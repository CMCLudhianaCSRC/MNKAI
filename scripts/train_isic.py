import argparse, os, time, math, signal, sys, logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import yaml
import timm
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from tqdm.auto import tqdm

# ---------- environment knobs that help stability on 8GB ----------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.6")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ---------------- utils (lightweight; rely on your src if present) ----------------
def load_yaml(p): 
    with open(p,"r",encoding="utf-8") as f: 
        return yaml.safe_load(f)

def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_tb(log_dir, fold=0):
    p = Path(log_dir) / f"run{fold}"
    p.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(p))

LOGGER_NAME = "mnkai.train"

def setup_logging(cfg):
    log_cfg = cfg.get("logging", {})
    log_dir = Path(log_cfg.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_cfg.get("log_file", "train_isic.log")
    level_name = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    fmt = logging.Formatter("%(asctime)s | %(levelname)7s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    file_path = log_dir / log_file
    file_handler = RotatingFileHandler(
        file_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setFormatter(fmt)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    logger.info("Log file -> %s", file_path.as_posix())
    return logger

def detect_cols(df):
    cols = {c.lower(): c for c in df.columns}
    path = cols.get("filepath") or cols.get("image") or cols.get("image_path") or cols.get("filename")
    label = cols.get("label_name") or cols.get("label") or cols.get("diagnosis") or cols.get("target")
    return path, label

def build_label_map(cfg):
    classes = [str(x).strip().lower() for x in cfg["classes"]["names"]]
    alias = cfg["classes"].get("aliases", {})
    table = {c:c for c in classes}
    for k, vs in alias.items():
        k = str(k).strip().lower()
        for v in vs:
            table[str(v).strip().lower()] = k
    ignore = set(str(s).lower() for s in cfg["classes"].get("ignore", []))
    return classes, table, ignore

def normalize_label(s, table, ignore):
    if s is None: return None
    t = str(s).strip().lower()
    if t in ignore: return None
    if t in table:  return table[t]
    t2 = " ".join(t.replace("-", " ").replace("_", " ").split())
    return table.get(t2, None)

# ---- simple transforms/dataset (we keep it minimal and stable) ----
from torchvision import transforms as T
from PIL import Image

def build_train_tfm(img_size, mean, std, aug_cfg):
    tfm = [
        T.RandomResizedCrop(img_size, scale=tuple(aug_cfg.get("random_crop_scale",[0.85,1.0])), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=float(aug_cfg.get("hflip",0.5))),
        T.RandomVerticalFlip(p=float(aug_cfg.get("vflip",0.2))),
        T.RandomRotation(int(aug_cfg.get("rotate",20))),
        T.ColorJitter(
            brightness=float(aug_cfg.get("brightness",0.1)),
            contrast=float(aug_cfg.get("contrast",0.1)),
            saturation=float(aug_cfg.get("saturation",0.05)),
            hue=float(aug_cfg.get("hue",0.02)),
        ),
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]
    return T.Compose(tfm)

def build_valid_tfm(img_size, mean, std, center=True):
    tfm = [
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]
    return T.Compose(tfm)

class SkinDataset(torch.utils.data.Dataset):
    def __init__(self, df, classes, img_size, tfm, images_dir=""):
        self.df = df.reset_index(drop=True)
        self.classes = classes
        self.cls2idx = {c:i for i,c in enumerate(classes)}
        self.tfm = tfm
        self.images_dir = images_dir
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        p = row["filepath"]
        if self.images_dir and not Path(p).is_absolute():
            p = str(Path(self.images_dir) / p)
        im = Image.open(p).convert("RGB")
        x = self.tfm(im)
        y = self.cls2idx[row["label"]]
        return {"image": x, "target": y}

# ---- Focal loss with label smoothing ----
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ls = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        ncls = logits.size(-1)
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        with torch.no_grad():
            td = torch.zeros_like(logits)
            td.fill_(self.ls / max(1, ncls - 1))
            td.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)
        pt = (td * p).sum(-1)
        fw = self.alpha * (1 - pt).pow(self.gamma)
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)[targets]
            fw = fw * weights
        loss = -(fw * (td * logp).sum(-1)).mean()
        return loss

# ---- metrics (macro-F1 minimal) ----
from sklearn.metrics import classification_report

@torch.no_grad()
def validate(model, dl, device, classes, tta=True, amp=False):
    model.eval()
    ys, yh, yp = [], [], []
    ce = torch.nn.CrossEntropyLoss()
    vloss = 0.0
    for batch in dl:
        x = batch["image"].to(device, non_blocking=True)
        y = torch.as_tensor(batch["target"], device=device)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(x)
            if tta:
                logits = 0.5 * (logits + model(torch.flip(x, dims=[3])))
            loss = ce(logits, y)
        vloss += float(loss.item())
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        yp.append(prob)
        ys.append(y.detach().cpu().numpy())
        yh.append(prob.argmax(1))
    if len(ys)==0:
        return {"loss": 0.0, "macro_f1": 0.0, "acc": 0.0}
    ys = np.concatenate(ys); yh = np.concatenate(yh); yp = np.concatenate(yp)
    rep = classification_report(ys, yh, target_names=classes, output_dict=True, zero_division=0)
    return {"loss": vloss/max(1,len(dl)), "macro_f1": float(rep["macro avg"]["f1-score"]), "acc": float(rep["accuracy"])}

# --------------------------- MAIN ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    logger = setup_logging(cfg)
    set_seed(int(cfg.get("seed",42)))

    hardware_cfg = cfg.get("hardware", {})
    torch.backends.cudnn.benchmark = bool(hardware_cfg.get("cudnn_benchmark", False))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(hardware_cfg.get("allow_tf32", True))
    channels_last = bool(hardware_cfg.get("channels_last", False))
    use_compile = bool(hardware_cfg.get("torch_compile", False)) and hasattr(torch, "compile")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(hardware_cfg.get("mixed_precision", True)) and device.type == "cuda"
    logger.info("Using device: %s", device)
    out_dir = Path(cfg.get("out_dir","checkpoints/run_stable"))
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(path: Path, state_dict, epoch_num: int, *, silent: bool = False):
        payload = {
            "epoch": epoch_num,
            "state_dict": state_dict,
            "classes": classes,
            "config": cfg,
        }
        try:
            torch.save(payload, path)
        except Exception as exc:
            if not silent:
                logger.warning("Failed to save checkpoint %s: %s", path, exc)

    # TensorBoard
    writer = None
    tb_cfg = cfg.get("tensorboard", {})
    if tb_cfg.get("enabled", True):
        writer = setup_tb(tb_cfg.get("log_dir", "runs"), fold=0)
        logger.info("TensorBoard -> %s", tb_cfg.get("log_dir", "runs"))

    classes, alias_table, ignore = build_label_map(cfg)

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    if train_cfg.get("use_folds"):
        logger.warning("Fold training requested but not supported in this script; using single validation split.")
    meta_csv = data_cfg.get("manifest_csv") or data_cfg.get("metadata_csv")
    if not meta_csv:
        raise RuntimeError("config.data.manifest_csv is required")
    raw = pd.read_csv(meta_csv)
    pcol, lcol = detect_cols(raw)
    strat_col = data_cfg.get("stratify_col", "label")
    group_col = data_cfg.get("group_column", "")

    df = pd.DataFrame({
        "filepath": raw[pcol].astype(str),
        "raw_label": raw[lcol]
    })
    if strat_col and strat_col not in df.columns and strat_col in raw.columns:
        df[strat_col] = raw[strat_col]
    if group_col and group_col not in df.columns and group_col in raw.columns:
        df[group_col] = raw[group_col]
    if strat_col and strat_col not in df.columns:
        logger.warning("Stratify column '%s' not found; defaulting to normalized labels.", strat_col)
    if group_col and group_col not in df.columns and group_col:
        logger.warning("Group column '%s' not found; proceeding without groups.", group_col)

    df["label"] = df["raw_label"].apply(lambda s: normalize_label(s, alias_table, ignore))
    df = df[df["label"].isin(classes)].copy()

    def _safe_is_file(p: str) -> bool:
        try:
            return Path(p).is_file()
        except OSError as exc:
            logger.warning("Skipping file %s due to access error: %s", p, exc)
            return False

    df = df[df["filepath"].map(_safe_is_file)].copy()
    df.reset_index(drop=True, inplace=True)
    logger.info("On-disk: %d images", len(df))

    seed = int(cfg.get("seed", 42))
    val_fraction = float(data_cfg.get("val_fraction", 0.20))
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("config.data.val_fraction must be between 0 and 1")

    strat_values = df["label"]
    if strat_col and strat_col in df.columns:
        strat_values = df[strat_col].fillna(df["label"])
    groups = None
    if group_col and group_col in df.columns:
        groups = df[group_col].fillna("").astype(str).values

    idx = np.arange(len(df))
    if groups is not None and len(np.unique(groups)) > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        tr_idx, va_idx = next(splitter.split(idx, groups=groups))
        logger.info("Split | train=%d | valid=%d (group-aware via GroupShuffleSplit)", len(tr_idx), len(va_idx))
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        tr_idx, va_idx = next(splitter.split(idx, strat_values))
        logger.info("Split | train=%d | valid=%d (stratified)", len(tr_idx), len(va_idx))

    train_df = df.iloc[tr_idx].reset_index(drop=True)
    valid_df = df.iloc[va_idx].reset_index(drop=True)

    # transforms/datasets
    img_size = int(data_cfg["img_size"])
    mean = tuple(data_cfg["normalize_mean"])
    std = tuple(data_cfg["normalize_std"])
    aug_cfg = cfg.get("augmentation", {})
    aug_train = build_train_tfm(img_size, mean, std, aug_cfg.get("train", {}))
    aug_valid = build_valid_tfm(
        img_size,
        mean,
        std,
        center=bool(aug_cfg.get("valid", {}).get("center_crop", True)),
    )

    images_dir = data_cfg.get("images_dir", "")
    ds_train = SkinDataset(train_df, classes, img_size, aug_train, images_dir=images_dir)
    ds_valid = SkinDataset(valid_df, classes, img_size, aug_valid, images_dir=images_dir)

    # sampler (balanced)
    cls_counts_series = train_df["label"].value_counts().reindex(classes).fillna(0)
    cls_counts = cls_counts_series.values.astype(np.int64)
    cls_w = 1.0 / np.maximum(cls_counts, 1)
    sw = train_df["label"].map(dict(zip(classes, cls_w))).values
    sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

    # class-balanced loss weights
    class_weights_loss = None
    balancing_mode = str(train_cfg.get("class_balancing", "")).lower()
    if balancing_mode == "effective_num":
        beta = float(train_cfg.get("effective_beta", 0.9999))
        weights = np.ones_like(cls_counts, dtype=np.float64)
        mask = cls_counts > 0
        if mask.any():
            weights[mask] = (1.0 - beta) / (1.0 - np.power(beta, cls_counts[mask]))
        total = weights.sum()
        if total > 0:
            weights = weights * (len(weights) / total)
        class_weights_loss = weights

    # loaders with fallback worker counts
    requested_workers = int(hardware_cfg.get("num_workers", 0))
    pin = bool(hardware_cfg.get("pin_memory", False))
    prefetch = int(hardware_cfg.get("prefetch_factor", 2))
    persistent = bool(hardware_cfg.get("persistent_workers", False))
    train_batch_size = int(train_cfg["batch_size"])

    def _make_loader(dataset, batch_size, *, sampler=None, shuffle=False, persistent_flag=False, workers_hint=requested_workers):
        worker_candidates = []
        if workers_hint > 0:
            worker_candidates.append(workers_hint)
            for fallback in (4, 2):
                if fallback < workers_hint and fallback not in worker_candidates:
                    worker_candidates.append(fallback)
        if 0 not in worker_candidates:
            worker_candidates.append(0)
        last_exc = None
        for workers in worker_candidates:
            try:
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    sampler=sampler,
                    shuffle=(sampler is None and shuffle),
                    num_workers=workers,
                    pin_memory=pin,
                    prefetch_factor=prefetch if workers > 0 else None,
                    persistent_workers=persistent_flag and workers > 0,
                )
                if workers != workers_hint:
                    logger.warning("DataLoader fallback: using %d workers (requested %d).", workers, workers_hint)
                return loader, workers
            except RuntimeError as exc:
                logger.warning("DataLoader init failed with %d workers: %s", workers, exc)
                last_exc = exc
        raise last_exc if last_exc is not None else RuntimeError("Failed to initialise DataLoader.")

    dl_train, train_workers_used = _make_loader(
        ds_train,
        train_batch_size,
        sampler=sampler,
        shuffle=False,
        persistent_flag=persistent,
        workers_hint=requested_workers,
    )
    dl_valid, valid_workers_used = _make_loader(
        ds_valid,
        batch_size=max(1, train_batch_size * 2),
        sampler=None,
        shuffle=False,
        persistent_flag=False,
        workers_hint=train_workers_used if train_workers_used > 0 else requested_workers,
    )
    logger.info("DataLoader workers -> train=%d | valid=%d", train_workers_used, valid_workers_used)

    # model
    model = timm.create_model(
        train_cfg["model_name"],
        pretrained=bool(train_cfg.get("pretrained", True)),
        num_classes=len(classes),
    )
    if bool(train_cfg.get("grad_checkpointing", False)) and hasattr(model, "set_grad_checkpointing"):
        try:
            model.set_grad_checkpointing(True)
            logger.info("Grad checkpointing enabled.")
        except Exception as exc:
            logger.warning("Grad checkpointing request failed: %s", exc)
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    if use_compile:
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
            logger.info("torch.compile enabled.")
        except Exception as exc:
            logger.warning("torch.compile failed: %s", exc)
            use_compile = False
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model %s | trainable params: %.2fM", train_cfg["model_name"], trainable_params / 1e6)

    # loss, opt, sched
    loss_name = str(train_cfg.get("loss", "cross_entropy")).lower()
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    focal_alpha = float(train_cfg.get("focal_alpha", 0.25))
    focal_gamma = float(train_cfg.get("focal_gamma", 2.0))
    class_weights_tensor = None
    if class_weights_loss is not None:
        class_weights_tensor = torch.tensor(class_weights_loss, dtype=torch.float32, device=device)

    if loss_name in {"cb_focal", "focal"}:
        criterion = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
            class_weights=class_weights_loss,
        )
    elif loss_name in {"ce", "cross_entropy"}:
        criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            weight=class_weights_tensor,
        )
    else:
        raise ValueError(f"Unsupported loss '{loss_name}'.")

    base_lr = float(train_cfg["lr"])
    optimizer = AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(train_cfg["weight_decay"]),
    )
    epochs = int(train_cfg["epochs"])
    warm = max(0, int(train_cfg.get("warmup_epochs", 0)))
    min_lr = float(train_cfg.get("min_lr", 0.0))
    scheduler_cfg = train_cfg.get("scheduler", "cosine")
    if isinstance(scheduler_cfg, dict):
        scheduler_name = str(scheduler_cfg.get("name", "cosine")).lower()
        scheduler_params = scheduler_cfg
    else:
        scheduler_name = str(scheduler_cfg).lower()
        scheduler_params = {}
    monitor_key = str(train_cfg.get("monitor", "macro_f1"))
    monitor_lower = monitor_key.lower()
    higher_is_better = not any(term in monitor_lower for term in ("loss", "err"))

    scheduler_requires_metric = False
    if scheduler_name in {"cosine", "cosine_annealing"}:
        min_scale = min_lr / base_lr if base_lr > 0 else 0.0
        def lr_lambda(ep):
            if ep < warm:
                return (ep + 1) / max(1, warm)
            prog = (ep - warm) / max(1, epochs - warm)
            cosine = 0.5 * (1.0 + math.cos(math.pi * prog))
            return min_scale + (1.0 - min_scale) * cosine
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_name in {"cosine_restart", "cosineannealingwarmrestarts"}:
        t0 = int(scheduler_params.get("t0", scheduler_params.get("T_0", max(1, epochs - warm))))
        tmult = int(scheduler_params.get("t_mult", 1))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t0,
            T_mult=max(1, tmult),
            eta_min=min_lr,
        )
    elif scheduler_name in {"reduce_on_plateau", "plateau"}:
        scheduler_requires_metric = True
        plateau_factor = float(train_cfg.get("plateau_factor", scheduler_params.get("factor", 0.5)))
        plateau_patience = int(train_cfg.get("plateau_patience", scheduler_params.get("patience", 2)))
        plateau_threshold = float(train_cfg.get("plateau_threshold", scheduler_params.get("threshold", 0.01)))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max" if higher_is_better else "min",
            factor=plateau_factor,
            patience=plateau_patience,
            threshold=plateau_threshold,
            threshold_mode="abs",
            min_lr=min_lr,
            verbose=True,
        )
    else:
        raise ValueError(f"Unsupported scheduler '{scheduler_name}'.")

    # EMA (cheap update every N steps)
    ema_enabled = bool(train_cfg.get("ema", True))
    ema_every = max(1, int(train_cfg.get("ema_every_steps", 1)))
    ema_decay  = float(train_cfg.get("ema_decay", 0.999))
    ema_model = deepcopy(model).to(device) if ema_enabled else None
    if ema_enabled:
        for p in ema_model.parameters():
            p.requires_grad_(False)

    swa_enabled = bool(train_cfg.get("swa", False))
    swa_start_frac = float(train_cfg.get("swa_start_frac", 0.0))
    swa_model = None
    swa_updates = 0

    # AMP scaler (new API)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # resume
    start_epoch = 0
    resume_cfg = train_cfg.get("resume", "")
    resume_path = ""
    if isinstance(resume_cfg, str):
        if resume_cfg == "auto_last":
            candidates = sorted(out_dir.glob("last*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                candidates = sorted(out_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                resume_path = candidates[0].as_posix()
                logger.info("Auto-resume checkpoint -> %s", resume_path)
        elif resume_cfg:
            resume_path = resume_cfg

    if resume_path:
        try:
            ck = torch.load(resume_path, map_location="cpu")
            sd = ck.get("state_dict", ck)
            model.load_state_dict(sd, strict=False)
            start_epoch = int(ck.get("epoch", 0))
            logger.info("Resumed from %s @ epoch %d", resume_path, start_epoch)
        except Exception as exc:
            logger.warning("Resume failed from %s: %s", resume_path, exc)
            start_epoch = 0

    if hasattr(scheduler, "last_epoch"):
        scheduler.last_epoch = max(start_epoch - 1, -1)

    # safety dirs
    out_dir.mkdir(parents=True, exist_ok=True)
    (Path(cfg.get("report_dir", out_dir / "reports"))).mkdir(parents=True, exist_ok=True)

    # training
    accum = int(train_cfg.get("grad_accum_steps", 1))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", train_cfg.get("max_grad_norm", 0.0)))
    mixup_alpha = float(train_cfg.get("mixup_alpha", 0.0))
    save_every_steps = int(train_cfg.get("checkpoint_every_n_steps", 250))
    save_every_minutes = int(train_cfg.get("checkpoint_every_minutes", 5))
    log_every = max(1, int(train_cfg.get("log_every_n_steps", 50)))
    tta_valid = bool(train_cfg.get("tta_valid", True))
    last_save_t = time.time()

    interrupted = {"flag": False}
    def _sigint(sig, frm): interrupted["flag"]=True
    try: signal.signal(signal.SIGINT, _sigint)
    except: pass

    global_step = 0
    best_f1 = -1.0

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        steps_per_epoch = max(1, len(dl_train))
        ema_counter = 0

        with tqdm(
            dl_train,
            total=len(dl_train),
            dynamic_ncols=True,
            leave=False,
            desc=f"epoch {epoch+1}/{epochs}",
        ) as loop:
            for step, batch in enumerate(loop, start=1):
                iter_start = time.perf_counter()
                x = batch["image"].to(device, non_blocking=True)
                y = torch.as_tensor(batch["target"], device=device)
                if channels_last:
                    x = x.to(memory_format=torch.channels_last)

                lam = 1.0
                y_perm = None
                if mixup_alpha > 0.0 and x.size(0) > 1:
                    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                    perm = torch.randperm(x.size(0), device=device)
                    x = lam * x + (1.0 - lam) * x[perm]
                    y_perm = y[perm]

                try:
                    with torch.amp.autocast("cuda", enabled=amp_enabled):
                        logits = model(x)
                        loss_primary = criterion(logits, y)
                        if y_perm is not None:
                            loss_secondary = criterion(logits, y_perm)
                            loss_value = lam * loss_primary + (1.0 - lam) * loss_secondary
                        else:
                            loss_value = loss_primary
                    loss = loss_value / accum

                    if not torch.isfinite(loss):
                        logger.error("[nan] Non-finite loss at epoch %d step %d. Skipping.", epoch + 1, step)
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    step_loss = float(loss_value.detach().item())
                    scaler.scale(loss).backward()

                    boundary = (step % accum == 0) or (step == steps_per_epoch)
                    if boundary:
                        if grad_clip_norm > 0.0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        ema_counter += 1

                        if ema_enabled and ema_model is not None and (ema_counter % ema_every == 0):
                            with torch.no_grad():
                                for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                                    p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

                    train_loss += step_loss

                    if (time.time() - last_save_t) >= save_every_minutes * 60:
                        last_save_t = time.time()
                        mid = out_dir / f"mid_epoch{epoch+1}_step{step}.pt"
                        _save_checkpoint(mid, model.state_dict(), epoch + 1, silent=True)

                    if save_every_steps > 0 and (step % save_every_steps == 0):
                        mid = out_dir / f"mid_epoch{epoch+1}_step{step}.pt"
                        _save_checkpoint(mid, model.state_dict(), epoch + 1, silent=True)

                    if writer and (step % log_every == 0):
                        global_idx = epoch * steps_per_epoch + step
                        writer.add_scalar("train/loss_step", step_loss, global_idx)
                        writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), global_idx)

                    iter_time = time.perf_counter() - iter_start
                    batch_sz = x.size(0)
                    ips = (batch_sz / iter_time) if iter_time > 0 else 0.0
                    lr_now = optimizer.param_groups[0]["lr"]
                    mem_gb = torch.cuda.memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
                    loop.set_postfix(
                        loss=f"{step_loss:.4f}",
                        lr=f"{lr_now:.2e}",
                        ips=f"{ips:.1f}",
                        mem=f"{mem_gb:.2f}GB",
                    )

                except torch.cuda.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    loop.set_postfix_str("oom -> cleared cache")
                    continue

                if interrupted["flag"]:
                    emer = out_dir / f"emergency_epoch{epoch+1}_step{step}.pt"
                    _save_checkpoint(emer, model.state_dict(), epoch + 1)
                    logger.warning("Interrupted. Emergency checkpoint saved.")
                    return

        if writer:
            writer.add_scalar("train/loss_epoch", float(train_loss / max(1, steps_per_epoch)), epoch)

        if swa_enabled:
            swa_start_epoch = max(1, int(math.ceil(swa_start_frac * epochs)))
            if (epoch + 1) >= swa_start_epoch:
                if swa_model is None:
                    swa_model = deepcopy(model).to(device)
                    if channels_last:
                        swa_model = swa_model.to(memory_format=torch.channels_last)
                    for p in swa_model.parameters():
                        p.requires_grad_(False)
                    swa_updates = 1
                else:
                    swa_updates += 1
                    beta = 1.0 / swa_updates
                    with torch.no_grad():
                        for p_swa, p in zip(swa_model.parameters(), model.parameters()):
                            p_swa.data.lerp_(p.data, beta)

        eval_model = ema_model if ema_enabled else model
        if not ema_enabled and swa_model is not None:
            eval_model = swa_model
        val = validate(eval_model, dl_valid, device, classes, tta=tta_valid, amp=amp_enabled)
        logger.info(
            "VALID epoch %d: loss=%.4f acc=%.4f macroF1=%.4f",
            epoch + 1,
            val["loss"],
            val["acc"],
            val["macro_f1"],
        )
        if writer:
            writer.add_scalar("valid/loss", val["loss"], epoch)
            writer.add_scalar("valid/macro_f1", val["macro_f1"], epoch)
            writer.add_scalar("valid/acc", val["acc"], epoch)

        last = out_dir / "last.pt"
        _save_checkpoint(last, model.state_dict(), epoch + 1)

        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best = out_dir / "best.pt"
            _save_checkpoint(best, eval_model.state_dict(), epoch + 1)

        if scheduler_requires_metric:
            metric_value = val.get(monitor_lower)
            if metric_value is None:
                fallback_key = "loss" if not higher_is_better else "macro_f1"
                metric_value = val.get(fallback_key, 0.0)
            scheduler.step(metric_value)
        else:
            scheduler.step()

    if writer:
        try: writer.close()
        except: pass

if __name__ == "__main__":
    main()
