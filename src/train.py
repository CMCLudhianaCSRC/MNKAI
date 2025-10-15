import argparse, json, math, os, signal, time
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from loguru import logger
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ---- Local modules (ensure src/ exists) ----
try:
    from src.dataset import SkinDataset
    from src.metrics import metrics_from_preds
    from src.transforms import AugConfig, build_train, build_valid
    from src.utils import (
        setup_logging, set_seed, seed_worker, get_device, configure_torch_backends,
        safe_mkdir, count_parameters,
    )
    from src.model import build_model
except Exception as e:
    print("Missing or broken local modules under src/. Ensure you have src/dataset.py, "
          "src/metrics.py, src/transforms.py, src/utils.py, src/model.py.")
    raise

# ----------------------------
# Config I/O
# ----------------------------
def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ----------------------------
# Label normalization helpers
# ----------------------------
def _build_alias_table(cfg):
    canonical = [str(x).strip().lower() for x in cfg["classes"]["names"]]
    aliases_cfg = cfg["classes"].get("aliases", {})
    table = {c: c for c in canonical}
    for can, alias_list in aliases_cfg.items():
        can = str(can).strip().lower()
        if can not in canonical:
            logger.warning(f"[class] alias key '{can}' not in classes.names; skipping")
            continue
        for a in alias_list:
            table[str(a).strip().lower()] = can
    ignore = set([str(s).lower() for s in cfg["classes"].get("ignore", [])])
    return canonical, table, ignore

def _normalize_label(raw: str, alias_table: dict, ignore_set: set[str]):
    if raw is None:
        return None, True
    s = str(raw).strip().lower()
    if s in ignore_set:
        return None, True
    if s in alias_table:
        return alias_table[s], False
    s2 = " ".join(s.replace("-", " ").replace("_", " ").replace("/", " ").replace("\\", " ").split())
    if s2 in alias_table:
        return alias_table[s2], False
    return None, True

# ----------------------------
# Build a single train/valid split (no folds path)
# ----------------------------
def build_single_split(meta_csv, images_dir, classes, alias_table, ignore_set, seed=42, val_fraction=0.20, stratify_col="label_name"):
    df_raw = pd.read_csv(meta_csv)
    # detect columns
    cols = {c.lower(): c for c in df_raw.columns}
    path_col = cols.get("filepath") or cols.get("image") or cols.get("image_path") or cols.get("filename")
    label_col = cols.get("label_name") or cols.get("label") or cols.get("diagnosis") or cols.get("dx")
    if path_col is None or label_col is None:
        raise ValueError("CSV must have image path (filepath/filename/image) and a label column (label_name/label/diagnosis).")

    root = Path(images_dir) if images_dir else None
    def _resolve(p: str) -> str:
        P = Path(str(p))
        return str((root / P).resolve()) if (root and not P.is_absolute()) else str(P)

    norm_labels, keep_mask = [], []
    for raw in df_raw[label_col].astype(str).tolist():
        lab, ign = _normalize_label(raw, alias_table, ignore_set)
        if ign or lab is None:
            keep_mask.append(False); norm_labels.append(None)
        else:
            keep_mask.append(True);  norm_labels.append(lab)

    df = pd.DataFrame({
        "filepath": df_raw[path_col].astype(str).map(_resolve),
        "label": norm_labels
    })
    df = df[keep_mask].reset_index(drop=True)

    exists = df["filepath"].map(lambda p: Path(p).is_file())
    n_ok = int(exists.sum())
    logger.info(f"[paths] Found {n_ok}/{len(df)} files on disk.")

    df = df[df["label"].isin(classes)].reset_index(drop=True)

    # stratified split
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    y = df["label"].values
    idx = np.arange(len(df))
    (train_idx, valid_idx), = sss.split(idx, y)  # type: ignore
    df["is_valid"] = 0
    df.loc[valid_idx, "is_valid"] = 1
    return df

# ----------------------------
# Loss
# ----------------------------
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.label_smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        pt = (true_dist * probs).sum(dim=-1)
        focal_weight = self.alpha * (1 - pt).pow(self.gamma)
        loss = -(focal_weight * (true_dist * log_probs).sum(dim=-1))
        return loss.mean()


def _rand_bbox(width: int, height: int, lam: float) -> Tuple[int, int, int, int]:
    """Sample a CutMix bounding box."""
    cut_ratio = math.sqrt(max(0.0, 1.0 - lam))
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    # ensure at least 1 pixel
    cut_w = max(1, cut_w)
    cut_h = max(1, cut_h)

    cx = np.random.randint(0, width)
    cy = np.random.randint(0, height)

    x1 = np.clip(cx - cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y2 = np.clip(cy + cut_h // 2, 0, height)
    return int(x1), int(y1), int(x2), int(y2)


def _mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0.0 or x.size(0) < 2:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    lam = max(0.0, min(1.0, lam))
    perm = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[perm]
    return mixed_x, y, y[perm], lam


def _cutmix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0.0 or x.size(-1) < 2 or x.size(-2) < 2 or x.size(0) < 2:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    lam = max(0.0, min(1.0, lam))
    perm = torch.randperm(x.size(0), device=x.device)
    target_a, target_b = y, y[perm]

    _, _, h, w = x.shape
    x1, y1, x2, y2 = _rand_bbox(w, h, lam)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    lam_adj = 1.0 - ((x2 - x1) * (y2 - y1) / max(1, w * h))
    return x, target_a, target_b, float(lam_adj)


def maybe_apply_mixture(
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: dict,
    epoch: int,
    total_epochs: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, str]:
    train_cfg = cfg["train"]
    mixup_alpha = float(train_cfg.get("mixup_alpha", 0.0))
    cutmix_alpha = float(train_cfg.get("cutmix_alpha", 0.0))
    mix_prob = float(train_cfg.get("mix_prob", 0.0))
    cutmix_prob = float(train_cfg.get("cutmix_prob", 0.5))
    start_frac = float(train_cfg.get("mix_start_frac", 0.0))
    end_frac = float(train_cfg.get("mix_end_frac", 1.0))

    start_epoch = max(0, int(total_epochs * start_frac))
    end_epoch = max(start_epoch, int(total_epochs * end_frac))

    if mix_prob <= 0.0 or (mixup_alpha <= 0.0 and cutmix_alpha <= 0.0):
        return x, None, None, 1.0, "none"
    if epoch < start_epoch or epoch >= end_epoch:
        return x, None, None, 1.0, "none"

    # anneal probability toward end to stabilise training
    if end_epoch > start_epoch:
        frac = (epoch - start_epoch) / max(1, end_epoch - start_epoch)
        epoch_prob = mix_prob * max(0.0, 1.0 - frac)
    else:
        epoch_prob = mix_prob

    if torch.rand(1).item() > epoch_prob:
        return x, None, None, 1.0, "none"

    use_cutmix = cutmix_alpha > 0.0 and torch.rand(1).item() < cutmix_prob
    if use_cutmix:
        mixed_x, ya, yb, lam = _cutmix_batch(x, y, cutmix_alpha)
        return mixed_x, ya, yb, lam, "cutmix"

    mixed_x, ya, yb, lam = _mixup_batch(x, y, mixup_alpha)
    return mixed_x, ya, yb, lam, "mixup"

# ----------------------------
# DataLoaders
# ----------------------------
def make_loaders(df, classes, cfg, aug_train, aug_valid):
    bs = int(cfg["train"]["batch_size"])
    img_size = int(cfg["data"]["img_size"])
    images_dir = cfg["data"]["images_dir"]
    hair_removal = bool(cfg["data"].get("hair_removal", False))

    train_df = df[df["is_valid"] == 0].reset_index(drop=True)
    valid_df = df[df["is_valid"] == 1].reset_index(drop=True)

    ds_train = SkinDataset(train_df, classes, img_size, transform=aug_train,
                           hair_removal=hair_removal, images_dir=images_dir)
    ds_valid = SkinDataset(valid_df, classes, img_size, transform=aug_valid,
                           hair_removal=hair_removal, images_dir=images_dir)

    # simple class-balanced sampler
    class_counts = train_df["label"].value_counts().reindex(classes).fillna(0).values
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = train_df["label"].map(dict(zip(classes, class_weights))).values
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    hw = cfg["hardware"]
    num_workers = int(hw["num_workers"])
    prefetch_factor = int(hw.get("prefetch_factor", 1))
    pin_memory = bool(hw.get("pin_memory", False))
    persistent = bool(hw.get("persistent_workers", False)) and num_workers > 0

    dl_train = DataLoader(
        ds_train,
        batch_size=bs,
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        persistent_workers=persistent,
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=max(1, bs * 2),
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        persistent_workers=persistent,
    )
    return dl_train, dl_valid

# ----------------------------
# Validation (optional TTA)
# ----------------------------
@torch.no_grad()
def run_validation(model, dl_valid, device, classes, use_tta: bool):
    model.eval()
    vy_true, vy_pred, vy_prob = [], [], []
    criterion = torch.nn.CrossEntropyLoss()
    val_loss = 0.0

    def tta_forward(x):
        outs = [model(x)]
        if use_tta:
            outs += [
                model(torch.flip(x, dims=[3])),
                model(torch.flip(x, dims=[2])),
                model(torch.flip(x, dims=[2, 3])),
            ]
        return torch.stack(outs, dim=0).mean(0)

    for batch in dl_valid:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        logits = tta_forward(x)
        loss = criterion(logits, y)
        val_loss += loss.item()

        prob = F.softmax(logits, dim=-1).cpu().numpy()
        vy_prob.append(prob)
        vy_true.append(y.cpu().numpy())
        vy_pred.append(prob.argmax(1))

    if len(vy_true) > 0:
        vy_true = np.concatenate(vy_true)
        vy_pred = np.concatenate(vy_pred)
        vy_prob = np.concatenate(vy_prob)
        vrpt = metrics_from_preds(vy_true, vy_pred, vy_prob, classes)
    else:
        vrpt = {"report": {"accuracy": 0.0, "macro avg": {"f1-score": 0.0}}}

    vrpt["loss"] = float(val_loss / max(1, len(dl_valid)))
    return vrpt

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)

    setup_logging(cfg["logging"]["log_dir"], cfg["logging"]["log_file"], cfg["logging"]["level"])
    set_seed(cfg["seed"])
    device = get_device()
    logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0) if device.type=='cuda' else device.type}")
    configure_torch_backends(
        allow_tf32=bool(cfg["hardware"].get("allow_tf32", True)),
        compile_mode=bool(cfg["hardware"].get("torch_compile", False)),
    )

    # TensorBoard
    writer = None
    if cfg.get("tensorboard", {}).get("enabled", True):
        tb_dir = Path(cfg.get("tensorboard", {}).get("log_dir", "runs")) / "run1"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        logger.info(f"[tb] TensorBoard -> {tb_dir}")

    # AMP dtype + channels_last
    amp_dtype_cfg = (cfg.get("hardware", {}).get("amp_dtype", "fp16") or "fp16").lower()
    autocast_dtype = torch.bfloat16 if amp_dtype_cfg in ("bf16", "bfloat16") else torch.float16
    use_channels_last = True

    # Classes and CSV → split
    classes, alias_table, ignore_set = _build_alias_table(cfg)
    logger.info(f"[classes] {len(classes)} classes: {classes}")

    df = build_single_split(
        cfg["data"]["manifest_csv"],
        cfg["data"]["images_dir"],
        classes, alias_table, ignore_set,
        seed=int(cfg["seed"]),
        val_fraction=0.20,
        stratify_col="label_name",
    )
    logger.info(f"[split] train={(df['is_valid']==0).sum()} | valid={(df['is_valid']==1).sum()}")

    # Augs
    aug_conf = AugConfig(
        img_size=int(cfg["data"]["img_size"]),
        mean=tuple(cfg["data"]["normalize_mean"]),
        std=tuple(cfg["data"]["normalize_std"]),
        hflip=float(cfg["augmentation"]["train"]["hflip"]),
        vflip=float(cfg["augmentation"]["train"]["vflip"]),
        rotate=int(cfg["augmentation"]["train"]["rotate"]),
        brightness=float(cfg["augmentation"]["train"]["brightness"]),
        contrast=float(cfg["augmentation"]["train"]["contrast"]),
        saturation=float(cfg["augmentation"]["train"]["saturation"]),
        hue=float(cfg["augmentation"]["train"]["hue"]),
        blur=float(cfg["augmentation"]["train"]["blur"]),
        jpeg=int(cfg["augmentation"]["train"]["jpeg"]),
        random_crop_scale=tuple(cfg["augmentation"]["train"]["random_crop_scale"]),
    )
    aug_train = build_train(aug_conf)
    aug_valid = build_valid(aug_conf, center_crop=bool(cfg["augmentation"]["valid"].get("center_crop", True)))

    # Loaders
    dl_train, dl_valid = make_loaders(df, classes, cfg, aug_train, aug_valid)

    # Model
    logger.info(f"[model] Building {cfg['train']['model_name']} (num_classes={len(classes)}, pretrained={cfg['train']['pretrained']})")
    model = build_model(cfg["train"]["model_name"], len(classes), cfg["train"]["pretrained"],
                        compile_model=bool(cfg["hardware"].get("torch_compile", False))).to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    if bool(cfg["train"].get("grad_checkpointing", False)) and hasattr(model, "set_grad_checkpointing"):
        try:
            model.set_grad_checkpointing(True)
            logger.info("[model] Grad checkpointing enabled.")
        except Exception:
            logger.warning("[model] Grad checkpointing unsupported; skipping.")

    logger.info(f"[model] Trainable parameters: {count_parameters(model):,}")

    # Loss
    if cfg["train"]["loss"] == "focal":
        criterion = FocalLoss(
            alpha=float(cfg["train"]["focal_alpha"]),
            gamma=float(cfg["train"]["focal_gamma"]),
            label_smoothing=float(cfg["train"]["label_smoothing"]),
        )
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=float(cfg["train"]["label_smoothing"]))

    # Optim + scheduler
    optimizer = AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    total_epochs = int(cfg["train"]["epochs"])
    def lr_lambda(epoch):
        warm = int(cfg["train"]["warmup_epochs"])
        if epoch < warm:
            return (epoch + 1) / max(1, warm)
        progress = (epoch - warm) / max(1, total_epochs - warm)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # EMA (off in stable config; wiring kept for later)
    ema_enabled = bool(cfg["train"].get("ema", False))
    ema_decay = float(cfg["train"].get("ema_decay", 0.999))
    ema_model = deepcopy(model).to(device) if ema_enabled else None
    if ema_enabled:
        for p in ema_model.parameters():
            p.requires_grad_(False)
        logger.info("[ema] enabled")

    # Safety dirs
    safe_mkdir("checkpoints"); safe_mkdir("exports")

    # AMP scaler (CUDA only)
    scaler = torch.amp.GradScaler('cuda', enabled=bool(cfg["hardware"]["mixed_precision"]) and device.type == "cuda")

    best_macro_f1 = -1.0
    epochs_no_improve = 0
    steps_per_epoch = max(1, len(dl_train))
    save_every = int(cfg["train"].get("checkpoint_every_n_steps", 0))
    grad_accum = max(1, int(cfg["train"].get("grad_accum_steps", 1)))
    max_grad_norm = float(cfg["train"].get("max_grad_norm", 0.0))

    logger.info(f"[train] grad_accum={grad_accum} | amp={scaler.is_enabled()} | save_every={save_every}")

    # Graceful interrupt → emergency save
    interrupted = {"flag": False}
    def _handle_sigint(signum, frame):
        interrupted["flag"] = True
        logger.warning("[signal] Ctrl-C caught; will save emergency checkpoint.")
    try:
        signal.signal(signal.SIGINT, _handle_sigint)
    except Exception:
        pass

    # Resume
    start_epoch = 0
    resume_path = cfg["train"].get("resume") or ""
    if isinstance(resume_path, str) and len(resume_path) > 0 and Path(resume_path).is_file():
        try:
            ckpt_r = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt_r["state_dict"], strict=False)
            if "optimizer" in ckpt_r:
                try:
                    optimizer.load_state_dict(ckpt_r["optimizer"])
                except Exception:
                    logger.warning("[resume] Optimizer mismatch; starting fresh.")
            if "epoch" in ckpt_r:
                start_epoch = int(ckpt_r["epoch"])
            logger.info(f"[resume] Loaded {resume_path}, resume from epoch index {start_epoch}.")
        except Exception as e:
            logger.warning(f"[resume] Failed to load '{resume_path}': {e}")
    scheduler.last_epoch = start_epoch - 1

    ema_every_steps = max(1, int(cfg["train"].get("ema_every_steps", 1)))
    ema_start_epoch = int(cfg["train"].get("ema_start_epoch", 0))
    ema_warmup_updates = max(0, int(cfg["train"].get("ema_warmup_updates", 0)))
    optimizer_updates = max(0, start_epoch * max(1, steps_per_epoch // grad_accum))

    swa_enabled = bool(cfg["train"].get("swa", False))
    swa_start_frac = float(cfg["train"].get("swa_start_frac", 1.0))
    swa_start_epoch = max(0, int(total_epochs * swa_start_frac))
    swa_model = AveragedModel(model).to(device) if swa_enabled else None
    swa_scheduler = None
    swa_updates = 0
    if swa_enabled:
        swa_lr = float(cfg["train"].get("swa_lr", cfg["train"]["lr"]))
        swa_anneal_epochs = max(1, int(cfg["train"].get("swa_anneal_epochs", 5)))
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr, anneal_epochs=swa_anneal_epochs)
        logger.info(f"[swa] enabled: start_epoch={swa_start_epoch}, swa_lr={swa_lr}, anneal_epochs={swa_anneal_epochs}")

    # ---------------- Train loop ----------------
    for epoch in range(start_epoch, total_epochs):
        epoch_t0 = time.time()
        model.train()
        train_loss = 0.0
        y_true, y_pred, y_prob = [], [], []

        optimizer.zero_grad(set_to_none=True)
        loop = tqdm(dl_train, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{total_epochs}", ncols=110)

        for step, batch in enumerate(loop, start=1):
            t0 = time.time()
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            if use_channels_last:
                images = images.to(memory_format=torch.channels_last)

            inputs, targets_a, targets_b, lam, mix_mode = maybe_apply_mixture(
                images, targets, cfg, epoch, total_epochs
            )
            if use_channels_last:
                inputs = inputs.to(memory_format=torch.channels_last)

            try:
                with torch.amp.autocast('cuda', enabled=scaler.is_enabled(), dtype=autocast_dtype):
                    logits = model(inputs)
                    if mix_mode != "none" and targets_a is not None and targets_b is not None:
                        loss = lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)
                    else:
                        loss = criterion(logits, targets)
                    loss = loss / grad_accum

                if not torch.isfinite(loss):
                    logger.error(f"[nan] Non-finite loss at epoch {epoch} step {step}. Skipping step.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()

                is_accum_boundary = (step % grad_accum) == 0 or (step == steps_per_epoch)
                if is_accum_boundary:
                    # unscale once per optimizer update
                    if max_grad_norm > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_updates += 1

                    # EMA update
                    if (
                        ema_enabled
                        and epoch >= ema_start_epoch
                        and optimizer_updates >= ema_warmup_updates
                        and (optimizer_updates % ema_every_steps) == 0
                    ):
                        with torch.no_grad():
                            for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                                p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

                    if swa_enabled and epoch >= swa_start_epoch:
                        swa_model.update_parameters(model)
                        swa_updates += 1

                # Stats
                train_loss += float(loss.item() * grad_accum)
                probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
                y_prob.append(probs)
                y_true.append(targets.detach().cpu().numpy())
                y_pred.append(probs.argmax(1))

                # Live progress
                step_time = time.time() - t0
                ips = inputs.size(0) / max(1e-6, step_time)
                pct = 100.0 * step / steps_per_epoch
                loop.set_postfix(loss=f"{(loss.item()*grad_accum):.4f}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                                 ips=f"{ips:.1f}/s",
                                 done=f"{pct:.1f}%",
                                 mix=mix_mode)

                # TB per-step
                if writer:
                    gs = epoch * steps_per_epoch + (step - 1)
                    writer.add_scalar("train/loss_step", float(loss.item() * grad_accum), gs)
                    writer.add_scalar("train/ips", float(ips), gs)
                    writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), gs)

                # Mid-epoch safety checkpoint
                if save_every > 0 and (step % save_every == 0 or step == steps_per_epoch):
                    ckpt_mid = Path("checkpoints") / f"mid_epoch{epoch+1}_step{step}.pt"
                    try:
                        torch.save(
                            {"epoch": epoch + 1, "step": step, "state_dict": model.state_dict(),
                             "classes": classes, "config": cfg, "optimizer": optimizer.state_dict()},
                            ckpt_mid,
                        )
                    except Exception as e:
                        logger.warning(f"[ckpt] Mid-epoch save failed: {e}")

            except torch.cuda.OutOfMemoryError:
                logger.error(f"[oom] CUDA OOM at epoch {epoch+1} step {step}. Skipping batch.")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                continue

            if interrupted["flag"]:
                emer = Path("checkpoints") / f"emergency_epoch{epoch+1}_step{step}.pt"
                try:
                    torch.save(
                        {"epoch": epoch + 1, "step": step, "state_dict": model.state_dict(),
                         "classes": classes, "config": cfg, "optimizer": optimizer.state_dict()},
                        emer,
                    )
                    logger.warning(f"[signal] Emergency checkpoint saved: {emer}. Exiting.")
                except Exception as e:
                    logger.error(f"[signal] Emergency checkpoint save FAILED: {e}")
                return

        # epoch train summary
        if len(y_true) > 0:
            y_true_np = np.concatenate(y_true)
            y_pred_np = np.concatenate(y_pred)
            y_prob_np = np.concatenate(y_prob)
            trpt = metrics_from_preds(y_true_np, y_pred_np, y_prob_np, classes)
            macro_f1 = float(trpt.get("report", {}).get("macro avg", {}).get("f1-score", 0.0))
        else:
            trpt = {"report": {"macro avg": {"f1-score": 0.0}}}
            macro_f1 = 0.0
        logger.info(f"[train] epoch {epoch+1}: loss={train_loss/max(1,len(dl_train)):.4f} macroF1={macro_f1:.4f}")
        if writer:
            writer.add_scalar("train/loss_epoch", float(train_loss / max(1, len(dl_train))), epoch)
            writer.add_scalar("train/macro_f1", macro_f1, epoch)

        # validation (use EMA if enabled)
        eval_model = ema_model if ema_enabled else model
        vrpt = run_validation(eval_model, dl_valid, device, classes, use_tta=bool(cfg["train"].get("tta_valid", False)))
        v_macro_f1 = float(vrpt["report"]["macro avg"]["f1-score"])
        v_acc = float(vrpt["report"]["accuracy"])
        v_auc = float(vrpt.get("macro_auc_ovr", -1.0))
        logger.info(f"[valid] epoch {epoch+1}: loss={vrpt['loss']:.4f} acc={v_acc:.4f} macroF1={v_macro_f1:.4f} auc={v_auc:.4f}")
        if writer:
            writer.add_scalar("valid/loss", vrpt["loss"], epoch)
            writer.add_scalar("valid/macro_f1", v_macro_f1, epoch)
            if v_auc >= 0:
                writer.add_scalar("valid/macro_auc_ovr", v_auc, epoch)

        # Step LR / SWA annealing
        if swa_enabled and epoch >= swa_start_epoch and swa_scheduler is not None:
            swa_scheduler.step()
        else:
            scheduler.step()

        # Early stop / checkpoints
        best_path = Path("checkpoints") / "best.pt"
        last_path = Path("checkpoints") / "last.pt"
        improved = v_macro_f1 > best_macro_f1
        if improved:
            best_macro_f1 = v_macro_f1
            epochs_no_improve = 0
            try:
                torch.save({"epoch": epoch + 1, "state_dict": eval_model.state_dict(),
                            "classes": classes, "config": cfg},
                           best_path)
                logger.success(f"[ckpt] New BEST: {best_path}")
            except Exception as e:
                logger.warning(f"[ckpt] Save failed: {e}")
        else:
            epochs_no_improve += 1

        try:
            torch.save({"epoch": epoch + 1, "state_dict": model.state_dict(),
                        "classes": classes, "config": cfg, "optimizer": optimizer.state_dict()},
                       last_path)
            logger.info(f"[ckpt] Saved LAST: {last_path}")
        except Exception as e:
            logger.warning(f"[ckpt] LAST save failed: {e}")

        if epochs_no_improve >= int(cfg["train"]["early_stopping_patience"]):
            logger.info("[early] Stopping early due to no improvement.")
            break

        if device.type == "cuda":
            torch.cuda.empty_cache()

    final_report = vrpt

    if swa_enabled and swa_updates > 0 and swa_model is not None:
        logger.info("[swa] Finalising SWA model with BN update + validation.")

        def _bn_iterator():
            for batch in dl_train:
                data = batch["image"].to(device, non_blocking=True)
                if use_channels_last:
                    data = data.to(memory_format=torch.channels_last)
                yield data, None

        try:
            update_bn(_bn_iterator(), swa_model, device=device)
        except Exception as e:
            logger.warning(f"[swa] BN update failed: {e}")

        swa_report = run_validation(
            swa_model, dl_valid, device, classes, use_tta=bool(cfg["train"].get("tta_valid", False))
        )
        swa_macro_f1 = float(swa_report["report"]["macro avg"]["f1-score"])
        swa_acc = float(swa_report["report"]["accuracy"])
        logger.info(f"[swa] validation: acc={swa_acc:.4f} macroF1={swa_macro_f1:.4f}")

        swa_path = Path("checkpoints") / "swa.pt"
        try:
            torch.save({"epoch": total_epochs, "state_dict": swa_model.state_dict(), "classes": classes, "config": cfg}, swa_path)
            logger.info(f"[ckpt] Saved SWA weights: {swa_path}")
        except Exception as e:
            logger.warning(f"[ckpt] Failed to save SWA checkpoint: {e}")

        if swa_macro_f1 > best_macro_f1:
            best_macro_f1 = swa_macro_f1
            best_path = Path("checkpoints") / "best.pt"
            try:
                torch.save({"epoch": total_epochs, "state_dict": swa_model.state_dict(), "classes": classes, "config": cfg}, best_path)
                logger.success(f"[ckpt] SWA model promoted to BEST: {best_path}")
            except Exception as e:
                logger.warning(f"[ckpt] Failed to overwrite BEST with SWA: {e}")
        final_report = swa_report

    # write final report
    rep_path = Path("checkpoints") / "valid_report.json"
    try:
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)
        logger.info(f"[report] Validation metrics saved to {rep_path}")
    except Exception as e:
        logger.warning(f"[report] Failed to save validation report: {e}")

if __name__ == "__main__":
    main()
