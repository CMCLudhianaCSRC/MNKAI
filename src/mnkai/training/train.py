"""Unified trainer for the new MNKAI pipeline.

Differences from the legacy `src/train.py`:

  - Group-aware splits: reads precomputed `data/splits/<name>.csv` (built by
    scripts.make_splits) instead of doing label-only StratifiedShuffleSplit.
    No patient bridges train and validation.

  - Real `cb_focal`: the loss factory in `mnkai.losses` actually implements
    class-balanced focal; unknown names raise instead of silently falling
    through to plain CE.

  - Multimodal-ready: if `model.tabular_dim > 0` and the dataset emits a
    `tabular` tensor, it is fed to `ImageTabularNet`.

  - Reproducible: `torch.Generator(seed)` for the DataLoader, optional
    deterministic mode, and per-run config + git hash snapshot.

  - Per-step BF16 autocast on Ampere+, channels_last memory format, optional
    grad checkpointing for tight VRAM.
"""
from __future__ import annotations

import json
import math
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mnkai.data.datasets import SkinDataset, TabularSpec
from mnkai.data.manifest import LabelNormalizer, class_counts, normalize_manifest
from mnkai.data.splits import fold_train_valid_indices, merge_manifest_splits
from mnkai.data.transforms import AugConfig, build_train, build_valid
from mnkai.evaluation.metrics import all_metrics
from mnkai.losses import build_loss
from mnkai.models.backbones import ModelConfig, build_model
from mnkai.training.ema import ModelEMA
from mnkai.training.sampler import make_weighted_sampler
from mnkai.training.schedulers import build_scheduler
from mnkai.utils.logging import setup_logging
from mnkai.utils.seed import seed_worker, set_seed, torch_generator


# --------------------------------------------------------------- mixup utils --

def _rand_bbox(w: int, h: int, lam: float) -> tuple[int, int, int, int]:
    cut_ratio = math.sqrt(max(0.0, 1.0 - lam))
    cw, ch = max(1, int(w * cut_ratio)), max(1, int(h * cut_ratio))
    cx = np.random.randint(0, w)
    cy = np.random.randint(0, h)
    x1 = max(0, cx - cw // 2)
    y1 = max(0, cy - ch // 2)
    x2 = min(w, cx + cw // 2)
    y2 = min(h, cy + ch // 2)
    return x1, y1, x2, y2


# (R20) Reproducibility of mix augmentation — reviewed, no change required.
# These helpers draw from the GLOBAL numpy/torch RNGs (np.random.beta, torch.randperm).
# That is reproducible here because fit_one_fold() calls set_seed(seed, deterministic=...)
# before the training loop, and set_seed seeds np.random, torch, and torch.cuda. So a run
# with a fixed seed replays the same mixup/cutmix draws. Threading an explicit Generator
# through would add plumbing without changing reproducibility; the one caveat is that
# these draws share global RNG state with any other global-RNG consumer in the loop, so
# adding or removing such a consumer shifts the mix stream (it stays deterministic, but
# is not stable across code changes).
def _mixup(x: torch.Tensor, y: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0 or x.size(0) < 2:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def _cutmix(x: torch.Tensor, y: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0 or x.size(0) < 2:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, h, w = x.shape
    x1, y1, x2, y2 = _rand_bbox(w, h, lam)
    out = x.clone()
    out[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam_adj = 1.0 - ((x2 - x1) * (y2 - y1) / max(1, w * h))
    return out, y, y[idx], lam_adj


def _maybe_mix(x, y, cfg, epoch, total_epochs):
    p = float(cfg["train"].get("mix_prob", 0.0))
    if p <= 0:
        return x, None, None, 1.0, "none"
    s = float(cfg["train"].get("mix_start_frac", 0.0))
    e = float(cfg["train"].get("mix_end_frac", 0.9))
    se, ee = int(total_epochs * s), int(total_epochs * e)
    if epoch < se or epoch >= ee:
        return x, None, None, 1.0, "none"
    # linearly anneal probability
    frac = (epoch - se) / max(1, ee - se)
    if torch.rand(1).item() > p * (1.0 - frac):
        return x, None, None, 1.0, "none"
    cmx = float(cfg["train"].get("cutmix_alpha", 0.0))
    mix = float(cfg["train"].get("mixup_alpha", 0.0))
    use_cutmix = cmx > 0 and (mix <= 0 or torch.rand(1).item() < float(cfg["train"].get("cutmix_prob", 0.5)))
    if use_cutmix:
        out, ya, yb, lam = _cutmix(x, y, cmx)
        return out, ya, yb, lam, "cutmix"
    out, ya, yb, lam = _mixup(x, y, mix)
    return out, ya, yb, lam, "mixup"


# ----------------------------------------------------------------- runners --


def _build_loaders(cfg: dict, manifest: pd.DataFrame, classes: list[str], tabular: TabularSpec | None,
                   train_idx: np.ndarray, valid_idx: np.ndarray, seed: int):
    img_size = int(cfg["data"]["img_size"])
    aug_cfg_train = AugConfig(
        img_size=img_size,
        mean=tuple(cfg["data"].get("normalize_mean", (0.485, 0.456, 0.406))),
        std=tuple(cfg["data"].get("normalize_std", (0.229, 0.224, 0.225))),
        shades_of_gray_p=float(cfg["augmentation"].get("shades_of_gray_p", 1.0)),
        sog_norm=float(cfg["augmentation"].get("sog_norm", 6.0)),
        hair_removal=bool(cfg["augmentation"].get("hair_removal", False)),
        hair_removal_p=float(cfg["augmentation"].get("hair_removal_p", 0.5)),
        hair_removal_max_side=int(cfg["augmentation"].get("hair_removal_max_side", 768)),
        hflip=float(cfg["augmentation"]["train"].get("hflip", 0.5)),
        vflip=float(cfg["augmentation"]["train"].get("vflip", 0.2)),
        rotate=int(cfg["augmentation"]["train"].get("rotate", 25)),
        random_crop_scale=tuple(cfg["augmentation"]["train"].get("random_crop_scale", (0.8, 1.0))),
        brightness=float(cfg["augmentation"]["train"].get("brightness", 0.15)),
        contrast=float(cfg["augmentation"]["train"].get("contrast", 0.15)),
        saturation=float(cfg["augmentation"]["train"].get("saturation", 0.10)),
        hue=float(cfg["augmentation"]["train"].get("hue", 0.03)),
        blur=float(cfg["augmentation"]["train"].get("blur", 0.15)),
        motion_blur=float(cfg["augmentation"]["train"].get("motion_blur", 0.15)),
        jpeg_lower=int(cfg["augmentation"]["train"].get("jpeg_lower", 60)),
        erase_p=float(cfg["augmentation"]["train"].get("erase_p", 0.30)),
        erase_max_holes=int(cfg["augmentation"]["train"].get("erase_max_holes", 3)),
        erase_max_frac=float(cfg["augmentation"]["train"].get("erase_max_frac", 0.18)),
    )
    aug_cfg_valid = AugConfig(img_size=img_size,
                              mean=aug_cfg_train.mean, std=aug_cfg_train.std,
                              shades_of_gray_p=aug_cfg_train.shades_of_gray_p, sog_norm=aug_cfg_train.sog_norm,
                              hair_removal=False)

    train_df = manifest.iloc[train_idx].reset_index(drop=True)
    valid_df = manifest.iloc[valid_idx].reset_index(drop=True)

    images_dir = cfg["data"].get("images_dir") or None
    train_ds = SkinDataset(train_df, classes, transform=build_train(aug_cfg_train),
                           images_dir=images_dir, tabular=tabular)
    valid_ds = SkinDataset(valid_df, classes, transform=build_valid(aug_cfg_valid),
                           images_dir=images_dir, tabular=tabular)

    # Optional: per-row sample weights from hard-negative mining
    sample_w = None
    hardneg_csv = cfg["train"].get("hardneg_weights_csv")
    if hardneg_csv and Path(hardneg_csv).is_file():
        try:
            hn = pd.read_csv(hardneg_csv)
            if "row_index" in hn.columns and "weight" in hn.columns:
                w_by_idx = dict(zip(hn["row_index"].astype(int), hn["weight"].astype(float)))
                sample_w = np.array([w_by_idx.get(i, 1.0) for i in range(len(train_df))],
                                     dtype=np.float64)
                logger.info(f"[train] hardneg weights loaded ({len(w_by_idx)} rows; mean={sample_w.mean():.3f}, "
                            f"max={sample_w.max():.2f})")
        except Exception as e:
            logger.warning(f"[train] failed to load hardneg weights: {e}")

    # Imbalance handling: the sampler re-balances WHICH samples are seen. Use
    # ONE balancing mechanism only — stacking a full-inverse-freq sampler
    # (sampler_power=1.0) with cb_focal caused the Phase-3 majority-class
    # collapse. `sampler_power` softens it (0.5 = sqrt sampling); set
    # `use_weighted_sampler: false` for natural frequency (pair with a loss-side
    # method like logit_adjusted / balanced-softmax).
    use_sampler = bool(cfg["train"].get("use_weighted_sampler", True))
    sampler_power = float(cfg["train"].get("sampler_power", 1.0))
    if use_sampler:
        sampler = make_weighted_sampler(
            train_df, classes,
            source_col=cfg["data"].get("source_col") or None,
            source_weights=cfg["data"].get("source_weights") or None,
            class_boost=cfg["train"].get("class_boost") or None,
            sample_weights=sample_w,
            power=sampler_power,
            seed=seed,
        )
        logger.info(f"[train] weighted sampler ON (power={sampler_power})")
    else:
        sampler = None
        logger.info("[train] weighted sampler OFF (natural frequency)")
    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["hardware"].get("num_workers", 4))
    g = torch_generator(seed)
    pin_memory = bool(cfg["hardware"].get("pin_memory", True))
    persistent = bool(cfg["hardware"].get("persistent_workers", False)) and nw > 0
    pf = int(cfg["hardware"].get("prefetch_factor", 2))

    dl_train = DataLoader(
        train_ds, batch_size=bs, sampler=sampler, shuffle=(sampler is None),
        num_workers=nw,
        pin_memory=pin_memory, persistent_workers=persistent,
        prefetch_factor=pf if nw > 0 else None,
        worker_init_fn=seed_worker, generator=g,
    )
    dl_valid = DataLoader(
        valid_ds, batch_size=max(1, bs * 2), shuffle=False, num_workers=nw,
        pin_memory=pin_memory, persistent_workers=persistent,
        prefetch_factor=pf if nw > 0 else None,
        worker_init_fn=seed_worker, generator=g,
    )
    return dl_train, dl_valid, train_df, valid_df


@torch.no_grad()
def _validate(model: torch.nn.Module, dl: DataLoader, device: torch.device, classes: list[str], use_tta: bool):
    model.eval()
    all_yt, all_yp, all_prob = [], [], []
    val_loss = 0.0
    crit = torch.nn.CrossEntropyLoss()
    for batch in dl:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        tab = batch.get("tabular")
        if tab is not None:
            tab = tab.to(device, non_blocking=True)
        kwargs = {"tabular": tab} if tab is not None else {}

        logits = model(x, **kwargs) if kwargs else model(x)
        if use_tta:
            for dims in ([3], [2], [2, 3]):
                logits = logits + (model(torch.flip(x, dims=dims), **kwargs) if kwargs
                                   else model(torch.flip(x, dims=dims)))
            logits = logits / 4.0
        loss = crit(logits, y)
        val_loss += float(loss.item())
        prob = F.softmax(logits, dim=-1).cpu().numpy()
        all_prob.append(prob)
        all_yt.append(y.cpu().numpy())
        all_yp.append(prob.argmax(1))

    yt = np.concatenate(all_yt) if all_yt else np.array([])
    yp = np.concatenate(all_yp) if all_yp else np.array([])
    prob = np.concatenate(all_prob) if all_prob else np.zeros((0, len(classes)))
    metrics = all_metrics(yt, yp, prob, classes)
    metrics["loss"] = float(val_loss / max(1, len(dl)))
    return metrics, yt, yp, prob


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    """Single-fold training loop. Returns best validation metrics."""
    out_dir = Path(cfg.get("out_dir", "checkpoints/run0"))
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.get("logging", {}).get("log_dir", "logs"),
                  cfg.get("logging", {}).get("log_file", "train.log"),
                  cfg.get("logging", {}).get("level", "INFO"))

    seed = int(cfg.get("seed", 42))
    set_seed(seed, deterministic=bool(cfg.get("deterministic", False)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"[train] GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
        # Performance: TF32 matmul + cuDNN autotune. Pure speedup on Ampere;
        # safe because the config runs with deterministic=False.
        if not bool(cfg.get("deterministic", False)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            logger.info("[train] perf: TF32 on, cuDNN benchmark on")

    # ---- manifest + splits
    manifest_csv = Path(cfg["data"]["manifest_csv"])
    raw = pd.read_csv(manifest_csv)
    classes = [c.lower() for c in cfg["classes"]["names"]]
    norm = LabelNormalizer(classes=classes)
    manifest = normalize_manifest(raw, norm)

    splits_csv = Path(cfg["data"]["splits_csv"])
    splits = pd.read_csv(splits_csv)
    # Fail-loud merge guard (R11): normalizes the join key on both sides and
    # aborts if a key-format mismatch silently shrinks the training set.
    manifest = merge_manifest_splits(manifest, splits)
    fold = int(cfg["train"].get("fold", 0))
    train_idx, valid_idx = fold_train_valid_indices(manifest, fold)
    logger.info(f"[train] fold={fold} train={len(train_idx)} valid={len(valid_idx)}")

    # ---- tabular
    tab_cfg = cfg.get("tabular", {}) or {}
    tabular = None
    tabular_dim = 0
    if bool(tab_cfg.get("enabled", False)):
        tabular = TabularSpec(
            use_sex=bool(tab_cfg.get("use_sex", True)),
            use_age=bool(tab_cfg.get("use_age", True)),
            use_site=bool(tab_cfg.get("use_site", True)),
            use_fitzpatrick=bool(tab_cfg.get("use_fitzpatrick", False)),
        )
        tabular_dim = tabular.dim
        logger.info(f"[train] tabular features enabled, dim={tabular_dim}")

    # ---- loaders
    dl_train, dl_valid, train_df, valid_df = _build_loaders(
        cfg, manifest, classes, tabular, train_idx, valid_idx, seed=seed,
    )
    counts = class_counts(train_df, classes)
    logger.info(f"[train] class counts: {dict(zip(classes, counts))}")

    # ---- model
    mcfg = ModelConfig(
        name=cfg["train"]["model_name"],
        pretrained=bool(cfg["train"].get("pretrained", True)),
        num_classes=len(classes),
        drop_rate=float(cfg["train"].get("drop_rate", 0.10)),
        drop_path_rate=float(cfg["train"].get("drop_path_rate", 0.10)),
        head_dropout=float(cfg["train"].get("head_dropout", 0.30)),
        tabular_dim=tabular_dim,
        tabular_hidden=int(cfg["train"].get("tabular_hidden", 64)),
        head_hidden=int(cfg["train"].get("head_hidden", 0)),
        # DINOv2 needs explicit img_size override (native is 518)
        img_size=int(cfg["data"].get("img_size", 0)) or None,
        fusion=str(cfg["train"].get("fusion", "concat")),
        cross_attn_heads=int(cfg["train"].get("cross_attn_heads", 4)),
        cross_attn_dim=int(cfg["train"].get("cross_attn_dim", 256)),
    )
    model = build_model(mcfg).to(device)
    if bool(cfg["hardware"].get("channels_last", True)) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if bool(cfg["train"].get("grad_checkpointing", False)):
        backbone = getattr(model, "backbone", model)
        if hasattr(backbone, "set_grad_checkpointing"):
            backbone.set_grad_checkpointing(True)
            logger.info("[train] gradient checkpointing enabled")

    # ---- curriculum: resume from a previous-stage checkpoint
    # Holds richer state (optimizer / scheduler / ema / epoch / best_macro_f1)
    # if the checkpoint was saved with the hardened format.
    resume_from = cfg["train"].get("resume_from") or ""
    _resume_payload: dict = {}
    if isinstance(resume_from, str) and resume_from and Path(resume_from).is_file():
        try:
            obj = torch.load(resume_from, map_location=device, weights_only=False)
            sd = obj.get("model_state") or obj.get("state_dict") or obj
            missing, unexpected = model.load_state_dict(sd, strict=False)
            logger.info(f"[train] resumed weights from {resume_from} "
                        f"(missing={len(missing)}, unexpected={len(unexpected)})")
            _resume_payload = obj if isinstance(obj, dict) else {}
        except Exception as e:
            logger.warning(f"[train] resume_from failed: {e}; continuing fresh")

    # ---- init-from-backbone (transfer learning across a DIFFERENT class set)
    # Used by Phase 3: load a Phase-1 checkpoint's backbone + fusion weights but
    # REINITIALISE the classifier head (9-class -> N-class) and start a FRESH
    # fine-tune (epoch 0, new optimizer). Distinct from resume_from, which is a
    # same-task continuation that also restores optimizer/scheduler/epoch.
    # Guard: only applies on a genuinely fresh start (no real resume payload),
    # so the orchestrator's crash-resume of Phase 3 itself is unaffected.
    init_bb = cfg["train"].get("init_from_backbone") or ""
    if isinstance(init_bb, str) and ("{n}" in init_bb or "{fold}" in init_bb):
        init_bb = init_bb.format(n=fold, fold=fold)   # per-fold init (e.g. cRT from Fix-C fold{n})
    if not _resume_payload and isinstance(init_bb, str) and init_bb and Path(init_bb).is_file():
        try:
            obj = torch.load(init_bb, map_location=device, weights_only=False)
            sd = obj.get("ema_state") or obj.get("model_state") or obj.get("state_dict") or obj
            msd = model.state_dict()
            # shape-aware filter: keep only tensors that exist AND match shape
            filt = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
            # cRT / decoupling: force a FRESH classifier head even when the class
            # set (and thus head shape) matches — load backbone+fusion, retrain head.
            if bool(cfg["train"].get("reinit_head", False)):
                _dropped = [k for k in list(filt) if k.startswith("head.")]
                for _k in _dropped:
                    filt.pop(_k, None)
                logger.info(f"[train] reinit_head=True: dropped {len(_dropped)} head "
                            f"tensors -> fresh classifier (cRT).")
            skipped = [k for k in sd if k not in filt]
            model.load_state_dict(filt, strict=False)
            logger.info(f"[train] init-from-backbone {init_bb}: loaded {len(filt)} "
                        f"tensors, skipped {len(skipped)} mismatched (head reinit). "
                        f"Fresh fine-tune from epoch 0.")
        except Exception as e:
            logger.warning(f"[train] init_from_backbone failed: {e}; random head + backbone")

    # ---- curriculum: optionally freeze backbone for N epochs
    freeze_n = int(cfg["train"].get("freeze_backbone_epochs", 0))
    backbone_module = getattr(model, "backbone", None)

    # ---- loss (supports per-class focal gamma + per-class boost on cb_focal)
    loss_name = cfg["train"]["loss"]
    # focal_gamma may be float (uniform) OR list (per-class). Cast accordingly.
    _fg = cfg["train"].get("focal_gamma", 2.0)
    if isinstance(_fg, (list, tuple)):
        focal_gamma = [float(x) for x in _fg]
        logger.info(f"[train] per-class focal_gamma = {focal_gamma}")
    else:
        focal_gamma = float(_fg)
    class_boost_loss = cfg["train"].get("class_boost_loss")
    if class_boost_loss is not None:
        class_boost_loss = [float(x) for x in class_boost_loss]
        logger.info(f"[train] per-class loss boost = {class_boost_loss}")
    _needs_counts = ("cb_focal", "ldam", "logit_adjusted", "balanced_softmax", "la", "bsm")
    crit = build_loss(
        loss_name,
        num_classes=len(classes),
        class_counts=counts if loss_name in _needs_counts else None,
        label_smoothing=float(cfg["train"].get("label_smoothing", 0.05)),
        focal_gamma=focal_gamma,
        focal_alpha=cfg["train"].get("focal_alpha"),
        cb_beta=float(cfg["train"].get("cb_beta", 0.9999)),
        class_boost=class_boost_loss,
        la_tau=float(cfg["train"].get("la_tau", 1.0)),
        ldam_max_m=float(cfg["train"].get("ldam_max_m", 0.5)),
        ldam_s=float(cfg["train"].get("ldam_s", 30.0)),
    ).to(device)

    # ---- optional knowledge-distillation teacher (e.g. fold-0 best.pt)
    # When kd_teacher_ckpt is set, load a frozen teacher; the student then
    # gets a KL-divergence soft-label loss in addition to the hard-label loss:
    #   L_total = (1 - kd_alpha) * L_hard  +  kd_alpha * T^2 * KL(student||teacher)
    # Reference: Hinton 2015 "Distilling the Knowledge in a Neural Network".
    teacher_model = None
    kd_alpha = float(cfg["train"].get("kd_alpha", 0.0))
    kd_temp = float(cfg["train"].get("kd_temperature", 4.0))
    kd_ckpt_path = cfg["train"].get("kd_teacher_ckpt") or ""
    if kd_alpha > 0 and kd_ckpt_path and Path(kd_ckpt_path).is_file():
        try:
            teacher_model = build_model(mcfg).to(device).eval()
            t_obj = torch.load(kd_ckpt_path, map_location=device, weights_only=False)
            t_sd = t_obj.get("model_state") or t_obj.get("ema_state") or t_obj
            teacher_model.load_state_dict(t_sd, strict=False)
            for p in teacher_model.parameters():
                p.requires_grad_(False)
            logger.info(f"[train] KD teacher loaded from {kd_ckpt_path} "
                        f"(alpha={kd_alpha:.2f}, T={kd_temp:.1f})")
        except Exception as e:
            logger.warning(f"[train] KD teacher load failed: {e}; disabling KD")
            teacher_model = None
            kd_alpha = 0.0

    # ---- optim + scheduler (cosine OR SGDR per config)
    optim = AdamW(model.parameters(),
                  lr=float(cfg["train"]["lr"]),
                  weight_decay=float(cfg["train"].get("weight_decay", 1e-4)))
    total_epochs = int(cfg["train"]["epochs"])
    warmup = int(cfg["train"].get("warmup_epochs", 2))
    sched = build_scheduler(
        cfg["train"].get("scheduler", "cosine"),
        optim, total_epochs,
        warmup_epochs=warmup,
        sgdr_first_cycle=int(cfg["train"].get("sgdr_first_cycle", 10)),
        sgdr_cycle_mult=float(cfg["train"].get("sgdr_cycle_mult", 2.0)),
        min_lr_ratio=float(cfg["train"].get("min_lr_ratio", 0.01)),
    )

    # ---- AMP
    amp_dtype_str = (cfg["hardware"].get("amp_dtype") or "bf16").lower()
    autocast_dtype = torch.bfloat16 if amp_dtype_str.startswith("bf") else torch.float16
    use_amp = bool(cfg["hardware"].get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and autocast_dtype == torch.float16)

    # ---- EMA
    ema = ModelEMA(model, decay=float(cfg["train"].get("ema_decay", 0.999))) if bool(cfg["train"].get("ema", True)) else None

    # ---- TB
    tb_dir = Path(cfg.get("tensorboard", {}).get("log_dir", "runs")) / out_dir.name
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    grad_accum = max(1, int(cfg["train"].get("grad_accum_steps", 1)))
    max_grad_norm = float(cfg["train"].get("max_grad_norm", 1.0))
    early_patience = int(cfg["train"].get("early_stopping_patience", 7))
    best_macro_f1 = -1.0
    epochs_no_improve = 0
    start_epoch = 0    # default: train from scratch
    interrupted = {"flag": False}

    # ---- restore richer state from resume checkpoint (if present)
    if _resume_payload:
        try:
            if "optimizer_state" in _resume_payload:
                optim.load_state_dict(_resume_payload["optimizer_state"])
                logger.info("[resume] optimizer state restored")
            if "scheduler_state" in _resume_payload and sched is not None:
                sched.load_state_dict(_resume_payload["scheduler_state"])
                logger.info("[resume] scheduler state restored")
            if "scaler_state" in _resume_payload and scaler is not None:
                scaler.load_state_dict(_resume_payload["scaler_state"])
                logger.info("[resume] AMP scaler state restored")
            if "ema_state" in _resume_payload and ema is not None:
                ema.module.load_state_dict(_resume_payload["ema_state"])
                logger.info("[resume] EMA state restored")
            if "epoch" in _resume_payload:
                start_epoch = int(_resume_payload["epoch"])
                logger.info(f"[resume] continuing from epoch {start_epoch+1}")
            if "best_macro_f1" in _resume_payload:
                best_macro_f1 = float(_resume_payload["best_macro_f1"])
                logger.info(f"[resume] best_macro_f1 baseline = {best_macro_f1:.4f}")
            if "epochs_no_improve" in _resume_payload:
                epochs_no_improve = int(_resume_payload["epochs_no_improve"])
        except Exception as e:
            logger.warning(f"[resume] partial state restore failed: {e}")

    def _sigint(signum, frame):
        interrupted["flag"] = True
        logger.warning("[signal] Ctrl-C received; will save emergency checkpoint")
    try:
        signal.signal(signal.SIGINT, _sigint)
    except Exception:
        pass

    # save run config snapshot
    with open(out_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        import yaml
        yaml.safe_dump(cfg, f, sort_keys=False)

    final_metrics: dict = {}

    for epoch in range(start_epoch, total_epochs):
        # curriculum: freeze backbone for the first N epochs, then unfreeze
        if freeze_n > 0 and backbone_module is not None:
            should_freeze = epoch < freeze_n
            for p in backbone_module.parameters():
                p.requires_grad_(not should_freeze)
            if epoch == 0:
                logger.info(f"[train] backbone frozen for first {freeze_n} epochs")
            elif epoch == freeze_n:
                logger.info("[train] backbone unfrozen")

        model.train()
        t0 = time.time()
        n_steps = len(dl_train)
        loop = tqdm(dl_train, total=n_steps, desc=f"epoch {epoch+1}/{total_epochs}", ncols=110)
        train_loss_sum = 0.0
        optim.zero_grad(set_to_none=True)

        for step, batch in enumerate(loop, start=1):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            tab = batch.get("tabular")
            if tab is not None:
                tab = tab.to(device, non_blocking=True)
            if bool(cfg["hardware"].get("channels_last", True)) and device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)

            inputs, ya, yb, lam, mode = _maybe_mix(x, y, cfg, epoch, total_epochs)
            if bool(cfg["hardware"].get("channels_last", True)) and device.type == "cuda":
                inputs = inputs.contiguous(memory_format=torch.channels_last)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=autocast_dtype):
                logits = model(inputs, tabular=tab) if tab is not None else model(inputs)
                if mode != "none":
                    loss_hard = lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)
                else:
                    loss_hard = crit(logits, y)
                if teacher_model is not None and kd_alpha > 0:
                    with torch.no_grad():
                        t_logits = (teacher_model(inputs, tabular=tab) if tab is not None
                                    else teacher_model(inputs))
                    # KL-div soft loss with temperature; (T^2) factor preserves
                    # gradient magnitudes at high T per Hinton 2015.
                    log_p_s = F.log_softmax(logits / kd_temp, dim=-1)
                    p_t = F.softmax(t_logits.detach() / kd_temp, dim=-1)
                    loss_kd = F.kl_div(log_p_s, p_t, reduction="batchmean") * (kd_temp ** 2)
                    loss = (1.0 - kd_alpha) * loss_hard + kd_alpha * loss_kd
                else:
                    loss = loss_hard
                loss = loss / grad_accum

            if not torch.isfinite(loss):
                logger.warning(f"[train] non-finite loss at step {step}; skipping")
                optim.zero_grad(set_to_none=True)
                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step % grad_accum) == 0 or step == n_steps:
                if max_grad_norm > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optim); scaler.update()
                else:
                    optim.step()
                optim.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)

            train_loss_sum += float(loss.item() * grad_accum)
            loop.set_postfix(loss=f"{loss.item()*grad_accum:.4f}",
                             lr=f"{optim.param_groups[0]['lr']:.2e}", mix=mode)

            if interrupted["flag"]:
                emer = out_dir / "emergency.pt"
                torch.save({"epoch": epoch, "step": step,
                            "model_state": model.state_dict(), "config": cfg},
                           emer)
                logger.warning(f"[signal] saved {emer}; exiting")
                writer.close()
                return final_metrics

        sched.step()
        train_loss = train_loss_sum / max(1, n_steps)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/lr", optim.param_groups[0]["lr"], epoch)

        # --- validate (use EMA if available)
        eval_model = ema.module if ema is not None else model
        v_metrics, _yt, _yp, _prob = _validate(
            eval_model, dl_valid, device, classes,
            use_tta=bool(cfg["train"].get("tta_valid", False)),
        )
        elapsed = time.time() - t0
        logger.info(
            f"[epoch {epoch+1}] loss={train_loss:.4f} | "
            f"val_loss={v_metrics['loss']:.4f} acc={v_metrics['accuracy']:.4f} "
            f"bal_acc={v_metrics['balanced_accuracy']:.4f} "
            f"macroF1={v_metrics['macro_f1']:.4f} "
            f"AUROC={v_metrics.get('macro_auroc_ovr', float('nan')):.4f} | "
            f"elapsed={elapsed:.1f}s"
        )
        for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auroc_ovr",
                  "weighted_f1", "loss", "ece"):
            if k in v_metrics and not (isinstance(v_metrics[k], float) and np.isnan(v_metrics[k])):
                writer.add_scalar(f"valid/{k}", v_metrics[k], epoch)

        # Hardened checkpoint format: includes optimizer, scheduler, scaler, EMA,
        # epoch counter, and best_macro_f1 so resume can pick up EXACTLY where
        # training stopped after a reboot / power loss / Ctrl-C.
        def _full_checkpoint(model_to_save, is_best: bool) -> dict:
            ckpt = {
                "epoch":            epoch + 1,
                "model_state":      model_to_save.state_dict(),
                "classes":          classes,
                "config":           cfg,
                "metrics":          v_metrics,
                "optimizer_state":  optim.state_dict(),
                "best_macro_f1":    best_macro_f1,
                "epochs_no_improve": epochs_no_improve,
            }
            try:
                if sched is not None:
                    ckpt["scheduler_state"] = sched.state_dict()
            except Exception:
                pass
            try:
                if scaler is not None:
                    ckpt["scaler_state"] = scaler.state_dict()
            except Exception:
                pass
            try:
                if ema is not None:
                    ckpt["ema_state"] = ema.module.state_dict()
            except Exception:
                pass
            return ckpt

        improved = v_metrics["macro_f1"] > best_macro_f1
        if improved:
            best_macro_f1 = v_metrics["macro_f1"]
            epochs_no_improve = 0
            torch.save(_full_checkpoint(eval_model, is_best=True), out_dir / "best.pt")
            logger.info(f"[train] saved best (macro_f1={best_macro_f1:.4f})")
            with open(out_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                json.dump(v_metrics, f, indent=2)
            final_metrics = v_metrics
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_patience:
                logger.info(f"[train] early stopping at epoch {epoch+1}")
                break

        # Always save last.pt with full state for crash-safe resume
        torch.save(_full_checkpoint(model, is_best=False), out_dir / "last.pt")

        # Optional: per-epoch snapshot for SWA averaging across the convergence
        # plateau. Enable in config with: train.save_epoch_ckpts_after: 30
        # Will save ckpt_epoch_NN.pt for epoch >= threshold, using the EMA state.
        save_after = int(cfg["train"].get("save_epoch_ckpts_after", -1))
        if save_after >= 0 and (epoch + 1) >= save_after:
            ep_ckpt = out_dir / f"ckpt_epoch_{epoch+1:02d}.pt"
            torch.save(_full_checkpoint(eval_model, is_best=False), ep_ckpt)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    writer.close()
    logger.info(f"[train] done; best macro_f1 = {best_macro_f1:.4f}")
    return final_metrics
