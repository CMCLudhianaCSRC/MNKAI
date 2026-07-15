"""Distill an ensemble of teachers into the mobile student (MobileNetV3-Large).

Standard knowledge-distillation:
    L = (1 - w) * CE(student, hard_label) + w * T^2 * KL(softmax(s/T) || softmax(t/T))
where t is the mean of the teacher logits.

    python -m scripts.distill \
        --student-config configs/mobilenetv3_large.yaml \
        --teachers checkpoints/convnext_tiny_fold0/best.pt,checkpoints/effnet_b3_fold0/best.pt
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from mnkai.data.datasets import SkinDataset
from mnkai.data.manifest import LabelNormalizer, normalize_manifest
from mnkai.data.splits import fold_train_valid_indices
from mnkai.data.transforms import AugConfig, build_train, build_valid
from mnkai.evaluation.metrics import all_metrics
from mnkai.models.backbones import ModelConfig, build_model
from mnkai.training.sampler import make_weighted_sampler
from mnkai.utils.config import load_config
from mnkai.utils.logging import setup_logging
from mnkai.utils.seed import seed_worker, set_seed, torch_generator


def _load_teacher(ckpt_path: str, num_classes: int, device) -> torch.nn.Module:
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = obj.get("config", {})
    name = cfg.get("train", {}).get("model_name") or "convnext_tiny.fb_in22k_ft_in1k"
    mcfg = ModelConfig(name=name, pretrained=False, num_classes=num_classes,
                       drop_rate=0.0, drop_path_rate=0.0, head_dropout=0.0)
    m = build_model(mcfg).to(device).eval()
    m.load_state_dict(obj.get("model_state") or obj["state_dict"], strict=False)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-config", required=True)
    ap.add_argument("--teachers", required=True, help="comma-separated teacher checkpoints")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--teacher-weight", type=float, default=None)
    args = ap.parse_args()

    setup_logging("logs", "distill.log", "INFO")
    cfg = load_config(args.student_config)
    distill_cfg = cfg["train"].get("distillation", {}) or {}
    T = float(args.temperature or distill_cfg.get("temperature", 4.0))
    w = float(args.teacher_weight or distill_cfg.get("teacher_weight", 0.7))

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = [c.lower() for c in cfg["classes"]["names"]]

    # data
    raw = pd.read_csv(cfg["data"]["manifest_csv"])
    manifest = normalize_manifest(raw, LabelNormalizer(classes=classes))
    splits = pd.read_csv(cfg["data"]["splits_csv"])
    if "filepath" in splits.columns:
        manifest = manifest.merge(splits[["filepath", "split", "fold"]], on="filepath", how="inner")
    elif "isic_id" in splits.columns:
        manifest = manifest.merge(splits[["isic_id", "split", "fold"]], on="isic_id", how="inner")
    fold = int(cfg["train"].get("fold", 0))
    tr_idx, va_idx = fold_train_valid_indices(manifest, fold)

    aug = AugConfig(img_size=int(cfg["data"]["img_size"]),
                    mean=tuple(cfg["data"].get("normalize_mean", (0.485, 0.456, 0.406))),
                    std=tuple(cfg["data"].get("normalize_std", (0.229, 0.224, 0.225))),
                    hair_removal=False)
    train_df = manifest.iloc[tr_idx].reset_index(drop=True)
    valid_df = manifest.iloc[va_idx].reset_index(drop=True)
    ds_tr = SkinDataset(train_df, classes, transform=build_train(aug))
    ds_va = SkinDataset(valid_df, classes, transform=build_valid(aug))

    sampler = make_weighted_sampler(train_df, classes, seed=seed)
    g = torch_generator(seed)
    bs = int(cfg["train"]["batch_size"])
    dl_tr = DataLoader(ds_tr, batch_size=bs, sampler=sampler, num_workers=4,
                       pin_memory=True, worker_init_fn=seed_worker, generator=g)
    dl_va = DataLoader(ds_va, batch_size=bs * 2, shuffle=False, num_workers=4,
                       pin_memory=True, worker_init_fn=seed_worker, generator=g)

    # student
    mcfg = ModelConfig(
        name=cfg["train"]["model_name"], pretrained=True, num_classes=len(classes),
        drop_rate=float(cfg["train"].get("drop_rate", 0.20)),
        drop_path_rate=float(cfg["train"].get("drop_path_rate", 0.10)),
        head_dropout=float(cfg["train"].get("head_dropout", 0.30)),
    )
    student = build_model(mcfg).to(device)
    if device.type == "cuda":
        student = student.to(memory_format=torch.channels_last)

    # teachers
    teacher_paths = [p.strip() for p in args.teachers.split(",") if p.strip()]
    teachers = [_load_teacher(p, len(classes), device) for p in teacher_paths]
    logger.info(f"[distill] T={T} w_teacher={w} teachers={len(teachers)}")

    optim = AdamW(student.parameters(), lr=float(cfg["train"]["lr"]),
                  weight_decay=float(cfg["train"].get("weight_decay", 5e-5)))
    total_epochs = int(cfg["train"]["epochs"])
    warmup = int(cfg["train"].get("warmup_epochs", 2))
    def _lr(epoch): return ((epoch+1) / max(1, warmup)) if epoch < warmup else \
        0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, total_epochs - warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, _lr)

    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(total_epochs):
        student.train()
        t0 = time.time()
        for batch in tqdm(dl_tr, desc=f"distill {epoch+1}/{total_epochs}", ncols=100):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    teacher_logits = torch.stack([t(x) for t in teachers], dim=0).mean(0)
                s_logits = student(x)
                hard = F.cross_entropy(s_logits, y, label_smoothing=0.05)
                soft = F.kl_div(F.log_softmax(s_logits / T, dim=-1),
                                F.softmax(teacher_logits / T, dim=-1),
                                reduction="batchmean") * (T * T)
                loss = (1 - w) * hard + w * soft
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        sched.step()

        # eval
        student.eval()
        all_y, all_p = [], []
        with torch.no_grad():
            for batch in dl_va:
                x = batch["image"].to(device, non_blocking=True)
                logits = student(x)
                all_p.append(F.softmax(logits, dim=-1).cpu().numpy())
                all_y.append(batch["target"].numpy())
        yt, prob = np.concatenate(all_y), np.concatenate(all_p)
        m = all_metrics(yt, prob.argmax(1), prob, classes)
        logger.info(f"[distill] epoch {epoch+1} val_macro_f1={m['macro_f1']:.4f} "
                    f"acc={m['accuracy']:.4f} elapsed={time.time()-t0:.1f}s")
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            torch.save({"epoch": epoch+1, "model_state": student.state_dict(),
                        "config": cfg, "metrics": m}, out / "best.pt")


if __name__ == "__main__":
    main()
