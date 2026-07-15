"""Robustness battery: re-run a checkpoint over a corruption suite.

For each corruption (jpeg q40, q20, motion blur, low light, WB drift, noise),
record macro_f1, balanced_acc, and primary-class AUROC. Output one row per
(corruption, metric) and a summary plot.

    python -m scripts.corruption_sweep \
        --config configs/convnext_tiny.yaml \
        --ckpt checkpoints/run0/best.pt \
        --split valid --fold 0 \
        --out-dir checkpoints/run0/robustness
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from mnkai.data.datasets import SkinDataset
from mnkai.data.manifest import LabelNormalizer, normalize_manifest
from mnkai.data.splits import external_test_indices, fold_train_valid_indices
from mnkai.data.transforms import AugConfig, build_valid
from mnkai.evaluation.corruption import corruption_suite
from mnkai.models.backbones import ModelConfig, build_model
from mnkai.utils.config import load_config
from mnkai.utils.logging import setup_logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=("valid", "external_test"), default="valid")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging("logs", "corruption_sweep.log", "INFO")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

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
    if args.split == "valid":
        _, idx = fold_train_valid_indices(manifest, args.fold)
    else:
        idx = external_test_indices(manifest)
    sub = manifest.iloc[idx].reset_index(drop=True)

    # model
    mcfg = ModelConfig(name=cfg["train"]["model_name"], pretrained=False, num_classes=len(classes))
    model = build_model(mcfg).to(device).eval()
    obj = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(obj.get("model_state") or obj.get("state_dict") or obj, strict=False)

    aug = AugConfig(
        img_size=int(cfg["data"]["img_size"]),
        mean=tuple(cfg["data"].get("normalize_mean", (0.485, 0.456, 0.406))),
        std=tuple(cfg["data"].get("normalize_std", (0.229, 0.224, 0.225))),
        shades_of_gray_p=float(cfg["augmentation"].get("shades_of_gray_p", 1.0)),
        sog_norm=float(cfg["augmentation"].get("sog_norm", 6.0)),
        hair_removal=False,
    )
    ds = SkinDataset(sub, classes, transform=build_valid(aug),
                     images_dir=cfg["data"].get("images_dir") or None)
    dl = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]) * 2, shuffle=False,
                    num_workers=int(cfg["hardware"].get("num_workers", 4)), pin_memory=True)

    primary_idx = classes.index(cfg.get("primary_class", "mel"))
    rows: list[dict] = []
    suite = corruption_suite()
    for cname, fn in suite.items():
        logger.info(f"[corruption] {cname}")
        all_y, all_prob = [], []
        with torch.no_grad():
            for batch in dl:
                x = batch["image"].to(device, non_blocking=True)
                y = batch["target"]
                xc = fn(x)
                logits = model(xc)
                all_prob.append(F.softmax(logits, dim=-1).cpu().numpy())
                all_y.append(y.numpy())
        yt = np.concatenate(all_y)
        prob = np.concatenate(all_prob)
        yp = prob.argmax(1)
        pos = (yt == primary_idx).astype(int)
        try:
            auroc = float(roc_auc_score(pos, prob[:, primary_idx]))
        except ValueError:
            auroc = float("nan")
        rows.append({
            "corruption": cname,
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "balanced_acc": float(balanced_accuracy_score(yt, yp)),
            f"auroc_{classes[primary_idx]}": auroc,
            "n": int(len(yt)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out / "corruption_breakdown.csv", index=False)
    with open(out / "corruption_breakdown.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=float)
    logger.info(f"[corruption] wrote {out/'corruption_breakdown.csv'}")
    logger.info("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
