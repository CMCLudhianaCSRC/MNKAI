"""Evaluate a trained checkpoint on a fold's validation split or the external test split.

Produces:
  - overall_metrics.json with bootstrapped 95% CIs for macro-F1, balanced
    accuracy, macro AUROC, primary-class AUROC.
  - subgroup_breakdown.csv (source / sex / age band / site / fitzpatrick).
  - per_class_metrics.csv.
  - confusion_matrix.png + ROC and PR curves.

Usage:
    python -m scripts.evaluate \
        --config configs/convnext_tiny.yaml \
        --ckpt checkpoints/run0/best.pt \
        --split valid --fold 0
    python -m scripts.evaluate \
        --config configs/convnext_tiny.yaml \
        --ckpt checkpoints/run0/best.pt \
        --split external_test
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

from mnkai.data.datasets import SkinDataset, TabularSpec
from mnkai.data.manifest import LabelNormalizer, normalize_manifest
from mnkai.data.splits import external_test_indices, fold_train_valid_indices
from mnkai.data.transforms import AugConfig, build_valid
from mnkai.evaluation.bootstrap import bootstrap_metric
from mnkai.evaluation.metrics import all_metrics
from mnkai.evaluation.subgroups import fitzpatrick_gap, subgroup_breakdown
from mnkai.models.backbones import ModelConfig, build_model
from mnkai.utils.config import load_config
from mnkai.utils.logging import setup_logging


def _load_ckpt(path: Path, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    return obj


@torch.no_grad()
def _predict(model, dl, device, use_tta: bool):
    model.eval()
    all_y, all_p = [], []
    for batch in dl:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["target"]
        tab = batch.get("tabular")
        if tab is not None:
            tab = tab.to(device, non_blocking=True)
        kwargs = {"tabular": tab} if tab is not None else {}
        logits = model(x, **kwargs) if kwargs else model(x)
        if use_tta:
            for dims in ([3], [2], [2, 3]):
                logits = logits + (model(torch.flip(x, dims=dims), **kwargs) if kwargs else model(torch.flip(x, dims=dims)))
            logits = logits / 4.0
        all_p.append(F.softmax(logits, dim=-1).cpu().numpy())
        all_y.append(y.numpy())
    return np.concatenate(all_y), np.concatenate(all_p)


def _macro_f1(yt, yp):
    return float(f1_score(yt, yp, average="macro", zero_division=0))


def _balanced_acc(yt, yp):
    return float(balanced_accuracy_score(yt, yp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=("valid", "external_test"), default="valid")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging("logs", "evaluate.log", "INFO")
    out_dir = Path(args.out_dir or Path(args.ckpt).parent / f"eval_{args.split}_fold{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = [c.lower() for c in cfg["classes"]["names"]]

    # data
    raw = pd.read_csv(cfg["data"]["manifest_csv"])
    manifest = normalize_manifest(raw, LabelNormalizer(classes=classes))
    splits = pd.read_csv(cfg["data"]["splits_csv"])
    if "filepath" in splits.columns:
        manifest = manifest.merge(splits[["filepath", "split", "fold"]], on="filepath", how="inner")
    elif "isic_id" in splits.columns and "isic_id" in manifest.columns:
        manifest = manifest.merge(splits[["isic_id", "split", "fold"]], on="isic_id", how="inner")
    else:
        raise ValueError("splits CSV must include 'filepath' or 'isic_id'")

    if args.split == "valid":
        _, idx = fold_train_valid_indices(manifest, args.fold)
    else:
        idx = external_test_indices(manifest)
    if len(idx) == 0:
        raise SystemExit(f"no rows found for split={args.split} fold={args.fold}")
    sub = manifest.iloc[idx].reset_index(drop=True)
    logger.info(f"[eval] {args.split}: n={len(sub)}")

    # model
    tabular = None
    tab_cfg = cfg.get("tabular", {}) or {}
    if bool(tab_cfg.get("enabled", False)):
        tabular = TabularSpec(
            use_sex=bool(tab_cfg.get("use_sex", True)),
            use_age=bool(tab_cfg.get("use_age", True)),
            use_site=bool(tab_cfg.get("use_site", True)),
            use_fitzpatrick=bool(tab_cfg.get("use_fitzpatrick", False)),
        )
    mcfg = ModelConfig(
        name=cfg["train"]["model_name"], pretrained=False, num_classes=len(classes),
        drop_rate=0.0, drop_path_rate=0.0, head_dropout=0.0,
        tabular_dim=tabular.dim if tabular is not None else 0,
        tabular_hidden=int(cfg["train"].get("tabular_hidden", 64)),
        head_hidden=int(cfg["train"].get("head_hidden", 0)),
    )
    model = build_model(mcfg).to(device)
    ckpt = _load_ckpt(Path(args.ckpt), device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    logger.info(f"[eval] loaded {args.ckpt}")

    aug = AugConfig(
        img_size=int(cfg["data"]["img_size"]),
        mean=tuple(cfg["data"].get("normalize_mean", (0.485, 0.456, 0.406))),
        std=tuple(cfg["data"].get("normalize_std", (0.229, 0.224, 0.225))),
        shades_of_gray_p=float(cfg["augmentation"].get("shades_of_gray_p", 1.0)),
        sog_norm=float(cfg["augmentation"].get("sog_norm", 6.0)),
        hair_removal=False,
    )
    ds = SkinDataset(sub, classes, transform=build_valid(aug),
                     images_dir=cfg["data"].get("images_dir") or None, tabular=tabular)
    dl = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]) * 2, shuffle=False,
                    num_workers=int(cfg["hardware"].get("num_workers", 4)),
                    pin_memory=bool(cfg["hardware"].get("pin_memory", True)))

    yt, prob = _predict(model, dl, device, use_tta=args.tta)
    yp = prob.argmax(1)

    metrics = all_metrics(yt, yp, prob, classes)

    # --- bootstrapped CIs
    primary_class = cfg.get("primary_class", "mel").lower()
    primary_idx = classes.index(primary_class)
    f1_ci = bootstrap_metric(yt, yp, _macro_f1, n_resamples=args.n_bootstrap, seed=0, stratify=yt)
    ba_ci = bootstrap_metric(yt, yp, _balanced_acc, n_resamples=args.n_bootstrap, seed=0, stratify=yt)
    # primary-class AUROC bootstrap
    pos = (yt == primary_idx).astype(int)
    if pos.sum() > 0 and pos.sum() < len(pos):
        auc_ci = bootstrap_metric(
            pos, prob[:, primary_idx],
            lambda y, s: float(roc_auc_score(y, s)),
            n_resamples=args.n_bootstrap, seed=0,
        )
    else:
        auc_ci = None

    metrics["macro_f1_ci95"] = [f1_ci.lo, f1_ci.hi]
    metrics["balanced_accuracy_ci95"] = [ba_ci.lo, ba_ci.hi]
    if auc_ci is not None:
        metrics[f"auroc_{primary_class}_ci95"] = [auc_ci.lo, auc_ci.hi]
    metrics["n"] = int(len(yt))

    with open(out_dir / "overall_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)

    pd.DataFrame(
        [{"class": c, **metrics["report"][c]} for c in classes if c in metrics["report"]]
    ).to_csv(out_dir / "per_class_metrics.csv", index=False)

    # --- subgroup
    sg = subgroup_breakdown(sub, yt, yp, prob, classes=classes, primary_class=primary_class,
                            n_bootstrap=min(args.n_bootstrap, 1000))
    sg.to_csv(out_dir / "subgroup_breakdown.csv", index=False)
    gap = fitzpatrick_gap(sg, primary_class=primary_class)
    with open(out_dir / "fitzpatrick_gap.json", "w", encoding="utf-8") as f:
        json.dump(gap, f, indent=2, default=float)

    # save raw predictions
    np.savez_compressed(out_dir / "predictions.npz", y_true=yt, y_pred=yp, y_prob=prob,
                        filepaths=sub.get("filepath", pd.Series([])).to_numpy())

    logger.info(
        f"[eval] {args.split} | macro_f1={metrics['macro_f1']:.4f} {f1_ci} "
        f"bal_acc={metrics['balanced_accuracy']:.4f} {ba_ci} "
        f"AUROC_{primary_class}={metrics.get(f'auroc_{primary_class}_ci95', '-')}"
    )


if __name__ == "__main__":
    main()
