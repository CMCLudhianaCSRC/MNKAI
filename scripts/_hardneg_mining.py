"""Hard-negative mining: identify worst-predicted samples + up-weight them.

After a model has trained to its initial convergence, the remaining error is
concentrated in HARD examples. This script:
  1. Runs inference on the full train set with the current best.pt
  2. Identifies the top-K hardest examples (lowest probability for true class)
  3. Saves a sample-weight CSV that the dataloader can use
  4. Recommends 3-5 additional epochs of training with these weights

Trade-off: tighter focus on hard cases vs. risk of overfitting to label noise.
The 5-retry fallback in our dataset already protects against label-noise cases.

Usage:
  python -m scripts._hardneg_mining --config ... --ckpt ... --out-weights ...
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnkai.utils.config import load_config
from mnkai.data.manifest import LabelNormalizer, normalize_manifest
from mnkai.data.splits import fold_train_valid_indices
from mnkai.data.datasets import SkinDataset, TabularSpec
from mnkai.data.transforms import AugConfig, build_valid
from mnkai.models.backbones import ModelConfig, build_model


@torch.no_grad()
def collect_train_probs(model, df, classes, cfg, device, batch_size=64):
    # CRITICAL: must match training's _build_loaders aug_cfg_valid (shades_of_gray_p).
    aug = AugConfig(
        img_size=int(cfg["data"]["img_size"]),
        mean=tuple(cfg["data"].get("normalize_mean", [0.485, 0.456, 0.406])),
        std=tuple(cfg["data"].get("normalize_std", [0.229, 0.224, 0.225])),
        shades_of_gray_p=float(cfg.get("augmentation", {}).get("shades_of_gray_p", 1.0)),
        sog_norm=float(cfg.get("augmentation", {}).get("sog_norm", 6.0)),
        hair_removal=False,
    )
    tfm = build_valid(aug)
    tab_cfg = cfg.get("tabular", {}) or {}
    tabular = None
    if tab_cfg.get("enabled", False):
        tabular = TabularSpec(use_sex=tab_cfg.get("use_sex",True),
                              use_age=tab_cfg.get("use_age",True),
                              use_site=tab_cfg.get("use_site",True),
                              use_fitzpatrick=tab_cfg.get("use_fitzpatrick",False))
    ds = SkinDataset(df, transform=tfm, classes=classes, images_dir=None, tabular=tabular)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    probs_all, labels_all, idx_all = [], [], []
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    model.eval()
    for batch in dl:
        x = batch["image"].to(device, non_blocking=True)
        tab = batch.get("tabular")
        if tab is not None:
            tab = tab.to(device, non_blocking=True)
            out = model(x, tab)
        else:
            out = model(x)
        probs = F.softmax(out.float(), dim=-1).cpu().numpy()
        probs_all.append(probs)
        labels_all.append(batch["target"].numpy())
        idx_all.append(batch["row_index"].numpy())
    return (np.concatenate(probs_all), np.concatenate(labels_all),
            np.concatenate(idx_all))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-weights", required=True,
                     help="CSV: row_index, weight")
    ap.add_argument("--top-k-percent", type=float, default=20.0,
                     help="Top-K% hardest examples get up-weighted")
    ap.add_argument("--up-weight", type=float, default=3.0,
                     help="Multiplier applied to hard examples (others = 1.0)")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--subsample", type=int, default=50000,
                     help="Subsample train for speed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.fold is not None:
        cfg["train"]["fold"] = args.fold
    fold = int(cfg["train"].get("fold", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = list(cfg["classes"]["names"])

    # ---- model
    tab_cfg = cfg.get("tabular", {}) or {}
    tab_dim = 0
    if tab_cfg.get("enabled", False):
        tab_dim = TabularSpec(use_sex=tab_cfg.get("use_sex",True),
                                use_age=tab_cfg.get("use_age",True),
                                use_site=tab_cfg.get("use_site",True),
                                use_fitzpatrick=tab_cfg.get("use_fitzpatrick",False)).dim
    mcfg = ModelConfig(
        name=cfg["train"]["model_name"], pretrained=False,
        num_classes=len(classes),
        drop_rate=float(cfg["train"].get("drop_rate", 0.10)),
        drop_path_rate=float(cfg["train"].get("drop_path_rate", 0.10)),
        head_dropout=float(cfg["train"].get("head_dropout", 0.30)),
        tabular_dim=tab_dim,
        tabular_hidden=int(cfg["train"].get("tabular_hidden", 128)),
        head_hidden=int(cfg["train"].get("head_hidden", 0)),
        img_size=int(cfg["data"].get("img_size", 0)) or None,
        fusion=str(cfg["train"].get("fusion", "concat")),
        cross_attn_heads=int(cfg["train"].get("cross_attn_heads", 4)),
        cross_attn_dim=int(cfg["train"].get("cross_attn_dim", 256)),
    )
    model = build_model(mcfg).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"[hardneg] loaded {args.ckpt}")

    # ---- manifest + train split
    norm = LabelNormalizer(classes=classes)
    manifest = pd.read_csv(cfg["data"]["manifest_csv"], low_memory=False)
    manifest = normalize_manifest(manifest, normalizer=norm)
    splits = pd.read_csv(cfg["data"]["splits_csv"], low_memory=False)
    train_idx, _ = fold_train_valid_indices(splits, fold=fold)
    train_df = manifest.iloc[train_idx].reset_index(drop=True)
    if args.subsample and len(train_df) > args.subsample:
        train_df = train_df.sample(n=args.subsample, random_state=42).reset_index(drop=True)
        print(f"[hardneg] subsampled train to {len(train_df):,}")

    # ---- score every train example
    probs, labels, idx = collect_train_probs(model, train_df, classes, cfg, device)

    # Hardness = 1 - p(true class)
    p_true = probs[np.arange(len(labels)), labels]
    hardness = 1.0 - p_true

    # Top-K% hardest
    k = int(args.top_k_percent / 100.0 * len(hardness))
    hard_mask = np.zeros(len(hardness), dtype=bool)
    hard_indices = np.argsort(-hardness)[:k]
    hard_mask[hard_indices] = True

    weights = np.ones(len(hardness))
    weights[hard_mask] = args.up_weight

    # Per-class summary of where the hardest examples come from
    print(f"\n[hardneg] top {args.top_k_percent}% hardest:")
    for k_idx, cls in enumerate(classes):
        cls_total = (labels == k_idx).sum()
        cls_hard = ((labels == k_idx) & hard_mask).sum()
        pct_hard = (cls_hard / max(1, cls_total)) * 100
        avg_p_true = p_true[labels == k_idx].mean() if cls_total > 0 else 0.0
        print(f"  {cls:>8}: {cls_hard}/{cls_total} ({pct_hard:.1f}%) hard, "
              f"avg p_true={avg_p_true:.3f}")

    # Save weights CSV
    out_df = pd.DataFrame({
        "row_index": idx,
        "hardness":  hardness,
        "weight":    weights,
        "is_hard":   hard_mask,
    })
    Path(args.out_weights).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_weights, index=False)
    print(f"\n[hardneg] saved weights -> {args.out_weights}")
    print(f"[hardneg] use in your sampler: WeightedRandomSampler(weights=...)")


if __name__ == "__main__":
    main()
