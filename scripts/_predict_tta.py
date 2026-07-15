"""Test-time augmentation (TTA) inference helper.

For each input, averages logits across 5 augmented views: original, horizontal
flip, vertical flip, both flips, and 90° rotation. The average is computed in
LOGIT space (more stable than averaging probabilities, especially for rare
classes).

Usage:
  python -m scripts._predict_tta \
      --config configs/phase1_dinov2_mnkai3.yaml \
      --ckpt   checkpoints/phase1_dinov2_fold0/best.pt \
      --fold   0 \
      --out    checkpoints/phase1_dinov2_fold0/posthoc/val_logits_tta.npz
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnkai.data.datasets import SkinDataset, TabularSpec
from mnkai.data.manifest import LabelNormalizer, normalize_manifest
from mnkai.data.splits import fold_train_valid_indices
from mnkai.data.transforms import AugConfig, build_valid
from mnkai.models.backbones import ModelConfig, build_model
from mnkai.utils.config import load_config


def _tta_logits(model, x, tab):
    """Average logits across 5 deterministic augmented views."""
    kwargs = {"tabular": tab} if tab is not None else {}
    views = [
        x,
        torch.flip(x, dims=[3]),       # hflip
        torch.flip(x, dims=[2]),       # vflip
        torch.flip(x, dims=[2, 3]),    # hflip+vflip
        torch.rot90(x, k=1, dims=[2, 3]),  # 90° rotation
    ]
    total = None
    for v in views:
        l = model(v, **kwargs) if kwargs else model(v)
        total = l if total is None else total + l
    return total / float(len(views))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=32)  # smaller bc 5× memory
    ap.add_argument("--split", choices=["valid", "train"], default="valid")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.fold is not None:
        cfg["train"]["fold"] = args.fold
    fold = int(cfg["train"].get("fold", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = list(cfg["classes"]["names"])

    # tabular
    tab_cfg = cfg.get("tabular", {}) or {}
    tabular = None
    if tab_cfg.get("enabled", False):
        tabular = TabularSpec(
            use_sex=tab_cfg.get("use_sex", True),
            use_age=tab_cfg.get("use_age", True),
            use_site=tab_cfg.get("use_site", True),
            use_fitzpatrick=tab_cfg.get("use_fitzpatrick", False),
        )

    # model
    mcfg = ModelConfig(
        name=cfg["train"]["model_name"], pretrained=False,
        num_classes=len(classes),
        drop_rate=float(cfg["train"].get("drop_rate", 0.10)),
        drop_path_rate=float(cfg["train"].get("drop_path_rate", 0.10)),
        head_dropout=float(cfg["train"].get("head_dropout", 0.30)),
        tabular_dim=(tabular.dim if tabular else 0),
        tabular_hidden=int(cfg["train"].get("tabular_hidden", 128)),
        head_hidden=int(cfg["train"].get("head_hidden", 0)),
        img_size=int(cfg["data"].get("img_size", 0)) or None,
        fusion=str(cfg["train"].get("fusion", "concat")),
        cross_attn_heads=int(cfg["train"].get("cross_attn_heads", 4)),
        cross_attn_dim=int(cfg["train"].get("cross_attn_dim", 256)),
    )
    model = build_model(mcfg).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("ema_state") or ckpt.get("model_state") or ckpt, strict=False)
    print(f"[tta] loaded {args.ckpt}")

    # manifest + split
    norm = LabelNormalizer(classes=classes)
    manifest = pd.read_csv(cfg["data"]["manifest_csv"], low_memory=False)
    manifest = normalize_manifest(manifest, normalizer=norm)
    splits = pd.read_csv(cfg["data"]["splits_csv"], low_memory=False)
    train_idx, valid_idx = fold_train_valid_indices(splits, fold=fold)
    idx = valid_idx if args.split == "valid" else train_idx
    df = manifest.iloc[idx].reset_index(drop=True)
    print(f"[tta] {args.split} set: {len(df):,}")

    # transform — match training's aug_cfg_valid exactly (SoG match!)
    aug = AugConfig(
        img_size=int(cfg["data"]["img_size"]),
        mean=tuple(cfg["data"].get("normalize_mean", [0.485, 0.456, 0.406])),
        std=tuple(cfg["data"].get("normalize_std", [0.229, 0.224, 0.225])),
        shades_of_gray_p=float(cfg.get("augmentation", {}).get("shades_of_gray_p", 1.0)),
        sog_norm=float(cfg.get("augmentation", {}).get("sog_norm", 6.0)),
        hair_removal=False,
    )
    tfm = build_valid(aug)
    ds = SkinDataset(df, classes=classes, transform=tfm, images_dir=None, tabular=tabular)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2,
                     pin_memory=True)

    logits_all, labels_all = [], []
    t0 = time.time()
    for i, batch in enumerate(dl):
        x = batch["image"].to(device, non_blocking=True)
        tab = batch.get("tabular")
        if tab is not None:
            tab = tab.to(device, non_blocking=True)
        l = _tta_logits(model, x, tab)
        logits_all.append(l.float().cpu().numpy())
        labels_all.append(batch["target"].numpy())
        if (i + 1) % 50 == 0:
            print(f"[tta]  {(i+1)*args.batch_size:,}/{len(df):,} "
                  f"({(time.time()-t0)/60:.1f}m)")
    logits = np.concatenate(logits_all)
    labels = np.concatenate(labels_all)
    print(f"[tta] done; elapsed={(time.time()-t0)/60:.1f}m")

    out_path = Path(args.out or (Path(args.ckpt).parent / "posthoc" /
                                 f"{args.split}_logits_tta.npz"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, logits=logits, labels=labels)
    print(f"[tta] saved -> {out_path}")


if __name__ == "__main__":
    main()
