"""5-fold ensemble inference.

Averages logits (NOT probabilities) across multiple checkpoints. Logit-space
averaging is more stable than probability-space for rare classes, and matches
how the per-fold temperature scaling expects to be applied.

Usage:
  python -m scripts._ensemble_predict \
      --config configs/phase1_dinov2_mnkai3.yaml \
      --ckpts  checkpoints/phase1_dinov2_fold0/best.pt \
               checkpoints/phase1_dinov2_fold1/best.pt \
               checkpoints/phase1_dinov2_fold2/best.pt \
               checkpoints/phase1_dinov2_fold3/best.pt \
               checkpoints/phase1_dinov2_fold4/best.pt \
      --use-tta \
      --out    checkpoints/ensemble/val_logits.npz

Notes:
  - Each fold's val set is DIFFERENT (group-aware splits). For ensemble eval
    you typically use a held-out test set OR the union of all folds' val sets.
  - Pass --eval-on-fold to evaluate on a specific fold's val set (e.g. 0).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, f1_score, balanced_accuracy_score, classification_report,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnkai.data.datasets import SkinDataset, TabularSpec
from mnkai.data.manifest import LabelNormalizer, normalize_manifest
from mnkai.data.splits import fold_train_valid_indices
from mnkai.data.transforms import AugConfig, build_valid
from mnkai.models.backbones import ModelConfig, build_model
from mnkai.utils.config import load_config


def _build_model(cfg, classes, tabular, device):
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
    return build_model(mcfg).to(device).eval()


def _tta_logits(model, x, tab):
    kwargs = {"tabular": tab} if tab is not None else {}
    views = [x,
             torch.flip(x, dims=[3]),
             torch.flip(x, dims=[2]),
             torch.flip(x, dims=[2, 3]),
             torch.rot90(x, k=1, dims=[2, 3])]
    total = None
    for v in views:
        l = model(v, **kwargs) if kwargs else model(v)
        total = l if total is None else total + l
    return total / float(len(views))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--eval-on-fold", type=int, default=0,
                     help="Which fold's val set to use for evaluation")
    ap.add_argument("--use-tta", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["train"]["fold"] = args.eval_on_fold
    fold = args.eval_on_fold
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

    # manifest + val split
    norm = LabelNormalizer(classes=classes)
    manifest = pd.read_csv(cfg["data"]["manifest_csv"], low_memory=False)
    manifest = normalize_manifest(manifest, normalizer=norm)
    splits = pd.read_csv(cfg["data"]["splits_csv"], low_memory=False)
    _, valid_idx = fold_train_valid_indices(splits, fold=fold)
    val_df = manifest.iloc[valid_idx].reset_index(drop=True)
    print(f"[ens] eval on fold {fold} val set: {len(val_df):,}")

    aug = AugConfig(
        img_size=int(cfg["data"]["img_size"]),
        mean=tuple(cfg["data"].get("normalize_mean", [0.485, 0.456, 0.406])),
        std=tuple(cfg["data"].get("normalize_std", [0.229, 0.224, 0.225])),
        shades_of_gray_p=float(cfg.get("augmentation", {}).get("shades_of_gray_p", 1.0)),
        sog_norm=float(cfg.get("augmentation", {}).get("sog_norm", 6.0)),
        hair_removal=False,
    )
    tfm = build_valid(aug)
    ds = SkinDataset(val_df, classes=classes, transform=tfm, images_dir=None,
                     tabular=tabular)
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2,
                    pin_memory=True)

    # Iterate each ckpt; sum logits across folds
    sum_logits = None
    labels_ref = None
    for ck_path in args.ckpts:
        print(f"[ens] loading {ck_path}")
        model = _build_model(cfg, classes, tabular, device)
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        sd = ck.get("ema_state") or ck.get("model_state") or ck
        model.load_state_dict(sd, strict=False)

        logits_all, labels_all = [], []
        t0 = time.time()
        for batch in dl:
            x = batch["image"].to(device, non_blocking=True)
            tab = batch.get("tabular")
            if tab is not None:
                tab = tab.to(device, non_blocking=True)
            l = _tta_logits(model, x, tab) if args.use_tta else (
                model(x, tabular=tab) if tab is not None else model(x))
            logits_all.append(l.float().cpu().numpy())
            labels_all.append(batch["target"].numpy())
        logits_k = np.concatenate(logits_all)
        labels_k = np.concatenate(labels_all)
        sum_logits = logits_k.copy() if sum_logits is None else sum_logits + logits_k
        labels_ref = labels_k if labels_ref is None else labels_ref
        print(f"[ens]   forward done in {(time.time()-t0)/60:.1f}m "
              f"shape={logits_k.shape}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    avg_logits = sum_logits / float(len(args.ckpts))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, logits=avg_logits, labels=labels_ref, classes=np.array(classes))
    print(f"[ens] saved ensemble logits -> {out}")

    # Quick eval
    probs = F.softmax(torch.from_numpy(avg_logits), dim=-1).numpy()
    preds = probs.argmax(1)
    print(f"\n[ens] ENSEMBLE metrics (fold-{fold} val, {len(args.ckpts)} models, "
          f"TTA={'on' if args.use_tta else 'off'}):")
    print(f"  accuracy           : {(preds == labels_ref).mean():.4f}")
    print(f"  balanced_accuracy  : {balanced_accuracy_score(labels_ref, preds):.4f}")
    print(f"  macro_f1           : {f1_score(labels_ref, preds, average='macro', zero_division=0):.4f}")
    print(f"  macro_auroc        : {roc_auc_score(labels_ref, probs, multi_class='ovr', average='macro'):.4f}")
    print()
    print(classification_report(labels_ref, preds, target_names=classes, zero_division=0))


if __name__ == "__main__":
    main()
