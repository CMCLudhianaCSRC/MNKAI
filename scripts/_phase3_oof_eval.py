"""Phase-3 out-of-fold (OOF) evaluation — the HONEST per-class report.

For each fold N: load checkpoints/phase3_indian_foldN/best.pt and run inference
on fold N's OWN held-out val set (which that model never trained on). Concatenate
all 5 folds' predictions -> every image is scored exactly once by a model that
didn't see it. Then compute:
  * overall accuracy, balanced accuracy, macro-F1
  * top-3 accuracy (referral-tool metric)
  * per-class precision/recall/F1/support
  * per-Fitzpatrick balanced accuracy (fairness: FST I-III vs IV-VI)

Outputs eval/phase3_oof/metrics.json + a printed report.

Run:
  PYTHONPATH=src .venv/Scripts/python.exe scripts/_phase3_oof_eval.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, precision_recall_fscore_support,
    accuracy_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mnkai.data.datasets import SkinDataset, TabularSpec
from mnkai.data.manifest import LabelNormalizer, normalize_manifest
from mnkai.data.splits import fold_train_valid_indices
from mnkai.data.transforms import AugConfig, build_valid
from mnkai.models.backbones import ModelConfig, build_model
from mnkai.utils.config import load_config

CONFIG = "configs/phase3_indian_25class.yaml"
CKPT_FMT = "checkpoints/phase3_indian_fold{n}/best.pt"
OUT = ROOT / "eval/phase3_oof"
FOLDS = [0, 1, 2, 3, 4]


def build(cfg, classes, tab, device):
    mcfg = ModelConfig(
        name=cfg["train"]["model_name"], pretrained=False, num_classes=len(classes),
        drop_rate=float(cfg["train"].get("drop_rate", 0.10)),
        drop_path_rate=float(cfg["train"].get("drop_path_rate", 0.10)),
        head_dropout=float(cfg["train"].get("head_dropout", 0.30)),
        tabular_dim=(tab.dim if tab else 0),
        tabular_hidden=int(cfg["train"].get("tabular_hidden", 128)),
        head_hidden=int(cfg["train"].get("head_hidden", 0)),
        img_size=int(cfg["data"].get("img_size", 0)) or None,
        fusion=str(cfg["train"].get("fusion", "concat")),
        cross_attn_heads=int(cfg["train"].get("cross_attn_heads", 4)),
        cross_attn_dim=int(cfg["train"].get("cross_attn_dim", 256)),
    )
    return build_model(mcfg).to(device).eval()


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(CONFIG)
    classes = list(cfg["classes"]["names"])
    tab_cfg = cfg.get("tabular", {}) or {}
    tab = TabularSpec(use_sex=tab_cfg.get("use_sex", True),
                      use_age=tab_cfg.get("use_age", True),
                      use_site=tab_cfg.get("use_site", True),
                      use_fitzpatrick=tab_cfg.get("use_fitzpatrick", True)) \
          if tab_cfg.get("enabled", False) else None

    norm = LabelNormalizer(classes=classes)
    manifest = pd.read_csv(cfg["data"]["manifest_csv"], low_memory=False)
    manifest = normalize_manifest(manifest, normalizer=norm)
    splits = pd.read_csv(cfg["data"]["splits_csv"], low_memory=False)
    if "filepath" in splits.columns:
        manifest = manifest.merge(splits[["filepath", "split", "fold"]], on="filepath", how="inner")

    aug = AugConfig(
        img_size=int(cfg["data"]["img_size"]),
        mean=tuple(cfg["data"].get("normalize_mean", [0.485, 0.456, 0.406])),
        std=tuple(cfg["data"].get("normalize_std", [0.229, 0.224, 0.225])),
        shades_of_gray_p=float(cfg.get("augmentation", {}).get("shades_of_gray_p", 0.0)),
        sog_norm=float(cfg.get("augmentation", {}).get("sog_norm", 6.0)),
        hair_removal=False)
    tfm = build_valid(aug)

    all_logits, all_labels, all_fst = [], [], []
    for n in FOLDS:
        cfg["train"]["fold"] = n
        _, vidx = fold_train_valid_indices(splits, fold=n)
        vdf = manifest.iloc[vidx].reset_index(drop=True)
        ds = SkinDataset(vdf, classes=classes, transform=tfm, images_dir=None, tabular=tab)
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=False)
        model = build(cfg, classes, tab, device)
        ck = torch.load(CKPT_FMT.format(n=n), map_location=device, weights_only=False)
        sd = ck.get("ema_state") or ck.get("model_state") or ck
        model.load_state_dict(sd, strict=False)
        t0 = time.time()
        logits = []
        for b in dl:
            x = b["image"].to(device, non_blocking=True)
            tb = b.get("tabular")
            if tb is not None: tb = tb.to(device)
            l = model(x, tabular=tb) if tb is not None else model(x)
            logits.append(l.float().cpu().numpy())
        all_logits.append(np.concatenate(logits))
        all_labels.append(vdf["label_name"].map({c: i for i, c in enumerate(classes)}).values)
        fst = vdf["fitzpatrick"] if "fitzpatrick" in vdf.columns else pd.Series([np.nan]*len(vdf))
        all_fst.append(fst.values)
        print(f"[oof] fold {n}: {len(vdf)} val imgs in {(time.time()-t0)/60:.1f}m", flush=True)
        del model
        if device.type == "cuda": torch.cuda.empty_cache()

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels).astype(int)
    fst = np.concatenate(all_fst)
    probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
    preds = probs.argmax(1)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    top3_acc = float(np.mean([labels[i] in top3[i] for i in range(len(labels))]))

    acc = float(accuracy_score(labels, preds))
    bal = float(balanced_accuracy_score(labels, preds))
    mf1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    p, r, f1c, sup = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(classes))), zero_division=0)

    per_class = sorted(
        [{"class": classes[i], "f1": round(float(f1c[i]), 3),
          "precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
          "support": int(sup[i])} for i in range(len(classes))],
        key=lambda d: d["f1"], reverse=True)

    # per-Fitzpatrick fairness (group by FST when present)
    fst_groups = {}
    fst_num = pd.to_numeric(pd.Series(fst), errors="coerce")
    for lo, hi, name in [(1, 3, "FST_I-III_light"), (4, 6, "FST_IV-VI_dark")]:
        m = (fst_num >= lo) & (fst_num <= hi)
        m = m.values
        if m.sum() >= 30:
            fst_groups[name] = {
                "n": int(m.sum()),
                "balanced_acc": round(float(balanced_accuracy_score(labels[m], preds[m])), 3),
                "macro_f1": round(float(f1_score(labels[m], preds[m], average="macro", zero_division=0)), 3),
                "accuracy": round(float(accuracy_score(labels[m], preds[m])), 3)}

    out = {
        "n_images": int(len(labels)), "n_classes": len(classes),
        "accuracy": round(acc, 4), "balanced_accuracy": round(bal, 4),
        "macro_f1": round(mf1, 4), "top3_accuracy": round(top3_acc, 4),
        "per_class": per_class, "fairness_by_fitzpatrick": fst_groups,
        "fst_coverage": int((~fst_num.isna()).sum()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2))

    print("\n========== PHASE-3 OOF EVALUATION ==========")
    print(f"images={out['n_images']}  classes={out['n_classes']}")
    print(f"accuracy={acc:.3f}  balanced_acc={bal:.3f}  macro_f1={mf1:.3f}  top3_acc={top3_acc:.3f}")
    print("\n-- per-class F1 (sorted) --")
    for d in per_class:
        print(f"  {d['class']:<22} F1={d['f1']:.3f}  P={d['precision']:.3f}  R={d['recall']:.3f}  n={d['support']}")
    print("\n-- fairness by Fitzpatrick --")
    if fst_groups:
        for k, v in fst_groups.items():
            print(f"  {k}: n={v['n']}  bal_acc={v['balanced_acc']}  macro_f1={v['macro_f1']}")
    else:
        print(f"  (insufficient FST labels; coverage={out['fst_coverage']}/{len(labels)})")
    print(f"\nsaved -> {OUT/'metrics.json'}")


if __name__ == "__main__":
    main()
