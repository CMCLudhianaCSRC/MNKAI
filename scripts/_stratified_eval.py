"""Stratified evaluation: per-FST, per-anatomical-site, per-source.

Required for the fairness publication story (Daneshjou 2022). Identifies
which subgroups the model underperforms on — critical for FDA SaMD review.

Run after training. Loads cached val_logits.npz from _posthoc_calibrate.py
and emits per-subgroup metric tables.

Outputs:
  posthoc/fst_stratified.json     : metrics by Fitzpatrick (1-3 vs 4-6)
  posthoc/source_stratified.json  : metrics by source (ISIC vs PAD vs SCIN vs ...)
  posthoc/site_stratified.json    : metrics by anatomical site (head, torso, etc.)
  posthoc/fairness_report.md      : human-readable summary
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score, accuracy_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnkai.utils.config import load_config
from mnkai.data.manifest import LabelNormalizer, normalize_manifest
from mnkai.data.splits import fold_train_valid_indices


def safe_auroc(y_true, y_score, labels, name=""):
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score, multi_class="ovr",
                                   average="macro", labels=labels))
    except Exception:
        return None


def subgroup_metrics(probs, labels, mask, classes):
    """Compute metrics on the subset where mask=True."""
    if mask.sum() < 20:
        return {"n": int(mask.sum()), "note": "subgroup too small (< 20)"}
    sub_probs = probs[mask]
    sub_labels = labels[mask]
    preds = np.argmax(sub_probs, axis=1)
    out = {
        "n":                  int(mask.sum()),
        "accuracy":           float(accuracy_score(sub_labels, preds)),
        "balanced_accuracy":  float(balanced_accuracy_score(sub_labels, preds)),
        "macro_f1":           float(f1_score(sub_labels, preds, average="macro",
                                              zero_division=0, labels=range(len(classes)))),
        "weighted_f1":        float(f1_score(sub_labels, preds, average="weighted",
                                              zero_division=0, labels=range(len(classes)))),
        "macro_auroc":        safe_auroc(sub_labels, sub_probs, range(len(classes))),
    }
    # Per-class for the subgroup
    out["per_class_f1"] = {}
    for k, cls in enumerate(classes):
        cls_mask = sub_labels == k
        if cls_mask.sum() > 0:
            cls_preds = preds[cls_mask] == k
            cls_f1 = float(f1_score((sub_labels == k).astype(int),
                                     (preds == k).astype(int), zero_division=0))
            out["per_class_f1"][cls] = {
                "support": int(cls_mask.sum()),
                "recall":  float(cls_preds.mean()),
                "f1":      cls_f1,
            }
    return out


def _fst_bucket(fst_raw):
    """Normalize FST values to '1-3' (light) or '4-6' (dark) buckets.
    Accepts numeric (1-6), strings (FST1..FST6, I..VI, etc.)."""
    if pd.isna(fst_raw):
        return None
    s = str(fst_raw).strip().upper().replace("FST", "").replace(" ", "")
    # Roman numerals
    roman = {"I":1,"II":2,"III":3,"IV":4,"V":5,"VI":6}
    if s in roman:
        v = roman[s]
    else:
        try:
            v = int(float(s))
        except Exception:
            return None
    if v <= 3:
        return "1-3"
    elif v <= 6:
        return "4-6"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--fold", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.fold is not None:
        cfg["train"]["fold"] = args.fold
    fold = int(cfg["train"].get("fold", 0))
    out_dir = Path(args.out_dir or cfg["out_dir"]) / "posthoc"

    # Load cached logits and val manifest
    arr = np.load(out_dir / "val_logits.npz")
    logits = arr["logits"]
    labels = arr["labels"]
    # Temperature-scaled probs
    T = json.load(open(out_dir / "temperature.json"))["temperature"]
    probs = _softmax(logits / T)
    print(f"[strat] loaded {len(logits):,} val logits, T={T:.3f}")

    classes = list(cfg["classes"]["names"])

    # Reconstruct val_df to align with logits ordering
    norm = LabelNormalizer(classes=classes)
    manifest = pd.read_csv(cfg["data"]["manifest_csv"], low_memory=False)
    manifest = normalize_manifest(manifest, normalizer=norm)
    splits = pd.read_csv(cfg["data"]["splits_csv"], low_memory=False)
    _, valid_idx = fold_train_valid_indices(splits, fold=fold)
    val_df = manifest.iloc[valid_idx].reset_index(drop=True)
    if len(val_df) != len(labels):
        print(f"[strat] WARN: manifest({len(val_df)}) != logits({len(labels)}); slicing")
        n = min(len(val_df), len(labels))
        val_df = val_df.iloc[:n]
        labels = labels[:n]
        probs = probs[:n]

    # ── FST stratification ──────────────────────────────────────────────────
    if "fitzpatrick" in val_df.columns:
        fst_bucket = val_df["fitzpatrick"].apply(_fst_bucket)
        fst_results = {}
        for bucket in ["1-3", "4-6", None]:
            mask = (fst_bucket == bucket).to_numpy() if bucket is not None \
                    else fst_bucket.isna().to_numpy()
            tag = bucket if bucket is not None else "missing_fst"
            fst_results[tag] = subgroup_metrics(probs, labels, mask, classes)
        # Disparity: 1-3 vs 4-6 gap
        if "1-3" in fst_results and "4-6" in fst_results \
                and isinstance(fst_results["1-3"], dict) \
                and isinstance(fst_results["4-6"], dict) \
                and "macro_auroc" in fst_results["1-3"] \
                and "macro_auroc" in fst_results["4-6"] \
                and fst_results["1-3"]["macro_auroc"] is not None \
                and fst_results["4-6"]["macro_auroc"] is not None:
            fst_results["disparity"] = {
                "auroc_gap_light_minus_dark":
                    fst_results["1-3"]["macro_auroc"] - fst_results["4-6"]["macro_auroc"],
                "balanced_acc_gap":
                    fst_results["1-3"]["balanced_accuracy"] - fst_results["4-6"]["balanced_accuracy"],
            }
        with open(out_dir / "fst_stratified.json", "w") as f:
            json.dump(fst_results, f, indent=2)
        print("[strat] FST stratification:")
        for k, v in fst_results.items():
            if isinstance(v, dict) and "macro_auroc" in v:
                print(f"  FST {k:>10}: n={v['n']:>6}, AUROC={v['macro_auroc'] or 0:.4f}, "
                      f"bal_acc={v['balanced_accuracy']:.4f}, macroF1={v['macro_f1']:.4f}")

    # ── Source stratification ──────────────────────────────────────────────
    if "source" in val_df.columns:
        src_results = {}
        for src in sorted(val_df["source"].dropna().unique()):
            mask = (val_df["source"] == src).to_numpy()
            src_results[str(src)] = subgroup_metrics(probs, labels, mask, classes)
        with open(out_dir / "source_stratified.json", "w") as f:
            json.dump(src_results, f, indent=2)
        print("\n[strat] Source stratification (sources with >=20 samples):")
        for src, v in sorted(src_results.items(),
                              key=lambda x: -x[1].get("n", 0) if isinstance(x[1], dict) else 0):
            if isinstance(v, dict) and "macro_auroc" in v and v.get("macro_auroc") is not None:
                print(f"  {src:>20}: n={v['n']:>6}, AUROC={v['macro_auroc']:.4f}, "
                      f"bal_acc={v['balanced_accuracy']:.4f}, macroF1={v['macro_f1']:.4f}")

    # ── Anatomical site stratification ─────────────────────────────────────
    site_col = next((c for c in val_df.columns
                      if "anatom" in c.lower() or c.lower().endswith("site")), None)
    if site_col:
        site_results = {}
        for site in sorted(val_df[site_col].dropna().unique()):
            mask = (val_df[site_col] == site).to_numpy()
            site_results[str(site)] = subgroup_metrics(probs, labels, mask, classes)
        with open(out_dir / "site_stratified.json", "w") as f:
            json.dump(site_results, f, indent=2)
        print(f"\n[strat] Anatomical site stratification (col={site_col}):")
        for site, v in sorted(site_results.items(),
                               key=lambda x: -x[1].get("n", 0) if isinstance(x[1], dict) else 0):
            if isinstance(v, dict) and "macro_auroc" in v and v.get("macro_auroc") is not None:
                print(f"  {site:>20}: n={v['n']:>6}, AUROC={v['macro_auroc']:.4f}, "
                      f"bal_acc={v['balanced_accuracy']:.4f}")


def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    main()
