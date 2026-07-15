"""Screening-mode per-class thresholds: find the threshold that gives at
least `min_sens` sensitivity for each class, maximising specificity.

Two operating modes are useful clinically:
  - Specialist mode (Youden's J)  : balanced sens/spec, current default
  - Screening mode  (sens >= X)   : high recall, accept more false alarms

For melanoma in particular, the FDA / clinical literature targets sens >= 95%
at spec >= 70-80%. This script computes those thresholds from cached
val_logits.npz produced by _posthoc_calibrate.py.

Outputs:
  posthoc/screening_thresholds.json
    { 'mel':   {threshold: 0.04, sens: 0.95, spec: 0.78},
      'bkl':   {threshold: 0.06, sens: 0.90, spec: 0.72},
      ... }

Usage:
  python -m scripts._screening_thresholds \
      --config configs/phase1_dinov2_mnkai3.yaml \
      --min-sens-mel 0.95 \
      --min-sens-bkl 0.90 \
      --min-sens-other 0.85
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mnkai.utils.config import load_config


def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def threshold_for_sens(y_true_binary: np.ndarray, y_score: np.ndarray, min_sens: float):
    """Find the largest threshold (= highest specificity) such that the
    achieved sensitivity is at least `min_sens`. Returns (threshold, sens, spec).
    """
    fpr, tpr, thr = roc_curve(y_true_binary, y_score)
    # Walk thresholds from most strict (high) to least strict (low). For each
    # threshold, sens = tpr, spec = 1 - fpr. Find first idx where tpr >= min_sens.
    feasible = tpr >= min_sens
    if not feasible.any():
        # Even threshold 0 doesn't reach min_sens — return the smallest
        i = int(np.argmax(tpr))
        return float(thr[i]), float(tpr[i]), float(1 - fpr[i])
    # Among feasible, pick the one with HIGHEST specificity = lowest fpr
    feasible_idx = np.where(feasible)[0]
    i = feasible_idx[np.argmin(fpr[feasible_idx])]
    return float(thr[i]), float(tpr[i]), float(1 - fpr[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--min-sens-mel", type=float, default=0.95)
    ap.add_argument("--min-sens-bkl", type=float, default=0.90)
    ap.add_argument("--min-sens-other", type=float, default=0.85)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir or cfg["out_dir"]) / "posthoc"
    classes = list(cfg["classes"]["names"])

    arr = np.load(out_dir / "val_logits.npz")
    logits, labels = arr["logits"], arr["labels"]
    T = json.load(open(out_dir / "temperature.json"))["temperature"]
    probs = _softmax(logits / T)
    print(f"[screen] loaded {len(logits):,} val logits, T={T:.3f}, classes={classes}")

    # Per-class min-sens map (mel and bkl get higher targets)
    targets = {}
    for cls in classes:
        if cls == "mel":
            targets[cls] = args.min_sens_mel
        elif cls == "bkl":
            targets[cls] = args.min_sens_bkl
        else:
            targets[cls] = args.min_sens_other

    out = {}
    for k, cls in enumerate(classes):
        y_true = (labels == k).astype(int)
        if y_true.sum() == 0:
            continue
        y_score = probs[:, k]
        thr, sens, spec = threshold_for_sens(y_true, y_score, targets[cls])
        out[cls] = {
            "target_min_sens": targets[cls],
            "threshold":       thr,
            "achieved_sens":   sens,
            "achieved_spec":   spec,
            "support":         int(y_true.sum()),
        }
        flag = "" if sens >= targets[cls] else "  ** below target **"
        print(f"  {cls:>8}: target sens>={targets[cls]:.2f}  =>  "
              f"thr={thr:.4f}  sens={sens:.3f}  spec={spec:.3f}{flag}")

    out_path = out_dir / "screening_thresholds.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[screen] saved -> {out_path}")


if __name__ == "__main__":
    main()
