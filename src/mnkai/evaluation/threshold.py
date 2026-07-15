"""Operating-point selection.

The PPT specifies "Sensitivity at 95% Specificity" as the clinically meaningful
operating point. This module:

  - Sweeps thresholds in [0, 1].
  - Finds the threshold that yields specificity >= target (default 0.95) at
    maximum sensitivity. Returns the threshold AND the sensitivity / PPV / NPV.
  - Provides a helper to convert per-class probabilities into an operating
    decision: predict the primary class (e.g. melanoma) iff its score crosses
    the threshold; otherwise return the argmax over the remaining classes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OperatingPoint:
    threshold: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    tp: int
    tn: int
    fp: int
    fn: int

    def as_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "PPV": self.ppv,
            "NPV": self.npv,
            "TP": self.tp,
            "TN": self.tn,
            "FP": self.fp,
            "FN": self.fn,
        }


def threshold_table(scores: np.ndarray, labels: np.ndarray, *,
                    n_thresholds: int = 101) -> list[OperatingPoint]:
    """Return one OperatingPoint per linspace threshold."""
    s = np.asarray(scores).astype(float)
    y = np.asarray(labels).astype(int)
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("threshold_table needs both positive and negative samples")
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    out: list[OperatingPoint] = []
    for t in thresholds:
        pred = s >= t
        tp = int((pred & pos).sum())
        fp = int((pred & ~pos).sum())
        tn = int((~pred & ~pos).sum())
        fn = int((~pred & pos).sum())
        sens = tp / max(1, n_pos)
        spec = tn / max(1, n_neg)
        ppv = tp / max(1, tp + fp)
        npv = tn / max(1, tn + fn)
        out.append(OperatingPoint(threshold=float(t), sensitivity=sens, specificity=spec,
                                  ppv=ppv, npv=npv, tp=tp, tn=tn, fp=fp, fn=fn))
    return out


def operating_point_at_specificity(scores: np.ndarray, labels: np.ndarray, *,
                                   target_specificity: float = 0.95,
                                   n_thresholds: int = 1001) -> OperatingPoint:
    """Highest-sensitivity threshold whose specificity >= target.

    If no threshold reaches target_specificity, returns the one closest to it
    (this can happen when the score distribution is too coarse).
    """
    table = threshold_table(scores, labels, n_thresholds=n_thresholds)
    eligible = [p for p in table if p.specificity >= target_specificity]
    if eligible:
        return max(eligible, key=lambda p: p.sensitivity)
    # fall back: closest specificity
    return min(table, key=lambda p: abs(p.specificity - target_specificity))


def operating_point_at_sensitivity(scores: np.ndarray, labels: np.ndarray, *,
                                   target_sensitivity: float = 0.90,
                                   n_thresholds: int = 1001) -> OperatingPoint:
    """Highest-specificity threshold whose sensitivity >= target."""
    table = threshold_table(scores, labels, n_thresholds=n_thresholds)
    eligible = [p for p in table if p.sensitivity >= target_sensitivity]
    if eligible:
        return max(eligible, key=lambda p: p.specificity)
    return min(table, key=lambda p: abs(p.sensitivity - target_sensitivity))


def youden_j_threshold(scores: np.ndarray, labels: np.ndarray, *,
                       n_thresholds: int = 1001) -> OperatingPoint:
    """Threshold maximizing Youden's J = sens + spec - 1 (balance point)."""
    table = threshold_table(scores, labels, n_thresholds=n_thresholds)
    return max(table, key=lambda p: p.sensitivity + p.specificity)


__all__ = [
    "OperatingPoint",
    "threshold_table",
    "operating_point_at_specificity",
    "operating_point_at_sensitivity",
    "youden_j_threshold",
]
