"""Risk-coverage and safe-abstain logic.

The PPT promises "risk-coverage (safe abstain mode) — model rejects
low-confidence predictions." This module:

  - Computes a risk-coverage curve: what error rate do we accept at coverage c?
  - Picks the abstention threshold that holds error <= target while maximizing
    coverage.
  - At inference, returns either a prediction OR an "abstain" sentinel.

The threshold can be confidence-on-max-class or per-class. Per-class is more
conservative for the primary (melanoma) class but harder to calibrate; we
expose both.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ABSTAIN = -1


@dataclass
class CoveragePoint:
    threshold: float
    coverage: float
    error_rate: float
    n_kept: int
    n_total: int


def risk_coverage_curve(probs: np.ndarray, labels: np.ndarray, *,
                        n_thresholds: int = 101) -> list[CoveragePoint]:
    """Sweep confidence thresholds; record (coverage, error among kept) for each."""
    p = np.asarray(probs)
    y = np.asarray(labels).astype(int)
    if p.ndim != 2:
        raise ValueError("probs must be (N, C)")
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    correct = (pred == y).astype(int)
    n = len(y)

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    out: list[CoveragePoint] = []
    for t in thresholds:
        kept = conf >= t
        nk = int(kept.sum())
        if nk == 0:
            err = 0.0
        else:
            err = 1.0 - correct[kept].mean()
        out.append(CoveragePoint(threshold=float(t), coverage=nk / max(1, n),
                                 error_rate=float(err), n_kept=nk, n_total=n))
    return out


def threshold_for_target_error(probs: np.ndarray, labels: np.ndarray, *,
                               target_error: float = 0.05,
                               min_coverage: float = 0.60,
                               n_thresholds: int = 1001) -> CoveragePoint:
    """Pick the threshold that keeps error_rate <= target_error AND coverage >= min_coverage,
    maximizing coverage.

    Fallback (never raises): if no threshold meets both constraints, relax the
    coverage floor and return the candidate whose error_rate is closest to
    `target_error` among those with coverage >= 0.10; if even that is empty,
    return the most conservative point (threshold 0.0). This function always
    returns a `CoveragePoint` and does not raise.
    """
    curve = risk_coverage_curve(probs, labels, n_thresholds=n_thresholds)
    eligible = [c for c in curve if c.error_rate <= target_error and c.coverage >= min_coverage]
    if eligible:
        return max(eligible, key=lambda c: c.coverage)
    # no eligible: relax coverage requirement and pick closest to target_error among
    # those with non-trivial coverage
    fallback = [c for c in curve if c.coverage >= 0.10]
    if fallback:
        return min(fallback, key=lambda c: abs(c.error_rate - target_error))
    return curve[0]


def predict_with_abstain(probs: np.ndarray, threshold: float,
                         per_class_thresholds: np.ndarray | None = None) -> np.ndarray:
    """Return predictions with `ABSTAIN` (-1) for low-confidence rows.

    If `per_class_thresholds` is given (length C), abstain when the predicted
    class's confidence is below its own threshold. Otherwise use the single
    `threshold` against the max-confidence.
    """
    p = np.asarray(probs)
    pred = p.argmax(axis=1)
    if per_class_thresholds is None:
        conf = p.max(axis=1)
        pred = np.where(conf >= threshold, pred, ABSTAIN)
        return pred
    per = np.asarray(per_class_thresholds, dtype=float)
    if per.shape != (p.shape[1],):
        raise ValueError("per_class_thresholds must have length C")
    confs = p[np.arange(len(p)), pred]
    cls_thr = per[pred]
    return np.where(confs >= cls_thr, pred, ABSTAIN)


def coverage_at_threshold(probs: np.ndarray, threshold: float) -> float:
    p = np.asarray(probs)
    return float((p.max(axis=1) >= threshold).mean())


__all__ = [
    "ABSTAIN",
    "CoveragePoint",
    "risk_coverage_curve",
    "threshold_for_target_error",
    "predict_with_abstain",
    "coverage_at_threshold",
]
