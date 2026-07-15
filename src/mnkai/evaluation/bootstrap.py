"""Bootstrapped confidence intervals + paired tests.

  - bootstrap_metric : 95% CI for any scalar metric via percentile bootstrap.
  - mcnemar_test     : exact / mid-p McNemar for paired classifier comparison.
  - delong_test      : DeLong's nonparametric AUROC comparison (Sun & Xu 2014
                       fast variant) — a paired test on AUROCs from two
                       classifiers evaluated on the same examples.

These three are what reviewers ask for. The legacy repo had none of them.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class CIResult:
    point: float
    lo: float
    hi: float
    n_resamples: int

    def __repr__(self) -> str:
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}]"


def bootstrap_metric(
    y_true: np.ndarray,
    y_other: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    stratify: np.ndarray | None = None,
) -> CIResult:
    """Percentile bootstrap CI for `metric_fn(y_true, y_other)`.

    `y_other` may be predicted labels or per-class probabilities; this
    function only resamples row indices and lets `metric_fn` interpret them.

    `stratify`, if given, drives stratified resampling within each unique
    value (e.g. by class label) — this gives narrower, more honest CIs for
    imbalanced metrics like macro-F1.
    """
    y_true = np.asarray(y_true)
    y_other = np.asarray(y_other)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n != y_other.shape[0]:
        raise ValueError(f"length mismatch: {n} vs {y_other.shape[0]}")

    point = float(metric_fn(y_true, y_other))

    if stratify is None:
        idx_all = np.arange(n)
        bag = [rng.choice(idx_all, size=n, replace=True) for _ in range(n_resamples)]
    else:
        stratify = np.asarray(stratify)
        groups = {}
        for k, v in enumerate(stratify):
            groups.setdefault(v, []).append(k)
        groups = {k: np.array(v) for k, v in groups.items()}
        bag = []
        for _ in range(n_resamples):
            parts = [rng.choice(idx, size=len(idx), replace=True) for idx in groups.values()]
            bag.append(np.concatenate(parts))

    vals = np.empty(n_resamples, dtype=np.float64)
    for i, idx in enumerate(bag):
        try:
            vals[i] = metric_fn(y_true[idx], y_other[idx])
        except (ValueError, ZeroDivisionError):
            vals[i] = np.nan
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return CIResult(point=point, lo=float("nan"), hi=float("nan"), n_resamples=0)

    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return CIResult(point=point, lo=float(lo), hi=float(hi), n_resamples=int(len(vals)))


# ---------------------------------------------------------------- mcnemar --

def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray, *, exact: bool = True) -> dict:
    """Paired McNemar test for whether two classifiers differ in error rate.

    Inputs are 0/1 arrays of correctness on the SAME examples.
    Returns {'b': discordant_a_right, 'c': discordant_b_right, 'p_value': ..., 'statistic': ...}.

    Uses the binomial exact test when (b+c) is small, chi-squared with continuity
    correction otherwise — this is the standard recommendation.
    """
    a = np.asarray(correct_a).astype(bool)
    b = np.asarray(correct_b).astype(bool)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    n_b = int((a & ~b).sum())  # A right, B wrong
    n_c = int((~a & b).sum())  # A wrong, B right
    n = n_b + n_c
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0, "statistic": 0.0, "test": "trivial"}

    if exact and n < 50:
        # exact binomial: P(X >= max(b,c) | n, 0.5) two-sided
        k = max(n_b, n_c)
        p = 2.0 * stats.binom.sf(k - 1, n, 0.5)
        p = min(1.0, p)
        return {"b": n_b, "c": n_c, "p_value": float(p), "statistic": float(k), "test": "exact"}
    # chi-squared with continuity correction
    stat = (abs(n_b - n_c) - 1.0) ** 2 / float(n)
    p = float(stats.chi2.sf(stat, df=1))
    return {"b": n_b, "c": n_c, "p_value": p, "statistic": float(stat), "test": "chi2"}


# ------------------------------------------------------------------ delong --

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Mid-rank with average ties (used in DeLong)."""
    j = np.argsort(x)
    z = np.empty_like(x, dtype=np.float64)
    z[j] = np.arange(1, len(x) + 1, dtype=np.float64)
    # average over ties
    sorted_x = x[j]
    i = 0
    n = len(x)
    while i < n:
        run_end = i + 1
        while run_end < n and sorted_x[run_end] == sorted_x[i]:
            run_end += 1
        if run_end > i + 1:
            avg = z[j[i:run_end]].mean()
            z[j[i:run_end]] = avg
        i = run_end
    return z


def _delong_internal(scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray):
    """Sun & Xu fast DeLong for two paired AUROCs on the same examples."""
    pos = labels == 1
    neg = ~pos
    m, n_neg = int(pos.sum()), int(neg.sum())
    if m == 0 or n_neg == 0:
        raise ValueError("DeLong needs both positive and negative examples")
    auc_var = np.zeros((2, 2), dtype=np.float64)
    aucs = np.zeros(2)
    v01 = np.zeros((2, m))
    v10 = np.zeros((2, n_neg))
    for k, scores in enumerate((scores_a, scores_b)):
        sx = scores[pos]
        sy = scores[neg]
        tx = _compute_midrank(sx)
        ty = _compute_midrank(sy)
        tz = _compute_midrank(np.concatenate([sx, sy]))
        v01[k] = (tz[:m] - tx) / n_neg
        v10[k] = 1.0 - (tz[m:] - ty) / m
        aucs[k] = tz[:m].sum() / (m * n_neg) - (m + 1.0) / (2.0 * n_neg)
    s01 = np.cov(v01)
    s10 = np.cov(v10)
    cov = s01 / m + s10 / n_neg
    return aucs, cov


def delong_test(scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray) -> dict:
    """Two-sided DeLong test for equality of two paired AUROCs.

    `scores_a`, `scores_b`: per-example positive-class scores (e.g. softmax prob
    for the class of interest) from two models on the SAME examples.
    `labels`: 0/1 per example.

    Returns {'auc_a', 'auc_b', 'cov', 'z', 'p_value'}.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    y = np.asarray(labels).astype(int)
    if a.shape != b.shape or a.shape != y.shape:
        raise ValueError("scores_a, scores_b, labels must have the same shape")
    aucs, cov = _delong_internal(a, b, y)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z = 0.0
        p = 1.0
    else:
        z = float(diff / np.sqrt(var))
        p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "diff": float(diff),
        "var": float(var),
        "z": z,
        "p_value": p,
    }


__all__ = ["CIResult", "bootstrap_metric", "mcnemar_test", "delong_test"]
