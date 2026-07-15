"""Split conformal prediction for FDA-grade uncertainty.

Reference:
  Angelopoulos & Bates (2021). "A Gentle Introduction to Conformal
  Prediction and Distribution-Free Uncertainty Quantification."
  arXiv:2107.07511.

Algorithm (split conformal, APS — adaptive prediction sets):
  1. Train a classifier on the train fold.
  2. Hold out a calibration set of size n.
  3. Compute APS non-conformity scores on calibration:
       s_i = sum_{k: pi_k <= pi_{y_i}} pi_k
     where pi is the softmax probability vector ranked descending.
  4. Pick threshold q_hat at the (1 - alpha)(1 + 1/n) quantile of s_i.
  5. At test time, output the smallest prediction set
     {k : sum_{j: pi_j <= pi_k} pi_j <= q_hat}
     guaranteed to contain the true label with probability >= 1 - alpha.

Why this matters:
  - Distribution-free coverage guarantee (no assumption on data distribution).
  - Regulator-friendly: FDA SaMD draft guidance (2023) explicitly mentions
    conformal as a recommended uncertainty method.
  - Singleton sets => "high confidence diagnosis".
  - Multi-element sets => "model is uncertain among these candidates".
  - Empty set (rare with APS) => "the model is confident none of the
    classes apply" — useful for the 'normal'/'no-lesion' use case.

Use:
  >>> calibrator = fit_split_conformal(cal_probs, cal_labels, alpha=0.05)
  >>> sets = calibrator.predict_set(test_probs)        # list[list[int]]
  >>> coverage_report(sets, test_labels)               # empirical coverage
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Non-conformity scores
# ──────────────────────────────────────────────────────────────────────────────
def _aps_scores(probs: np.ndarray, labels: np.ndarray, randomize: bool = True,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Adaptive Prediction Sets (APS) score.

    For each (probs[i], labels[i]):
      - Sort probs[i] descending into pi.
      - Cumulative sum cum until we reach the position of the true label.
      - Score = cum (with optional randomisation to ensure exact coverage).

    Romano et al. (2020) — "Classification with Valid and Adaptive Coverage."
    """
    n, k = probs.shape
    if rng is None:
        rng = np.random.default_rng(42)
    # Sort probs descending, get indices
    order = np.argsort(-probs, axis=1)            # (n, k)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_probs, axis=1)         # (n, k)

    # Position of true label in sorted order
    pos = np.zeros(n, dtype=int)
    for i in range(n):
        pos[i] = int(np.where(order[i] == labels[i])[0][0])

    rows = np.arange(n)
    score = cum[rows, pos]                        # cumsum up to and incl. true label
    if randomize:
        # u ~ Uniform(0,1), subtract u * prob_at_true_label for tighter sets
        u = rng.random(n)
        score = score - u * sorted_probs[rows, pos]
    return score


def _aps_predict_set(probs: np.ndarray, q_hat: float, randomize: bool = True,
                     rng: np.random.Generator | None = None) -> list[list[int]]:
    """Return the smallest APS prediction set per row that has cum_prob > q_hat."""
    n, k = probs.shape
    if rng is None:
        rng = np.random.default_rng(42)
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_probs, axis=1)

    sets: list[list[int]] = []
    for i in range(n):
        # Smallest j such that cum[i, j] >= q_hat
        if randomize:
            u = rng.random()
            adj = cum[i] - u * sorted_probs[i]
            j = int(np.searchsorted(adj, q_hat, side="left"))
        else:
            j = int(np.searchsorted(cum[i], q_hat, side="left"))
        j = min(j, k - 1)
        sets.append(order[i, : j + 1].tolist())
    return sets


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class APSScores:
    scores: np.ndarray       # calibration set non-conformity scores (n,)
    q_hat: float             # threshold at (1-alpha)(1+1/n) quantile
    alpha: float             # target miscoverage
    n_classes: int


class ConformalCalibrator:
    """Split-conformal calibrator using APS non-conformity scores.

    Fit once on a held-out calibration set, then call .predict_set(probs)
    on test data to get prediction sets with >=1-alpha coverage.
    """

    def __init__(self, q_hat: float, alpha: float, n_classes: int,
                 randomize: bool = True, seed: int = 42):
        self.q_hat       = float(q_hat)
        self.alpha       = float(alpha)
        self.n_classes   = int(n_classes)
        self.randomize   = bool(randomize)
        self._seed       = int(seed)

    @classmethod
    def fit(cls, cal_probs: np.ndarray, cal_labels: np.ndarray,
            alpha: float = 0.05, randomize: bool = True,
            seed: int = 42) -> "ConformalCalibrator":
        cal_probs  = np.asarray(cal_probs, dtype=np.float64)
        cal_labels = np.asarray(cal_labels, dtype=int)
        if cal_probs.ndim != 2:
            raise ValueError(f"cal_probs must be (n, k); got shape {cal_probs.shape}")
        n = cal_probs.shape[0]
        rng = np.random.default_rng(seed)
        scores = _aps_scores(cal_probs, cal_labels, randomize=randomize, rng=rng)
        # Conservative quantile correction for finite calibration set
        q_level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n))
        q_hat = float(np.quantile(scores, q_level, method="higher"))
        return cls(q_hat=q_hat, alpha=alpha, n_classes=cal_probs.shape[1],
                   randomize=randomize, seed=seed)

    def predict_set(self, test_probs: np.ndarray) -> list[list[int]]:
        test_probs = np.asarray(test_probs, dtype=np.float64)
        rng = np.random.default_rng(self._seed)
        return _aps_predict_set(test_probs, self.q_hat,
                                randomize=self.randomize, rng=rng)

    def predict_size(self, test_probs: np.ndarray) -> np.ndarray:
        """Return prediction-set sizes per example."""
        return np.array([len(s) for s in self.predict_set(test_probs)])

    def to_dict(self) -> dict:
        return {"q_hat": self.q_hat, "alpha": self.alpha,
                "n_classes": self.n_classes, "randomize": self.randomize,
                "seed": self._seed}

    @classmethod
    def from_dict(cls, d: dict) -> "ConformalCalibrator":
        return cls(q_hat=d["q_hat"], alpha=d["alpha"],
                   n_classes=d["n_classes"],
                   randomize=d.get("randomize", True),
                   seed=d.get("seed", 42))


def fit_split_conformal(cal_probs, cal_labels, alpha: float = 0.05,
                        randomize: bool = True, seed: int = 42) -> ConformalCalibrator:
    return ConformalCalibrator.fit(cal_probs, cal_labels, alpha=alpha,
                                   randomize=randomize, seed=seed)


# ──────────────────────────────────────────────────────────────────────────────
# Coverage diagnostics
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class CoverageReport:
    coverage:        float       # empirical coverage
    avg_set_size:    float       # average |C(x)|
    singleton_rate:  float       # fraction of examples with |C(x)| == 1
    set_size_hist:   dict        # {set_size: count}


def coverage_report(prediction_sets: list[list[int]],
                    test_labels: np.ndarray) -> CoverageReport:
    """Empirical coverage / set-size statistics."""
    test_labels = np.asarray(test_labels, dtype=int)
    n = len(prediction_sets)
    covered = sum(1 for s, y in zip(prediction_sets, test_labels) if y in s)
    sizes = [len(s) for s in prediction_sets]
    hist: dict[int, int] = {}
    for s in sizes:
        hist[s] = hist.get(s, 0) + 1
    return CoverageReport(
        coverage       = covered / max(1, n),
        avg_set_size   = float(np.mean(sizes)) if sizes else 0.0,
        singleton_rate = sum(1 for s in sizes if s == 1) / max(1, n),
        set_size_hist  = dict(sorted(hist.items())),
    )


__all__ = [
    "ConformalCalibrator",
    "APSScores",
    "fit_split_conformal",
    "coverage_report",
    "CoverageReport",
]
