"""Compact metric helpers used by the evaluator.

Differences from the legacy `src/metrics.py`:
  - Adds balanced accuracy and per-class precision-recall AUC.
  - Returns sklearn `classification_report` plus a flat dict of macro / weighted
    summaries that's easier to log to TensorBoard / W&B.
  - Computes Expected Calibration Error and Brier multiclass score.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Multiclass ECE: bin by max-class confidence; mean |acc - conf|."""
    if y_prob is None or len(y_prob) == 0:
        return float("nan")
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    correct = (predictions == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def brier_multiclass(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error vs one-hot."""
    n_classes = y_prob.shape[1]
    onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_prob - onehot) ** 2, axis=1)))


def safe_auroc_ovr(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """OvR AUROC per class + macro; tolerant to single-class folds."""
    out: dict = {"per_class": {}, "macro": float("nan")}
    n_classes = y_prob.shape[1]
    aucs = []
    for c in range(n_classes):
        y_c = (y_true == c).astype(int)
        if y_c.sum() == 0 or y_c.sum() == len(y_c):
            out["per_class"][c] = float("nan")
            continue
        try:
            auc = roc_auc_score(y_c, y_prob[:, c])
        except ValueError:
            auc = float("nan")
        out["per_class"][c] = float(auc)
        if not np.isnan(auc):
            aucs.append(auc)
    if aucs:
        out["macro"] = float(np.mean(aucs))
    return out


def safe_pr_auc_ovr(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    out: dict = {"per_class": {}, "macro": float("nan")}
    n_classes = y_prob.shape[1]
    aps = []
    for c in range(n_classes):
        y_c = (y_true == c).astype(int)
        if y_c.sum() == 0:
            out["per_class"][c] = float("nan")
            continue
        try:
            ap = average_precision_score(y_c, y_prob[:, c])
        except ValueError:
            ap = float("nan")
        out["per_class"][c] = float(ap)
        if not np.isnan(ap):
            aps.append(ap)
    if aps:
        out["macro"] = float(np.mean(aps))
    return out


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None,
                class_names: Sequence[str]) -> dict:
    """One-shot metric bundle for an eval split."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).astype(int).tolist(),
        "report": classification_report(y_true, y_pred, target_names=list(class_names),
                                        output_dict=True, zero_division=0),
    }
    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[0] == y_true.shape[0]:
        roc = safe_auroc_ovr(y_true, y_prob)
        pr = safe_pr_auc_ovr(y_true, y_prob)
        out["macro_auroc_ovr"] = roc["macro"]
        out["per_class_auroc"] = {class_names[c]: v for c, v in roc["per_class"].items()}
        out["macro_pr_auc_ovr"] = pr["macro"]
        out["per_class_pr_auc"] = {class_names[c]: v for c, v in pr["per_class"].items()}
        out["ece"] = expected_calibration_error(y_true, y_prob)
        out["brier"] = brier_multiclass(y_true, y_prob)
    return out


__all__ = [
    "all_metrics",
    "expected_calibration_error",
    "brier_multiclass",
    "safe_auroc_ovr",
    "safe_pr_auc_ovr",
]
