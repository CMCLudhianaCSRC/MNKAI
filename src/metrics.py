from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def macro_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    C = y_prob.shape[1]
    aucs = []
    for c in range(C):
        y = (y_true == c).astype(int)
        try:
            aucs.append(roc_auc_score(y, y_prob[:, c]))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else 0.0


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro"))


def ece_score(probs, labels, n_bins: int = 15):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    preds = probs.argmax(1)
    conf = probs.max(1)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (conf > lo) & (conf <= hi)
        if not np.any(sel):
            continue
        acc = np.mean(preds[sel] == labels[sel])
        cavg = np.mean(conf[sel])
        ece += (np.sum(sel) / len(labels)) * abs(acc - cavg)
    return float(ece)

def _safe_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    average: str,
    labels: List[str],
) -> Dict[str, Any]:
    """Compute AUCs defensively; return -1.0 when not computable."""
    out: Dict[str, Any] = {
        "macro_auc_ovr": -1.0,
        "macro_auc_ovo": -1.0,
        "per_class_auc": {},
    }
    try:
        # one-vs-rest
        ovr = roc_auc_score(
            y_true,
            y_prob,
            average="macro",
            multi_class="ovr",
        )
        out["macro_auc_ovr"] = float(ovr)
    except Exception:
        pass

    try:
        # one-vs-one
        ovo = roc_auc_score(
            y_true,
            y_prob,
            average="macro",
            multi_class="ovo",
        )
        out["macro_auc_ovo"] = float(ovo)
    except Exception:
        pass

    # Per-class AUCs (OvR) if possible
    try:
        n_classes = y_prob.shape[1]
        for c in range(n_classes):
            y_true_c = (y_true == c).astype(int)
            auc_c = roc_auc_score(y_true_c, y_prob[:, c])
            out["per_class_auc"][labels[c]] = float(auc_c)
    except Exception:
        # leave empty on failure
        pass

    return out


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> Optional[float]:
    """
    ECE for multiclass: bin by confidence (max prob); compare accuracy vs confidence per bin.
    Returns None if inputs are empty.
    """
    if y_prob is None or len(y_prob) == 0:
        return None
    try:
        confidences = y_prob.max(axis=1)
        predictions = y_prob.argmax(axis=1)
        correct = (predictions == y_true).astype(float)

        # Bin edges from 0..1
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
            if not np.any(mask):
                continue
            acc_bin = correct[mask].mean()
            conf_bin = confidences[mask].mean()
            weight = mask.mean()
            ece += weight * abs(acc_bin - conf_bin)
        return float(ece)
    except Exception:
        return None


def metrics_from_preds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Compute a compact set of metrics the trainer expects.
    Returns:
      {
        "report": <sklearn classification_report dict>,
        "confusion_matrix": <2D list>,
        "accuracy": <float>,
        "macro_auc_ovr": <float or -1.0>,
        "macro_auc_ovo": <float or -1.0>,
        "per_class_auc": {class: auc, ...},
        "ece": <float or None>,
      }
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_prob is not None:
        y_prob = np.asarray(y_prob)

    # Basic metrics
    try:
        rpt = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
    except Exception:
        # Fallback if labels mismatch
        rpt = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    cm_list = cm.astype(int).tolist()

    # AUCs (only if probabilities provided)
    aucs = {"macro_auc_ovr": -1.0, "macro_auc_ovo": -1.0, "per_class_auc": {}}
    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[0] == y_true.shape[0]:
        try:
            aucs = _safe_auc(y_true, y_prob, average="macro", labels=class_names)
        except Exception:
            pass

    # Calibration (ECE)
    ece_val = expected_calibration_error(y_true, y_prob) if y_prob is not None else None

    # Make a tidy object similar to what your trainer logs
    out: Dict[str, Any] = {
        "report": rpt,
        "confusion_matrix": cm_list,
        "accuracy": acc,
        "macro_auc_ovr": float(aucs.get("macro_auc_ovr", -1.0)),
        "macro_auc_ovo": float(aucs.get("macro_auc_ovo", -1.0)),
        "per_class_auc": aucs.get("per_class_auc", {}),
        "ece": ece_val,
    }
    return out
