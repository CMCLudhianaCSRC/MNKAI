"""Subgroup metric breakdowns.

For each subgroup column (source, sex, age band, anatomical site, fitzpatrick),
compute macro-F1, balanced accuracy, AUROC for the primary class (mel by
default), and bootstrapped 95% CIs.

The legacy `eval/run40/robustness_breakdown.csv` had only sex × {male, female}.
This module is the proper replacement.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from mnkai.evaluation.bootstrap import bootstrap_metric


def _age_bands(age: pd.Series) -> pd.Series:
    """Standard dermatology age bands."""
    bins = [-0.1, 17, 39, 59, 79, 200]
    labels = ["<=17", "18-39", "40-59", "60-79", "80+"]
    out = pd.cut(pd.to_numeric(age, errors="coerce"), bins=bins, labels=labels).astype(str)
    out = out.fillna("unknown").replace({"nan": "unknown"})
    return out


def _macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _balanced_acc(y_true, y_pred):
    return float(balanced_accuracy_score(y_true, y_pred))


def _primary_auroc(scores_for_primary: np.ndarray, y_true: np.ndarray, primary_idx: int):
    y = (y_true == primary_idx).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, scores_for_primary))


def subgroup_breakdown(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    *,
    classes: Sequence[str],
    primary_class: str = "mel",
    subgroup_cols: Sequence[str] = ("source", "sex", "anatom_site_general", "fitzpatrick"),
    add_age_bands: bool = True,
    n_bootstrap: int = 1000,
    min_subgroup_n: int = 25,
    seed: int = 0,
) -> pd.DataFrame:
    """Return long-format DataFrame of subgroup metrics with 95% CIs.

    Columns: group, value, n, macro_f1, macro_f1_lo, macro_f1_hi,
             balanced_acc, balanced_acc_lo, balanced_acc_hi,
             primary_auroc, primary_auroc_lo, primary_auroc_hi.
    """
    classes = [c.lower() for c in classes]
    primary_idx = classes.index(primary_class)
    primary_scores = y_prob[:, primary_idx]

    df = df.reset_index(drop=True)
    if df.shape[0] != y_true.shape[0]:
        raise ValueError(f"df has {len(df)} rows; predictions have {y_true.shape[0]}")

    rows = []
    cols = list(subgroup_cols)
    if add_age_bands and "age_approx" in df.columns and "age_band" not in cols:
        df = df.copy()
        df["age_band"] = _age_bands(df["age_approx"])
        cols.append("age_band")

    for col in cols:
        if col not in df.columns:
            continue
        for value, sub in df.groupby(col, dropna=False):
            if len(sub) < min_subgroup_n:
                continue
            idx = sub.index.to_numpy()
            yt, yp = y_true[idx], y_pred[idx]
            yprob = y_prob[idx]
            sp = primary_scores[idx]

            f1 = bootstrap_metric(yt, yp, _macro_f1, n_resamples=n_bootstrap, seed=seed)
            ba = bootstrap_metric(yt, yp, _balanced_acc, n_resamples=n_bootstrap, seed=seed)
            try:
                auc = bootstrap_metric(
                    yt, sp,
                    lambda t, s, _i=primary_idx: _primary_auroc(s, t, _i),
                    n_resamples=n_bootstrap, seed=seed,
                )
            except ValueError:
                auc = type("X", (), {"point": float("nan"), "lo": float("nan"), "hi": float("nan")})()

            rows.append({
                "group": col,
                "value": str(value) if not pd.isna(value) else "missing",
                "n": int(len(sub)),
                "macro_f1": f1.point, "macro_f1_lo": f1.lo, "macro_f1_hi": f1.hi,
                "balanced_acc": ba.point, "balanced_acc_lo": ba.lo, "balanced_acc_hi": ba.hi,
                f"auroc_{primary_class}": auc.point,
                f"auroc_{primary_class}_lo": auc.lo,
                f"auroc_{primary_class}_hi": auc.hi,
            })

    return pd.DataFrame(rows).sort_values(["group", "value"]).reset_index(drop=True)


def fitzpatrick_gap(report_df: pd.DataFrame, primary_class: str = "mel") -> dict:
    """Compute headline Δ between FST I-II (light) and FST V-VI (dark).

    Returns dict of metric -> dict(light, dark, gap). Useful for the paper headline.
    """
    if "group" not in report_df.columns or (report_df["group"] == "fitzpatrick").sum() == 0:
        return {}
    fst = report_df[report_df["group"] == "fitzpatrick"]
    light = fst[fst["value"].isin(["1", "2", "1.0", "2.0"])]
    dark = fst[fst["value"].isin(["5", "6", "5.0", "6.0"])]
    out = {}
    for col in ["macro_f1", "balanced_acc", f"auroc_{primary_class}"]:
        if col not in fst.columns:
            continue
        l = light[col].mean() if len(light) else float("nan")
        d = dark[col].mean() if len(dark) else float("nan")
        out[col] = {"light_I_II": float(l), "dark_V_VI": float(d), "gap_light_minus_dark": float(l - d)}
    return out


__all__ = ["subgroup_breakdown", "fitzpatrick_gap"]
