"""Group-aware splits for dermatology data.

The previous pipeline's splitter (`StratifiedShuffleSplit` on label only) put the
SAME PATIENT in train and validation, inflating reported metrics by an unknown
margin. This module fixes that with two independent guards:

  1. An *external-test* holdout taken by source domain (e.g. all MSK images).
     This simulates dataset-shift evaluation and is never seen during model
     selection.

  2. K-fold cross-validation on the remaining "in-distribution" data using
     `GroupKFold(group_id)` (patient/lesion ID) so no patient straddles
     train/val boundaries. Stratification by class is approximated greedily:
     groups are assigned to the fold that currently has the lowest count of
     the group's majority class, an LPT-style heuristic that converges to
     near-balanced class counts across folds.

Inputs
------
A manifest CSV with at least: `filepath`, `label_name`, `group_id`.
Optional: `source` (e.g. "ISIC_HAM", "ISIC_BCN", "ISIC_MSK"), `sex`,
`age_approx`, `anatom_site_general`, `fitzpatrick`.

Outputs
-------
A new CSV with columns added:
  - `split`        : one of {"train_val", "external_test"}
  - `fold`         : integer in [0, k_folds) for train_val rows, -1 for external_test
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class SplitConfig:
    k_folds: int = 5
    external_source: str | None = "MSK"  # held-out source domain (case-insensitive substring match)
    group_col: str = "group_id"
    label_col: str = "label_name"
    source_col: str = "source"
    seed: int = 42


def _normalize_source(s: str | float) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip().upper()


def _identify_external_mask(df: pd.DataFrame, cfg: SplitConfig) -> pd.Series:
    if cfg.external_source is None or cfg.external_source == "":
        return pd.Series(False, index=df.index)
    if cfg.source_col not in df.columns:
        logger.warning(f"[splits] external_source='{cfg.external_source}' requested but "
                       f"manifest has no '{cfg.source_col}' column; skipping external holdout")
        return pd.Series(False, index=df.index)
    needle = cfg.external_source.upper()
    sources = df[cfg.source_col].apply(_normalize_source)
    return sources.str.contains(needle, na=False)


def _group_majority_label(df: pd.DataFrame, group_col: str, label_col: str) -> dict[str, str]:
    """For each group, return its most common label."""
    out: dict[str, str] = {}
    for g, sub in df.groupby(group_col):
        out[g] = Counter(sub[label_col].astype(str).str.lower()).most_common(1)[0][0]
    return out


def _greedy_balanced_group_kfold(
    df: pd.DataFrame, k: int, group_col: str, label_col: str, seed: int
) -> np.ndarray:
    """Assign each group to one of k folds using a longest-processing-time heuristic
    that approximately balances per-class counts across folds.

    Returns
    -------
    fold_per_row : np.ndarray of shape (len(df),) with integer fold ids in [0, k).
    """
    rng = np.random.default_rng(seed)

    # 1. group sizes
    group_sizes: dict[str, int] = df.groupby(group_col).size().to_dict()
    group_label = _group_majority_label(df, group_col, label_col)

    # 2. order groups by size desc; tie-break randomly
    items = list(group_sizes.items())  # list of (group, size)
    rng.shuffle(items)
    items.sort(key=lambda kv: -kv[1])

    # 3. running per-fold class counts
    fold_class_counts: list[Counter] = [Counter() for _ in range(k)]
    fold_total: list[int] = [0] * k
    group_to_fold: dict[str, int] = {}

    for g, size in items:
        cls = group_label[g]
        # pick fold with lowest count of THIS class (to balance the rare class first);
        # tie-break by total fold size (smaller wins), then random.
        scores = []
        for f in range(k):
            scores.append((fold_class_counts[f][cls], fold_total[f], rng.random(), f))
        scores.sort()
        chosen = scores[0][3]
        group_to_fold[g] = chosen
        fold_class_counts[chosen][cls] += size
        fold_total[chosen] += size

    fold_per_row = df[group_col].map(group_to_fold).to_numpy()
    return fold_per_row.astype(int)


def make_splits(df: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    """Return a copy of `df` with `split` and `fold` columns added.

    Validates that `group_col` and `label_col` exist; raises otherwise.
    """
    for col in (cfg.group_col, cfg.label_col):
        if col not in df.columns:
            raise ValueError(f"manifest is missing required column '{col}'; have {list(df.columns)}")

    df = df.copy().reset_index(drop=True)
    df[cfg.label_col] = df[cfg.label_col].astype(str).str.strip().str.lower()

    ext_mask = _identify_external_mask(df, cfg)
    n_ext = int(ext_mask.sum())
    logger.info(f"[splits] external_test ({cfg.external_source}): {n_ext} rows")

    df["split"] = np.where(ext_mask, "external_test", "train_val")
    df["fold"] = -1

    inner = df.loc[~ext_mask].copy()
    if len(inner) == 0:
        raise RuntimeError("after external holdout, no rows remain for training")

    folds = _greedy_balanced_group_kfold(
        inner, k=cfg.k_folds, group_col=cfg.group_col, label_col=cfg.label_col, seed=cfg.seed
    )
    df.loc[~ext_mask, "fold"] = folds

    # leakage check: same group must never appear across folds
    leaked = (
        df.loc[df["split"] == "train_val"]
        .groupby(cfg.group_col)["fold"]
        .nunique()
        .gt(1)
        .sum()
    )
    if leaked:
        raise RuntimeError(f"[splits] BUG: {leaked} groups span multiple folds")

    # also: external test groups should not reappear in train_val
    ext_groups = set(df.loc[df["split"] == "external_test", cfg.group_col].unique())
    inner_groups = set(df.loc[df["split"] == "train_val", cfg.group_col].unique())
    leaked_ext = ext_groups & inner_groups
    if leaked_ext:
        logger.warning(f"[splits] {len(leaked_ext)} groups appear in BOTH external_test and train_val; "
                       f"forcing those rows into external_test")
        bad = df[cfg.group_col].isin(leaked_ext)
        df.loc[bad, "split"] = "external_test"
        df.loc[bad, "fold"] = -1

    _log_split_summary(df, cfg)
    return df


def _log_split_summary(df: pd.DataFrame, cfg: SplitConfig) -> None:
    n_total = len(df)
    n_ext = int((df["split"] == "external_test").sum())
    n_inner = n_total - n_ext
    logger.info(f"[splits] total={n_total} train_val={n_inner} external_test={n_ext}")

    inner = df[df["split"] == "train_val"]
    if cfg.label_col in inner.columns:
        per_fold = inner.pivot_table(index="fold", columns=cfg.label_col, aggfunc="size", fill_value=0)
        logger.info(f"[splits] per-fold class counts:\n{per_fold}")

    # group leakage sanity check (already enforced above; reported here too)
    g_per_fold = inner.groupby("fold")[cfg.group_col].nunique()
    logger.info(f"[splits] groups per fold: {dict(g_per_fold)}")


def merge_manifest_splits(manifest: pd.DataFrame, splits: pd.DataFrame,
                          *, min_keep_frac: float = 0.90) -> pd.DataFrame:
    """Inner-merge a normalized `manifest` with a `splits` table, failing loudly
    if a join-key format mismatch silently shrinks the training set.

    The join key is `filepath` (preferred) or `isic_id`; it is normalized on BOTH
    sides (whitespace strip, and for `filepath` backslash->forward-slash) so a path
    separator or trailing-space mismatch cannot cause the inner merge to quietly
    drop rows. If fewer than `min_keep_frac` of the manifest rows survive the merge,
    a RuntimeError is raised with a diagnostic instead of training on a fraction of
    the data.

    Returns the merged DataFrame (with `split` and `fold` columns attached).
    """
    n_before = len(manifest)
    manifest = manifest.copy()
    splits = splits.copy()
    if "filepath" in splits.columns and "filepath" in manifest.columns:
        key = "filepath"
        manifest[key] = manifest[key].astype(str).str.strip().str.replace("\\", "/", regex=False)
        splits[key] = splits[key].astype(str).str.strip().str.replace("\\", "/", regex=False)
    elif "isic_id" in splits.columns and "isic_id" in manifest.columns:
        key = "isic_id"
        manifest[key] = manifest[key].astype(str).str.strip()
        splits[key] = splits[key].astype(str).str.strip()
    else:
        raise ValueError(
            "splits table must share a 'filepath' or 'isic_id' column with the "
            f"manifest to merge; manifest cols={list(manifest.columns)}, "
            f"splits cols={list(splits.columns)}"
        )

    merged = manifest.merge(splits[[key, "split", "fold"]], on=key, how="inner")
    n_after = len(merged)
    keep_frac = n_after / max(1, n_before)
    logger.info(f"[splits] manifest<->splits merge on '{key}': "
                f"{n_before} -> {n_after} rows (dropped {n_before - n_after}, kept {keep_frac:.1%})")
    if n_after == 0:
        raise RuntimeError(f"[splits] manifest<->splits merge on '{key}' produced 0 rows")
    if keep_frac < min_keep_frac:
        raise RuntimeError(
            f"[splits] manifest<->splits inner merge on '{key}' kept only {keep_frac:.1%} "
            f"of rows ({n_after}/{n_before}); expected >= {min_keep_frac:.0%}. This usually "
            f"means the '{key}' key format differs between the manifest and the splits table "
            f"(path separators, prefixes, or extensions). Reconcile the key format so the "
            f"merge is near-lossless instead of silently shrinking the training set."
        )
    return merged


def fold_train_valid_indices(df: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Given a manifest with `split` and `fold` columns, return (train_idx, valid_idx) for `fold`.

    Train: all train_val rows whose fold != `fold`.
    Valid: all train_val rows whose fold == `fold`.
    External test is never returned by this function.
    """
    if "split" not in df.columns or "fold" not in df.columns:
        raise ValueError("dataframe must have 'split' and 'fold' columns; run make_splits() first")
    inner = df["split"].to_numpy() == "train_val"
    fold_arr = df["fold"].to_numpy()
    train_idx = np.where(inner & (fold_arr != fold))[0]
    valid_idx = np.where(inner & (fold_arr == fold))[0]
    return train_idx, valid_idx


def external_test_indices(df: pd.DataFrame) -> np.ndarray:
    return np.where(df["split"].to_numpy() == "external_test")[0]


# --------- assertions used by both production code and tests ---------

def assert_no_group_leakage(df: pd.DataFrame, group_col: str = "group_id") -> None:
    """Raise if any group appears in more than one (split, fold) bucket among train_val."""
    inner = df[df["split"] == "train_val"]
    bad = inner.groupby(group_col)["fold"].nunique().gt(1)
    if bad.any():
        offenders = bad[bad].index.tolist()[:10]
        raise AssertionError(f"group leakage across folds; first offenders: {offenders}")


__all__ = [
    "SplitConfig",
    "make_splits",
    "merge_manifest_splits",
    "fold_train_valid_indices",
    "external_test_indices",
    "assert_no_group_leakage",
]
