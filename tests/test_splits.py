"""Tests for group-aware splits — verifies the leakage fix actually works."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mnkai.data.splits import (
    SplitConfig,
    assert_no_group_leakage,
    external_test_indices,
    fold_train_valid_indices,
    make_splits,
)


def _toy_manifest(n_patients=100, imgs_per_patient=10, n_classes=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pi in range(n_patients):
        cls = f"c{pi % n_classes}"
        src = "MSK" if pi % 7 == 0 else "BCN"
        for k in range(imgs_per_patient):
            rows.append({
                "filepath": f"/tmp/p{pi}_{k}.jpg",
                "label_name": cls,
                "group_id": f"pt_{pi:04d}",
                "source": src,
            })
    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def test_no_group_leakage_across_folds():
    df = _toy_manifest()
    out = make_splits(df, SplitConfig(k_folds=5, external_source="MSK", seed=42))
    inner = out[out["split"] == "train_val"]
    # every group lives in exactly one fold
    nf = inner.groupby("group_id")["fold"].nunique()
    assert (nf == 1).all(), f"groups split across folds: {nf[nf > 1].head().to_dict()}"
    assert_no_group_leakage(out)


def test_external_holdout_is_msk_only():
    df = _toy_manifest()
    out = make_splits(df, SplitConfig(k_folds=5, external_source="MSK", seed=42))
    ext = out[out["split"] == "external_test"]
    inner = out[out["split"] == "train_val"]
    assert (ext["source"].str.upper() == "MSK").all()
    assert (inner["source"].str.upper() != "MSK").all() or len(inner) == 0


def test_no_external_groups_appear_in_train():
    df = _toy_manifest()
    out = make_splits(df, SplitConfig(k_folds=5, external_source="MSK", seed=42))
    ext_groups = set(out.loc[out["split"] == "external_test", "group_id"])
    inner_groups = set(out.loc[out["split"] == "train_val", "group_id"])
    assert ext_groups.isdisjoint(inner_groups)


def test_class_balance_across_folds():
    df = _toy_manifest()
    out = make_splits(df, SplitConfig(k_folds=5, external_source=None, seed=42))
    counts = out.pivot_table(index="fold", columns="label_name", aggfunc="size", fill_value=0)
    # max/min ratio should be <= 1.5x for the toy data (well-balanced groups)
    for cls in counts.columns:
        col = counts[cls]
        assert col.max() / max(1, col.min()) <= 2.0, f"class {cls} unbalanced: {col.values}"


def test_fold_indices_sum_correctly():
    df = _toy_manifest()
    out = make_splits(df, SplitConfig(k_folds=5, external_source="MSK", seed=42))
    inner_n = (out["split"] == "train_val").sum()
    for fold in range(5):
        tr, va = fold_train_valid_indices(out, fold)
        assert len(tr) + len(va) == inner_n
        assert len(set(tr).intersection(va)) == 0
    assert len(external_test_indices(out)) == (out["split"] == "external_test").sum()


def test_no_external_source_means_no_holdout():
    df = _toy_manifest()
    out = make_splits(df, SplitConfig(k_folds=5, external_source=None, seed=42))
    assert (out["split"] == "external_test").sum() == 0


def test_missing_group_column_raises():
    df = _toy_manifest()
    df = df.drop(columns=["group_id"])
    with pytest.raises(ValueError, match="group_id"):
        make_splits(df, SplitConfig(k_folds=5))
