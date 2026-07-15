"""ML-critical invariant tests (cheap, data-free) added with the safe-improvement
batch (R09/R11/R13/R15/R18/R22).

Each test uses `pytest.importorskip` for its heavier optional deps so the suite
degrades gracefully in a minimal sandbox. Everything here runs on tiny synthetic
arrays / frames — NO real data, NO model download, NO training.

Invariants pinned:
  (i)   build_loss dispatch + effective_number_weights math + LDAM/cb-focal weight norm
  (ii)  class_balanced_weights power semantics
  (iii) group-aware split has zero train/val group overlap
  (iv)  RAPS `_raps_set` set-size monotonicity in the threshold + the temperature/bias
        transform in demo_app/pipeline.py
  (v)   the alias-collision guard (manifest.LabelNormalizer)
  (vi)  the manifest<->splits merge guard (splits.merge_manifest_splits)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# (i) build_loss dispatch + effective_number_weights + LDAM / cb-focal norm
# ---------------------------------------------------------------------------
def test_build_loss_dispatch_and_weight_math():
    pytest.importorskip("torch")
    from mnkai.losses.focal import (
        ClassBalancedFocal,
        FocalLoss,
        LabelSmoothingCE,
        LDAMLoss,
        LogitAdjustedCE,
        build_loss,
        effective_number_weights,
    )

    counts = [10, 100, 1000]

    # --- dispatch: each name resolves to the right class
    assert isinstance(build_loss("ce", num_classes=3), LabelSmoothingCE)
    assert isinstance(build_loss("focal", num_classes=3), FocalLoss)
    assert isinstance(
        build_loss("cb_focal", num_classes=3, class_counts=counts), ClassBalancedFocal
    )
    assert isinstance(
        build_loss("logit_adjusted", num_classes=3, class_counts=counts), LogitAdjustedCE
    )
    assert isinstance(build_loss("ldam", num_classes=3, class_counts=counts), LDAMLoss)

    # --- unknown name raises (no silent fallback)
    with pytest.raises(ValueError):
        build_loss("does_not_exist", num_classes=3)

    # --- class_counts length mismatch raises
    with pytest.raises(ValueError):
        build_loss("cb_focal", num_classes=3, class_counts=[1, 2])

    # --- effective_number_weights: exact Cui et al. formula, normalized to sum C
    beta = 0.9999
    w = effective_number_weights(counts, beta=beta).numpy()
    assert w.shape == (3,)
    assert abs(float(w.sum()) - 3.0) < 1e-5          # normalized to sum == C
    # closed-form: w_i propto (1-beta)/(1-beta^{n_i}); rarer class => heavier
    raw = (1.0 - beta) / (1.0 - np.power(beta, np.asarray(counts, dtype=np.float64)))
    raw = raw / raw.sum() * 3.0
    assert np.allclose(w, raw, atol=1e-5)
    assert w[0] > w[1] > w[2]

    # --- effective_number_weights handles zero counts without NaN/inf
    wz = effective_number_weights([0, 5, 50]).numpy()
    assert np.all(np.isfinite(wz))


def test_cb_focal_class_boost_renormalizes_to_mean_one():
    pytest.importorskip("torch")
    import torch

    from mnkai.losses.focal import ClassBalancedFocal

    counts = [10, 100, 1000]
    boost = [5.0, 1.0, 1.0]
    loss = ClassBalancedFocal(class_counts=counts, class_boost=boost)
    alpha = loss.alpha  # per-class effective weights * boost, renormalized to mean 1
    assert alpha.numel() == 3
    assert abs(float(alpha.mean().item()) - 1.0) < 1e-5
    # forward runs and returns a finite scalar
    logits = torch.randn(8, 3)
    target = torch.randint(0, 3, (8,))
    val = loss(logits, target)
    assert torch.isfinite(val)


def test_ldam_margins_scaled_to_max_m():
    pytest.importorskip("torch")
    import torch

    from mnkai.losses.focal import LDAMLoss

    counts = [10, 100, 1000]
    max_m = 0.5
    loss = LDAMLoss(class_counts=counts, max_m=max_m)
    m = loss.m_list.numpy()
    # margins scaled so the LARGEST margin equals max_m, rare class gets the largest
    assert abs(float(m.max()) - max_m) < 1e-6
    assert m[0] > m[1] > m[2]
    logits = torch.randn(8, 3)
    target = torch.randint(0, 3, (8,))
    assert torch.isfinite(loss(logits, target))


# ---------------------------------------------------------------------------
# (ii) class_balanced_weights power semantics
# ---------------------------------------------------------------------------
def test_class_balanced_weights_power_semantics():
    pytest.importorskip("torch")
    pytest.importorskip("pandas")
    from mnkai.training.sampler import class_balanced_weights

    # 3 classes: counts 1, 3, 6
    labels = [0] + [1] * 3 + [2] * 6
    num_classes = 3

    labels_arr = np.asarray(labels)
    # helper: the (unique) per-class weight, which is what gets mean-normalized to 1
    def per_class_weight(w):
        return np.array([w[labels_arr == c][0] for c in range(num_classes)])

    # power=0 -> no rebalancing: every weight equal to 1
    w0 = class_balanced_weights(labels, num_classes, power=0.0)
    assert np.allclose(w0, 1.0, atol=1e-9)

    # power=1 -> full inverse frequency; per-CLASS weight ∝ 1/count, mean-normalized to 1
    w1 = class_balanced_weights(labels, num_classes, power=1.0)
    assert abs(float(per_class_weight(w1).mean()) - 1.0) < 1e-9   # per-class mean == 1
    # the single rare-class sample (label 0) must outweigh a common-class sample
    assert w1[0] > w1[-1]
    # each class contributes equal TOTAL mass under full inverse frequency
    mass = [w1[labels_arr == c].sum() for c in range(num_classes)]
    assert np.allclose(mass, mass[0], atol=1e-6)

    # power=0.5 -> gentler than power=1: rare/common ratio strictly smaller
    w_half = class_balanced_weights(labels, num_classes, power=0.5)
    ratio_full = w1[0] / w1[-1]
    ratio_half = w_half[0] / w_half[-1]
    assert 1.0 < ratio_half < ratio_full


# ---------------------------------------------------------------------------
# (iii) group-aware split has zero train/val group overlap
# ---------------------------------------------------------------------------
def _synthetic_manifest(n_groups_per_class=6, rows_per_group=2, n_classes=3):
    import pandas as pd

    rows = []
    gid = 0
    for c in range(n_classes):
        for _ in range(n_groups_per_class):
            gid += 1
            for r in range(rows_per_group):
                rows.append(
                    {
                        "filepath": f"img_{c}_{gid}_{r}.jpg",
                        "label_name": f"class{c}",
                        "group_id": f"grp_{gid}",
                    }
                )
    return pd.DataFrame(rows)


def test_group_split_zero_overlap():
    pytest.importorskip("pandas")
    pytest.importorskip("loguru")
    from mnkai.data.splits import (
        SplitConfig,
        assert_no_group_leakage,
        fold_train_valid_indices,
        make_splits,
    )

    df = _synthetic_manifest()
    cfg = SplitConfig(k_folds=3, external_source=None, seed=0)
    out = make_splits(df, cfg)

    # global assertion helper must pass on a clean split
    assert_no_group_leakage(out, group_col="group_id")

    # explicit per-fold check: train groups and valid groups are disjoint
    for fold in sorted(set(out.loc[out["split"] == "train_val", "fold"])):
        tr_idx, va_idx = fold_train_valid_indices(out, int(fold))
        assert len(tr_idx) > 0 and len(va_idx) > 0
        tr_groups = set(out.iloc[tr_idx]["group_id"])
        va_groups = set(out.iloc[va_idx]["group_id"])
        assert tr_groups.isdisjoint(va_groups), f"group overlap in fold {fold}"


def test_assert_no_group_leakage_detects_overlap():
    pytest.importorskip("pandas")
    pytest.importorskip("loguru")
    import pandas as pd

    from mnkai.data.splits import assert_no_group_leakage

    # same group_id 'g1' straddles fold 0 and fold 1 -> must raise
    bad = pd.DataFrame(
        {
            "group_id": ["g1", "g1", "g2"],
            "fold": [0, 1, 0],
            "split": ["train_val", "train_val", "train_val"],
        }
    )
    with pytest.raises(AssertionError):
        assert_no_group_leakage(bad, group_col="group_id")


# ---------------------------------------------------------------------------
# (iv) RAPS _raps_set monotonicity + temperature/bias transform
# ---------------------------------------------------------------------------
def _import_pipeline():
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    import importlib

    return importlib.import_module("demo_app.pipeline")


def test_raps_set_size_monotone_in_threshold():
    pipe = _import_pipeline()
    rng = np.random.default_rng(0)
    logits = rng.normal(size=10)
    prob = np.exp(logits) / np.exp(logits).sum()

    lam, k_reg = 0.05, 2
    sizes = []
    for qhat in np.linspace(0.05, 0.999, 12):
        s = pipe._raps_set(prob, float(qhat), lam, k_reg)
        sizes.append(len(s))
        # set is always non-empty and within [1, C]
        assert 1 <= len(s) <= len(prob)
        # returned indices are valid and unique
        assert len(set(s.tolist())) == len(s)
    # higher coverage threshold => (weakly) larger prediction set
    assert all(b >= a for a, b in zip(sizes, sizes[1:])), sizes


def test_calibrated_adjust_transform():
    pipe = _import_pipeline()
    pytest.importorskip("scipy")

    p = np.array([0.6, 0.25, 0.1, 0.05])

    # T == 1, zero bias: cal is (up to the log round-trip) the input; adj ~= cal
    cal1, adj1 = pipe._calibrated_adjust(p, 1.0, np.zeros(4))
    assert np.allclose(cal1.sum(), 1.0)
    assert np.allclose(adj1.sum(), 1.0)
    assert np.allclose(cal1, p, atol=1e-6)
    assert np.argmax(adj1) == np.argmax(p)

    # T > 1 softens the distribution: entropy increases, peak prob decreases
    calT, _ = pipe._calibrated_adjust(p, 2.5, np.zeros(4))
    ent = lambda v: float(-np.sum(v * np.log(np.clip(v, 1e-12, 1))))
    assert calT.max() < cal1.max()
    assert ent(calT) > ent(cal1)

    # a positive per-class bias raises that class's adjusted probability
    bias = np.array([0.0, 3.0, 0.0, 0.0])   # boost class 1
    _, adj_b = pipe._calibrated_adjust(p, 1.0, bias)
    assert adj_b[1] > adj1[1]


# ---------------------------------------------------------------------------
# (v) alias-collision guard
# ---------------------------------------------------------------------------
def test_alias_collision_guard_records_and_resolves_last():
    pytest.importorskip("pandas")
    pytest.importorskip("loguru")
    from mnkai.data.manifest import LabelNormalizer

    norm = LabelNormalizer(
        classes=["a", "b"],
        aliases={"a": ["x", "shared"], "b": ["shared"]},
        ignore=[],
    )
    # collision recorded: 'shared' claimed by both a and b
    assert set(norm._alias_claims["shared"]) == {"a", "b"}
    assert len(norm._alias_claims["shared"]) == 2
    # resolves to the LAST claimant (dict insertion order => 'b')
    assert norm.map("shared") == "b"
    # a non-colliding alias has exactly one claimant
    assert norm._alias_claims["x"] == ["a"]
    assert norm.map("x") == "a"


def test_exanthema_collision_resolved_in_shipped_taxonomy():
    pytest.importorskip("pandas")
    pytest.importorskip("loguru")
    from mnkai.data.manifest import PHASE3_CLASSES, LabelNormalizer

    norm = LabelNormalizer(classes=PHASE3_CLASSES)
    # R15: 'exanthema' is now owned by exactly one canonical class (drug_eruption)
    assert norm._alias_claims["exanthema"] == ["drug_eruption"]
    assert norm.map("exanthema") == "drug_eruption"


# ---------------------------------------------------------------------------
# (vi) manifest<->splits merge guard
# ---------------------------------------------------------------------------
def test_merge_guard_happy_path_and_separator_normalization():
    pytest.importorskip("pandas")
    pytest.importorskip("loguru")
    import pandas as pd

    from mnkai.data.splits import merge_manifest_splits

    manifest = pd.DataFrame(
        {
            "filepath": ["a/b/1.jpg", "a/b/2.jpg", "a/b/3.jpg"],
            "label_name": ["mel", "nv", "bcc"],
        }
    )
    # splits uses backslashes + trailing whitespace on the SAME logical paths
    splits = pd.DataFrame(
        {
            "filepath": ["a\\b\\1.jpg ", "a\\b\\2.jpg", "a\\b\\3.jpg"],
            "split": ["train_val", "train_val", "external_test"],
            "fold": [0, 1, -1],
        }
    )
    merged = merge_manifest_splits(manifest, splits)
    assert len(merged) == 3                    # normalization prevents silent drops
    assert set(merged.columns) >= {"filepath", "label_name", "split", "fold"}


def test_merge_guard_raises_on_key_mismatch():
    pytest.importorskip("pandas")
    pytest.importorskip("loguru")
    import pandas as pd

    from mnkai.data.splits import merge_manifest_splits

    manifest = pd.DataFrame(
        {"filepath": [f"img_{i}.jpg" for i in range(10)], "label_name": ["mel"] * 10}
    )
    # splits keyed by a DIFFERENT format => almost nothing merges
    splits = pd.DataFrame(
        {
            "filepath": [f"PREFIX/img_{i}.jpg" for i in range(10)],
            "split": ["train_val"] * 10,
            "fold": [0] * 10,
        }
    )
    with pytest.raises(RuntimeError):
        merge_manifest_splits(manifest, splits)


def test_merge_guard_raises_on_missing_key():
    pytest.importorskip("pandas")
    pytest.importorskip("loguru")
    import pandas as pd

    from mnkai.data.splits import merge_manifest_splits

    manifest = pd.DataFrame({"filepath": ["1.jpg"], "label_name": ["mel"]})
    splits = pd.DataFrame({"nope": ["1.jpg"], "split": ["train_val"], "fold": [0]})
    with pytest.raises(ValueError):
        merge_manifest_splits(manifest, splits)
