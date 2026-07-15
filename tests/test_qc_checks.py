"""Data-free tests for scripts/qc_checks.py (synthetic arrays only, no real data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# scripts/ is not a package — put it on the path so `import qc_checks` works.
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

qc = pytest.importorskip("qc_checks")


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def test_healthy_cache_passes():
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((200, 32)).astype(np.float32)
    y = np.repeat(np.arange(4), 50)
    rep = qc.validate_feature_cache(feats, y=y)
    assert rep.ok
    assert rep.n_rows == 200 and rep.n_dims == 32
    assert rep.stats["n_classes"] == 4
    assert not rep.problems


def test_all_zero_features_fail():
    feats = np.zeros((50, 16), dtype=np.float32)
    rep = qc.validate_feature_cache(feats)
    assert not rep.ok
    assert any("all-zero" in p for p in rep.problems)


def test_constant_features_fail():
    feats = np.full((50, 16), 3.14, dtype=np.float32)
    rep = qc.validate_feature_cache(feats)
    assert not rep.ok
    assert any("constant" in p or "all-zero" in p for p in rep.problems)


def test_nonfinite_fails():
    rng = np.random.default_rng(1)
    feats = rng.standard_normal((40, 8)).astype(np.float32)
    feats[3, 2] = np.nan
    feats[7, 1] = np.inf
    rep = qc.validate_feature_cache(feats)
    assert not rep.ok
    assert any("non-finite" in p for p in rep.problems)


def test_partial_zero_rows_fail():
    rng = np.random.default_rng(2)
    feats = rng.standard_normal((100, 8)).astype(np.float32)
    feats[:20] = 0.0  # 20% all-zero rows (a partial-extraction signature)
    rep = qc.validate_feature_cache(feats, max_zero_row_frac=0.01)
    assert not rep.ok
    assert any("all-zero rows" in p for p in rep.problems)


def test_expected_dims_mismatch_fails():
    rng = np.random.default_rng(3)
    feats = rng.standard_normal((30, 20)).astype(np.float32)
    rep = qc.validate_feature_cache(feats, expected_dims=768)
    assert not rep.ok
    assert any("dim mismatch" in p for p in rep.problems)


def test_wrong_ndim_fails():
    feats = np.zeros((10, 3, 3), dtype=np.float32)
    rep = qc.validate_feature_cache(feats)
    assert not rep.ok
    assert any("2-D" in p for p in rep.problems)


def test_extreme_imbalance_warns_but_passes():
    rng = np.random.default_rng(4)
    feats = rng.standard_normal((1010, 8)).astype(np.float32)
    y = np.array([0] * 1000 + [1] * 10)  # 100x imbalance -> above default? default is 200x
    rep = qc.validate_feature_cache(feats, y=y, warn_imbalance_ratio=50.0)
    assert rep.ok  # imbalance is a warning, not a failure
    assert any("imbalance" in w for w in rep.warnings)


def test_empty_class_fails():
    rng = np.random.default_rng(5)
    feats = rng.standard_normal((30, 8)).astype(np.float32)
    y = np.array([0] * 30)  # only one class present, but min_per_class check is per present class
    # Force a "declared but empty" style check via min_per_class on a rare class:
    y = np.array([0] * 29 + [1])
    rep = qc.validate_feature_cache(feats, y=y, min_per_class=2)
    assert not rep.ok
    assert any("below 2 samples" in p for p in rep.problems)


# --------------------------------------------------------------------------
# leakage audit
# --------------------------------------------------------------------------
def test_leakage_detects_duplicates():
    rng = np.random.default_rng(6)
    train = rng.standard_normal((300, 32)).astype(np.float32)
    # external: 10 exact copies of train rows + 40 fresh rows
    dupes = train[:10].copy()
    fresh = rng.standard_normal((40, 32)).astype(np.float32)
    external = np.concatenate([dupes, fresh], axis=0)
    rep = qc.leakage_audit(train, external, threshold=0.98)
    assert not rep.ok
    assert rep.n_flagged >= 10
    assert rep.max_cosine > 0.999
    # worst offenders point back at the right train rows
    top = {o["external_idx"] for o in rep.offenders[:10]}
    assert top == set(range(10))


def test_no_leakage_passes():
    rng = np.random.default_rng(7)
    train = rng.standard_normal((300, 64)).astype(np.float32)
    external = rng.standard_normal((80, 64)).astype(np.float32)
    rep = qc.leakage_audit(train, external, threshold=0.98)
    assert rep.ok
    assert rep.n_flagged == 0


def test_leakage_dim_mismatch_raises():
    train = np.zeros((10, 32), dtype=np.float32)
    external = np.zeros((5, 16), dtype=np.float32)
    with pytest.raises(ValueError):
        qc.leakage_audit(train, external)


def test_max_cosine_argmax_correct():
    rng = np.random.default_rng(8)
    train = rng.standard_normal((50, 16)).astype(np.float32)
    external = train[[7, 20, 3]].copy()
    best, arg = qc.max_cosine_to_reference(external, train, return_argmax=True)
    assert np.allclose(best, 1.0, atol=1e-4)
    assert arg.tolist() == [7, 20, 3]


# --------------------------------------------------------------------------
# loading round-trip
# --------------------------------------------------------------------------
def test_load_npz_and_npy(tmp_path):
    rng = np.random.default_rng(9)
    feats = rng.standard_normal((20, 8)).astype(np.float32)
    y = np.arange(20) % 4
    split = np.array(["train_val"] * 15 + ["external_test"] * 5)
    npz = tmp_path / "c.npz"
    np.savez(npz, features=feats, y=y, split=split, classes=np.array(["a", "b", "c", "d"]))
    loaded = qc.load_features(npz)
    assert loaded.features.shape == (20, 8)
    assert loaded.y is not None and loaded.split is not None

    npy = tmp_path / "m.npy"
    np.save(npy, feats)
    lm = qc.load_features(npy)
    assert lm.features.shape == (20, 8)
    assert lm.y is None


def test_cli_validate_and_leak(tmp_path, capsys):
    rng = np.random.default_rng(10)
    feats = rng.standard_normal((60, 8)).astype(np.float32)
    split = np.array(["train_val"] * 40 + ["external_test"] * 20)
    # inject 5 leaks: copy first 5 train rows into external region
    feats[40:45] = feats[:5]
    npz = tmp_path / "c.npz"
    np.savez(npz, features=feats, y=np.arange(60) % 3, split=split)

    assert qc.main(["validate", "--cache", str(npz)]) == 0
    rc = qc.main(["leak-audit", "--cache", str(npz),
                  "--train-split", "train_val", "--external-split", "external_test"])
    assert rc == 1  # leakage present -> non-zero exit
