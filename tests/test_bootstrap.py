"""Stat tests — verify bootstrap CIs, McNemar, and DeLong behave correctly."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from mnkai.evaluation.bootstrap import bootstrap_metric, delong_test, mcnemar_test


def _macro_f1(yt, yp):
    return float(f1_score(yt, yp, average="macro", zero_division=0))


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(0)
    yt = rng.integers(0, 4, 500)
    yp = yt.copy()
    yp[rng.choice(500, 100, replace=False)] = (yp[rng.choice(500, 100, replace=False)] + 1) % 4
    ci = bootstrap_metric(yt, yp, _macro_f1, n_resamples=500, seed=1)
    assert ci.lo <= ci.point <= ci.hi


def test_bootstrap_stratified_does_not_crash_on_rare_class():
    rng = np.random.default_rng(0)
    yt = np.zeros(200, dtype=int); yt[:5] = 1
    yp = yt.copy(); yp[3] = 0
    ci = bootstrap_metric(yt, yp, _macro_f1, n_resamples=200, seed=0, stratify=yt)
    assert np.isfinite(ci.point)


def test_mcnemar_identical_classifiers_p_eq_1():
    a = np.array([1, 0, 1, 1, 0, 0, 1])
    b = a.copy()
    out = mcnemar_test(a, b)
    assert out["p_value"] == pytest.approx(1.0)


def test_mcnemar_strong_difference_low_p():
    rng = np.random.default_rng(0)
    n = 200
    a = rng.integers(0, 2, n)  # random
    b = np.ones(n, dtype=int)  # always right
    yt = np.ones(n, dtype=int)
    correct_a = (a == yt).astype(int)
    correct_b = (b == yt).astype(int)
    out = mcnemar_test(correct_a, correct_b)
    assert out["p_value"] < 1e-5


def test_delong_identical_scores_p_eq_1():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = rng.random(300)
    out = delong_test(s, s, y)
    assert out["p_value"] == pytest.approx(1.0)
    assert out["auc_a"] == pytest.approx(out["auc_b"])


def test_delong_better_classifier_detected():
    rng = np.random.default_rng(0)
    n = 1000
    y = rng.integers(0, 2, n)
    # signal_b is highly informative; signal_a is mostly noise
    a = rng.random(n)
    b = y + 0.3 * rng.standard_normal(n)
    out = delong_test(b, a, y)  # b is better
    auc_b = roc_auc_score(y, b)
    auc_a = roc_auc_score(y, a)
    assert out["auc_a"] == pytest.approx(auc_b, abs=1e-6)
    assert out["auc_b"] == pytest.approx(auc_a, abs=1e-6)
    assert out["p_value"] < 1e-5
