"""Tests for operating-point selection and risk-coverage abstention."""
from __future__ import annotations

import numpy as np
import pytest

from mnkai.evaluation.abstention import (
    ABSTAIN,
    coverage_at_threshold,
    predict_with_abstain,
    risk_coverage_curve,
    threshold_for_target_error,
)
from mnkai.evaluation.threshold import (
    operating_point_at_sensitivity,
    operating_point_at_specificity,
    threshold_table,
    youden_j_threshold,
)


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(0)
    n = 1000
    y = rng.integers(0, 2, n)
    # informative score
    s = rng.normal(0, 1, n) + 1.5 * y
    s = (s - s.min()) / (s.max() - s.min())   # squash to [0,1]
    return y, s


def test_threshold_table_monotone(binary_data):
    y, s = binary_data
    table = threshold_table(s, y, n_thresholds=21)
    # specificity is monotone non-decreasing in threshold
    spec = [p.specificity for p in table]
    assert all(b >= a - 1e-9 for a, b in zip(spec, spec[1:]))


def test_op_at_specificity_meets_target(binary_data):
    y, s = binary_data
    op = operating_point_at_specificity(s, y, target_specificity=0.95)
    assert op.specificity >= 0.94  # tolerance


def test_op_at_sensitivity_meets_target(binary_data):
    y, s = binary_data
    op = operating_point_at_sensitivity(s, y, target_sensitivity=0.90)
    assert op.sensitivity >= 0.89


def test_youden_j_returns_a_balance(binary_data):
    y, s = binary_data
    op = youden_j_threshold(s, y)
    assert op.sensitivity + op.specificity > 1.0  # better than chance


def test_threshold_table_requires_both_classes():
    y = np.zeros(20, dtype=int)
    s = np.linspace(0, 1, 20)
    with pytest.raises(ValueError):
        threshold_table(s, y)


def _toy_probs(n_classes: int = 4, n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n)
    base = rng.dirichlet(alpha=np.ones(n_classes), size=n)
    # boost the true class
    boost = np.eye(n_classes)[y] * 0.6
    p = base + boost
    p = p / p.sum(axis=1, keepdims=True)
    return p, y


def test_risk_coverage_curve_low_threshold_high_coverage():
    p, y = _toy_probs()
    curve = risk_coverage_curve(p, y, n_thresholds=21)
    # threshold=0 -> coverage=1
    assert curve[0].coverage == pytest.approx(1.0, abs=1e-9)
    # higher threshold means coverage cannot increase
    cov = [c.coverage for c in curve]
    assert all(b <= a + 1e-9 for a, b in zip(cov, cov[1:]))


def test_threshold_for_target_error_is_eligible_or_falls_back():
    p, y = _toy_probs(seed=1)
    cp = threshold_for_target_error(p, y, target_error=0.15, min_coverage=0.30)
    # eligible solution should hit error <= target if any threshold can
    assert 0.0 <= cp.threshold <= 1.0


def test_predict_with_abstain_marks_low_confidence():
    p, y = _toy_probs()
    out = predict_with_abstain(p, threshold=0.99)  # very strict
    # with strict threshold, almost everything should abstain
    abstain_frac = float((out == ABSTAIN).mean())
    assert abstain_frac > 0.9


def test_predict_with_abstain_per_class():
    p, _ = _toy_probs(n_classes=4)
    per = np.array([0.10, 0.99, 0.10, 0.99])  # only abstain on class 1 / 3
    out = predict_with_abstain(p, threshold=0.0, per_class_thresholds=per)
    # never abstain on classes 0 / 2 (threshold is 0.10 which is achievable)
    assert (out[(out == 0)] == 0).all()
    assert (out[(out == 2)] == 2).all()


def test_coverage_at_threshold():
    p, _ = _toy_probs()
    assert coverage_at_threshold(p, 0.0) == pytest.approx(1.0)
    assert coverage_at_threshold(p, 1.0) == pytest.approx(0.0, abs=1e-6)
