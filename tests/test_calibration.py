"""Tests for temperature calibration."""
from __future__ import annotations

import numpy as np
import pytest

from mnkai.evaluation.calibration import (
    GroupTemperatureScaler,
    TemperatureScaler,
)


def _logits_overconfident(seed=0, n=500, c=4, scale=4.0):
    """Generate logits and labels where the model is informative but
    over-confident (T should converge to >1)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, c, n)
    base = rng.standard_normal((n, c)) * 0.3
    base[np.arange(n), y] += 1.5     # signal
    return base * scale, y           # multiply -> over-confident


def test_temperature_scaler_reduces_ece_when_overconfident():
    # Generate genuinely miscalibrated predictions: scale logits *and* corrupt
    # 25% of labels so that confidence outpaces accuracy -> ECE > 0, T > 1.
    rng = np.random.default_rng(0)
    n, c = 1500, 4
    y = rng.integers(0, c, n)
    base = rng.standard_normal((n, c)) * 0.4
    base[np.arange(n), y] += 1.4
    logits = base * 4.0  # over-confident
    flip = rng.random(n) < 0.25
    y_obs = np.where(flip, (y + 1) % c, y)  # 25% wrong labels visible to scaler

    scaler = TemperatureScaler()
    rep = scaler.fit(logits, y_obs, max_iter=200, lr=0.1)
    assert rep.ece_after <= rep.ece_before + 1e-6
    assert rep.nll_after <= rep.nll_before + 1e-6
    assert scaler.T > 1.0  # over-confident -> T > 1


def test_temperature_scaler_T_is_finite():
    logits, y = _logits_overconfident()
    scaler = TemperatureScaler()
    scaler.fit(logits, y)
    assert np.isfinite(scaler.T) and scaler.T > 0


def test_group_temperature_scaler_fits_groups_and_global():
    logits, y = _logits_overconfident(n=1000, c=4, scale=3.5)
    groups = np.array(["a"] * 500 + ["b"] * 500)
    scaler = GroupTemperatureScaler(group_keys=["a", "b"])
    rep = scaler.fit(logits, y, list(groups))
    assert rep.ece_after <= rep.ece_before + 1e-6
    Ts = rep.temperature
    assert isinstance(Ts, dict)
    assert "a" in Ts and "b" in Ts
    assert all(t > 0 for t in Ts.values())


def test_group_temperature_tiny_groups_fall_back_to_global():
    logits, y = _logits_overconfident(n=200, c=4, scale=3.0)
    groups = np.array(["a"] * 195 + ["b"] * 5)
    scaler = GroupTemperatureScaler(group_keys=["a", "b"])
    rep = scaler.fit(logits, y, list(groups))
    # (R16) b has fewer than 30 samples -> it now falls back to the fitted GLOBAL
    # temperature instead of being left uncalibrated at T=1 (the old buggy no-op).
    global_T = scaler.temperature_for("an_unknown_group")   # returns the global T
    assert rep.temperature["b"] == pytest.approx(global_T, abs=1e-2)
    assert abs(rep.temperature["b"] - 1.0) > 1e-2
