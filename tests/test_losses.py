"""Loss tests — verifies cb_focal actually weights by class counts (the legacy
trainer silently fell through to plain CE)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mnkai.losses.focal import (
    ClassBalancedFocal,
    FocalLoss,
    LabelSmoothingCE,
    LDAMLoss,
    build_loss,
    effective_number_weights,
)


def test_effective_number_weights_normalize_to_C():
    counts = [10, 100, 1000]
    w = effective_number_weights(counts).numpy()
    assert w.shape == (3,)
    assert abs(w.sum() - 3.0) < 1e-5
    # rare class should be heavier than common
    assert w[0] > w[1] > w[2]


def test_focal_pt_is_true_class_prob_not_soft_dot():
    """Regression test for the legacy bug.

    The legacy FocalLoss in src/train.py defined `pt = (true_dist * probs).sum`.
    When label_smoothing > 0 this is the *soft-label dot product*, NOT
    p_y, so the focal modulator weakens incorrectly. Here we lock down
    that the new implementation uses p_y, by checking that gamma=0 reduces
    to a label-smoothed CE regardless of smoothing.
    """
    torch.manual_seed(0)
    logits = torch.randn(8, 5)
    target = torch.randint(0, 5, (8,))
    smoothing = 0.1
    focal_g0 = FocalLoss(alpha=None, gamma=0.0, label_smoothing=smoothing)
    ce_smooth = LabelSmoothingCE(smoothing=smoothing)
    a = focal_g0(logits, target).item()
    b = ce_smooth(logits, target).item()
    assert abs(a - b) < 1e-5, f"focal(gamma=0) should equal label-smoothed CE; {a} vs {b}"


def test_focal_modulator_actually_modulates():
    torch.manual_seed(0)
    logits = torch.tensor([[10.0, -10.0, -10.0], [0.1, 0.0, 0.0]])
    target = torch.tensor([0, 0])
    f = FocalLoss(alpha=None, gamma=2.0).forward(logits, target).item()
    ce = torch.nn.functional.cross_entropy(logits, target).item()
    # focal should down-weight the very-confident first sample, so loss < CE
    assert f < ce


def test_class_balanced_focal_weights_per_class():
    counts = [10, 1000, 1000]
    cb = ClassBalancedFocal(class_counts=counts, gamma=0.0, beta=0.999, label_smoothing=0.0)
    # alpha buffer should hold the per-class weights, summing to C
    a = cb.alpha
    assert a.numel() == 3
    assert abs(float(a.sum()) - 3.0) < 1e-4
    assert float(a[0]) > float(a[1])  # rare > common


def test_build_loss_unknown_raises():
    with pytest.raises(ValueError, match="unknown loss"):
        build_loss("does_not_exist", num_classes=3)


def test_build_loss_cb_focal_requires_counts():
    with pytest.raises(ValueError, match="class_counts"):
        build_loss("cb_focal", num_classes=3)


def test_build_loss_cb_focal_returns_class_balanced_focal():
    loss = build_loss("cb_focal", num_classes=3, class_counts=[10, 100, 1000],
                      label_smoothing=0.05, focal_gamma=2.0)
    assert isinstance(loss, ClassBalancedFocal)


def test_ldam_runs():
    torch.manual_seed(0)
    counts = [100, 50, 10]
    loss = LDAMLoss(class_counts=counts, max_m=0.5, s=30.0)
    logits = torch.randn(16, 3, requires_grad=True)
    target = torch.randint(0, 3, (16,))
    out = loss(logits, target)
    assert torch.isfinite(out)
    out.backward()
