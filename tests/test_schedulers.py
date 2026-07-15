"""Tests for SGDR / cosine scheduler."""
from __future__ import annotations

import torch

from mnkai.training.schedulers import build_scheduler


def _trace(sched, total):
    out = []
    for _ in range(total):
        out.append(sched.get_last_lr()[0])
        sched.step()
    return out


def test_cosine_with_warmup_starts_low_and_decays():
    p = torch.nn.Parameter(torch.zeros(1))
    optim = torch.optim.SGD([p], lr=1.0)
    sched = build_scheduler("cosine", optim, total_epochs=20, warmup_epochs=3)
    lrs = _trace(sched, 20)
    # warmup: monotonically increasing for first 3 epochs
    assert lrs[0] < lrs[1] < lrs[2]
    # final lr below initial (cosine decay)
    assert lrs[-1] < lrs[3]
    assert all(x >= -1e-9 for x in lrs)


def test_sgdr_has_restarts():
    p = torch.nn.Parameter(torch.zeros(1))
    optim = torch.optim.SGD([p], lr=1.0)
    sched = build_scheduler("sgdr", optim, total_epochs=30, warmup_epochs=0,
                            sgdr_first_cycle=5, sgdr_cycle_mult=2.0,
                            min_lr_ratio=0.01)
    lrs = _trace(sched, 30)
    # At cycle boundary (epoch 5), lr should jump back up vs lr just before
    assert lrs[5] > lrs[4]
    # lr stays in [min_lr_ratio - eps, 1.0 + eps]
    assert min(lrs) >= 0.01 - 1e-6
    assert max(lrs) <= 1.0 + 1e-6


def test_unknown_scheduler_raises():
    p = torch.nn.Parameter(torch.zeros(1))
    optim = torch.optim.SGD([p], lr=1.0)
    try:
        build_scheduler("nonsense", optim, total_epochs=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
