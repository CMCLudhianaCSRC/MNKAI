"""Learning-rate schedulers.

Two options used in the new trainer:
  - cosine_with_warmup       : single cosine decay, linear warmup
  - cosine_warm_restarts     : SGDR (Loshchilov & Hutter 2017). The PPT
                               promises this; the legacy code only had
                               plain cosine.
"""
from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


def cosine_with_warmup(optim: Optimizer, total_epochs: int, warmup_epochs: int = 0) -> torch.optim.lr_scheduler.LambdaLR:
    def lam(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        prog = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optim, lam)


def cosine_warm_restarts(
    optim: Optimizer,
    total_epochs: int,
    *,
    first_cycle: int = 10,
    cycle_mult: float = 2.0,
    warmup_epochs: int = 0,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """SGDR-style cosine annealing with warm restarts (per-epoch step).

    Cycle k has length first_cycle * cycle_mult**k. Within each cycle, LR
    decays from base_lr to min_lr_ratio * base_lr. Across restarts, base_lr is
    held constant (no decay across restarts) so that the schedule is
    interpretable; this matches Loshchilov & Hutter 2017.
    """
    def lam(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        e = epoch - warmup_epochs
        # find current cycle index k and position within it
        cycle_len = first_cycle
        cycle_start = 0
        k = 0
        while True:
            if e < cycle_start + cycle_len:
                pos = (e - cycle_start) / max(1, cycle_len)
                cos = 0.5 * (1.0 + math.cos(math.pi * pos))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cos
            cycle_start += cycle_len
            cycle_len = max(1, int(round(cycle_len * cycle_mult)))
            k += 1
            if cycle_start > total_epochs * 4:  # runaway safety
                return min_lr_ratio
    return torch.optim.lr_scheduler.LambdaLR(optim, lam)


def build_scheduler(name: str, optim: Optimizer, total_epochs: int, *,
                    warmup_epochs: int = 0,
                    sgdr_first_cycle: int = 10,
                    sgdr_cycle_mult: float = 2.0,
                    min_lr_ratio: float = 0.01) -> torch.optim.lr_scheduler.LambdaLR:
    name = name.lower().strip()
    if name in ("cosine", "cosine_warmup"):
        return cosine_with_warmup(optim, total_epochs, warmup_epochs)
    if name in ("sgdr", "cosine_restarts", "cosine_warm_restarts"):
        return cosine_warm_restarts(
            optim, total_epochs,
            first_cycle=sgdr_first_cycle, cycle_mult=sgdr_cycle_mult,
            warmup_epochs=warmup_epochs, min_lr_ratio=min_lr_ratio,
        )
    raise ValueError(f"unknown scheduler '{name}'; expected cosine or sgdr")


__all__ = ["build_scheduler", "cosine_with_warmup", "cosine_warm_restarts"]
