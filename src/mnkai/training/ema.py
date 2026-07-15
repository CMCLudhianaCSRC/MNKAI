"""Lightweight EMA wrapper."""
from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


class ModelEMA:
    """Exponential moving average of model parameters.

    Maintained as a duplicate model on the same device. Use `.module` for eval.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach().to(v.dtype), alpha=1.0 - d)
            else:
                v.copy_(msd[k])
