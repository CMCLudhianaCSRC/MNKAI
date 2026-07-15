"""Loss functions for class-imbalanced multiclass classification.

Includes:
  - LabelSmoothingCE   : standard CE with label smoothing
  - FocalLoss          : standard multiclass focal loss (correct pt = p_y, NOT
                         the soft-label dot product the legacy trainer used)
  - ClassBalancedFocal : Cui et al. 2019 class-balanced focal loss using the
                         "effective number of samples" reweighting
                         (https://arxiv.org/abs/1901.05555)
  - LDAMLoss           : Cao et al. 2019 label-distribution-aware margin loss
                         (https://arxiv.org/abs/1906.07413)

The previous code path silently fell through `cb_focal` to plain CE because the
trainer only matched `loss == "focal"`. `build_loss()` here is the single
factory used by the new trainer; unknown names raise.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def effective_number_weights(class_counts: Sequence[int], beta: float = 0.9999) -> torch.Tensor:
    """Cui et al. effective-number-of-samples weights, normalized so they sum to C.

    Args
    ----
    class_counts : per-class image counts (length C)
    beta         : 0.999 / 0.9999 typical; closer to 1 = stronger reweighting

    Returns
    -------
    weights : float32 tensor of shape (C,) with mean 1.0
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    if (counts <= 0).any():
        # avoid divide-by-zero for empty classes
        counts = np.maximum(counts, 1.0)
    eff_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(eff_num, 1e-12)
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32)


# -------------------------------------------------------------------- losses --


class LabelSmoothingCE(nn.Module):
    """Thin wrapper around `F.cross_entropy(label_smoothing=...)`."""

    def __init__(self, smoothing: float = 0.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.smoothing = float(smoothing)
        if weight is None:
            self.register_buffer("weight", torch.empty(0))
        else:
            self.register_buffer("weight", weight.float())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.weight if self.weight.numel() > 0 else None
        return F.cross_entropy(logits, target, weight=w, label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    """Standard multiclass focal loss with optional per-class gamma.

    L_i = -alpha_{y_i} * (1 - p_{y_i})^{gamma_{y_i}} * log p_{y_i}

    `alpha` may be a scalar (broadcast to all classes) or a per-class tensor of
    shape (C,) (the latter lets you combine focal with class-balanced weights).

    `gamma` may be a scalar (uniform focal strength) OR a per-class sequence
    of length C — this lets you make the model "paranoid" about specific classes
    (e.g. higher gamma on mel/bkl to prioritise their sensitivity).
    """

    def __init__(
        self,
        alpha: float | Sequence[float] | torch.Tensor | None = None,
        gamma: float | Sequence[float] | torch.Tensor = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.label_smoothing = float(label_smoothing)

        # gamma may be scalar or per-class
        if isinstance(gamma, (int, float)):
            self.gamma = float(gamma)
            self.register_buffer("gamma_vec", torch.empty(0))
        else:
            g = gamma if isinstance(gamma, torch.Tensor) else torch.tensor(list(gamma), dtype=torch.float32)
            self.gamma = float(g.mean().item())  # bookkeeping only
            self.register_buffer("gamma_vec", g.float())

        if alpha is None:
            self.register_buffer("alpha", torch.empty(0))
            self._scalar_alpha = None
        elif isinstance(alpha, (int, float)):
            self.register_buffer("alpha", torch.empty(0))
            self._scalar_alpha = float(alpha)
        else:
            a = alpha if isinstance(alpha, torch.Tensor) else torch.tensor(list(alpha), dtype=torch.float32)
            self.register_buffer("alpha", a.float())
            self._scalar_alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # CE per-sample (with optional label smoothing) — relies on PyTorch's
        # canonical convention so that focal(gamma=0) == cross_entropy.
        ce = F.cross_entropy(logits, target, reduction="none", label_smoothing=self.label_smoothing)
        # focal modulator must come from the TRUE class probability (the legacy
        # bug used the soft-label dot product instead, weakening focal under
        # label smoothing).
        with torch.no_grad():
            pt = F.log_softmax(logits, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1).exp()

        if self.gamma_vec.numel() > 0:
            g = self.gamma_vec.to(logits.device, dtype=ce.dtype)[target]
            loss = (1.0 - pt).pow(g) * ce
        else:
            loss = (1.0 - pt).pow(self.gamma) * ce

        if self._scalar_alpha is not None:
            loss = self._scalar_alpha * loss
        elif self.alpha.numel() > 0:
            loss = self.alpha.to(loss.device, dtype=loss.dtype)[target] * loss
        return loss.mean()


class ClassBalancedFocal(FocalLoss):
    """Class-balanced focal loss (Cui et al. 2019).

    Same formula as FocalLoss, but `alpha` is set to the per-class
    effective-number weights derived from `class_counts`.

    Additional knobs for clinically-prioritised classes:
      - class_boost : per-class multiplicative factor applied AFTER the
                      effective-number weights are computed. Use this to make
                      melanoma / bkl losses count more (e.g. [5,1,1,1,1,3,1,1,1]
                      for the 9-class layout).
      - gamma can be either scalar or a length-C sequence — per-class gamma
                      lets you make the model "paranoid" about specific classes.
    """

    def __init__(
        self,
        class_counts: Sequence[int],
        gamma: float | Sequence[float] = 2.0,
        beta: float = 0.9999,
        label_smoothing: float = 0.0,
        class_boost: Sequence[float] | None = None,
    ):
        weights = effective_number_weights(class_counts, beta=beta)
        if class_boost is not None:
            cb = torch.tensor(list(class_boost), dtype=torch.float32)
            if cb.numel() != weights.numel():
                raise ValueError(
                    f"class_boost length {cb.numel()} != num_classes {weights.numel()}"
                )
            weights = weights * cb
            # Re-normalise so the loss magnitude stays comparable.
            weights = weights / weights.mean()
        super().__init__(alpha=weights, gamma=gamma, label_smoothing=label_smoothing)
        self.beta = float(beta)
        self.class_counts = list(int(x) for x in class_counts)
        self.class_boost = list(float(x) for x in class_boost) if class_boost is not None else None


class LogitAdjustedCE(nn.Module):
    """Logit-adjusted / Balanced-Softmax cross-entropy (Menon et al. 2021;
    Ren et al. 2020, https://arxiv.org/abs/2007.07314 / 2007.10740).

    A SINGLE-MECHANISM long-tail method: train on the NATURAL class frequency
    (no resampling, no loss reweighting) and instead add a per-class log-prior
    offset to the logits inside the softmax:

        loss = CE( logits + tau * log(prior),  y )

    This makes the trained model Bayes-optimal w.r.t. a UNIFORM prior, so at
    inference you use the raw logits directly (no offset) and rare classes are
    no longer suppressed. It avoids the prior-overshoot that resampling +
    reweighting (the Phase-3 collapse) produced, because the correction is
    applied exactly once and is analytically calibrated to the data prior.

    Pair with `use_weighted_sampler: false` (natural-frequency sampling).
    tau=1.0 is the standard Balanced-Softmax; tau in [0.5, 1.5] can be tuned.
    """

    def __init__(self, class_counts: Sequence[float], tau: float = 1.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        counts = np.asarray(class_counts, dtype=np.float64)
        counts = np.maximum(counts, 1.0)
        prior = counts / counts.sum()
        log_prior = np.log(prior + 1e-12) * float(tau)
        self.register_buffer("log_prior", torch.tensor(log_prior, dtype=torch.float32))
        self.tau = float(tau)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        adj = logits + self.log_prior.to(logits.device, dtype=logits.dtype).unsqueeze(0)
        return F.cross_entropy(adj, target, label_smoothing=self.label_smoothing)


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin loss (Cao et al. 2019).

    Adds a per-class additive margin in logit space, where rare classes get
    larger margins. Should be combined with the deferred-reweighting schedule
    (linearly ramp class weights up after a few epochs of plain training).
    """

    def __init__(
        self,
        class_counts: Sequence[int],
        max_m: float = 0.5,
        s: float = 30.0,
        weight: torch.Tensor | None = None,
    ):
        super().__init__()
        counts = np.asarray(class_counts, dtype=np.float64)
        m_list = 1.0 / np.sqrt(np.sqrt(np.maximum(counts, 1.0)))
        m_list = m_list * (max_m / m_list.max())
        self.register_buffer("m_list", torch.tensor(m_list, dtype=torch.float32))
        self.s = float(s)
        if weight is None:
            self.register_buffer("weight", torch.empty(0))
        else:
            self.register_buffer("weight", weight.float())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, target.unsqueeze(1), True)
        batch_m = self.m_list.to(logits.device).unsqueeze(0).expand_as(logits)  # (B, C)
        # subtract margin from the true-class logit, then scale.
        logits_adj = logits - batch_m * index.float()
        logits_adj = self.s * logits_adj
        w = self.weight if self.weight.numel() > 0 else None
        return F.cross_entropy(logits_adj, target, weight=w)


# ------------------------------------------------------------------- factory --


def build_loss(name: str, *, num_classes: int, class_counts: Sequence[int] | None = None,
               label_smoothing: float = 0.0,
               focal_gamma: float | Sequence[float] = 2.0,
               focal_alpha: float | None = None,
               cb_beta: float = 0.9999,
               class_boost: Sequence[float] | None = None,
               la_tau: float = 1.0,
               ldam_max_m: float = 0.5, ldam_s: float = 30.0) -> nn.Module:
    """Return the loss requested by config. Raises on unknown name (no silent fallback).

    Extras for clinical prioritisation:
      - focal_gamma may be a length-C sequence for per-class focal strength.
      - class_boost (cb_focal only) multiplies per-class loss weights by a
        per-class factor (e.g. [5,1,1,1,1,3,1,1,1] to up-weight mel & bkl).
    """
    name = name.lower().strip()
    if name in ("ce", "cross_entropy", "label_smoothing_ce"):
        return LabelSmoothingCE(smoothing=label_smoothing)
    if name == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma, label_smoothing=label_smoothing)
    if name in ("cb_focal", "class_balanced_focal"):
        if class_counts is None:
            raise ValueError("cb_focal requires class_counts")
        if len(class_counts) != num_classes:
            raise ValueError(f"class_counts length {len(class_counts)} != num_classes {num_classes}")
        # validate per-class gamma / boost
        if not isinstance(focal_gamma, (int, float)):
            if len(focal_gamma) != num_classes:
                raise ValueError(f"focal_gamma length {len(focal_gamma)} != num_classes {num_classes}")
        if class_boost is not None and len(class_boost) != num_classes:
            raise ValueError(f"class_boost length {len(class_boost)} != num_classes {num_classes}")
        return ClassBalancedFocal(class_counts=class_counts, gamma=focal_gamma, beta=cb_beta,
                                  label_smoothing=label_smoothing, class_boost=class_boost)
    if name in ("logit_adjusted", "balanced_softmax", "la", "bsm"):
        if class_counts is None:
            raise ValueError("logit_adjusted requires class_counts")
        if len(class_counts) != num_classes:
            raise ValueError(f"class_counts length {len(class_counts)} != num_classes {num_classes}")
        return LogitAdjustedCE(class_counts=class_counts, tau=la_tau,
                               label_smoothing=label_smoothing)
    if name == "ldam":
        if class_counts is None:
            raise ValueError("ldam requires class_counts")
        weights = None
        if cb_beta is not None and cb_beta > 0.0:
            weights = effective_number_weights(class_counts, beta=cb_beta)
        return LDAMLoss(class_counts=class_counts, max_m=ldam_max_m, s=ldam_s, weight=weights)
    raise ValueError(f"unknown loss '{name}'; expected one of: ce, focal, cb_focal, "
                     f"logit_adjusted, ldam")


__all__ = [
    "LabelSmoothingCE",
    "FocalLoss",
    "ClassBalancedFocal",
    "LogitAdjustedCE",
    "LDAMLoss",
    "effective_number_weights",
    "build_loss",
]
