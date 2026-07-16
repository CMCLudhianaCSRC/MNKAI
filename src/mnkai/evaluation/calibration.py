"""Confidence calibration.

  - TemperatureScaler            : scalar temperature on logits (Guo 2017).
  - GroupTemperatureScaler       : one temperature per subgroup (e.g. per-FST)
                                   so calibration is equitable across skin tones.
  - vector_temperature_scale     : per-class temperatures, fit jointly.

Returned models are tiny torch.nn.Modules; their `forward(logits) -> probs`
applies the scaling and softmax. Fitting uses LBFGS on NLL.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CalibrationFitReport:
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float
    temperature: float | list[float] | dict[str, float]


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not m.any():
            continue
        ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


class TemperatureScaler(nn.Module):
    """Single scalar T s.t. probs = softmax(logits / T)."""

    def __init__(self, init_T: float = 1.0):
        super().__init__()
        # parameterize as log_T to keep T > 0 without constraints
        self.log_T = nn.Parameter(torch.tensor(float(np.log(init_T))))

    @property
    def T(self) -> float:
        return float(self.log_T.exp().detach().cpu())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits / self.log_T.exp(), dim=-1)

    def fit(self, logits: np.ndarray, labels: np.ndarray, *,
            max_iter: int = 100, lr: float = 0.05) -> CalibrationFitReport:
        """Fit T on a held-out (val) set's logits + labels."""
        logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
        labels_t = torch.as_tensor(np.asarray(labels), dtype=torch.long)

        # before
        with torch.no_grad():
            p_before = F.softmax(logits_t, dim=-1).cpu().numpy()
        nll_b = _nll(p_before, labels_t.numpy())
        ece_b = _ece(p_before, labels_t.numpy())

        opt = torch.optim.LBFGS([self.log_T], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            T = self.log_T.exp()
            loss = F.cross_entropy(logits_t / T, labels_t)
            loss.backward()
            return loss

        opt.step(closure)

        with torch.no_grad():
            p_after = self.forward(logits_t).cpu().numpy()
        nll_a = _nll(p_after, labels_t.numpy())
        ece_a = _ece(p_after, labels_t.numpy())

        return CalibrationFitReport(
            nll_before=nll_b, nll_after=nll_a, ece_before=ece_b, ece_after=ece_a,
            temperature=self.T,
        )


class GroupTemperatureScaler(nn.Module):
    """One temperature per subgroup. `groups` is a numpy/torch tensor of group IDs.

    Use case: per-Fitzpatrick calibration. Calling forward without `groups`
    falls back to a "global" temperature fit on all data.
    """

    def __init__(self, group_keys: list[str]):
        super().__init__()
        self.group_keys = list(group_keys)
        self.idx = {k: i for i, k in enumerate(self.group_keys)}
        self.log_T = nn.Parameter(torch.zeros(len(self.group_keys)))  # T_init = 1
        self._global = TemperatureScaler()

    def temperature_for(self, key: str) -> float:
        if key not in self.idx:
            return self._global.T
        return float(self.log_T[self.idx[key]].exp().detach().cpu())

    def forward(self, logits: torch.Tensor, groups: list[str] | None = None) -> torch.Tensor:
        if groups is None:
            return self._global(logits)
        T = torch.empty(len(groups), 1, device=logits.device)
        for i, g in enumerate(groups):
            j = self.idx.get(g)
            if j is None:
                T[i] = self._global.log_T.exp().to(logits.device)
            else:
                T[i] = self.log_T[j].exp().to(logits.device)
        return F.softmax(logits / T, dim=-1)

    def fit(self, logits: np.ndarray, labels: np.ndarray, groups: list[str], *,
            max_iter: int = 100, lr: float = 0.05) -> CalibrationFitReport:
        logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
        labels_t = torch.as_tensor(np.asarray(labels), dtype=torch.long)
        n = len(labels)
        if len(groups) != n:
            raise ValueError("groups length must match labels")
        # build per-row temperature index
        gi = np.array([self.idx.get(g, -1) for g in groups])
        # rows with unknown group are calibrated by the global scaler
        # — fit global on all data first
        self._global.fit(logits, labels, max_iter=max_iter, lr=lr)
        # fit per-group
        for k, j in self.idx.items():
            mask = gi == j
            if mask.sum() < 30:  # (R16) too few samples to fit a reliable subgroup T:
                # fall back to the fitted GLOBAL temperature rather than leaving this
                # known subgroup at T=1 (uncalibrated) — the global estimate is
                # strictly better and preserves the equitable-calibration goal.
                with torch.no_grad():
                    self.log_T[j] = self._global.log_T.detach().reshape(-1)[0]
                continue
            sub_logits = logits_t[mask]
            sub_labels = labels_t[mask]
            log_t = nn.Parameter(self.log_T[j].detach().clone())
            opt = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter,
                                    line_search_fn="strong_wolfe")
            def closure():
                opt.zero_grad()
                T = log_t.exp()
                loss = F.cross_entropy(sub_logits / T, sub_labels)
                loss.backward()
                return loss
            opt.step(closure)
            with torch.no_grad():
                self.log_T[j] = log_t.detach()

        # report metrics with combined per-row T
        T_per_row = torch.where(
            torch.tensor(gi >= 0), self.log_T[torch.tensor(np.maximum(gi, 0))].exp(),
            torch.tensor(self._global.T)
        ).unsqueeze(1)
        with torch.no_grad():
            p_after = F.softmax(logits_t / T_per_row, dim=-1).cpu().numpy()
            p_before = F.softmax(logits_t, dim=-1).cpu().numpy()
        return CalibrationFitReport(
            nll_before=_nll(p_before, labels_t.numpy()),
            nll_after=_nll(p_after, labels_t.numpy()),
            ece_before=_ece(p_before, labels_t.numpy()),
            ece_after=_ece(p_after, labels_t.numpy()),
            temperature={k: self.temperature_for(k) for k in self.group_keys},
        )


__all__ = [
    "TemperatureScaler",
    "GroupTemperatureScaler",
    "CalibrationFitReport",
]
