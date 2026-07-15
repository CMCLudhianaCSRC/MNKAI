"""Sampling strategies."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


def class_balanced_weights(labels: Sequence[int], num_classes: int,
                            power: float = 1.0) -> np.ndarray:
    """Per-sample weights = 1 / (class_count ** power). Mean-normalized to 1.0.

    `power` controls how aggressively classes are re-balanced:
      power=1.0  -> full inverse-frequency (every class equal per-epoch share).
                    AGGRESSIVE: stacked with cb_focal this caused the Phase-3
                    majority-class collapse (the effective prior overshoots far
                    past balanced into rare-class-dominant).
      power=0.5  -> "square-root" sampling: gentle re-balance that preserves
                    some natural-frequency signal (recommended default for
                    long-tail problems where the common classes are the targets).
      power=0.0  -> no re-balancing (natural frequency; use with a loss-side
                    method like logit-adjustment instead).
    """
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    cls_w = 1.0 / np.power(counts, float(power))
    cls_w = cls_w / cls_w.mean()
    return cls_w[labels]


def make_weighted_sampler(
    df: pd.DataFrame, classes: Sequence[str], *, label_col: str = "label_name",
    source_col: str | None = None, source_weights: dict | None = None,
    class_boost: dict | None = None,
    sample_weights: np.ndarray | None = None,
    power: float = 1.0,
    seed: int = 0,
) -> WeightedRandomSampler:
    """Class-balanced + (optionally) source-weighted random sampler.

    The class-balance term gives every class equal expected representation per
    epoch. The source term (if given) up- or down-weights specific data sources;
    e.g. set source_weights={'INDIA': 0.7, 'ISIC': 0.3} for the Indian fine-tune
    phase.

    Extras:
      class_boost   : per-class extra multiplier on top of class balance, e.g.
                      {'mel': 1.5, 'bkl': 1.3} ups how often those classes are
                      sampled (clinical prioritisation).
      sample_weights: per-row pre-computed weights from hard-negative mining
                      (multiplied with the class-balanced weights).
    """
    classes = [c.lower() for c in classes]
    cls_idx = {c: i for i, c in enumerate(classes)}
    labels = df[label_col].astype(str).str.lower().map(cls_idx).to_numpy()
    weights = class_balanced_weights(labels, num_classes=len(classes), power=power)

    if class_boost:
        boost_vec = np.ones(len(classes), dtype=np.float64)
        for cls_name, factor in class_boost.items():
            i = cls_idx.get(str(cls_name).lower())
            if i is None:
                continue
            boost_vec[i] = float(factor)
        weights = weights * boost_vec[labels]
        weights = weights / max(weights.mean(), 1e-12)

    if source_col and source_weights and source_col in df.columns:
        src = df[source_col].astype(str).str.upper()
        sw = np.ones(len(df), dtype=np.float64)
        for k, v in source_weights.items():
            sw[src.str.contains(str(k).upper(), na=False)] = float(v)
        weights = weights * sw
        weights = weights / max(weights.mean(), 1e-12)

    if sample_weights is not None:
        if len(sample_weights) != len(weights):
            raise ValueError(
                f"sample_weights length {len(sample_weights)} != df length {len(weights)}"
            )
        weights = weights * np.asarray(sample_weights, dtype=np.float64)
        weights = weights / max(weights.mean(), 1e-12)

    g = torch.Generator()
    g.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(df),
        replacement=True,
        generator=g,
    )


__all__ = ["make_weighted_sampler", "class_balanced_weights"]
