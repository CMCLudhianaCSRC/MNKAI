"""Unified SkinDataset.

Replaces the two parallel datasets in the legacy repo:

  - `src_legacy/dataset.py`           (image only)
  - `src_legacy/utils/datasets/isic_streaming_mm.py`  (image + tabular)

This single class can return either modality based on the `tabular_features`
argument; downstream models pick what to consume.

The dataset is *manifest-driven*: it never walks the filesystem. The manifest
must contain at minimum `filepath` and `label_name` columns; everything else
(group_id, source, sex, age, anatomical site, fitzpatrick) is optional.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


SEX_MAP = {"male": 0, "female": 1}
SITE_VOCAB = (
    "anterior torso", "posterior torso", "lateral torso", "head/neck",
    "upper extremity", "lower extremity", "palms/soles", "oral/genital",
    "unknown",
)


@dataclass
class TabularSpec:
    """Defines which manifest columns become tabular features."""
    use_sex: bool = True
    use_age: bool = True
    use_site: bool = True
    use_fitzpatrick: bool = False
    age_max: float = 100.0  # for normalization
    site_vocab: tuple[str, ...] = field(default_factory=lambda: SITE_VOCAB)

    @property
    def dim(self) -> int:
        d = 0
        if self.use_sex:
            d += 3   # male, female, unknown one-hot
        if self.use_age:
            d += 2   # normalized age + missing-flag
        if self.use_site:
            d += len(self.site_vocab)
        if self.use_fitzpatrick:
            d += 7   # 6 FST classes + missing
        return d


def _normalize_str(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip().lower()


def _site_index(site: str, vocab: Sequence[str]) -> int:
    s = _normalize_str(site)
    for i, v in enumerate(vocab):
        if v in s or s in v:
            return i
    return len(vocab) - 1  # unknown bucket


def encode_tabular(row: pd.Series, spec: TabularSpec) -> np.ndarray:
    parts: list[np.ndarray] = []
    if spec.use_sex:
        sex = _normalize_str(row.get("sex"))
        v = np.zeros(3, dtype=np.float32)
        if sex == "male":
            v[0] = 1.0
        elif sex == "female":
            v[1] = 1.0
        else:
            v[2] = 1.0
        parts.append(v)
    if spec.use_age:
        age_raw = row.get("age_approx")
        try:
            age = float(age_raw)
            ok = np.isfinite(age)
        except (TypeError, ValueError):
            age, ok = 0.0, False
        v = np.array([age / spec.age_max if ok else 0.0, 0.0 if ok else 1.0], dtype=np.float32)
        parts.append(v)
    if spec.use_site:
        idx = _site_index(row.get("anatom_site_general", ""), spec.site_vocab)
        v = np.zeros(len(spec.site_vocab), dtype=np.float32)
        v[idx] = 1.0
        parts.append(v)
    if spec.use_fitzpatrick:
        fst = row.get("fitzpatrick")
        v = np.zeros(7, dtype=np.float32)
        try:
            fi = int(fst)
            if 1 <= fi <= 6:
                v[fi - 1] = 1.0
            else:
                v[6] = 1.0
        except (TypeError, ValueError):
            v[6] = 1.0  # missing
        parts.append(v)
    return np.concatenate(parts, axis=0) if parts else np.zeros(0, dtype=np.float32)


class SkinDataset(Dataset):
    """Image (+ optional tabular) dataset for dermatology classification.

    Args
    ----
    df : DataFrame slice with `filepath` and `label_name` columns. The slice's
         row order is preserved as the dataset's __getitem__ order.
    classes : canonical class name list (lowercase). Defines class-index mapping.
    transform : callable mapping PIL.Image -> torch.Tensor (use build_train/build_valid).
    images_dir : optional prefix prepended to relative `filepath`s.
    tabular : TabularSpec or None. If None, returns only image+target.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        classes: Sequence[str],
        transform,
        *,
        images_dir: str | Path | None = None,
        tabular: TabularSpec | None = None,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.classes = [str(c).strip().lower() for c in classes]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform
        self.images_dir = Path(images_dir) if images_dir else None
        self.tabular = tabular

        # auto-detect column names
        cols = {c.lower(): c for c in self.df.columns}
        self._path_col = (
            cols.get("filepath") or cols.get("image") or cols.get("image_path") or cols.get("filename")
        )
        self._label_col = cols.get("label_name") or cols.get("label") or cols.get("diagnosis")
        if self._path_col is None or self._label_col is None:
            raise ValueError(
                f"manifest needs 'filepath' and 'label_name' (or aliases); got {list(self.df.columns)}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def _resolve(self, p: str) -> Path:
        P = Path(p)
        if P.is_absolute():
            return P
        if self.images_dir is not None:
            return self.images_dir / P
        return P

    def get_label_index(self, label: str) -> int:
        idx = self.class_to_idx.get(_normalize_str(label), -1)
        if idx < 0:
            raise ValueError(f"unknown label '{label}'; expected one of {self.classes}")
        return idx

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # (R10) Resilient to *transient* USB-drive flakiness, but FAIL LOUD on a
        # genuinely unreadable file — never silently substitute a different sample or
        # a zero image. The old behaviour (retry a same-label sample, then emit a
        # hardcoded 3x224x224 zero image carrying the true label) had three defects:
        #   1. a disconnected drive made every read miss, so the model trained on
        #      all-zero images with valid labels and only ONE warning (silent garbage);
        #   2. same-label substitution biased the sampled distribution toward the
        #      classes whose files happened to be readable;
        #   3. the zero image was hardcoded at 224px and breaks the default collate
        #      at the 518px training resolution anyway.
        # We now retry the SAME path a few times (handles a briefly-sleeping USB drive)
        # and then raise. On a valid run the first open() succeeds -> behaviour is
        # unchanged and no result is affected. (`time` imported at module scope.)
        row = self.df.iloc[idx]
        path = self._resolve(str(row[self._path_col]))
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                img = Image.open(path).convert("RGB")
                img_t = self.transform(img)
                target = self.get_label_index(row[self._label_col])
                out: dict[str, Any] = {
                    "image": img_t,
                    "target": torch.tensor(target, dtype=torch.long),
                    "path": str(path),
                    "row_index": int(idx),
                }
                if self.tabular is not None:
                    out["tabular"] = torch.from_numpy(encode_tabular(row, self.tabular))
                return out
            except (FileNotFoundError, OSError) as e:
                last_err = e
                if attempt == 0:
                    try:
                        from loguru import logger as _log
                        _log.warning(f"[dataset] read miss {path} (row {idx}); retrying")
                    except Exception:
                        pass
                time.sleep(0.2)   # give a flaky USB drive a moment to wake up
        raise FileNotFoundError(
            f"[dataset] could not read image for row {idx} after 3 attempts: {path}. "
            f"Refusing to fall back to a zero image (that once let a disconnected drive "
            f"train the whole model on all-zero inputs). Check the data drive is mounted "
            f"and the manifest path is correct. Last error: {last_err}"
        )


__all__ = ["SkinDataset", "TabularSpec", "encode_tabular", "SITE_VOCAB"]
