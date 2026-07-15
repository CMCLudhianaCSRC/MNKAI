"""Image QC + EXIF stripping.

Functions:
  - corrupt_or_blank      : returns reason or None
  - blur_score            : edge-map (PIL FIND_EDGES) variance, a Laplacian-style
                            sharpness proxy (lower = blurrier)
  - exposure_outlier      : flags very dark or very bright images
  - phash                 : perceptual hash (used for cross-source dedup)
  - strip_exif            : write a copy without EXIF / metadata (PPT ethics)
  - run_qc                : iterate a manifest, emit a per-row qc_report

The PPT's ethics slide commits to EXIF removal; the new ingestion pipeline
runs `strip_exif` over every image before training and inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image, ImageFilter


@dataclass
class QCResult:
    path: str
    ok: bool
    reason: str = ""
    blur: float | None = None
    brightness: float | None = None
    width: int | None = None
    height: int | None = None
    phash: str | None = None


def corrupt_or_blank(path: str | Path) -> str | None:
    try:
        with Image.open(path) as im:
            im.verify()
    except Exception as e:
        return f"corrupt: {e}"
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            arr = np.asarray(im)
        if arr.size == 0:
            return "empty"
        if arr.std() < 2.0:
            return "blank/near-uniform"
    except Exception as e:
        return f"unreadable: {e}"
    return None


def blur_score(path: str | Path) -> float:
    """Edge-map (PIL ``FIND_EDGES``) response variance — a Laplacian-style sharpness
    proxy. Lower = blurrier. <50 is usually unusable.

    Note: this uses PIL's ``FIND_EDGES`` convolution rather than a true Laplacian
    (no OpenCV dependency); it is monotonically related to sharpness the same way,
    but the absolute values differ from ``cv2.Laplacian(...).var()``.
    """
    with Image.open(path) as im:
        g = im.convert("L")
        l = g.filter(ImageFilter.FIND_EDGES)
        arr = np.asarray(l, dtype=np.float32)
    return float(arr.var())


def exposure_score(path: str | Path) -> float:
    """Mean luminance (0-255)."""
    with Image.open(path) as im:
        g = im.convert("L")
        arr = np.asarray(g, dtype=np.float32)
    return float(arr.mean())


def phash(path: str | Path, hash_size: int = 16) -> str:
    """Perceptual hash; uses imagehash if available, else a tiny fallback."""
    try:
        import imagehash
        with Image.open(path) as im:
            return str(imagehash.phash(im.convert("RGB"), hash_size=hash_size))
    except Exception:
        # tiny fallback: dHash on resized grayscale
        with Image.open(path) as im:
            g = im.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
            arr = np.asarray(g, dtype=np.int32)
        diff = arr[:, 1:] > arr[:, :-1]
        return "".join("1" if x else "0" for x in diff.flatten().tolist())


def strip_exif(src: str | Path, dst: str | Path) -> None:
    """Write a JPEG copy of `src` to `dst` with EXIF and metadata removed.

    PPT ethics commitment. We re-encode at quality 95 (visually lossless) and
    drop all PIL/Exif info.
    """
    src = Path(src); dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        # build a fresh image with no info dict (drops exif/icc/etc.)
        clean = Image.new(im.mode, im.size)
        clean.putdata(list(im.getdata()))
        clean.save(dst, format="JPEG", quality=95, optimize=True)


def run_qc(paths: Iterable[str | Path], *, blur_min: float = 50.0,
           dark_max: float = 20.0, bright_min: float = 235.0) -> pd.DataFrame:
    """Iterate paths and return a per-image QC report.

    Includes a `phash` column suitable for `df.duplicated(subset='phash')`
    deduplication across sources.
    """
    rows: list[QCResult] = []
    for p in paths:
        p = str(p)
        reason = corrupt_or_blank(p)
        if reason is not None:
            rows.append(QCResult(path=p, ok=False, reason=reason))
            continue
        try:
            with Image.open(p) as im:
                w, h = im.size
            b = blur_score(p)
            e = exposure_score(p)
            ph = phash(p)
        except Exception as ex:
            rows.append(QCResult(path=p, ok=False, reason=f"qc_error: {ex}"))
            continue
        bad = []
        if b < blur_min:
            bad.append(f"blur({b:.1f})")
        if e < dark_max:
            bad.append(f"too_dark({e:.1f})")
        if e > bright_min:
            bad.append(f"too_bright({e:.1f})")
        rows.append(QCResult(path=p, ok=not bad, reason=";".join(bad),
                             blur=b, brightness=e, width=w, height=h, phash=ph))
    df = pd.DataFrame([r.__dict__ for r in rows])
    if len(df):
        n_ok = int(df["ok"].sum())
        logger.info(f"[qc] {n_ok}/{len(df)} pass; failures: {df.loc[~df['ok'], 'reason'].value_counts().to_dict()}")
    return df


__all__ = [
    "QCResult",
    "corrupt_or_blank",
    "blur_score",
    "exposure_score",
    "phash",
    "strip_exif",
    "run_qc",
]
