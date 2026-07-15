"""Unify multiple per-source manifests into one master manifest.

Inputs (each a CSV with at least an ID + label-like field):
  - ISIC bundle (one CSV per collection: HAM10000, BCN20000, ISIC2019, ISIC2020,
    MILK10k, PROVe-AI, BALD acral, etc.) — fields per `download.isic._flatten_meta`.
  - Fitzpatrick17k metadata.
  - DermaCon-IN metadata (whatever the dataset ships with).

Output:
  one CSV with the canonical column set:
      filepath, isic_id (optional), label_name, group_id, source, modality,
      sex, age_approx, anatom_site_general, fitzpatrick

Steps:
  - Resolve filepaths to actual files on disk (T7 SSD layout).
  - Drop rows whose label maps to None via LabelNormalizer.
  - Cross-source perceptual-hash dedup (optional; needs imagehash).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
from loguru import logger

from mnkai.data.manifest import LabelNormalizer, normalize_manifest


SOURCE_DEFAULTS = {
    "ham10000": "ISIC_HAM",
    "bcn20000": "ISIC_BCN",
    "isic2019": "ISIC_2019",
    "isic2020": "ISIC_2020",
    "msk": "ISIC_MSK",
    "milk10k": "ISIC_MILK10k",
    "prove_ai": "ISIC_PROVE_AI",
    "bald_acral": "ISIC_BALD_ACRAL",
    "fitzpatrick17k": "FITZ17K",
    "dermacon_in": "INDIA_DERMACON",
}


def _attach_source_and_path(df: pd.DataFrame, *, source: str, image_root: Path,
                            shard_fn=None, id_col: str = "isic_id") -> pd.DataFrame:
    """Infer `filepath` for each row by sharding `id_col` under `image_root`."""
    df = df.copy()
    df["source"] = source
    if "filepath" not in df.columns or df["filepath"].isna().any():
        if shard_fn is None:
            def shard_fn(i: str) -> Path:  # noqa: E306
                if not isinstance(i, str) or not i:
                    return Path("")
                body = i.split("_")[1] if "_" in i else i
                a, b = body[:2], body[2:4] if len(body) >= 4 else "00"
                return image_root / a / b / f"{i}.jpg"
        df["filepath"] = df[id_col].apply(lambda x: str(shard_fn(x)) if pd.notna(x) else "")
    return df


def _ensure_group_id(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Pick the best `group_id` we can from available columns.

    Order of preference: lesion_id, patient_id, md5hash, isic_id (= no grouping).
    """
    df = df.copy()
    if "group_id" in df.columns and df["group_id"].notna().all():
        return df
    if "lesion_id" in df.columns and df["lesion_id"].notna().any():
        df["group_id"] = df["lesion_id"].fillna(df.get("patient_id", ""))
    elif "patient_id" in df.columns and df["patient_id"].notna().any():
        df["group_id"] = df["patient_id"]
    elif "md5hash" in df.columns:
        df["group_id"] = source + "_" + df["md5hash"].astype(str)
    elif "isic_id" in df.columns:
        df["group_id"] = df["isic_id"].astype(str)
    else:
        df["group_id"] = [f"{source}_{i:08d}" for i in range(len(df))]
    df["group_id"] = df["group_id"].astype(str).fillna("")
    df.loc[df["group_id"] == "", "group_id"] = [
        f"{source}_orphan_{i:08d}" for i in range((df["group_id"] == "").sum())
    ]
    return df


def unify_manifests(
    bundles: Sequence[dict],
    *,
    classes: Sequence[str] | None = None,
    dedup_phash: bool = False,
) -> pd.DataFrame:
    """Each bundle: {meta_csv, image_root, source, label_col?, id_col?}.

    Returns the unified dataframe. Caller writes it to disk.
    """
    keep = ["filepath", "isic_id", "label_name", "group_id", "source", "modality",
            "sex", "age_approx", "anatom_site_general", "fitzpatrick"]
    norm = LabelNormalizer(classes=list(classes)) if classes else LabelNormalizer()
    parts: list[pd.DataFrame] = []

    for b in bundles:
        meta_csv = Path(b["meta_csv"])
        if not meta_csv.exists():
            logger.warning(f"[unify] missing {meta_csv}; skipping")
            continue
        raw = pd.read_csv(meta_csv)
        # source column
        src = b.get("source") or SOURCE_DEFAULTS.get(meta_csv.stem.lower(), meta_csv.stem)
        # which column is the label
        label_col = b.get("label_col") or _detect_label_column(raw)
        if label_col is None:
            logger.warning(f"[unify] no label column in {meta_csv}; skipping")
            continue
        if label_col != "label_name":
            raw = raw.rename(columns={label_col: "label_name"})
        # filepath
        raw = _attach_source_and_path(raw, source=src,
                                       image_root=Path(b["image_root"]),
                                       id_col=b.get("id_col", "isic_id"))
        raw = _ensure_group_id(raw, source=src)
        normd = normalize_manifest(raw, norm)
        parts.append(normd)

    if not parts:
        raise RuntimeError("unify: no bundles produced rows")
    df = pd.concat(parts, axis=0, ignore_index=True, sort=False)
    # ensure all keep columns exist
    for c in keep:
        if c not in df.columns:
            df[c] = ""
    df = df[keep + [c for c in df.columns if c not in keep]]

    n_before = len(df)
    if dedup_phash:
        df = _dedup_phash(df)
        logger.info(f"[unify] dedup_phash: {n_before} -> {len(df)}")

    logger.info(f"[unify] final manifest: {len(df)} rows from {df['source'].nunique()} sources")
    logger.info(f"[unify] per-source counts:\n{df['source'].value_counts()}")
    logger.info(f"[unify] per-class counts:\n{df['label_name'].value_counts()}")
    return df


def _detect_label_column(df: pd.DataFrame) -> str | None:
    for c in ("label_name", "label", "diagnosis", "dx", "diagnosis_3", "three_partition_label"):
        if c in df.columns:
            return c
    # case-insensitive
    lc = {c.lower(): c for c in df.columns}
    for c in ("label_name", "label", "diagnosis"):
        if c in lc:
            return lc[c]
    return None


def _dedup_phash(df: pd.DataFrame) -> pd.DataFrame:
    """Drop near-duplicate rows across sources using perceptual hash on filepaths.

    Slow (one open per row); only run when you suspect dataset overlap (e.g.
    HAM10000 vs ISIC 2019). Keeps the row whose source is preferred per the
    `_SOURCE_PREF` order.
    """
    try:
        import imagehash
        from PIL import Image
    except Exception:
        logger.warning("[unify] imagehash not installed; skipping dedup_phash")
        return df

    pref = ["INDIA_DERMACON", "FITZ17K", "ISIC_HAM", "ISIC_BCN", "ISIC_2019",
            "ISIC_2020", "ISIC_MSK", "ISIC_MILK10k", "ISIC_PROVE_AI", "ISIC_BALD_ACRAL"]
    rank = {s: i for i, s in enumerate(pref)}

    hashes: dict[str, list[int]] = {}
    for i, p in enumerate(df["filepath"].tolist()):
        if not p or not Path(p).exists():
            continue
        try:
            with Image.open(p) as im:
                h = str(imagehash.phash(im.convert("RGB")))
        except Exception:
            continue
        hashes.setdefault(h, []).append(i)

    keep = set(range(len(df)))
    for _h, idxs in hashes.items():
        if len(idxs) <= 1:
            continue
        # keep the row from the most-preferred source; drop rest
        idxs.sort(key=lambda i: rank.get(df.iloc[i]["source"], 999))
        for i in idxs[1:]:
            keep.discard(i)
    return df.iloc[sorted(keep)].reset_index(drop=True)


__all__ = ["unify_manifests", "SOURCE_DEFAULTS"]
