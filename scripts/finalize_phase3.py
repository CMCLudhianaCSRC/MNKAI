"""Phase 3 post-processing orchestrator.

Run AFTER all downloads are complete. Builds the master manifest, runs QC,
deduplicates by perceptual hash, and emits the K-fold splits with MSK as the
external test domain.

  python -m scripts.finalize_phase3 \
      --isic-meta data/manifests/isic_combined.csv \
      --isic-root D:\AIFINAL\images\isic \
      --fitz17k-meta data/manifests/fitzpatrick17k.csv \
      --fitz17k-root D:\AIFINAL\images\fitz17k \
      --dermacon-root D:\AIFINAL\images\dermacon_in \
      --out-master data/manifests/master.csv \
      --out-clean  data/manifests/master_clean.csv \
      --out-splits data/splits/master.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from mnkai.data.qc import run_qc
from mnkai.data.splits import SplitConfig, assert_no_group_leakage, make_splits
from mnkai.data.unify import unify_manifests
from mnkai.utils.logging import setup_logging


def _find_image_for_dermacon(dermacon_root: Path, image_id: str) -> str:
    """DermaCon-IN extracts to a doubly-nested folder:
       dermacon_in/DATASET/DATASET_<n>/DATASET_<n>/IMG_NNNN.jpg
    Search the tree for a filename match. Cache the lookup by walking once.
    """
    cache: dict[str, str] = getattr(_find_image_for_dermacon, "_cache", {})
    if not cache:
        for p in dermacon_root.rglob("*.jpg"):
            cache[p.stem] = str(p)
            cache[p.name] = str(p)
        _find_image_for_dermacon._cache = cache
    return cache.get(image_id, "") or cache.get(f"{image_id}.jpg", "")


def _materialize_dermacon_metadata(dermacon_root: Path, out_csv: Path) -> Path | None:
    """DermaCon-IN ships `Skin_Metadata.tab`. Rebuild a CSV with `filepath`,
    `label_name`, `group_id`, plus the FST column if present."""
    md_tab = None
    for cand in dermacon_root.rglob("Skin_Metadata*"):
        md_tab = cand
        break
    if md_tab is None:
        logger.warning(f"[finalize] no Skin_Metadata.tab under {dermacon_root}")
        return None
    df = pd.read_csv(md_tab, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    image_col = (cols.get("image_id") or cols.get("filename") or cols.get("image")
                 or cols.get("img_id") or cols.get("file_name"))
    label_col = (cols.get("diagnosis") or cols.get("label") or cols.get("class")
                 or cols.get("disease") or cols.get("category"))
    if image_col is None or label_col is None:
        logger.warning(f"[finalize] DermaCon metadata missing image/label cols; got {list(df.columns)}")
        return None
    out = pd.DataFrame()
    out["dermacon_id"] = df[image_col].astype(str)
    out["filepath"] = out["dermacon_id"].apply(lambda i: _find_image_for_dermacon(dermacon_root, i))
    out["label_name"] = df[label_col].astype(str)
    if "patient_id" in cols:
        out["patient_id"] = df[cols["patient_id"]]
    for fst_alias in ("fitzpatrick", "fst", "skin_type", "skin_tone"):
        if fst_alias in cols:
            out["fitzpatrick"] = df[cols[fst_alias]]
            break
    if "sex" in cols:
        out["sex"] = df[cols["sex"]]
    elif "gender" in cols:
        out["sex"] = df[cols["gender"]]
    if "age_approx" in cols:
        out["age_approx"] = df[cols["age_approx"]]
    elif "age" in cols:
        out["age_approx"] = df[cols["age"]]
    n_resolved = int((out["filepath"].astype(str) != "").sum())
    logger.info(f"[finalize] DermaCon: {n_resolved}/{len(out)} rows resolved to images")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    logger.info(f"[finalize] wrote DermaCon-IN metadata: {out_csv} ({len(out)} rows)")
    return out_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isic-meta", required=True)
    ap.add_argument("--isic-root", required=True)
    ap.add_argument("--fitz17k-meta", required=True)
    ap.add_argument("--fitz17k-root", required=True)
    ap.add_argument("--dermacon-root", required=True)
    ap.add_argument("--out-master", required=True)
    ap.add_argument("--out-clean", required=True)
    ap.add_argument("--out-splits", required=True)
    ap.add_argument("--external-source", default="MSK")
    ap.add_argument("--k-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dedup-phash", action="store_true")
    ap.add_argument("--qc-blur-min", type=float, default=50.0)
    args = ap.parse_args()

    setup_logging("logs", "finalize_phase3.log", "INFO")

    # 1) DermaCon-IN metadata
    dermacon_root = Path(args.dermacon_root)
    dermacon_meta = Path("data/manifests/dermacon_in.csv")
    _materialize_dermacon_metadata(dermacon_root, dermacon_meta)

    # 2) unify
    bundles = [
        {"meta_csv": args.isic_meta, "image_root": args.isic_root, "source": "ISIC", "id_col": "isic_id"},
        {"meta_csv": args.fitz17k_meta, "image_root": args.fitz17k_root,
         "source": "FITZ17K", "id_col": "md5hash", "label_col": "three_partition_label"},
    ]
    if dermacon_meta.exists():
        # find the actual extracted images dir for shard_fn
        bundles.append({
            "meta_csv": str(dermacon_meta),
            "image_root": str(dermacon_root),
            "source": "INDIA_DERMACON",
            "id_col": "isic_id",
        })

    df = unify_manifests(bundles, dedup_phash=args.dedup_phash)
    Path(args.out_master).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_master, index=False)
    logger.info(f"[finalize] master: {args.out_master} ({len(df)} rows)")

    # 3) QC
    rep = run_qc(df["filepath"].tolist(), blur_min=args.qc_blur_min)
    rep.to_csv(Path(args.out_master).with_name("qc_report.csv"), index=False)
    keep = set(rep.loc[rep["ok"], "path"])
    clean = df[df["filepath"].isin(keep)].reset_index(drop=True)
    clean.to_csv(args.out_clean, index=False)
    logger.info(f"[finalize] clean: {args.out_clean} ({len(clean)} rows)")

    # 4) splits
    cfg = SplitConfig(k_folds=args.k_folds, external_source=args.external_source,
                      seed=args.seed)
    splits = make_splits(clean, cfg)
    assert_no_group_leakage(splits)
    Path(args.out_splits).parent.mkdir(parents=True, exist_ok=True)
    keep_cols = ["filepath", "label_name", "group_id", "split", "fold"]
    if "isic_id" in splits.columns:
        keep_cols = ["isic_id"] + keep_cols
    if "source" in splits.columns:
        keep_cols.append("source")
    splits[keep_cols].to_csv(args.out_splits, index=False)
    logger.info(f"[finalize] splits: {args.out_splits}")
    logger.info("\n" + splits.groupby(["split", "source"]).size().to_string())


if __name__ == "__main__":
    main()
