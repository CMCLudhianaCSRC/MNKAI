"""Build group-aware K-fold splits + external-source holdout for a manifest.

Usage:
    python -m scripts.make_splits \
        --manifest data_legacy/manifest_unified_labeled.csv \
        --out data/splits/isic48k.csv \
        --k-folds 5 --external-source MSK
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from mnkai.data.manifest import LabelNormalizer, normalize_manifest
from mnkai.data.splits import SplitConfig, assert_no_group_leakage, make_splits
from mnkai.utils.logging import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="path to labeled manifest CSV")
    ap.add_argument("--out", required=True, help="output CSV path (will be created)")
    ap.add_argument("--k-folds", type=int, default=5)
    ap.add_argument("--external-source", default="MSK", help="source-name substring for the external holdout (e.g. MSK, BCN, HAM); '' to disable")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group-col", default="group_id")
    args = ap.parse_args()

    setup_logging("logs", "make_splits.log", "INFO")

    raw = pd.read_csv(args.manifest)
    logger.info(f"[splits] loaded {len(raw)} rows from {args.manifest}")
    df = normalize_manifest(raw, LabelNormalizer())

    cfg = SplitConfig(
        k_folds=int(args.k_folds),
        external_source=args.external_source if args.external_source else None,
        group_col=args.group_col,
        seed=int(args.seed),
    )
    out_df = make_splits(df, cfg)
    assert_no_group_leakage(out_df, group_col=args.group_col)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = ["filepath", "label_name", args.group_col, "split", "fold"]
    if "isic_id" in out_df.columns:
        keep_cols = ["isic_id"] + keep_cols
    if "source" in out_df.columns:
        keep_cols.append("source")
    out_df[keep_cols].to_csv(out_path, index=False)
    logger.info(f"[splits] wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
