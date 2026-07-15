"""Run image QC over a manifest. Drops corrupt/blank/blurry/over-exposed rows.

    python -m scripts.run_qc --manifest data/manifests/master.csv \
        --out-report data/manifests/qc_report.csv \
        --out-clean data/manifests/master_clean.csv
"""
from __future__ import annotations

import argparse

import pandas as pd
from loguru import logger

from mnkai.data.qc import run_qc
from mnkai.utils.logging import setup_logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-report", required=True)
    ap.add_argument("--out-clean", required=True)
    ap.add_argument("--blur-min", type=float, default=50.0)
    ap.add_argument("--dark-max", type=float, default=20.0)
    ap.add_argument("--bright-min", type=float, default=235.0)
    ap.add_argument("--dedup-phash", action="store_true")
    args = ap.parse_args()

    setup_logging("logs", "run_qc.log", "INFO")
    df = pd.read_csv(args.manifest)
    logger.info(f"[qc] {len(df)} rows in manifest")
    rep = run_qc(df["filepath"].tolist(),
                 blur_min=args.blur_min, dark_max=args.dark_max, bright_min=args.bright_min)
    rep.to_csv(args.out_report, index=False)
    keep = rep.loc[rep["ok"], "path"].tolist()
    keep_set = set(keep)
    clean = df[df["filepath"].isin(keep_set)].reset_index(drop=True)
    if args.dedup_phash and "phash" in rep.columns:
        ph = rep.set_index("path")["phash"]
        clean["phash"] = clean["filepath"].map(ph)
        before = len(clean)
        clean = clean.drop_duplicates(subset="phash", keep="first").reset_index(drop=True)
        logger.info(f"[qc] phash dedup: {before} -> {len(clean)}")
        clean = clean.drop(columns=["phash"], errors="ignore")
    clean.to_csv(args.out_clean, index=False)
    logger.info(f"[qc] kept {len(clean)} / {len(df)} -> {args.out_clean}")


if __name__ == "__main__":
    main()
