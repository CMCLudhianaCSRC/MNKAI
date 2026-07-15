"""Filter ISIC metadata down to rows whose diagnosis maps to our 8-class taxonomy.

Reads `data/manifests/isic_combined.csv` (75k rows, all 8 collections) and
writes `data/manifests/isic_filtered.csv` containing only rows whose
diagnosis_3 (or diagnosis fallback) maps to mel/nv/bcc/scc/df/bkl/ak/vasc.

Also caps the dominant `nv` and `bcc` classes to a configurable max so we
don't waste bandwidth downloading 23k nevus images we'd just down-sample.

    python -m scripts.filter_isic_to_8class \
        --in data/manifests/isic_combined.csv \
        --out data/manifests/isic_filtered.csv \
        --cap-nv 8000 --cap-bcc 6000 --cap-mel 8000 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from mnkai.data.manifest import LabelNormalizer
from mnkai.utils.logging import setup_logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-nv", type=int, default=8000)
    ap.add_argument("--cap-bcc", type=int, default=6000)
    ap.add_argument("--cap-mel", type=int, default=8000)
    ap.add_argument("--cap-bkl", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    setup_logging("logs", "filter_isic.log", "INFO")
    df = pd.read_csv(args.inp, low_memory=False)
    logger.info(f"[filter] input: {len(df)} rows, columns={list(df.columns)}")

    norm = LabelNormalizer()
    # Prefer diagnosis_3 then fall back to diagnosis
    label_src = df["diagnosis_3"].fillna("").astype(str)
    if "diagnosis" in df.columns:
        label_src = label_src.where(label_src != "", df["diagnosis"].fillna("").astype(str))
    df["label_name"] = [norm.map(s) for s in label_src.tolist()]
    n_drop = int(df["label_name"].isna().sum())
    df = df[df["label_name"].notna()].reset_index(drop=True)
    logger.info(f"[filter] mapped {len(df)} rows; dropped {n_drop} unmapped")

    counts = df["label_name"].value_counts()
    logger.info(f"[filter] before cap:\n{counts}")

    caps = {"nv": args.cap_nv, "bcc": args.cap_bcc,
            "mel": args.cap_mel, "bkl": args.cap_bkl}
    rng = np.random.default_rng(args.seed)
    keep_idx = []
    for cls, sub in df.groupby("label_name"):
        cap = caps.get(cls, len(sub))
        if len(sub) <= cap:
            keep_idx.extend(sub.index.tolist())
        else:
            picked = rng.choice(sub.index.to_numpy(), size=cap, replace=False)
            keep_idx.extend(picked.tolist())
    df_out = df.loc[sorted(keep_idx)].reset_index(drop=True)
    logger.info(f"[filter] after cap:\n{df_out['label_name'].value_counts()}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out, index=False)
    logger.info(f"[filter] wrote {len(df_out)} rows to {args.out}")


if __name__ == "__main__":
    main()
