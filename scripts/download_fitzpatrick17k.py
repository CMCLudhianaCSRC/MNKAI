"""Download Fitzpatrick17k images + metadata.

Default: use Hugging Face mirror `ZYXue/Fitzpatrick_17k` (Parquet, ~88 MB).
Fallback (`--legacy`): the original GitHub CSV + atlas URLs (DEAD as of 2025).

    python -m scripts.download_fitzpatrick17k \
        --meta-out data/manifests/fitzpatrick17k.csv \
        --image-root D:\AIFINAL\images\fitz17k
"""
from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from mnkai.data.download.fitzpatrick17k import (
    download_dataset,
    download_dataset_hf,
    fetch_metadata,
)
from mnkai.utils.logging import setup_logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-out", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--meta-only", action="store_true")
    ap.add_argument("--legacy", action="store_true",
                    help="use the original GitHub-CSV-+-atlas-URLs path (now broken)")
    ap.add_argument("--hf-repo", default="ZYXue/Fitzpatrick_17k")
    args = ap.parse_args()

    setup_logging("logs", "download_fitz17k.log", "INFO")
    if args.legacy:
        fetch_metadata(args.meta_out)
        if args.meta_only:
            return
        stats = download_dataset(args.meta_out, Path(args.image_root),
                                 concurrency=args.concurrency)
    else:
        stats = download_dataset_hf(Path(args.image_root), meta_out=args.meta_out,
                                    repo=args.hf_repo)
    logger.info(f"[fitz17k] done: {stats}")


if __name__ == "__main__":
    main()
