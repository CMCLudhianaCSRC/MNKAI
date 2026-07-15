"""Download DermaCon-IN from Harvard Dataverse (DOI 10.7910/DVN/W7OUZM).

    python -m scripts.download_dermacon_in --out-dir D:\AIFINAL\images\dermacon_in
"""
from __future__ import annotations

import argparse

from loguru import logger

from mnkai.data.download.dermacon_in import DOI, download_dataset
from mnkai.utils.logging import setup_logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--doi", default=DOI)
    ap.add_argument("--no-unzip", action="store_true")
    ap.add_argument("--exclude-ext", default="",
                    help="comma-separated extensions to skip, e.g. '.pth,.bin'")
    ap.add_argument("--include-only-ext", default="",
                    help="comma-separated extensions to keep, all others skipped")
    args = ap.parse_args()

    setup_logging("logs", "download_dermacon_in.log", "INFO")
    excl = tuple(e.strip() for e in args.exclude_ext.split(",") if e.strip())
    incl = tuple(e.strip() for e in args.include_only_ext.split(",") if e.strip())
    stats = download_dataset(args.out_dir, doi=args.doi, unzip=not args.no_unzip,
                             exclude_exts=excl, include_only_exts=incl)
    logger.info(f"[dermacon] done: {stats}")


if __name__ == "__main__":
    main()
