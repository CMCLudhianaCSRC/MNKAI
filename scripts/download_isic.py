"""Download metadata + images for one or more ISIC collections.

Examples:
    # discover collections matching name patterns
    python -m scripts.download_isic --list "BCN|HAM|MILK|PROVe|BALD|2019 Challenge|2020 Challenge"

    # fetch metadata for ISIC 2019 Challenge - Training and BCN20000, then images
    python -m scripts.download_isic \
        --name "ISIC 2019 Challenge - Training,BCN_20000" \
        --meta-out data/manifests/isic_combined.csv \
        --image-root D:\AIFINAL\images\isic_combined --concurrency 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from mnkai.data.download.isic import (
    download_ids,
    fetch_metadata_for,
    find_collections,
    list_collections,
)
from mnkai.utils.logging import setup_logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default=None, help="regex of collection names to print and exit")
    ap.add_argument("--name", default=None, help="comma-separated EXACT collection names to download")
    ap.add_argument("--patterns", default=None,
                    help="comma-separated regex patterns; matches across collection names")
    ap.add_argument("--meta-out", default=None)
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="cap rows per collection (debug)")
    ap.add_argument("--meta-only", action="store_true")
    ap.add_argument("--skip-meta-if-exists", action="store_true",
                    help="reuse the existing meta CSV instead of re-fetching (idempotent restart)")
    args = ap.parse_args()

    setup_logging("logs", "download_isic.log", "INFO")

    if args.list:
        results = find_collections([args.list])[args.list]
        for c in results:
            flag = "(public)" if c["public"] else "(private)"
            print(f"  id={c['id']:<6} {flag} {c['name']}")
        return

    if not args.name and not args.patterns:
        raise SystemExit("must pass --name or --patterns or --list")

    coll_ids: list[int] = []
    name_map = {c["name"]: c["id"] for c in list_collections()}
    if args.name:
        for n in args.name.split(","):
            n = n.strip()
            if n in name_map:
                coll_ids.append(name_map[n])
            else:
                logger.warning(f"[isic] no exact match for '{n}'; try --patterns")
    if args.patterns:
        m = find_collections([p.strip() for p in args.patterns.split(",")])
        for plist in m.values():
            for c in plist:
                if c["id"] not in coll_ids:
                    coll_ids.append(c["id"])

    if not coll_ids:
        raise SystemExit("no collections resolved")
    logger.info(f"[isic] resolved {len(coll_ids)} collection ids: {coll_ids}")

    if args.meta_out is None:
        raise SystemExit("--meta-out required")
    from pathlib import Path as _P
    meta_path = _P(args.meta_out)
    have_meta = meta_path.exists() and meta_path.stat().st_size > 1024
    if args.skip_meta_if_exists and have_meta:
        logger.info(f"[isic] reusing existing metadata at {meta_path}")
    else:
        fetch_metadata_for(coll_ids, args.meta_out, limit=args.limit)
    if args.meta_only:
        return

    if args.image_root is None:
        raise SystemExit("--image-root required for image download")
    df = pd.read_csv(args.meta_out, low_memory=False)
    ids = [str(x) for x in df["isic_id"].dropna().tolist()]
    logger.info(f"[isic] downloading {len(ids)} images to {args.image_root}")
    stats = download_ids(ids, Path(args.image_root), concurrency=args.concurrency)
    logger.info(f"[isic] done: {stats}")


if __name__ == "__main__":
    main()
