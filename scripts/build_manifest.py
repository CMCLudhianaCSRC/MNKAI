"""Unify multiple per-source manifests into one master manifest.

Usage (after Phase 3 downloads):
    python -m scripts.build_manifest \
        --bundle "name=isic_combined,csv=data/manifests/isic_combined.csv,root=D:\AIFINAL\images\isic_combined,source=ISIC" \
        --bundle "name=fitz17k,csv=data/manifests/fitzpatrick17k.csv,root=D:\AIFINAL\images\fitz17k,source=FITZ17K,id_col=md5hash,label_col=three_partition_label" \
        --bundle "name=dermacon_in,csv=data/manifests/dermacon_in.csv,root=D:\AIFINAL\images\dermacon_in,source=INDIA_DERMACON" \
        --out data/manifests/master.csv \
        --dedup-phash
"""
from __future__ import annotations

import argparse

from mnkai.data.unify import unify_manifests
from mnkai.utils.logging import setup_logging


def parse_bundle(s: str) -> dict:
    out = {}
    for kv in s.split(","):
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        out[k.strip()] = v.strip()
    if "csv" not in out or "root" not in out:
        raise ValueError(f"--bundle must include csv= and root=: got {s}")
    return {
        "meta_csv": out["csv"],
        "image_root": out["root"],
        "source": out.get("source"),
        "id_col": out.get("id_col", "isic_id"),
        "label_col": out.get("label_col"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", action="append", required=True,
                    help="repeat per source: 'csv=...,root=...,source=...,id_col=isic_id,label_col=...'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dedup-phash", action="store_true")
    args = ap.parse_args()

    setup_logging("logs", "build_manifest.log", "INFO")
    bundles = [parse_bundle(b) for b in args.bundle]
    df = unify_manifests(bundles, dedup_phash=args.dedup_phash)
    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
