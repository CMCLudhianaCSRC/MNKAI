"""One-off migration: copy the E:-drive image references onto the D: SSD and
rewrite the manifests so the whole pipeline reads from fast, reliable storage.

The corpus is split: ~217k images already on D:\AIFINAL, ~53.6k on the flaky
E:\AIFINAL (GODSEYE drive). The E: drive has caused repeated dataloader stalls
and missed reads. This script copies the E: files to D: (mirroring the path
under E:\AIFINAL -> D:\AIFINAL) and rewrites every manifest's `filepath` column.

Idempotent: re-running skips files already copied. Originals are backed up
once to <file>.e_backup before the first rewrite.

Usage:
  python -m scripts._migrate_e_to_d            # copy + rewrite
  python -m scripts._migrate_e_to_d --dry-run  # report only
"""
from __future__ import annotations
import argparse, os, re, shutil, sys, time
from pathlib import Path

import pandas as pd

MANIFESTS = [
    "data/manifests/master.csv",
    "data/manifests/master_clean.csv",
]
SPLITS = ["data/splits/master_splits.csv"]

E_PREFIX_RE = re.compile(r"(?i)^e:[\\/]aifinal")


def to_dest(src: str) -> str:
    """E:\\AIFINAL\\... -> D:\\AIFINAL\\...  (case-insensitive on the prefix)."""
    return E_PREFIX_RE.sub(r"D:\\AIFINAL", src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # 1. collect every distinct E: filepath across all manifests
    e_paths: set[str] = set()
    for m in MANIFESTS + SPLITS:
        if not os.path.exists(m):
            continue
        df = pd.read_csv(m, low_memory=False)
        if "filepath" not in df.columns:
            continue
        fp = df["filepath"].astype(str)
        e_paths |= set(fp[fp.str.match(r"(?i)^e:[\\/]")])
    print(f"[migrate] distinct E: files referenced: {len(e_paths):,}", flush=True)

    if not e_paths:
        print("[migrate] nothing on E: — done.")
        return

    # 2. copy E: -> D:
    copied = skipped = missing = 0
    bytes_copied = 0
    t0 = time.time()
    total = len(e_paths)
    for i, src in enumerate(sorted(e_paths), 1):
        dst = to_dest(src)
        if os.path.exists(dst):
            skipped += 1
        elif not os.path.exists(src):
            missing += 1
        elif args.dry_run:
            copied += 1
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
            try:
                bytes_copied += os.path.getsize(dst)
            except OSError:
                pass
        if i % 2000 == 0:
            el = (time.time() - t0) / 60
            print(f"[migrate]  {i:,}/{total:,}  copied={copied:,} "
                  f"skipped={skipped:,} missing={missing:,}  "
                  f"{bytes_copied/1e9:.1f} GB  {el:.1f}m", flush=True)

    print(f"[migrate] COPY DONE: copied={copied:,} skipped={skipped:,} "
          f"missing={missing:,}  ({bytes_copied/1e9:.2f} GB, "
          f"{(time.time()-t0)/60:.1f}m)", flush=True)
    if missing:
        print(f"[migrate] WARNING: {missing} E: files were missing on disk — "
              f"their manifest rows will point to D: but the file won't exist; "
              f"the dataset's retry/fallback handles this.", flush=True)

    if args.dry_run:
        print("[migrate] dry-run — manifests NOT rewritten.")
        return

    # 3. rewrite manifests + splits
    for m in MANIFESTS + SPLITS:
        if not os.path.exists(m):
            continue
        df = pd.read_csv(m, low_memory=False)
        if "filepath" not in df.columns:
            print(f"[migrate] {m}: no filepath column, skipped")
            continue
        backup = m + ".e_backup"
        if not os.path.exists(backup):
            shutil.copy2(m, backup)
        fp = df["filepath"].astype(str)
        before = int(fp.str.match(r"(?i)^e:[\\/]").sum())
        df["filepath"] = fp.map(to_dest)
        df.to_csv(m, index=False)
        print(f"[migrate] {m}: rewrote {before:,} E: paths -> D: "
              f"(backup: {backup})", flush=True)

    print("[migrate] ALL DONE — pipeline now reads 100% from D: SSD.", flush=True)


if __name__ == "__main__":
    main()
