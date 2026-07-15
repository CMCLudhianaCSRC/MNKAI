"""One-shot status report for Phase 3 downloads."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\AIFINAL\images")


def _fmt_size(n_bytes: int) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {u}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def _du(p: Path) -> int:
    if not p.exists():
        return 0
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def _count_files(p: Path, suffix: str | None = None) -> int:
    if not p.exists():
        return 0
    if suffix is None:
        return sum(1 for _ in p.rglob("*") if _.is_file())
    return sum(1 for _ in p.rglob(f"*{suffix}") if _.is_file())


def main():
    print("===== Phase 3 download status =====")
    # ISIC
    isic_root = ROOT / "isic"
    n_isic = _count_files(isic_root, ".jpg")
    sz_isic = _du(isic_root)
    print(f"ISIC images:       {n_isic:>7,d}   {_fmt_size(sz_isic):>10s}  (target: ~75,338)")
    for isic_meta in (Path("data/manifests/isic_filtered.csv"),
                      Path("data/manifests/isic_combined.csv")):
        if isic_meta.exists():
            try:
                df = pd.read_csv(isic_meta, low_memory=False)
                print(f"  meta {isic_meta.name}: {len(df):>7,d} rows")
                if "fitzpatrick" in df.columns:
                    fst_present = int(df["fitzpatrick"].notna().sum())
                    print(f"    with FST label: {fst_present:>7,d} ({100*fst_present/max(1,len(df)):.1f}%)")
                if "label_name" in df.columns:
                    print("    class counts:", df["label_name"].value_counts().to_dict())
            except Exception as e:
                print(f"  meta {isic_meta.name}: error: {e}")

    # Fitz17k
    fitz_root = ROOT / "fitz17k"
    n_fitz = _count_files(fitz_root, ".jpg")
    sz_fitz = _du(fitz_root)
    print(f"Fitzpatrick17k:    {n_fitz:>7,d}   {_fmt_size(sz_fitz):>10s}")

    # DermaCon
    derm_root = ROOT / "dermacon_in"
    n_derm = _count_files(derm_root)
    sz_derm = _du(derm_root)
    derm_meta_path = next(derm_root.rglob("Skin_Metadata*"), None) if derm_root.exists() else None
    print(f"DermaCon-IN:       {n_derm:>7,d} files {_fmt_size(sz_derm):>10s}  meta={derm_meta_path}")

    # Disk
    import shutil
    total, used, free = shutil.disk_usage("D:\\")
    print(f"D:\\ free:          {_fmt_size(free)} of {_fmt_size(total)}")


if __name__ == "__main__":
    main()
