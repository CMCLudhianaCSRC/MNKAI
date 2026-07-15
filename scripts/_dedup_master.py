"""Perceptual-hash deduplication for the master manifest.

Goals:
 1. Compute pHash for every file in master.csv (resumable cache).
 2. Find near-duplicates (Hamming distance ≤ DUP_THRESHOLD on 64-bit pHash).
 3. Across-source dedup: keep ONE row per duplicate group, prefer:
    higher-quality source (D:ISIC > E:fine_pool > E:lbl > E:shard > E:extra
                            > minority > HAM_supp > MED-NODE > Fitz17k > DermaCon)
 4. Within a duplicate group, prefer rows with non-null fitzpatrick / sex / age.
 5. Emit:
      data/manifests/master_dedup.csv     - master after dedup
      data/manifests/dedup_groups.csv     - which IDs were merged

Usage:
  python scripts/_dedup_master.py
"""
from __future__ import annotations

import sys, time
from pathlib import Path
from collections import defaultdict

import pandas as pd
from PIL import Image
import imagehash
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

MASTER_CSV    = Path("data/manifests/master.csv")
HASH_CACHE    = Path("data/manifests/_phash_cache.csv")
OUT_DEDUP     = Path("data/manifests/master_dedup.csv")
OUT_GROUPS    = Path("data/manifests/dedup_groups.csv")

DUP_THRESHOLD = 4    # Hamming distance ≤ 4 on 64-bit pHash = near-dup
                     # 0 = identical pixels, 4 = JPEG re-encode tolerance

# Source priority (lower = preferred — keep this row in dedup group)
SOURCE_PRIORITY = {
    "ISIC_D":             0,   # filtered, MILK-quality
    "ISIC_E_LBL":         1,
    "ISIC_E_MSK":         1,   # external test domain - keep
    "ISIC_E_POOL":        2,
    "ISIC_E_SHARD":       3,
    "HAM10000_SUPP":      3,
    "ISIC_E_EXTRA":       4,
    "ISIC_MINORITY":      5,
    "MED-NODE":           6,
    "PAD-UFES-20":        4,
    "DERM7PT":            5,
    "BCN20000_KAGGLE":    5,
    "ISIC_2019_KAGGLE":   5,
    "SCIN_P1":            5,
    "ISIC_2024":          6,
    "ISIC_2024_MSK":      1,   # MSK-attributed in 2024 — same external domain
    "FITZ17K":            7,
    "DERMACON_IN":        8,
}


def compute_one(args):
    iid, fp = args
    try:
        img = Image.open(fp)
        if img.mode != "RGB":
            img = img.convert("RGB")
        h = imagehash.phash(img, hash_size=8)   # 64-bit hash
        return (iid, fp, str(h))
    except Exception:
        return (iid, fp, "")


def compute_hashes(rows: list[tuple[str, str]], cache_path: Path) -> dict[str, str]:
    """Compute pHash for each (iid, fp). Returns id→hash. Resumable.

    ThreadPoolExecutor (not Process) — PIL.Image releases the GIL during file
    read/decode so threads parallelise well, and we avoid the heavy Windows
    spawn cost (each ProcessPoolExecutor worker re-imports pandas/PIL/numpy
    which can take 30+ seconds per worker on Windows).

    Cache loading uses vectorised pandas (no iterrows — was the 2nd bottleneck).
    """
    cache: dict[str, str] = {}
    if cache_path.exists():
        t = time.time()
        df = pd.read_csv(cache_path, low_memory=False)
        # Vectorised: build dict from two Series directly
        ids   = df["isic_id"].astype(str).tolist()
        hashs = df["phash"].astype(str).tolist() if "phash" in df.columns else [""] * len(ids)
        cache = dict(zip(ids, hashs))
        print(f"[dedup] cache loaded: {len(cache):,} hashes "
              f"in {time.time()-t:.1f}s", flush=True)

    todo = [(iid, fp) for iid, fp in rows if iid not in cache]
    print(f"[dedup] hashing {len(todo):,} new files (16 threads)", flush=True)
    if not todo:
        return cache

    t0 = time.time()
    completed = 0
    buffer = []

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(compute_one, item): item for item in todo}
        for fut in as_completed(futures):
            try:
                iid, fp, h = fut.result()
            except Exception:
                continue
            cache[iid] = h
            buffer.append({"isic_id": iid, "filepath": fp, "phash": h})
            completed += 1
            if completed % 1000 == 0:
                # checkpoint to cache
                file_existed = cache_path.exists() and cache_path.stat().st_size > 100
                pd.DataFrame(buffer).to_csv(cache_path, mode="a",
                                            header=not file_existed, index=False)
                buffer = []
                rate = completed / (time.time() - t0)
                eta = (len(todo) - completed) / rate / 60 if rate else 0
                print(f"  {completed:,}/{len(todo):,}  {rate:.0f}/s  "
                      f"elapsed {(time.time()-t0)/60:.1f}m  eta {eta:.1f}m", flush=True)

    if buffer:
        file_existed = cache_path.exists() and cache_path.stat().st_size > 100
        pd.DataFrame(buffer).to_csv(cache_path, mode="a",
                                    header=not file_existed, index=False)
    print(f"[dedup] hashing done in {(time.time()-t0)/60:.1f} min", flush=True)
    return cache


def hex_to_int(h: str) -> int | None:
    if not h or len(h) != 16:
        return None
    try:
        return int(h, 16)
    except Exception:
        return None


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def find_dup_groups(id_to_hash: dict[str, str]) -> list[set]:
    """LSH-bucketed near-duplicate groups within DUP_THRESHOLD Hamming distance."""
    # Bucket by 16-bit prefix; within bucket compare pairwise
    print(f"[dedup] grouping near-dups (threshold={DUP_THRESHOLD})...", flush=True)
    buckets: dict[int, list[tuple[str, int]]] = defaultdict(list)
    valid: dict[str, int] = {}
    for iid, h in id_to_hash.items():
        v = hex_to_int(h)
        if v is None:
            continue
        valid[iid] = v
        # 4 buckets of 16 bits each so similar hashes cluster
        for i in range(4):
            shift = i * 16
            key = (v >> shift) & 0xFFFF
            buckets[(i, key)].append((iid, v))

    # Union-find
    parent: dict[str, str] = {iid: iid for iid in valid}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    n_pairs = 0
    for items in buckets.values():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if hamming(items[i][1], items[j][1]) <= DUP_THRESHOLD:
                    union(items[i][0], items[j][0])
                    n_pairs += 1
        if n_pairs > 0 and n_pairs % 100000 == 0:
            print(f"  pairs found so far: {n_pairs:,}", flush=True)

    groups: dict[str, set] = defaultdict(set)
    for iid in valid:
        groups[find(iid)].add(iid)
    return [g for g in groups.values() if len(g) > 1]


def pick_best(group: set, master: pd.DataFrame) -> str:
    """Pick the best row from a duplicate group.

    Order: hard label > weak label, then source priority, then metadata richness.
    """
    sub = master[master["isic_id"].astype(str).isin(group)]
    sub = sub.copy()
    if "weak_label" in sub.columns:
        sub["_weak"] = sub["weak_label"].fillna(False).astype(bool).astype(int)
    else:
        sub["_weak"] = 0
    sub["_prio"] = sub["source"].map(lambda s: SOURCE_PRIORITY.get(str(s), 99))
    sub["_meta"] = (sub["fitzpatrick"].notna().astype(int) +
                    sub["sex"].notna().astype(int) +
                    sub["age_approx"].notna().astype(int))
    # _weak ascending (False=0 first), _prio ascending, _meta descending
    sub = sub.sort_values(["_weak", "_prio", "_meta"], ascending=[True, True, False])
    return str(sub.iloc[0]["isic_id"])


def main():
    print(f"[dedup] loading {MASTER_CSV}", flush=True)
    master = pd.read_csv(MASTER_CSV, low_memory=False)
    master["isic_id"] = master["isic_id"].astype(str)
    print(f"  master: {len(master):,} rows", flush=True)

    # Vectorised filter — skip per-row Path.exists() (was 30 min on USB drive
    # for 271k rows). Trust integrators' upstream existence check.
    fp_series = master["filepath"].astype(str)
    valid = (fp_series.str.len() > 0) & (fp_series != "nan")
    master_valid = master[valid]
    rows: list[tuple[str, str]] = list(zip(
        master_valid["isic_id"].astype(str).tolist(),
        master_valid["filepath"].astype(str).tolist(),
    ))
    print(f"  rows with filepath: {len(rows):,}", flush=True)

    id_to_hash = compute_hashes(rows, HASH_CACHE)
    groups = find_dup_groups(id_to_hash)
    print(f"[dedup] near-duplicate groups: {len(groups):,} "
          f"(total IDs in groups: {sum(len(g) for g in groups):,})", flush=True)

    # Emit dedup_groups.csv
    grp_rows = []
    drops: set[str] = set()
    for gi, g in enumerate(groups):
        keep = pick_best(g, master)
        for iid in g:
            grp_rows.append({"group_id": gi, "isic_id": iid, "kept": iid == keep})
            if iid != keep:
                drops.add(iid)
    pd.DataFrame(grp_rows).to_csv(OUT_GROUPS, index=False)
    print(f"  wrote {OUT_GROUPS} ({len(grp_rows):,} rows)", flush=True)

    # Emit master_dedup.csv
    dedup = master[~master["isic_id"].isin(drops)].copy()
    dedup.to_csv(OUT_DEDUP, index=False)
    print(f"  wrote {OUT_DEDUP} ({len(dedup):,} rows; "
          f"{len(master) - len(dedup):,} duplicates removed)", flush=True)

    # Class distribution before/after
    print("\nBEFORE dedup (8-class):")
    isic = master[master["label_name"].isin(["mel","nv","bcc","scc","df","bkl","ak","vasc"])]
    print(isic["label_name"].value_counts().to_string())
    print("\nAFTER dedup (8-class):")
    isic2 = dedup[dedup["label_name"].isin(["mel","nv","bcc","scc","df","bkl","ak","vasc"])]
    print(isic2["label_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
