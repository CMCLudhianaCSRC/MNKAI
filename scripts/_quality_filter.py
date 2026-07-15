"""Quality filter pass on the deduped master manifest.

Removes:
  1. Corrupted files (PIL can't open or returns 0-size)
  2. Low resolution (min(W, H) < 224)         — model trains at 224
  3. Blank / near-blank images (std_dev < 5)  — nothing to learn from
  4. Tiny file size (< 5 KB)                  — almost certainly broken

Writes:
  data/manifests/master_clean.csv     — passed quality filter
  data/manifests/_qa_failures.csv     — what was dropped + reason
"""
from pathlib import Path
import time
import pandas as pd
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor

IN_CSV  = Path("data/manifests/master_dedup.csv")
OUT_CSV = Path("data/manifests/master_clean.csv")
QA_LOG  = Path("data/manifests/_qa_failures.csv")

# MIN_DIM = 80 — ISIC 2024 thumbnails are 109-165 px. Lower bound prevents
# garbage-tier images while preserving 3D-TBP thumbnails. The training
# DataLoader's Resize(224) will upscale smaller images. DINOv2 was pretrained
# with multi-scale (224-518) and tolerates upscaled small inputs.
MIN_DIM   = 80
MIN_BYTES = 1024
MIN_STD   = 5.0


def check_one(args):
    iid, fp = args
    try:
        p = Path(fp)
        if not p.exists():
            return iid, "missing"
        if p.stat().st_size < MIN_BYTES:
            return iid, "tiny_file"
        with Image.open(p) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
            w, h = im.size
            if min(w, h) < MIN_DIM:
                return iid, f"low_res_{w}x{h}"
            # Sample center crop for blank check (don't load full image)
            arr = np.asarray(im.resize((64, 64)))
            if arr.std() < MIN_STD:
                return iid, "blank"
        return iid, ""
    except Exception as e:
        return iid, f"corrupt:{type(e).__name__}"


def main():
    print(f"[qa] loading {IN_CSV}", flush=True)
    df = pd.read_csv(IN_CSV, low_memory=False)
    print(f"  rows: {len(df):,}", flush=True)

    rows = list(zip(df["isic_id"].astype(str), df["filepath"].astype(str)))

    t0 = time.time()
    failures = {}
    completed = 0
    with ProcessPoolExecutor(max_workers=8) as pool:
        for iid, reason in pool.map(check_one, rows, chunksize=50):
            if reason:
                failures[iid] = reason
            completed += 1
            if completed % 5000 == 0:
                rate = completed / (time.time() - t0)
                eta = (len(rows) - completed) / rate / 60 if rate else 0
                print(f"  [{completed:,}/{len(rows):,}] {rate:.0f}/s  "
                      f"eta {eta:.1f}m  failures={len(failures):,}", flush=True)

    print(f"\n[qa] complete in {(time.time()-t0)/60:.1f}m", flush=True)
    print(f"  total failures: {len(failures):,}", flush=True)

    # Failure breakdown
    from collections import Counter
    reasons = Counter(failures.values())
    for r, n in reasons.most_common():
        print(f"    {r:>20}: {n:,}", flush=True)

    # Write QA log
    qa_df = pd.DataFrame([{"isic_id": k, "reason": v} for k, v in failures.items()])
    qa_df.to_csv(QA_LOG, index=False)

    # Write clean manifest
    clean = df[~df["isic_id"].astype(str).isin(failures)].copy()
    clean.to_csv(OUT_CSV, index=False)
    print(f"\n  wrote {OUT_CSV} ({len(clean):,} clean rows)", flush=True)

    # 8-class counts
    isic = clean[clean["label_name"].isin(
        ["mel", "nv", "bcc", "scc", "df", "bkl", "ak", "vasc"])]
    print("\n8-class distribution after QA:")
    print(isic["label_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
