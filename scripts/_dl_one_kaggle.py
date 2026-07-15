"""Download a single Kaggle dataset by slug. Used to parallelize downloads.

Usage: python scripts/_dl_one_kaggle.py <slug> <dest_dir>
"""
import os, sys, time
from pathlib import Path

if "KAGGLE_API_TOKEN" not in os.environ:
    print("ERROR: KAGGLE_API_TOKEN not set", file=sys.stderr); sys.exit(1)

slug = sys.argv[1]
dest = Path(sys.argv[2])
dest.mkdir(parents=True, exist_ok=True)

import kaggle
kaggle.api.authenticate()

# Skip if already extracted with images
existing = sum(1 for p in dest.rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png'))
if existing > 100:
    print(f"  {slug}: {existing:,} images already extracted -> skip", flush=True)
    sys.exit(0)

t0 = time.time()
print(f"Downloading {slug} -> {dest} ...", flush=True)
kaggle.api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
print(f"\nDONE {slug} in {(time.time()-t0)/60:.1f}m", flush=True)
