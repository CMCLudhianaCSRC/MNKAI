"""Download a single Kaggle competition by slug.

Usage: python scripts/_dl_one_competition.py <competition_slug> <dest_dir>

Requires user to have accepted competition rules on the Kaggle web UI.
"""
import os, sys, time, zipfile
from pathlib import Path

if "KAGGLE_API_TOKEN" not in os.environ:
    print("ERROR: KAGGLE_API_TOKEN not set", file=sys.stderr); sys.exit(1)

slug = sys.argv[1]
dest = Path(sys.argv[2])
dest.mkdir(parents=True, exist_ok=True)

import kaggle
kaggle.api.authenticate()

existing = sum(1 for p in dest.rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png'))
if existing > 100:
    print(f"  {slug}: {existing:,} images already extracted -> skip", flush=True)
    sys.exit(0)

t0 = time.time()
print(f"Downloading competition {slug} -> {dest} ...", flush=True)
try:
    kaggle.api.competition_download_files(slug, path=str(dest), quiet=False)
except Exception as e:
    print(f"DOWNLOAD FAILED: {e}", flush=True)
    print(f"  Hint: visit https://www.kaggle.com/competitions/{slug}/rules and click Accept.")
    sys.exit(2)

# Extract any zip files
for zip_path in dest.glob("*.zip"):
    print(f"  Extracting {zip_path.name} ({zip_path.stat().st_size:,} bytes)", flush=True)
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
    except Exception as e:
        print(f"  extract failed: {e}", flush=True)

print(f"\nDONE {slug} in {(time.time()-t0)/60:.1f}m", flush=True)
