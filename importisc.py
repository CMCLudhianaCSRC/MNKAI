import os
import sys
import csv
import json
import time
import math
import random
import ctypes
import ctypes.wintypes
from pathlib import Path
from typing import Dict, Any, Optional

import requests
from PIL import Image
from io import BytesIO

# ==============================
# CONFIG (edit if you want)
# ==============================
API_BASE = "https://api.isic-archive.com/api/v2"
ROOT_DIR = Path(r"D:\AIFINAL")           # <— target root
IMG_DIR = ROOT_DIR / "images"            # where JPGs go
LOG_DIR = ROOT_DIR / "logs"              # logs + state
MANIFEST = ROOT_DIR / "manifest.csv"     # csv manifest
STATE = LOG_DIR / "state.json"
ERROR_LOG = LOG_DIR / "errors.log"

BATCH_LIMIT = 100            # API page size (max 100)
FILTER_DERMOSCOPIC = True    # only dermoscopic images
MAX_IMAGES = None            # None = all available; or set e.g. 500_000
REQUEST_TIMEOUT = 60         # seconds per request
BASE_SLEEP = 0.5             # polite delay per image (seconds)
PAGE_SLEEP = 1.0             # polite delay per page (seconds)

# Retry/backoff
MAX_RETRIES = 6
BACKOFF_BASE = 0.75          # exponential backoff seconds
JITTER = (0.0, 0.5)          # random jitter added to backoff

# ==============================
# Prevent Windows Sleep Helpers
# ==============================
ES_CONTINUOUS        = 0x80000000
ES_SYSTEM_REQUIRED   = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040  # ok for media/background tasks

def prevent_sleep_start():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
        )
    except Exception:
        pass  # non-Windows or failed—ignore

def prevent_sleep_stop():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:
        pass

# ==============================
# Utilities
# ==============================
def ensure_dirs():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST.exists():
        with MANIFEST.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["isic_id", "diagnosis", "benign_malignant",
                             "modality", "source", "filepath"])

def load_state() -> Dict[str, Any]:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"offset": 0, "downloaded": 0, "skipped": 0, "errors": 0}

def save_state(state: Dict[str, Any]):
    tmp = STATE.with_suffix(".json.part")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE)

def append_manifest(row):
    with MANIFEST.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

def log_error(msg: str):
    with ERROR_LOG.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")

def existing_ids_from_manifest() -> set:
    ids = set()
    if MANIFEST.exists():
        with MANIFEST.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                ids.add(r["isic_id"])
    return ids

def polite_sleep(sec):
    time.sleep(sec)

# Robust GET with retries/backoff
def robust_get(session: requests.Session, url: str, params: Optional[dict] = None) -> requests.Response:
    last_exc = None
    for i in range(MAX_RETRIES):
        try:
            r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            # Handle transient statuses explicitly
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"Transient HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            backoff = BACKOFF_BASE * (2 ** i) + random.uniform(*JITTER)
            time.sleep(backoff)
    # one last try (will raise if fails)
    if last_exc:
        raise last_exc

def verify_image_bytes(b: bytes) -> bool:
    try:
        Image.open(BytesIO(b)).verify()
        return True
    except Exception:
        return False

# ==============================
# Main Download Loop
# ==============================
def main():
    ensure_dirs()
    state = load_state()
    offset = int(state.get("offset", 0))
    downloaded = int(state.get("downloaded", 0))
    skipped = int(state.get("skipped", 0))
    errors = int(state.get("errors", 0))

    # Build a quick set of already-recorded images to skip dup work
    already = existing_ids_from_manifest()

    params = {"limit": BATCH_LIMIT, "offset": offset}
    if FILTER_DERMOSCOPIC:
        # Using acquisition.image_type filter (API v2 metadata shape)
        params["metadata.acquisition.image_type"] = "dermoscopic"

    session = requests.Session()

    print(f"Starting at offset={offset} | downloaded={downloaded} | skipped={skipped} | errors={errors}")
    print(f"Saving to: {IMG_DIR}")
    print("Sleep prevention: ON")
    prevent_sleep_start()
    try:
        while True:
            # Respect MAX_IMAGES if set
            if MAX_IMAGES is not None and downloaded >= MAX_IMAGES:
                print(f"Reached MAX_IMAGES={MAX_IMAGES}. Stopping.")
                break

            # Pull one page
            try:
                resp = robust_get(session, f"{API_BASE}/images", params=params)
                data = resp.json()
            except Exception as e:
                log_error(f"[PAGE] offset={params.get('offset')} error={repr(e)}")
                # back off and try next loop iteration (will retry)
                time.sleep(3)
                continue

            # v2 API returns an object with a 'results' list
            results = data.get("results", []) if isinstance(data, dict) else data
            if not results:
                print("No more images returned by API. Done.")
                break

            page_count = 0

            for img in results:
                isic_id = img.get("isic_id")
                if not isic_id:
                    continue

                # Skip if known
                if isic_id in already:
                    skipped += 1
                    page_count += 1
                    continue

                meta = img.get("metadata", {}) or {}
                clinical = meta.get("clinical", {}) or {}
                acq = meta.get("acquisition", {}) or {}

                diagnosis = clinical.get("diagnosis", "") or ""
                benign_malignant = clinical.get("benign_malignant", "") or ""
                modality = acq.get("image_type", "") or ""
                source = img.get("attribution", "") or ""

                # final path (partition into subdirs to avoid giant directories)
                subdir = isic_id.split("_")[1][:3] if "_" in isic_id else isic_id[:3]
                target_dir = IMG_DIR / subdir
                target_dir.mkdir(parents=True, exist_ok=True)

                jpg_path = target_dir / f"{isic_id}.jpg"
                part_path = jpg_path.with_suffix(".jpg.part")

                # If file exists (even without manifest), skip and add to manifest
                if jpg_path.exists():
                    append_manifest([isic_id, diagnosis, benign_malignant, modality, source, str(jpg_path)])
                    already.add(isic_id)
                    downloaded += 1
                    page_count += 1
                    # polite tiny sleep
                    polite_sleep(0.02)
                    # State checkpoint occasionally
                    if downloaded % 200 == 0:
                        state.update({"offset": offset, "downloaded": downloaded, "skipped": skipped, "errors": errors})
                        save_state(state)
                    continue

                # Download the image bytes robustly
                try:
                    files = img.get("files", {}) or {}
                    full = files.get("full", {}) or {}
                    img_url = full.get("url")
                    if not img_url:
                        img_url = f"{API_BASE}/images/{isic_id}/download"
                    rimg = robust_get(session, img_url)
                    b = rimg.content
                    if not verify_image_bytes(b):
                        raise ValueError("PIL.verify failed")
                    # write .part then rename
                    with part_path.open("wb") as f:
                        f.write(b)
                    part_path.replace(jpg_path)

                    # add to manifest & cache set
                    append_manifest([isic_id, diagnosis, benign_malignant, modality, source, str(jpg_path)])
                    already.add(isic_id)
                    downloaded += 1
                except Exception as e:
                    errors += 1
                    log_error(f"[IMG] {isic_id} error={repr(e)}")
                    # remove corrupt partial
                    try:
                        if part_path.exists():
                            part_path.unlink()
                    except Exception:
                        pass

                page_count += 1

                # per-image polite delay
                polite_sleep(BASE_SLEEP)

                # Periodic state save
                if (downloaded + skipped) % 200 == 0:
                    state.update({"offset": offset, "downloaded": downloaded, "skipped": skipped, "errors": errors})
                    save_state(state)

                # Hard stop if MAX_IMAGES reached mid-page
                if MAX_IMAGES is not None and downloaded >= MAX_IMAGES:
                    break

            # Next page
            offset += BATCH_LIMIT
            params["offset"] = offset

            # save page-level progress
            state.update({"offset": offset, "downloaded": downloaded, "skipped": skipped, "errors": errors})
            save_state(state)

            # polite delay per page
            polite_sleep(PAGE_SLEEP)

            # Safety: if API starts returning empty repeatedly (should break earlier)
            if page_count == 0:
                print("Page empty after filtering—continuing.")
                continue

            # Console heartbeat
            print(f"Offset {offset}: downloaded={downloaded}, skipped={skipped}, errors={errors}")

        print(f"✅ Finished. downloaded={downloaded}, skipped={skipped}, errors={errors}")

    except KeyboardInterrupt:
        print("\nInterrupted by user. State saved.")
    finally:
        prevent_sleep_stop()
        print("Sleep prevention: OFF")

if __name__ == "__main__":
    # Basic dependency check
    try:
        import requests  # noqa
        from PIL import Image  # noqa
    except Exception as e:
        print("Please install dependencies first: pip install requests pillow")
        sys.exit(1)

    main()
