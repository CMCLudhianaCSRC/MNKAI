"""Fetch ISIC API metadata for 51k E: drive IDs that lack labels.

CRITICAL FIX (Apr 2026):
  Old version passed isic_id=id1,id2,... in query string. The ISIC API IGNORES
  this filter and just returns the first N images by default order. The result
  was 51,700 rows that all referenced the same 100 IDs (ISIC_0000000-099) on
  repeat. The bad CSV was useless.

  New version: individual GET /api/v2/images/{isic_id}/ requests with a
  ThreadPoolExecutor (30 workers). Resumable via on-disk checkpointing every
  500 IDs.

Run: python scripts/_fetch_isic_labels.py
"""
import json, time, csv, requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

IDS_FILE = Path("data/manifests/e_ids_need_labels.json")
OUT_CSV  = Path("data/manifests/isic_extra_labels.csv")
WORKERS  = 30
TIMEOUT  = 20
RETRIES  = 2

COLS = [
    "isic_id", "diagnosis_3", "diagnosis_1", "benign_malignant", "sex",
    "age_approx", "anatom_site_general", "melanocytic", "attribution", "image_type",
]


def fetch_one(isic_id: str) -> dict:
    """Fetch single image metadata."""
    url = f"https://api.isic-archive.com/api/v2/images/{isic_id}/"
    for attempt in range(RETRIES + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                d    = r.json()
                clin = d.get("metadata", {}).get("clinical", {})
                acq  = d.get("metadata", {}).get("acquisition", {})
                return {
                    "isic_id":             d.get("isic_id", isic_id),
                    "diagnosis_3":         clin.get("diagnosis_3", "") or "",
                    "diagnosis_1":         clin.get("diagnosis_1", "") or "",
                    "benign_malignant":    clin.get("benign_malignant", "") or "",
                    "sex":                 clin.get("sex", "") or "",
                    "age_approx":          clin.get("age_approx", "") or "",
                    "anatom_site_general": clin.get("anatom_site_general", "") or "",
                    "melanocytic":         clin.get("melanocytic", "") or "",
                    "attribution":         d.get("attribution", "") or "",
                    "image_type":          acq.get("image_type", "") or "",
                }
            elif r.status_code == 404:
                return {"isic_id": isic_id, **{c: "" for c in COLS if c != "isic_id"}}
            elif r.status_code in (429, 502, 503, 504) and attempt < RETRIES:
                time.sleep(1.5 * (attempt + 1))
                continue
        except Exception:
            if attempt < RETRIES:
                time.sleep(1.0 * (attempt + 1))
                continue
    return {"isic_id": isic_id, **{c: "" for c in COLS if c != "isic_id"}}


def load_done_ids() -> set:
    if not OUT_CSV.exists() or OUT_CSV.stat().st_size < 100:
        return set()
    done = set()
    with open(OUT_CSV, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            iid = r.get("isic_id", "").strip()
            if iid:
                done.add(iid)
    return done


def append_rows(rows):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = (not OUT_CSV.exists()) or OUT_CSV.stat().st_size < 100
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})


def main():
    with open(IDS_FILE) as f:
        all_ids = json.load(f)
    print(f"Total IDs: {len(all_ids):,}", flush=True)

    done = load_done_ids()
    remaining = [iid for iid in all_ids if iid not in done]
    print(f"Already fetched: {len(done):,}  Remaining: {len(remaining):,}", flush=True)
    if not remaining:
        return

    t0 = time.time()
    completed = 0
    buffer = []
    BUFFER_SIZE = 500
    LOG_EVERY   = 500

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch_one, iid): iid for iid in remaining}
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except Exception:
                row = {"isic_id": futures[fut], **{c: "" for c in COLS if c != "isic_id"}}
            buffer.append(row)
            completed += 1

            if len(buffer) >= BUFFER_SIZE:
                append_rows(buffer)
                buffer = []

            if completed % LOG_EVERY == 0:
                elapsed = time.time() - t0
                rate    = completed / elapsed if elapsed > 0 else 0
                eta     = (len(remaining) - completed) / rate if rate > 0 else 0
                print(f"  [{completed:,}/{len(remaining):,}] "
                      f"rate={rate:.1f}/s  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m",
                      flush=True)

    if buffer:
        append_rows(buffer)

    print(f"\nFetch complete in {(time.time()-t0)/60:.1f} minutes.", flush=True)
    print(f"Saved: {OUT_CSV}", flush=True)

    import pandas as pd
    df = pd.read_csv(OUT_CSV, low_memory=False)
    df = df[df["isic_id"].notna()]
    labeled = df[df["diagnosis_3"].astype(str).str.len() > 0]
    print(f"Total rows: {len(df):,}  with labels: {len(labeled):,}  "
          f"unique IDs: {df['isic_id'].nunique():,}")
    print("Top 15 diagnoses:")
    print(labeled["diagnosis_3"].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
