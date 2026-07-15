"""Download more images for underrepresented 8-class categories from ISIC API.

REAL ISIC availability (confirmed via search API + image detail inspection, Apr 2026):
  ak    : "Solar or actinic keratosis"=2,838 + Keratoacanthoma=324  = 3,162 total
           we have 1,458 → ~1,700 more downloadable  ← MAIN TARGET
  scc   : "Squamous cell carcinoma, NOS"=1,717 + "in situ"=662      = 2,379 total
           we have 1,619 → ~760 more downloadable
  df    : Dermatofibroma=773 total → we have 727 → only ~46 more
  vasc  : Hemangioma=298 + Angiokeratoma=79 + "Pyogenic granuloma"=23 = 400 total
           we have 318 → ~82 more
  bcc   : "Basal cell carcinoma"=12,314 → we have 9,455 (well covered)

CRITICAL FIX (Apr 2026):
  - Download URL: use S3 direct  https://isic-archive.s3.amazonaws.com/images/{id}.jpg
    NOT the old /api/v2/images/{id}/files/full.jpg  (that returned 404)
  - Multi-word queries MUST use double-quoted Elasticsearch term syntax:
    query='diagnosis_3:"Solar or actinic keratosis"'
    Single-word queries work without quotes: query='diagnosis_3:Keratoacanthoma'

Run: python scripts/_dl_minority_classes.py
"""
import sys, time, requests, pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── config ────────────────────────────────────────────────────────────────────
OUT_DIR      = Path(r"D:\AIFINAL\images\isic_minority")
MANIFEST_OUT = Path("data/manifests/isic_minority.csv")
PAGE_SIZE    = 100
WORKERS      = 30    # parallel download threads (S3 handles this fine)
DELAY        = 0.15  # seconds between API list calls

# ISIC endpoints
SEARCH_URL = "https://api.isic-archive.com/api/v2/images/search/"
# CORRECT S3 download URL (confirmed working Apr 2026)
S3_URL     = "https://isic-archive.s3.amazonaws.com/images/{isic_id}.jpg"
S3_THUMB   = "https://isic-archive.s3.amazonaws.com/thumbnails/{isic_id}_thumbnail.jpg"

# CONFIRMED exact diagnosis_3 values (from ISIC API inspection Apr 2026)
# Multi-word values require double-quote wrapping in search query
TARGETS = {
    "ak": {
        # Total available: ~3,162 (2,838 + 324) | we have 1,458 | need ~1,700
        "queries": [
            'diagnosis_3:"Solar or actinic keratosis"',  # 2,838 available
            "diagnosis_3:Keratoacanthoma",               # 324 available
        ],
        "want": 3200,   # get all we can
    },
    "scc": {
        # Total available: ~2,379 | we have 1,619 | need ~760
        "queries": [
            'diagnosis_3:"Squamous cell carcinoma, NOS"',     # 1,717 available
            'diagnosis_3:"Squamous cell carcinoma in situ"',  # 662 available
        ],
        "want": 2500,
    },
    "df": {
        # Only 773 total in ISIC — near ceiling
        "queries": [
            "diagnosis_3:Dermatofibroma",  # 773 available
        ],
        "want": 800,    # realistic ceiling
    },
    "vasc": {
        # ~400 total across all vascular subtypes
        "queries": [
            "diagnosis_3:Hemangioma",              # 298 available
            "diagnosis_3:Angiokeratoma",           # 79 available
            'diagnosis_3:"Pyogenic granuloma"',    # 23 available
        ],
        "want": 450,
    },
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)


def query_isic_search(query: str, limit: int = 5000) -> list[dict]:
    """Use ISIC /images/search/ endpoint (confirmed working) to get image IDs."""
    results = []
    offset  = 0
    while len(results) < limit:
        params = {"query": query, "limit": PAGE_SIZE, "offset": offset}
        try:
            r = requests.get(SEARCH_URL, params=params, timeout=30)
            if r.status_code != 200:
                print(f"  Search HTTP {r.status_code} for: {query}")
                break
            data  = r.json()
            items = data.get("results", [])
            if not items:
                break
            total = data.get("count", 0)
            for item in items:
                clin = item.get("metadata", {}).get("clinical", {})
                results.append({
                    "isic_id":     item.get("isic_id", ""),
                    "attribution": item.get("attribution", ""),
                    "diagnosis_3": clin.get("diagnosis_3", query.split(":")[-1]),
                    "sex":         clin.get("sex", ""),
                    "age_approx":  clin.get("age_approx", ""),
                })
            print(f"    query={query[:50]} offset={offset} got={len(items)} total={total}")
            offset += PAGE_SIZE
            if offset >= total or offset >= limit:
                break
            time.sleep(DELAY)
        except Exception as e:
            print(f"  API error for '{query}': {e}")
            time.sleep(2)
            break
    return results


def download_image(isic_id: str, out_dir: Path) -> str | None:
    """Download one image via S3; return filepath or None on failure.

    Correct S3 URL format (confirmed Apr 2026):
      https://isic-archive.s3.amazonaws.com/images/{isic_id}.jpg
    Thumbnail fallback:
      https://isic-archive.s3.amazonaws.com/thumbnails/{isic_id}_thumbnail.jpg
    """
    shard = out_dir / isic_id[:7] / (isic_id + ".jpg")
    shard.parent.mkdir(parents=True, exist_ok=True)
    if shard.exists() and shard.stat().st_size > 0:
        return str(shard)

    # Primary: S3 full image
    url = S3_URL.format(isic_id=isic_id)
    try:
        r = requests.get(url, timeout=60, stream=True)
        if r.status_code == 200:
            tmp = shard.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    if chunk:
                        f.write(chunk)
            tmp.replace(shard)
            return str(shard)
        # Fallback: S3 thumbnail
        url2 = S3_THUMB.format(isic_id=isic_id)
        r2 = requests.get(url2, timeout=30)
        if r2.status_code == 200:
            shard.write_bytes(r2.content)
            return str(shard)
    except Exception:
        pass
    return None


def main():
    all_rows = []

    # Load existing minority manifest to avoid re-adding rows
    existing_ids: set = set()
    if MANIFEST_OUT.exists() and MANIFEST_OUT.stat().st_size > 100:
        ex = pd.read_csv(MANIFEST_OUT, low_memory=False)
        ex = ex[ex["isic_id"].notna()]
        existing_ids = {str(x) for x in ex["isic_id"].tolist() if str(x) and str(x) != "nan"}
        all_rows = ex.to_dict("records")
        print(f"Resuming from CSV: {len(existing_ids):,} IDs already in manifest", flush=True)

    # ALSO scan the on-disk download dir as source of truth: any *.jpg there
    # is already downloaded and we should not retry it. This protects against
    # CSV corruption or bookkeeping mismatches.
    on_disk = set()
    if OUT_DIR.exists():
        for p in OUT_DIR.rglob("*.jpg"):
            if p.stat().st_size > 0:
                on_disk.add(p.stem)
    print(f"Files already on disk: {len(on_disk):,}", flush=True)
    existing_ids |= on_disk

    for cls, cfg in TARGETS.items():
        want    = cfg["want"]
        already = sum(1 for r in all_rows if r.get("label_name") == cls)
        needed  = max(0, want - already)
        print(f"\n{'='*60}", flush=True)
        print(f"CLASS: {cls} | have={already} | want={want} | need={needed}", flush=True)

        if needed == 0:
            print(f"  Already have enough for {cls}, skipping", flush=True)
            continue

        # Collect candidate IDs from API using search endpoint
        candidates = []
        for q in cfg["queries"]:
            print(f"  Querying: {q}", flush=True)
            items = query_isic_search(q, limit=needed * 4)  # over-fetch for safety
            candidates.extend(items)
            print(f"    -> {len(items)} candidates", flush=True)

        # Deduplicate and filter out already-downloaded
        seen_ids = set()
        fresh = []
        n_already = 0
        for c in candidates:
            iid = str(c.get("isic_id", ""))
            if not iid:
                continue
            if iid in existing_ids:
                n_already += 1
                continue
            if iid in seen_ids:
                continue
            fresh.append(c)
            seen_ids.add(iid)

        print(f"  After dedup: {len(fresh)} fresh, {n_already} already done, "
              f"{len(candidates) - len(fresh) - n_already} dup-within-query", flush=True)

        # Don't over-cap; if we have fewer fresh than needed, take all of them
        fresh = fresh[:needed + 500]  # buffer for download failures
        print(f"  Downloading {len(fresh)} images for {cls}...", flush=True)

        downloaded = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(download_image, c["isic_id"], OUT_DIR): c
                       for c in fresh}
            for fut in as_completed(futures):
                c = futures[fut]
                try:
                    fp = fut.result()
                    if fp:
                        all_rows.append({
                            "isic_id":    c["isic_id"],
                            "filepath":   fp,
                            "label_name": cls,
                            "diagnosis_3": c.get("diagnosis_3", ""),
                            "source":     "ISIC_MINORITY",
                            "group_id":   c["isic_id"],
                            "sex":        c.get("sex", ""),
                            "age_approx": c.get("age_approx", ""),
                            "fitzpatrick": None,
                        })
                        existing_ids.add(c["isic_id"])
                        downloaded += 1
                        if downloaded % 100 == 0:
                            print(f"    [{cls}] {downloaded}/{len(fresh)} downloaded")
                except Exception as e:
                    print(f"    Error {c['isic_id']}: {e}")

        print(f"  {cls}: downloaded {downloaded} new images")

        # Save checkpoint after each class
        df = pd.DataFrame(all_rows)
        df.to_csv(MANIFEST_OUT, index=False)
        print(f"  Checkpoint saved: {MANIFEST_OUT} ({len(df)} rows)")

    # Final save + summary
    df = pd.DataFrame(all_rows)
    df.to_csv(MANIFEST_OUT, index=False)
    print(f"\n{'='*60}")
    print(f"DONE. Saved {MANIFEST_OUT} ({len(df):,} rows)")
    if len(df) > 0 and "label_name" in df.columns:
        print("Label distribution:")
        print(df["label_name"].value_counts().to_string())
    else:
        print("No rows saved.")


if __name__ == "__main__":
    main()
