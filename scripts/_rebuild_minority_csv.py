"""Rebuild data/manifests/isic_minority.csv from on-disk files.

We have 702 .jpg files in D:\AIFINAL\images\isic_minority but no longer have
the CSV mapping each to its class. Recover by:
 1. Listing every file on disk and reading the ISIC ID from the filename.
 2. Querying ISIC search API for each minority class (ak/scc/df/vasc).
 3. Cross-referencing: if ID is on disk AND in class search results, label it.

Run: python scripts/_rebuild_minority_csv.py
"""
import requests, pandas as pd
from pathlib import Path

OUT_DIR      = Path(r"D:\AIFINAL\images\isic_minority")
MANIFEST_OUT = Path("data/manifests/isic_minority.csv")
SEARCH       = "https://api.isic-archive.com/api/v2/images/search/"
PAGE_SIZE    = 100

# Same query->class mapping used in _dl_minority_classes.py
QUERY_TO_CLASS = {
    'diagnosis_3:"Solar or actinic keratosis"':       "ak",
    "diagnosis_3:Keratoacanthoma":                    "ak",
    'diagnosis_3:"Squamous cell carcinoma, NOS"':     "scc",
    'diagnosis_3:"Squamous cell carcinoma in situ"':  "scc",
    "diagnosis_3:Dermatofibroma":                     "df",
    "diagnosis_3:Hemangioma":                         "vasc",
    "diagnosis_3:Angiokeratoma":                      "vasc",
    'diagnosis_3:"Pyogenic granuloma"':               "vasc",
}


def query_all(query: str) -> list[dict]:
    """Page through all results for one query."""
    out = []
    offset = 0
    while True:
        params = {"query": query, "limit": PAGE_SIZE, "offset": offset}
        r = requests.get(SEARCH, params=params, timeout=30)
        if r.status_code != 200:
            break
        data  = r.json()
        items = data.get("results", [])
        if not items:
            break
        for it in items:
            clin = it.get("metadata", {}).get("clinical", {})
            out.append({
                "isic_id":     it.get("isic_id", ""),
                "diagnosis_3": clin.get("diagnosis_3", ""),
                "sex":         clin.get("sex", ""),
                "age_approx":  clin.get("age_approx", ""),
                "attribution": it.get("attribution", ""),
            })
        total = data.get("count", 0)
        offset += PAGE_SIZE
        if offset >= total or offset >= 5000:
            break
    return out


def main():
    # 1. List on-disk files
    on_disk = {}
    for p in OUT_DIR.rglob("*.jpg"):
        if p.stat().st_size > 0:
            on_disk[p.stem] = str(p)
    print(f"On-disk files: {len(on_disk):,}")

    # 2. Build ID->class lookup from search API
    id_to_class: dict[str, tuple[str, dict]] = {}
    for query, cls in QUERY_TO_CLASS.items():
        print(f"  Querying {query} -> {cls}")
        items = query_all(query)
        print(f"    -> {len(items)} returned")
        for it in items:
            iid = it.get("isic_id")
            if iid and iid not in id_to_class:
                id_to_class[iid] = (cls, it)

    print(f"Class lookup built: {len(id_to_class):,} unique IDs across all minority queries")

    # 3. Match on-disk files to class
    rows = []
    matched = 0
    for iid, fp in on_disk.items():
        if iid in id_to_class:
            cls, meta = id_to_class[iid]
            rows.append({
                "isic_id":     iid,
                "filepath":    fp,
                "label_name":  cls,
                "diagnosis_3": meta.get("diagnosis_3", ""),
                "source":      "ISIC_MINORITY",
                "group_id":    iid,
                "sex":         meta.get("sex", ""),
                "age_approx":  meta.get("age_approx", ""),
                "fitzpatrick": None,
            })
            matched += 1

    print(f"Matched: {matched}/{len(on_disk)} files to class")

    df = pd.DataFrame(rows)
    df.to_csv(MANIFEST_OUT, index=False)
    print(f"Saved {MANIFEST_OUT} ({len(df)} rows)")
    if len(df):
        print("Class distribution:")
        print(df["label_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
