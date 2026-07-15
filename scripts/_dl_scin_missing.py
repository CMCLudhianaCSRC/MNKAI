"""Download the SCIN images we don't already have, from the PUBLIC GCS bucket
(no credentials). SCIN's condition mix (eczema, contact dermatitis, urticaria,
psoriasis, tinea, impetigo/bacterial, acne, drug rash) is exactly our weak tail.

Existing SCIN images live in D:\\AIFINAL\\images\\external\\scin\\dataset\\images\\;
we have 5,231 of 10,380. This pulls the ~5,149 missing (mostly extra views of the
same cases -> share case_id group -> no leakage, more tail signal).

    python scripts/_dl_scin_missing.py
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests

DL_DIR = Path("D:/AIFINAL/downloads/scin_full")
IMG_DIR = Path("D:/AIFINAL/images/external/scin/dataset/images")
BASE = "https://storage.googleapis.com/dx-scin-public-data"
MANIFEST_OUT = Path("data/manifests/scin_new_images.csv")


def load_targets():
    c = pd.read_csv(DL_DIR / "scin_cases.csv", low_memory=False)
    rows = []
    for _, r in c.iterrows():
        for col in ("image_1_path", "image_2_path", "image_3_path"):
            p = r.get(col)
            if isinstance(p, str) and p.strip() and p.lower() != "nan":
                rows.append({"case_id": r["case_id"], "gcs_path": p.strip(),
                             "basename": os.path.basename(p.strip()),
                             "fitzpatrick": r.get("fitzpatrick_skin_type")})
    return pd.DataFrame(rows).drop_duplicates("basename")


def fetch(row):
    dst = IMG_DIR / row["basename"]
    if dst.exists() and dst.stat().st_size > 0:
        return ("skip", row["basename"])
    url = f"{BASE}/{row['gcs_path']}"
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and len(r.content) > 100:
                dst.write_bytes(r.content)
                return ("ok", row["basename"])
            return (f"http{r.status_code}", row["basename"])
        except Exception as e:
            time.sleep(1.5 * (attempt + 1))
    return ("fail", row["basename"])


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    tgt = load_targets()
    have = {p.name for p in IMG_DIR.glob("*.png")}
    todo = tgt[~tgt["basename"].isin(have)].reset_index(drop=True)
    print(f"[scin] full={len(tgt)} have={len(have)} to_download={len(todo)}", flush=True)
    tgt.to_csv(MANIFEST_OUT, index=False)
    print(f"[scin] wrote target manifest -> {MANIFEST_OUT}", flush=True)

    ok = skip = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(fetch, r) for _, r in todo.iterrows()]
        for i, f in enumerate(as_completed(futs)):
            st, _ = f.result()
            if st == "ok": ok += 1
            elif st == "skip": skip += 1
            else: fail += 1
            if (i + 1) % 250 == 0:
                print(f"[scin] {i+1}/{len(todo)}  ok={ok} fail={fail} "
                      f"({(time.time()-t0)/60:.1f}m)", flush=True)
    print(f"[scin] DONE ok={ok} skip={skip} fail={fail} in {(time.time()-t0)/60:.1f}m", flush=True)
    print(f"[scin] images now in {IMG_DIR}: {len(list(IMG_DIR.glob('*.png')))}", flush=True)


if __name__ == "__main__":
    main()
