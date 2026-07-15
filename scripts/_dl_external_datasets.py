"""Download external dermatology datasets into D:\AIFINAL\images\external.

Targets (Apr 2026, what's actually downloadable without auth):
  1. 142 missing HAM10000 IDs (ISIC S3 - confirmed)
  2. MED-NODE: 170 images (70 mel + 100 nv) - cs.rug.nl direct zip
  3. PH2: 200 images (80 nv + 80 atypical nv + 40 mel) - try various URLs
  4. Derm7pt: ~1011 dermo+clinical pairs - try cs.sfu.ca

Outputs:
  - Files into D:\AIFINAL\images\external\<dataset>\
  - data/manifests/external_datasets.csv with rows ready to merge into master
"""
import csv, json, time, zipfile, io, requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

EXT_ROOT     = Path(r"D:\AIFINAL\images\external")
OUT_CSV      = Path("data/manifests/external_datasets.csv")
WORKERS      = 20
TIMEOUT      = 60

EXT_ROOT.mkdir(parents=True, exist_ok=True)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

ALL_ROWS = []


# ── Source 1: 142 missing HAM10000 IDs ───────────────────────────────────────
def download_missing_ham10000():
    miss_file = Path("data/manifests/_missing_ham10000_ids.json")
    if not miss_file.exists():
        print("  no _missing_ham10000_ids.json - skipping HAM supplement", flush=True)
        return
    with open(miss_file) as f:
        ids = json.load(f)
    print(f"\n=== HAM10000 supplement: {len(ids)} missing IDs ===", flush=True)

    # Need their labels too - cross-reference with HAM metadata
    import pandas as pd
    ham = pd.read_csv("data/manifests/HAM10000_metadata.tab", sep="\t")
    id_to_label  = dict(zip(ham["image_id"].astype(str), ham["dx"].astype(str)))
    id_to_lesion = dict(zip(ham["image_id"].astype(str), ham["lesion_id"].astype(str)))

    out_dir = EXT_ROOT / "ham10000_supplement"
    out_dir.mkdir(parents=True, exist_ok=True)

    def dl(iid):
        dest = out_dir / f"{iid}.jpg"
        if dest.exists() and dest.stat().st_size > 0:
            return iid, str(dest), True
        url = f"https://isic-archive.s3.amazonaws.com/images/{iid}.jpg"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                dest.write_bytes(r.content)
                return iid, str(dest), True
        except Exception:
            pass
        return iid, "", False

    ok = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as p:
        for iid, fp, success in p.map(dl, ids):
            if success:
                ok += 1
                lbl = id_to_label.get(iid, "")
                # Map HAM dx to 8-class
                dx_map = {"akiec":"ak", "bcc":"bcc", "bkl":"bkl", "df":"df",
                          "mel":"mel", "nv":"nv", "vasc":"vasc"}
                cls = dx_map.get(lbl, "")
                if cls:
                    ALL_ROWS.append({
                        "isic_id":    iid,
                        "filepath":   fp,
                        "label_name": cls,
                        "source":     "HAM10000_SUPP",
                        "group_id":   id_to_lesion.get(iid, iid),
                        "fitzpatrick": None,
                        "sex": None,
                        "age_approx": None,
                    })
    print(f"  downloaded {ok}/{len(ids)} HAM supplement images", flush=True)


# ── Source 2: MED-NODE (70 mel + 100 nv = 170 macroscopic) ───────────────────
def download_mednode():
    out_dir = EXT_ROOT / "mednode"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_url = "https://www.cs.rug.nl/~imaging/databases/melanoma_naevi/complete_mednode_dataset.zip"
    print(f"\n=== MED-NODE: downloading {zip_url} ===", flush=True)

    zip_path = out_dir / "complete_mednode_dataset.zip"
    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
        try:
            r = requests.get(zip_url, timeout=300, stream=True)
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
            print(f"  zip size: {zip_path.stat().st_size:,} bytes", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            return

    # Extract
    extract_dir = out_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
    except Exception as e:
        print(f"  extract failed: {e}", flush=True)
        return

    # Walk and label
    n_mel, n_nv = 0, 0
    for img in extract_dir.rglob("*"):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        # MED-NODE folder structure: melanoma/ and naevus/
        parts = [p.lower() for p in img.parts]
        if any("melanoma" in p for p in parts):
            cls = "mel"; n_mel += 1
        elif any("naev" in p or "nv" in p for p in parts):
            cls = "nv"; n_nv += 1
        else:
            continue
        ALL_ROWS.append({
            "isic_id":    f"MEDNODE_{img.stem}",
            "filepath":   str(img),
            "label_name": cls,
            "source":     "MED-NODE",
            "group_id":   f"MEDNODE_{img.stem}",
            "fitzpatrick": None,
            "sex": None,
            "age_approx": None,
        })
    print(f"  MED-NODE: {n_mel} mel + {n_nv} nv = {n_mel+n_nv} images", flush=True)


# ── Source 3: PH2 (try multiple URLs) ────────────────────────────────────────
def download_ph2():
    out_dir = EXT_ROOT / "ph2"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== PH2: searching for working URL ===", flush=True)

    candidates = [
        "https://www.fc.up.pt/addi/PH2Dataset.rar",
        "https://www.fc.up.pt/addi/PH2_Dataset.zip",
        "https://www.fc.up.pt/addi/PH2Dataset.zip",
        "http://www.fc.up.pt/addi/Downloads/PH2Dataset.rar",
    ]
    found_url = None
    for url in candidates:
        try:
            r = requests.head(url, timeout=15, allow_redirects=True)
            ct = r.headers.get("Content-Type", "")
            if r.status_code == 200 and "html" not in ct:
                found_url = url; break
            print(f"  {r.status_code} {url}", flush=True)
        except Exception:
            pass

    if not found_url:
        print(f"  PH2: no working direct URL found - skipping (manual download required)", flush=True)
        return

    # download...
    fname = Path(found_url).name
    out_zip = out_dir / fname
    print(f"  downloading {fname} from {found_url}", flush=True)
    r = requests.get(found_url, timeout=600, stream=True)
    with open(out_zip, "wb") as f:
        for chunk in r.iter_content(1 << 16):
            f.write(chunk)
    print(f"  zip size: {out_zip.stat().st_size:,} bytes", flush=True)


# ── Source 4: Derm7pt (try multiple URLs) ────────────────────────────────────
def download_derm7pt():
    print(f"\n=== Derm7pt: searching for working URL ===", flush=True)
    candidates = [
        "http://derm.cs.sfu.ca/Welcome_files/release_v0.zip",
        "http://derm.cs.sfu.ca/release/release_v0.zip",
        "https://derm.cs.sfu.ca/release_v0.zip",
        "https://derm.cs.sfu.ca/Welcome_files/release_v0.zip",
    ]
    for url in candidates:
        try:
            r = requests.head(url, timeout=15, allow_redirects=True)
            ct = r.headers.get("Content-Type", "")
            if r.status_code == 200 and "html" not in ct:
                print(f"  found: {url} ({r.headers.get('Content-Length','?')} bytes)", flush=True)
                # Only download if reasonable size (Derm7pt is ~1.5GB)
                # Save URL only for now; user can decide on manual download
                return url
            print(f"  {r.status_code} {url}", flush=True)
        except Exception:
            pass
    print(f"  Derm7pt: no direct URL found - skipping", flush=True)


def main():
    download_missing_ham10000()
    download_mednode()
    download_ph2()
    download_derm7pt()

    # Save manifest
    if ALL_ROWS:
        import pandas as pd
        df = pd.DataFrame(ALL_ROWS)
        df.to_csv(OUT_CSV, index=False)
        print(f"\nSaved {OUT_CSV} ({len(df)} rows)", flush=True)
        print(df["label_name"].value_counts().to_string(), flush=True)
        print(df["source"].value_counts().to_string(), flush=True)
    else:
        print("\nNo external rows added.", flush=True)


if __name__ == "__main__":
    main()
