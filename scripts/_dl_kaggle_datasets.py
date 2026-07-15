"""Download dermatology datasets from Kaggle using the user's API token.

Uses KAGGLE_API_TOKEN env var (set by parent process - never written to disk).

Targets (priority order):
  1. PAD-UFES-20: maxjen/pad-ufes-20  — 2,298 smartphone clinical images, 6 classes
  2. Fitzpatrick17k: anaranjank/fitzpatrick17k-skin-condition-dataset
                     OR mrinmoyas/fitzpatrick17k-cleaned (try several)
  3. ISIC 2020 challenge: cdeotte/jpeg-isic2020-256x256 (small, fast)
  4. Skin-Cancer-MNIST (HAM10000): kmader/skin-cancer-mnist-ham10000 (already have most)

Outputs:
  D:\AIFINAL\images\external\<dataset_name>\
  data/manifests/kaggle_external.csv with rows ready to merge into master
"""
import os, sys, time, zipfile, csv
from pathlib import Path

# Lazy kaggle import — only needed when actually downloading.
# Integration-only callers (e.g. _integrators_v3.py) don't need auth.
def _ensure_kaggle():
    if "KAGGLE_API_TOKEN" not in os.environ:
        raise RuntimeError("KAGGLE_API_TOKEN not set — cannot download")
    import kaggle  # type: ignore
    kaggle.api.authenticate()
    return kaggle

EXT_ROOT = Path(r"D:\AIFINAL\images\external")
EXT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_CSV = Path("data/manifests/kaggle_external.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

ALL_ROWS = []


def safe_unzip(zip_path: Path, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {zip_path.name} ({zip_path.stat().st_size:,} bytes) -> {target}", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target)


def download_dataset(slug: str, dest_dir: Path) -> Path | None:
    """Download a Kaggle dataset, return path to extracted dir."""
    out = dest_dir
    out.mkdir(parents=True, exist_ok=True)

    # Skip if already extracted (look for any .jpg/.png inside)
    n = sum(1 for _ in out.rglob('*')) if out.exists() else 0
    if n > 5:
        print(f"  {slug}: already extracted ({n} files in {out}); skipping download", flush=True)
        return out

    print(f"\n  Downloading {slug} ...", flush=True)
    t0 = time.time()
    try:
        kaggle = _ensure_kaggle()
        kaggle.api.dataset_download_files(slug, path=str(out), unzip=True, quiet=False)
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)
        return None
    print(f"  Done in {(time.time()-t0)/60:.1f}m", flush=True)
    return out


# ── PAD-UFES-20 ──────────────────────────────────────────────────────────────
def integrate_pad_ufes(extracted_dir: Path):
    """Walk extracted PAD-UFES dir, find metadata.csv, create manifest rows."""
    print(f"\n  Integrating PAD-UFES-20 from {extracted_dir}", flush=True)

    # Find metadata.csv
    meta_files = list(extracted_dir.rglob("metadata*.csv"))
    if not meta_files:
        print("    no metadata.csv found", flush=True)
        return
    meta_path = meta_files[0]
    print(f"    metadata: {meta_path}", flush=True)

    import pandas as pd
    meta = pd.read_csv(meta_path)
    print(f"    metadata rows: {len(meta)}, columns: {list(meta.columns)[:10]}")

    # Find diagnostic column
    dx_col = None
    for c in ['diagnostic', 'diagnosis', 'dx', 'class', 'label']:
        if c in meta.columns:
            dx_col = c; break
    if not dx_col:
        print(f"    no diagnostic column found in {list(meta.columns)}", flush=True)
        return

    # Build image lookup by name
    img_lookup = {}
    for ext in (".png", ".jpg", ".jpeg"):
        for p in extracted_dir.rglob(f"*{ext}"):
            img_lookup[p.name] = str(p)
            img_lookup[p.stem] = str(p)

    print(f"    images on disk: {len(img_lookup)//2:,} unique stems", flush=True)

    # Find image filename column
    img_col = None
    for c in ['img_id', 'image_id', 'image', 'filename', 'file_name', 'path']:
        if c in meta.columns:
            img_col = c; break
    if not img_col:
        print(f"    no image-name column. Trying lesion_id...", flush=True)
        return

    # PAD-UFES dx values are: ACK, BCC, MEL, NEV, SCC, SEK
    # Map to our 8-class:
    pad_to_8 = {
        "ACK": "ak",   # Actinic Keratosis
        "BCC": "bcc",
        "MEL": "mel",
        "NEV": "nv",
        "SCC": "scc",
        "SEK": "bkl",  # Seborrheic Keratosis -> BKL
    }

    matched = 0
    for _, r in meta.iterrows():
        img_name = str(r[img_col])
        cls_pad  = str(r[dx_col]).upper().strip()
        cls8     = pad_to_8.get(cls_pad)
        if not cls8:
            continue
        fp = img_lookup.get(img_name) or img_lookup.get(Path(img_name).stem)
        if not fp:
            continue
        # Group by patient if possible
        group = r.get('patient_id', r.get('lesion_id', Path(fp).stem))
        ALL_ROWS.append({
            "isic_id":    f"PAD_{Path(fp).stem}",
            "filepath":   fp,
            "label_name": cls8,
            "source":     "PAD-UFES-20",
            "group_id":   f"PAD_{group}",
            "fitzpatrick": str(r.get("fitspatrick", r.get("fitzpatrick", ""))),
            "sex":         str(r.get("gender", "")),
            "age_approx":  str(r.get("age", "")),
        })
        matched += 1

    print(f"    matched: {matched}/{len(meta)} rows", flush=True)


# ── ISIC 2019 ────────────────────────────────────────────────────────────────
def integrate_isic2019(extracted_dir: Path):
    """ISIC 2019: ~25,331 8-class images. CSV labels in `ISIC_2019_Training_GroundTruth.csv`."""
    print(f"\n  Integrating ISIC 2019 from {extracted_dir}", flush=True)
    import pandas as pd

    # Find ground truth CSV
    gt_files = list(extracted_dir.rglob("*GroundTruth*.csv"))
    if not gt_files:
        print("    no GroundTruth CSV found", flush=True)
        return
    gt_path = gt_files[0]
    print(f"    metadata: {gt_path}", flush=True)
    gt = pd.read_csv(gt_path)
    print(f"    GT shape: {gt.shape}, columns: {list(gt.columns)}", flush=True)

    # ISIC 2019 columns: image, MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK
    cls_cols = [c for c in ["MEL","NV","BCC","AK","BKL","DF","VASC","SCC"] if c in gt.columns]
    cls_to_8 = {"MEL":"mel","NV":"nv","BCC":"bcc","AK":"ak",
                "BKL":"bkl","DF":"df","VASC":"vasc","SCC":"scc"}

    img_lookup: dict[str, str] = {}
    for ext in (".jpg", ".jpeg", ".png"):
        for p in extracted_dir.rglob(f"*{ext}"):
            img_lookup[p.stem] = str(p)

    print(f"    images on disk: {len(img_lookup):,}", flush=True)

    matched = 0
    for _, r in gt.iterrows():
        img_id = str(r.get("image", "")).strip()
        if not img_id: continue
        # Find class with value == 1
        cls8 = None
        for col in cls_cols:
            try:
                if float(r[col]) >= 0.5:
                    cls8 = cls_to_8[col]; break
            except Exception:
                continue
        if not cls8: continue
        fp = img_lookup.get(img_id) or img_lookup.get(Path(img_id).stem)
        if not fp: continue
        ALL_ROWS.append({
            "isic_id":    img_id,
            "filepath":   fp,
            "label_name": cls8,
            "source":     "ISIC_2019_KAGGLE",
            "group_id":   img_id,
            "fitzpatrick": None,
            "sex": None,
            "age_approx": None,
        })
        matched += 1
    print(f"    matched: {matched}/{len(gt)} rows", flush=True)


# ── Derm7pt ──────────────────────────────────────────────────────────────────
def integrate_derm7pt(extracted_dir: Path):
    """Derm7pt: ~1011 dermo+clinical pairs. CSV in meta/."""
    print(f"\n  Integrating Derm7pt from {extracted_dir}", flush=True)
    import pandas as pd

    # Find meta CSV
    meta_files = list(extracted_dir.rglob("meta*.csv")) + list(extracted_dir.rglob("*.csv"))
    if not meta_files:
        print("    no meta CSV found", flush=True)
        return
    meta_path = meta_files[0]
    print(f"    metadata: {meta_path}", flush=True)
    meta = pd.read_csv(meta_path)
    print(f"    columns: {list(meta.columns)}", flush=True)

    # Derm7pt diagnosis values include: melanoma, naevus, basal cell carcinoma, etc
    dx_col = None
    for c in ['diagnosis', 'diagnostic', 'dx', 'class', 'label']:
        if c in meta.columns:
            dx_col = c; break
    if not dx_col:
        print(f"    no diagnostic column", flush=True)
        return

    # Derm7pt has both clinical and dermoscopy images
    img_cols = [c for c in ['clinic', 'derm', 'image', 'derm_path', 'clinic_path'] if c in meta.columns]

    img_lookup: dict[str, str] = {}
    for ext in (".jpg", ".jpeg", ".png"):
        for p in extracted_dir.rglob(f"*{ext}"):
            img_lookup[p.stem] = str(p)
            img_lookup[p.name] = str(p)
            # also relative path
            try:
                rel = p.relative_to(extracted_dir)
                img_lookup[str(rel).replace("\\","/")] = str(p)
            except Exception: pass
    print(f"    images on disk: {len(img_lookup):,} keys", flush=True)

    # Map derm7pt dx to 8-class
    derm_to_8 = {
        "melanoma": "mel",
        "melanoma (in situ)": "mel",
        "melanoma metastasis": "mel",
        "nevus": "nv", "naevus": "nv",
        "basal cell carcinoma": "bcc",
        "seborrheic keratosis": "bkl",
        "lentigo": "bkl",
        "solar lentigo": "bkl",
        "dermatofibroma": "df",
        "vascular lesion": "vasc",
        "miscellaneous": None,
        "melanosis": None,  # not 8-class
        "clark nevus": "nv",
        "dermal nevus": "nv",
        "blue nevus": "nv",
        "combined nevus": "nv",
        "congenital nevus": "nv",
        "recurrent nevus": "nv",
        "reed or spitz nevus": "nv",
    }

    matched = 0
    for _, r in meta.iterrows():
        dx_raw = str(r[dx_col]).strip().lower()
        # First try exact match
        cls8 = derm_to_8.get(dx_raw)
        # Substring match for "melanoma (less than 0.76 mm)" etc.
        if cls8 is None and "melanoma" in dx_raw:
            cls8 = "mel"
        if not cls8: continue
        # Try each image column
        for img_col in img_cols:
            img_path = str(r.get(img_col, ""))
            if not img_path: continue
            fp = (img_lookup.get(img_path) or
                  img_lookup.get(Path(img_path).stem) or
                  img_lookup.get(img_path.replace("\\","/")))
            if not fp: continue
            ALL_ROWS.append({
                "isic_id":    f"DERM7_{Path(fp).stem}",
                "filepath":   fp,
                "label_name": cls8,
                "source":     "DERM7PT",
                "group_id":   f"DERM7_{r.get('case_num', Path(fp).stem)}",
                "fitzpatrick": None,
                "sex": str(r.get("sex", "")) if "sex" in meta.columns else None,
                "age_approx": None,
            })
            matched += 1
    print(f"    matched: {matched} rows from {len(meta)} cases", flush=True)


# ── BCN20000 ─────────────────────────────────────────────────────────────────
def integrate_bcn20000(extracted_dir: Path):
    """BCN20000: 18,946 ISIC 8-class images. Has metadata.csv with text diagnosis labels."""
    print(f"\n  Integrating BCN20000 from {extracted_dir}", flush=True)
    import pandas as pd
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from mnkai.data.manifest import LabelNormalizer

    md = list(extracted_dir.rglob("metadata.csv"))
    if not md:
        print("    no metadata.csv found", flush=True); return
    meta = pd.read_csv(md[0], low_memory=False)
    print(f"    metadata: {md[0]}, shape={meta.shape}", flush=True)

    img_lookup: dict[str, str] = {}
    for ext in (".jpg", ".jpeg", ".png"):
        for p in extracted_dir.rglob(f"*{ext}"):
            img_lookup[p.stem] = str(p)
    print(f"    images on disk: {len(img_lookup):,}", flush=True)

    norm = LabelNormalizer()
    matched = 0
    skipped_lbl = 0
    for _, r in meta.iterrows():
        iid = str(r["isic_id"]).strip()
        # Try diagnosis_3 first, then diagnosis (more specific = better)
        for col in ["diagnosis_3", "diagnosis"]:
            dx = str(r.get(col, "") or "").strip()
            if dx:
                cls8 = norm.map(dx)
                if cls8: break
        else:
            cls8 = None

        if not cls8:
            skipped_lbl += 1
            continue
        fp = img_lookup.get(iid)
        if not fp: continue
        ALL_ROWS.append({
            "isic_id":    iid,
            "filepath":   fp,
            "label_name": cls8,
            "source":     "BCN20000_KAGGLE",
            "group_id":   str(r.get("lesion_id", iid)) if pd.notna(r.get("lesion_id", None)) else iid,
            "fitzpatrick": None,
            "sex":         str(r.get("sex", "")) if "sex" in meta.columns else None,
            "age_approx":  r.get("age_approx", None),
        })
        matched += 1
    print(f"    matched: {matched}/{len(meta)} rows (skipped {skipped_lbl} non-8class)", flush=True)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    # 1) PAD-UFES-20 — try mahdavi1202/skin-cancer (most downloaded, 3.6 GB)
    pad_dir = EXT_ROOT / "pad-ufes-20"
    pad_slugs = ["mahdavi1202/skin-cancer", "maxjen/pad-ufes-20", "paulaiskanderrr/pad-ufes-20"]
    for slug in pad_slugs:
        res = download_dataset(slug, pad_dir)
        if res:
            integrate_pad_ufes(res)
            break

    # 2) ISIC 2019 (8-class gold standard, 25k images, includes BCN20000 + HAM)
    isic19_dir = EXT_ROOT / "isic-2019"
    res = download_dataset("andrewmvd/isic-2019", isic19_dir)
    if res:
        integrate_isic2019(res)

    # 3) Derm7pt (small, 1011 images)
    derm7_dir = EXT_ROOT / "derm7pt"
    res = download_dataset("menakamohanakumar/derm7pt", derm7_dir)
    if res:
        integrate_derm7pt(res)

    # 4) BCN20000 standalone (1 GB)
    bcn_dir = EXT_ROOT / "bcn20000"
    res = download_dataset("radwahashiesh/bcn20000", bcn_dir)
    if res:
        integrate_bcn20000(res)

    # Save manifest
    if ALL_ROWS:
        import pandas as pd
        df = pd.DataFrame(ALL_ROWS)
        df.to_csv(OUT_CSV, index=False)
        print(f"\nSaved {OUT_CSV} ({len(df)} rows)", flush=True)
        print(df["label_name"].value_counts().to_string(), flush=True)
        print(df["source"].value_counts().to_string(), flush=True)
    else:
        print("\nNo Kaggle rows added.", flush=True)


if __name__ == "__main__":
    main()
