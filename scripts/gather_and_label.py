#!/usr/bin/env python
import argparse, os, re
from pathlib import Path
import pandas as pd

def scan_images(root: Path):
    rows = []
    for p in root.rglob("ISIC_*.*"):
        if p.suffix.lower() not in (".jpg",".jpeg",".png",".bmp",".tif",".tiff"): 
            continue
        m = re.search(r"(ISIC_\d+)", p.name)
        if not m: 
            continue
        rows.append({"isic_id": m.group(1), "filepath": str(p.resolve())})
    return pd.DataFrame(rows).drop_duplicates(subset=["isic_id"])

def gather_metadata(roots):
    frames = []
    for r in roots:
        meta = Path(r) / "metadata.csv"
        if meta.exists():
            try:
                df = pd.read_csv(meta, low_memory=False)
                df["__source_dir__"] = str(Path(r).resolve())
                frames.append(df)
            except Exception as e:
                print(f"[warn] failed to read {meta}: {e}")
    if not frames:
        raise SystemExit("No metadata.csv files found.")
    meta_all = pd.concat(frames, ignore_index=True).drop_duplicates()
    # normalize an isic_id column
    cand = [c for c in meta_all.columns if c.lower() in ("isic_id","name","id")]
    if not cand:
        raise SystemExit("No isic_id/name/id column in metadata.")
    col = cand[0]
    meta_all["isic_id"] = meta_all[col].astype(str)
    meta_all["isic_id"] = meta_all["isic_id"].str.replace(r"[^A-Za-z0-9_]", "", regex=True)
    meta_all["isic_id"] = meta_all["isic_id"].apply(lambda s: s if s.startswith("ISIC_") else (f"ISIC_{s}" if s.isdigit() else s))
    return meta_all

# light-weight mapper (same as earlier join script)
ALIASES = {
    "mel": ["mel","melanoma","melanoma invasive","melanoma in situ","superficial spreading melanoma","nodular melanoma","lentigo maligna","lentigo malignant melanoma","acral lentiginous melanoma","mucosal melanoma","mm","melanocytic malignant"],
    "nv":  ["nv","nevus","naevus","melanocytic nevus","common nevus","atypical nevus","dysplastic nevus","compound nevus","intradermal nevus","junctional nevus","melanocytic nevi","nevi"],
    "bcc": ["bcc","basal cell carcinoma","basalioma"],
    "scc": ["scc","squamous cell carcinoma","keratinocyte carcinoma","squamous carcinoma","akiec_scc"],
    "df":  ["df","dermatofibroma","histiocytoma"],
    "bkl": ["bkl","benign keratosis","seborrheic keratosis","solar lentigo","lentigo","lichen planus-like keratosis","lplk","sk","sebk"],
    "ak":  ["ak","actinic keratosis","akiec","actinic keratoses"],
    "vasc":["vasc","vascular","angioma","hemangioma","pyogenic granuloma","venous lake","vascular lesion"],
}
CLASS_MAP = {k:i for i,k in enumerate(["mel","nv","bcc","scc","df","bkl","ak","vasc"])}

def norm(s): 
    return re.sub(r"\s+"," ", s.strip().lower()) if isinstance(s,str) else ""

def pick_diag(row):
    for k in ("diagnosis_3","diagnosis 3","metadata.clinical.diagnosis_3",
              "diagnosis_2","diagnosis 2","metadata.clinical.diagnosis_2",
              "diagnosis","metadata.clinical.diagnosis","dx"):
        if k in row and str(row[k]).strip():
            return norm(str(row[k]))
    # fallback: any diagnosis-like col
    for k in row.index:
        if "diagnosis" in k.lower() and str(row[k]).strip():
            return norm(str(row[k]))
    return ""

def to_fine(diag):
    s = norm(diag)
    if not s: return ""
    for k, vals in ALIASES.items():
        for v in vals:
            if s == v or v in s: return k
    if "seborrheic keratosis" in s or "sebk" in s: return "bkl"
    if any(v in s for v in ["angioma","hemangioma","vascular"]): return "vasc"
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="Folders that contain images + metadata.csv (e.g., D:\\AIFINAL\\labeled_images D:\\AIFINAL\\labeled\\coll_70 ...)")
    ap.add_argument("--out_manifest", default="data/manifest_unified.csv")
    ap.add_argument("--out_meta", default="data/isic_meta_ALL_LABELED.csv")
    ap.add_argument("--out_labeled", default="data/manifest_unified_labeled.csv")
    ap.add_argument("--label_strategy", choices=["fine","coarse"], default="fine")
    args = ap.parse_args()

    # 1) images
    imgs = pd.concat([scan_images(Path(r)) for r in args.roots], ignore_index=True).drop_duplicates(subset=["isic_id"])
    Path(args.out_manifest).parent.mkdir(parents=True, exist_ok=True)
    imgs.to_csv(args.out_manifest, index=False)
    print(f"[manifest] {len(imgs)} rows -> {args.out_manifest}")

    # 2) metadata
    meta = gather_metadata(args.roots)
    meta["diagnosis_norm"] = meta.apply(pick_diag, axis=1)
    meta["fine_label"] = meta["diagnosis_norm"].apply(to_fine)
    # coarse by heuristic
    def coarse_from_fine(s):
        if s in ("mel","bcc","scc"): return "malignant"
        if s in ("nv","df","bkl","ak","vasc"): return "benign"
        return ""
    meta["coarse_label"] = meta["fine_label"].apply(coarse_from_fine)
    meta[["isic_id","diagnosis_norm","fine_label","coarse_label"]].to_csv(args.out_meta, index=False)
    print(f"[meta] {len(meta)} rows -> {args.out_meta}")

    # 3) join
    merged = imgs.merge(meta[["isic_id","diagnosis_norm","fine_label","coarse_label"]], on="isic_id", how="left")
    if args.label_strategy == "fine":
        merged["label_name"] = merged["fine_label"]
        merged["label_id"] = merged["label_name"].map(CLASS_MAP)
    else:
        merged["label_name"] = merged["coarse_label"]
        merged["label_id"] = merged["label_name"].map({"benign":0,"malignant":1})

    labeled = merged[merged["label_name"].astype(str).str.len() > 0].copy()
    labeled.to_csv(args.out_labeled, index=False)
    print(f"[labeled] kept {len(labeled)} rows -> {args.out_labeled}")

if __name__ == "__main__":
    main()
