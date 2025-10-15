import argparse
from pathlib import Path
import pandas as pd
import re

# Map free-text diagnosis to canonical multi-class labels
def canon_label(dx: str) -> str:
    if not dx: return "other"
    s = re.sub(r"\\s+", " ", dx.strip().lower())
    # common ISIC classes & synonyms
    if any(k in s for k in ["melanoma"]): return "mel"
    if any(k in s for k in ["nevus", "naevus", "melanocytic naevus", "melanocytic nevus"]): return "nv"
    if any(k in s for k in ["basal cell", "bcc"]): return "bcc"
    if any(k in s for k in ["actinic keratosis", "solar keratosis", "intraepithelial carcinoma", "akiec"]): return "ak"
    if any(k in s for k in ["seborrheic keratosis", "bkl", "benign keratosis", "lichen", "lentigo"]): return "bkl"
    if any(k in s for k in ["dermatofibroma", "df"]): return "df"
    if any(k in s for k in ["squamous cell", "scc", "keratoacanthoma"]): return "scc"
    if any(k in s for k in ["vascular", "angioma", "hemangioma", "pyogenic granuloma", "vasc"]): return "vasc"
    return "other"

def coarse_label(dx_label: str, benign_malignant: str) -> str:
    # Prefer explicit benign_malignant if available
    if benign_malignant in ("benign", "malignant"):
        return benign_malignant
    # Otherwise decide from canonical multi-class
    if dx_label in ("mel", "bcc", "scc", "ak"):  # AKIEC is intraepithelial carcinoma (in situ) → often treated as malignant for screening
        return "malignant"
    if dx_label in ("nv", "bkl", "df", "vasc"):
        return "benign"
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths_csv", default="data/manifest_unified.csv",
                    help="Your existing manifest with filepaths (has isic_id, filepath, etc.)")
    ap.add_argument("--meta_csv", default="data/isic_meta.csv",
                    help="Output of fetch_isic_metadata.py")
    ap.add_argument("--out_csv", default="data/manifest_unified_labeled.csv")
    ap.add_argument("--label_strategy", choices=["fine", "coarse"], default="fine",
                    help="'fine' → mel/nv/bcc/..., 'coarse' → benign/malignant")
    ap.add_argument("--drop_unknown", action="store_true",
                    help="Drop rows with unknown label after labeling")
    args = ap.parse_args()

    p = pd.read_csv(args.paths_csv)
    m = pd.read_csv(args.meta_csv)

    if "isic_id" not in p.columns or "isic_id" not in m.columns:
        raise SystemExit("Both CSVs must have an 'isic_id' column")

    df = p.merge(m, on="isic_id", how="left", suffixes=("", "_meta"))

    # If 'modality' missing on path CSV, keep from meta
    if "modality" not in p.columns and "modality" in df.columns:
        pass  # already merged in

    # Build canonical multi-class label
    dx = df["diagnosis"].fillna("").astype(str)
    df["label_fine"] = dx.map(canon_label)

    # Build coarse label
    bm = df["benign_malignant"].fillna("").astype(str).str.lower()
    df["label_coarse"] = [coarse_label(f, b) for f, b in zip(df["label_fine"], bm)]

    # Choose final label column
    if args.label_strategy == "fine":
        df["label"] = df["label_fine"]
    else:
        df["label"] = df["label_coarse"]

    # Ensure dermoscopic only if you want (optional)
    if "modality" in df.columns:
        df = df[df["modality"].astype(str).str.lower().str.contains("dermo") | df["modality"].isna()]

    # Filter rows that actually have a filepath
    df = df[df["filepath"].astype(str).str.len() > 0].copy()

    if args.drop_unknown:
        df = df[df["label"].astype(str) != "unknown"].copy()

    # Keep tidy columns for the trainer
    keep = ["isic_id", "filepath", "label", "diagnosis", "benign_malignant",
            "sex", "age_approx", "anatom_site_general", "modality", "source"]
    for k in keep:
        if k not in df.columns:
            df[k] = ""

    Path(Path(args.out_csv).parent).mkdir(parents=True, exist_ok=True)
    df[keep].to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} with {len(df)} rows")
    print("Label distribution:")
    print(df["label"].value_counts(dropna=False).sort_index())

if __name__ == "__main__":
    main()
