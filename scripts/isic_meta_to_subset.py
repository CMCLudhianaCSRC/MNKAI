#!/usr/bin/env python
import argparse, logging, os, subprocess, sys, re, random, shutil
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

CLASS_ORDER = ["mel", "nv", "bcc", "scc", "df", "bkl", "ak", "vasc"]
CLASS_MAP = {k: i for i, k in enumerate(CLASS_ORDER)}
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
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()) if isinstance(s, str) else ""
def to_fine_label(diag: str) -> str:
    s = norm(diag)
    if not s: return ""
    for k, vals in ALIASES.items():
        for v in vals:
            if s == v or v in s: return k
    if "seborrheic keratosis" in s or "sebk" in s: return "bkl"
    if any(v in s for v in ["angioma","hemangioma","vascular"]): return "vasc"
    return ""
def coarse_from_any(row: pd.Series, fine: str) -> str:
    for key in row.index:
        kl = key.lower()
        if "benign_malignant" in kl:
            val = norm(str(row[key]))
            if val in ("benign","malignant"): return val
        if kl.endswith(".target") or kl == "target":
            val = str(row[key]).strip()
            if val in ("1","1.0"): return "malignant"
            if val in ("0","0.0"): return "benign"
    if fine in ("mel","bcc","scc"): return "malignant"
    if fine in ("nv","df","bkl","ak","vasc"): return "benign"
    return ""
def setup_log(log_file: str, level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")])
def which_isic(user_path: Optional[str]) -> str:
    if user_path and Path(user_path).exists(): return str(Path(user_path))
    for name in ("isic.exe","isic"):
        hit = shutil.which(name)
        if hit: return hit
    raise SystemExit("isic CLI not found. Put isic.exe in PATH or pass --isic-exe.")
def run_isic_meta(isic_bin: str, collection_id: str, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists() and out_csv.stat().st_size > 0:
        logging.info(f"[skip] {collection_id} -> {out_csv.name} exists")
        return
    # Try --outfile (newer) then --out (older) for compatibility
    cmd1 = [isic_bin, "metadata", "download", "--collections", str(collection_id), "--outfile", str(out_csv)]
    cmd2 = [isic_bin, "metadata", "download", "--collections", str(collection_id), "--out", str(out_csv)]
    for cmd in (cmd1, cmd2):
        logging.info(f"[isic] {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as e:
            logging.warning(f"[isic] failed (exit {e.returncode}) with flag {cmd[-2]}; trying alternative...")
    raise SystemExit(f"[isic] both --outfile and --out failed for collection {collection_id}")
def normalize_isic_id(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.lower() in ("isic_id","isic id","name","id")]
    if not cols: raise SystemExit("Metadata CSV has no isic_id/name/id column.")
    c = cols[0]
    df["isic_id"] = df[c].astype(str)
    df["isic_id"] = df["isic_id"].str.replace(r"[^A-Za-z0-9_]", "", regex=True)
    df["isic_id"] = df["isic_id"].apply(lambda s: s if s.startswith("ISIC_") else (f"ISIC_{s}" if s.isdigit() else s))
    return df
def pick_first_nonempty(row: pd.Series, keys_pref: List[str]) -> str:
    for k in keys_pref:
        if k in row and isinstance(row[k], str) and row[k].strip(): return row[k]
        if k in row and pd.notna(row[k]) and str(row[k]).strip(): return str(row[k])
    for k in row.index:
        if "diagnosis" in k.lower():
            val = str(row[k]).strip()
            if val: return val
    return ""
def build_labels(meta: pd.DataFrame, label_strategy: str, dermo_only: bool) -> pd.DataFrame:
    df = meta.copy()
    df.columns = [c.strip() for c in df.columns]
    df = normalize_isic_id(df)
    diag_keys_pref = [
        "diagnosis_3","diagnosis 3","metadata.clinical.diagnosis_3",
        "diagnosis_2","diagnosis 2","metadata.clinical.diagnosis_2",
        "diagnosis","metadata.clinical.diagnosis","dx"
    ]
    df["diagnosis_norm"] = df.apply(lambda r: pick_first_nonempty(r, diag_keys_pref), axis=1).astype(str)
    df["fine_label"] = df["diagnosis_norm"].apply(to_fine_label)
    df["coarse_label"] = df.apply(lambda r: coarse_from_any(r, r["fine_label"]), axis=1)
    if dermo_only:
        mod_cols = [c for c in df.columns if any(k in c.lower() for k in ["image_type","modality","acquisition.image_type","metadata.acquisition.image_type","metadata.image_type"])]
        if mod_cols:
            mods = df[mod_cols].astype(str).agg(lambda s: " ".join([x.lower() for x in s.values]), axis=1)
            df = df[mods.str.contains("dermo") | mods.eq("")]
    if label_strategy == "fine":
        df["label_name"] = df["fine_label"]
        df["label_id"] = df["label_name"].map(CLASS_MAP)
    else:
        df["label_name"] = df["coarse_label"]
        df["label_id"] = df["label_name"].map({"benign": 0, "malignant": 1})
    return df
def join_with_manifest(paths_csv: Path, labeled_meta: pd.DataFrame, drop_unlabeled: bool, skip_missing_files: bool) -> pd.DataFrame:
    manifest = pd.read_csv(paths_csv)
    if "isic_id" not in manifest.columns: raise SystemExit("manifest must have 'isic_id'")
    if "filepath" not in manifest.columns: raise SystemExit("manifest must have 'filepath'")
    manifest["isic_id"] = manifest["isic_id"].astype(str)
    merged = manifest.merge(labeled_meta, on="isic_id", how="left")
    unlabeled_mask = merged["label_name"].isna() | (merged["label_name"].astype(str).str.len() == 0)
    if drop_unlabeled:
        merged = merged[~unlabeled_mask].copy()
    if skip_missing_files:
        merged = merged[merged["filepath"].astype(str).apply(lambda p: Path(p).exists())]
    keep = ["isic_id","filepath","label_name","label_id","diagnosis_norm","coarse_label","fine_label"]
    extras = [c for c in ("sex","age_approx","anatom_site_general") if c in merged.columns]
    merged = merged[keep+extras].reset_index(drop=True)
    return merged
def balanced_subset(df: pd.DataFrame, total: int, label_col="label_name", seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    classes = df[label_col].dropna().unique().tolist()
    k = len(classes)
    total = min(total, len(df)) if total > 0 else len(df)
    base = total // k
    picks = []
    for cls in classes:
        d = df[df[label_col] == cls]
        n = min(len(d), base)
        if n > 0: picks.append(d.sample(n=n, random_state=seed))
    picked = pd.concat(picks, ignore_index=True) if picks else df.iloc[0:0]
    remain = total - len(picked)
    if remain > 0:
        pool = df.loc[~df.index.isin(picked.index)]
        sizes = pool.groupby(label_col).size()
        if sizes.sum() > 0:
            probs = sizes / sizes.sum()
            alloc = (probs * remain).round().astype(int)
            diff = remain - alloc.sum()
            if diff != 0:
                residuals = (probs * remain) - alloc
                order = residuals.sort_values(ascending=(diff < 0)).index.tolist()
                for cls in order[:abs(diff)]:
                    alloc[cls] += 1 if diff > 0 else -1
            parts = []
            for cls, n in alloc.items():
                if n > 0:
                    g = pool[pool[label_col] == cls]
                    parts.append(g.sample(n=min(n, len(g)), random_state=seed))
            if parts:
                picked = pd.concat([picked] + parts, ignore_index=True)
    return picked.sample(frac=1.0, random_state=seed).reset_index(drop=True).head(total)
def load_collections_arg(arg: str, file_path: Optional[str]) -> List[str]:
    if file_path:
        ids = [l.strip() for l in Path(file_path).read_text(encoding="utf-8").splitlines() if l.strip()]
        return ids
    return [s.strip() for s in str(arg).split(",") if s.strip()]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collections", required=True, help="Comma-separated IDs (e.g. 70,65,212) or 'file'")
    ap.add_argument("--collections-file", default=None, help="Text file with one collection ID per line")
    ap.add_argument("--ids_csv", required=True, help="manifest_unified.csv (has isic_id, filepath)")
    ap.add_argument("--out_meta", default="data/isic_meta_ALL_LABELED.csv")
    ap.add_argument("--out_labeled", default="data/manifest_unified_labeled.csv")
    ap.add_argument("--subset_out", default="data/manifest_subset_120k.csv")
    ap.add_argument("--subset_total", type=int, default=120000)
    ap.add_argument("--label_strategy", choices=["fine","coarse"], default="fine")
    ap.add_argument("--dermo_only", action="store_true", default=True)
    ap.add_argument("--drop_unlabeled", action="store_true", default=True)
    ap.add_argument("--skip_missing_files", action="store_true", default=True)
    ap.add_argument("--isic-exe", default=None)
    ap.add_argument("--workdir", default="data/isic_meta_parts")
    ap.add_argument("--log-file", default="logs/isic_meta_to_subset.log")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    setup_log(args.log_file, args.log_level)
    random.seed(args.seed); np.random.seed(args.seed)

    isic_bin = which_isic(args.isic_exe)
    logging.info(f"Using isic CLI: {isic_bin}")

    ids = load_collections_arg(args.collections, args.collections_file)
    if not ids: raise SystemExit("No collection IDs provided.")
    parts_dir = Path(args.workdir); parts_dir.mkdir(parents=True, exist_ok=True)

    # 1) Fetch metadata for each collection
    for cid in ids:
        run_isic_meta(isic_bin, cid, parts_dir / f"meta_{cid}.csv")

    # 2) Merge all metadata parts
    frames = []
    for p in sorted(parts_dir.glob("meta_*.csv")):
        try:
            dfp = pd.read_csv(p, low_memory=False)
            frames.append(dfp)
            logging.info(f"[merge] {p.name}: {len(dfp)} rows")
        except Exception as e:
            logging.warning(f"[merge] failed to read {p}: {e}")
    if not frames: raise SystemExit("No metadata CSVs read; aborting.")
    meta = pd.concat(frames, ignore_index=True).drop_duplicates()

    # 3) Build labels
    labeled_meta = build_labels(meta, label_strategy=args.label_strategy, dermo_only=args.dermo_only)
    labeled_meta[["isic_id","diagnosis_norm","fine_label","coarse_label","sex","age_approx","anatom_site_general"]].to_csv(args.out_meta, index=False)
    logging.info(f"[meta] combined labeled metadata -> {args.out_meta} (rows={len(labeled_meta)})")

    # 4) Join with your manifest
    labeled_manifest = join_with_manifest(Path(args.ids_csv), labeled_meta, args.drop_unlabeled, args.skip_missing_files)
    labeled_manifest.to_csv(args.out_labeled, index=False)
    logging.info(f"[join] labeled manifest -> {args.out_labeled} (rows={len(labeled_manifest)})")

    # Coverage diagnostic
    try:
        manifest_rows = len(pd.read_csv(args.ids_csv))
        logging.info(f"[coverage] manifest rows={manifest_rows} | labeled in-join={len(labeled_manifest)}")
    except Exception:
        pass
    logging.info("Class counts:\n" + str(labeled_manifest["label_name"].value_counts(dropna=False)))

    # 5) Balanced subset
    want = args.subset_total
    have = len(labeled_manifest)
    if have < want:
        logging.warning(f"[subset] Requested {want} but only {have} labeled available. Using {have}.")
        want = have
    subset = balanced_subset(labeled_manifest, total=want, label_col="label_name", seed=args.seed)
    subset.to_csv(args.subset_out, index=False)
    logging.info(f"[subset] {want} rows -> {args.subset_out}")
    logging.info("Subset class counts:\n" + str(subset["label_name"].value_counts(dropna=False)))
    logging.info("✅ Done.")
if __name__ == "__main__":
    main()
