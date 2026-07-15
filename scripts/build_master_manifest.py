"""Build the unified master manifest from ALL data sources.

Sources:
  1. D:/AIFINAL/images/isic          – filtered 8-class ISIC (29,512 labeled rows, 29,954 on disk)
  2. E:/AIFINAL/fine_pool            – pre-labeled 8-class pool (19,074 images)
  3. E:/AIFINAL/labeled_images       – flat ISIC dump (66,737; cross-ref via isic_combined)
  4. E:/AIFINAL/images               – numbered shards (121,729; cross-ref via isic_combined)
  4b. isic_extra_labels.csv         – 51,631 E: images labelled via ISIC API Apr 2026
                                       (40,843 nv + 10,857 mel; in shards or labeled_images)
  5. D:/AIFINAL/images/dermacon_in   – Indian skin 5,450 imgs (FST 3-6 diversity; kept separate)
  6. D:/AIFINAL/images/fitz17k       – Fitzpatrick17k (downloading; supplemental FST diversity)
  7. D:/AIFINAL/images/isic_minority – Newly downloaded minority class images (ak/scc/df/vasc)

Output: data/manifests/master.csv
"""
from __future__ import annotations

import sys, hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from loguru import logger
from mnkai.data.manifest import LabelNormalizer, DEFAULT_CLASSES

logger.add("logs/build_master_manifest.log", rotation="10 MB")

# ─── constants ───────────────────────────────────────────────────────────────
D_ISIC       = Path(r"D:\AIFINAL\images\isic")
E_FINE_POOL  = Path(r"E:\AIFINAL\fine_pool")
E_LBL_IMGS   = Path(r"E:\AIFINAL\labeled_images")
E_SHARDS     = Path(r"E:\AIFINAL\images")
D_DERMACON   = Path(r"D:\AIFINAL\images\dermacon_in")
D_FITZ17K    = Path(r"D:\AIFINAL\images\fitz17k")
D_MINORITY   = Path(r"D:\AIFINAL\images\isic_minority")

MANIFEST_DIR = Path("data/manifests")
OUT_MASTER   = MANIFEST_DIR / "master.csv"
CLASSES      = DEFAULT_CLASSES


def _norm_label(norm: LabelNormalizer, raw: str) -> str | None:
    return norm.map(str(raw)) if pd.notna(raw) and str(raw).strip() else None


# ──────────────────────────────────────────────────────────────────────────────
# SOURCE 1: D:\AIFINAL\images\isic  (filtered 8-class, best quality)
# ──────────────────────────────────────────────────────────────────────────────
def load_d_isic() -> pd.DataFrame:
    meta = MANIFEST_DIR / "isic_filtered.csv"
    if not meta.exists():
        logger.warning(f"[src1] {meta} missing – skipping"); return pd.DataFrame()
    df = pd.read_csv(meta, low_memory=False)
    norm = LabelNormalizer()

    # Build filepath: D:\AIFINAL\images\isic\ISIC_xxxxxxx.jpg
    isic_files: dict[str, str] = {}
    if D_ISIC.exists():
        for p in D_ISIC.rglob("*.jpg"):
            isic_files[p.stem] = str(p)
    logger.info(f"[src1] D: isic on-disk files: {len(isic_files):,}")

    rows = []
    for _, r in df.iterrows():
        iid = str(r["isic_id"])
        fp  = isic_files.get(iid, "")
        lbl = str(r.get("label_name", "")) if pd.notna(r.get("label_name", "")) else ""
        if not lbl:
            lbl = _norm_label(norm, r.get("diagnosis_3", "")) or ""
        if not fp or not lbl:
            continue
        rows.append({
            "isic_id": iid, "filepath": fp, "label_name": lbl,
            "source": "ISIC_D",
            "group_id": str(r.get("lesion_id", "") or iid),
            "fitzpatrick": r.get("fitzpatrick", None),
            "sex": r.get("sex", None),
            "age_approx": r.get("age_approx", None),
        })
    out = pd.DataFrame(rows)
    logger.info(f"[src1] D: isic → {len(out):,} rows | {out['label_name'].value_counts().to_dict()}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# SOURCE 2: E:\AIFINAL\fine_pool  (pre-labeled by class subdirectory)
# ──────────────────────────────────────────────────────────────────────────────
def load_e_fine_pool(existing_ids: set) -> pd.DataFrame:
    if not E_FINE_POOL.exists():
        logger.warning("[src2] E:\\fine_pool missing"); return pd.DataFrame()
    rows = []
    for cls_dir in sorted(E_FINE_POOL.iterdir()):
        if not cls_dir.is_dir(): continue
        label = cls_dir.name.lower()
        if label not in CLASSES:
            logger.warning(f"[src2] unknown class dir: {cls_dir.name}"); continue
        for img in cls_dir.rglob("*"):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"): continue
            iid = img.stem
            if iid in existing_ids: continue  # already in D: source
            rows.append({
                "isic_id": iid, "filepath": str(img), "label_name": label,
                "source": "ISIC_E_POOL",
                "group_id": iid,
                "fitzpatrick": None, "sex": None, "age_approx": None,
            })
    out = pd.DataFrame(rows) if rows else pd.DataFrame()
    if len(out):
        logger.info(f"[src2] E: fine_pool → {len(out):,} rows | {out['label_name'].value_counts().to_dict()}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# SOURCE 3: E:\AIFINAL\labeled_images  (flat ISIC dump, cross-ref via combined)
# ──────────────────────────────────────────────────────────────────────────────
def load_e_labeled_images(existing_ids: set) -> pd.DataFrame:
    if not E_LBL_IMGS.exists():
        logger.warning("[src3] E:\\labeled_images missing"); return pd.DataFrame()

    combined_path = MANIFEST_DIR / "isic_combined.csv"
    if not combined_path.exists():
        logger.warning("[src3] isic_combined.csv missing"); return pd.DataFrame()

    combined = pd.read_csv(combined_path, low_memory=False)
    norm = LabelNormalizer()
    # build label lookup
    lbl_lookup: dict[str, str] = {}
    for _, r in combined.iterrows():
        lbl = _norm_label(norm, r.get("diagnosis_3", ""))
        if lbl:
            lbl_lookup[str(r["isic_id"])] = lbl

    rows = []
    for img in E_LBL_IMGS.iterdir():
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"): continue
        iid = img.stem
        if iid in existing_ids: continue
        lbl = lbl_lookup.get(iid)
        if not lbl: continue
        rows.append({
            "isic_id": iid, "filepath": str(img), "label_name": lbl,
            "source": "ISIC_E_LBL",
            "group_id": iid,
            "fitzpatrick": None, "sex": None, "age_approx": None,
        })
    out = pd.DataFrame(rows) if rows else pd.DataFrame()
    if len(out):
        logger.info(f"[src3] E: labeled_images → {len(out):,} rows | {out['label_name'].value_counts().to_dict()}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# SOURCE 4: E:\AIFINAL\images (numbered shards, cross-ref via combined)
# ──────────────────────────────────────────────────────────────────────────────
def load_e_shards(existing_ids: set) -> pd.DataFrame:
    if not E_SHARDS.exists():
        logger.warning("[src4] E:\\images missing"); return pd.DataFrame()

    combined_path = MANIFEST_DIR / "isic_combined.csv"
    if not combined_path.exists():
        return pd.DataFrame()

    combined = pd.read_csv(combined_path, low_memory=False)
    norm = LabelNormalizer()
    lbl_lookup: dict[str, str] = {}
    meta_lookup: dict[str, dict] = {}
    for _, r in combined.iterrows():
        iid = str(r["isic_id"])
        lbl = _norm_label(norm, r.get("diagnosis_3", ""))
        if lbl:
            lbl_lookup[iid] = lbl
            meta_lookup[iid] = {
                "fitzpatrick": r.get("fitzpatrick"),
                "sex": r.get("sex"),
                "age_approx": r.get("age_approx"),
                "source_attr": str(r.get("attribution", "") or ""),
            }

    logger.info(f"[src4] scanning E:\\images shards for new 8-class images...")
    rows = []
    for p in E_SHARDS.rglob("*.jpg"):
        iid = p.stem
        if iid in existing_ids: continue
        lbl = lbl_lookup.get(iid)
        if not lbl: continue
        meta = meta_lookup.get(iid, {})
        # detect if MSK (external test domain)
        src_attr = meta.get("source_attr", "")
        source = "ISIC_E_MSK" if "Memorial Sloan Kettering" in src_attr else "ISIC_E_SHARD"
        rows.append({
            "isic_id": iid, "filepath": str(p), "label_name": lbl,
            "source": source,
            "group_id": iid,
            "fitzpatrick": meta.get("fitzpatrick"),
            "sex": meta.get("sex"),
            "age_approx": meta.get("age_approx"),
        })
    out = pd.DataFrame(rows) if rows else pd.DataFrame()
    if len(out):
        logger.info(f"[src4] E: shards → {len(out):,} rows | {out['label_name'].value_counts().to_dict()}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# SOURCE 4b: isic_extra_labels.csv  — 51k E: drive images labelled Apr 2026
#   via ISIC API batch fetch (_fetch_isic_labels.py).
#   All 51,700 IDs returned labels: 40,843 nv + 10,857 mel.
#   Physical files can live under E:\AIFINAL\images (shards)
#   OR E:\AIFINAL\labeled_images (flat dump) — scan both.
# ──────────────────────────────────────────────────────────────────────────────
def load_e_extra_labels(existing_ids: set) -> pd.DataFrame:
    extra_csv = MANIFEST_DIR / "isic_extra_labels.csv"
    if not extra_csv.exists():
        logger.warning("[src4b] isic_extra_labels.csv not found"); return pd.DataFrame()

    df = pd.read_csv(extra_csv, low_memory=False)
    norm = LabelNormalizer()

    # Build filepath scan of ALL E:\AIFINAL sub-directories where images live
    # (labeled_images = flat dump, images = numbered shards)
    logger.info("[src4b] scanning E: drive for extra-label file paths (shards + labeled_images)...")
    e_file_map: dict[str, str] = {}
    for scan_dir in [E_SHARDS, E_LBL_IMGS]:
        if scan_dir.exists():
            for p in scan_dir.rglob("*.jpg"):
                if p.stem not in e_file_map:   # shards take priority if duplicate
                    e_file_map[p.stem] = str(p)
    logger.info(f"[src4b] E: combined scan: {len(e_file_map):,} files")

    rows = []
    for _, r in df.iterrows():
        iid = str(r["isic_id"])
        if iid in existing_ids:
            continue
        lbl = _norm_label(norm, r.get("diagnosis_3", ""))
        if not lbl:
            continue
        fp = e_file_map.get(iid, "")
        if not fp:
            continue
        rows.append({
            "isic_id":    iid,
            "filepath":   fp,
            "label_name": lbl,
            "source":     "ISIC_E_EXTRA",
            "group_id":   iid,
            "fitzpatrick": r.get("fitzpatrick", None),
            "sex":         r.get("sex", None),
            "age_approx":  r.get("age_approx", None),
        })

    out = pd.DataFrame(rows) if rows else pd.DataFrame()
    if len(out):
        logger.info(f"[src4b] E: extra labels -> {len(out):,} rows | "
                    f"{out['label_name'].value_counts().to_dict()}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# SOURCE 7: D:\AIFINAL\images\isic_minority  (newly downloaded minority classes)
# ──────────────────────────────────────────────────────────────────────────────
def load_minority(existing_ids: set) -> pd.DataFrame:
    minority_csv = MANIFEST_DIR / "isic_minority.csv"
    if not minority_csv.exists() or minority_csv.stat().st_size < 100:
        logger.info("[src7] isic_minority.csv not yet available"); return pd.DataFrame()

    df = pd.read_csv(minority_csv, low_memory=False)
    if "label_name" not in df.columns or len(df) == 0:
        return pd.DataFrame()

    rows = []
    for _, r in df.iterrows():
        iid = str(r["isic_id"])
        if iid in existing_ids:
            continue
        fp = str(r.get("filepath", ""))
        if not fp or not Path(fp).exists():
            continue
        lbl = str(r.get("label_name", ""))
        if lbl not in CLASSES:
            continue
        rows.append({
            "isic_id":    iid,
            "filepath":   fp,
            "label_name": lbl,
            "source":     "ISIC_MINORITY",
            "group_id":   iid,
            "fitzpatrick": r.get("fitzpatrick", None),
            "sex":         r.get("sex", None),
            "age_approx":  r.get("age_approx", None),
        })

    out = pd.DataFrame(rows) if rows else pd.DataFrame()
    if len(out):
        logger.info(f"[src7] minority downloads -> {len(out):,} rows | "
                    f"{out['label_name'].value_counts().to_dict()}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# SOURCE 8: external_datasets.csv  — HAM10000 supplement, MED-NODE, etc.
# ──────────────────────────────────────────────────────────────────────────────
def load_external_datasets(existing_ids: set) -> pd.DataFrame:
    ext_csv = MANIFEST_DIR / "external_datasets.csv"
    if not ext_csv.exists() or ext_csv.stat().st_size < 100:
        logger.info("[src8] external_datasets.csv not yet available"); return pd.DataFrame()

    df = pd.read_csv(ext_csv, low_memory=False)
    if len(df) == 0 or "isic_id" not in df.columns:
        return pd.DataFrame()

    rows = []
    for _, r in df.iterrows():
        iid = str(r["isic_id"])
        if iid in existing_ids:
            continue
        fp = str(r.get("filepath", ""))
        if not fp or not Path(fp).exists():
            continue
        lbl = str(r.get("label_name", ""))
        if lbl not in CLASSES:
            continue
        rows.append({
            "isic_id":    iid,
            "filepath":   fp,
            "label_name": lbl,
            "source":     str(r.get("source", "EXTERNAL")),
            "group_id":   str(r.get("group_id", iid)),
            "fitzpatrick": r.get("fitzpatrick", None),
            "sex":         r.get("sex", None),
            "age_approx":  r.get("age_approx", None),
        })

    out = pd.DataFrame(rows) if rows else pd.DataFrame()
    if len(out):
        logger.info(f"[src8] external -> {len(out):,} rows | "
                    f"sources={out['source'].value_counts().to_dict()}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# SOURCE 9: kaggle_external.csv  — PAD-UFES-20, ISIC2019, Derm7pt, BCN20000
# ──────────────────────────────────────────────────────────────────────────────
def load_kaggle_external(existing_ids: set) -> pd.DataFrame:
    kag_csv = MANIFEST_DIR / "kaggle_external.csv"
    if not kag_csv.exists() or kag_csv.stat().st_size < 100:
        logger.info("[src9] kaggle_external.csv not yet available"); return pd.DataFrame()

    df = pd.read_csv(kag_csv, low_memory=False)
    if len(df) == 0 or "isic_id" not in df.columns:
        return pd.DataFrame()

    # Fast vectorised filtering — integrators already verified files exist,
    # so skip per-row Path.exists() checks (~30 min on USB drive for 200k rows).
    df = df[~df["isic_id"].astype(str).isin(existing_ids)]
    df = df[df["label_name"].astype(str).isin(CLASSES)]
    df = df[df["filepath"].astype(str).str.len() > 0]

    out = pd.DataFrame({
        "isic_id":     df["isic_id"].astype(str),
        "filepath":    df["filepath"].astype(str),
        "label_name":  df["label_name"].astype(str),
        "source":      df.get("source", "KAGGLE").astype(str) if "source" in df.columns else "KAGGLE",
        "group_id":    df.get("group_id", df["isic_id"]).astype(str) if "group_id" in df.columns else df["isic_id"].astype(str),
        "fitzpatrick": df.get("fitzpatrick"),
        "sex":         df.get("sex"),
        "age_approx":  df.get("age_approx"),
        "weak_label":  df.get("weak_label", False).fillna(False).astype(bool) if "weak_label" in df.columns else False,
    })

    if len(out):
        n_weak = int(out["weak_label"].sum()) if "weak_label" in out.columns else 0
        logger.info(f"[src9] kaggle external -> {len(out):,} rows "
                    f"({n_weak:,} weak-labelled) | "
                    f"sources={out['source'].value_counts().to_dict()}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# SOURCE 5: DermaCon-IN  (Indian skin, FST 3-6 diversity)
# ──────────────────────────────────────────────────────────────────────────────
def load_dermacon() -> pd.DataFrame:
    meta_tab = None
    for cand in D_DERMACON.rglob("Skin_Metadata*"):
        meta_tab = cand; break
    if meta_tab is None:
        logger.warning("[src5] DermaCon Skin_Metadata.tab not found"); return pd.DataFrame()

    df = pd.read_csv(meta_tab, sep="\t")
    # build image file lookup
    img_lookup: dict[str, str] = {}
    for p in D_DERMACON.rglob("*.jpg"):
        img_lookup[p.stem] = str(p)
        img_lookup[p.name] = str(p)
    logger.info(f"[src5] DermaCon images on disk: {len(img_lookup)//2:,}")

    rows = []
    for _, r in df.iterrows():
        iname = str(r.get("Image_name", "") or "").strip()
        fp = img_lookup.get(iname) or img_lookup.get(Path(iname).stem, "")
        if not fp: continue
        disease = str(r.get("Disease_label", "") or "").strip()
        main_cls = str(r.get("Main_class", "") or "").strip()
        fst_raw = str(r.get("Fitzpatrick", "") or "").strip()
        fst = fst_raw.replace("FST ", "") if fst_raw.startswith("FST") else None
        rows.append({
            "isic_id": Path(fp).stem,
            "filepath": fp,
            "label_name": disease,            # keep original disease label (not 8-class)
            "dermacon_main_class": main_cls,
            "source": "DERMACON_IN",
            "group_id": str(r.get("Subject_ID", Path(fp).stem)),
            "fitzpatrick": fst,
            "sex": r.get("Sex", None),
            "age_approx": r.get("Age", None),
        })
    out = pd.DataFrame(rows) if rows else pd.DataFrame()
    logger.info(f"[src5] DermaCon-IN → {len(out):,} rows | FST dist: {out['fitzpatrick'].value_counts().to_dict()}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# SOURCE 6: Fitzpatrick17k HF  (if already downloaded)
# ──────────────────────────────────────────────────────────────────────────────
def load_fitz17k() -> pd.DataFrame:
    hf_meta = MANIFEST_DIR / "fitz17k_hf.csv"
    if not hf_meta.exists():
        logger.info("[src6] fitz17k_hf.csv not yet available (still downloading)"); return pd.DataFrame()
    df = pd.read_csv(hf_meta, low_memory=False)
    logger.info(f"[src6] fitz17k_hf: {len(df)} rows, cols={list(df.columns)}")

    # keep rows where the image file actually exists
    df = df[df["filepath"].apply(lambda p: Path(str(p)).exists())]
    df["source"] = "FITZ17K"
    if "fitzpatrick_scale" in df.columns:
        df = df.rename(columns={"fitzpatrick_scale": "fitzpatrick"})
    logger.info(f"[src6] Fitz17k with existing files: {len(df):,}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    logger.info("=" * 70)
    logger.info("BUILD MASTER MANIFEST")
    logger.info("=" * 70)

    # ── collect 8-class ISIC sources ─────────────────────────────────────────
    src1 = load_d_isic()
    seen_ids: set = set(src1["isic_id"].tolist()) if len(src1) else set()

    src2 = load_e_fine_pool(seen_ids)
    if len(src2): seen_ids.update(src2["isic_id"].tolist())

    src3 = load_e_labeled_images(seen_ids)
    if len(src3): seen_ids.update(src3["isic_id"].tolist())

    src4 = load_e_shards(seen_ids)
    if len(src4): seen_ids.update(src4["isic_id"].tolist())

    # Source 4b: newly-labelled E: drive images (Apr 2026 ISIC API fetch)
    src4b = load_e_extra_labels(seen_ids)
    if len(src4b): seen_ids.update(src4b["isic_id"].tolist())

    # Source 7: freshly downloaded minority class images
    src7 = load_minority(seen_ids)
    if len(src7): seen_ids.update(src7["isic_id"].tolist())

    # Source 8: external datasets (HAM10000 supp, MED-NODE, PH2 if downloaded, etc.)
    src8 = load_external_datasets(seen_ids)
    if len(src8): seen_ids.update(src8["isic_id"].tolist())

    # Source 9: Kaggle datasets (PAD-UFES-20, ISIC 2019, Derm7pt, BCN20000)
    src9 = load_kaggle_external(seen_ids)
    if len(src9): seen_ids.update(src9["isic_id"].tolist())

    # ── diversity sources (kept with original labels / separate source tag) ──
    src5 = load_dermacon()
    src6 = load_fitz17k()

    # ── combine ───────────────────────────────────────────────────────────────
    canonical_cols = ["isic_id", "filepath", "label_name", "source",
                      "group_id", "fitzpatrick", "sex", "age_approx"]
    parts = []
    for df, name in [(src1,"D:ISIC"), (src2,"E:fine_pool"), (src3,"E:lbl_imgs"),
                     (src4,"E:shards"), (src4b,"E:extra"), (src7,"minority"),
                     (src8,"external"), (src9,"kaggle"),
                     (src5,"DermaCon"), (src6,"Fitz17k")]:
        if not len(df): continue
        for c in canonical_cols:
            if c not in df.columns: df[c] = None
        parts.append(df[canonical_cols + [c for c in df.columns if c not in canonical_cols]])

    master = pd.concat(parts, ignore_index=True)
    master.to_csv(OUT_MASTER, index=False)

    logger.info("=" * 70)
    logger.info(f"MASTER MANIFEST: {len(master):,} total rows → {OUT_MASTER}")
    logger.info("─" * 70)

    # ── summary table ─────────────────────────────────────────────────────────
    sep = "=" * 70
    div = "-" * 70
    print("\n" + sep)
    print("MASTER MANIFEST SUMMARY")
    print(sep)
    print(f"Total rows   : {len(master):,}")
    print(f"Output CSV   : {OUT_MASTER}")
    print()

    # per-source
    print("-- Rows by source " + "-" * 52)
    print(master["source"].value_counts().to_string())
    print()

    # 8-class only rows
    isic_cls = master[master["label_name"].isin(CLASSES)]
    print(f"-- 8-class rows only (ISIC taxonomy): {len(isic_cls):,} " + "-" * 28)
    print(isic_cls["label_name"].value_counts().to_string())
    print()

    # FST distribution
    print("-- Fitzpatrick skin type distribution (all sources) " + "-" * 19)
    fst_all = master["fitzpatrick"].value_counts(dropna=False)
    print(fst_all.to_string())
    print()

    # source x class cross-tab
    print("-- 8-class images by source " + "-" * 43)
    ct = isic_cls.groupby(["source", "label_name"]).size().unstack(fill_value=0)
    ct["TOTAL"] = ct.sum(axis=1)
    print(ct.to_string())
    print()

    # DermaCon breakdown
    if len(src5):
        print("-- DermaCon-IN top disease labels " + "-" * 37)
        print(src5["label_name"].value_counts().head(15).to_string())
        print()
        print("-- DermaCon-IN FST distribution " + "-" * 39)
        print(src5["fitzpatrick"].value_counts(dropna=False).to_string())

    print(sep)
    return master


if __name__ == "__main__":
    main()
