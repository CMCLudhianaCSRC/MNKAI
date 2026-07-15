"""Build the CLINICAL-PHOTO binary cancer dataset v2 — the "all 3" fix for DDI:
  (1) CleanPatrick-cleaned Fitzpatrick17k (drop label-errors / off-topic / near-dups),
  (2) DDI folded into 5-fold CROSS-VALIDATION (the only clean dark-skin cancer data —
      trained + tested via CV instead of held fully out),
  (3) PAD-UFES clinical cancers.

Binary: malignant vs benign (AK / non-neoplastic -> benign). 5-fold group-disjoint CV;
DDI folds stratified by (skin_tone x malignant) so each held-out fold is representative.

    python scripts/_build_cancer_data.py
"""
from __future__ import annotations
import re, glob, os
from pathlib import Path
import numpy as np
import pandas as pd

FITZ_LABELS = "data/manifests/fitzpatrick17k.csv"
FITZ_IMG_DIR = "D:/AIFINAL/downloads/fitz17k_full"
CLEANPATRICK = "D:/AIFINAL/downloads/cleanpatrick/combined_annotations.csv"
MASTER = "data/manifests/master.csv"
DDI_DIR = "D:/AIFINAL/images/ddi"
OUT_MANIFEST = "data/manifests/cancer_cv.csv"
OUT_SPLITS = "data/splits/cancer_cv_splits.csv"
NONNEO_CAP = 2500
K = 5
SEED = 0


def fst_group(f):
    try:
        f = int(float(f)); return "I-III" if f <= 3 else "IV-VI"
    except Exception:
        return "unk"


def build_fitz():
    lab = pd.read_csv(FITZ_LABELS, low_memory=False).set_index("md5hash")
    cp = pd.read_csv(CLEANPATRICK).set_index("md5hash")   # label_error, off_topic_sample, nd_clique
    imgs = {Path(p).stem: p for p in glob.glob(FITZ_IMG_DIR + "/**/*.jpg", recursive=True)}
    seen_clique = set(); rows = []
    for md5, fp in imgs.items():
        if md5 not in lab.index or md5 not in cp.index:
            continue
        c = cp.loc[md5]
        if int(c.get("label_error", 0)) == 1 or int(c.get("off_topic_sample", 0)) == 1:
            continue                                        # CleanPatrick: drop noisy/off-topic
        clq = int(c.get("nd_clique", 0))
        if clq > 0:                                         # near-duplicate: keep only first of clique
            if clq in seen_clique:
                continue
            seen_clique.add(clq)
        tp = str(lab.loc[md5].get("three_partition_label", ""))
        y = {"malignant": "malignant", "benign": "benign", "non-neoplastic": "nonneo"}.get(tp)
        if y is None:
            continue
        rows.append({"isic_id": f"FITZ_{md5[:10]}", "filepath": fp, "raw": y, "source": "FITZ17K_CLEAN",
                     "group_id": f"FITZ_{md5}", "fitzpatrick": lab.loc[md5].get("fitzpatrick_scale", np.nan)})
    df = pd.DataFrame(rows)
    non = df[df.raw == "nonneo"].sample(min(NONNEO_CAP, (df.raw == "nonneo").sum()), random_state=SEED)
    df = pd.concat([df[df.raw != "nonneo"], non], ignore_index=True)
    df["label_name"] = np.where(df.raw == "malignant", "malignant", "benign")
    print(f"[fitz] after CleanPatrick: {len(df)} (dropped label-errors/off-topic/near-dups)", flush=True)
    return df.drop(columns="raw")


def build_pad():
    m = pd.read_csv(MASTER, low_memory=False)
    p = m[m["source"] == "PAD-UFES-20"].copy()
    p["label_name"] = np.where(p["label_name"].isin({"bcc", "scc", "mel"}), "malignant", "benign")
    p["group_id"] = p["filepath"].map(lambda fp: "PAD_" + (re.match(r"(PAT_\d+_\d+)", Path(str(fp)).stem) or
                                                            re.match(r"(.+)", Path(str(fp)).stem)).group(1))
    p["isic_id"] = "PAD_" + p.index.astype(str)
    return p[["isic_id", "filepath", "label_name", "source", "group_id", "fitzpatrick"]]


def build_ddi():
    meta = pd.read_csv(Path(DDI_DIR) / "ddi_metadata.csv")
    tone_fst = {12: 2, 34: 4, 56: 6}
    return pd.DataFrame({
        "isic_id": "DDI_" + meta["DDI_ID"].astype(str),
        "filepath": [str(Path(DDI_DIR) / f) for f in meta["DDI_file"]],
        "label_name": np.where(meta["malignant"].astype(bool), "malignant", "benign"),
        "source": "DDI", "group_id": "DDI_" + meta["DDI_ID"].astype(str),
        "fitzpatrick": meta["skin_tone"].map(tone_fst).astype(float),
        "_ddi_stratum": meta["skin_tone"].astype(str) + "_" + meta["malignant"].astype(str),
    })


def assign_folds(df, rng):
    """Group-disjoint K-fold. DDI stratified by (tone x malignant) for representative held-out folds."""
    fold = pd.Series(-1, index=df.index)
    ddi = df["source"] == "DDI"
    # DDI: stratified round-robin per stratum
    for _, idx in df[ddi].groupby("_ddi_stratum").groups.items():
        idx = list(idx); rng.shuffle(idx)
        for i, ix in enumerate(idx):
            fold[ix] = i % K
    # others: group-disjoint round-robin over shuffled groups
    other_groups = df.loc[~ddi, "group_id"].unique(); rng.shuffle(other_groups)
    g2f = {g: i % K for i, g in enumerate(other_groups)}
    fold[~ddi] = df.loc[~ddi, "group_id"].map(g2f).values
    return fold.astype(int)


def main():
    parts = [build_fitz(), build_pad(), build_ddi()]
    for p in parts:
        if "_ddi_stratum" not in p.columns:
            p["_ddi_stratum"] = ""
    df = pd.concat(parts, ignore_index=True)
    df = df[df["filepath"].map(lambda p: os.path.exists(str(p)))].reset_index(drop=True)
    df["fst_group"] = df["fitzpatrick"].map(fst_group)
    rng = np.random.default_rng(SEED)
    df["fold"] = assign_folds(df, rng)

    print("\n=== dataset (source x label) ===")
    print(df.groupby(["source", "label_name"]).size().to_string())
    print("\n=== dark-skin (FST IV-VI) x label ===")
    print(pd.crosstab(df["fst_group"], df["label_name"]).to_string())
    print("\n=== DDI per-fold (held-out test each fold) ===")
    ddi = df[df.source == "DDI"]
    print(pd.crosstab(ddi["fold"], ddi["label_name"]).to_string())

    split = pd.DataFrame({"filepath": df["filepath"], "split": "train_val", "fold": df["fold"]})
    Path(OUT_MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_SPLITS).parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["_ddi_stratum", "fold"]).to_csv(OUT_MANIFEST, index=False)  # fold lives ONLY in splits (avoid merge collision)
    split.to_csv(OUT_SPLITS, index=False)
    nmal = int((df.label_name == "malignant").sum())
    print(f"\n[out] {len(df)} imgs (malignant {nmal} / benign {len(df)-nmal}) | {K}-fold CV | -> {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
