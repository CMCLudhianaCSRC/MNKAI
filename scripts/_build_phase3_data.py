"""Build the unified Phase-3 (Indian) manifest + group-aware splits.

Best-method decisions (documented):
  * Unified 25-class set = base-9 (capped rehearsal to prevent catastrophic
    forgetting of skin cancers) + 16 Indian-relevant classes.
  * Drop `melasma` (only 43 images — unlearnable, pollutes macro-F1).
  * Cap `viral_infection` 37k -> 8k to break its 58% dominance.
  * Base-9 rehearsal caps keep cancers alive without dominating.
  * Drop junk / non-image / missing files.
  * Group-aware 5-fold splits by group_id (zero patient leakage).

Outputs:
  data/manifests/master_phase3_clean.csv
  data/splits/phase3_splits.csv
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
rng = np.random.RandomState(SEED)

PHASE1_MANIFEST = ROOT / "data/manifests/master_clean.csv"
PHASE3_MANIFEST = ROOT / "data/manifests/phase3_external.csv"
OUT_MANIFEST    = ROOT / "data/manifests/master_phase3_clean.csv"
OUT_SPLITS      = ROOT / "data/splits/phase3_splits.csv"

BASE9 = ["mel", "nv", "bcc", "scc", "df", "bkl", "ak", "vasc", "normal"]
BASE9_CAP = {"mel":500,"nv":1000,"bcc":500,"scc":500,"df":300,
             "bkl":500,"ak":500,"vasc":300,"normal":500}

DROP_INDIAN = {"melasma"}             # too few (43)
INDIAN_CAP = {"viral_infection": 8000}  # break 58% dominance

VALID_EXT = (".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff")
N_FOLDS = 5
TEST_FRAC = 0.15   # stratified, group-aware held-out external test set

KEEP_COLS = ["isic_id","filepath","label_name","source","group_id",
             "fitzpatrick","sex","age_approx"]


def log(m): print(f"[p3data] {m}", flush=True)


def clean_paths(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df[df["filepath"].astype(str).str.lower().str.endswith(VALID_EXT)]
    log(f"dropped {n0-len(df)} non-image-extension rows")
    # basename junk filter (e.g. Keywords.jpg)
    bad = df["filepath"].astype(str).str.lower().str.contains("keywords")
    df = df[~bad]
    log(f"dropped {int(bad.sum())} 'keywords' junk rows")
    return df.reset_index(drop=True)


def cap_per_class(df, caps):
    parts = []
    for cls, g in df.groupby("label_name"):
        cap = caps.get(cls)
        if cap is not None and len(g) > cap:
            g = g.sample(n=cap, random_state=SEED)
        parts.append(g)
    return pd.concat(parts).reset_index(drop=True)


def main():
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    # ---- Indian classes ----
    p3 = pd.read_csv(PHASE3_MANIFEST, low_memory=False)
    p3 = p3[[c for c in KEEP_COLS if c in p3.columns]].copy()
    p3 = clean_paths(p3)
    p3 = p3[~p3["label_name"].isin(DROP_INDIAN)]
    indian_classes = sorted(p3["label_name"].unique().tolist())
    log(f"Indian classes ({len(indian_classes)}): {indian_classes}")

    # ---- de-confound viral_infection: source-aware cap --------------------
    # ~92% of viral images come from single-disease MPOX_* datasets that contain
    # NO other class -> "from the mpox dataset" perfectly predicts viral, a
    # dataset-as-label shortcut. Keep ALL clinical/mixed-source viral first
    # (DermNet/SkinDisease/SCIN), then fill the cap with MPOX, so viral is far
    # less dataset-separable and the model must learn real viral morphology.
    vir = p3[p3["label_name"] == "viral_infection"].copy()
    if len(vir) > INDIAN_CAP["viral_infection"]:
        cap = INDIAN_CAP["viral_infection"]
        is_mpox = vir["source"].astype(str).str.upper().str.startswith("MPOX")
        clinical = vir[~is_mpox]
        mpox = vir[is_mpox]
        keep_mpox = mpox.sample(n=max(0, cap - len(clinical)), random_state=SEED) \
                    if len(clinical) < cap else mpox.iloc[0:0]
        vir = pd.concat([clinical, keep_mpox]).reset_index(drop=True)
        log(f"viral de-confound: kept {len(clinical)} clinical + {len(keep_mpox)} mpox "
            f"= {len(vir)} ({100*len(clinical)/max(len(vir),1):.0f}% clinical-source)")
    p3 = pd.concat([p3[p3["label_name"] != "viral_infection"], vir], ignore_index=True)
    # other Indian classes uncapped (INDIAN_CAP only contained viral)

    # ---- base-9 rehearsal reservoir ----
    p1 = pd.read_csv(PHASE1_MANIFEST, low_memory=False)
    p1 = p1[[c for c in KEEP_COLS if c in p1.columns]].copy()
    p1 = p1[p1["label_name"].isin(BASE9)]
    p1 = clean_paths(p1)
    p1 = cap_per_class(p1, BASE9_CAP)
    log(f"base-9 rehearsal rows: {len(p1)}")

    # ---- unify ----
    uni = pd.concat([p1, p3], ignore_index=True)
    # ensure all keep-cols exist
    for c in KEEP_COLS:
        if c not in uni.columns:
            uni[c] = np.nan
    uni = uni[KEEP_COLS]
    # group_id fallback (unique per row) so splitter never leaks
    miss = uni["group_id"].isna()
    uni.loc[miss, "group_id"] = ["solo_%d" % i for i in range(int(miss.sum()))]

    # dedupe by filepath (merge in train.py is on filepath; must be unique)
    n_before = len(uni)
    uni = uni.drop_duplicates(subset=["filepath"]).reset_index(drop=True)
    if len(uni) != n_before:
        log(f"dropped {n_before-len(uni)} duplicate filepaths")

    classes = sorted(uni["label_name"].unique().tolist())
    log(f"UNIFIED classes ({len(classes)}): {classes}")
    log("class counts:\n" + uni["label_name"].value_counts().to_string())

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    uni.to_csv(OUT_MANIFEST, index=False)
    log(f"wrote {OUT_MANIFEST} ({len(uni)} rows)")

    # ---- HELD-OUT EXTERNAL TEST SET + group-aware 5-fold splits ----
    # Carve out a stratified, GROUP-AWARE external test set (~TEST_FRAC) BEFORE
    # the CV folds. Mandatory for a high-tier clinical publication: the 5 CV
    # models / ensemble are measured on data NONE of them ever saw. Whole groups
    # go to test or train (never split) -> zero patient leakage. Stratified
    # per-class so every class appears in test.
    # group -> (majority class, size). Most groups are single-class.
    grp = uni.groupby("group_id")
    grp_cls = grp["label_name"].agg(lambda s: s.value_counts().index[0])
    grp_sz = grp.size()
    split_of = {}   # group_id -> 'external_test' | 'inner'
    for cls in classes:
        cls_groups = grp_cls[grp_cls == cls].index.tolist()
        rng.shuffle(cls_groups)
        cls_total = int(grp_sz[cls_groups].sum())
        target = TEST_FRAC * cls_total
        acc = 0
        for gid in cls_groups:
            if acc < target:
                split_of[gid] = "external_test"
                acc += int(grp_sz[gid])
            else:
                split_of[gid] = "inner"
    # any group not assigned (edge cases) -> inner
    for gid in grp_sz.index:
        split_of.setdefault(gid, "inner")

    # 5-fold over INNER groups only (greedy load-balance)
    inner_groups = [g for g in grp_sz.sort_values(ascending=False).index
                    if split_of[g] == "inner"]
    fold_of = {}
    load = np.zeros(N_FOLDS)
    for gid in inner_groups:
        f = int(np.argmin(load))
        fold_of[gid] = f
        load[f] += int(grp_sz[gid])

    def row_split(gid):
        return "external_test" if split_of[gid] == "external_test" else "train_val"
    def row_fold(gid):
        return -1 if split_of[gid] == "external_test" else int(fold_of[gid])

    split = pd.DataFrame({
        "isic_id": uni["isic_id"].values,
        "filepath": uni["filepath"].values,      # train.py merges on filepath
        "group_id": uni["group_id"].values,
        "split": uni["group_id"].map(row_split).values,
        "fold": uni["group_id"].map(row_fold).values,
    })
    OUT_SPLITS.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(OUT_SPLITS, index=False)
    log(f"wrote {OUT_SPLITS}")
    n_ext = int((split["split"] == "external_test").sum())
    n_inner = len(split) - n_ext
    log(f"external_test={n_ext} ({100*n_ext/len(split):.1f}%)  train_val={n_inner}")
    log("inner fold sizes: " + str(split[split.split=='train_val']["fold"]
                                   .value_counts().sort_index().to_dict()))

    # leakage checks
    span = split.groupby("group_id")["fold"].nunique().max()
    log(f"max folds-per-group incl test (must be 1): {span}")
    ext_g = set(split.loc[split.split=='external_test','group_id'])
    inn_g = set(split.loc[split.split=='train_val','group_id'])
    log(f"groups in BOTH test and train_val (must be 0): {len(ext_g & inn_g)}")
    # per-class test coverage (every class must be present in test)
    test_cls = uni.loc[uni["group_id"].map(split_of)=="external_test","label_name"].nunique()
    log(f"classes present in external_test: {test_cls}/{len(classes)}")
    print("\nPHASE3_CLASSES=" + ",".join(classes))


if __name__ == "__main__":
    main()
