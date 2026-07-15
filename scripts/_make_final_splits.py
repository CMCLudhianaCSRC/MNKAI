"""Generate 5-fold group-aware splits on the cleaned + deduped master.

Input:  data/manifests/master_clean.csv  (after dedup + QA)
Output: data/splits/master_splits.csv
        - external test split for MSK images
        - 5 group-aware folds for everything else
        - leakage assertion verified before write
"""
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnkai.data.splits import SplitConfig, make_splits, assert_no_group_leakage
from mnkai.data.manifest import LabelNormalizer, DEFAULT_CLASSES

IN_CSV  = Path("data/manifests/master_clean.csv")
OUT_CSV = Path("data/splits/master_splits.csv")

CLASSES = DEFAULT_CLASSES


def main():
    print(f"Loading {IN_CSV}", flush=True)
    df = pd.read_csv(IN_CSV, low_memory=False)
    print(f"  rows: {len(df):,}", flush=True)

    # Filter to 8-class
    df8 = df[df["label_name"].astype(str).isin(CLASSES)].copy()
    print(f"  8-class rows: {len(df8):,}", flush=True)
    print(df8["label_name"].value_counts().to_string(), flush=True)

    # Ensure group_id
    df8["isic_id"]  = df8["isic_id"].astype(str)
    df8["group_id"] = df8["group_id"].astype(str)
    mask = df8["group_id"].isin(["nan", "None", ""])
    df8.loc[mask, "group_id"] = ["row_" + str(i) for i in df8.index[mask]]
    print(f"  unique groups: {df8['group_id'].nunique():,}", flush=True)

    # Build splits
    cfg = SplitConfig(k_folds=5, external_source="MSK", seed=42)
    out_df = make_splits(df8, cfg)

    # Verify no leakage
    assert_no_group_leakage(out_df)
    print(f"  no-leakage assertion PASSED", flush=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(out_df):,} rows)", flush=True)

    # Summary
    print(f"\nSplit assignment:")
    print(out_df["split"].value_counts().to_string())
    print(f"\nFold balance (train+val rows):")
    tv = out_df[out_df["split"] != "external_test"]
    print(tv["fold"].value_counts().sort_index().to_string())
    print(f"\nClass × split:")
    print(out_df.groupby(["split", "label_name"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
