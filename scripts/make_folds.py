from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified cross-validation folds")
    parser.add_argument("--in_csv", default="data/manifest_unified.csv", help="Input manifest CSV")
    parser.add_argument("--out_csv", default="data/folds_all.csv", help="Output CSV with folds")
    parser.add_argument("--label_col", default=None, help="Column containing class labels. If omitted, common names are tried.")
    parser.add_argument("--group_col", default=None, help="Column used for grouping (patient/source). If omitted, common names are tried.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for StratifiedGroupKFold")
    parser.add_argument("--random_state", type=int, default=2025, help="Random seed for shuffling")
    parser.add_argument("--allow_missing_paths", action="store_true", help="Keep rows with missing filepath instead of dropping them")
    return parser.parse_args()


def pick_column(df: pd.DataFrame, requested: Optional[str], candidates: Iterable[str], *, required: bool = True) -> Optional[str]:
    if requested:
        if requested in df.columns:
            return requested
        raise KeyError(f"Column '{requested}' not found in input CSV")

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    if required:
        raise KeyError(f"Could not find any of the columns: {', '.join(candidates)}")
    return None


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.in_csv)

    filepath_col = pick_column(df, None, ["filepath", "path", "image_path"], required=False)
    if filepath_col and not args.allow_missing_paths:
        df = df[df[filepath_col].notnull()].copy()

    label_col = pick_column(df, args.label_col, ["label", "label_name", "diagnosis", "diagnosis_3"])
    labels = df[label_col].astype(str)

    group_col = pick_column(df, args.group_col, ["group_id", "patient_id", "patient", "subject", "source"], required=False)
    if group_col:
        group = df[group_col].astype(str).fillna("UNKNOWN")
    else:
        group = pd.Series(["GLOBAL"] * len(df))

    skf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    folds = [-1] * len(df)
    for fold_idx, (_, va_idx) in enumerate(skf.split(df, labels, groups=group)):
        for row_idx in va_idx:
            folds[row_idx] = fold_idx

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out = df[["isic_id"]].copy()
    out["fold"] = folds
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({args.n_splits} folds)")


if __name__ == "__main__":
    main()
