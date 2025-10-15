import argparse, pandas as pd, os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--in_csv', default='data/manifest_unified.csv')
ap.add_argument('--out_csv', default='data/manifest_subset_120k.csv')
ap.add_argument('--total', type=int, default=120000)
ap.add_argument('--label_col', type=str, default=None, help="Column name to balance by. If not provided, auto-detects.")
args = ap.parse_args()

Path(os.path.dirname(args.out_csv) or '.').mkdir(parents=True, exist_ok=True)
df = pd.read_csv(args.in_csv)
df = df[df['filepath'].notnull()].copy()

# Decide which column to use for labels
candidates = [
    args.label_col,
    'label',
    'label_name',
    'fine_label',
    'coarse_label',
    'diagnosis_norm',
]
label_col = next((c for c in candidates if c and c in df.columns), None)
if label_col is None:
    raise KeyError(f"No suitable label column found. Tried: {[c for c in candidates if c]}. Available columns: {list(df.columns)}")

labs = df[label_col].fillna('unknown').astype(str).str.strip().str.lower()
classes = sorted(labs.unique())
per = max(1, args.total // max(1, len(classes)))

parts = []
for c in classes:
    dc = df[labs == c]
    if len(dc) == 0:
        continue
    take = min(per, len(dc))
    parts.append(dc.sample(n=take, random_state=2025))

sub = pd.concat(parts).sample(frac=1.0, random_state=2025).reset_index(drop=True)
sub.to_csv(args.out_csv, index=False)
print(f"Balanced by '{label_col}'. Wrote {args.out_csv} with {len(sub)} rows across {len(classes)} classes (~{per} each)")
