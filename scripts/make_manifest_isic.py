"""
Builds a unified manifest CSV from your downloader's CSV.
Input (default): D:\AIFINAL\manifest.csv
Output: data/manifest_unified.csv

Columns produced (if available):
 isic_id,label,diagnosis,benign_malignant,modality,source,age_approx,sex,anatom_site_general,filepath
"""
import argparse, os
import pandas as pd
from pathlib import Path

DEFAULT_IN = r"D:\\AIFINAL\\manifest.csv"
DEFAULT_OUT = "data/manifest_unified.csv"

ap = argparse.ArgumentParser()
ap.add_argument('--in_csv', default=DEFAULT_IN)
ap.add_argument('--out_csv', default=DEFAULT_OUT)
args = ap.parse_args()

Path(os.path.dirname(args.out_csv) or '.').mkdir(parents=True, exist_ok=True)

df = pd.read_csv(args.in_csv)
# Normalize column names (case-insensitive map)
lower_map = {c.lower().strip(): c for c in df.columns}

def ensure_col(name):
    if name in df.columns: return
    if name in lower_map: df[name] = df[lower_map[name]]
    else: df[name] = ""

for col in ['isic_id','diagnosis','benign_malignant','modality','source','filepath',
            'age_approx','sex','anatom_site_general','label']:
    ensure_col(col)

# Derive label if missing/empty
if df['label'].isna().all() or (df['label'].astype(str).str.len()==0).all():
    if 'diagnosis' in df.columns:
        df['label'] = df['diagnosis'].fillna('unknown')
    elif 'benign_malignant' in df.columns:
        df['label'] = df['benign_malignant'].fillna('unknown')
    else:
        df['label'] = 'unknown'

keep = ['isic_id','label','diagnosis','benign_malignant','modality','source',
        'age_approx','sex','anatom_site_general','filepath']

# Filter to dermoscopic + has filepath
df = df[df['modality'].astype(str).str.lower().eq('dermoscopic')]
df = df[df['filepath'].astype(str).str.len() > 0].copy()

# Clean
df['label'] = df['label'].astype(str).str.strip().str.lower()
df['source'] = df['source'].astype(str).str.strip().replace({'': 'ISIC'})

out = df[keep].drop_duplicates(subset=['isic_id']).reset_index(drop=True)
Path('data').mkdir(exist_ok=True)
out.to_csv(args.out_csv, index=False)
print(f"Wrote {args.out_csv} with {len(out)} rows")
