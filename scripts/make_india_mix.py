import argparse, pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--isic_csv', default='data/manifest_unified.csv')
ap.add_argument('--india_csv', required=True, help='Your Indian dataset manifest (same columns as unified)')
ap.add_argument('--out_csv', default='data/manifest_india_mix.csv')
ap.add_argument('--india_frac', type=float, default=0.7, help='target fraction India in mix')
ap.add_argument('--cap', type=int, default=200000, help='total rows in mixed manifest')
args = ap.parse_args()

Path('data').mkdir(exist_ok=True)
A = pd.read_csv(args.isic_csv)
B = pd.read_csv(args.india_csv)
B['source'] = 'INDIA'

n = args.cap
nB = int(n * args.india_frac)

B2 = B.sample(n=min(nB, len(B)), random_state=2025) if len(B) else B
A2 = A.sample(n=min(n-nB, len(A)), random_state=2025) if len(A) else A

mix = pd.concat([A2,B2]).sample(frac=1.0, random_state=2025).reset_index(drop=True)
mix.to_csv(args.out_csv, index=False)
print(f"Wrote {args.out_csv} with {len(mix)} rows; INDIA≈{(len(B2)/max(len(mix),1)):.2%}")
