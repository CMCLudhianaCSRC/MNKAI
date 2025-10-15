import argparse, os, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--reports_dir', default='checkpoints/isic_120k/reports')
ap.add_argument('--fold_tag', default='fold0')
ap.add_argument('--specificity', type=float, default=0.95)
args = ap.parse_args()

probs = np.load(os.path.join(args.reports_dir, f'val_probs_{args.fold_tag}.npy'))
labels = np.load(os.path.join(args.reports_dir, f'val_labels_{args.fold_tag}.npy'))

C = probs.shape[1]
th = [0.5]*C
for c in range(C):
    y = (labels==c).astype(int)
    p = probs[:,c]
    # sweep thresholds (fast grid)
    grid = np.linspace(0,1,1001)
    best = 0.5
    for g in grid:
        tp = ((p>=g) & (y==1)).sum()
        tn = ((p< g) & (y==0)).sum()
        fp = ((p>=g) & (y==0)).sum()
        fn = ((p< g) & (y==1)).sum()
        spec = tn / max(tn+fp,1)
        if spec >= args.specificity:
            best = g; break
    th[c] = float(best)

with open(os.path.join(args.reports_dir, f'thresholds_{args.fold_tag}.json'),'w') as f:
    json.dump({'specificity_target': args.specificity, 'per_class': th}, f, indent=2)
print("Saved thresholds JSON.")
