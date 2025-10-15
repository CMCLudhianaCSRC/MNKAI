import os, json, glob, argparse
import numpy as np
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--reports_dir', default='checkpoints/isic_120k/reports')
args = ap.parse_args()

os.makedirs(os.path.join(args.reports_dir, 'figs'), exist_ok=True)

fold_jsons = sorted(glob.glob(os.path.join(args.reports_dir, 'valid_report_fold*.json')))
auc, f1, ece, names = [], [], [], []
for p in fold_jsons:
    with open(p,'r') as f:
        j = json.load(f)
    auc.append(j.get('auc',0))
    f1.append(j.get('f1',0))
    ece.append(j.get('ece',0))
    names.append(os.path.splitext(os.path.basename(p))[0])

if names:
    plt.figure(figsize=(8,4)); plt.bar(names, auc); plt.xticks(rotation=45, ha='right'); plt.ylim(0,1)
    plt.title('AUC by fold'); plt.tight_layout();
    plt.savefig(os.path.join(args.reports_dir,'figs','auc_by_fold.png'), dpi=200); plt.close()

    plt.figure(figsize=(8,4)); plt.bar(names, f1); plt.xticks(rotation=45, ha='right'); plt.ylim(0,1)
    plt.title('F1 by fold'); plt.tight_layout();
    plt.savefig(os.path.join(args.reports_dir,'figs','f1_by_fold.png'), dpi=200); plt.close()

    plt.figure(figsize=(8,4)); plt.bar(names, ece); plt.xticks(rotation=45, ha='right'); plt.ylim(0,1)
    plt.title('ECE by fold (lower is better)'); plt.tight_layout();
    plt.savefig(os.path.join(args.reports_dir,'figs','ece_by_fold.png'), dpi=200); plt.close()

print('Saved fold metric figures to figs/.')
