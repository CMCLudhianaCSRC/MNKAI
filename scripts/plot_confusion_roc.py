import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

ap = argparse.ArgumentParser()
ap.add_argument('--reports_dir', default='checkpoints/isic_120k/reports')
ap.add_argument('--fold_tag', default='fold0')
args = ap.parse_args()

rd = args.reports_dir
probs = np.load(os.path.join(rd, f'val_probs_{args.fold_tag}.npy'))
labels = np.load(os.path.join(rd, f'val_labels_{args.fold_tag}.npy'))

preds = probs.argmax(1)
cm = confusion_matrix(labels, preds, normalize='true')

plt.figure(figsize=(6,5))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion (normalized)')
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(rd, 'cm_argmax.png'), dpi=200); plt.close()

# Macro ROC
C = probs.shape[1]
fprs, tprs, rocs = [], [], []
for c in range(C):
    y_true = (labels==c).astype(int)
    fpr, tpr, _ = roc_curve(y_true, probs[:,c])
    roc_auc = auc(fpr, tpr)
    fprs.append(fpr); tprs.append(tpr); rocs.append(roc_auc)

plt.figure(figsize=(5,4))
for c,(fpr,tpr,ro) in enumerate(zip(fprs,tprs,rocs)):
    plt.plot(fpr, tpr, lw=1, alpha=0.5)
plt.plot([0,1],[0,1],'--', lw=1)
plt.title(f'ROC (macro AUC≈{np.mean(rocs):.3f})')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.tight_layout()
plt.savefig(os.path.join(rd, 'roc_macro.png'), dpi=200); plt.close()
print("Saved cm_argmax.png and roc_macro.png")
