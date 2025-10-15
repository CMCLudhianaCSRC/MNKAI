import argparse, json, os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import pandas as pd

from src.datasets.isic_streaming import ISICStreaming
from src.datasets.isic_streaming_mm import ISICStreamingMM
from src.models.mm_classifier import ImageTabularNet
from src.utils.metrics import macro_auc, macro_f1, ece_score

ap = argparse.ArgumentParser()
ap.add_argument('--manifest', default='data/manifest_subset_120k.csv')  # sanity: eval on the same subset
ap.add_argument('--cache_dir', default=r'D:\AIFINAL\images')
ap.add_argument('--ckpt', required=True)
ap.add_argument('--use_tabular', action='store_true')
ap.add_argument('--out_dir', default='checkpoints/isic_120k/reports')
ap.add_argument('--fold_tag', default='fold0')
args = ap.parse_args()

TFM = T.Compose([T.Resize((384,384)), T.ToTensor()])
if args.use_tabular:
    ds = ISICStreamingMM(args.manifest, args.cache_dir, transforms=TFM)
else:
    ds = ISICStreaming(args.manifest, args.cache_dir, transforms=TFM)

model = ImageTabularNet(num_classes=len(ds.label_to_idx), tab_dim=getattr(ds,'tab_dim',0)).cuda()
state = torch.load(args.ckpt, map_location='cpu'); model.load_state_dict(state['model']); model.eval()

loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
all_logits, all_labels, all_src, all_ids = [], [], [], []
with torch.no_grad():
    for batch in loader:
        if args.use_tabular:
            x, tab, y, isid, src = batch
            x, tab = x.cuda(), tab.cuda(); logits = model(x, tab)
        else:
            x, y, isid, src = batch
            x = x.cuda(); logits = model(x)
        all_logits.append(logits.cpu()); all_labels.append(y); all_src.extend(src); all_ids.extend(isid)

logits = torch.cat(all_logits); labels = torch.cat(all_labels).numpy()
probs = torch.softmax(logits, dim=1).numpy(); preds = probs.argmax(1)
res = {'overall': {'auc': macro_auc(labels, probs), 'f1': macro_f1(labels, preds), 'ece': ece_score(probs, labels)}}

src_series = pd.Series(all_src)
for src in sorted(src_series.unique()):
    idx = np.where(src_series.values==src)[0]
    res[src] = {'auc': macro_auc(labels[idx], probs[idx]),
                'f1': macro_f1(labels[idx], preds[idx]),
                'n': int(len(idx))}

os.makedirs(args.out_dir, exist_ok=True)
np.save(os.path.join(args.out_dir, f'val_probs_{args.fold_tag}.npy'), probs)
np.save(os.path.join(args.out_dir, f'val_labels_{args.fold_tag}.npy'), labels)
with open(os.path.join(args.out_dir, f'sources_{args.fold_tag}.json'),'w') as f: json.dump(all_src, f)
with open(os.path.join(args.out_dir, f'isic_ids_{args.fold_tag}.json'),'w') as f: json.dump(all_ids, f)
print(json.dumps(res, indent=2))
