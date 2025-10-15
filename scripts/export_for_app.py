# scripts/export_for_app.py
import argparse, torch, json

ap = argparse.ArgumentParser()
ap.add_argument("--in_ckpt", required=True, help="trainer checkpoint (has keys: model, label_to_idx)")
ap.add_argument("--out_ckpt", required=True, help="path to write app checkpoint (keys: state_dict, classes)")
args = ap.parse_args()

ckpt = torch.load(args.in_ckpt, map_location="cpu")
if "model" not in ckpt or "label_to_idx" not in ckpt:
    raise SystemExit("Input checkpoint missing keys {'model','label_to_idx'}")

label_to_idx = ckpt["label_to_idx"]
# build classes list in index order
classes = [None] * len(label_to_idx)
for label, idx in label_to_idx.items():
    classes[idx] = str(label)

out = {
    "state_dict": ckpt["model"],
    "classes": classes,
}
torch.save(out, args.out_ckpt)
print(f"Wrote {args.out_ckpt} with {len(classes)} classes")
