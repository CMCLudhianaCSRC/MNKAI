"""Precompute lesion bounding-box crops for every row in a manifest.

Saves crops to a parallel directory tree and writes a new manifest pointing at
the cropped JPEGs. Use the cropper trained by `scripts.train_cropper`.

    python -m scripts.precompute_crops \
        --manifest data/manifests/master_clean.csv \
        --cropper checkpoints/cropper/best.pt \
        --out-root D:\AIFINAL\images\cropped \
        --out-manifest data/manifests/master_cropped.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mnkai.models.segmentation import TimmUNet, predict_bbox
from mnkai.utils.logging import setup_logging


class _Stream(Dataset):
    def __init__(self, paths: list[str], img_size: int):
        self.paths = paths; self.img_size = int(img_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                W, H = im.size
                resized = im.resize((self.img_size, self.img_size), Image.BILINEAR)
            x = np.asarray(resized, dtype=np.float32) / 255.0
            x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            return {"image": torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32)),
                    "orig_w": W, "orig_h": H, "path": p, "ok": 1}
        except Exception:
            return {"image": torch.zeros(3, self.img_size, self.img_size),
                    "orig_w": 0, "orig_h": 0, "path": p, "ok": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cropper", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pad", type=float, default=0.10)
    args = ap.parse_args()

    setup_logging("logs", "precompute_crops.log", "INFO")
    df = pd.read_csv(args.manifest)
    paths = df["filepath"].astype(str).tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj = torch.load(args.cropper, map_location=device, weights_only=False)
    encoder = obj.get("encoder", "mobilenetv3_large_100")
    img_size = int(args.img_size or obj.get("img_size", 384))
    model = TimmUNet(encoder_name=encoder, pretrained=False).to(device).eval()
    model.load_state_dict(obj["model_state"], strict=False)

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    new_paths: list[str] = []

    ds = _Stream(paths, img_size)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                    shuffle=False, pin_memory=True)

    idx = 0
    with torch.no_grad():
        for batch in tqdm(dl, desc="crop"):
            x = batch["image"].to(device, non_blocking=True)
            bboxes = predict_bbox(model, x, threshold=0.5, pad=args.pad)
            for j in range(len(bboxes)):
                if not int(batch["ok"][j]):
                    new_paths.append(""); idx += 1; continue
                src = batch["path"][j]
                W = int(batch["orig_w"][j]); H = int(batch["orig_h"][j])
                x1, y1, x2, y2 = bboxes[j]
                # rescale bbox from img_size space to original
                sx, sy = W / img_size, H / img_size
                x1, x2 = int(x1 * sx), int(x2 * sx)
                y1, y2 = int(y1 * sy), int(y2 * sy)
                # build output path mirroring input shard
                src_p = Path(src)
                rel = src_p.name
                # 2-level shard from filename to avoid 100k in one folder
                a = rel[:2] if len(rel) >= 2 else "00"
                dst = out_root / a / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(src) as im:
                    im = im.convert("RGB").crop((max(0, x1), max(0, y1),
                                                 min(W, x2), min(H, y2)))
                    im.save(dst, format="JPEG", quality=95)
                new_paths.append(str(dst))
                idx += 1

    df["filepath_cropped"] = new_paths
    out_manifest = Path(args.out_manifest); out_manifest.parent.mkdir(parents=True, exist_ok=True)
    # write a copy where filepath is the cropped one
    out_df = df.copy()
    keep = out_df["filepath_cropped"].astype(str) != ""
    out_df = out_df[keep].copy()
    out_df["filepath_orig"] = out_df["filepath"]
    out_df["filepath"] = out_df["filepath_cropped"]
    out_df = out_df.drop(columns=["filepath_cropped"])
    out_df.to_csv(out_manifest, index=False)
    logger.info(f"[crop] wrote {len(out_df)} rows to {out_manifest}")


if __name__ == "__main__":
    main()
