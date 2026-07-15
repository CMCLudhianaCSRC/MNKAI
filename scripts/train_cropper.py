"""Train a binary lesion-segmentation U-Net on ISIC 2018 Task 1 data.

Output: a small checkpoint (`checkpoints/cropper/best.pt`) that
`scripts/precompute_crops.py` then uses to produce per-image bounding boxes
that we pre-cache for classifier training.

The dataset spec is a CSV with `filepath, mask_path` columns.

    python -m scripts.train_cropper \
        --pairs data/manifests/isic2018_task1_pairs.csv \
        --epochs 25 --img-size 384 --out-dir checkpoints/cropper
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mnkai.models.segmentation import TimmUNet
from mnkai.utils.logging import setup_logging
from mnkai.utils.seed import seed_worker, set_seed, torch_generator


class SegPairs(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int, train: bool):
        self.df = df.reset_index(drop=True)
        self.img_size = int(img_size)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["filepath"]).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        msk = Image.open(row["mask_path"]).convert("L").resize((self.img_size, self.img_size), Image.NEAREST)
        x = np.asarray(img, dtype=np.float32) / 255.0
        m = (np.asarray(msk, dtype=np.float32) > 127).astype(np.float32)
        if self.train and np.random.rand() < 0.5:
            x = x[:, ::-1, :].copy(); m = m[:, ::-1].copy()
        if self.train and np.random.rand() < 0.5:
            x = x[::-1, :, :].copy(); m = m[::-1, :].copy()
        x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return {"image": torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32)),
                "mask": torch.from_numpy(m).unsqueeze(0)}


def dice_loss(logits, target):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + 1.0
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
    return (1 - num / den).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="CSV with filepath, mask_path columns")
    ap.add_argument("--encoder", default="mobilenetv3_large_100")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="checkpoints/cropper")
    args = ap.parse_args()

    setup_logging("logs", "train_cropper.log", "INFO")
    set_seed(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.pairs)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(df))
    n_val = max(1, int(len(df) * args.val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    ds_tr = SegPairs(df.iloc[tr_idx], args.img_size, train=True)
    ds_va = SegPairs(df.iloc[val_idx], args.img_size, train=False)

    g = torch_generator(args.seed)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=4,
                       pin_memory=True, worker_init_fn=seed_worker, generator=g)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=4,
                       pin_memory=True, worker_init_fn=seed_worker, generator=g)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimmUNet(encoder_name=args.encoder, pretrained=True).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(dl_tr, desc=f"epoch {epoch+1}/{args.epochs}", ncols=100):
            x = batch["image"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = 0.5 * bce(logits, m) + 0.5 * dice_loss(logits.float(), m)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

        # validation Dice
        model.eval()
        dice_sum, n = 0.0, 0
        with torch.no_grad():
            for batch in dl_va:
                x = batch["image"].to(device, non_blocking=True)
                m = batch["mask"].to(device, non_blocking=True)
                logits = model(x)
                p = (torch.sigmoid(logits) > 0.5).float()
                num = 2 * (p * m).sum(dim=(1, 2, 3))
                den = p.sum(dim=(1, 2, 3)) + m.sum(dim=(1, 2, 3))
                dice = (num + 1) / (den + 1)
                dice_sum += float(dice.sum().item()); n += int(dice.numel())
        dice_avg = dice_sum / max(1, n)
        logger.info(f"[cropper] epoch {epoch+1} dice={dice_avg:.4f}")
        if dice_avg > best:
            best = dice_avg
            torch.save({"epoch": epoch + 1, "model_state": model.state_dict(),
                        "encoder": args.encoder, "img_size": args.img_size}, out / "best.pt")
            logger.info(f"[cropper] saved best dice={best:.4f}")


if __name__ == "__main__":
    main()
