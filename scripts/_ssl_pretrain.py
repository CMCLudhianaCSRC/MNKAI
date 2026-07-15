"""Self-supervised pretraining on the full MNKAI-III corpus.

In-domain SSL pretraining gives the highest gain-per-effort of any remaining
lever (per the audit). Pretrains a DINOv2-init ViT-B/14 with a SimCLR-style
contrastive objective on 271,058 images (all sources combined, labels ignored).

After SSL, save backbone weights. Folds 1-4 use these as init instead of
public DINOv2 weights → expected +0.03-0.05 macro_F1 per fold.

Reference: Chen et al 2020 (SimCLR); Caron et al 2021 (DINO); Oquab et al 2024
(DINOv2). Our setup is SimCLR for simplicity on RTX 4050.

Usage:
  python -m scripts._ssl_pretrain --config configs/ssl_pretrain.yaml
"""
from __future__ import annotations
import argparse, sys, time, math
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import timm
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ──────────────────────────────────────────────────────────────────────────────
# SimCLR-style augmentation pipeline
# ──────────────────────────────────────────────────────────────────────────────
def simclr_transform(img_size: int = 224):
    """Strong augmentation for SimCLR.
    Standard recipe: RandomResizedCrop + ColorJitter + RandomGrayscale + Flip + Blur.
    """
    import torchvision.transforms as T
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4,
                                       saturation=0.4, hue=0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Dataset: yields 2 augmented views of each image, ignores labels
# ──────────────────────────────────────────────────────────────────────────────
class SSLDataset(Dataset):
    """Memory-frugal: holds only the filepath list (a ~20 MB list of strings),
    NOT the full 271k-row DataFrame. Each spawned DataLoader worker pickles a
    copy of the dataset, so dropping the unused columns cuts per-worker memory
    from ~300 MB to ~20 MB — the fix for the earlier RAM-swap stalls."""

    def __init__(self, df: pd.DataFrame, img_size: int = 224):
        self.paths = df["filepath"].astype(str).tolist()
        self.t = simclr_transform(img_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        fp = self.paths[idx]
        # Robust to missing files: pick another path if this one fails
        img = None
        for _ in range(5):
            try:
                with Image.open(fp) as _im:          # context-managed: closes
                    img = _im.convert("RGB")          # fully loads into memory
                break
            except Exception:
                idx = (idx + 1) % len(self.paths)
                fp = self.paths[idx]
        if img is None:
            img = Image.new("RGB", (224, 224))
        # Two augmented views (positive pair)
        v1 = self.t(img)
        v2 = self.t(img)
        return v1, v2


# ──────────────────────────────────────────────────────────────────────────────
# Model: DINOv2 backbone + projection head
# ──────────────────────────────────────────────────────────────────────────────
class SimCLRModel(nn.Module):
    def __init__(self, backbone_name: str = "vit_base_patch14_dinov2.lvd142m",
                  img_size: int = 224, proj_dim: int = 128, hidden: int = 2048):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True,
                                            num_classes=0, img_size=img_size)
        feat = self.backbone.num_features
        # 2-layer MLP projection head
        self.proj = nn.Sequential(
            nn.Linear(feat, hidden),
            nn.GELU(),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, x):
        h = self.backbone(x)
        z = F.normalize(self.proj(h), dim=-1)
        return h, z


# ──────────────────────────────────────────────────────────────────────────────
# Contrastive loss (NT-Xent, SimCLR original)
# ──────────────────────────────────────────────────────────────────────────────
def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.5):
    """Normalized temperature-scaled cross-entropy on a batch of N pairs.
    z1, z2 are already L2-normalised (B, D)."""
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)              # (2B, D)
    sim = z @ z.t() / temp                       # (2B, 2B)
    # Mask self-similarity
    mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))
    # Positive pair for i in [0..B-1] is i+B; for i in [B..2B-1] is i-B
    targets = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, targets)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/master.csv")
    ap.add_argument("--output", default="checkpoints/ssl_pretrained/backbone.pt")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--subsample", type=int, default=0,
                     help="If >0, randomly subsample the corpus to this many "
                          "images (seed 42) — for feasible SSL on limited GPU.")
    ap.add_argument("--compile", action="store_true",
                     help="Enable torch.compile (opt-in; flaky on Windows).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ---- performance: TF32 matmul + cuDNN autotune (Ampere; pure speedup) ----
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True   # autotune conv algos (fixed shapes)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        logger.info("[ssl] perf: TF32 on, cuDNN benchmark on")

    # Load manifest — use ALL rows (no class filter — SSL uses no labels)
    df = pd.read_csv(args.manifest, low_memory=False)
    # Filter only rows where filepath exists (rough heuristic — skip per-row checks)
    df = df[df["filepath"].astype(str).str.len() > 5].reset_index(drop=True)
    if args.subsample and len(df) > args.subsample:
        df = df.sample(n=args.subsample, random_state=42).reset_index(drop=True)
        logger.info(f"[ssl] subsampled corpus to {len(df):,} images")
    logger.info(f"[ssl] using {len(df):,} images for SSL pretraining")

    ds = SSLDataset(df, img_size=args.img_size)
    # Memory-bounded dataloader: persistent_workers=False so workers are
    # re-spawned each epoch (releases any accumulated memory); prefetch_factor=2
    # caps queued batches. The earlier 12-worker / prefetch-4 run blew system
    # RAM to 42 GB commit and swap-stalled. Keep workers modest (<=6).
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                     num_workers=args.num_workers, pin_memory=True,
                     persistent_workers=False, drop_last=True,
                     prefetch_factor=(2 if args.num_workers > 0 else None))
    logger.info(f"[ssl] dataloader: {len(dl)} steps/epoch "
                f"(batch={args.batch_size}, workers={args.num_workers}, "
                f"prefetch=2, persistent=False)")

    model = SimCLRModel(img_size=args.img_size).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"[ssl] model params: {n_params:.1f}M")

    # ---- optional torch.compile (opt-in; Windows Triton support is flaky) ----
    if args.compile:
        try:
            model = torch.compile(model)
            logger.info("[ssl] torch.compile enabled (default mode)")
        except Exception as e:
            logger.warning(f"[ssl] torch.compile failed ({e}); continuing uncompiled")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    # Cosine schedule
    total_steps = args.epochs * (len(dl) // args.grad_accum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps,
                                                        eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        losses = []
        t0 = time.time()
        accum_count = 0
        for step, (v1, v2) in enumerate(dl):
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda",
                                    dtype=torch.float16):
                _, z1 = model(v1)
                _, z2 = model(v2)
                loss = nt_xent_loss(z1, z2, temp=args.temp) / args.grad_accum
            scaler.scale(loss).backward()
            accum_count += 1
            if accum_count >= args.grad_accum:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                sched.step()
                accum_count = 0
            losses.append(loss.item() * args.grad_accum)
            if step % 100 == 0:
                lr = optim.param_groups[0]["lr"]
                logger.info(f"[ssl] e{epoch+1} step {step}/{len(dl)} "
                            f"loss={loss.item()*args.grad_accum:.4f} lr={lr:.2e}")

        mean_loss = float(np.mean(losses))
        logger.info(f"[ssl] epoch {epoch+1}/{args.epochs} mean_loss={mean_loss:.4f} "
                    f"elapsed={(time.time()-t0)/60:.1f}m")

        # Save backbone only (no proj head — we'll discard it for downstream)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save({
                "backbone_state": model.backbone.state_dict(),
                "backbone_name":  "vit_base_patch14_dinov2.lvd142m",
                "img_size":       args.img_size,
                "ssl_loss":       mean_loss,
                "epoch":          epoch + 1,
            }, args.output)
            logger.info(f"[ssl] saved best backbone -> {args.output} (loss={mean_loss:.4f})")

    logger.info(f"[ssl] done. best_loss={best_loss:.4f}")


if __name__ == "__main__":
    main()
