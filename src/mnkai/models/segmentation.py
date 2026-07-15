"""Lightweight lesion-segmentation model.

Used as a preprocessing step: predict a binary lesion mask, then crop the
classification input to the bounding box of the mask (with padding). Trained
once on ISIC 2018 Task 1 (lesion segmentation), cached, then frozen.

Architecture: U-Net with a MobileNetV3-Large encoder (timm features). Small
enough to run on 6 GB VRAM at 384x384 with batch 8.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class TimmUNet(nn.Module):
    """Encoder-decoder for binary segmentation. The encoder is a timm model
    in features-only mode; the decoder is a vanilla U-Net upsampling stack.
    """

    def __init__(self, encoder_name: str = "mobilenetv3_large_100",
                 pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=pretrained,
                                         features_only=True)
        feat_dims = self.encoder.feature_info.channels()  # e.g. [16, 24, 40, 112, 960]
        self.feat_dims = feat_dims

        ups: list[nn.Module] = []
        decs: list[nn.Module] = []
        prev = feat_dims[-1]
        for skip in reversed(feat_dims[:-1]):
            ups.append(nn.ConvTranspose2d(prev, skip, kernel_size=2, stride=2))
            decs.append(_DoubleConv(skip * 2, skip))
            prev = skip
        self.ups = nn.ModuleList(ups)
        self.decs = nn.ModuleList(decs)
        self.head = nn.Sequential(
            nn.ConvTranspose2d(prev, prev // 2, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(prev // 2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)  # list low->high
        x = feats[-1]
        skips = list(reversed(feats[:-1]))
        for up, dec, skip in zip(self.ups, self.decs, skips):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                                  align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))
        out = self.head(x)
        return out  # (B, 1, H, W) logits


@torch.no_grad()
def predict_bbox(model: TimmUNet, x: torch.Tensor, *, threshold: float = 0.5,
                 pad: float = 0.10) -> list[tuple[int, int, int, int]]:
    """Per-image bounding box of the predicted mask (as int (x1,y1,x2,y2)).

    Returns the full image bbox if the mask is empty (defensive default).
    """
    logits = model(x)
    masks = torch.sigmoid(logits) > threshold
    out = []
    H, W = x.shape[-2:]
    for m in masks:
        ys, xs = torch.where(m[0])
        if len(xs) == 0:
            out.append((0, 0, W, H)); continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        # padding
        bw, bh = x2 - x1, y2 - y1
        px = int(bw * pad); py = int(bh * pad)
        x1 = max(0, x1 - px); y1 = max(0, y1 - py)
        x2 = min(W, x2 + px); y2 = min(H, y2 + py)
        out.append((x1, y1, x2, y2))
    return out


__all__ = ["TimmUNet", "predict_bbox"]
