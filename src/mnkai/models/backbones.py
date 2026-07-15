"""Model factory.

Image-only and image+tabular variants. The image-only path is a thin wrapper
around timm; the multimodal path supports either:

  fusion='concat'        : concatenate pooled features + tabular features
                           into a single MLP head (legacy).
  fusion='cross_attn'    : single-layer cross-attention where the tabular
                           token attends to image patch tokens (when the
                           backbone exposes them) before fusion. Strictly
                           more expressive than concat, used by Pacheco
                           et al. 2021 in dermatology multimodal fusion.

Backbones supported (timm names):
  - convnext_*           : ConvNeXt v1/v2 family
  - tf_efficientnetv2_*  : EfficientNet V2 (current default)
  - vit_base_patch14_dinov2.lvd142m : DINOv2 ViT-B/14 (Oquab 2024)
                                      ImageNet-free SSL backbone, 86M params,
                                      ~3.5 GB VRAM at 224 px / fp16 / bs 24.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import timm
import torch
import torch.nn as nn
from loguru import logger


@dataclass
class ModelConfig:
    name: str = "convnext_tiny"
    pretrained: bool = True
    num_classes: int = 8
    drop_rate: float = 0.10           # backbone drop_rate (timm)
    drop_path_rate: float = 0.10      # stochastic depth
    head_dropout: float = 0.30        # additional dropout in our head
    tabular_dim: int = 0              # 0 disables tabular branch
    tabular_hidden: int = 128
    head_hidden: int = 0              # 0 -> linear head; >0 -> mlp
    fusion: str = "concat"            # 'concat' | 'cross_attn'
    cross_attn_heads: int = 4         # only used when fusion=='cross_attn'
    cross_attn_dim: int = 256         # shared dim for cross-attention
    img_size: int | None = None       # override timm default (esp. needed for DINOv2)


def _create_model_with_fallback(name: str, kwargs: dict) -> nn.Module:
    """timm.create_model with a kwarg-by-kwarg TypeError fallback.

    On TypeError we retry after removing the purely-optional `drop_path_rate`
    kwarg first (some backbones — e.g. certain DINOv2 variants — reject it). We
    deliberately do NOT silently drop `img_size`: dropping it would downgrade the
    model to its default resolution (e.g. 518 -> 224 for DINOv2), which silently
    changes the feature geometry the entire probe/eval pipeline depends on. If
    `img_size` is genuinely rejected, we re-raise so the failure is loud instead
    of producing a wrong-resolution backbone.
    """
    try:
        return timm.create_model(name, **kwargs)
    except TypeError as e:
        # Retry dropping optional kwargs one at a time, drop_path_rate first.
        if "drop_path_rate" in kwargs:
            retry = dict(kwargs)
            retry.pop("drop_path_rate", None)
            try:
                logger.warning(f"[model] '{name}' rejected drop_path_rate; retrying without it")
                return timm.create_model(name, **retry)
            except TypeError:
                pass
        # Still failing. If img_size is set, refuse to silently drop it.
        if kwargs.get("img_size") is not None:
            raise TypeError(
                f"[model] '{name}' rejected create_model kwargs even after dropping "
                f"drop_path_rate, and img_size={kwargs['img_size']} is set. Refusing to "
                f"silently drop img_size (that would downgrade to the backbone's default "
                f"resolution and change feature geometry). Original error: {e}"
            ) from e
        raise


def build_image_only(cfg: ModelConfig) -> nn.Module:
    logger.info(f"[model] timm.create_model('{cfg.name}', pretrained={cfg.pretrained}, "
                f"img_size={cfg.img_size})")
    kwargs: dict = {
        "pretrained": cfg.pretrained,
        "num_classes": cfg.num_classes,
        "drop_rate": cfg.drop_rate,
    }
    if cfg.drop_path_rate > 0:
        kwargs["drop_path_rate"] = cfg.drop_path_rate
    if cfg.img_size is not None:
        kwargs["img_size"] = cfg.img_size
    return _create_model_with_fallback(cfg.name, kwargs)


class _CrossAttnFusion(nn.Module):
    """Single-block cross-attention fusion.

    Tabular MLP output becomes a single 'query token' that attends to image
    patch tokens (or a single pooled token when patch tokens unavailable).
    The attended representation is concatenated with the pooled image
    feature for the final head. This is structurally similar to perceiver-IO
    / late-fusion variants and outperforms naive concat for medical
    multimodal in published dermatology benchmarks (Pacheco 2021).
    """

    def __init__(self, image_dim: int, tab_dim: int, fusion_dim: int,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj_img = nn.Linear(image_dim, fusion_dim)
        self.proj_tab = nn.Linear(tab_dim,   fusion_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, image_tokens: torch.Tensor, tab_emb: torch.Tensor) -> torch.Tensor:
        # image_tokens: (B, N, image_dim) or (B, image_dim) -> add seq dim
        if image_tokens.dim() == 2:
            image_tokens = image_tokens.unsqueeze(1)  # (B, 1, image_dim)
        kv = self.proj_img(image_tokens)              # (B, N, fusion_dim)
        q  = self.proj_tab(tab_emb).unsqueeze(1)      # (B, 1, fusion_dim)
        attn_out, _ = self.attn(q, kv, kv)            # (B, 1, fusion_dim)
        return self.norm(attn_out.squeeze(1) + q.squeeze(1))


class ImageTabularNet(nn.Module):
    """Two-stream model: timm backbone (no head) + tabular MLP -> fused head.

    Fusion modes (set via cfg.fusion):
      - 'concat'     : pooled image features ++ tabular features -> MLP
      - 'cross_attn' : tab token cross-attends to image patches (when
                       backbone exposes them via forward_features), then
                       concatenated with pooled image features for the head.

    For DINOv2 backbones, set drop_rate / drop_path_rate to 0 — DINOv2
    Lightning prefers no extra regularisation on top of the SSL pretrain.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.tabular_dim <= 0:
            raise ValueError("ImageTabularNet requires tabular_dim > 0")
        self.cfg = cfg
        # Some backbones (DINOv2) ignore drop_rate; that's fine.
        backbone_kwargs = dict(
            pretrained=cfg.pretrained,
            num_classes=0,
            drop_rate=cfg.drop_rate,
        )
        if cfg.drop_path_rate > 0:
            backbone_kwargs["drop_path_rate"] = cfg.drop_path_rate
        if cfg.img_size is not None:
            backbone_kwargs["img_size"] = cfg.img_size
        # Kwarg-by-kwarg TypeError fallback (drops drop_path_rate first; never
        # silently drops img_size — see _create_model_with_fallback).
        self.backbone = _create_model_with_fallback(cfg.name, backbone_kwargs)

        feat_dim = self.backbone.num_features
        self._has_patch_tokens = hasattr(self.backbone, "forward_features")

        self.tab_mlp = nn.Sequential(
            nn.Linear(cfg.tabular_dim, cfg.tabular_hidden),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout * 0.5),
            nn.Linear(cfg.tabular_hidden, cfg.tabular_hidden),
            nn.GELU(),
        )

        if cfg.fusion == "cross_attn":
            self.fuse = _CrossAttnFusion(
                image_dim=feat_dim, tab_dim=cfg.tabular_hidden,
                fusion_dim=cfg.cross_attn_dim,
                num_heads=cfg.cross_attn_heads,
                dropout=cfg.head_dropout * 0.5,
            )
            fused_dim = feat_dim + cfg.cross_attn_dim
        else:
            self.fuse = None
            fused_dim = feat_dim + cfg.tabular_hidden

        if cfg.head_hidden > 0:
            self.head = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Dropout(cfg.head_dropout),
                nn.Linear(fused_dim, cfg.head_hidden),
                nn.GELU(),
                nn.Dropout(cfg.head_dropout),
                nn.Linear(cfg.head_hidden, cfg.num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Dropout(cfg.head_dropout),
                nn.Linear(fused_dim, cfg.num_classes),
            )

    def _image_features(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (pooled_features, patch_tokens_or_None)."""
        if self.cfg.fusion == "cross_attn" and self._has_patch_tokens:
            # forward_features returns full token output (cls + patches) for ViT
            # or feature map (B, C, H, W) for CNNs.
            feats = self.backbone.forward_features(image)
            if feats.dim() == 4:
                # CNN feature map -> flatten spatial -> (B, H*W, C)
                B, C, H, W = feats.shape
                tokens = feats.permute(0, 2, 3, 1).reshape(B, H * W, C)
                pooled = feats.mean(dim=(2, 3))
            elif feats.dim() == 3:
                # ViT: (B, 1+N, C); pooled via cls token (index 0) or mean
                tokens = feats
                pooled = feats[:, 0] if feats.shape[1] > 1 else feats.mean(dim=1)
            else:
                # Unknown shape; fall back to standard forward
                pooled = self.backbone(image)
                tokens = pooled.unsqueeze(1) if pooled.dim() == 2 else pooled
            return pooled, tokens
        else:
            pooled = self.backbone(image)
            return pooled, None

    def forward(self, image: torch.Tensor, tabular: torch.Tensor | None = None) -> torch.Tensor:
        pooled, tokens = self._image_features(image)
        if tabular is None:
            tabular = torch.zeros(pooled.size(0), self.cfg.tabular_dim,
                                  device=pooled.device, dtype=pooled.dtype)
        t = self.tab_mlp(tabular)
        if self.fuse is not None and tokens is not None:
            attended = self.fuse(tokens, t)               # (B, cross_attn_dim)
            z = torch.cat([pooled, attended], dim=1)
        else:
            z = torch.cat([pooled, t], dim=1)
        return self.head(z)


def build_model(cfg: ModelConfig) -> nn.Module:
    """Build either the image-only model or the multimodal model based on cfg."""
    if cfg.tabular_dim and cfg.tabular_dim > 0:
        return ImageTabularNet(cfg)
    return build_image_only(cfg)


__all__ = ["ModelConfig", "build_model", "ImageTabularNet", "build_image_only"]
