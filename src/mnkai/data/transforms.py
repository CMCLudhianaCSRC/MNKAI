"""Image transforms.

Three notable additions vs the legacy code:

  1. *Shades-of-Gray color constancy* (Finlayson et al., Minkowski p-norm) as a
     pre-step. Dermatology images come from many cameras + light sources;
     SoG removes the global illuminant and is a near-zero-cost domain
     normalizer that consistently lifts cross-source AUROC.

  2. *Skin-tone-aware hair removal* (off by default). The legacy implementation
     blackhat-thresholded raw grayscale at threshold=10, which fires on melanin
     in Fitzpatrick V-VI skin and silently corrupts the very samples we care
     about. The replacement gates the inpaint mask by HSV "is-skin" pixels and
     by hair-like elongation, drastically reducing false positives on dark skin.

  3. Augmentation policy tuned for noisy clinical input:
       - JPEG compression jitter
       - mild defocus + motion blur
       - random shadow + brightness jitter
       - random erasing (heavier than CoarseDropout)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image


# -------------------------------------------------------- color constancy --

def shades_of_gray(img: np.ndarray, p: float = 6.0, eps: float = 1e-8) -> np.ndarray:
    """Minkowski-norm color constancy with order `p` (Finlayson 2004).

    Args
    ----
    img : H x W x 3 uint8 or float RGB image
    p   : Minkowski norm exponent. p=1 is gray-world, p=inf is max-RGB,
          p=6 is the canonical Shades-of-Gray and the strongest single
          domain-shift defense for dermoscopy.

    Returns
    -------
    out : same shape and dtype as `img`, color-corrected.
    """
    is_uint8 = img.dtype == np.uint8
    x = img.astype(np.float32)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"shades_of_gray expects HxWx3 RGB; got {x.shape}")
    norms = np.power(np.power(x, p).mean(axis=(0, 1)) + eps, 1.0 / p)  # (3,)
    illum = norms / (norms.mean() + eps)
    illum = np.where(illum > 0, illum, 1.0)
    out = x / illum
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8) if is_uint8 else out


class ShadesOfGray(A.ImageOnlyTransform):
    """Albumentations wrapper for shades-of-gray color constancy."""

    def __init__(self, p: float = 1.0, norm_p: float = 6.0):
        super().__init__(p=p)
        self.norm_p = float(norm_p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return shades_of_gray(img, p=self.norm_p)

    def get_transform_init_args_names(self):
        return ("norm_p",)


# ----------------------------------------------------------- hair removal --


def _is_skin_mask_hsv(bgr: np.ndarray) -> np.ndarray:
    """Permissive skin-color gate that includes Fitzpatrick V-VI.

    HSV ranges chosen empirically to cover light through deeply pigmented skin
    while excluding pure black hair. Returns a uint8 mask (0 / 255).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # H: 0-25 OR 160-179 (red/brown), S: > 20, V: > 30
    skin = ((((h <= 25) | (h >= 160)) & (s > 20) & (v > 30))).astype(np.uint8) * 255
    return skin


def _hair_mask_blackhat(bgr: np.ndarray) -> np.ndarray:
    """Detect elongated dark structures (hair) using directional black-hat.

    Multi-orientation black-hat is much more selective than the single 17x17
    rectangular kernel used in the legacy code.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    masks = []
    for angle in (0, 45, 90, 135):
        kx, ky = (15, 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        # rotate kernel by approximating with directional structuring elements
        if angle == 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        elif angle == 90:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        elif angle == 45:
            k = np.eye(15, dtype=np.uint8)
        else:  # 135
            k = np.flipud(np.eye(15, dtype=np.uint8))
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
        masks.append(bh)
    bh = np.maximum.reduce(masks)
    # adaptive threshold: only keep strong responses
    thr = max(20, int(np.percentile(bh, 99)))
    mask = (bh > thr).astype(np.uint8) * 255
    # clean up noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mask


def remove_hair_safe(img_rgb: np.ndarray, max_side: int = 768) -> np.ndarray:
    """Skin-tone-aware hair removal. Operates on RGB uint8.

    Will not inpaint pixels that are not in the skin-color HSV cone, so it
    cannot wholesale repaint melanin in Fitzpatrick V-VI samples.
    """
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    h, w = bgr.shape[:2]
    s = max(h, w)
    scale = 1.0 if s <= max_side else max_side / float(s)
    if scale != 1.0:
        small = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    else:
        small = bgr

    skin = _is_skin_mask_hsv(small)
    hair = _hair_mask_blackhat(small)
    mask = cv2.bitwise_and(hair, skin)  # only inpaint where it's both hair-like AND on skin

    # inpaint
    inpainted = cv2.inpaint(small, mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)

    if scale != 1.0:
        inpainted = cv2.resize(inpainted, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)


class SafeHairRemoval(A.ImageOnlyTransform):
    def __init__(self, p: float = 0.5, max_side: int = 768):
        super().__init__(p=p)
        self.max_side = int(max_side)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return remove_hair_safe(img, max_side=self.max_side)

    def get_transform_init_args_names(self):
        return ("max_side",)


# ------------------------------------------------------------------ config --


@dataclass
class AugConfig:
    img_size: int = 384
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # color constancy
    shades_of_gray_p: float = 1.0  # apply with prob 1.0 by default
    sog_norm: float = 6.0

    # hair removal
    hair_removal: bool = False
    hair_removal_p: float = 0.5
    hair_removal_max_side: int = 768

    # geometric
    hflip: float = 0.5
    vflip: float = 0.2
    rotate: int = 25
    random_crop_scale: tuple[float, float] = (0.8, 1.0)

    # color
    brightness: float = 0.15
    contrast: float = 0.15
    saturation: float = 0.10
    hue: float = 0.03

    # noise
    blur: float = 0.15
    motion_blur: float = 0.15
    jpeg_lower: int = 60  # 0 to disable

    # erasing
    erase_p: float = 0.30
    erase_max_holes: int = 3
    erase_max_frac: float = 0.18

    extra: dict = field(default_factory=dict)


def _np_from(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        x = np.array(image)
    else:
        x = np.asarray(image)
    if x.ndim == 2:
        x = np.stack([x] * 3, axis=-1)
    if x.shape[-1] != 3:
        raise ValueError("expected HxWx3 RGB")
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x


class _AlbWrap:
    def __init__(self, compose: A.Compose):
        self.compose = compose

    def __call__(self, image):
        return self.compose(image=_np_from(image))["image"]


def build_train(aug: AugConfig) -> Callable:
    size = int(aug.img_size)
    aug_list: list = []

    if aug.shades_of_gray_p > 0:
        aug_list.append(ShadesOfGray(p=aug.shades_of_gray_p, norm_p=aug.sog_norm))
    if aug.hair_removal:
        aug_list.append(SafeHairRemoval(p=aug.hair_removal_p, max_side=aug.hair_removal_max_side))

    aug_list += [
        A.RandomResizedCrop(size=(size, size), scale=tuple(aug.random_crop_scale), ratio=(0.85, 1.15),
                            interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=aug.hflip),
        A.VerticalFlip(p=aug.vflip),
        A.Affine(rotate=(-aug.rotate, aug.rotate), translate_percent=(0.0, 0.05),
                 scale=(0.95, 1.05), p=0.7, mode=cv2.BORDER_REFLECT_101),
        A.OneOf([
            A.ColorJitter(brightness=aug.brightness, contrast=aug.contrast,
                          saturation=aug.saturation, hue=aug.hue, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=aug.brightness, contrast_limit=aug.contrast, p=1.0),
        ], p=0.6),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
            A.Defocus(radius=(1, 3), p=1.0),
        ], p=max(aug.blur, aug.motion_blur)),
    ]
    if aug.jpeg_lower > 0 and aug.jpeg_lower < 100:
        aug_list.append(A.ImageCompression(quality_range=(aug.jpeg_lower, 100), p=0.3))

    aug_list += [
        A.CoarseDropout(num_holes_range=(1, max(1, aug.erase_max_holes)),
                        hole_height_range=(int(size * 0.04), int(size * aug.erase_max_frac)),
                        hole_width_range=(int(size * 0.04), int(size * aug.erase_max_frac)),
                        p=aug.erase_p),
        A.Normalize(mean=aug.mean, std=aug.std),
        ToTensorV2(),
    ]
    return _AlbWrap(A.Compose(aug_list))


def build_valid(aug: AugConfig) -> Callable:
    size = int(aug.img_size)
    aug_list: list = []
    if aug.shades_of_gray_p > 0:
        aug_list.append(ShadesOfGray(p=1.0, norm_p=aug.sog_norm))
    aug_list += [
        A.LongestMaxSize(max_size=int(size * 1.15), interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_REFLECT_101),
        A.CenterCrop(height=size, width=size),
        A.Normalize(mean=aug.mean, std=aug.std),
        ToTensorV2(),
    ]
    return _AlbWrap(A.Compose(aug_list))


__all__ = [
    "AugConfig",
    "ShadesOfGray",
    "SafeHairRemoval",
    "shades_of_gray",
    "remove_hair_safe",
    "build_train",
    "build_valid",
]
