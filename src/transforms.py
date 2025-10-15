from dataclasses import dataclass
from typing import Callable, Tuple

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image


@dataclass
class AugConfig:
    img_size: int = 384
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    hflip: float = 0.5
    vflip: float = 0.0
    rotate: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    hue: float = 0.0
    blur: float = 0.0
    jpeg: int = 100
    random_crop_scale: Tuple[float, float] = (0.9, 1.0)

    def __post_init__(self) -> None:
        self.mean = tuple(float(x) for x in self.mean)
        self.std = tuple(float(x) for x in self.std)
        rcs = self.random_crop_scale
        if rcs is None:
            self.random_crop_scale = (0.9, 1.0)
        elif isinstance(rcs, (list, tuple)) and len(rcs) == 2:
            self.random_crop_scale = (float(rcs[0]), float(rcs[1]))
        else:
            raise ValueError("random_crop_scale must be a length-2 sequence or None")
        self.hflip = float(self.hflip)
        self.vflip = float(self.vflip)
        self.rotate = float(self.rotate)
        self.brightness = float(self.brightness)
        self.contrast = float(self.contrast)
        self.saturation = float(self.saturation)
        self.hue = float(self.hue)
        self.blur = float(self.blur)
        self.jpeg = int(self.jpeg)


def _ensure_np(image) -> np.ndarray:
    """Accept PIL or ndarray input and return an RGB numpy array."""
    if isinstance(image, Image.Image):
        np_img = np.array(image)
    elif isinstance(image, np.ndarray) or hasattr(image, "__array__"):
        np_img = np.asarray(image)
    else:
        raise TypeError("Expected PIL.Image or numpy array for transforms")

    if np_img.ndim == 2:  # grayscale -> RGB
        np_img = np.stack([np_img] * 3, axis=-1)
    if np_img.shape[-1] != 3:
        raise ValueError("Expected image with 3 channels (RGB)")
    if np_img.dtype != np.uint8:
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)
    return np_img


class _AlbumentationsWrapper:
    def __init__(self, compose: A.Compose):
        self.compose = compose

    def __call__(self, image):
        np_img = _ensure_np(image)
        out = self.compose(image=np_img)
        return out["image"]


def build_train(aug: AugConfig) -> Callable:
    """Training augmentations (Albumentations -> torch tensor)."""
    size = int(aug.img_size)
    crop_scale = (max(0.1, aug.random_crop_scale[0]), min(1.0, aug.random_crop_scale[1]))
    aug_list = [
        A.RandomResizedCrop(height=size, width=size, scale=crop_scale, ratio=(0.9, 1.1), interpolation=1),
    ]

    if aug.hflip > 0:
        aug_list.append(A.HorizontalFlip(p=min(1.0, aug.hflip)))
    if aug.vflip > 0:
        aug_list.append(A.VerticalFlip(p=min(1.0, aug.vflip)))
    if aug.rotate > 0:
        aug_list.append(
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=aug.rotate,
                border_mode=0,
                p=0.7,
            )
        )

    if any(x > 0 for x in (aug.brightness, aug.contrast, aug.saturation, aug.hue)):
        aug_list.append(
            A.ColorJitter(
                brightness=aug.brightness,
                contrast=aug.contrast,
                saturation=aug.saturation,
                hue=aug.hue,
                p=0.4,
            )
        )

    if aug.blur > 0:
        aug_list.append(A.GaussianBlur(blur_limit=(3, 7), p=min(1.0, aug.blur)))

    if aug.jpeg < 100:
        aug_list.append(
            A.ImageCompression(quality_lower=max(10, aug.jpeg), quality_upper=100, p=0.2)
        )

    aug_list.extend(
        [
            A.CoarseDropout(max_holes=4, max_height=size // 10, max_width=size // 10, p=0.3),
            A.Normalize(mean=aug.mean, std=aug.std),
            ToTensorV2(),
        ]
    )
    return _AlbumentationsWrapper(A.Compose(aug_list, p=1.0))


def build_valid(aug: AugConfig, center_crop: bool = True) -> Callable:
    """Validation/eval pipeline producing a normalized tensor."""
    size = int(aug.img_size)
    aug_list = [
        A.Resize(height=size, width=size, interpolation=1),
    ]
    if center_crop:
        aug_list.append(A.CenterCrop(height=size, width=size))
    aug_list.extend([
        A.Normalize(mean=aug.mean, std=aug.std),
        ToTensorV2(),
    ])
    return _AlbumentationsWrapper(A.Compose(aug_list))
