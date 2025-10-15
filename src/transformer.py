# src/transforms.py
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
from torchvision import transforms as T

@dataclass
class AugConfig:
    img_size: int = 384
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float]  = (0.229, 0.224, 0.225)

def _to_tensor(np_rgb: np.ndarray) -> torch.Tensor:
    # (H,W,3) uint8 -> (3,H,W) float32 [0,1]
    x = torch.from_numpy(np_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    return x

def build_valid(aug: AugConfig, center_crop: bool = True):
    """
    Returns a callable similar to Albumentations pipelines:
      out = tf(image=np_rgb); out["image"] is a normalized tensor.
    """
    size = int(aug.img_size)
    ops = [T.ToTensor()]  # uses PIL-like path, but we'll feed np->PIL on the fly
    if center_crop:
        ops = [T.ToPILImage(), T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
               T.CenterCrop(size), T.ToTensor()]
    normalize = T.Normalize(mean=aug.mean, std=aug.std)
    tfm = T.Compose(ops + [normalize])

    def _apply(image: np.ndarray) -> Dict[str, Any]:
        # Accepts np RGB; applies resize/crop+normalize; returns dict for consistency
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        out = tfm(image)
        return {"image": out}

    return _apply
