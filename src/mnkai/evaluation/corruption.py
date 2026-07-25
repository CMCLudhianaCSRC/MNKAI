"""Corruption sweep — smartphone-like degradations, applied at test time.

Operates on tensors AFTER the standard valid transform has produced
normalized 3xHxW float tensors. Each corruption returns a NEW tensor; the
original is not modified. Use `corruption_suite()` for the full battery.

Implemented (PPT clinical-realism asks):
  - jpeg_compression(q)        : encode→decode through JPEG
  - motion_blur(kernel_size)
  - low_light(gamma)
  - white_balance_drift(scale_per_channel)
  - gaussian_noise(sigma)
"""
from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# Default normalization constants (ImageNet). (R21) These are the DEFAULT only —
# the denormalize/renormalize round-trip is only correct if they match the statistics
# the evaluated model was actually normalized with. Pass mean/std explicitly (e.g. from
# the eval AugConfig) whenever a model uses different constants, or the corruption
# severities will be applied in the wrong colour space.
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _as_stat(v, default: torch.Tensor) -> torch.Tensor:
    """Accept None / a 3-sequence / an already-shaped tensor -> (1,3,1,1) tensor."""
    if v is None:
        return default
    if isinstance(v, torch.Tensor):
        return v.reshape(1, 3, 1, 1)
    return torch.tensor(list(v), dtype=torch.float32).view(1, 3, 1, 1)


def _denormalize(x: torch.Tensor, mean=None, std=None) -> torch.Tensor:
    m, s = _as_stat(mean, _MEAN), _as_stat(std, _STD)
    return (x * s.to(x.device) + m.to(x.device)).clamp(0, 1)


def _normalize(x: torch.Tensor, mean=None, std=None) -> torch.Tensor:
    """Inverse of _denormalize. Pass the SAME mean/std used to denormalize, or the
    round-trip silently shifts the image's colour statistics. (R21)"""
    m, s = _as_stat(mean, _MEAN), _as_stat(std, _STD)
    return (x - m.to(x.device)) / s.to(x.device)


def jpeg_compression(x: torch.Tensor, quality: int = 50) -> torch.Tensor:
    out = []
    raw = (_denormalize(x).cpu() * 255.0).clamp(0, 255).to(torch.uint8)
    for img in raw:
        pil = Image.fromarray(img.permute(1, 2, 0).numpy())
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        decoded = np.array(Image.open(buf).convert("RGB"))
        out.append(torch.from_numpy(decoded).permute(2, 0, 1).float() / 255.0)
    y = torch.stack(out, dim=0).to(x.device)
    return _normalize(y)


def motion_blur(x: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    k = max(3, int(kernel_size) | 1)
    kernel = torch.zeros(k, k, device=x.device)
    kernel[k // 2, :] = 1.0 / k
    kernel = kernel.view(1, 1, k, k).expand(3, 1, k, k)
    pad = k // 2
    return F.conv2d(x, kernel, padding=pad, groups=3)


def low_light(x: torch.Tensor, gamma: float = 2.5) -> torch.Tensor:
    raw = _denormalize(x)
    raw = raw.pow(gamma)
    return _normalize(raw)


def white_balance_drift(x: torch.Tensor, scale_per_channel=(1.10, 0.95, 0.90)) -> torch.Tensor:
    raw = _denormalize(x)
    s = torch.tensor(scale_per_channel, device=x.device).view(1, 3, 1, 1)
    raw = (raw * s).clamp(0, 1)
    return _normalize(raw)


def gaussian_noise(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    raw = _denormalize(x)
    raw = raw + torch.randn_like(raw) * float(sigma)
    raw = raw.clamp(0, 1)
    return _normalize(raw)


def corruption_suite() -> dict:
    """Default suite of (name, callable) pairs run by `corruption_sweep_eval`."""
    return {
        "clean": lambda x: x,
        "jpeg_q40": lambda x: jpeg_compression(x, 40),
        "jpeg_q20": lambda x: jpeg_compression(x, 20),
        "motion_blur_k5": lambda x: motion_blur(x, 5),
        "motion_blur_k9": lambda x: motion_blur(x, 9),
        "low_light_g2_0": lambda x: low_light(x, 2.0),
        "low_light_g3_0": lambda x: low_light(x, 3.0),
        "wb_drift": lambda x: white_balance_drift(x, (1.15, 0.95, 0.85)),
        "noise_s0_05": lambda x: gaussian_noise(x, 0.05),
        "noise_s0_10": lambda x: gaussian_noise(x, 0.10),
    }


__all__ = [
    "jpeg_compression",
    "motion_blur",
    "low_light",
    "white_balance_drift",
    "gaussian_noise",
    "corruption_suite",
]
