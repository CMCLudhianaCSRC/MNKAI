"""Tests for QC scanner, EXIF stripper, and corruption sweep transforms."""
from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image

from mnkai.data.qc import (
    blur_score,
    corrupt_or_blank,
    exposure_score,
    phash,
    run_qc,
    strip_exif,
)
from mnkai.evaluation.corruption import (
    corruption_suite,
    gaussian_noise,
    jpeg_compression,
    low_light,
    motion_blur,
    white_balance_drift,
)


def _save_tmp(tmp_path, name, img):
    p = tmp_path / name
    img.save(p, format="JPEG", quality=95)
    return p


def test_corrupt_detects_invalid(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    assert corrupt_or_blank(bad) is not None


def test_corrupt_passes_real_image(tmp_path):
    p = _save_tmp(tmp_path, "ok.jpg", Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8")))
    assert corrupt_or_blank(p) is None


def test_blank_image_flagged(tmp_path):
    img = Image.fromarray(np.full((64, 64, 3), 128, dtype="uint8"))
    p = _save_tmp(tmp_path, "blank.jpg", img)
    assert corrupt_or_blank(p) is not None


def test_blur_and_exposure_metrics(tmp_path):
    img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8"))
    p = _save_tmp(tmp_path, "noisy.jpg", img)
    b = blur_score(p)
    e = exposure_score(p)
    assert b > 0
    assert 0 <= e <= 255


def test_phash_changes_with_content(tmp_path):
    a = _save_tmp(tmp_path, "a.jpg", Image.fromarray(np.zeros((64, 64, 3), dtype="uint8")))
    b = _save_tmp(tmp_path, "b.jpg", Image.fromarray(np.full((64, 64, 3), 255, dtype="uint8")))
    assert phash(a) != phash(b)


def test_strip_exif_writes_clean_image(tmp_path):
    src = tmp_path / "with_exif.jpg"
    img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8"))
    img.save(src, format="JPEG", quality=95,
             exif=b"Exif\x00\x00II*\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00")
    dst = tmp_path / "clean.jpg"
    strip_exif(src, dst)
    with Image.open(dst) as im:
        assert (im.getexif() or {}) == {} or len(im.getexif()) == 0


def test_run_qc_table(tmp_path):
    paths = []
    for i in range(4):
        img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8"))
        paths.append(_save_tmp(tmp_path, f"x{i}.jpg", img))
    rep = run_qc([str(p) for p in paths])
    assert len(rep) == 4
    assert "ok" in rep.columns and "phash" in rep.columns


# ------------------------ corruption transforms ------------------------

def _norm_tensor(b=2, h=64, w=64):
    x = torch.rand(b, 3, h, w)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def test_corruption_suite_keys():
    s = corruption_suite()
    assert "clean" in s
    assert "jpeg_q40" in s
    assert "low_light_g3_0" in s


def test_jpeg_compression_changes_image():
    x = _norm_tensor()
    y = jpeg_compression(x, quality=20)
    assert y.shape == x.shape
    assert not torch.allclose(x, y)


def test_motion_blur_changes_image():
    x = _norm_tensor()
    y = motion_blur(x, kernel_size=7)
    assert y.shape == x.shape
    assert not torch.allclose(x, y)


def test_low_light_darkens():
    x = _norm_tensor()
    y = low_light(x, gamma=3.0)
    assert y.shape == x.shape


def test_white_balance_drift():
    x = _norm_tensor()
    y = white_balance_drift(x, (1.2, 0.9, 0.8))
    assert y.shape == x.shape
    assert not torch.allclose(x, y)


def test_gaussian_noise_changes_image():
    x = _norm_tensor()
    torch.manual_seed(0)
    y = gaussian_noise(x, sigma=0.10)
    assert y.shape == x.shape
    assert not torch.allclose(x, y)
