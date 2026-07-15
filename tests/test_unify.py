"""Tests for the manifest unifier."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from mnkai.data.unify import unify_manifests


def _write_csv(p, rows):
    pd.DataFrame(rows).to_csv(p, index=False)


def test_unify_from_two_sources(tmp_path: Path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, [
        {"isic_id": "ISIC_00000001", "diagnosis": "melanoma", "patient_id": "pt_a01",
         "sex": "male", "age_approx": 60, "anatom_site_general": "head/neck"},
        {"isic_id": "ISIC_00000002", "diagnosis": "nevus", "patient_id": "pt_a01"},
    ])
    _write_csv(b, [
        {"isic_id": "X1", "label": "basal cell carcinoma", "patient_id": "pt_b01"},
        {"isic_id": "X2", "label": "vascular", "patient_id": "pt_b02"},
    ])
    img_root = tmp_path / "img"
    out = unify_manifests([
        {"meta_csv": str(a), "image_root": str(img_root / "a"), "source": "ISIC_2019"},
        {"meta_csv": str(b), "image_root": str(img_root / "b"), "source": "INDIA_DERMACON"},
    ])
    assert len(out) == 4
    assert set(out["source"]) == {"ISIC_2019", "INDIA_DERMACON"}
    assert set(out["label_name"]) == {"mel", "nv", "bcc", "vasc"}
    # group_id should be set from patient_id
    assert (out.loc[out["source"] == "ISIC_2019", "group_id"] == "pt_a01").all()
