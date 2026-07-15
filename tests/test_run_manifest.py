"""Data-free tests for mnkai.utils.run_manifest (stdlib-only module)."""
from __future__ import annotations

import json

import pytest

rm = pytest.importorskip("mnkai.utils.run_manifest")


def test_config_hash_is_stable_and_order_independent():
    a = {"lr": 0.001, "head": "mlp", "layers": [1, 2, 3]}
    b = {"layers": [1, 2, 3], "head": "mlp", "lr": 0.001}  # reordered keys
    assert rm.config_hash(a) == rm.config_hash(b)
    # a real change must change the hash
    c = dict(a, lr=0.002)
    assert rm.config_hash(a) != rm.config_hash(c)


def test_config_hash_int_float_normalized():
    assert rm.config_hash({"k": 1}) == rm.config_hash({"k": 1.0})


def test_file_hash_roundtrip(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    h1 = rm.file_hash(p)
    assert h1 and len(h1) == 16
    # identical content -> identical hash; changed content -> different
    p2 = tmp_path / "y.bin"
    p2.write_bytes(b"hello world")
    assert rm.file_hash(p2) == h1
    p2.write_bytes(b"hello world!")
    assert rm.file_hash(p2) != h1


def test_file_hash_missing_returns_none(tmp_path):
    assert rm.file_hash(tmp_path / "nope.bin") is None


def test_package_versions_reports_known_and_unknown():
    vers = rm.package_versions(("numpy", "definitely-not-a-real-package-xyz"))
    assert "numpy" in vers
    assert vers["definitely-not-a-real-package-xyz"] == "not-installed"


def test_build_manifest_captures_core_fields(tmp_path):
    cfg = {"head": "mlp", "img_size": 518}
    inp = tmp_path / "feats.npz"
    inp.write_bytes(b"not really an npz but bytes are hashed")
    man = rm.build_run_manifest(
        "probe_multiclass",
        seeds={"numpy": 0, "sklearn_random_state": 0},
        config=cfg,
        inputs=[inp],
        outputs=[tmp_path / "out.json"],  # does not exist -> recorded as absent
        extra={"head": "mlp"},
    )
    d = man.to_dict()
    assert d["stage"] == "probe_multiclass"
    assert d["seeds"]["sklearn_random_state"] == 0
    assert d["config_hash"] == rm.config_hash(cfg)
    assert d["schema_version"] == rm._SCHEMA_VERSION
    assert d["timestamp_utc"]  # non-empty ISO timestamp
    assert d["python"]
    # inputs hashed + existence recorded
    assert d["inputs"][0]["exists"] is True
    assert d["inputs"][0]["sha256_16"] is not None
    assert d["outputs"][0]["exists"] is False
    assert d["outputs"][0]["sha256_16"] is None
    assert "numpy" in d["packages"]


def test_write_manifest_creates_valid_json(tmp_path):
    dest = tmp_path / "sub" / "run.manifest.json"
    out = rm.write_run_manifest(
        dest,
        stage="ddi_ensemble_ci",
        seeds={"point_estimate_seed": 0},
        config={"members": 6},
    )
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["stage"] == "ddi_ensemble_ci"
    assert loaded["seeds"]["point_estimate_seed"] == 0
    assert loaded["config_hash"] == rm.config_hash({"members": 6})


def test_config_path_hashed_when_no_config(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("head: mlp\nimg_size: 518\n", encoding="utf-8")
    man = rm.build_run_manifest("stage", config_path=cfg_file)
    assert man.config_hash == rm.file_hash(cfg_file)
    assert man.config_path == str(cfg_file)
