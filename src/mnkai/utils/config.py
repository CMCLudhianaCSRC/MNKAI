from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`. Returns a new dict."""
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config; if it contains `_base_`, transitively merge those first.

    `_base_` may be a string path or a list of paths. Later bases override earlier ones;
    the current file's keys override all bases.
    """
    path = Path(path)
    cfg = load_yaml(path)
    bases = cfg.pop("_base_", None)
    if bases is None:
        return cfg
    if isinstance(bases, str):
        bases = [bases]
    merged: dict[str, Any] = {}
    for b in bases:
        bp = (path.parent / b).resolve()
        merged = deep_update(merged, load_config(bp))
    return deep_update(merged, cfg)
