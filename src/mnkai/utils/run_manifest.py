"""Structured per-run manifest — the "what produced this artifact" record.

A run manifest is a small JSON blob capturing everything needed to reason about
*whether two runs are comparable*: the git SHA of the code, the seeds, a stable
hash of the config, the input files (with content hashes), timestamps, and the
package versions of the libraries that materially affect results. Write one next
to every artifact (a probe JSON, `models/clinical_25class.pkl`, a DDI result) so
a number can always be traced back to the exact inputs that made it.

Design constraints:
  * **No heavy dependencies** — stdlib only (``importlib.metadata`` for versions,
    ``subprocess`` for a best-effort git SHA). Importing this never pulls torch.
  * **Fail-soft on environment probing** — a missing git binary or an uninstalled
    package yields ``None``/``"not-installed"`` rather than raising, so writing a
    manifest never breaks a run.
  * **Stable hashing** — config hashing is canonical (sorted keys, normalized
    numbers) so the same logical config always hashes the same string.

Typical use inside a pipeline script::

    from mnkai.utils.run_manifest import write_run_manifest

    write_run_manifest(
        "eval/probe/probe_25class_518_mlp.manifest.json",
        stage="probe_multiclass",
        seeds={"numpy": 0, "sklearn_random_state": 0},
        config_path="configs/phase3_indian_25class.yaml",
        inputs=["eval/probe/feats_25class_518.npz"],
        outputs=["eval/probe/probe_25class_518_mlp.json"],
        extra={"head": "mlp", "img_size": 518},
    )
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Packages whose versions materially affect a result — recorded by default.
DEFAULT_TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "torch",
    "torchvision",
    "timm",
    "Pillow",
)

_SCHEMA_VERSION = "1.0"


# ============================================================================
# hashing helpers
# ============================================================================
def _canonical(obj: Any) -> Any:
    """Recursively normalize a JSON-ish object so equal configs hash identically."""
    if isinstance(obj, dict):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, float):
        # collapse -0.0 / integral floats so 1 and 1.0 don't diverge
        if obj == int(obj):
            return int(obj)
        return round(obj, 12)
    return obj


def config_hash(config: Any) -> str:
    """SHA-256 (first 16 hex chars) of a config object, canonicalized first."""
    blob = json.dumps(_canonical(config), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def file_hash(path: str | Path, *, chunk: int = 1 << 20) -> str | None:
    """SHA-256 (first 16 hex chars) of a file's bytes, or ``None`` if absent."""
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _file_record(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return {
        "path": str(path),
        "exists": p.is_file(),
        "sha256_16": file_hash(p),
        "bytes": p.stat().st_size if p.is_file() else None,
    }


# ============================================================================
# environment probing (all fail-soft)
# ============================================================================
def git_sha(short: bool = False) -> str | None:
    """Best-effort current git commit SHA; ``None`` if not a repo / git missing."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        return None
    return None


def git_dirty() -> bool | None:
    """True if the working tree has uncommitted changes; ``None`` if undeterminable."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except Exception:
        return None
    return None


def package_versions(packages: tuple[str, ...] = DEFAULT_TRACKED_PACKAGES) -> dict[str, str]:
    """Installed version per package name; ``"not-installed"`` when absent."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:  # pragma: no cover - importlib.metadata is stdlib on 3.8+
        return {p: "unknown" for p in packages}
    out: dict[str, str] = {}
    for p in packages:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = "not-installed"
        except Exception:
            out[p] = "unknown"
    return out


# ============================================================================
# manifest
# ============================================================================
@dataclass
class RunManifest:
    """Serializable record of the conditions under which an artifact was produced."""

    stage: str
    schema_version: str = _SCHEMA_VERSION
    timestamp_utc: str = ""
    git_sha: str | None = None
    git_dirty: bool | None = None
    seeds: dict[str, Any] = field(default_factory=dict)
    config_hash: str | None = None
    config_path: str | None = None
    inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    packages: dict[str, str] = field(default_factory=dict)
    python: str = ""
    platform: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_run_manifest(
    stage: str,
    *,
    seeds: dict[str, Any] | None = None,
    config: Any | None = None,
    config_path: str | Path | None = None,
    inputs: list[str | Path] | None = None,
    outputs: list[str | Path] | None = None,
    packages: tuple[str, ...] = DEFAULT_TRACKED_PACKAGES,
    extra: dict[str, Any] | None = None,
) -> RunManifest:
    """Assemble a `RunManifest`. `config` (a dict) is hashed; `config_path` (a file)
    is both recorded and, if `config` is not given, hashed by its file bytes.

    ``inputs`` / ``outputs`` are hashed by content so a manifest pins the exact
    bytes consumed and produced.
    """
    cfg_hash: str | None = None
    if config is not None:
        cfg_hash = config_hash(config)
    elif config_path is not None:
        cfg_hash = file_hash(config_path)

    return RunManifest(
        stage=stage,
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017
        git_sha=git_sha(),
        git_dirty=git_dirty(),
        seeds=dict(seeds or {}),
        config_hash=cfg_hash,
        config_path=str(config_path) if config_path is not None else None,
        inputs=[_file_record(p) for p in (inputs or [])],
        outputs=[_file_record(p) for p in (outputs or [])],
        packages=package_versions(packages),
        python=platform.python_version(),
        platform=platform.platform(),
        extra=dict(extra or {}),
    )


def write_run_manifest(path: str | Path, stage: str, **kwargs: Any) -> Path:
    """Build a manifest and write it as indented JSON to ``path``. Returns the path.

    ``**kwargs`` are forwarded to `build_run_manifest` (``seeds``, ``config``,
    ``config_path``, ``inputs``, ``outputs``, ``packages``, ``extra``).
    """
    manifest = build_run_manifest(stage, **kwargs)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return p


__all__ = [
    "RunManifest",
    "build_run_manifest",
    "write_run_manifest",
    "config_hash",
    "file_hash",
    "git_sha",
    "git_dirty",
    "package_versions",
    "DEFAULT_TRACKED_PACKAGES",
]
