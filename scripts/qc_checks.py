"""Feature-cache QC + frozen-feature leakage audit (data-cheap, importable + CLI).

This module exists to *catch the real bugs this project actually hit* before they
silently poison a metric:

  * **all-zero / degenerate features** — when the D: image drive was disconnected,
    feature extraction produced all-zero (or constant) embeddings and everything
    downstream still "ran". `validate_feature_cache` fails loud on that.
  * **train/test leakage** — ~33% of the 25-class external test were near-duplicate
    cross-source atlas images of training rows. `leakage_audit` re-finds that class
    of bug via frozen-feature cosine nearest-neighbour between two splits.

Both jobs operate on the **already-cached** features
(`eval/probe/*.npz`, `eval/probe/ddi_backbones/*.npy`). Nothing here touches the
image drive (D:), loads an image, or re-extracts features.

CLI
---
    # validate one cache (npz cache or bare .npy feature matrix)
    python scripts/qc_checks.py validate --cache eval/probe/feats_25class_518.npz
    python scripts/qc_checks.py validate --cache eval/probe/ddi_backbones/dinov2_b.npy

    # leakage audit between two splits *inside* one npz cache (uses the 'split' array)
    python scripts/qc_checks.py leak-audit --cache eval/probe/feats_25class_518.npz \
        --train-split train_val --external-split external_test --threshold 0.98

    # leakage audit between two separate feature files
    python scripts/qc_checks.py leak-audit --train A.npy --external B.npy --threshold 0.98

Exit code is non-zero when a hard check fails, so this doubles as a CI / pre-run gate.

Importable
----------
    from qc_checks import (
        load_features, validate_feature_cache, leakage_audit,
        ValidationReport, LeakageReport,
    )

Only hard dependency is NumPy (a core project dep).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ------- keys we accept when pulling arrays out of a saved cache ---------------
_FEATURE_KEYS = ("features", "feats", "X", "embeddings", "emb")
_LABEL_KEYS = ("y", "labels", "label", "target")
_SPLIT_KEYS = ("split", "splits")
_CLASS_KEYS = ("classes", "class_names")

_EPS = 1e-8


# ============================================================================
# loading
# ============================================================================
@dataclass
class LoadedCache:
    """A feature cache resolved into arrays, regardless of source format."""

    features: np.ndarray
    y: np.ndarray | None = None
    split: np.ndarray | None = None
    classes: np.ndarray | None = None
    source: str = ""

    @property
    def n(self) -> int:
        return int(self.features.shape[0]) if self.features.ndim >= 1 else 0


def _first_key(store, keys: tuple[str, ...]):
    for k in keys:
        if k in store:
            return store[k]
    return None


def load_features(path: str | Path) -> LoadedCache:
    """Load a `.npz` cache or a bare `.npy` feature matrix into a `LoadedCache`.

    `.npz` is expected to carry a feature matrix under one of `_FEATURE_KEYS`, and
    *optionally* labels / split / class-name arrays. A `.npy` is treated as the
    feature matrix alone.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"feature cache not found: {path}")

    if path.suffix == ".npy":
        feats = np.load(path, allow_pickle=False)
        return LoadedCache(features=np.asarray(feats), source=str(path))

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as store:
            feats = _first_key(store, _FEATURE_KEYS)
            if feats is None:
                raise KeyError(
                    f"{path}: none of {_FEATURE_KEYS} present; keys = {list(store.files)}"
                )
            y = _first_key(store, _LABEL_KEYS)
            split = _first_key(store, _SPLIT_KEYS)
            classes = _first_key(store, _CLASS_KEYS)
            return LoadedCache(
                features=np.asarray(feats),
                y=None if y is None else np.asarray(y),
                split=None if split is None else np.asarray(split),
                classes=None if classes is None else np.asarray(classes),
                source=str(path),
            )

    raise ValueError(f"unsupported cache format {path.suffix!r} (expected .npy or .npz)")


# ============================================================================
# validation
# ============================================================================
@dataclass
class ValidationReport:
    """Structured result of `validate_feature_cache`. `ok` is the go/no-go flag."""

    source: str
    ok: bool
    n_rows: int
    n_dims: int
    problems: list[str] = field(default_factory=list)      # hard failures -> ok=False
    warnings: list[str] = field(default_factory=list)      # soft flags -> ok stays True
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_feature_cache(
    cache: LoadedCache | np.ndarray | str | Path,
    *,
    y: np.ndarray | None = None,
    expected_dims: int | None = None,
    expected_rows: int | None = None,
    max_zero_row_frac: float = 0.01,
    max_dead_dim_frac: float = 0.10,
    min_per_class: int = 1,
    warn_imbalance_ratio: float = 200.0,
) -> ValidationReport:
    """Validate that a frozen-feature cache is non-degenerate and shaped as expected.

    Hard failures (set ``ok=False``):
      * features are not a 2-D array;
      * any non-finite value (NaN / inf) — a corrupt or partial extraction;
      * the whole matrix is all-zero or constant — the disconnected-drive signature;
      * more than ``max_zero_row_frac`` of rows are all-zero;
      * more than ``max_dead_dim_frac`` of feature dims are constant (zero variance);
      * an ``expected_dims`` / ``expected_rows`` mismatch, if supplied;
      * a class with fewer than ``min_per_class`` samples, if labels are available.

    Soft warnings (do not fail): extreme but plausible class imbalance
    (max/min count above ``warn_imbalance_ratio``).

    Accepts a `LoadedCache`, a raw ndarray, or a path (loaded via `load_features`).
    """
    if isinstance(cache, (str, Path)):
        cache = load_features(cache)
    if isinstance(cache, np.ndarray):
        cache = LoadedCache(features=np.asarray(cache), source="<ndarray>")
    if y is None:
        y = cache.y

    feats = np.asarray(cache.features)
    problems: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {}

    n_rows = int(feats.shape[0]) if feats.ndim >= 1 else 0
    n_dims = int(feats.shape[1]) if feats.ndim >= 2 else 0

    # --- shape ---------------------------------------------------------------
    if feats.ndim != 2:
        problems.append(f"features must be 2-D (rows x dims); got shape {feats.shape}")
        return ValidationReport(cache.source, False, n_rows, n_dims, problems, warnings, stats)
    if n_rows == 0 or n_dims == 0:
        problems.append(f"empty feature matrix: shape {feats.shape}")
        return ValidationReport(cache.source, False, n_rows, n_dims, problems, warnings, stats)
    if expected_dims is not None and n_dims != expected_dims:
        problems.append(f"dim mismatch: expected {expected_dims}, got {n_dims}")
    if expected_rows is not None and n_rows != expected_rows:
        problems.append(f"row mismatch: expected {expected_rows}, got {n_rows}")

    # --- finiteness ----------------------------------------------------------
    finite = np.isfinite(feats)
    n_nonfinite = int(finite.size - finite.sum())
    stats["n_nonfinite"] = n_nonfinite
    if n_nonfinite:
        problems.append(f"{n_nonfinite} non-finite (NaN/inf) values present")

    # work on a finite-safe view for the numeric summaries below
    safe = np.where(finite, feats, 0.0)
    abs_max = float(np.abs(safe).max())
    # peak-to-peak (max-min) is exact for identical values, unlike float32 std
    global_range = float(safe.max() - safe.min())
    stats["abs_max"] = abs_max
    stats["global_range"] = global_range
    stats["global_std"] = float(safe.std())
    stats["mean"] = float(safe.mean())

    # --- all-zero / constant matrix (the disconnected-drive signature) -------
    if abs_max <= _EPS:
        problems.append("features are all-zero (max |x| ~ 0) — likely a disconnected "
                        "image source / failed extraction")
    elif global_range <= _EPS:
        problems.append("features are constant (zero variance across the matrix) — "
                        "degenerate cache")

    # --- fraction of all-zero rows ------------------------------------------
    row_absmax = np.abs(safe).max(axis=1)
    zero_rows = int((row_absmax <= _EPS).sum())
    zero_row_frac = zero_rows / n_rows
    stats["zero_rows"] = zero_rows
    stats["zero_row_frac"] = round(zero_row_frac, 6)
    if zero_row_frac > max_zero_row_frac:
        problems.append(
            f"{zero_rows} all-zero rows ({zero_row_frac:.2%} > "
            f"{max_zero_row_frac:.2%} allowed) — partial extraction failure"
        )

    # --- dead (constant) dimensions -----------------------------------------
    dim_range = safe.max(axis=0) - safe.min(axis=0)
    dead_dims = int((dim_range <= _EPS).sum())
    dead_dim_frac = dead_dims / n_dims
    stats["dead_dims"] = dead_dims
    stats["dead_dim_frac"] = round(dead_dim_frac, 6)
    if dead_dim_frac > max_dead_dim_frac:
        problems.append(
            f"{dead_dims} constant feature dims ({dead_dim_frac:.2%} > "
            f"{max_dead_dim_frac:.2%} allowed)"
        )

    # --- class balance -------------------------------------------------------
    if y is not None:
        y = np.asarray(y).ravel()
        if y.shape[0] != n_rows:
            problems.append(f"label length {y.shape[0]} != n_rows {n_rows}")
        else:
            uniq, counts = np.unique(y, return_counts=True)
            stats["n_classes"] = int(uniq.shape[0])
            stats["class_counts"] = {str(k): int(v)
                                     for k, v in zip(uniq.tolist(), counts.tolist(), strict=False)}
            stats["min_class_count"] = int(counts.min())
            stats["max_class_count"] = int(counts.max())
            if counts.min() < min_per_class:
                empties = [str(k) for k, v in zip(uniq.tolist(), counts.tolist(), strict=False)
                           if v < min_per_class]
                problems.append(f"class(es) below {min_per_class} samples: {empties}")
            ratio = counts.max() / max(counts.min(), 1)
            stats["imbalance_ratio"] = round(float(ratio), 2)
            if ratio > warn_imbalance_ratio:
                warnings.append(
                    f"extreme class imbalance: max/min = {ratio:.0f}x "
                    f"(> {warn_imbalance_ratio:.0f}x) — check sampling/weights"
                )

    ok = len(problems) == 0
    return ValidationReport(cache.source, ok, n_rows, n_dims, problems, warnings, stats)


# ============================================================================
# leakage audit
# ============================================================================
@dataclass
class LeakageReport:
    """Structured result of `leakage_audit`. `ok` is False when leakage exceeds tol."""

    ok: bool
    threshold: float
    n_train: int
    n_external: int
    n_flagged: int                 # external rows whose max cosine to train > threshold
    flagged_frac: float
    max_cosine: float
    offenders: list[dict[str, Any]] = field(default_factory=list)  # worst external rows
    percentiles: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms <= _EPS, 1.0, norms)
    return x / norms


def max_cosine_to_reference(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    chunk: int = 2048,
    return_argmax: bool = False,
):
    """For each `query` row, the max cosine similarity to any `reference` row.

    Chunked matmul over normalized vectors — O(n_query * n_ref) time but bounded
    memory (``chunk * n_ref`` per step). Returns the per-query max (and, optionally,
    the index of the nearest reference row).
    """
    q = _l2_normalize(query)
    r = _l2_normalize(reference)
    n = q.shape[0]
    best = np.empty(n, dtype=np.float32)
    arg = np.empty(n, dtype=np.int64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sims = q[s:e] @ r.T                     # (e-s, n_ref)
        best[s:e] = sims.max(axis=1)
        if return_argmax:
            arg[s:e] = sims.argmax(axis=1)
    if return_argmax:
        return best, arg
    return best


def leakage_audit(
    train: np.ndarray,
    external: np.ndarray,
    *,
    threshold: float = 0.98,
    top_k: int = 25,
    max_flagged_frac: float = 0.0,
    chunk: int = 2048,
) -> LeakageReport:
    """Frozen-feature near-duplicate audit of an external split against a train split.

    For every external row we compute its maximum cosine similarity to any training
    row; rows above ``threshold`` are flagged as probable train duplicates (the
    cross-source atlas-image leakage this project hit). ``ok`` is False when the
    flagged fraction exceeds ``max_flagged_frac`` (default 0.0 = any leak fails).

    Returns counts, the flagged fraction, a similarity percentile summary, and the
    ``top_k`` worst offenders (external index, nearest train index, cosine).
    """
    train = np.asarray(train)
    external = np.asarray(external)
    if train.ndim != 2 or external.ndim != 2:
        raise ValueError("train and external must both be 2-D feature matrices")
    if train.shape[1] != external.shape[1]:
        raise ValueError(
            f"dim mismatch: train {train.shape[1]} vs external {external.shape[1]}"
        )

    best, arg = max_cosine_to_reference(external, train, chunk=chunk, return_argmax=True)
    flagged_mask = best > threshold
    n_flagged = int(flagged_mask.sum())
    flagged_frac = n_flagged / max(external.shape[0], 1)

    order = np.argsort(-best)[:top_k]
    offenders = [
        {
            "external_idx": int(i),
            "nearest_train_idx": int(arg[i]),
            "cosine": round(float(best[i]), 6),
        }
        for i in order
        if best[i] > threshold or top_k > 0  # always list the worst, mark via cosine
    ]

    pct = {
        f"p{p}": round(float(np.percentile(best, p)), 6)
        for p in (50, 90, 99, 100)
    }

    ok = flagged_frac <= max_flagged_frac
    return LeakageReport(
        ok=ok,
        threshold=threshold,
        n_train=int(train.shape[0]),
        n_external=int(external.shape[0]),
        n_flagged=n_flagged,
        flagged_frac=round(float(flagged_frac), 6),
        max_cosine=round(float(best.max()), 6),
        offenders=offenders,
        percentiles=pct,
    )


# ============================================================================
# CLI
# ============================================================================
def _cmd_validate(args: argparse.Namespace) -> int:
    cache = load_features(args.cache)
    rep = validate_feature_cache(
        cache,
        expected_dims=args.expected_dims,
        max_zero_row_frac=args.max_zero_row_frac,
        max_dead_dim_frac=args.max_dead_dim_frac,
    )
    _emit(rep.to_dict(), args.json)
    if not args.json:
        print(f"\n[validate] {rep.source}")
        print(f"  shape: {rep.n_rows} x {rep.n_dims}  ok={rep.ok}")
        for w in rep.warnings:
            print(f"  WARN: {w}")
        for p in rep.problems:
            print(f"  FAIL: {p}")
        print(f"  => {'PASS' if rep.ok else 'FAIL'}")
    return 0 if rep.ok else 1


def _cmd_leak_audit(args: argparse.Namespace) -> int:
    if args.cache:
        cache = load_features(args.cache)
        if cache.split is None:
            print(f"FAIL: {args.cache} has no 'split' array; use --train/--external instead",
                  file=sys.stderr)
            return 2
        split = np.asarray(cache.split).astype(str)
        tr_mask = split == args.train_split
        ex_mask = split == args.external_split
        if tr_mask.sum() == 0 or ex_mask.sum() == 0:
            print(f"FAIL: split values found = {sorted(set(split.tolist()))}; "
                  f"train='{args.train_split}' (n={int(tr_mask.sum())}) "
                  f"external='{args.external_split}' (n={int(ex_mask.sum())})",
                  file=sys.stderr)
            return 2
        train = cache.features[tr_mask]
        external = cache.features[ex_mask]
    else:
        train = load_features(args.train).features
        external = load_features(args.external).features

    rep = leakage_audit(
        train, external,
        threshold=args.threshold, top_k=args.top_k,
        max_flagged_frac=args.max_flagged_frac,
    )
    _emit(rep.to_dict(), args.json)
    if not args.json:
        print(f"\n[leak-audit] train={rep.n_train}  external={rep.n_external}  "
              f"threshold={rep.threshold}")
        print(f"  flagged (cosine>{rep.threshold}): {rep.n_flagged} "
              f"({rep.flagged_frac:.2%})   max cosine={rep.max_cosine}")
        print(f"  cosine percentiles: {rep.percentiles}")
        if rep.offenders:
            print(f"  worst offenders (up to {args.top_k}):")
            for o in rep.offenders[:10]:
                print(f"    ext[{o['external_idx']}] ~ train[{o['nearest_train_idx']}] "
                      f"cos={o['cosine']}")
        print(f"  => {'PASS' if rep.ok else 'LEAKAGE DETECTED'}")
    return 0 if rep.ok else 1


def _emit(obj: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, indent=2))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="qc_checks",
        description="Feature-cache QC + frozen-feature leakage audit "
                    "(operates on cached features only; never touches the image drive).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="check a cache is non-degenerate")
    v.add_argument("--cache", required=True, help=".npz cache or .npy feature matrix")
    v.add_argument("--expected-dims", type=int, default=None)
    v.add_argument("--max-zero-row-frac", type=float, default=0.01)
    v.add_argument("--max-dead-dim-frac", type=float, default=0.10)
    v.add_argument("--json", action="store_true", help="emit the report as JSON")
    v.set_defaults(func=_cmd_validate)

    la = sub.add_parser("leak-audit", help="frozen-feature near-duplicate audit between splits")
    la.add_argument("--cache", help="one .npz containing a 'split' array")
    la.add_argument("--train-split", default="train_val")
    la.add_argument("--external-split", default="external_test")
    la.add_argument("--train", help="feature file for the train split (alt to --cache)")
    la.add_argument("--external", help="feature file for the external split (alt to --cache)")
    la.add_argument("--threshold", type=float, default=0.98)
    la.add_argument("--top-k", type=int, default=25)
    la.add_argument("--max-flagged-frac", type=float, default=0.0,
                    help="tolerated leak fraction before failing (default 0 = any leak fails)")
    la.add_argument("--json", action="store_true", help="emit the report as JSON")
    la.set_defaults(func=_cmd_leak_audit)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
