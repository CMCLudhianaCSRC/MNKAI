# Contributing — reproducibility, QC, and CI

This project is **rigor-first**: every headline number must be reconstructable, and
the specific bugs that already bit us (all-zero features from a disconnected image
drive, ~33% train/test leakage, double-balancing collapse) must be *catchable by a
script*, not just by memory. This document covers the infrastructure that enforces
that. All of it is data-cheap and runs on the laptop.

## The reproducibility toolkit

| Tool | What it does | Touches the image drive? |
|---|---|---|
| `scripts/reproduce.py` | Documents + pre-flights the exact pipeline order, seeds, inputs/outputs. Never executes heavy stages. | No |
| `scripts/qc_checks.py` | Validates a feature cache is non-degenerate; audits two splits for near-duplicate leakage. | No |
| `src/mnkai/utils/run_manifest.py` | Writes a per-run JSON provenance record (git SHA, seeds, config + input hashes, package versions). | No |
| `.github/workflows/ci.yml` | Lint + byte-compile smoke + the data-free pytest suite on every push/PR. | No |

Everything operates on **already-cached** artifacts (`eval/probe/*.npz`,
`eval/probe/ddi_backbones/*.npy`). Nothing here re-extracts features, loads an image,
reads `D:/`, or regenerates a model.

## 1. `reproduce.py` — the canonical recipe

`reproduce.py` is the single source of truth for *how the numbers are produced and in
what order*. It describes two sub-pipelines:

- **A (`25class`)**: `cache_features → probe_multiclass → trust_refresh → save_deployable`
- **B (`ddi`)**: `ddi_ensemble → ddi_ensemble_ci`

```bash
python scripts/reproduce.py plan              # print the ordered recipe (nothing runs)
python scripts/reproduce.py check             # verify inputs exist + QC-validate caches
python scripts/reproduce.py manifest --out eval/reproduce   # write per-stage manifests
```

It intentionally does **not** run the stages. Regenerating `models/clinical_25class.pkl`
or any metric is a deliberate, controlled operation you perform by hand from the printed
plan — not a side effect of a convenience script.

## 2. `qc_checks.py` — the bug-catchers

**Validation** — is a feature cache non-degenerate? Fails loud on the exact
disconnected-drive signature (all-zero / constant / NaN / a spike of all-zero rows),
a shape/label regression, or an empty class:

```bash
python scripts/qc_checks.py validate --cache eval/probe/feats_25class_518.npz
python scripts/qc_checks.py validate --cache eval/probe/ddi_backbones/dinov2_b.npy
```

**Leakage audit** — frozen-feature near-duplicate detection between a train split and
an external / OOF split (cosine nearest-neighbour). This is how the ~33% cross-source
atlas-image leakage was found and re-confirmed:

```bash
# split-aware (one npz that carries a 'split' array)
python scripts/qc_checks.py leak-audit --cache eval/probe/feats_25class_518.npz \
    --train-split train_val --external-split external_test --threshold 0.98

# two separate feature files
python scripts/qc_checks.py leak-audit --train A.npy --external B.npy --threshold 0.98
```

Both commands **exit non-zero on failure**, so they double as CI / pre-run gates. Add
`--json` for a machine-readable report. The module is importable too:

```python
from qc_checks import validate_feature_cache, leakage_audit, load_features
rep = validate_feature_cache(load_features("eval/probe/feats_25class_518.npz"))
assert rep.ok, rep.problems
```

## 3. Run manifests — provenance per artifact

Write a manifest next to any artifact so a number can be traced to the exact inputs
that produced it. Stdlib-only, no heavy imports:

```python
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
```

A manifest records: `git_sha` + `git_dirty`, `seeds`, a canonical `config_hash`,
content hashes (`sha256_16`) of every input/output, package versions of the libraries
that move results (numpy/scipy/scikit-learn/pandas/torch/torchvision/timm/Pillow), the
Python version, and the platform. `config_hash` is order-independent, so the same
logical config always hashes the same string.

## 4. Continuous integration

`.github/workflows/ci.yml` runs on every push / PR to `main`:

1. **lint + compile smoke** — `ruff check src scripts` and `python -m compileall src scripts`.
2. **pytest (data-free)** — installs the CPU-only science stack and runs `tests/`.

The suite is data-free: tests needing an optional dependency or real data self-skip via
`pytest.importorskip`. Nothing in CI reads the image drive or downloads datasets.

## Local checks before a PR

```bash
# lint (matches CI)
ruff check src scripts

# data-free tests
PYTHONPATH=src pytest -q tests

# targeted: just the reproducibility/QC infra
PYTHONPATH=src pytest -q tests/test_qc_checks.py tests/test_run_manifest.py
```

## Guardrails (do not regress these)

- **Additive / safe changes only.** Do not regenerate the shipped model or any metric
  as a side effect. Model / metric regeneration is a controlled, deliberate re-run.
- **The external test set stays pristine** — never tuned on. Run `qc_checks leak-audit`
  whenever a split changes.
- **One rebalancing mechanism at a time** (a sampler stacked on a reweighted loss was
  the original collapse).
- **Never read/echo/modify `secrets/`** or touch `D:/`.
- After any model-touching step, re-fit temperature + conformal and re-confirm the
  melanoma high-sensitivity operating point.
