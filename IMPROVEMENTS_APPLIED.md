# IMPROVEMENTS_APPLIED — safe auto-apply batch (Agent 5, project improvement)

Traceable log of the additive, result-preserving changes applied from
`reports/review/04_consolidated_recommendations.md`. Scope was strictly the
synthesis-approved **auto-apply set** (R09, R11, R13-partial, R18, R22) plus two
pure-correctness items (R15 alias collision + guard) and a data-free pytest suite.

**Explicitly NOT done here** (result/artifact-regenerating or require a run):
R01, R02 (determinism re-run), R03, R04, R05, R06, R07, R08, R10, R12, R14, R16,
R17, R19, R20, R21. No training, feature-extraction, or heavy compute was run;
no model/`*.pkl`, no `data/`/results, and no `D:/` paths were modified.

Verification: every edited file byte-compiles (`python -m py_compile`); the new
test invariants were executed against the real code with lightweight stubs for the
sandbox-absent deps (loguru/scipy) — all six groups pass. `pytest`, `scikit-learn`,
`scipy`, `loguru`, and `timm` are not installed in the sandbox, so the suite itself
could not be run under `pytest` here; it is written to run in the project env
(deps pinned in `requirements.txt`), and each test uses `pytest.importorskip` to
degrade gracefully.

---

## R09 — Cite Kumar et al. 2022 (LP-FT, ICLR) + reframe novelty
- **File:** `PAPER_DRAFT.md`
  - **What:** Reframed abstract contribution (4) and §7 to position the
    frozen-probe finding as a *clinical, data-scarce, dark-skin demonstration* of
    the established fine-tuning-distortion effect (not a new phenomenon); added an
    explicit "Relation to prior work" paragraph in §7; added an LP-FT future-work
    note in §8 (Limitations); added a **References** section (Kumar 2022, DINOv2,
    Angelopoulos & Bates conformal, DDI/Daneshjou).
  - **Why:** Synthesis conflict-resolution #1 — keep the thesis but reposition
    novelty onto the deployable trust stack + dark-skin demonstration; cite the
    prior art the reviewers flagged as missing. Pure documentation edit.
- **File:** `PROJECT_REPORT.md`
  - **What:** Added a paragraph to "Honest framing" stating the frozen-vs-fine-tune
    effect itself is not claimed novel (cites Kumar et al. 2022), and that novelty
    is the clinical dark-skin demonstration + trust stack; noted LP-FT as future work.
  - **Why:** Same as above; keep the two headline docs consistent.

## R11 — Guard the manifest↔splits inner merge
- **File:** `src/mnkai/data/splits.py`
  - **What:** Added `merge_manifest_splits(manifest, splits, *, min_keep_frac=0.90)`:
    normalizes the join key (`filepath` preferred, else `isic_id`) on BOTH sides
    (whitespace strip; `filepath` backslash→forward-slash), does the inner merge,
    logs dropped-row counts, and raises `RuntimeError` if <90% of rows survive
    (or 0 rows) or `ValueError` if no shared key exists. Added to `__all__`.
  - **Why:** A path-format mismatch previously shrank the training set silently.
    Fail-loud guard; no behaviour change on a valid manifest/splits pair.
- **File:** `src/mnkai/training/train.py`
  - **What:** Replaced the inline `manifest.merge(...)` block (train.py ~L287) with
    a call to `merge_manifest_splits(...)`; imported it from `mnkai.data.splits`.
  - **Why:** Reuse the guarded, unit-testable helper; the logic is behaviour-
    identical on valid data. (Guard lives in the lightweight `splits` module so it
    is testable without importing the heavy trainer.)

## R13 (partial, safe part) — Reproducibility package
- **File:** `requirements.txt`
  - **What:** Restored from git (was deleted in the working tree) and pinned:
    lower bounds kept (tested versions) plus upper bounds capping the next MAJOR
    to prevent silent breaking drift; added missing runtime dep `loguru`, plus
    `imagehash`, `fastapi`, `uvicorn`, and dev `pytest`; added a header documenting
    the exact-freeze procedure (`pip freeze > requirements.lock.txt`).
  - **Why:** Reproducibility; the sandbox is not the project env so exact `==`
    pins were not fabricated — ranges + a documented lock procedure instead.
- **File:** `scripts/_probe_backbone.py`
  - **What:** Removed the hardcoded `DDI = Path("D:/AIFINAL/images/ddi")`; added a
    `--ddi-root` CLI arg and `MNKAI_DDI_ROOT` env var, with the original D: path as
    the default; `main()` now reads `--ddi-root`.
  - **Why:** Portability; the D: path is no longer baked into source. Default
    preserves existing behaviour, so no result change.
- **File:** `docs/MODEL_CARD.md` (new)
  - **What:** Authored the model-card skeleton for `models/clinical_25class.pkl`:
    intended use, provenance/data, honest subgroup performance (incl. the DDI
    ensemble AUROC), operating point + decision logic, failure modes,
    reproducibility, and the "decision support, not diagnosis" framing. Class list
    read from `eval/probe/feats_25class_518.npz` (data-only read; the `.pkl` was
    not opened/modified).
  - **Why:** R13 reproducibility/governance deliverable. New documentation only.

## R18 — Backbone TypeError fallback should not silently drop `img_size`
- **File:** `src/mnkai/models/backbones.py`
  - **What:** Added `_create_model_with_fallback(name, kwargs)`: on `TypeError`,
    retries after dropping `drop_path_rate` first; if still failing and `img_size`
    is set, re-raises with a diagnostic (refuses to silently drop `img_size`).
    Used it in both `build_image_only` and `ImageTabularNet.__init__`, replacing
    the old "pop img_size and drop_path_rate together" fallback.
  - **Why:** The old fallback could silently downgrade DINOv2 518→default
    resolution, changing feature geometry. Only the exception path changed; the
    success path (shipped DINOv2@518) is untouched → no result change.

## R22 — Static code-hygiene bundle (safe, additive)
- **File:** `src/mnkai/training/train.py`
  - **What:** Removed the dead ternary `best_macro_f1 if is_best else best_macro_f1`
    → `best_macro_f1` (checkpoint dict).
- **File:** `src/mnkai/data/datasets.py`
  - **What:** Moved the per-call `import time, random` (inside `SkinDataset.__getitem__`)
    to module scope; left a comment noting the module-scope import.
- **File:** `src/mnkai/data/download/isic.py`
  - **What:** Replaced the `rows = [{...}][:0]` dead-list build (builds a dict then
    slices to empty) with `rows = []` plus a clarifying comment.
- **File:** `src/mnkai/data/qc.py`
  - **What:** Fixed `blur_score` docstring (and the module-header line): it uses
    PIL `FIND_EDGES`, not a true Laplacian — documented as a Laplacian-style
    sharpness proxy.
- **File:** `src/mnkai/evaluation/abstention.py`
  - **What:** Fixed `threshold_for_target_error` docstring: it never raises; it
    always returns a `CoveragePoint` via the documented fallback.
  - **Why (all R22):** Docstring/dead-code hygiene only; zero behaviour change.

## R15 — Resolve the `exanthema` alias collision + add a collision guard
- **File:** `src/mnkai/data/manifest.py`
  - **What:** Removed the duplicate `"exanthema"` alias from `viral_infection`
    (it was also claimed by `drug_eruption`, which already won at runtime as the
    later dict key), leaving `drug_eruption` as the sole owner — mapping unchanged.
    Added a collision guard in `LabelNormalizer.__post_init__` that records all
    claimants per alias (`self._alias_claims`) and logs a warning for any alias
    claimed by >1 canonical class, reporting the last-wins resolution.
  - **Why:** Pure correctness fix + guard. Because `drug_eruption` already won,
    removing the duplicate changes no label mapping; the guard surfaces any future
    collision instead of it silently mislabeling images.

## New tests — ML-critical invariants (data-free, cheap)
- **File:** `tests/test_improvements.py` (new)
  - **What:** pytest suite on tiny synthetic arrays/frames (no real data, no model
    download) pinning: (i) `build_loss` dispatch + `effective_number_weights` math
    + LDAM/cb-focal weight normalization; (ii) `class_balanced_weights` power
    semantics; (iii) group-aware split zero train/val group overlap +
    `assert_no_group_leakage` detection; (iv) `_raps_set` set-size monotonicity in
    the threshold + the temperature/bias transform (`_calibrated_adjust`) in
    `demo_app/pipeline.py`; (v) the alias-collision guard (incl. `exanthema`→
    `drug_eruption`); (vi) the `merge_manifest_splits` guard (happy path with
    separator normalization, key-mismatch raises, missing-key raises). Each test
    uses `pytest.importorskip` for its optional deps.
  - **Why:** Lock the invariants the review flagged so future edits can't silently
    break them.
- **File:** `demo_app/pipeline.py`
  - **What:** Made `timm` import lazy (moved into `_load`), removed the top-level
    `scipy.special` import, and extracted the temperature+bias transform into a
    pure module-level `_calibrated_adjust(p, temperature, bias)` that `assess`
    now calls (behaviour-identical). This lets the pure helpers be imported and
    unit-tested without the backbone/scipy stack.
  - **Why:** Testability for invariant (iv); zero change to serving behaviour
    (same operations, same order; heavy imports merely deferred).

---

# Agent 7 — Pipeline enhancement (reproducibility + QC infra)

Additive, safe infrastructure whose purpose is to **catch the real bugs this
project hit** (all-zero features from a disconnected D: drive; ~33% train/test
leakage; double-balancing collapse) and to make every run reconstructable. No
training, feature-extraction, or heavy compute was run; no model/`*.pkl`, no
`data/`/results, no `D:/`, and no `secrets/` were touched. Maps to the roadmap's
reproducibility package (`reports/review/09_prioritized_roadmap.md`, R02/R06/R13)
and the gap analysis' missing QC/leakage tooling.

**Verification:** `python -m py_compile` on all new modules; `ruff check` clean on
all new files (project `pyproject.toml` ruleset); the 23 new tests pass under the
project venv; `qc_checks` and `reproduce` CLIs smoke-tested against the real cached
features read-only (leak-audit correctly re-detected the known 25-class external
near-duplicates; validate PASSed all `eval/probe` caches).

## scripts/qc_checks.py (new) — feature-cache QC + leakage audit
- **What:** Importable module + CLI operating on **already-cached** features only
  (`.npz`/`.npy`), never the image drive. Two jobs:
  1. `validate_feature_cache(...)` — fails loud on the disconnected-drive signature
     (all-zero / constant matrix via exact peak-to-peak range; NaN/inf; a spike of
     all-zero rows; too many dead dims), plus shape/`expected_dims` mismatch and
     empty-class checks; extreme class imbalance is a soft warning.
  2. `leakage_audit(train, external, threshold=0.98)` — frozen-feature cosine
     nearest-neighbour near-duplicate detection between two splits (chunked, memory-
     bounded), reporting flagged count/fraction, cosine percentiles, and worst
     offenders. `max_cosine_to_reference(...)` exposed for reuse.
  - CLI: `qc_checks.py validate --cache ...` and `qc_checks.py leak-audit --cache ...
    --train-split train_val --external-split external_test` (or `--train/--external`
    file pair); `--json` for machine-readable output; non-zero exit on failure so it
    doubles as a CI / pre-run gate. NumPy-only dependency.
- **Why:** R06 (frozen-feature duplicate audit) + the missing "would have caught the
  all-zero cache" QC gate. Reusable across `eval/probe/*.npz` and
  `eval/probe/ddi_backbones/*.npy`.

## scripts/reproduce.py (new) — single documented pipeline entrypoint
- **What:** Canonical, machine-readable description of the frozen-probe pipeline order
  with seed pinning and config/input hashing. Sub-pipeline A
  (`cache_features → probe_multiclass → trust_refresh → save_deployable`) and B
  (`ddi_ensemble → ddi_ensemble_ci`). Three sub-commands: `plan` (print the recipe),
  `check` (verify declared inputs exist + QC-validate caches via `qc_checks`),
  `manifest` (write per-stage run manifests). **Deliberately does not execute** any
  heavy stage or regenerate the model. cp1252-safe stdout.
- **Why:** R13 reproducibility package — a run is reconstructable from one documented
  entrypoint; pre-flight would have caught the missing/degenerate-cache failures.

## src/mnkai/utils/run_manifest.py (new) — structured run-manifest helper
- **What:** Importable, stdlib-only (no heavy deps). `build_run_manifest(...)` /
  `write_run_manifest(...)` emit per-run JSON with git SHA + dirty flag, seeds,
  canonical order-independent `config_hash`, `sha256_16` content hashes of every
  input/output, tracked package versions (numpy/scipy/scikit-learn/pandas/torch/
  torchvision/timm/Pillow), Python version, and platform. Environment probing is
  fail-soft (missing git / uninstalled package never raises).
- **Why:** R02/R13 — provenance so a number is traceable to the exact inputs and code
  that produced it. Added to `utils` alongside `seed.py`/`config.py`.

## .github/workflows/ci.yml (new) — data-free CI
- **What:** On push/PR to `main` (+ manual): a `lint-smoke` job (`ruff check src
  scripts` + `python -m compileall src scripts`) and a `tests` job (CPU-only torch
  wheel + core science stack, then `pytest tests`). Data-free: optional-dep/real-data
  tests self-skip via `pytest.importorskip`; nothing reads the image drive.
- **Why:** R13 — automated guard so the QC/reproducibility invariants can't silently
  regress.

## tests/test_qc_checks.py + tests/test_run_manifest.py (new)
- **What:** 23 data-free tests on synthetic arrays. QC: healthy-passes, all-zero /
  constant / non-finite / partial-zero-rows / wrong-ndim / dim-mismatch / empty-class
  failures, imbalance warning, leakage detects exact-duplicate rows (with correct
  argmax back-pointers) and passes on disjoint data, dim-mismatch raises, npz/npy load
  round-trip, and both CLIs (validate exit 0, leak-audit exit 1 on injected leaks).
  Manifest: stable order-independent config hash, int/float hash normalization, file
  hash round-trip + missing→None, package-version known/unknown, full manifest field
  capture, JSON write round-trip, config-path fallback hashing.
- **Why:** Lock the invariants so future edits can't silently break the bug-catchers.

## Docs + hygiene
- **File:** `README.md` — rewrote "Reproduce the science" to lead with
  `reproduce.py plan/check/manifest` + `qc_checks validate/leak-audit` as the
  pre-flight, then the ordered manual stages (track A then B); pointer to
  `CONTRIBUTING.md`.
- **File:** `CONTRIBUTING.md` (new) — reproducibility/QC/CI reference: the toolkit
  table, `reproduce.py`, `qc_checks.py`, run manifests, CI, local pre-PR checks, and
  the do-not-regress guardrails.
- **File:** `.gitignore` — ignore `eval/reproduce/` (per-run manifests are regenerated
  on demand and carry absolute local paths).
