# Agent 1A — Code Review: Core Library (`src/mnkai/`)

Scope: every module under `src/mnkai/` read start-to-finish. Focus: bugs, silent-failure
paths, numerical issues, ML-math correctness (loss factories, conformal, calibration,
group-aware splits), reproducibility, duplication, maintainability.

Overall: the library is well-structured and the *scientific* math (conformal APS,
temperature scaling, DeLong, class-balanced/logit-adjusted/LDAM losses, group-aware
splits) is correct and carefully cited. The highest-risk issues are **silent-failure
data paths** (the zero-image dataset fallback, and the manifest↔splits inner-join) and
two **silent kwarg/label collisions** (backbone `img_size` drop; the `exanthema` alias).
None of the confirmed issues change the *already-reported* frozen-probe headline numbers,
because that pipeline uses `scripts/_cache_features.py` + `_probe_*` rather than this
end-to-end trainer — but they matter for any run that *does* use this trainer, and for
maintainability/reproducibility claims in the paper.

---

## HIGH

### H1. `data/datasets.py` — zero-image silent fallback corrupts training on a dead drive
`SkinDataset.__getitem__` (lines 174-238). On `FileNotFoundError/OSError`, it retries 20×
(0.2 s sleep each) picking a *different same-label* row, then **returns an all-zero
`3×224×224` image carrying the ORIGINAL true label**. Consequences:

1. **Silent data corruption**: if the T7/USB drive disconnects, *every* read fails, so the
   dataset returns all-zero images with valid labels for the whole epoch. Training/eval
   proceeds with no error and no crash — exactly the failure this codebase already learned
   about. Only a single `logger.warning` per bad item marks it.
2. **Distribution skew**: the same-label retry (`random.choice(same_label)`, lines 208-212)
   substitutes a *different* readable sample for the requested index. Transient misses
   therefore silently bias which samples/classes are actually seen, independent of the
   sampler weights.
3. **Broken at the real resolution**: the fallback is hardcoded `torch.zeros(3, 224, 224)`
   (line 226), but the project trains DINOv2 at **518 px**. A zero image of the wrong size
   makes the default collate `torch.stack` raise a size-mismatch — so the "graceful" path
   actually *crashes* at 518 px instead of degrading, i.e. it doesn't do what it claims.
4. **Non-determinism**: uses the unseeded global `random` module, so retries are not
   reproducible.
5. **Hidden slowdowns**: 20 × 0.2 s = up to 4 s stall per bad item × workers reads as
   "slow training".

Fix: fail loud after N retries (raise), OR track a global miss-rate and abort the run when
it exceeds a small threshold; never substitute a zero image. If a placeholder is truly
wanted, size it from the transform output and return a mask/flag so the loss can ignore it.
Not auto-applyable (changes training/eval behavior).

---

## MEDIUM

### M1. `training/train.py` — manifest↔splits `inner` merge can silently drop rows
Lines 287-292: `manifest = manifest.merge(splits[["filepath","split","fold"]],
on="filepath", how="inner")`. If `filepath` formatting differs between the normalized
manifest and the splits CSV (absolute vs relative, `/` vs `\`, prefix differences), the
inner join silently yields few or zero matching rows. The run then trains on a silently
shrunken (or empty) set; empty folds produce NaN/degenerate metrics rather than an obvious
error. Add an assertion that the merged row count ≈ the pre-merge count (or ≥ some fraction),
and log dropped-row counts. Medium-high operational risk.

### M2. `losses/focal.py` — focal modulator detached under `torch.no_grad()`
Lines 117-118: `pt` (the true-class probability driving `(1-pt)^gamma`) is computed inside
`torch.no_grad()`, so the modulating factor is treated as a constant per-sample weight and
its gradient w.r.t. the logits is dropped. Canonical focal loss (Lin et al. 2017) keeps the
modulator in the graph; this is the "detached-weight" variant and optimizes a different
objective. The docstring frames the `no_grad` as a correctness fix for the label-smoothing
`pt`, but the side effect is a real change to the focal gradient. If intentional, document
it explicitly as a deviation from standard focal; if not, remove the `no_grad` (or detach
only for the smoothing-vs-hard-pt bookkeeping). Scientific — do NOT auto-apply.

### M3. `models/backbones.py` — TypeError fallback silently drops `img_size` with `drop_path_rate`
`ImageTabularNet.__init__` lines 128-134 (and analogously `build_image_only` 60-65): the
`except TypeError` retry pops **both** `drop_path_rate` and `img_size`. For DINOv2 the
resolution (518) is scientifically load-bearing; if `create_model` raises `TypeError` for
any reason, the retry silently rebuilds the backbone at its default resolution with no
error surfaced. Retry by removing kwargs one at a time (drop_path_rate first, re-raise if
img_size is genuinely rejected), or validate the resulting `default_cfg` resolution. Medium.

### M4. `data/manifest.py` — alias collision: `exanthema` maps to two classes
`DEFAULT_ALIASES`: `"exanthema"` is listed under both `viral_infection` (line 131) and
`drug_eruption` (line 134, alongside `"exanthem"`). `LabelNormalizer.__post_init__` builds
`self._table` by iterating the dict and letting later keys overwrite earlier ones, so
`exanthema` deterministically resolves to `drug_eruption` and the viral mapping is silently
shadowed. Any `exanthema`-labeled images are assigned to `drug_eruption` regardless of
intent. Resolve the overlap explicitly (pick one class, or add a collision check in
`__post_init__` that warns when an alias is claimed by >1 canonical class). Medium
data-labeling correctness. (A collision-detection guard is a safe additive check.)

### M5. `calibration/calibration.py` — small subgroups left at T=1 instead of global T
`GroupTemperatureScaler.fit` (lines 143-160): subgroups with `<30` samples `continue`,
leaving `log_T[j]=0` → T=1 (no calibration), while a fitted global temperature `_global`
already exists and would be a better estimate. Meanwhile `forward`/`temperature_for` only
fall back to global T for *unknown* keys, not for these known-but-tiny groups. Result: a
small FST subgroup gets *no* calibration even though the equitable-calibration goal would be
better served by the global T. Fix: initialize small-group `log_T[j]` from the fitted global
`_global.log_T` (or route small known groups through global in `forward`). Medium.

---

## LOW / MAINTAINABILITY

### L1. `training/train.py` — dead ternary in checkpoint dict
Line 650: `"best_macro_f1": best_macro_f1 if is_best else best_macro_f1` — both branches are
identical. Harmless but misleading (suggests the value should differ on non-best saves).
Simplify to `"best_macro_f1": best_macro_f1`. Cosmetic; cannot change results. Safe to
auto-apply.

### L2. `training/train.py` — mixup/cutmix use un-seeded global RNG
`_mixup`/`_cutmix`/`_maybe_mix` use `np.random.*` and `torch.rand(...)` (global RNGs) rather
than the run's seeded `torch.Generator`. Even with `deterministic=True`, augmentation
mixing is not bit-reproducible. Thread the seeded generator through, or document that mix
augmentation is intentionally excluded from determinism.

### L3. `data/datasets.py` — per-call `import time, random`
Line 181 imports inside `__getitem__`, executed on every sample. Move to module scope.
Trivial. Safe to auto-apply.

### L4. `data/qc.py` — `blur_score` docstring says "Laplacian variance" but uses `FIND_EDGES`
Lines 57-63 compute the variance of PIL `ImageFilter.FIND_EDGES`, not a Laplacian. The
threshold guidance ("<50 unusable") was presumably tuned for one and not the other. Either
switch to a real Laplacian (`cv2.Laplacian`) or fix the docstring/threshold. Low.

### L5. `evaluation/abstention.py` — docstring/behavior mismatch
`threshold_for_target_error` docstring (lines 64-66) says it "raises and returns the
closest-to-target candidate" when nothing is eligible, but the code never raises — it
silently returns a fallback. Fix the docstring (or add the promised warning). Low.

### L6. `evaluation/corruption.py` — hardcoded ImageNet mean/std
`_MEAN`/`_STD` (lines 25-26) are fixed ImageNet constants. If any config uses different
normalization, the denormalize→corrupt→renormalize round-trip is applied with the wrong
stats. Parameterize from the eval `AugConfig`. Low (matches current default).

### L7. `data/download/isic.py` — convoluted dead code
Lines 153-157 build a list literal then slice `[:0]` to an empty list; the subsequent
`fieldnames` branch already handles the empty case. Remove the dead construction. Trivial.

---

## Verified-correct (spot-checked, no action)
- `calibration/conformal.py`: APS scores, randomized set construction (monotonicity of the
  randomized cumulative adjustment holds), and the finite-sample `(1-α)(1+1/n)` quantile are
  correct; `predict_set` uses a fixed-seed RNG so results are reproducible.
- `evaluation/bootstrap.py`: percentile bootstrap, stratified resampling, exact/χ² McNemar,
  and fast DeLong (`_delong_internal`, cov via `np.cov`) are correct.
- `evaluation/metrics.py`: equal-width ECE binning has no double-counting; multiclass Brier
  correct; OvR AUROC/AP tolerant to single-class folds.
- `data/splits.py`: group-aware external holdout + greedy-balanced GroupKFold with an
  explicit cross-fold and external↔inner leakage assertion is sound.
- `training/schedulers.py`, `training/ema.py`, `utils/seed.py` (`seed_worker` precedence is
  correctly parenthesized), `utils/config.py`: correct.
