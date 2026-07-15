# Code Review — Analysis Scripts (`scripts/`)

**Reviewer:** Agent 1B (static review, no execution)
**Scope:** frozen-probe method + evaluation pipeline: `_cache_features.py`, `_probe_multiclass.py`, `_probe_backbone.py`, `_ddi_ensemble.py`, `_trust_refresh_25class.py`, `_save_deployable.py`, `_optimize_thresholds.py`, `_conformal_fix.py`, `_coarse_eval.py`, `_bootstrap_ci.py`, `_ood_gate.py`, `_build_cancer_data.py`, plus `_clean_eval_25class.py`, `_concat_probe_25.py`, `_probe_meta_25class.py` and a scan of the ~77 remaining scripts.

## Executive summary

The core methodology is sound and the leakage-awareness is genuinely good: OOF folds are used to fit the per-class bias / temperature / conformal quantiles and those are *frozen* before touching the external test; per-fold `StandardScaler` is fit only on the fold-train split (no scaler leakage); the near-duplicate dedup (cosine > 0.99 to nearest train image) is a real and correctly-implemented control; `StratifiedKFold` and bootstrap RNGs are seeded. The coordinate-ascent bias is honestly tuned on OOF and applied to external — not fit on test.

Two systemic problems undercut the "rigor-first, reproducible" claim:

1. **No `random_state` on any `MLPClassifier`/`LogisticRegression`** across the entire probe pipeline. Every headline number (25-class 0.573, ECE 0.016, conformal set size, the *shipped* `models/clinical_25class.pkl`) comes from a stochastic MLP with `early_stopping=True` (random internal validation split). Numbers are not reproducible run-to-run, and the deployed artifact differs each time it is regenerated.
2. **Conformal RAPS hyperparameters (`k_reg`, `lambda`) are selected on the evaluation set.** `_conformal_fix.py` and `_trust_refresh_25class.py` grid-search the config that minimizes mean set size *measured on the test/clean-external split*, then report that minimum. The quantile is calibrated honestly, but the model-selection step is test-tuned, so the reported set size is optimistically biased.

Secondary: the `ok` valid-image mask produced by `_cache_features.py` is never consumed downstream (failed image loads become zero-vector features that silently pollute train and test); heavy verbatim duplication of the dedup + OOF-probe blocks across 6 scripts; hard-coded `D:/AIFINAL/...` paths and hard-coded relative input paths in the "final" scripts.

Nothing here suggests fabricated results — the honest-reporting discipline (leaky FULL vs CLEAN side by side, OOF-frozen bias) is present throughout. The issues are reproducibility and one selection-on-test bias in the conformal numbers.

---

## Findings

### F1 (HIGH) — No `random_state`: entire probe pipeline is non-reproducible, incl. the shipped model
`MLPClassifier(...)` and `LogisticRegression(...)` are instantiated **without `random_state`** in every probe/eval script:
- `_probe_multiclass.py:26-27`
- `_trust_refresh_25class.py:34-35`
- `_save_deployable.py:27,39`  ← produces `models/clinical_25class.pkl`
- `_clean_eval_25class.py:59,62`
- `_concat_probe_25.py:17,19`
- `_probe_meta_25class.py:52,54`
- `_probe_backbone.py:80`, `_ddi_ensemble.py:61` (LR — lbfgs is effectively deterministic, low risk, but still unset for consistency)

`MLPClassifier` with `early_stopping=True` draws a random 10% internal validation split and random weight init; with no seed the OOF probabilities, the temperature `T`, the coordinate-ascent bias, the conformal `qhat`, and the final external macro-F1 all shift between runs. For a project whose thesis is a specific number (0.573) and a deployable `.pkl`, this is the single most important gap. **Fix:** add `random_state=0` to every `MLPClassifier`/`LogisticRegression`. (Not auto-applied because it pins to one particular seed and will change the currently-recorded numbers by a small amount — should be a deliberate re-run.)

### F2 (HIGH) — Conformal RAPS hyperparameters selected on the evaluation set (optimistic set size)
`_conformal_fix.py:122-130` and `_trust_refresh_25class.py:48-53` loop over `(k_reg, lambda)` and keep the config that **minimizes `sz.mean()` computed on the test/clean-external split** (subject to a coverage floor). The quantile `qhat` is fit on the calibration/OOF split correctly, but choosing which regularization minimizes set size *on the same data you then report* is selection-on-test. The reported "mean set ~4 @ 90%" is biased low. **Fix:** select `(k_reg, lambda)` on a held-out calibration split (or a nested split of the calibration data), then evaluate the single chosen config on test once. `_save_deployable.py:34` hardcodes `K_REG, LAM = 2, 0.05`, which presumably came from this test-tuned search, so the deployed qhat inherits the bias.

### F3 (MEDIUM) — `ok` valid-image mask is written but never used; failed loads pollute features
`_cache_features.py:62-74` catches image-load failures, substitutes `torch.zeros(3,H,W)`, marks `ok[idx]=False`, and saves `ok` into the `.npz`. **No downstream script reads `ok`** (confirmed: `_probe_multiclass`, `_clean_eval_25class`, `_trust_refresh_25class`, `_save_deployable`, `_probe_meta_25class`, `_concat_probe_25` never reference it). Any unreadable image therefore enters training and evaluation as a zero-image feature vector (a valid-looking DINOv2 embedding of a black frame) with its real label — silently corrupting both the probe fit and the reported metrics in proportion to the number of failures. **Fix:** filter rows by `ok` when loading features in every downstream probe (`m = z["ok"]; X, y, split, fold, fst = X[m], ...`), or drop failed rows in `_cache_features.py` before saving.

### F4 (MEDIUM) — Massive duplication of the dedup + OOF-probe blocks (drift risk)
The near-duplicate mask block
```
tvn = Xtv/(norm...); exn = Xex/(norm...); for i in range(0,len(exn),256): mx[...] = (exn@tvn.T).max(1); clean = mx <= 0.99
```
is copy-pasted verbatim in `_trust_refresh_25class.py:26-29`, `_clean_eval_25class.py:48-52`, `_concat_probe_25.py:11-14`, `_probe_meta_25class.py:46-49`. The full "OOF 5-fold MLP + `coordinate_ascent` bias + external predict" block is duplicated across `_probe_multiclass`, `_trust_refresh_25class`, `_save_deployable`, `_clean_eval_25class`, `_concat_probe_25`, `_probe_meta_25class`. The dup threshold `0.99` and the MLP hyperparameters `(512,), max_iter=300, alpha=1e-3` are hardcoded independently in each — a change in one place (e.g. adding `random_state`, per F1) will not propagate, so the shipped model and the reported eval can silently diverge. **Fix:** extract `dedup_mask(Xtv, Xex, thr)` and `fit_oof_probe(Xtv, ytv, foldtv, C)` / `make_head()` into a shared module (e.g. `src/mnkai/evaluation/probe.py` or `scripts/_probe_common.py`) and import.

### F5 (MEDIUM) — Hard-coded absolute and relative input paths in "final" scripts
`D:/AIFINAL/images/ddi` is hardcoded in `_probe_backbone.py:26`, `_ddi_ensemble.py:23`, `_build_cancer_data.py:22`; `_build_cancer_data.py:18-23` also hardcodes `D:/AIFINAL/downloads/...`. `_trust_refresh_25class.py:21`, `_save_deployable.py:18`, `_probe_meta_25class.py:31`, `_concat_probe_25.py:8` hardcode relative input/output paths (`eval/probe/feats_25class_518.npz`, `models/clinical_25class.pkl`) and depend on the current working directory. These make the scripts non-portable and CWD-fragile. **Fix:** promote to `argparse` args (matching the pattern already used by the well-parameterized scripts) or a single config constant.

### F6 (LOW) — OOF-after-bias reported as a performance figure (mildly optimistic)
`_optimize_thresholds.py:76,92` reports `OOF macro_f1_after` (macro-F1 on OOF *after* applying the bias that was fit to maximize exactly that quantity). This is an in-sample-optimized number and will overstate generalization; only the `EXTERNAL` figure is honest. It is labeled, so this is a presentation caveat, not a bug — but the OOF-after number should not be quoted as achieved performance. (`_probe_multiclass.py:72` correctly reports OOF macro-F1 *without* bias — good.)

### F7 (LOW) — Brittle coarse/binary indexing assumes all class names present
`_coarse_eval.py:49` (`grp[j, cls_to_coarse[c]]`), `:52`, and `_bootstrap_ci.py:43` (`classes.index("mel")`) raise `KeyError`/`ValueError` if a class in the feature set is absent from the `COARSE` map or if `"mel"` is missing (e.g. reused on the 9-class task). `_probe_multiclass.py:41` handles this with `.get(..., 0)` but that silently maps unknown classes to group 0, skewing coarse accuracy. **Fix:** validate the class set against `COARSE` up front and fail loudly, rather than KeyError deep in a matrix build or silently defaulting to group 0.

### F8 (LOW) — `_conformal_fix.py` clustered method: top-1-cluster qhat can break marginal coverage
`_conformal_fix.py:162-171` builds the clustered set size using only the **top-1 class's cluster qhat** (`qh_i = cluster_qhat[ranks[order[i][0]]]`), by the author's own inline admission ("simpler: use the qhat of the top-1 class's cluster"). Applying a single cluster's threshold to a cumulative-probability set that spans classes from other clusters has no coverage guarantee and makes the "clustered" row not comparable to APS/RAPS. Low severity because clustered is a comparison baseline, not the shipped method — but the row should be labeled approximate or fixed to accumulate per-class cluster quantiles.

### F9 (INFO) — Minor cosmetics / non-issues confirmed OK
- `_cache_features.py:67` `ok[i+len(imgs)-1]=False` indexing is **correct** (global index of the just-appended failure).
- `_ood_gate.py:61` tied covariance `np.cov(Xc.T)` orientation is **correct** (d variables × n obs) with ridge; seeds set; reproducible.
- `_build_cancer_data.py` fold assignment is group-disjoint (patient `group_id`) with DDI stratified round-robin — no patient leakage across folds; CleanPatrick near-dup cliques collapsed to first member. Sound.
- `_bootstrap_ci.py` and `_clean_eval_25class.py:28` percentile bootstraps are standard and seeded.
- `TONE = {12:"FST I-III", 34:"FST III-IV", 56:"FST V-VI"}` (`_probe_backbone.py:27`, `_ddi_ensemble.py:25`) has cosmetic label overlap ("III" in two bands); does not affect computation (keyed by integer code).
