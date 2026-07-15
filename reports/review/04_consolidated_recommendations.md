# MNKAI-III — Consolidated & Prioritized Recommendations (Agent 4: Cross-Validation & Synthesis)

This document merges the outputs of four upstream review agents — **code review** (library `01a`, scripts `01b`, app `01c`), **scientific review** (`02`), **gap analysis** (`03`), and the **innovation** pass (returned null) — plus the **adversarial verdicts** on the two high-severity code bugs. Overlapping findings are merged into single items with all sources noted; conflicts are resolved explicitly; every item is scored on scientific impact (1–5), implementation effort (1–5, 1=trivial), and publication value (1–5), then assigned an overall priority (1 = do first).

`auto_apply` = safe, additive, static, laptop-feasible edits that **cannot change any scientific result** (docstrings, dead-code removal, defensive guards, new documentation, reproducibility scaffolding). Everything that re-generates a metric, model artifact, or requires running a probe is **not** auto-applied — it is queued for a controlled re-run.

---

## Conflict resolutions

1. **Rigor vs. novelty (gap analysis vs. scientific review).** Gap analysis calls the central "frozen-probe beats fine-tuning" thesis *under-supported and confounded*; scientific review calls the *qualitative* conclusion sound but the *point estimates* oversold. **Resolution:** keep the thesis, but (a) reframe it as a *clinical, data-scarce, dark-skin demonstration* of an already-established phenomenon (Kumar et al., ICLR 2022, LP-FT), not a new phenomenon; (b) gate the causal "freezing wins" wording behind two missing controls — a supervised-ImageNet frozen baseline (R08) and a protocol-matched DDI head-to-head (R17). Novelty is repositioned onto the **deployable trust stack + dark-skin demonstration**, which no reviewer disputes.

2. **Severity of the two "high" code bugs (code review vs. adversarial verdict).** Both confirmed high-severity script bugs (no `random_state`; conformal RAPS tuned on the eval split) were **downgraded to medium** by the adversarial verifier — they yield *non-deterministic / optimistically-biased* numbers, not *wrong* outputs. **Resolution:** adopt **medium severity** but retain **high priority (P2, P5)**, because both touch headline numbers and the shipped `clinical_25class.pkl`, and both are cheap to fix.

3. **DDI group-CV concern — is it real? (scientific review).** Scientific review flags that the breakthrough 0.789 came from image-level `StratifiedKFold` (no patient grouping) and offers a cheap escape hatch: *if DDI is genuinely one-image-per-lesion/patient, the concern retires.* **Resolution:** R01 does the 10-minute verification first; only if patients contribute multiple images does the full `StratifiedGroupKFold` re-run become load-bearing. Either way the paper must state which is true.

4. **DDI confidence-interval item raised twice (scientific review #2 + gap analysis).** Merged into **R03**. Note the existing `scripts/_bootstrap_ci.py` is wired to the 25-class fine-tune predictions, not the DDI probe scores — the fix is to re-point it, not to write it from scratch.

---

## Ranked recommendations

| P | ID | Recommendation (merged) | Sources | Impact | Effort | PubVal | Auto | Phase |
|---|----|--------------------------|---------|:------:|:------:|:------:|:----:|:-----:|
| 1 | R01 | **Re-run the headline DDI AUROC under genuine patient/lesion-grouped folds.** Verify DDI image→patient multiplicity first; if >1 image/patient, switch `scripts/_probe_backbone.py` from `StratifiedKFold` to `StratifiedGroupKFold` and re-report 0.748/0.789. Reconcile with the "group-disjoint" claim in docs & `_build_cancer_data.py`. | science, gaps | 5 | 2 | 5 | no | 1 |
| 2 | R02 | **Make the whole probe/eval pipeline deterministic.** Add `random_state=0` to every `MLPClassifier`/`LogisticRegression` (`_probe_multiclass.py`, `_trust_refresh_25class.py`, `_save_deployable.py`, `_clean_eval_25class.py`, `_concat_probe_25.py`, `_probe_meta_25class.py`, `_probe_backbone.py`, `_ddi_ensemble.py`) **and** set the global seed *before* the OOF loop so `T`, bias, and `qhat` inherit determinism. Re-run `_save_deployable.py` once; re-record 0.573 / ECE 0.016 / conformal set-size. | code, gaps, science | 5 | 1 | 5 | no | 1 |
| 3 | R03 | **Stratified bootstrap 95% CIs on every DDI AUROC (overall + per-tone), across multiple CV seeds.** Re-point `_bootstrap_ci.py` at the DDI probe scores. Reframe "reaches the ceiling" → "point estimate 0.789, 95% CI [x,y], consistent with the published 0.76–0.78 ceiling." | science, gaps | 5 | 1 | 5 | no | 1 |
| 4 | R04 | **Add DeLong / paired-bootstrap ΔAUROC significance tests to every comparative AUROC claim** (frozen 0.748 vs fine-tune 0.579; 518 vs 224; DINOv2 vs baselines). No such test currently exists in `scripts/` despite `docs/AUDIT.md` claiming a fix. | gaps, science | 5 | 2 | 5 | no | 1 |
| 5 | R05 | **Eliminate model-selection-on-the-evaluation-set (winner's curse).** (a) Select the best config (224 vs 518, LR vs MLP, DDI backbone) on OOF, then score the single chosen config *once* on the untouched external. (b) Select conformal `(k_reg, λ)` on a held-out calibration slice, freeze, then evaluate once — fixes the optimistically-low "set ~4 @ 90%". | science, code | 4 | 2 | 5 | no | 1 |
| 6 | R06 | **Extend the frozen-feature duplicate audit to OOF folds and the 9-class folds.** The leakage tool ran only train-vs-external; the same cross-source atlas duplication can contaminate OOF folds (where bias is tuned) and the 197k 9-class folds. Re-report 0.5948, the tuned bias, and the 9-class 0.656. | science | 4 | 2 | 5 | no | 1 |
| 7 | R07 | **Honest, powered per-subgroup reporting.** Surface the weak 25-class per-tone numbers (FST I–III bal-acc 0.264 / n=642; IV–VI 0.454 / n=135) alongside the DDI win; add per-subgroup n / positive counts, precision-of-estimate statements, and a multiple-comparisons note across tone bins. | gaps, science | 4 | 1 | 5 | no | 1 |
| 8 | R08 | **Add a supervised-ImageNet frozen-probe baseline to de-confound the central claim.** Populate the 2×2 {DINOv2-SSL, supervised-IN21k} × {frozen-probe, fine-tune}. Separates "freezing wins" (the claim) from "DINOv2's representation is simply better." Needs one feature-extraction pass → deferred until the running ensemble frees GPU. | gaps | 5 | 3 | 5 | no | 2 |
| 9 | R09 | **Cite Kumar et al. 2022 (LP-FT, ICLR) and reframe novelty.** Position the finding as the clinical dark-skin/data-scarce demonstration + deployable trust stack, not a new phenomenon. Pure edit to `PAPER_DRAFT.md` / `PROJECT_REPORT.md`. | gaps, science | 4 | 1 | 5 | **yes** | 1 |
| 10 | R10 | **Fix the zero-image silent fallback in `SkinDataset.__getitem__`** (`datasets.py:216`). Fail loud after N retries (or abort on a global miss-rate threshold); never substitute an all-zero image carrying the true label; stop same-label index substitution; size any placeholder from the transform (currently hardcoded 3×224×224 → breaks collate at 518); use a seeded RNG. | code | 4 | 3 | 3 | no | 1 |
| 11 | R11 | **Guard the manifest↔splits inner merge** (`train.py:288`). Normalize `filepath` on both sides; assert merged rows ≥ a high fraction of pre-merge; log dropped rows so a path-format mismatch fails loudly instead of silently shrinking the training set. | code | 3 | 1 | 3 | **yes** | 1 |
| 12 | R12 | **Dedup-threshold sensitivity analysis.** Plot the corrected external macro-F1 as a function of the cosine cutoff (0.95–0.995) to show 0.573 is not a knife-edge artifact of the single 0.99 choice. | gaps | 3 | 2 | 4 | no | 1 |
| 13 | R13 | **Reproducibility package.** Restore/pin `requirements.txt` (deleted in working tree), remove hardcoded `D:/AIFINAL` paths in `_probe_backbone.py`, add exact repro commands + data/code-availability statements, and author a **model card** for `models/clinical_25class.pkl` (intended use, provenance, subgroup performance, operating point, failure modes). | gaps | 3 | 3 | 4 | partial | 1 |
| 14 | R14 | **DDI operating-point + decision-curve + score-calibration analysis.** Report sensitivity @ fixed high specificity, net-benefit / decision curve, and temperature-scale the DDI cancer score itself (only the 25-class model is currently calibrated). AUROC alone is insufficient for a malignancy screen. | gaps | 3 | 3 | 4 | no | 2 |
| 15 | R15 | **Resolve the `exanthema` alias collision** (`manifest.py:131/134`) — currently claimed by both `viral_infection` and `drug_eruption`; later key silently wins, mislabeling images. Assign one canonical class **and** add a collision check in `__post_init__` that warns on any alias claimed twice. | code | 3 | 1 | 3 | no | 1 |
| 16 | R16 | **Fix `GroupTemperatureScaler` small-subgroup fallback** (`calibration.py:145`). Groups with <30 samples are left at T=1 (uncalibrated) instead of falling back to the fitted global T — undermines the equitable-calibration goal. Init small-group `log_T` from the global scaler. | code | 3 | 1 | 3 | no | 1 |
| 17 | R17 | **Protocol-matched fine-tune vs frozen-probe on identical DDI folds & resolution.** The flagship 0.58→0.75 gap currently confounds freezing with training data (DDI-only vs CleanPatrick+DDI+PAD), class mix, and resolution. Needs end-to-end fine-tuning → GPU. Promote the partially-controlled All-3 CV fine-tune (0.579) to primary evidence meanwhile. | science, gaps | 4 | 4 | 5 | no | 3 |
| 18 | R18 | **Backbone `TypeError` fallback should not silently drop `img_size`** (`backbones.py:132`). Retry by removing kwargs one at a time (`drop_path_rate` first) and re-raise if `img_size` is genuinely rejected; optionally validate `default_cfg` resolution. Prevents a silent 518→default downgrade for DINOv2. | code | 3 | 1 | 2 | **yes** | 1 |
| 19 | R19 | **Focal-loss modulator under `torch.no_grad`** (`focal.py:117`) detaches the focal gradient — a genuine deviation from Lin et al. 2017. Either differentiate the modulator or document explicitly as an intentional deviation. Only matters on re-train → low urgency. | code | 2 | 1 | 2 | no | 2 |
| 20 | R20 | **Seed mixup/cutmix RNG** (`train.py:69`) — uses global `np.random`/`torch.rand` so mix augmentation is non-reproducible even with `deterministic=True`. Thread the seeded generator or document the exclusion. | code | 2 | 2 | 2 | no | 2 |
| 21 | R21 | **Parameterize corruption-sweep mean/std** (`corruption.py:25`) from the eval `AugConfig` instead of hardcoded ImageNet constants, so the denorm/renorm round-trip is correct if normalization ever differs. | code | 2 | 1 | 2 | no | 2 |
| 22 | R22 | **Static code-hygiene bundle (safe, additive).** Dead ternary `train.py:650`; per-call `import time, random` in `datasets.py:181`; `[:0]` dead-list build in `download/isic.py:153`; `blur_score` docstring says Laplacian but uses `FIND_EDGES` (`qc.py:57`); `threshold_for_target_error` docstring claims it raises but never does (`abstention.py:64`). | code | 1 | 1 | 2 | **yes** | 1 |

---

## Auto-apply set (safe for the implementation stage to apply now)

These change **no** scientific result — they are docstrings, dead-code removal, defensive guards, new documentation, and reproducibility scaffolding:

- **R09** — add Kumar 2022 citation + novelty reframing (doc edit).
- **R11** — manifest↔splits merge assertion + drop-count logging (guard only; no result change under valid data).
- **R13 (partial)** — restore `requirements.txt` from git, author the model-card skeleton, remove hardcoded `D:/AIFINAL` paths (the *analysis* portions of R13 are not auto).
- **R18** — backbone kwarg-by-kwarg retry + re-raise (fail-loud guard on the exception path only).
- **R22** — the five static hygiene fixes.

**Explicitly NOT auto-applied** (result- or artifact-changing, or require a run): R01, R02, R03, R04, R05, R06, R07, R08, R10, R12, R14, R15, R16, R17, R19, R20, R21. R02 in particular is a trivial edit but *regenerates the shipped model and shifts every headline number*, so it must go through a controlled, recorded re-run — not a silent auto-apply.

---

## Note on the running ensemble constraint

All Phase-1 *analysis* items (R01, R03–R06, R12) run on **already-cached frozen features via CPU sklearn** and touch none of the forbidden paths (`_ddi_ensemble.py`, `_cache_features.py`, `models/*.pkl`, `D:/`). They are authored now as ready-to-run patches/scripts and executed once the ensemble job completes. R08/R17 require fresh feature extraction / fine-tuning and are correctly pushed to later phases.
