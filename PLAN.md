# MNKAI-III — Master Plan (updated post-review, 2026-06)

Clinical-grade dermatology AI for Indian deployment. Goals: (1) high-tier journal
publication under college endorsement, (2) deployable clinical product.
Constraints: NO synthetic data, NO scraping. Hardware: RTX 4050 6GB / 16GB RAM.

## Status snapshot
- **Phase 1 (9-class lesion):** ✅ DONE — 5-fold CV macro-F1 **0.637 ± 0.009**, AUROC ~0.95.
- **Phase 3 v1 (Indian 25-class):** ⚠️ DONE but **majority-class collapse** (double-balancing bug).
  Cancers transferred well; common Indian classes (eczema/psoriasis/tinea) collapsed to ~0 recall.
- **Comprehensive review:** ✅ DONE (14-agent, adversarially verified). Verdict: **fix + retrain, do NOT restart.**
- **Corrected infrastructure:** ✅ BUILT (this is the active plan below).

## Strategic decision (from the cost-benefit review)
**Option 2 (targeted fix + retrain) → Option 3 (selective rebuild).** NOT a restart —
the collapse is a ~1-line weighting bug; backbone, data pipeline, and group-aware CV are sound.

---

## PHASE 3-CORRECTED (active) — scientific ablation + publication eval

Root cause: double-balancing (full-inverse-freq sampler × cb_focal β=0.9999, ~500–766× range).
Fix = use ONE balancing mechanism. We ABLATE three single-mechanism options and pick the
winner on the held-out external test.

**Infrastructure (built, CPU-verified):**
- `sampler.py` `power` knob; `train.py` `sampler_power` + `use_weighted_sampler`.
- `focal.py` `LogitAdjustedCE` (Balanced-Softmax) loss.
- External held-out test set (15%, stratified, group-aware, zero leakage) in `phase3_splits.csv`.
- viral_infection source-aware de-confound (36% clinical-source, was 8%).
- Ablation configs: `phase3_fixA_ce_sampler` / `phase3_fixB_balsoftmax` / `phase3_fixC_cbfocal_only`.
- `_phase3_eval_full.py`: OOF + external ensemble + calibration (ECE/Brier) + conformal (APS) + per-class AUROC + per-Fitzpatrick fairness.
- `_phase3_ablation_driver.py`: the autonomous engine.

**Steps:**
1. [RUN] Ablation bake-off: train fold-0 of FixA/B/C → eval each on external test (~6h).
2. [AUTO] Pick winner by external balanced-acc + recovered common-class recall.
3. [AUTO] Train winner's folds 1–4 (~8h).
4. [AUTO] Full 5-fold publication eval (OOF + external ensemble + calibration + conformal + fairness).
   **Target:** macro-F1 0.339 → ~0.45–0.55; common-class recall ~0 → 0.3–0.6.

Launch: `nohup .venv/Scripts/python.exe scripts/_phase3_ablation_driver.py > logs/ablation_driver.log 2>&1 &`

---

## PHASE 4 — novelty + paper (after Phase 3-corrected)
- Per-class × per-Fitzpatrick fairness audit (+ mitigation if gaps > 0.10; Mondrian conformal).
- Hierarchical-taxonomy-loss ablation (coarse super-class → fine 25) vs the flat winner — novelty.
- Source-stratified viral_infection reporting (disclose + quantify the de-confounded result).
- Counterfactual fairness analysis.
- Figures: reliability/calibration, coverage-risk, per-class, fairness, confusion.
- Manuscript: methods/results/limitations (incl. the diagnosed-and-fixed collapse as a contribution).

## PHASE 5 — clinical (CMC Ludhiana prospective)
- Prospective external validation on real clinic photos (the ultimate generalization test).
- Fitzpatrick IV–VI collection target (close the fairness coverage gap).
- Regulatory/CDSCO considerations; inference-latency targets.

## Carried-over improvements to fold in (from the review)
- img_size 224-vs-higher ablation (Phase 1 fold-0); multimodal (tabular) ablation.
- Backbone comparison (DINOv2-B vs L / derm-foundation) — efficiency-for-hardware framing.
- Redundant storage + backups (the D: external SSD is a single point of failure).
- Early-stop on balanced-acc / per-class floor (not macro-F1).
