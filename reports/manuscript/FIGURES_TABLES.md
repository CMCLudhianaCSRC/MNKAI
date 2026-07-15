# MNKAI-III Manuscript — Figures & Tables Specification

This document specifies every figure and table for the manuscript, the exact data/script that produces each, and its caption. Existing assets are reused where possible; one **new** figure (the ensemble forest plot / per-tone AUROC-with-CI) must be generated.

**Important asset note:** the `eval/run40/*.png` files are tracked in git but are currently *deleted from the working tree* (git status shows `D eval/run40/...`). Before building the figure package, restore them with:

```
git checkout -- eval/run40/
```

All numbers below are copied from the honest, leak-free artifacts; do not substitute other values.

---

## FIGURE 1 (NEW — must be generated) — DDI per-tone AUROC forest plot with 95% CIs

**Purpose.** The paper's headline. Shows the frozen six-backbone ensemble vs the single-backbone baseline, overall and per Fitzpatrick stratum, with stratified bootstrap 95% CIs and significance annotation.

**Type.** Horizontal forest / dot-and-whisker plot. Y-axis: strata (Overall, FST I–III, FST III–IV, FST V–VI). Two series per stratum: single DINOv2-B baseline (open marker) and score-average ensemble (filled marker). X-axis: AUROC, range 0.60–0.95. Vertical reference line at 0.5 optional; annotate the paired-bootstrap Delta and p-value beside each stratum.

**Data source.** `eval/probe/ddi_ensemble_ci.json` (point_estimate, bootstrap_ci95, delta_ensemble_minus_baseline). Values to plot:

| Stratum | Baseline AUROC [95% CI] | Ensemble AUROC [95% CI] | Delta [95% CI], p |
|---|---|---|---|
| Overall | 0.748 [0.704, 0.791] | 0.826 [0.789, 0.862] | +0.078 [0.049, 0.107], p<0.001 |
| FST I–III | 0.732 [0.643, 0.811] | 0.782 [0.701, 0.856] | +0.050 [−0.003, 0.104], p=0.032 |
| FST III–IV | 0.739 [0.670, 0.806] | 0.830 [0.771, 0.882] | +0.091 [0.042, 0.141], p=0.00025 |
| FST V–VI | 0.789 [0.713, 0.859] | 0.872 [0.808, 0.929] | +0.083 [0.034, 0.133], p=0.0005 |

**Producing script.** Add a small plotting script (e.g. `scripts/_plot_ddi_forest.py`) that reads `ddi_ensemble_ci.json` and renders `eval/probe/ddi_forest.png`. Pure matplotlib, no compute, no GPU — reads the existing JSON only. **TODO-author:** create this script (≈40 lines).

**Caption.** "Figure 1. Malignant-vs-benign AUROC on the DDI dark-skin benchmark (656 biopsy-confirmed clinical photographs, image-level 5-fold stratified cross-validation), by Fitzpatrick stratum. A score-averaged ensemble of six frozen foundation backbones (filled) significantly exceeds a single frozen DINOv2-B backbone (open) overall and in the darker strata; all members are frozen, so the gain reflects architectural diversity, not adaptation. Whiskers are stratified bootstrap 95% CIs; paired-bootstrap Delta-AUROC and one-sided p-values annotated. On the darkest skin (FST V–VI), AUROC rises from 0.789 to 0.872 (Delta +0.083, p = 0.0005)."

---

## FIGURE 2 (NEW, optional companion) — Per-backbone contribution bar chart

**Purpose.** Show that no single backbone reaches the ensemble; motivates score-averaging.

**Type.** Grouped bar chart, one bar per backbone (DINOv2-B, DINOv2-B-reg, DINOv2-L, EVA-02-B, ConvNeXt-V2-B, SigLIP-B) plus concat-ensemble and score-avg ensemble, for Overall and FST V–VI.

**Data source.** `eval/probe/ddi_ensemble.json` (`per_backbone`, `ensemble_concat`, `ensemble_scoreavg`). Overall values: 0.748, 0.747, 0.791, 0.766, 0.755, 0.753 (backbones); concat 0.808; score-avg 0.826. FST V–VI: 0.789, 0.766, 0.818, 0.799, 0.791, 0.791; concat 0.834; score-avg 0.872.

**Producing script.** Same new plotting script or a sibling; reads `ddi_ensemble.json`. **TODO-author.**

**Caption.** "Figure 2. Per-backbone DDI AUROC (frozen, class-balanced logistic probe) and the two ensembling strategies. Score-averaging the six frozen backbones' malignancy scores outperforms the best single backbone (DINOv2-L) and feature concatenation, on both the overall set and the darkest (FST V–VI) stratum."

---

## FIGURE 3 — Calibration reliability, before vs after temperature scaling

**Purpose.** Trust stack, panel 1: ECE 0.200 → 0.016.

**Type.** Two-panel reliability diagram (predicted vs empirical accuracy, with the diagonal), before and after.

**Data source (reuse).** `eval/run40/reliability_before.png` and `eval/run40/reliability_after_tempscale.png` (restore from git). Supporting numbers: `eval/probe/trust_25class.json` (`temperature` 2.3, `ECE_before` 0.2002, `ECE_after` 0.0164) and `eval/run40/calibration.json`.

**Caption.** "Figure 3. Reliability diagrams for the 25-condition frozen-probe classifier on the duplicate-corrected external test before (left) and after (right) temperature scaling (T = 2.3). Expected calibration error falls from 0.200 to 0.016."

---

## FIGURE 4 — Coverage–risk (selective prediction) curve

**Purpose.** Trust stack, panel 2: retained accuracy vs referral rate.

**Type.** Line plot; x = referral (coverage) rate, y = retained accuracy and retained macro-F1.

**Data source (reuse + data).** `eval/run40/coverage_risk_curve.png` (restore from git) OR regenerate from `trust_25class.json` `abstention` array: (0.0, 0.637, 0.567), (0.10, 0.681, 0.601), (0.20, 0.723, 0.631), (0.30, 0.766, 0.665).

**Caption.** "Figure 4. Selective-prediction trade-off on the 25-condition external test. Deferring the least-confident cases raises retained accuracy from 0.637 (no referral) to 0.766 at a 30% referral rate. TODO-author: overlay area-under-risk-coverage (AURC) and a max-softmax baseline (gap-analysis R-item)."

---

## FIGURE 5 — ROC and PR curves (one-vs-rest), 25-condition

**Type.** Two panels: ROC one-vs-rest and PR one-vs-rest.

**Data source (reuse).** `eval/run40/roc_curves_one_vs_rest.png`, `eval/run40/pr_curves_one_vs_rest.png` (restore from git). NOTE — **TODO-verify**: confirm these run40 curves correspond to the frozen-probe pipeline and not the older fine-tuned model; if they are the fine-tuned model's, either relabel the caption accordingly or regenerate for the probe (gap-analysis notes run40 robustness/curves target the old model).

**Caption.** "Figure 5. One-vs-rest ROC (left) and precision–recall (right) curves for the 25-condition classifier on the external test. TODO-verify pipeline provenance."

---

## FIGURE 6 — Normalized confusion matrix, 25-condition

**Data source (reuse).** `eval/run40/confusion_matrix_normalized.png` (restore from git); per-class detail in `eval/run40/per_class_metrics.csv`.

**Caption.** "Figure 6. Row-normalized confusion matrix for the 25-condition frozen-probe classifier on the external test, illustrating the recovery of common-class recall after the majority-class-collapse fix (e.g., eczema 0.004 → 0.503). TODO-verify provenance vs probe pipeline."

---

## FIGURE 7 — Corruption / robustness sweep

**Data source (reuse).** `eval/run40/corruption_sweep.png` and `eval/run40/robustness_breakdown.csv` (restore from git).

**Caption.** "Figure 7. Robustness of the classifier under synthetic image corruptions. TODO-verify: gap analysis flags that the run40 corruption sweep targets the older fine-tuned model; re-run on the frozen probe or caption as the fine-tuned baseline's robustness."

---

## TABLE 1 — DDI headline results (in manuscript §4)

Reproduced from `eval/probe/ddi_ensemble_ci.json` and `ddi_ensemble.json`. See MANUSCRIPT.md Table 1. Columns: Model | Overall | FST V–VI | FST III–IV | FST I–III, with 95% CIs on baseline and score-avg ensemble rows. Include the concat-ensemble and DINOv2-L rows as intermediate points (no CI computed for those — mark "—").

## TABLE 2 — 25-condition macro-F1, frozen probe vs cRT (in manuscript §5)

From `eval/probe/CLEAN_25class.json`. Corrected external (n=3,813): 0.573 [0.553, 0.592] vs 0.527 [0.510, 0.543]. Full external (n=5,886): 0.619 [0.602, 0.635] vs 0.540 [0.523, 0.555]. Disjoint CIs on the corrected partition.

## TABLE 3 — Selective prediction (in manuscript §5)

From `trust_25class.json` `abstention`. See Figure 4 data.

## TABLE 4 (supplementary) — Resolution & head ablation, 25-condition

**Purpose.** Show 518 > 224 and MLP head effect; address winner's-curse framing.

**Data source.** `eval/probe/probe_25class_224_mlp.json` (OOF 0.5689, EXT+bias 0.5942), `probe_25class_518_mlp.json` (OOF 0.5948, EXT+bias 0.6261), plus the LR-head JSONs (`probe_25class_224_lr.json`, `probe_25class_518_lr.json`). **TODO-author:** consolidate into one table; note these are on the *uncorrected* external unless re-run, and state the config-selection procedure (select on OOF, report once on corrected external) to avoid selection bias.

## TABLE 5 (supplementary) — Per-Fitzpatrick 25-class balanced accuracy (honest/weak)

**Data source.** `CLEAN_25class.json` `fairness_clean_probe`: FST I–III bal-acc 0.264 (n=642); FST IV–VI 0.454 (n=135). Also `trust_25class.json` `by_tone` (acc 0.358 / 0.415; mean confidence 0.479 / 0.514). Caption must state these are small-n, unstable, and direction-flipping across models — reported for honesty, not as demonstrated fairness.

## TABLE 6 (supplementary) — Published-model DDI comparison

**Purpose.** Contextualize against the DDI ceiling. **TODO-author/verify:** tabulate published DDI AUROCs (DeepDerm, ModelDerm, HAM10000-trained; Daneshjou et al. 2022 ~0.56–0.67 off-the-shelf, ~0.73–0.90 adapted) beside our 0.826 / V–VI 0.872. Cite the source AUROCs; do not reproduce on our folds unless a public model's features are probed. PanDerm is excluded (mis-loaded checkpoint, non-commercial license).

---

## Asset checklist

| Asset | Status | Action |
|---|---|---|
| `eval/probe/ddi_ensemble.json` | present | reuse |
| `eval/probe/ddi_ensemble_ci.json` | present | reuse (Fig 1 source) |
| `eval/probe/CLEAN_25class.json` | present | reuse (Tables 2, 5) |
| `eval/probe/trust_25class.json` | present | reuse (Figs 3–4, Table 3) |
| `eval/probe/probe_25class_*.json` | present | reuse (Table 4) |
| `eval/probe/ddi_dinov2_518.json` | present | reuse |
| `eval/run40/*.png` | tracked but deleted in tree | `git checkout -- eval/run40/` |
| `eval/probe/ddi_forest.png` (Fig 1) | not created | author `scripts/_plot_ddi_forest.py` |
| Per-backbone bar (Fig 2) | not created | author plotting script |
