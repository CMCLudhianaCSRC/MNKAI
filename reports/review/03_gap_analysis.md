# MNKAI-III — Gap Analysis for a High-Impact Publication

*Agent 3 (independent gap review). Static review of docs + code + `eval/` results only. No compute run; the DDI ensemble job and feature-extraction scripts were not touched.*

**Scope of assessment:** the honest, leak-free claims in `PAPER_DRAFT.md` / `PROJECT_REPORT.md` / `README.md`, validated against the actual result JSONs in `eval/` and the method scripts in `scripts/`.

---

## 0. Bottom line

The project is **methodologically honest and unusually rigorous for a solo, laptop-only effort** — the collapse diagnosis, the trust layer, and the self-caught 33% leakage are genuine strengths that most published derm-AI papers lack. But as it stands the manuscript would **not clear peer review at npj Digital Medicine / Lancet Digital Health**, and would face a hard fight even at a strong ML4H/MICCAI venue. The gaps are not cosmetic: the two headline scientific claims are **not yet supported by the statistical tests that would make them defensible**, and one of them is **confounded by design**.

The single most important realization: the central thesis — *"frozen-probe beats end-to-end fine-tuning"* — is currently demonstrated by comparing runs that differ in **backbone treatment AND training data AND evaluation protocol simultaneously**, with **no significance test on any comparison**. Fixing that is almost entirely free (feature-space work on a laptop) and is the highest-leverage thing left to do.

**Verdict:** `needs-work` (see §7). Most of the highest-leverage fixes are laptop-feasible.

---

## 1. Baselines — the biggest scientific gap

### 1.1 No supervised-ImageNet frozen-probe baseline (confounds the central claim) — HIGH, feasible
The thesis is "**frozen** features beat **fine-tuning**." But the only frozen backbone probed for the main results is DINOv2 (SSL). There is no **supervised ImageNet-pretrained frozen probe** (e.g. `vit_base_patch16_224.augreg2_in21k_ft_in1k`, or a supervised ViT-B/16). Without it, the reader cannot separate two very different hypotheses:
- (A) *Freezing* beats fine-tuning (the paper's claim), vs.
- (B) *DINOv2's SSL representation* is simply better than what fine-tuning produces from any init.

A reviewer will ask for the 2×2: {DINOv2, supervised-IN} × {frozen-probe, fine-tune}. Only one cell is currently populated well. This is pure feature extraction + sklearn — fully laptop-feasible, no GPU training.

### 1.2 The frozen-vs-fine-tune comparison is not protocol-matched — HIGH, partly feasible
On DDI (`logs/DDI_BREAKTHROUGH.md`, `eval/cancer_cv/ddi_cv.json` vs `eval/probe/ddi_dinov2_518.json`):
- **Frozen probe:** DINOv2 @518, trained on **DDI-only**, 5-fold stratified CV → 0.748 / V–VI 0.789.
- **"Fine-tune" baseline:** trained on **CleanPatrick + DDI + PAD** (8,471 imgs, DDI ~8%), @224 → 0.579 / V–VI 0.53.

These differ in **training data, resolution, and class mix** all at once. The paper attributes the entire 0.58→0.75 gap to "freezing," but part of it is resolution (224→518, which the probe table itself shows is worth ~0.01–0.05) and part is the fine-tune being dragged by a non-DDI majority. A clean claim needs **fine-tune vs frozen on the identical DDI-only 5-fold split at the identical resolution**. A light 518px fine-tune on 656 images fits in 6 GB (small data, gradient checkpointing) — feasible; at minimum a matched 224px head-to-head is trivially feasible.

### 1.3 No published-model benchmark reproduced on the same DDI split — MEDIUM, cite-feasible
The paper claims frozen-probe reaches "the published ceiling (Daneshjou 2022: 0.76–0.78 V–VI)." Daneshjou 2022 also reports **DeepDerm / ModelDerm / HAM10000 baselines on this exact DDI set** (~0.56–0.67). None are tabulated next to your number on the *same folds*. At minimum, cite the specific published AUROCs in a comparison table; ideally probe one publicly-available derm model's features on your folds. (Note: PanDerm was mis-loaded — `logs/DDI_BREAKTHROUGH.md` flags 48 unexpected keys — so that comparison is currently invalid and should either be fixed or dropped, not reported.)

---

## 2. Statistical analysis — claims made without the supporting tests

### 2.1 No confidence intervals on the headline dark-skin AUROC — HIGH, trivially feasible
`eval/probe/ddi_dinov2_518.json` reports point estimates only: overall 0.748, **V–VI 0.789 with n=207 and only 48 malignant**. There is **no CI**. A bootstrap CI on an AUROC with 48 positives will be wide (likely ±0.08–0.12), and honesty demands showing it. `_bootstrap_ci.py` exists but is wired to the 25-class fine-tune preds, not the DDI probe scores. Extending it to the DDI held-out scores is minutes of work and is expected by any clinical venue.

### 2.2 No DeLong (or paired bootstrap) test on any AUROC comparison — HIGH, trivially feasible
`docs/AUDIT.md` even lists "no DeLong" as a known legacy gap and claims a fix, but **no DeLong test is implemented anywhere** in `scripts/` (grep confirms zero occurrences). Every comparative claim — frozen vs fine-tune (0.748 vs 0.579), 518 vs 224, DINOv2 vs supervised — is a naked point-estimate comparison. The 25-class macro-F1 win is at least defended via disjoint bootstrap CIs (`CLEAN_25class.json`: 0.573 [0.553–0.592] vs 0.527 [0.510–0.543]); the AUROC comparisons have nothing. Add DeLong or a paired bootstrap Δ-AUROC with a CI that excludes 0.

### 2.3 Subgroup claims are underpowered and uncorrected — HIGH, feasible (analysis + honest framing)
- DDI V–VI: 48 positives. "0.789 on the darkest skin" rests on ~48 malignant cases.
- 25-class external fairness (`CLEAN_25class.json`): **FST I–III balanced-acc 0.264 (n=642), FST IV–VI 0.454 (n=135)** — these are strikingly low versus the 0.573 headline and are barely mentioned. The paper foregrounds the DDI subgroup win but buries that the 25-class per-tone balanced accuracy is poor and tiny-n.
- No multiple-comparison adjustment across the tone subgroups.
A power/precision statement per subgroup, and honest reporting of the weak 25-class per-tone numbers, is needed. (Data-limited, but the *reporting* is free.)

### 2.4 Calibration reported without CIs or class-wise breakdown — MEDIUM, feasible
`trust_25class.json` gives ECE 0.200→0.016 as point values. No CI on ECE, no class-wise ECE / adaptive ECE, no reliability by subgroup. ECE is sample-size sensitive; a bootstrap CI and a class-wise ECE table are standard.

### 2.5 Selective prediction lacks AURC and a baseline — MEDIUM, feasible
The abstention curve (refer 30% → 0.766 retained acc) is a nice table but has no **area-under-risk-coverage (AURC)** summary and no comparison against a trivial baseline (e.g. max-softmax vs the conformal-set-size criterion). Report AURC and at least one alternative selection rule.

---

## 3. Reproducibility gaps

### 3.1 The headline 25-class model is non-deterministic — HIGH, trivial fix
`_probe_multiclass.py` builds the main MLP head as `MLPClassifier(hidden_layer_sizes=(512,), ...)` with **no `random_state`**. The 0.573 headline therefore varies run-to-run (weight init + early-stopping split are stochastic). Set the seed and report the mean ± SD over N seeds. As written, the number is not exactly reproducible even by the authors.

### 3.2 Hardcoded absolute paths, no environment lock — MEDIUM
`_probe_backbone.py` hardcodes `DDI = D:/AIFINAL/images/ddi`; feature paths are machine-specific. `requirements.txt` is deleted in the working tree (git status shows `D requirements.txt`). A journal needs a pinned environment (versions of timm/torch/sklearn — sklearn's MLP/LogReg numerics are version-sensitive), a data-availability statement, and a code-availability statement with exact commands. None are assembled.

### 3.3 Dedup threshold has no sensitivity analysis — MEDIUM, feasible
The leakage correction drops pairs at a single arbitrary cosine threshold (0.99). Since this directly moves the headline (0.626→0.573), show the metric as a function of threshold (0.95–0.995) so the reader sees the result is not a knife-edge artifact. This strengthens what is otherwise one of the paper's best contributions.

### 3.4 No model card / no reproducible config for the deployed probe — MEDIUM
`models/clinical_25class.pkl` bundles scaler+MLP+temp+bias+conformal, but there is no model card (intended use, training data provenance, subgroup performance, failure modes, operating point) — a near-mandatory artifact for a clinical-AI venue.

---

## 4. Missing experiments / ablations

- **Resolution ablation (224 vs 518) as a CI'd table** — data exist across the probe JSONs but are not consolidated with significance. LOW effort.
- **Head ablation isolating the per-class bias term** — `_probe_multiclass.py` computes raw vs +bias; surface it as an ablation (how much of 0.573 is the bias post-hoc vs the head). LOW effort.
- **DDI operating-point analysis** — only AUROC is reported for the probe. A clinical cancer detector needs **sensitivity at a fixed high specificity (or vice-versa), a decision-curve / net-benefit analysis, and calibration of the cancer score itself** (currently only the 25-class model is calibrated). MEDIUM effort, feasible.
- **Robustness/corruption sweep on the frozen-probe pipeline** — `corruption_sweep.py` / `_robustness_sweep.py` target the old fine-tuned model, not the deployed probe. Re-run on the probe. MEDIUM, feasible.
- **OOD gate evaluated end-to-end on the probe** — `_ood_gate.py` exists but its acceptance on the deployed pipeline (non-skin rejection rate, in-distribution retention) isn't reported as a metric. MEDIUM.
- **Multi-backbone ensemble** — running per constraints; note as pending, not a gap to action here.

---

## 5. Datasets / external validation — the clinically-decisive gaps

- **No external clinical test set for the dark-skin claim beyond DDI.** The entire dark-skin thesis rests on one dataset (656 images, ~171 malignant). DDI is acknowledged as essentially the only public dark-skin clinical cancer set, so this is data-limited — **not laptop-feasible** to close, but it caps the strength of the claim and must be stated as such. A second dark-skin source (PASSION, HIBA, or a Stanford AIMI set once registered) would materially de-risk the finding.
- **No prospective or Indian-population clinical validation.** The stated goal is Indian deployment, yet there is no Indian clinical test set with biopsy/clinician labels in the evaluation. DermaCon-IN is in the training mix but there is no held-out Indian clinical benchmark reported. Not laptop-feasible; a top venue will flag it.
- **No biopsy confirmation / label-quality analysis for the 25-class Indian labels.** DDI is biopsy-proven (good); the 25-class labels' provenance, class definitions, and any inter-rater agreement are undocumented. At least document label sources and known noise.

---

## 6. Citations & novelty positioning

- **The central claim's prior art is uncited.** "Fine-tuning distorts pretrained features; linear probing wins under distribution shift" is **Kumar et al. 2022 (ICLR, LP-FT)** — the exact phenomenon, with theory. `DDI_BREAKTHROUGH.md` gestures at "FEDD, MICCAI 2023" but the manuscript must cite and *differentiate* from Kumar 2022, or reviewers will read the core finding as a known result re-observed. **Frame the novelty as the clinical, data-scarce, dark-skin demonstration + the deployable trust stack — not the phenomenon itself.**
- **Other must-cite anchors currently absent from the draft's reference list:** Oquab 2023 (DINOv2), Daneshjou 2022 (DDI), Groh 2021 (Fitzpatrick17k), Cui 2019 (already used in code), Guo 2017 (temperature scaling), Angelopoulos 2021 (RAPS conformal), Tschandl/HAM10000, ISIC. A proper Related Work section positioning against derm-FM work (PanDerm, DermFoundation, SCIN) is missing.
- **Novelty risk:** none of the individual components (frozen probe, temperature scaling, RAPS, focal/cRT) is new. The defensible novelty is the **integration + the honest dark-skin case study + the leakage-audit method**. The paper should say this plainly rather than implying the frozen-probe advantage is a discovery.

---

## 7. Journal-readiness verdict

**Assessment: `needs-work`.**

For a **high-impact clinical venue (npj Digital Medicine / Lancet Digital Health):** not yet. Those venues effectively require (a) a **clinician/reader comparison or a genuine external clinical test set**, (b) full statistical treatment (CIs + significance tests on every headline comparison), and (c) a model card + reproducibility package. (a) is not laptop-feasible; (b) and (c) are. Without (a) the natural home is a strong ML4H / MICCAI-workshop / *Journal of Biomedical Informatics*-tier venue, where the methodological-rigor framing (collapse fix + trust layer + leakage audit + frozen-probe case study) is genuinely competitive **once the statistics are completed**.

**Highest-leverage, laptop-feasible additions (do these first):**
1. Bootstrap CIs on DDI AUROC + DeLong/paired-bootstrap on every AUROC comparison (§2.1, §2.2).
2. Supervised-ImageNet frozen-probe baseline to de-confound the central claim (§1.1).
3. Protocol-matched fine-tune-vs-frozen on identical DDI folds/resolution (§1.2).
4. Seed the MLP head + report variance; assemble the reproducibility package (§3.1–3.4).
5. Honest, powered subgroup reporting incl. the weak 25-class per-tone numbers (§2.3).
6. Dedup-threshold sensitivity curve (§3.3) and the Kumar-2022 citation/positioning (§6).

**Not laptop-feasible (state as limitations, or pursue with new data/collaborators):** dermatologist reader study; prospective Indian clinical validation; a second dark-skin cancer dataset.
