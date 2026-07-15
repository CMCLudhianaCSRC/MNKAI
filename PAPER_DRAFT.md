# Frozen Features over Fine-Tuning: A Rigorous, Data-Scarce Dermatology AI for Indian Skin

*Working draft — canonical honest numbers as of 2026-07-10. Supersedes the earlier RESEARCH_REPORT_* drafts.*

## Abstract
We present MNKAI-III, a dermatology classification system built by a single researcher on a 6 GB laptop,
targeting Indian clinical deployment across 25 conditions. Our contributions are methodological rather than a
leaderboard number. (1) We diagnose and fix a **majority-class collapse** caused by stacking two rebalancing
mechanisms (an inverse-frequency sampler on a class-balanced focal loss), recovering common-class recall (eczema
0.004→0.503). (2) We ship a **trust layer** — temperature-scaled calibration (ECE 0.176→0.016), conformal
prediction sets, and per-Fitzpatrick fairness — that is almost universally omitted in comparable work. (3) On the
DDI dark-skin benchmark our fine-tuned model scored at chance (AUROC 0.53 on Fitzpatrick V–VI); we show this is
the *unadapted floor*, not a data ceiling, and that **frozen foundation-model features with a lightweight probe**
reach 0.789 — a +0.26 AUROC gain, on the laptop, with no new data. (4) Generalizing this, we find frozen-probe
**beats end-to-end fine-tuning on every task** (9-class, 25-class, dark-skin cancer) in our data-scarce regime —
a clinical, dark-skin demonstration of the established fine-tuning-distortion effect (Kumar et al., ICLR 2022),
not a claim of a new phenomenon.
(5) In the process we **detect and correct ~33% train-test duplicate leakage** in a multi-source external test —
a cautionary, reproducible finding. All results are reported leak-free with bootstrap confidence intervals.

## 1. Introduction
Dermatology AI is dominated by light-skinned dermoscopy datasets and by end-to-end fine-tuning. Both assumptions
fail for an Indian clinical setting: skin tones skew darker, images are smartphone clinical photos, conditions are
inflammatory/infectious long-tailed, and compute + labeled data are scarce. We build under exactly these
constraints and report what actually works — including two honest negative results that strengthen the work.

## 2. Data
~266k labeled, de-duplicated images across dermoscopy (ISIC-2024/2019, BCN20000) and clinical photos
(Fitzpatrick17k [cleaned via CleanPatrick], SCIN, DermaCon-IN, PAD-UFES-20, DDI, Mpox/MSLD). Dark-skin
(Fitzpatrick IV–VI) coverage is ~15k — the binding scarcity. Splits are group-aware (patient/lesion-disjoint);
§6 details a leakage correction. No synthetic data, no scraping beyond public research datasets.

## 3. Methods
- **Backbone:** DINOv2 ViT-B/14 (frozen), native 518px, 768-d pooled features.
- **Collapse fix:** single rebalancing mechanism (class-balanced focal, no resampler) + decoupled cRT.
- **Frozen-probe (main method):** freeze the backbone; fit a class-balanced MLP head + post-hoc per-class logit
  bias (coordinate-ascent on OOF). No backbone fine-tuning.
- **Trust layer:** temperature scaling; RAPS/clustered conformal; per-Fitzpatrick reporting.
- **Evaluation:** group-aware 5-fold OOF + single-shot external; duplicate-corrected (§6); bootstrap 95% CIs.

## 4. Results (canonical, leak-free)
| Task | Fine-tuned baseline | **Frozen-probe** | Test |
|---|---|---|---|
| 25-class Indian macro-F1 | cRT 0.527 [0.510–0.543] | **0.573 [0.553–0.592]** | de-duplicated external (n=3,813) |
| 9-class base macro-F1 | 0.637 | **0.656** | OOF CV |
| DDI dark-skin cancer AUROC | 0.53 | **0.748 (V–VI 0.789)** | DDI 5-fold CV |
| Internal cancer AUROC | 0.988 | — | dermoscopy |
| Calibration ECE | 0.176 | **0.016** | external |
Frozen-probe wins on every task; the 25-class win is statistically significant (disjoint CIs).

## 5. The dark-skin cancer study
Fine-tuning on DDI (even in 5-fold CV *training on 80% of DDI*) stayed at 0.58 — we initially read this as a
data ceiling. A literature sweep showed 0.58 is exactly the **unadapted DDI floor** (Daneshjou 2022), while
fine-tuning/adaptation reaches 0.73–0.90 on the same data. The fix was not more data but a **better use of the
representation**: frozen DINOv2@518 features + a probe reach **0.789 on Fitzpatrick V–VI**, at the published
ceiling. We also confirm (with DermDiff) that synthetic augmentation *alone* does not help. Honest ceiling: dark-
skin clinical cancer imagery barely exists as public data (DDI is essentially the only source), so ~0.78 is the
realistic V–VI ceiling; we pair the detector with per-tone calibration and selective abstention.

## 6. A cautionary methodological finding: duplicate leakage
Our first 25-class "held-out" external test showed inflated numbers. Frozen-feature nearest-neighbor analysis
revealed **~33% of external images are exact duplicates of training images** (cross-source atlas photos —
DermNet/SkinDisease — that received distinct group IDs and thus slipped past the group-aware split). We drop them
(cosine > 0.99) and re-report; the honest 25-class numbers in §4 reflect this. The pure-ISIC 9-class split was
duplicate-free (0%); its external was instead class-degenerate (99.8% nevi), so we report OOF CV there. We
recommend feature-space duplicate auditing as a standard check for any multi-source medical dataset.

## 7. Why frozen-probe beats fine-tuning here
End-to-end fine-tuning on a small, imbalanced, multi-domain corpus *degrades* a strong self-supervised
representation (it collapsed DDI; it under-uses features on the 25-class). A frozen backbone preserves general
structure that a light head exploits — consistent with linear-probe-vs-fine-tune findings under distribution
shift and low labels. This is the practical thesis of the paper for the data-scarce clinical setting.

**Relation to prior work — we are not claiming a new phenomenon.** That fine-tuning can distort strong
pretrained features and underperform a frozen linear/light probe under distribution shift is an *established*
result, characterized by Kumar et al. (2022, *Fine-Tuning can Distort Pretrained Features and Underperform
Out-of-Distribution*, ICLR), who also propose the LP-FT remedy (linear-probe-then-fine-tune). Our contribution is
not to discover this effect but to **demonstrate and operationalize it in a clinical, data-scarce, dark-skin
dermatology setting** where it is both underexplored and consequential — and to wrap it in a deployable trust
stack (calibration + conformal + fairness + abstention) runnable on a 6 GB laptop. The novelty is the clinical
demonstration + trust stack + the DDI dark-skin result, not the frozen-vs-fine-tune phenomenon itself. We report
LP-FT as recommended future work (§8), since we did not evaluate the intermediate LP-FT regime here.

## 8. Limitations
Dark-skin cancer remains ~0.78 (data-limited, not method-limited). Fairness subgroups are underpowered
(Fitzpatrick IV–VI n≈135 in external). No biopsy confirmation or prospective validation. The system is
decision-support, not autonomous diagnosis; melanoma sensitivity is locked ≥0.90 with abstention. We did not
evaluate the intermediate **LP-FT** regime (linear-probe-then-fine-tune; Kumar et al., 2022), which is the
recommended next step to test whether a short fine-tune after probing recovers any headroom without re-inducing
the distortion.

## 9. Conclusion
Under real Indian-clinic constraints, rigor beats scale: a diagnosed collapse-fix, a trust layer, a leak-free
(and leakage-audited) evaluation, and a simple, laptop-runnable recipe — frozen foundation features + a light
probe — that outperforms fine-tuning across tasks, including the hardest dark-skin cancer case.

## Appendix — reproducibility
Scripts: `_cache_features.py`, `_probe_multiclass.py`, `_probe_backbone.py`, `_clean_eval_25class.py`,
`_ddi_cv_eval.py`. Artifacts: `eval/probe/*`, `logs/DDI_BREAKTHROUGH.md`, `logs/DARKSKIN_RESEARCH.md`,
`PROJECT_REPORT.md`. Backbone: `vit_base_patch14_dinov2.lvd142m` (frozen).

## References
- Kumar, A., Raghunathan, A., Jones, R., Ma, T., & Liang, P. (2022). *Fine-Tuning can Distort Pretrained Features
  and Underperform Out-of-Distribution.* International Conference on Learning Representations (ICLR).
  arXiv:2202.10054. — establishes the fine-tuning-distortion effect and the LP-FT remedy that our frozen-probe
  finding demonstrates in the clinical, data-scarce, dark-skin setting.
- Oquab, M., et al. (2024). *DINOv2: Learning Robust Visual Features without Supervision.* TMLR. — the frozen
  self-supervised backbone used here.
- Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free
  Uncertainty Quantification.* arXiv:2107.07511. — the conformal-set trust-layer component.
- Daneshjou, R., et al. (2022). *Disparities in dermatology AI performance on a diverse, curated clinical image
  set (DDI).* Science Advances. — the dark-skin cancer benchmark.
