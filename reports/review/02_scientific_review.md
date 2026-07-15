# MNKAI-III — Scientific Review (Agent 2)

*Static review of documentation, method scripts, and existing `eval/` artifacts. No compute was run; the DDI ensemble job and feature caches were not touched.*

## Scope

I evaluated the science behind five headline claims:

1. Central thesis — *frozen foundation-model features + light probe beat end-to-end fine-tuning* in this data-scarce, multi-domain, dark-skin regime.
2. The honest metrics — 25-class macro-F1 **0.573** (deduped external), 9-class **0.656** (OOF CV), DDI dark-skin cancer AUROC **0.748 / 0.789 (V–VI)**, ECE 0.176→0.016, RAPS conformal, selective prediction.
3. Statistical rigor — CIs, 5-fold CV, OOF construction, the DDI AUROCs.
4. Fairness across Fitzpatrick tones.
5. Whether the conclusions are justified.

Primary sources read: `PAPER_DRAFT.md`, `STUDY_GUIDE.md`, `PROJECT_REPORT.md`, `logs/FINAL_RESULTS.md`, `logs/DDI_BREAKTHROUGH.md`, `logs/DARKSKIN_RESEARCH.md`; scripts `_probe_multiclass.py`, `_probe_backbone.py`, `_optimize_thresholds.py`, `_clean_eval_25class.py`, `_build_cancer_data.py`; results `eval/probe/*.json`.

---

## Overall verdict

The **central thesis is scientifically sound and well-aligned with the literature.** "Fine-tuning distorts pretrained features and underperforms out-of-distribution / under low labels" is exactly the finding of Kumar et al. (ICLR 2022, *Fine-Tuning can Distort Pretrained Features*) and the LP-FT line of work; FEDD (MICCAI 2023) shows linear probing wins on DDI in the low-label regime specifically. The team's diagnosis that DDI 0.58 is the *unadapted floor* (matching Daneshjou et al., Sci. Adv. 2022) rather than a data ceiling is correct and well-argued.

The **honesty of the reporting is genuinely above field norm**: they caught and corrected ~33% train↔test duplicate leakage, revised numbers down (0.626→0.573, 0.539→0.527), flagged the degenerate 9-class external, ship a trust layer, and repeatedly state subgroups are underpowered. This is a rigor-first project and it mostly earns that label.

However, several **specific empirical claims are not as cleanly supported as stated**, and the strongest single headline (DDI V–VI 0.789) rests on the weakest statistics. The conclusions are *directionally* justified but individual point estimates are oversold. Details below, ordered by severity.

---

## Findings

### C1 (HIGH) — The headline DDI number was NOT produced by group-disjoint CV, contradicting the documented method

`logs/DDI_BREAKTHROUGH.md` cites `scripts/_probe_backbone.py --img-size 518` as the source of the 0.748 / **0.789** result, and `eval/probe/ddi_dinov2_518.json` matches. But that script uses:

```python
skf = StratifiedKFold(5, shuffle=True, random_state=0)   # by image, no groups
for tr, te in skf.split(X, strat): ...
```

This is **image-level** stratified CV with **no patient/lesion grouping**. Meanwhile `_build_cancer_data.py` / the documentation explicitly claim "5-fold **group-disjoint** CV." DDI contains ~656 images from ~570 patients, so some patients contribute multiple images; plain `StratifiedKFold` can place two images of the same patient/lesion across train and held-out folds → optimistic leakage on the exact number used as the breakthrough headline. The two evaluation paths disagree and the *less rigorous* one produced the reported number. This must be reconciled: re-run the 0.789 result under genuine group-disjoint folds before publishing it.

### C2 (HIGH) — No confidence intervals on any DDI AUROC; the V–VI subgroup is too small to claim "reaches the ceiling"

`ddi_dinov2_518.json` reports V–VI AUROC 0.789 as a bare point estimate. The V–VI stratum is roughly a third of 656 images (~200), with a malignant prevalence low enough that only a few dozen positives drive the subgroup AUROC. A subgroup AUROC on ~50 positives has a 95% CI on the order of ±0.06–0.09 — i.e. 0.789 is statistically compatible with anything from ~0.71 to ~0.87. The claim "at the published ceiling (0.76–0.78)" is a defensible *point estimate* but is presented with more certainty than the data support. Every other headline carries a bootstrap CI (25-class does); the DDI numbers, which are the emotional core of the paper, carry none. Add stratified bootstrap CIs and, ideally, repeat the CV over multiple seeds (currently a single `random_state=0`).

### C3 (MEDIUM) — "Frozen-probe beats fine-tuning" on DDI is confounded by training-data access

The comparison table pits "old fine-tuned model (0.58, never trained on DDI)" against "frozen probe (5-fold CV, **trains on 80% of DDI**)". These differ on *two* axes simultaneously — frozen-vs-finetuned **and** out-of-domain-vs-in-domain training data — so the DDI improvement cannot be attributed to the frozen representation alone. The partially-controlled comparison ("All-3 CV fine-tune" 0.579, which does include DDI in CV) does support the thesis and should be promoted to the primary evidence; but the cleanest test — end-to-end fine-tune DINOv2 on the *same* DDI-only folds vs. frozen probe on those folds — is not reported. Without it, the DDI leg of the central claim is suggestive, not demonstrated. (The 25-class leg, C-note below, is clean.)

### C4 (MEDIUM) — OOF and 9-class numbers were never audited for the same cross-source duplication that contaminated the external test

The leakage audit (`_clean_eval_25class.py`) only removes **train↔external** near-duplicates (cosine > 0.99). The root cause was cross-source atlas images (DermNet/SkinDisease) receiving *distinct group IDs*, defeating the group-aware split. That identical mechanism can place near-duplicates across **OOF folds within `train_val`**, inflating the OOF macro-F1 (0.5948) — which is (a) a reported headline for the 9-class task (**0.656 OOF CV**, the only 9-class number since its external is degenerate) and (b) the surface on which the per-class bias is tuned. The OOF folds and the 9-class 197k-image folds have not been de-duplicated by the same feature-NN audit. This is an unquantified optimistic-bias risk on two reported numbers; it should be checked with the tool they already built.

### C5 (MEDIUM) — Winner's-curse / multiple-comparisons across configurations with no model-selection split

Reported numbers are the **maximum** over configuration grids evaluated on the same data: image {224, 518} × head {LR, MLP} × several backbones for DDI; LR vs MLP for the multiclass. The 518+MLP and DDI@518 were selected because they scored highest (DDI 0.789@518 vs 0.740@224). With no separate selection/validation split and no CI, the reported max is upward-biased. State the selection procedure, or select configs on OOF and report the chosen config once on the untouched external — otherwise the external is partly a tuning set.

### C6 (LOW–MEDIUM) — Calibration, conformal, and abstention appear to be fit and evaluated on the same clean external set

`trust_25class.json`: temperature 2.3, ECE 0.176→0.016, RAPS coverage 0.912 @ target 0.90, abstention retained-acc 0.766 @ 30% referral — all on `n_clean_external = 3813`. Split-conformal's coverage guarantee and an honest ECE both require a **held-out calibration set disjoint from the evaluation set**. If qhat/temperature/abstention-threshold were fit on the same 3,813 images they are scored on, coverage is satisfied in-sample (near-tautological) and ECE-after is optimistically low. Single-parameter temperature scaling overfits little, so ECE is low-severity; the conformal coverage claim is the one to verify has a proper calibration/test split.

### C7 (LOW–MEDIUM) — Fairness story leans almost entirely on the DDI binary task; the 25-class subgroup evidence is thin and internally odd

The paper's fairness headline is DDI V–VI 0.789. But the *25-class* per-tone numbers (the deployed multiclass system) are weak and unstable: `CLEAN_25class.json` gives FST I–III bal-acc **0.264** (n=642) vs FST IV–VI **0.454** (n=135) — dark skin scoring *higher* than light, both low, on 135 images. `FINAL_RESULTS.md` reports the opposite direction for an earlier model (0.245 vs 0.209). Direction-flipping across models on n=135 confirms these estimates are noise. The team says so ("underpowered," "thin coverage") — good — but the abstract's clean "fairness" framing should not let a 656-image binary-cancer AUROC stand in for demonstrated fairness of the 25-class product. These are different tasks.

### C8 (LOW) — Minor internal inconsistencies in the duplicate rate and headline rounding

Duplicate fraction is "~33%" in the abstract but `CLEAN_25class.json` records `pct_duplicates = 0.352` (35%), and `FINAL_RESULTS.md` says "35%." Cosmetic, but a reviewer will notice; pick one number. Similarly `ddi_dinov2_518.json` labels the darkest stratum "FST V-VI" while `_build_cancer_data.py` maps tones via a `{12,34,56}` code — worth confirming the mapping so the 0.789 is unambiguously the darkest tier.

---

## What is solid (credit where due)

- **25-class frozen-probe vs cRT is a clean, fair comparison**: same deduped external (n=3,813), frozen-probe 0.573 [0.553–0.592] vs cRT 0.527 [0.510–0.543], **disjoint 95% CIs**. This is the strongest single piece of evidence for the thesis and it holds. (Caveat: the CIs capture test-set sampling variance only, not model-training variance; both models are single runs.)
- **The bias/threshold optimization is done correctly** — `coordinate_ascent` is fit on OOF (`ytv`) and frozen before touching the external. No threshold leakage.
- **Leakage detection and honest downward revision** — exemplary; the feature-NN duplicate audit is a genuinely reusable contribution.
- **Correct diagnosis of the DDI floor** — the literature reading (Daneshjou floor vs fine-tune ceiling; DermDiff shows synthesis alone fails) is accurate and the refusal to chase 0.9+/0.99 is scientifically honest.
- **The degenerate 9-class external (99.8% nevi, EXT macro-F1 0.11) is disclosed, not buried**, and OOF CV is used instead — the right call, transparently made.

---

## Are the conclusions justified?

**Mostly yes, with two calibrations.** (1) The *qualitative* thesis — frozen-probe ≥ fine-tuning in this regime — is supported by the clean 25-class result and is consistent with established literature; it should be stated as a robust qualitative finding, which it is. (2) The *specific point estimates*, above all DDI V–VI 0.789, are oversold: they lack CIs, rest on small subgroups, were selected as the max over a config grid, and — for DDI — came from image-level (not group-disjoint) CV that trains in-domain, unlike the fine-tuned comparator. None of these individually breaks the paper, but together they mean the headline numbers should be reported with CIs, a stated selection procedure, group-disjoint DDI folds, and a same-fold end-to-end-fine-tune comparator before they carry the weight the draft puts on them. The project's own framing ("contributions are methodological rather than a leaderboard number") is the honest and correct posture; the review recommends the numbers be presented to match that humility.
