# A Frozen Multi-Backbone Foundation-Feature Probe with a Deployable Trust Stack for Dark-Skin Dermatology Decision Support

*Manuscript draft — target venue: npj Digital Medicine. All quantitative claims are drawn from the honest, leakage-corrected result artifacts in `eval/probe/` and `logs/`. No numbers in this draft were invented; items that still require an author check are flagged **TODO-verify**.*

---

## Abstract

**Background.** Dermatology artificial intelligence is dominated by light-skinned dermoscopy datasets and by end-to-end fine-tuning of convolutional or transformer backbones. Both assumptions transfer poorly to a resource-constrained, dark-skin, clinical-photograph setting such as Indian teledermatology, where labeled data and compute are scarce and disease prevalence is long-tailed and inflammatory rather than dermoscopic. There is a well-documented performance gap on darker Fitzpatrick skin types, and most deployed systems ship without calibration, uncertainty quantification, or subgroup fairness reporting.

**Methods.** We assembled approximately 266,000 labeled, de-duplicated images from eight public sources spanning dermoscopy (ISIC-2019, ISIC-2024, BCN20000) and clinical photography (Fitzpatrick17k/CleanPatrick, SCIN, DermaCon-IN, PAD-UFES-20, DDI, Mpox/MSLD), of which roughly 15,000 are Fitzpatrick IV–VI — the binding scarcity. Frozen-backbone features for the entire corpus were extracted once in a single forward pass and cached; every subsequent experiment is a lightweight head fit on those cached features, so the complete decision-support pipeline was built on a single 6 GB consumer laptop with no backbone gradient step in the primary method. The 25-condition Indian-focused task is trained on 33,250 group-aware (patient/lesion-disjoint) images and evaluated on a duplicate-audited external test set (5,886 images; 3,813 after near-duplicate removal); the 9-condition dermoscopy task is trained on 197,137 images and evaluated out-of-fold because its external partition is class-degenerate. On the Diverse Dermatology Images (DDI) benchmark of 656 biopsy-confirmed clinical photographs (one lesion per image), we evaluate malignant-versus-benign classification with image-level 5-fold stratified cross-validation, the standard DDI protocol. We compare a single frozen DINOv2 ViT-B/14 probe against a score-averaged ensemble of six frozen foundation backbones (DINOv2-B, DINOv2-B-reg, DINOv2-L, EVA-02-B, ConvNeXt-V2-B, SigLIP-B). Statistical support is provided by stratified bootstrap 95% confidence intervals (CIs), paired-bootstrap Delta-AUROC tests, and multi-seed cross-validation. Separately, on a 25-condition Indian-focused classification task we compare the frozen probe against a matched fine-tuning baseline (classifier re-training, cRT) on a duplicate-audited external test set, and we ship a trust stack: temperature-scaled calibration, RAPS conformal prediction, and per-Fitzpatrick selective prediction. Finally, a photo-grounded, confidence-gated large-language-model (LLM) triage layer engages only when the classifier is uncertain: it is shown the lesion photograph and the live differential, asks targeted history questions, and fuses the answers as per-class log-likelihood-ratios into a Bayesian posterior, bounded so that history can escalate but never clear a malignancy flag. It is integrated into the deployed command-line and web applications and is offered as an engineering contribution — it is not formally evaluated and carries no clinical claim.

**Results.** On DDI, the frozen six-backbone score-averaged ensemble reaches overall AUROC 0.826 (95% CI 0.789–0.862) and, on the darkest skin (Fitzpatrick V–VI), AUROC 0.872 (95% CI 0.808–0.929), significantly above the single-backbone frozen baseline (V–VI 0.789): paired-bootstrap Delta-AUROC overall +0.078 (95% CI 0.049–0.107, p < 0.001); V–VI +0.083 (95% CI 0.034–0.133, p = 0.0005). All ensemble members are frozen, so the gain is attributable to architectural diversity rather than adaptation. Multi-seed cross-validation is stable (overall 0.830 ± 0.009; V–VI 0.874 ± 0.007). Clinically, the ensemble reaches sensitivity 0.696 at 80% specificity overall and 0.750 within V–VI, its net benefit exceeds treat-all and treat-none at every threshold (decision-curve analysis, pt 0.05–0.50), and its malignancy score is near-calibrated (temperature 1.25; ECE 0.057 uncalibrated, 0.055 after temperature scaling). A supervised-ImageNet frozen control shows that at matched architecture and resolution self-supervision does not significantly beat supervised pretraining (0.736 vs 0.729, p = 0.37), so the gain is attributable to the frozen-representation-plus-probe recipe rather than to the SSL objective. A 4-backbone subset retains V–VI AUROC 0.870 (of 0.872) at one-third less inference cost. On the 25-condition task, the frozen probe attains macro-F1 0.573 (95% CI 0.552–0.591) versus 0.527 (95% CI 0.510–0.543) for the matched fine-tuned baseline on the duplicate-corrected external test (disjoint CIs). Under a deterministic re-run (fixed seeds, R02), temperature scaling (T = 2.2) reduces expected calibration error (ECE) from 0.21 to 0.042; RAPS conformal achieves 0.892 empirical coverage at target 0.90 (mean set size 3.44); selective prediction at a 30% referral rate raises retained accuracy to 0.766. During evaluation we detected and corrected roughly 35% train–test near-duplicate leakage in the multi-source external test via frozen-feature nearest-neighbor auditing, revising the 25-class headline down (0.626 → 0.573).

**Conclusions.** In a data-scarce, dark-skin clinical regime, keeping a foundation backbone frozen and probing an ensemble of diverse frozen representations is a simple, laptop-feasible recipe that reaches, at the point estimate, published dark-skin cancer performance ceilings, while a deployable trust stack supplies calibration, conformal coverage, and per-tone selective prediction. This is a clinical demonstration of the known linear-probe-versus-fine-tune phenomenon, not a new learning phenomenon; the contributions are the frozen multi-backbone dark-skin result with full statistical treatment, the deployable trust stack, and two honest-methodology artifacts (a frozen-feature leakage audit and a majority-class-collapse diagnosis). The system is decision support, not autonomous diagnosis.

---

## 1. Introduction

Skin disease is among the most common reasons for primary-care presentation worldwide, and image-based triage has obvious appeal in settings with few dermatologists per capita. Yet the evidence base for dermatology AI is skewed. Public benchmarks are dominated by dermoscopic images of light skin (ISIC, HAM10000), and the field's default recipe — take an ImageNet-pretrained backbone and fine-tune it end-to-end on the target task — implicitly assumes abundant in-distribution labeled data. Neither assumption holds for the setting we target: an Indian teledermatology deployment operating on smartphone clinical photographs, across darker Fitzpatrick skin types, with a long-tailed, inflammatory and infectious disease mix, and on a single consumer laptop.

Two failure modes motivate this work. First, **fairness**: multiple audits have shown that dermatology models underperform on darker skin, and the Diverse Dermatology Images (DDI) benchmark was created specifically to expose this gap on biopsy-confirmed clinical photographs. Second, **trust**: deployed models are typically shipped as bare classifiers, without calibrated probabilities, without an uncertainty set, and without any subgroup performance disclosure — precisely the artifacts a clinician needs to know when to defer.

We report what actually worked under these constraints, including two negative results that strengthen rather than weaken the account. Our central technical observation is that, in this regime, **fine-tuning degrades a strong self-supervised representation, whereas a frozen backbone with a lightweight probe preserves and exploits it.** This is not a new phenomenon: it is exactly the linear-probe-versus-fine-tune (LP-FT) behavior characterized theoretically by Kumar et al. (ICLR 2022), who show that fine-tuning can distort pretrained features and underperform out-of-distribution. We therefore frame our contribution as a **clinical, data-scarce, dark-skin demonstration** of that known phenomenon, and place the novelty elsewhere:

1. **A frozen multi-backbone ensemble for dark-skin cancer.** Score-averaging six frozen foundation backbones lifts DDI dark-skin (Fitzpatrick V–VI) AUROC to 0.872 (95% CI 0.808–0.929), significantly above the single-backbone frozen probe (0.789), with every member frozen — the gain is architectural diversity, not adaptation. This is our new headline result and, to our knowledge, the strongest reported frozen (training-free-backbone) dark-skin cancer AUROC on DDI, reported with full CIs and significance tests.
2. **A deployable trust stack** — temperature scaling, RAPS conformal prediction sets, and per-Fitzpatrick selective prediction — integrated into the shipped pipeline, an artifact almost universally omitted in comparable work.
3. **Two honest-methodology contributions**: a frozen-feature near-duplicate **leakage audit** that caught ~35% train–test contamination missed by group-aware splitting, and a **majority-class-collapse diagnosis** that traced a common-class recall failure to double-balancing.
4. **A photo-grounded, confidence-gated LLM triage layer** that only engages when the image model is uncertain and whose textual evidence is bounded so it cannot silently override a malignancy flag.

**Objectives of this research programme.** The work has two declared objectives, of unequal weight.

- **Primary objective — the vision-model research.** Develop and internally validate a trustworthy, deployable image-classification system for dermatology across skin tones under real resource constraints. This objective is complete and is what the present manuscript reports.
- **Secondary objective — a prospective image-collection sub-study (institutionally approved).** Prospectively collect 3,000–5,000 consented, dermatologist-labelled dermatology photographs across two North Indian hospitals over approximately one year, with histopathological confirmation for all suspected malignancies, to create the first North Indian dermatology image dataset. It is a sub-study of, not a successor to, the primary objective.

The secondary objective exists because of a specific and measurable weakness in the primary one. Our dark-skin cancer result rests entirely on DDI — 656 images from a single United States academic clinic — and our 25-condition per-tone estimates rest on roughly 135 darker-skin external images spread across 25 conditions. Neither supports a claim about Indian patients. The sub-study is designed to convert that limitation into evidence; its protocol, capture standard operating procedure, metadata schema and labelling/biopsy-linkage workflow are specified in the accompanying study documents and summarised in Section 8.2. Data collection had not begun at the time of writing, and **no result in this manuscript depends on it.**

Throughout, every clinical claim is hedged: this is decision support, not diagnosis.

---

## 2. Related Work

**Foundation models and linear probing.** Self-supervised vision transformers such as DINOv2 (Oquab et al., 2023) produce general-purpose features that transfer strongly under linear evaluation. Kumar et al. (2022) formalized when fine-tuning helps versus hurts: under distribution shift, full fine-tuning can distort pretrained features and a linear probe on frozen features generalizes better. In dermatology specifically, frozen-feature and linear-probe approaches have been reported to be competitive with or superior to fine-tuning in low-label regimes (e.g., FEDD, MICCAI 2023, **TODO-verify exact venue/claim**). Our work is a clinical instantiation of this line, extended to a multi-backbone frozen ensemble.

**Dark-skin dermatology and fairness.** Groh et al. (2021) introduced Fitzpatrick17k and documented tone-dependent performance. Daneshjou et al. (2022, *Science Advances*) released DDI, a biopsy-confirmed clinical-photograph benchmark, and showed that off-the-shelf models degrade sharply on Fitzpatrick V–VI. Their reported off-the-shelf AUROCs are 0.56–0.67 **overall** (DeepDerm 0.56, ModelDerm 0.65, HAM10000-trained 0.67) and 0.50–0.57 **within FST V–VI**; after fine-tuning on DDI, V–VI performance rose to 0.76 (DeepDerm) and 0.78 (HAM10000-trained). We position our frozen-ensemble point estimate against that adapted ceiling, noting throughout that their evaluation used a randomized 60/20/20 split whereas ours uses 5-fold cross-validation.

**Domain-specific dermatology foundation models.** Recent efforts train derm-specific encoders — PanDerm (a multimodal clinical-dermatology vision foundation model), Google's **Derm Foundation** embedding model (released within Health AI Developer Foundations), and DermINO; SCIN is a crowdsourced dermatology *dataset* rather than an encoder and is cited as such. We did not use these as our primary backbone: one candidate (PanDerm) mis-loaded in our environment (unexpected checkpoint keys) and its license is non-commercial, so we report only general-purpose frozen backbones that are fairly evaluated and deployable.

**Trustworthy prediction.** Temperature scaling (Guo et al., 2017) is the standard post-hoc calibration method. Conformal prediction (Vovk et al.) and its adaptive variants — APS (Romano et al., 2020) and RAPS (Angelopoulos et al., 2021) — provide finite-sample coverage guarantees; clustered/Mondrian variants extend this to subgroups. Selective prediction / classification with a reject option provides an accuracy–coverage trade-off. We assemble these into a single deployable stack and report them per Fitzpatrick tone.

**Long-tailed learning.** Class-balanced losses (Cui et al., 2019), LDAM and deferred re-weighting (Cao et al., 2019), and decoupled classifier re-training (cRT; Kang et al., 2020, **TODO-verify**) address class imbalance. We use class-balanced focal loss and cRT as the fine-tuning comparator and diagnose a collapse caused by naively stacking two balancing mechanisms.

**Backbones benchmarked in the ensemble.** DINOv2 (Oquab et al., 2023), EVA-02 (Fang et al., 2023), ConvNeXt-V2 (Woo et al., 2023), and SigLIP (Zhai et al., 2023) span self-supervised, masked-image, convolutional, and contrastive language-image pretraining respectively — the architectural diversity we exploit.

---

## 3. Methods

### 3.1 Data

The training corpus comprises approximately 266,000 labeled, de-duplicated images spanning dermoscopy (ISIC-2019, ISIC-2024, BCN20000) and clinical photographs (Fitzpatrick17k cleaned via CleanPatrick, SCIN, DermaCon-IN, PAD-UFES-20, DDI, and Mpox/MSLD). Darker-skin (Fitzpatrick IV–VI) coverage is approximately 15,000 images — the binding scarcity. No synthetic data and no scraping beyond public research datasets were used. Splits are group-aware (patient/lesion-disjoint); Section 3.6 details a residual leakage correction.

From this corpus we derive two multiclass tasks. The **25-condition Indian-focused task** comprises 39,136 images — 33,250 in the training/cross-validation partition and 5,886 in a held-out external test partition (reduced to 3,813 after the near-duplicate correction of Section 3.6). The **9-condition dermoscopy task** comprises 248,172 images — 197,137 training/cross-validation and 51,035 external; because its external partition is 99.8% nevi (class-degenerate), we report out-of-fold cross-validation for that task instead. Backbone features for every image in both tasks were extracted once and cached (a single inference pass, no gradients), after which all probe fitting, calibration, conformal, and ablation experiments run on CPU from cache — this is what makes the full study laptop-feasible.

**Table 0. Task partitions.**

| Task | Classes | Train / CV | External test | Evaluation used |
|---|---|---|---|---|
| 25-condition (Indian-focused) | 25 | 33,250 | 5,886 → **3,813** deduped | de-duplicated external |
| 9-condition (dermoscopy) | 9 | 197,137 | 51,035 (99.8% nevi) | out-of-fold CV |
| DDI (dark-skin cancer) | 2 | — | 656 (all) | image-level 5-fold CV |

**DDI (dark-skin cancer benchmark).** DDI contains 656 biopsy-confirmed clinical photographs drawn from **570 unique patients**, with Fitzpatrick skin-type annotations. The publicly released metadata (`ddi_metadata.csv`) provides a unique image identifier, skin-tone group, malignancy label and disease name, but **no patient identifier**, so patient-grouped cross-validation is not constructible from the public release. We therefore use image-level 5-fold cross-validation, stratified by tone × malignancy.

Two consequences must be stated plainly rather than assumed away. First, because 656 images come from 570 patients, roughly 80–90 images share a patient with another image; a small number of same-patient pairs can therefore fall on opposite sides of a fold boundary. This is a residual source of optimism that we cannot eliminate with the released data, and we disclose it rather than claim lesion-level disjointness (an earlier internal description of these folds as "group-disjoint" was incorrect and has been withdrawn). Second, our protocol is *not* the protocol of the original DDI paper: Daneshjou et al. (2022) evaluated published models on a randomized 60/20/20 train–validation–test split, whereas we report 5-fold cross-validation because our probe is trained on DDI itself. Comparisons to their reported numbers are therefore protocol-mismatched and are framed accordingly throughout.

### 3.2 Frozen-probe recipe (primary method)

The backbone (DINOv2 ViT-B/14, `vit_base_patch14_dinov2.lvd142m`) is frozen at native 518 px input, producing a 768-dimensional pooled feature per image. Features are cached once. For each cross-validation fold, features are standardized with a `StandardScaler` fit on the training folds only, and a class-balanced logistic-regression head (`LogisticRegression(class_weight="balanced", C=0.5)`) is fit for the DDI binary task. For the 25-condition task, the head is a shallow class-balanced MLP (single 512-unit hidden layer) plus a post-hoc per-class logit bias optimized by coordinate ascent on out-of-fold (OOF) predictions. No backbone gradient step is taken in the primary method. The scaler and head see only training-fold features, and features are label-free, so no label leakage crosses folds.

### 3.3 Frozen multi-backbone ensemble (headline result)

Six frozen backbones are probed independently under the identical DDI protocol: DINOv2-B @518, DINOv2-B-reg @518, DINOv2-L @518, EVA-02-B @448, ConvNeXt-V2-B @384, and SigLIP-B @384. For each, per-fold features are standardized and a class-balanced logistic head yields OOF P(malignant). The **ensemble is the arithmetic mean of the six backbones' OOF malignancy scores** (score-averaging outperformed feature concatenation: overall 0.826 vs 0.808). All members are frozen; the ensemble introduces no fine-tuning.

### 3.4 Trust layer

On the 25-condition external test we fit and evaluate three post-hoc components. **Temperature scaling** learns a single scalar T on held-out logits to minimize calibration error. **RAPS conformal prediction** (regularized adaptive prediction sets) produces label sets with a target marginal coverage of 0.90, using regularization parameters k_reg = 2, λ = 0.05. **Selective prediction** sweeps a referral rate and reports retained accuracy and macro-F1 on the non-referred fraction. Results are additionally broken out per Fitzpatrick tone group. We note as a limitation (Section 7) that the conformal/temperature parameters should be fit on a calibration split strictly disjoint from the evaluation set; a controlled re-run to guarantee this split is queued and is not part of tonight's frozen artifacts.

### 3.5 Photo-grounded LLM triage

Image alone is an incomplete clinical input: duration, itch, bleeding, evolution and exposure history routinely separate diagnoses that look alike. We therefore add a triage layer that behaves like a junior clinician taking a focused history — but only when the image model admits it is unsure.

**Gating.** The layer is *confidence-gated*: it engages only when the classifier is uncertain, defined as a diffuse posterior (effective number of classes above a threshold, k = 4) **or** a small top-1/top-2 margin (τ = 0.15). When the image model is confident, no interview occurs and no language model is called — the classifier's answer stands unmodified. This keeps the image model the decision-maker and bounds both latency and cost.

**Photo-grounded question generation.** When triggered, the lesion photograph itself is passed to a vision-capable language model (Gemini 2.5 Flash) together with the live differential and any structured metadata. The model is instructed to look at the lesion — morphology, colour, scale, distribution — and to ask the questions whose answers, *combined with what it sees*, best separate the competing diagnoses. In practice the questions reference visible features (e.g. for an ambiguous pigmented lesion it asked "Has this reddish, somewhat irregular lesion changed in size, shape, colour or texture recently?"), which is the behaviour that distinguishes this from a fixed history questionnaire.

**Bayesian fusion.** Free-text answers are converted, per candidate condition, into a log-likelihood-ratio in [−2, 2] expressing how much the *history alone* supports or argues against that condition; the model is explicitly instructed not to restate the image probabilities. These ratios update the calibrated image prior in log space, yielding a refined posterior, together with any red flags and a one-line rationale. In a worked example, an image differential of seborrhoeic keratosis 31% / naevus 28% / melanoma 22% moved to melanoma 73% after a history of three months' growth, darkening and one bleeding episode, with "growth and darkening" and "bled once" surfaced as red flags.

**Safety bound.** The text channel may *raise* concern freely but its downward influence on malignant classes is bounded, so no history can pull a calibrated malignant probability below the referral line; the layer can escalate but never clear a malignancy flag.

**Implementation status.** The layer is integrated into the deployed system: it runs in the command-line pipeline (`demo_app/pipeline.py`) and in the web application, where an uncertain case renders the generated questions, accepts answers, and displays the before/after differential with the per-class change. It is offered as an **engineering contribution and is not formally evaluated**: we report no accuracy, calibration, or reader-study metric for it, and we make no clinical claim on its behalf. Prospective evaluation — including whether the fused posterior is better calibrated than the image posterior, and a certified melanoma-sensitivity floor — is future work (Section 8).

### 3.6 Leakage audit

The first 25-condition external evaluation produced inflated numbers. A frozen-feature nearest-neighbor analysis (cosine similarity in the same 768-d space) revealed that approximately 35% of external images were near-exact duplicates of training images — cross-source atlas photographs (e.g., DermNet, SkinDisease) that had received distinct group identifiers and thus evaded the group-aware split. We removed pairs above cosine 0.99 and re-reported; all 25-class numbers in Section 5 reflect this correction. The pure-ISIC 9-class split was duplicate-free (0%) but class-degenerate in its external partition (99.8% nevi), so we report OOF cross-validation there instead. We recommend feature-space duplicate auditing as a standard check for any multi-source medical dataset.

### 3.7 Majority-class-collapse diagnosis

An early fine-tuned model exhibited majority-class collapse (e.g., eczema recall 0.004). We traced this to stacking two rebalancing mechanisms — an inverse-frequency sampler on top of a class-balanced focal loss — which double-counts the correction and drives the head toward degenerate solutions. Removing the sampler and using a single balancing mechanism plus decoupled cRT recovered common-class recall (eczema 0.004 → 0.503).

### 3.8 Statistics

For DDI we compute stratified bootstrap 95% CIs (resampling within tone strata) on every AUROC, overall and per tone. Comparative claims use a **paired bootstrap Delta-AUROC** test (resampling the same held-out scores for both models jointly) with a one-sided p-value for the ensemble exceeding the baseline. Stability is assessed by repeating the full cross-validation over five fold seeds and reporting mean ± SD. For the 25-condition task, macro-F1 CIs are bootstrap over the external test rows; the frozen-probe-vs-cRT comparison is read from disjoint CIs. These CIs capture test-set sampling variance; they do not capture model-training variance (both 25-class models are single runs), which we flag as a limitation.

### 3.9 Clinical utility, ensemble efficiency, and a supervised-pretraining control

Three additional analyses, all on cached frozen features (no retraining, no GPU), strengthen the DDI leg and de-confound the central claim.

**Clinical operating characteristics (`scripts/_ddi_clinical.py`, `eval/probe/ddi_clinical.json`).** For the frozen six-backbone score-average ensemble we read sensitivity at fixed specificity (and vice versa) off the empirical OOF ROC, overall and within FST V–VI; compute a **decision-curve net-benefit** analysis across threshold probabilities pt = 0.05–0.50 (NB = TP/n − FP/n · pt/(1−pt), compared against treat-all and treat-none defaults); and calibrate the ensemble malignancy **score itself** by temperature scaling, with the temperature fit by NLL and cross-fit over the five folds for an honest 10-bin ECE. This addresses the operating-point / decision-curve / score-calibration gap (R14) previously listed as owed.

**Ensemble efficiency frontier (`scripts/_ensemble_subset.py`, `eval/probe/ensemble_subset.json`).** Each backbone's 5-fold stratified OOF P(malignant) is computed once under the identical protocol, then score-averaged over all 63 non-empty subsets of the six backbones. We report the accuracy-vs-cost frontier (best subset per size k by FST V–VI AUROC) and a recommended deployable subset.

**Supervised-ImageNet frozen control (`scripts/_ddi_supervised_control.py`, `eval/probe/ddi_supervised_baseline.json`).** To separate "freezing wins" from "DINOv2's SSL objective wins", we probe a **supervised** ImageNet ViT-B (`vit_base_patch16_224.augreg2_in21k_ft_in1k`) against a **self-supervised** DINOv2 ViT-B, both frozen, both at matched 224 px, under the identical DDI probe recipe, and compare with a paired bootstrap Delta-AUROC (4000 stratified resamples). The DINOv2-B@224 arm reproduced the reference `ddi_dinov2.json` overall AUROC (0.7364) exactly, validating the pipeline before comparison. This is the supervised-frozen control (R08) previously listed as missing.

**Intra-corpus OOF leakage audit (`eval/probe/leakage_oof.json`).** We extended the frozen-feature near-duplicate audit from the external test to the multiclass **OOF folds themselves**: for every validation row we measured cosine similarity to its own fold's training partition and counted same-label near-duplicates at thresholds 0.98/0.99. This tests whether the 25-class OOF macro-F1 (0.592) and 9-class OOF macro-F1 (0.660) are themselves optimistic from within-corpus duplication (R06).

---

## 4. Results — DDI dark-skin cancer (headline)

The frozen six-backbone score-averaged ensemble is the strongest configuration overall and on both darker strata; on the lightest stratum (FST I–II) the single DINOv2-L backbone is marginally higher (0.787 vs 0.782), well within overlapping confidence intervals. Table 1 gives point estimates with 95% CIs; per-backbone AUROCs are in the supplementary artifact `eval/probe/ddi_ensemble.json`.

**Table 1. DDI malignant-vs-benign AUROC (image-level 5-fold stratified CV, 656 images).**

| Model | Overall | FST V–VI (dark) | FST III–IV | FST I–II |
|---|---|---|---|---|
| Single DINOv2-B @518 (frozen baseline) | 0.748 [0.704, 0.791] | 0.789 [0.713, 0.859] | 0.739 [0.670, 0.806] | 0.732 [0.643, 0.811] |
| DINOv2-L @518 (best single backbone) | 0.791 | 0.818 | 0.781 | 0.787 |
| Ensemble — concat features (6 backbones) | 0.808 | 0.834 | 0.816 | 0.777 |
| **Ensemble — score-average (6 backbones)** | **0.826 [0.789, 0.862]** | **0.872 [0.808, 0.929]** | **0.830 [0.771, 0.882]** | **0.782 [0.701, 0.856]** |

CIs are stratified bootstrap 95%. Baseline per-tone CIs from `eval/probe/ddi_ensemble_ci.json`.

**Significance.** The paired-bootstrap Delta-AUROC (ensemble − single-backbone baseline) is significant overall and in the darker strata:

- Overall: +0.078 (95% CI 0.049–0.107), p < 0.001.
- FST V–VI: +0.083 (95% CI 0.034–0.133), p = 0.0005.
- FST III–IV: +0.091 (95% CI 0.042–0.141), p = 0.00025.
- FST I–II: +0.050 (95% CI −0.003–0.104), p = 0.032 (the only stratum whose CI touches zero, consistent with its smaller malignant count).

**Stability.** Across five fold seeds, overall AUROC is 0.830 ± 0.009 and FST V–VI is 0.874 ± 0.007, so the point estimates are not lucky-fold artifacts.

**Reading.** Dark-skin (V–VI) AUROC improves from 0.789 to 0.872, a real and significant gain. For orientation — with the protocol caveat of Section 3.1 firmly in mind — Daneshjou et al. (2022) reported off-the-shelf models on DDI at overall AUROC 0.56–0.67 (0.50–0.57 within FST V–VI), and after fine-tuning *on DDI itself* reached FST V–VI AUROC of 0.76 (DeepDerm) and 0.78 (HAM10000-trained). Our frozen ensemble's V–VI point estimate of 0.872 is therefore above that published adapted range, though its confidence interval (0.808–0.929) is wide and the two protocols differ (their 60/20/20 split versus our 5-fold cross-validation), so we present this as *reaching and plausibly exceeding* the published ceiling rather than as a like-for-like win. Because all six members are frozen, the improvement is attributable to architectural diversity across self-supervised, masked-image, convolutional, and contrastive backbones, not to adapting any one of them. We deliberately do not claim 0.9+/0.99: dark-skin clinical cancer imagery is genuinely scarce (DDI is essentially the only public source), the V–VI stratum rests on a modest number of malignant cases, and the resulting CI is correspondingly wide.

**Fairness (light–dark AUROC gap).** Daneshjou et al. (2022, *Science Advances* 8:eabq6147) evaluated three published models on DDI **zero-shot** (no DDI adaptation) and reported a substantial *light-minus-dark* AUROC gap in every case: DeepDerm 0.61 (FST I–II) vs 0.50 (V–VI), a gap of +0.11; ModelDerm 0.64 vs 0.55, +0.09; and a HAM10000-trained model 0.72 vs 0.57, +0.15 (overall AUROCs 0.53, 0.60 and 0.65 respectively) — i.e., all three are markedly worse on dark skin, and DeepDerm is at chance there. We stress that those are **zero-shot** evaluations whereas ours is cross-validated on DDI, so the comparison speaks to the *presence or absence of a tone gap* under each recipe, not to a head-to-head accuracy ranking. Our frozen ensemble's gap is −0.09 (numerically slightly better on dark skin: V–VI 0.872 vs I–II 0.782), but the light and dark stratum CIs overlap, so the honest reading is **no statistically significant disparity across tone**, not "better on dark skin." We report this as the absence of the usual dark-skin penalty under the frozen-ensemble recipe, with the strong caveat that the V–VI stratum is small-n (see Limitations).

### 4.1 Clinical utility of the ensemble malignancy score

AUROC is threshold-free; a screen also needs operating points, net benefit, and a calibrated score. All figures below are for the frozen six-backbone score-average ensemble on DDI (n = 656, 171 malignant, prevalence 0.261; FST V–VI n = 207, 48 malignant), from `eval/probe/ddi_clinical.json`.

**Operating points (empirical ROC).** Overall, sensitivity at 80% specificity is **0.696** and at 90% specificity is **0.538**; specificity at 90% sensitivity is 0.538. Within FST V–VI the dark-skin operating points are consistently *better* than overall — Se@Sp0.80 **0.750**, Se@Sp0.90 **0.625**, Sp@Se0.90 0.635 — consistent with the higher V–VI AUROC (0.872).

**Table 1b. DDI ensemble operating points.**

| Operating point | Overall | FST V–VI |
|---|---|---|
| Sensitivity @ 80% specificity | 0.696 | 0.750 |
| Sensitivity @ 90% specificity | 0.538 | 0.625 |
| Specificity @ 90% sensitivity | 0.538 | 0.635 |

*Note.* Overall-stratum operating points are read at essentially the requested thresholds (achieved specificity 0.804 and 0.901; achieved sensitivity 0.901). The FST V–VI points are read at the nearest achievable step of the empirical ROC — specificity 0.943 for the Se@Sp0.90 row and sensitivity 0.917 for the Sp@Se0.90 row — because that stratum contains only 207 images (48 malignant), so the ROC advances in coarse steps.

**Net benefit (decision curve).** Across every threshold probability pt = 0.05–0.50, the model's net benefit exceeds both treat-all and treat-none; treat-all turns negative just above pt = 0.26 (its zero-crossing is the prevalence, 0.261) while the model stays positive throughout (e.g., overall pt = 0.30: NB_model 0.111 vs treat-all −0.056). The advantage over the best default peaks in the mid-threshold band (overall delta ≈ 0.114 at pt = 0.25). The same pattern holds within FST V–VI.

**Score calibration.** Temperature scaling of the ensemble logit gives T = 1.25 (cross-fit, mild over-confidence — the ensemble score is already near-calibrated). ECE (10-bin) falls from 0.0571 to 0.0553 overall and from 0.0831 to 0.0819 within FST V–VI. The small gain reflects the near-unit temperature; V–VI remains somewhat less calibrated than overall (a fairness caveat, and ECE over only 207 V–VI images across 10 bins is noisy). We hedge these as favorable but not clinically validated operating characteristics on a single small dark-skin source.

### 4.2 Ensemble efficiency — how many backbones to deploy

Score-averaging all 63 non-empty subsets of the six backbones (identical protocol) traces an accuracy-vs-cost frontier (best subset per size k by FST V–VI AUROC), from `eval/probe/ensemble_subset.json`.

**Table 1c. Ensemble-size frontier (best subset per k, by FST V–VI AUROC).**

| k | Overall | FST V–VI | Members |
|---|---|---|---|
| 1 | 0.791 | 0.818 | DINOv2-L |
| 2 | 0.812 | 0.838 | + SigLIP-B |
| 3 | 0.816 | 0.859 | + DINOv2-B |
| **4 (recommended)** | **0.820** | **0.870** | **+ ConvNeXt-V2-B** |
| 5 | 0.827 | 0.870 | + EVA-02-B |
| 6 (full) | 0.826 | 0.872 | + DINOv2-B-reg |

The deployment sweet spot is the **4-backbone** ensemble {DINOv2-B, ConvNeXt-V2-B, SigLIP-B, DINOv2-L} at V–VI 0.870 / overall 0.820 — dropping two of six backbones (DINOv2-B-reg and EVA-02-B) cuts inference cost by one third while giving up only **0.0021 V–VI AUROC** (0.872 → 0.870) versus the full ensemble. Returns diminish sharply after k = 4 (k=4→5→6 add only +0.0003 and +0.0018 V–VI). Every frontier subset includes DINOv2-L, and the non-DINOv2 backbones (SigLIP at k=2, ConvNeXt-V2 at k=4) drive most of the marginal dark-skin gains — reinforcing that the ensemble benefit comes from backbone diversity.

### 4.3 Supervised-pretraining control — is it freezing, or is it DINOv2?

At matched architecture (ViT-B) and resolution (224 px), a **frozen supervised ImageNet** ViT-B scores overall AUROC **0.729 [0.683, 0.774]** versus **frozen DINOv2**'s **0.736 [0.692, 0.779]** on DDI — a statistical tie (paired Delta-AUROC +0.007 [−0.039, 0.052], p = 0.365). On the target dark-skin V–VI subgroup the supervised model is in fact numerically *higher* (0.781 vs 0.740; Delta −0.041, p = 0.85). SSL edges ahead only on FST III–IV (+0.065, p = 0.047, borderline). The DINOv2-B@224 arm reproduced the published 0.7364 exactly.

**Interpretation.** This is an honest negative result that tightens what we can claim. At matched arch/resolution, self-supervision does **not** significantly beat supervised pretraining on DDI dark-skin cancer. DINOv2's large gain over the old fine-tuned baseline (~0.53–0.58 → ~0.74) is therefore attributable to using a **frozen strong representation + probe**, not specifically to the SSL objective. The genuine levers remain the multi-backbone **ensemble** (0.826 / V–VI 0.872) and higher **resolution** (@518 → 0.748 single-backbone), not SSL-vs-supervised. The paper can claim "frozen foundation-model probe beats fine-tuning"; it cannot claim "self-supervision is the reason."

---

## 5. Results — 25-condition classification and the trust stack

**Classification.** On the duplicate-corrected external test (n = 3,813), the frozen probe (DINOv2 @518 + MLP + per-class bias) attains macro-F1 **0.573 (95% CI 0.552–0.591)**, versus **0.527 (95% CI 0.510–0.543)** for the matched fine-tuned classifier-re-training (cRT) baseline. The CIs are disjoint, so the frozen-probe advantage on this task is statistically supported. On the uncorrected external test (n = 5,886) the same models score 0.619 and 0.540 respectively; the ~0.05 downward revision of the frozen-probe number (0.619 → 0.573) is entirely the effect of the leakage correction, and we report the corrected number as canonical.

**Table 2. 25-condition macro-F1, frozen probe vs matched fine-tune.**

| Test partition | Frozen probe | cRT fine-tune |
|---|---|---|
| Duplicate-corrected external (n = 3,813) | **0.573 [0.553, 0.592]** | 0.527 [0.510, 0.543] |
| Full external (n = 5,886, leakage-inflated) | 0.619 [0.602, 0.635] | 0.540 [0.523, 0.555] |

**9-condition task.** On the ISIC-derived 9-class task, whose external partition is class-degenerate, we report OOF cross-validation macro-F1: frozen probe **0.660 (95% CI 0.654–0.665; stratified bootstrap, `rigor_main_ci.json`)** versus fine-tuned baseline 0.637. The frozen-feature OOF duplicate audit (§3.9, `leakage_oof.json`) confirms this 9-class OOF is essentially free of within-corpus near-duplicates (0.46% same-label), so — unlike the 25-class OOF — it is not duplicate-inflated.

**Resolution.** At matched head and protocol, the 518 px probe beats the 224 px probe on 25-class OOF macro-F1 by +0.024 (95% CI 0.018–0.032, paired bootstrap, p < 0.001; 518 OOF 0.592 [0.585, 0.599] vs 224 OOF 0.568 [0.560, 0.574]), confirming input resolution — not the pretraining objective — as a genuine lever.

**Calibration.** Under the deterministic re-run (fixed seeds, R02), temperature scaling with T = 2.2 reduces ECE from **0.21 to 0.042** on the external test. (The previously reported 0.016 was a non-deterministic artifact of the fixed-bin ECE estimator on the clean external and is superseded; 0.042 is the honest, reproducible value.)

**Conformal coverage.** RAPS conformal prediction achieves **0.892 empirical marginal coverage** at target 0.90, with a mean prediction-set size of 3.44 labels (λ = 0.05).

**Selective prediction.** Deferring the least-confident cases monotonically improves the retained cohort:

**Table 3. Selective prediction (25-class external).**

| Referral rate | Retained accuracy | Retained macro-F1 |
|---|---|---|
| 0.0 | 0.637 | 0.567 |
| 0.10 | 0.681 | 0.601 |
| 0.20 | 0.723 | 0.631 |
| 0.30 | **0.766** | 0.665 |

**Fairness by tone (25-class).** We report the per-Fitzpatrick balanced accuracy honestly, including its weakness. On the corrected external test the frozen probe attains FST I–III balanced accuracy 0.264 (n = 642) and FST IV–VI 0.454 (n = 135). These subgroup estimates are small-n and unstable (an earlier model showed the opposite direction), so we do **not** present the 25-class product as demonstrably fair across tones; the DDI binary-cancer AUROC and its per-tone CIs are a different, better-powered task and should not be read as fairness of the 25-class classifier. Mean confidence is comparable across the two bins (0.548 vs 0.545).

---

## 6. Discussion

The results cohere into a single practical thesis for the data-scarce, dark-skin clinical regime: **do not fine-tune the backbone; freeze it, probe it, and diversify the frozen backbones.** On DDI, where an earlier fine-tuned model sat at chance on dark skin (V–VI AUROC 0.53), a frozen single-backbone probe reaches 0.789 and a frozen six-backbone ensemble reaches 0.872 with a significant, seed-stable margin. On the 25-condition task, the frozen probe beats a matched fine-tune with disjoint CIs. This is consistent with, and we argue is a clinical instance of, the LP-FT phenomenon of Kumar et al. (2022): fine-tuning on a small, imbalanced, multi-domain corpus distorts a strong self-supervised representation, whereas a frozen backbone preserves the general structure that a light head exploits.

We are careful about attribution. The DDI frozen-vs-fine-tune comparison in our earlier internal reports differed on more than backbone treatment (training data, resolution, class mix), so we present the DDI leg as a *demonstration* consistent with the thesis rather than a fully controlled causal test; the clean, protocol-matched causal evidence is the 25-class frozen-probe-vs-cRT comparison on identical features. The novelty we assert is not the phenomenon but (i) the frozen multi-backbone dark-skin result with full statistics, (ii) the deployable trust stack, (iii) the leakage-audit and collapse-diagnosis methodology, and (iv) the bounded LLM triage design.

The **ensemble gain is architectural, not adaptive.** The six backbones span self-supervised (DINOv2), masked-image (EVA-02), purely convolutional (ConvNeXt-V2), and contrastive language-image (SigLIP) pretraining. Averaging their malignancy scores reduces variance and, evidently, exploits complementary error structure, most strongly on dark skin. This is an attractive property for deployment: no backbone is trained, so the ensemble is reproducible from cached features on a laptop.

The **trust stack** is what makes the system deployable rather than merely accurate. A clinician needs calibrated probabilities (ECE 0.042), an uncertainty set with a coverage guarantee (RAPS 0.892 at target 0.90), and a principled referral rule (selective prediction). Shipping these together, and reporting them per tone, is uncommon in comparable dermatology-AI work and is, in our view, a precondition for responsible use.

---

## 7. Limitations

- **Residual patient clustering in the DDI folds (cannot be eliminated).** DDI's 656 images come from 570 unique patients, but the public metadata carries no patient identifier, so patient-grouped folds are not constructible. Approximately 80–90 images therefore share a patient with another image, and some same-patient pairs will straddle a fold boundary, which biases the cross-validated AUROC slightly upward. We previously described these folds as group-disjoint; that was incorrect and is withdrawn. The magnitude is bounded — at most ~13% of images are affected and only the subset splitting across folds contributes — but it is a real, unquantified source of optimism in the headline 0.872, and it would be removed only by a release of DDI patient identifiers or by an independent dark-skin cohort.
- **Single dark-skin source.** The entire dark-skin cancer claim rests on DDI (656 images, single US academic-clinic source). There is no second dark-skin clinical cancer benchmark and no Indian held-out clinical cancer test, so external validity is unestablished. DDI is acknowledged as essentially the only public dark-skin clinical cancer set, so this is a data limitation, not a method limitation, but it caps the strength of the claim.
- **Underpowered subgroups.** The FST V–VI AUROC rests on a modest number of malignant cases, yielding a wide CI (0.808–0.929); the 25-class per-tone balanced accuracies (n = 642 and n = 135) are small-n, unstable, and weak (0.264 / 0.454). We report them but do not claim demonstrated 25-class fairness. No multiple-comparison correction is applied across tone bins.
- **Supervised-ImageNet frozen control (R08) — now run; it weakens the SSL attribution.** We probed a frozen supervised ImageNet ViT-B against a frozen DINOv2 ViT-B at matched arch/resolution (§4.3). Overall AUROC is a tie (0.729 vs 0.736, p = 0.365) and on FST V–VI the supervised model is numerically higher (0.781 vs 0.740). The frozen-probe advantage over fine-tuning holds, but we can **no longer** attribute it to the self-supervised objective; we have amended the claims accordingly.
- **Intra-corpus OOF near-duplication (R06) — audited; the 25-class OOF number is optimistic.** The frozen-feature audit of the multiclass OOF folds (`leakage_oof.json`) shows the 25-class OOF has **~30% of validation rows with a same-label near-duplicate (cosine > 0.98) in their own fold's training partition** (33.4% at any label; max cosine ≈ 1.0), so the 25-class OOF macro-F1 (0.592) is inflated by within-corpus atlas duplication — the corrected external test (n = 3,813; macro-F1 0.573) remains the canonical 25-class number, not the OOF value. By contrast the **9-class OOF is essentially clean** (only 0.46% same-label near-duplicates, p99 cosine ≈ 0.97), so the 9-class OOF macro-F1 (0.660) is *not* materially duplicate-inflated. A stratified-bootstrap CI on the 9-class OOF has since been produced (0.660, 95% CI 0.654–0.665; `rigor_main_ci.json`), closing that owed item.
- **Protocol-matched fine-tune on DDI folds (R17).** A clean end-to-end fine-tune on the identical DDI-only folds and resolution is not reported; it requires GPU training. Until then the DDI frozen-vs-fine-tune leg is suggestive, not fully controlled.
- **Operating point / decision curve (R14) — now reported for DDI.** We added ensemble operating points (Se@Sp0.80 0.696 overall / 0.750 V–VI), a decision-curve net-benefit analysis (model > treat-all and treat-none at all pt 0.05–0.50), and calibration of the cancer score itself (T = 1.25; ECE 0.057 → 0.055 overall, 0.083 → 0.082 V–VI). These are single-source, small-n (especially the 207-image V–VI ECE) and are reported as favorable but not clinically validated.
- **Winner's curse and calibration split (R05) — now run; the model-selection concern is retired, one caveat remains.** Two risks were tested (`r05_winners_curse.json`). First, *model selection*: all four candidate configurations (224/518 px × logistic/MLP head) were scored on out-of-fold data only, with the external test unseen. Selection on out-of-fold macro-F1 chooses **518 px + MLP** — the configuration already reported — and the choice is identical whether or not the per-class bias is applied, so the headline configuration was not selected by peeking at the test set and the 0.573 is not inflated by a winner's curse. Second, *conformal fitting*: with the temperature and conformal quantile fit on a calibration slice disjoint from the evaluation set (T = 1.9, q̂ = 0.971), empirical coverage on the external test is **0.925** against a 0.90 target with a mean set size of **3.45** — coverage holds, and the set size is slightly smaller than the previously reported ~4. **Caveat:** the R05 script scored the selected configuration on the *full* external partition (n = 5,886), which still contains the near-duplicate contamination of Section 3.6; its 0.629 therefore reproduces the known leakage-inflated value and is **not** a corrected version of 0.573. The de-duplicated 0.573 (n = 3,813) remains canonical. A re-run of R05 restricted to the de-duplicated partition is the small remaining task.
- **Determinism (R02) — now applied.** A fixed `random_state` (seed 0) was added to every sklearn probe/head fit and CV split, plus `np.random.seed(0)` before each OOF loop, and the headline artifacts were regenerated (`r02_determinism.json`). The honest external 25-class macro-F1 is unchanged (0.573 → 0.5728, Δ −0.0002, within tolerance); temperature and conformal set size moved trivially; the ECE-after value settled deterministically at 0.042 (from the prior non-deterministic 0.016). Recipe, architecture, splits, and hyperparameters were untouched. A full multi-seed mean ± SD re-issue of every number is still a reasonable next step.
- **No reader study.** There is no dermatologist comparison or prospective validation. The system is decision support, not autonomous diagnosis.

---

## 8. Future Work

Several previously-listed items are now complete: the supervised-ImageNet frozen control (R08, §4.3), the DDI operating-point / decision-curve / score-calibration analysis (R14, §4.1), the OOF duplicate audit for both the 25-class and 9-class caches (R06, §7), the deterministic controlled re-run (R02, §7), and the main-metric CIs including the 9-class OOF (0.660 [0.654, 0.665]) and the 518-vs-224 paired test (`rigor_main_ci.json`). Remaining laptop-feasible items: the dedup-threshold sensitivity curve (0.95–0.995) to show the 0.573 number is not a knife-edge artifact; a full multi-seed mean ± SD re-issue of every headline number; and a re-run of the winner's-curse / calibration-split check (R05) restricted to the de-duplicated external partition (the completed run scored the full, leakage-inflated partition; see Section 7). GPU-dependent items: a protocol-matched DDI fine-tune (R17). Data-dependent items: a second dark-skin cancer benchmark and a held-out Indian clinical test set. Methodological extensions suggested by the frozen-feature geometry include per-Fitzpatrick Mondrian conformal with small-sample coverage certificates, tone-conditional feature alignment as a fairness-adaptation probe of whether ~0.87 is a data or feature-usage ceiling, and formal calibration of the LLM-evidence fusion with a certified melanoma-sensitivity floor.

### 8.2 Secondary objective: a prospective North Indian image collection

The single most consequential limitation of this work is not methodological but evidential: there is no Indian dermatology image dataset with which to test the system on the population it was built for, and the world's only public dark-skin clinical cancer benchmark is American. The approved secondary objective addresses this directly.

**Design.** Two hospitals, approximately one year, targeting 3,000–5,000 consented images, run as two arms. A **prospective arm** accrues the common North Indian outpatient conditions (eczema/dermatitis, tinea and other fungal infections, psoriasis, vitiligo and pigmentary disorders, acne, urticaria, scabies, leprosy, lichen planus, bacterial and viral infections, drug eruptions). A **retrospective pathology-archive arm** supplies histopathologically confirmed malignancies, because skin cancer is comparatively rare in brown skin and the target of ≥170 biopsy-confirmed cancers — the number needed to match DDI's 171 — will not accrue prospectively within a year. DDI itself was assembled this way.

**Targets and their basis.** Roughly 96–100 test positives per condition are required to estimate a sensitivity near 0.80 with a 95% confidence interval of about ±0.08; ≥150–300 images per skin-tone bin give reasonable power to detect a ~10% tone gap. These are planning estimates, not guarantees.

**Capture.** Smartphone photography at a defined minimum standard (≥12 MP main camera with autofocus, main lens only, default photo mode, originals retained), with a 3D-printed spacer fixing camera-to-skin distance, a colour-reference card and ruler in frame, and a clip-on dermoscope at lesion stations. Following DermaCon-IN, devices are deliberately *not* standardised to a single model — device variation is recorded per image and treated as clinical realism.

**Labelling.** Two dermatologists label independently and blinded to each other and to the model's prediction, with third-reader adjudication of disagreements; every image carries an explicit evidence grade (histopathology / dermoscopy-supported / clinical consensus / provisional), and no case counts toward the malignant target until its pathology report has been reconciled to its study identifier.

**Governance.** Institutional Ethics Committee approval precedes any collection; consent is obtained before photography; images carry study identifiers only, exclude faces and identifying marks, and are stored encrypted on institution-approved, India-resident infrastructure in accordance with the Digital Personal Data Protection Act 2023.

**Expected output.** A publishable dataset resource in its own right — the first from North India, and the first anywhere pairing Indian clinical photography with a biopsy-confirmed cancer arm — plus the patient-level and temporally held-out test sets needed to state, or refute, that this system works for Indian patients.

---

## 9. Conclusion

Under authentic resource-constrained, dark-skin clinical conditions, rigor and representation reuse beat scale and fine-tuning. Freezing a foundation backbone and probing an ensemble of diverse frozen representations reaches, at the point estimate, published dark-skin cancer ceilings on DDI (V–VI AUROC 0.872, 95% CI 0.808–0.929), significantly above a single frozen backbone, with no backbone training. A deployable trust stack supplies calibration, conformal coverage, and per-tone selective prediction, and an honest evaluation — including a self-caught 35% leakage correction and a majority-class-collapse fix — keeps the numbers defensible. We present this as a clinical demonstration of a known linear-probe-versus-fine-tune phenomenon, with the novelty in the frozen multi-backbone dark-skin result, the trust stack, and the honest-methodology artifacts. The system is intended as decision support, not diagnosis, and its dark-skin claim awaits external and prospective validation.

---

## 10. Data and Code Availability

All datasets are public research datasets (ISIC-2019/2024, BCN20000, Fitzpatrick17k/CleanPatrick, SCIN, DermaCon-IN, PAD-UFES-20, DDI, Mpox/MSLD); access terms follow the respective providers. DDI requires registration with its host (Stanford AIMI, **TODO-verify current access route**). Result artifacts underlying every number in this manuscript are in `eval/probe/` (`ddi_ensemble.json`, `ddi_ensemble_ci.json`, `CLEAN_25class.json`, `trust_25class.json`, `probe_25class_*_mlp.json`, `probe_9class_224_mlp.json`, `ddi_dinov2_518.json`). Primary scripts: `scripts/_cache_features.py`, `_probe_multiclass.py`, `_probe_backbone.py`, `_ddi_ensemble.py`, `_ddi_ensemble_ci.py`, `_clean_eval_25class.py`, `_trust_refresh_25class.py`. Backbone: `vit_base_patch14_dinov2.lvd142m` (frozen). A pinned `requirements.txt` and a model card for the deployed 25-class probe are to be restored/authored as part of the reproducibility package (**TODO-verify** exact versions of `torch`, `timm`, `scikit-learn`). No GPU is required to reproduce the frozen-probe and ensemble results from cached features.

---

## 11. Ethics

This work uses only de-identified public research datasets and performs no interaction with human subjects. The intended use is clinical decision support to assist, not replace, a clinician, and the system abstains rather than issuing low-confidence decisions. We report subgroup performance honestly, including weak and underpowered subgroups, to avoid overstating fairness. The dark-skin cancer result derives from a single US-clinic source (DDI) and must not be interpreted as validated for autonomous use in any population, including the Indian population that motivates the project, absent external and prospective validation. Melanoma-referral sensitivity is preserved by design regardless of the text-triage channel. No personally identifying data is processed, stored, or transmitted by the pipeline.

---

*Word count target 4,000–5,000; see `FIGURES_TABLES.md` for figure/table specifications and `REFERENCES.md` for the reference list. Numbers flagged **TODO-verify** require an author check before submission.*
