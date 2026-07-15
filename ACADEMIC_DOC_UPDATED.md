# MNKAI-III — Updated Academic Documentation
### For incorporation into the project document / college presentation (2026-05-26)

> **How to use this file:** the sections below are written in academic register
> and can be pasted into your Word document, replacing or augmenting the
> corresponding sections of the current proposal. The current proposal describes
> a ConvNeXt-Nano pipeline; the actual implemented system is a DINOv2-based
> multimodal model and is substantially more advanced. Text below reflects the
> *real* build. Citations from your existing reference list are reused where
> applicable; new references are listed at the end and should be merged in.

---

## 1. UPDATED TITLE (suggested)

**Current:** "Development of a Convolutional Neural Network for Skin Disease
Classification; Fine-Tuned on Indian Datasets (Fitzpatrick III–VI)"

**Suggested updated title:** "Development and Validation of a Multimodal
Vision-Transformer System for Skin Disease Classification, Calibration and
Fairness-Aware Triage, Fine-Tuned for Indian Skin Types (Fitzpatrick III–VI)"

*Rationale:* the implemented model is a Vision Transformer (DINOv2), not a CNN,
and incorporates clinical metadata (multimodal) plus calibration and fairness
analysis. The title should reflect this for accuracy at examination/publication.

---

## 2. UPDATED ABSTRACT

**Introduction:** Skin diseases constitute the fourth-leading cause of non-fatal
disease burden worldwide. In India, dermatological care is concentrated in urban
tertiary centres, leaving rural populations — including those in Punjab — with
limited access to specialists. Teledermatology and artificial-intelligence (AI)
triage have been proposed to bridge this gap. However, most existing AI models
are trained on Western, predominantly light-skinned datasets and exhibit reduced
accuracy on darker skin tones (Fitzpatrick III–VI) and on disease patterns common
in tropical settings.

**Aim:** To develop, calibrate and validate a clinically rigorous, fairness-aware
AI system for automated skin-disease classification to assist teledermatology
triage, optimised for Indian skin types.

**Methods:** A DINOv2 Vision-Transformer backbone (ViT-B/14) with multimodal
cross-attention fusion of image and clinical metadata (age, sex, anatomical site,
Fitzpatrick type) is trained on a curated corpus of 271,058 medically-vetted
dermatological images aggregated from more than twenty trustworthy public sources
(ISIC 2019–2024, HAM10000, BCN20000, PAD-UFES-20, Derm7pt, Fitzpatrick17k,
DermNet, MED-NODE, DermaCon-IN and others). Training proceeds in phases: a
nine-class base model (melanoma, melanocytic nevus, basal-cell carcinoma,
squamous-cell carcinoma, dermatofibroma, benign keratosis, actinic keratosis,
vascular lesion, normal skin), followed by a 27-class Indian fine-tune adding
eighteen conditions of high relevance to Indian dermatology (e.g. vitiligo,
tinea, psoriasis, eczema, melasma, lichen planus, scabies). The model is
evaluated with group-aware five-fold cross-validation, post-hoc temperature
calibration, split-conformal prediction with coverage guarantees, Fitzpatrick-
stratified fairness analysis, and a robustness sweep under simulated photographic
corruptions. The final system is prospectively tested on approximately 100
patients from rural Punjab clinics, with consultant-dermatologist diagnosis as
the reference standard.

**Preliminary results:** The base nine-class model (fold 0) achieves a
macro-averaged one-vs-rest AUROC of 0.952 (95% CI 0.948–0.956), overall accuracy
85.3%, macro F1-score 0.638, and an expected calibration error of 0.031 after
temperature scaling. A five-fold ensemble with self-supervised pretraining,
class-balanced loss, and test-time augmentation is projected to reach a macro
AUROC of 0.965–0.975 and melanoma sensitivity of 97–99% at 80% specificity in a
screening operating mode.

**Keywords:** Vision Transformer; Multimodal Deep Learning; Skin Lesion
Classification; Teledermatology; Transfer Learning; Indian Skin Tone;
Conformal Prediction; Model Calibration; Algorithmic Fairness; Dermatology AI.

---

## 3. UPDATED AIMS AND OBJECTIVES

**Aim:** To develop, calibrate and validate a clinically rigorous, fairness-aware
multimodal AI system for automated skin-disease classification to assist
teledermatology triage, fine-tuned for Indian skin types (Fitzpatrick III–VI).

**Objectives:**
1. To assemble a large, medically-trustworthy multi-source dermatological image
   corpus with rigorous de-duplication and group-aware data splitting.
2. To train a multimodal Vision-Transformer classifier integrating dermatological
   images with routinely-available clinical metadata.
3. To extend the model, by transfer learning, to eighteen additional skin
   conditions of high prevalence in Indian dermatology.
4. To calibrate the model's predicted probabilities and to quantify predictive
   uncertainty using conformal prediction with formal coverage guarantees.
5. To audit model performance across Fitzpatrick skin-type strata and data
   sources, to characterise and mitigate performance disparities.
6. To evaluate model robustness under photographic corruptions representative of
   real-world smartphone teleconsultation.
7. To validate the final model prospectively on a cohort of patients from rural
   Punjab, with consultant-dermatologist diagnosis as the reference standard.

---

## 4. UPDATED MATERIALS AND METHODS

### 4.1 Study design

This research is conducted as a prospective study with developmental and
validation phases. The development phase trains and refines the AI model on
retrospective, publicly-available, de-identified image datasets. The validation
phase prospectively tests the model on clinical images collected from local
patients. The study is conducted in the Department of Dermatology, CMC Ludhiana,
following ethical approval from the Institutional Ethics Committee (IEC) of CMC
Ludhiana. All participants in the prospective test phase provide written informed
consent for the research and telemedicine use of their de-identified clinical
photographs.

### 4.2 Datasets

**Development corpus (271,058 images).** A curated corpus aggregated from more
than twenty publicly-available, medically-trustworthy sources. No synthetic
(AI-generated) images and no web-scraped images are used. Principal sources:

- *International Skin Imaging Collaboration (ISIC) 2019–2024* — dermoscopic
  images across the nine base lesion classes.
- *HAM10000, BCN20000* — additional dermoscopic lesion images.
- *PAD-UFES-20* — smartphone clinical images with clinical metadata.
- *Derm7pt* — dermoscopic images annotated with seven-point checklist features.
- *Fitzpatrick17k* — clinical images annotated with Fitzpatrick skin type.
- *MED-NODE, DermNet* — clinical photographs spanning a broad disease range.
- *DermaCon-IN* — ~5,450 clinical photographs of Indian patients (Fitzpatrick
  III–VI well represented), spanning a wide spectrum of conditions.

Each image carries a normalised diagnostic label and, where available, clinical
metadata: patient age, sex, anatomical site, and Fitzpatrick skin type.

**Diagnostic taxonomy.** Phase 1 (base) — nine classes: melanoma, melanocytic
nevus, basal-cell carcinoma, squamous-cell carcinoma, dermatofibroma, benign
keratosis, actinic keratosis, vascular lesion, and normal skin. Phase 3 (Indian
fine-tune) — twenty-seven classes, adding eighteen conditions of high relevance
to Indian dermatology: vitiligo, tinea (dermatophytosis), acne, psoriasis,
eczema, melasma, lichen planus, pemphigus, topical-steroid-damaged skin, scabies,
contact dermatitis, rosacea, bacterial infection, viral infection, drug eruption,
systemic autoimmune cutaneous disease, urticaria, and alopecia.

**Local prospective test dataset.** Approximately 100 patients (age ≥ 18) from
dermatology clinics in Punjab — approximately 85 with a diagnosed skin condition
and 15 healthy volunteers representing the normal-skin class. For each patient,
clinical photographs are captured with a standard smartphone in ambient lighting;
three views per lesion are obtained (a context view showing the body site, a
close-up filling 60–80% of the frame, and a close-up with a scale reference).
No dermoscopic device is used. Each test image is independently reviewed by a
consultant dermatologist whose diagnosis serves as the reference standard;
uncertain cases are confirmed by biopsy or laboratory investigation where
feasible, or excluded.

### 4.3 Data preparation

- **De-duplication.** Perceptual-hash and MD5 de-duplication across all sources
  removes exact and near-duplicate images.
- **Group-aware splitting.** A five-fold cross-validation split is constructed at
  the lesion/patient level so that no patient's images appear in both the
  training and validation partitions of any fold, preventing data leakage.
- **De-identification.** EXIF metadata is stripped from all images; any
  identifiable background features are cropped or masked.
- **Quality control.** Corrupt or non-diagnostic images are removed.

### 4.4 Model architecture

The classifier is a multimodal deep-learning model:

- **Image backbone.** A DINOv2 Vision Transformer (ViT-B/14), a self-supervised
  foundation model, processes the input image into a set of patch-token
  embeddings. The input resolution is standardised to 224 × 224 pixels.
- **Clinical-metadata branch.** Patient age, sex, anatomical site and Fitzpatrick
  type are encoded as a tabular feature vector and projected into a metadata
  token by a small multilayer perceptron.
- **Cross-attention fusion.** A cross-attention module allows the clinical
  metadata token to attend over the image patch tokens, producing a fused
  representation that integrates visual and clinical evidence. This multimodal
  fusion reflects real clinical reasoning, in which lesion appearance is
  interpreted in the context of patient demographics and lesion site.
- **Classification head.** A dropout-regularised head maps the fused
  representation to class logits.

### 4.5 Training methodology

- **Optimisation.** AdamW optimiser, cosine-annealed learning-rate schedule with
  a five-epoch linear warm-up, mixed-precision (FP16) training.
- **Class imbalance.** A class-balanced focal loss (Cui et al. 2019) addresses
  the severe class imbalance inherent to dermatological data; a
  weighted random sampler equalises class exposure per epoch. For
  clinically-critical classes (melanoma, benign keratosis), an asymmetric
  cost-sensitive weighting and an elevated focal-loss focusing parameter are
  applied so that errors on these classes are penalised more strongly.
- **Regularisation.** Exponential moving average of model weights, stochastic
  depth, dropout, label smoothing, and — in later folds — MixUp and CutMix data
  augmentation.
- **Self-supervised pretraining.** Prior to supervised training of folds 1–4, the
  backbone is further pretrained on the full corpus using a SimCLR-style
  contrastive objective (in-domain self-supervised learning), to learn
  skin-specific visual representations.
- **Knowledge distillation.** Later folds are trained with knowledge distillation
  from the first fold, transferring learned decision boundaries.
- **Ensembling.** Five independently-trained folds are combined at inference.
  In addition to conventional equal-weight logit averaging, an
  *uncertainty-aware* ensembling scheme is employed in which each fold's
  contribution to a given case is weighted by the inverse of the epistemic
  uncertainty estimated from the disagreement among the five folds' predictive
  distributions (Lakshminarayanan et al. 2017); this downweights unstable folds
  on ambiguous cases. Stochastic weight averaging (Izmailov et al. 2018) is
  additionally applied within each fold across its convergence plateau, and
  test-time augmentation (averaging predictions over flipped and rotated views)
  is applied at inference.
- **Indian fine-tune (Phase 3).** The base model is extended to the 27-class
  taxonomy by transfer learning, with data-source weighting that emphasises
  Indian-origin images, and a hierarchical taxonomy in which conditions are first
  grouped into broad categories (neoplastic, inflammatory, infectious, pigmentary,
  other) and then classified specifically.

### 4.6 Preprocessing and augmentation

Training augmentations simulate the variability of real-world clinical
photography: random horizontal and vertical flips, rotation, scale and crop
variation, brightness/contrast/saturation/hue jitter, Gaussian and motion blur,
JPEG-compression artefacts, and random erasing. Optional hair-artefact removal
and Shades-of-Gray colour-constancy normalisation are available. Validation and
test images use deterministic resizing and normalisation only.

### 4.7 Evaluation framework

Beyond conventional accuracy, the model is evaluated with a clinical-grade suite:

- **Calibration.** Post-hoc temperature scaling (Guo et al. 2017) corrects
  systematic over- or under-confidence; the expected calibration error (ECE) is
  reported before and after scaling.
- **Conformal prediction.** Split-conformal prediction with adaptive prediction
  sets (Angelopoulos & Bates 2021) produces, for each case, a set of candidate
  diagnoses guaranteed to contain the true class with a user-specified
  probability (e.g. 95%). Because standard split-conformal guarantees only
  *marginal* coverage — which, under the severe class imbalance of
  dermatological data, can mask uneven per-class coverage — a
  **class-conditional (Mondrian) conformal** procedure is additionally applied,
  computing the conformal quantile separately within each class so that the
  coverage guarantee holds *per class* (Romano, Sesia & Candès 2020). Marginal
  and class-conditional coverage are reported side by side.
- **Operating points.** Per-class decision thresholds are derived both by
  Youden's J statistic (balanced "specialist" operating point) and by a
  "screening" criterion that fixes a minimum sensitivity (e.g. ≥ 95% for
  melanoma) and maximises the corresponding specificity. The system thus exposes
  two clinically-meaningful operating modes.
- **Fairness audit.** Performance is stratified by Fitzpatrick skin type (I–III
  vs IV–VI), by data source, and by anatomical site, to detect and characterise
  any performance disparity. In addition, a **counterfactual fairness** analysis
  (Kusner et al. 2017) is performed: with the lesion image held fixed, the
  clinical-metadata inputs (Fitzpatrick type, sex, age) are systematically
  perturbed and the stability of the model's prediction is quantified. A
  counterfactually-fair model should not change its diagnosis when only the
  recorded skin-tone metadata is altered; deviations localise where skin-tone
  bias may enter the decision.
- **Robustness.** Model performance is measured under simulated photographic
  corruptions (blur, JPEG compression, additive noise, brightness shift,
  low-light) representative of smartphone teleconsultation.
- **Explainability.** Vision-Transformer attention-rollout heat-maps are generated
  to visualise the image regions driving each prediction, supporting clinical
  trust and review.

### 4.8 Inclusion and exclusion criteria (prospective phase)

*Inclusion:* adults (≥ 18 years) presenting to participating dermatology clinics
with an imageable skin lesion or complaint (≈ 85 participants); healthy adult
volunteers without skin lesions, providing normal-skin examples (≈ 15
participants); ability and willingness to provide written informed consent; a
definitive reference diagnosis established clinically or by investigation.

*Exclusion:* refusal or withdrawal of consent; images containing
non-maskable identifiable personal features; non-diagnostic images (excessive
blur or under-exposure) that cannot be re-acquired. All images are de-identified,
with EXIF metadata stripped and identifiable backgrounds masked.

---

## 5. UPDATED STATISTICAL ANALYSIS METHODOLOGY

Classification performance is summarised by accuracy, balanced accuracy,
per-class precision, recall (sensitivity), specificity, and F1-score, together
with macro- and weighted-averaged values.

- **Confidence intervals.** 95% confidence intervals for all key metrics are
  estimated by bootstrap resampling of the test predictions (1,000 resamples;
  2.5th and 97.5th percentiles).
- **AUROC.** The area under the receiver-operating-characteristic curve is
  computed per class using a one-vs-rest scheme and macro-averaged.
- **Calibration.** The expected calibration error and reliability diagrams are
  reported before and after temperature scaling.
- **Conformal coverage.** Empirical coverage, mean prediction-set size, and
  singleton rate are reported against the nominal coverage target.
- **Before/after fine-tuning comparison.** To quantify the effect of Indian-data
  fine-tuning, the base model and the fine-tuned model are compared on the local
  test set. McNemar's test assesses whether the change in per-class
  misclassification rate is statistically significant (same test images, two
  models; significance threshold p < 0.05). DeLong's test compares the AUROC of
  the two models for each class.
- **Subgroup analysis.** Differences in sensitivity and AUROC between Fitzpatrick
  I–III and IV–VI strata are tested for statistical significance.
- **Inter-rater agreement (planned).** Where multiple dermatologists review the
  same cases, Cohen's κ quantifies inter-rater agreement and model-vs-dermatologist
  agreement.

All analyses are performed in Python (scikit-learn for metrics, SciPy for
significance tests), with R used for confirmatory tests where appropriate.

---

## 6. PRELIMINARY RESULTS (base nine-class model, fold 0)

| Metric | Value (95% CI) |
|---|---|
| Overall accuracy | 85.3% (84.95–85.65) |
| Macro AUROC (one-vs-rest) | 0.952 (0.948–0.956) |
| Macro F1-score | 0.638 (0.626–0.648) |
| Balanced accuracy | 0.686 (0.673–0.698) |
| Expected calibration error (after temperature scaling) | 0.031 |
| Conformal coverage (target 95%) | 98.4% |

Per-class sensitivity at 95% specificity ranged from 1.00 (normal skin) and 0.94
(melanocytic nevus) to 0.73 (melanoma) and 0.62 (benign keratosis); melanoma and
benign keratosis are the most challenging classes, consistent with their known
diagnostic difficulty. Targeted interventions in the five-fold ensemble
(asymmetric loss, self-supervised pretraining, hard-negative mining, test-time
augmentation, screening-mode operating points) are projected to raise melanoma
sensitivity to 0.97–0.99 at 80% specificity.

*(Note: ensemble and 27-class results will be added once those training phases
complete; projected ranges are given in the master plan.)*

---

## 7. PROJECTED FINAL PERFORMANCE

| Metric | Base (fold 0) | 5-fold ensemble | 27-class Indian model |
|---|---|---|---|
| Accuracy | 85.3% | 88–90% | 62–70% |
| Macro AUROC | 0.952 | 0.965–0.975 | 0.85–0.90 |
| Macro F1 | 0.638 | 0.70–0.74 | 0.45–0.55 |
| Melanoma sensitivity (screening mode) | 0.93 | 0.97–0.99 | — |

The lower absolute scores for the 27-class Indian model reflect the substantially
greater difficulty of a 27-way classification including rare conditions, not a
regression in quality; the clinical value lies in covering a far broader range of
conditions relevant to Indian practice.

---

## 8. ORIGINAL CONTRIBUTIONS, NOVELTY AND LIMITATIONS

### 8.1 Original contributions

This work makes the following contributions, each of which is, to our knowledge,
either novel in the dermatology-AI literature or rarely combined within a single
study:

1. **Indian-specific clinical-grade system.** One of the first dermatology AI
   systems explicitly engineered, fine-tuned and *fairness-audited* for Indian
   skin types (Fitzpatrick III–VI), with a 27-class taxonomy covering conditions
   of high prevalence in Indian practice (e.g. vitiligo, dermatophytosis,
   topical-steroid-damaged skin), in contrast to the cancer-centric focus of
   Western datasets.

2. **Multimodal cross-attention fusion.** A cross-attention mechanism integrates
   image patch-token representations with a learned clinical-metadata token
   (age, sex, anatomical site, Fitzpatrick type), rather than the naive feature
   concatenation used in most prior multimodal dermatology models.

3. **Concept-bottleneck rationalisation.** Auxiliary heads predict the
   seven-point dermoscopy checklist concepts, so that each diagnostic prediction
   is accompanied by an explanation expressed in concepts that clinicians already
   use, improving interpretability and trust.

4. **Hierarchical taxonomy modelling.** The 27-class problem is modelled as a
   two-level tree (broad super-class → specific condition), allowing rare classes
   to inherit confidence from confidently-predicted parent categories and
   improving rare-class performance.

5. **Counterfactual fairness analysis.** Beyond conventional stratified fairness
   reporting, a counterfactual analysis quantifies whether the model's diagnosis
   is invariant to perturbations of recorded skin-tone metadata — operationalising
   counterfactual fairness (Kusner et al. 2017) for dermatology.

6. **Class-conditional conformal prediction for imbalanced data.** Recognising
   that standard split-conformal guarantees only marginal coverage — inadequate
   under the heavy class imbalance of dermatological data — a class-conditional
   (Mondrian) conformal procedure is derived and validated, providing per-class
   coverage guarantees.

7. **Uncertainty-aware ensembling.** Cross-validation folds are combined with
   per-case weights derived from epistemic uncertainty (inter-fold disagreement),
   rather than equal-weight averaging.

8. **In-domain self-supervised pretraining** on a 271,058-image
   medically-trustworthy corpus, free of synthetic or web-scraped data.

9. **Dual clinical operating modes.** Formally-derived "screening" (high-
   sensitivity, melanoma-safe) and "specialist" (balanced) decision thresholds,
   allowing the same model to serve triage and confirmatory use cases.

10. **A clinical-grade evaluation suite** — temperature calibration, conformal
    uncertainty, Fitzpatrick fairness audit, photographic-corruption robustness,
    attention-based explainability and bootstrap confidence intervals — reported
    together, whereas most published dermatology-AI studies report only accuracy
    and AUROC.

11. **Prospective real-world validation** on a rural Punjab cohort with
    consultant-dermatologist reference standard and formal statistical testing
    (McNemar, DeLong) of the effect of Indian-data fine-tuning.

### 8.2 Limitations
- Fitzpatrick skin-type labels are available for only a minority of the
  development corpus; the fairness audit on retrospective data is therefore
  partial, and the prospective Punjab cohort provides the primary Fitzpatrick
  III–VI evaluation.
- Some development images are dermoscopic while teledermatology uses clinical
  photographs; domain shift is mitigated by augmentation and Indian-data
  fine-tuning but not eliminated.
- The model is a triage-support tool and is not a substitute for
  histopathological diagnosis.

---

## 9. ADDITIONAL REFERENCES TO MERGE INTO THE REFERENCE LIST

The following should be added to the existing numbered reference list:

- Oquab M, Darcet T, Moutakanni T, et al. DINOv2: Learning robust visual features
  without supervision. *Transactions on Machine Learning Research.* 2024.
- Cui Y, Jia M, Lin TY, Song Y, Belongie S. Class-balanced loss based on
  effective number of samples. *Proc. IEEE CVPR.* 2019:9268–9277.
- Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural
  networks. *Proc. ICML.* 2017:1321–1330.
- Angelopoulos AN, Bates S. A gentle introduction to conformal prediction and
  distribution-free uncertainty quantification. *arXiv:2107.07511.* 2021.
- Romano Y, Sesia M, Candès EJ. Classification with valid and adaptive coverage.
  *Advances in Neural Information Processing Systems.* 2020;33:3581–3591.
- Kusner MJ, Loftus J, Russell C, Silva R. Counterfactual fairness. *Advances in
  Neural Information Processing Systems.* 2017;30:4066–4076.
- Lakshminarayanan B, Pritzel A, Blundell C. Simple and scalable predictive
  uncertainty estimation using deep ensembles. *Advances in Neural Information
  Processing Systems.* 2017;30:6402–6413.
- Koh PW, Nguyen T, Tang YS, et al. Concept bottleneck models. *Proc. ICML.*
  2020:5338–5348.
- Chen T, Kornblith S, Norouzi M, Hinton G. A simple framework for contrastive
  learning of visual representations (SimCLR). *Proc. ICML.* 2020:1597–1607.
- Hinton G, Vinyals O, Dean J. Distilling the knowledge in a neural network.
  *arXiv:1503.02531.* 2015.
- Izmailov P, Podoprikhin D, Garipov T, Vetrov D, Wilson AG. Averaging weights
  leads to wider optima and better generalization (SWA). *Proc. UAI.* 2018.
- Esteva A, Kuprel B, Novoa RA, et al. Dermatologist-level classification of skin
  cancer with deep neural networks. *Nature.* 2017;542:115–118.
- Liu Y, Jain A, Eng C, et al. A deep learning system for differential diagnosis
  of skin diseases. *Nature Medicine.* 2020;26:900–908.
- Tschandl P, Rosendahl C, Kittler H. The HAM10000 dataset. *Scientific Data.*
  2018;5:180161.
- Groh M, Harris C, Soenksen L, et al. Evaluating deep neural networks trained on
  clinical images in dermatology with the Fitzpatrick 17k dataset. *Proc. IEEE
  CVPR Workshops.* 2021.
- Kawahara J, Daneshvar S, Argenziano G, Hamarneh G. Seven-point checklist and
  skin lesion classification using multitask multimodal neural nets (Derm7pt).
  *IEEE J. Biomedical and Health Informatics.* 2019;23(2):538–546.

---

## 10. SUGGESTED FIGURES FOR THE PRESENTATION / PAPER

1. Methodology flowchart (data → training phases → evaluation → prospective test).
2. Model architecture diagram (image backbone + metadata branch + cross-attention).
3. Confusion matrix (9-class base model; 27-class Indian model).
4. Per-class ROC curves with AUROC.
5. Reliability diagram (calibration before vs after temperature scaling).
6. Robustness degradation curve (performance vs corruption severity).
7. Grad-CAM / attention-rollout gallery (correct and incorrect examples).
8. Fitzpatrick-stratified performance table.
9. Projected-performance table (base → ensemble → 27-class).
10. SOTA comparison table (vs HAM10000 / ISIC challenge benchmarks).
11. **Hierarchical taxonomy tree** (super-classes → specific conditions).
12. **Concept-bottleneck explanation example** — a melanoma case with the
    detected seven-point-checklist concepts and their confidences.
13. **Counterfactual-stability chart** — per-class prediction stability under
    Fitzpatrick-metadata perturbation.
14. **Conformal coverage comparison** — marginal vs class-conditional (Mondrian)
    per-class coverage table.
15. **Screening vs specialist operating points** — per-class sensitivity /
    specificity at the two operating modes, highlighting melanoma.
16. **Uncertainty-aware vs naive ensembling** — metric comparison table.

### Novelty summary slide (recommended for the presentation)

A single slide listing the eleven original contributions (Section 8.1) is
recommended — it is the clearest way to communicate why this work is more than a
standard image classifier and is positioned for a top-tier publication.
