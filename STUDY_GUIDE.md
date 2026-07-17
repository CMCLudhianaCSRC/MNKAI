# MNKAI-III — Complete Study Guide (start to end)

*Paste this whole file into a new chat to have full context, then ask about any section.*
This explains **every technique** used and **every important file's logic**, from the ML concepts down to the code.

---

# PART A — The tech topics (study checklist)

Group these; each line = a concept + where we use it. Learn them in this order.

### 1. Deep-learning foundations
- **CNN vs Vision Transformer (ViT)** — our backbone is a ViT (image split into patches → transformer). File: `models/backbones.py`.
- **Transfer learning** — reuse a model pretrained on huge data, adapt to our task. The whole project.
- **Self-supervised learning (SSL)** — learn features from *unlabeled* images (no labels). **DINOv2** (our backbone) is DINO-style SSL; **MAE/CAE** are masked-image-modeling SSL (planned). Concept behind "foundation models."
- **Fine-tuning vs linear/frozen probing** — *the central lesson.* Fine-tuning = update the whole network on your data (can *degrade* features when data is scarce). Frozen-probe = keep the backbone fixed, train only a small head. **Frozen-probe won here.**
- **Embeddings / feature vectors** — the backbone turns an image into a 768-d vector; classifiers/probes work on those.

### 2. Multimodal fusion
- **Cross-attention fusion** — combine image patch tokens with a clinical-metadata token (age/sex/site/Fitzpatrick). File: `models/backbones.py` (`ImageTabularNet`, `CrossAttentionFusion`).
- **Tabular encoding** — turning metadata into a feature vector (`TabularSpec` in `data/datasets.py`).

### 3. Class imbalance (long-tail) — the "collapse" story
- **Focal loss** — down-weights easy examples, focuses on hard ones. `losses/focal.py`.
- **Class-balanced focal (Cui 2019)** — reweight by "effective number of samples." `effective_number_weights`.
- **LDAM + DRW** — margin loss for rare classes + deferred reweighting. `losses/focal.py`.
- **Logit adjustment / balanced-softmax** — add a per-class prior term to logits. `LogitAdjustedCE`.
- **Weighted samplers** (`sampler.py`, `power` param) — oversample rare classes. **Stacking a sampler on cb_focal = double-balancing = the majority-class collapse we diagnosed and fixed.**
- **Decoupling / cRT (Kang 2020)** — train features on natural data, then retrain only the classifier with balancing.
- **Per-class threshold / logit bias** — post-hoc shift per class to maximize macro-F1. `scripts/_optimize_thresholds.py`.

### 4. Training techniques
- **MixUp / CutMix** — blend images/labels for regularization. `training/train.py`.
- **EMA (exponential moving average)** of weights — a smoothed "shadow" model, usually better. `training/ema.py`.
- **Label smoothing** — soften one-hot targets. `losses/focal.py`.
- **Cosine LR schedule + warmup + freeze-then-unfreeze backbone.** `training/schedulers.py`, `train.py`.
- **Gradient checkpointing / mixed precision (AMP)** — fit big models in 6 GB. `train.py`.

### 5. Trustworthy ML (our differentiator)
- **Calibration + temperature scaling + ECE** — make predicted probabilities honest. `evaluation/calibration.py`, `_trust_refresh_25class.py`.
- **Conformal prediction (APS / RAPS / clustered)** — output a *set* of classes with a coverage guarantee. `calibration/conformal.py`, `_conformal_fix.py`.
- **Selective prediction / abstention** — refer low-confidence cases to a human. `evaluation/abstention.py`.
- **OOD detection (Mahalanobis on features)** — reject non-skin / unfamiliar input. `_ood_gate.py`.
- **Fairness (per-Fitzpatrick subgroups)** — report accuracy per skin tone. `evaluation/subgroups.py`.

### 6. Honest evaluation
- **Group-aware cross-validation** — split by patient/lesion so the same lesion isn't in train+test. `data/splits.py`.
- **External (held-out) test** — a single-shot honest generalization number.
- **Bootstrap confidence intervals** — resample to get CIs on every metric. `evaluation/bootstrap.py`, `_bootstrap_ci.py`.
- **Near-duplicate leakage detection** — we found 33% train↔test duplicates via *feature nearest-neighbor cosine*; a reusable audit. (In `_probe_multiclass`-adjacent scripts.)
- **Test-time augmentation (TTA)** — average predictions over crops/flips. `evaluation/tta.py`.

### 7. The dark-skin cancer study
- **DDI benchmark**, **CleanPatrick** (label cleaning), **domain shift** (dermoscopy→clinical, light→dark), the **frozen-probe fix**, the **honest ceiling** (~0.78, data-limited).

### 8. Foundation-model ensembling (the leaderboard push)
- **Frozen multi-backbone ensemble** — DINOv2 + registers + EVA-02 + ConvNeXt-V2 + SigLIP + DINOv2-Large; concat features or average scores. `scripts/_ddi_ensemble.py`.
- **Derm foundation models** — PanDerm, DermINO, Google Derm-Foundation.

### 9. The LLM layer
- **Vision-language model (VLM)** — a model that sees images + text. We use **Gemini** (vision) via API.
- **Confidence-gated interview** — only ask questions when the classifier is unsure.
- **Photo-grounded question generation** — the model looks at the lesion to ask relevant history questions.
- **Bayesian fusion** — combine image prior × history likelihood-ratios. `demo_app/llm.py`.
- **Distillation / LoRA / adapters** — how a *trained* compact VLM (v2) would work. `LLM_PLAN.md`.

---

# PART B — Codebase map

```
MNKAIII/
├── src/mnkai/                 # the library (installable package)
│   ├── models/
│   │   ├── backbones.py       # ViT/DINOv2 backbone + cross-attn multimodal fusion + build_model
│   │   └── segmentation.py    # (optional) lesion segmentation head
│   ├── data/
│   │   ├── manifest.py        # LabelNormalizer + normalize_manifest (clean/standardize the CSV)
│   │   ├── datasets.py        # SkinDataset (loads image+label+tabular), TabularSpec, _resolve, zero-fallback
│   │   ├── splits.py          # group-aware K-fold + fold_train_valid_indices (no leakage)
│   │   ├── transforms.py      # AugConfig, build_train/build_valid, shades-of-gray color constancy, hair removal
│   │   ├── qc.py, unify.py     # quality control + multi-source unification
│   │   └── download/          # ISIC / DermaCon-IN / Fitzpatrick17k downloaders
│   ├── losses/focal.py        # LabelSmoothingCE, FocalLoss, ClassBalancedFocal, LDAM, LogitAdjustedCE, build_loss
│   ├── training/
│   │   ├── train.py           # THE training loop (fit_one_fold): loads data, builds model+loss, trains, evals
│   │   ├── sampler.py         # class-balanced weighted sampler (the `power` knob = collapse cause)
│   │   ├── ema.py             # exponential moving average of weights
│   │   └── schedulers.py      # cosine LR + warmup
│   ├── calibration/conformal.py   # APS/RAPS conformal prediction sets
│   ├── evaluation/            # metrics, bootstrap CIs, calibration/ECE, abstention, tta, subgroups, threshold, corruption
│   └── utils/                 # config.py (YAML + _base_ inheritance), logging, seed
├── scripts/                  # 77 runnable scripts (train, eval, and the frozen-probe pipeline)
├── demo_app/                 # the deployable app + LLM + clinical pipeline
├── configs/                  # YAML experiment configs (base.yaml + per-experiment)
├── models/clinical_25class.pkl   # the deployable frozen-probe head (scaler+MLP+temp+bias+conformal)
└── eval/, logs/, docs/       # results + reports
```

---

# PART C — How each core file works (the logic)

### models/backbones.py
- `ModelConfig` (dataclass): name, num_classes, dropout, `tabular_dim`, `fusion` ('concat'|'cross_attn'), img_size.
- `build_model(cfg)`: if `tabular_dim>0` → `ImageTabularNet` (multimodal); else → `build_image_only` (pure image head).
- `ImageTabularNet`: runs the ViT backbone (`forward_features` → patch tokens), a small MLP on the tabular vector → a "clinical token", then **CrossAttentionFusion** (the clinical token attends over the image patches) → concat → classifier head. This is what the fine-tuned Phase-1/Phase-3 models used.
- Key idea: the backbone can be **frozen** (features only) or **fine-tuned**. The frozen-probe method uses `forward_features` and ignores this whole head — it puts an sklearn head on top instead.

### data/manifest.py
- `LabelNormalizer`: maps raw label strings (aliases, typos) → the canonical class set; drops unknowns.
- `normalize_manifest(df, normalizer)`: cleans the manifest CSV (lowercase labels, drop rows with unknown labels, keep `filepath`). Used everywhere before training/eval.

### data/datasets.py
- `SkinDataset(df, classes, transform, tabular)`: PyTorch dataset. `__getitem__` → `_resolve(filepath)` → `Image.open` → transform → tensor; returns image + label index + optional tabular vector. **Has a zero-image fallback** (returns zeros if a file is missing — this is why a disconnected D: drive silently produced garbage until we added the path-exists check).
- `TabularSpec`: encodes age/sex/site/Fitzpatrick into a fixed-width (21-d) vector; `dim` gives the width.

### data/splits.py
- Group-aware K-fold: assigns whole `group_id`s (patient/lesion) to folds so no lesion appears in both train and val. `fold_train_valid_indices(manifest, fold)` → the train/valid row indices. **This is what prevents most leakage** (but not the cross-source atlas duplicates we later caught by feature-NN).

### data/transforms.py
- `AugConfig` + `build_train`/`build_valid`: albumentations pipelines. Includes **shades-of-gray color constancy** (normalize illumination — helps clinical photos) and hair removal. `build_valid` is the deterministic eval transform used for feature caching.

### losses/focal.py  (the imbalance toolkit)
- `effective_number_weights(counts, beta)`: Cui-2019 class weights.
- `ClassBalancedFocal`, `FocalLoss`, `LDAMLoss`, `LogitAdjustedCE`, `LabelSmoothingCE`.
- `build_loss(name, ...)`: the factory the trainer calls ('cb_focal', 'logit_adjusted', 'ce', ...). Unknown name → raises (so a typo can't silently fall back to CE, which was the original bug).

### training/sampler.py  (the collapse cause)
- `class_balanced_weights(labels, C, power)`: per-sample weight = 1/count^power. `power=1.0` = full inverse-freq (aggressive), `0.5` = sqrt (gentle), `0.0` = none. **Stacking power=1.0 on cb_focal double-counts the rebalancing → majority classes annihilated → the Phase-3 collapse.** The fix: use ONE mechanism.
- `make_weighted_sampler`: adds `class_boost` (extra multiplier for cancers), `source_weights` (up/down-weight datasets).

### training/train.py  (the training loop)
- `fit_one_fold(cfg, fold)`: loads manifest+splits (merge on `filepath`), builds the model (`build_model`), loss (`build_loss`), optional sampler/EMA/MixUp; supports `freeze_backbone_epochs`, `reinit_head` (fresh head for cRT), and `init_from_backbone` with a `{n}` template (per-fold init). Trains with AMP + grad checkpointing; validates each epoch (macro-F1, AUROC); saves `best.pt`/`last.pt`.
- This produced Phase-1 (9-class), Phase-3 (25-class), Fix-C, and cRT models — all *fine-tuned* (later beaten by the frozen-probe).

### calibration/conformal.py & evaluation/
- Conformal: build prediction *sets* (APS/RAPS) with coverage guarantee.
- `evaluation/`: `metrics.py` (macro-F1/AUROC), `calibration.py` (temperature, ECE), `bootstrap.py` (CIs), `abstention.py` (selective prediction), `tta.py` (test-time aug), `subgroups.py` (per-Fitzpatrick), `threshold.py` (operating points), `corruption.py` (robustness sweep).

### utils/config.py
- `load_config(path)`: reads YAML, resolves `_base_:` inheritance (deep-merge), returns the config dict. All experiments are configs.

---

# PART D — The FINAL method pipeline (frozen-probe), file by file

This is the method that won and is deployed. Study these scripts in order:

1. **`scripts/_cache_features.py`** — load a config's images → **frozen DINOv2@518** (timm, `forward_features`) → save 768-d features + labels + split/fold to `.npz`. (Extract once, reuse.)
2. **`scripts/_probe_multiclass.py`** — load cached features → 5-fold OOF + external → **class-balanced MLP head** (sklearn) → per-class **bias** (`_optimize_thresholds.coordinate_ascent`) → macro-F1, top-k, coarse, per-tone. The 25-class result (0.573).
3. **`scripts/_probe_backbone.py`** — the same idea for **binary DDI cancer** (5-fold stratified CV, logistic head) → the 0.789 dark-skin result. Supports any backbone + `--img-size`.
4. **`scripts/_ddi_ensemble.py`** — the **multi-backbone frozen ensemble** (leaderboard push): extract 6 backbones, ensemble by concat-features and score-average.
5. **`scripts/_trust_refresh_25class.py`** — temperature calibration (ECE 0.21→0.042, T=2.2, deterministic R02) + RAPS conformal (~4-set) + **abstention curve** (refer 30% → 0.766 retained acc), per tone.
6. **`scripts/_save_deployable.py`** — retrain the final head on all data, save `models/clinical_25class.pkl` (scaler+MLP+temperature+bias+conformal qhat).
7. **`demo_app/pipeline.py`** — the runtime: photo → DINOv2@518 feature → load the pkl → calibrated differential → conformal set + margin → **abstention gate** → if uncertain, call the LLM interviewer.
8. **`demo_app/llm.py`** — Gemini interviewer: `should_interview` (gate), `propose_questions(..., image_b64=)` (**sees the photo**, asks history questions), `fuse_history` (Bayesian update: image prior × history LR).

Supporting analysis scripts: `_conformal_fix.py` (RAPS/clustered), `_coarse_eval.py` (7-superclass + binary-cancer readout), `_bootstrap_ci.py` (CIs), `_ood_gate.py` (Mahalanobis OOD), `_build_cancer_data.py` / `_ddi_cv_eval.py` (the DDI cross-validation), `_dump_predictions.py` (per-image probs for any model).

---

# PART E — The project narrative (start → end)

1. **Phase 1** — 9-class dermoscopy base (fine-tuned DINOv2 + fusion), macro-F1 0.637.
2. **Phase 3** — 25-class Indian task; hit a **majority-class collapse** (double-balancing) → diagnosed → single-mechanism fix (eczema 0.004→0.503) → external 0.484.
3. **Tier 0/1** — per-class bias, calibration, conformal, OOD, then **cRT** → 0.539 (later found leaky → 0.527 honest).
4. **Dark-skin failure** — the cancer model scored 0.53 (chance) on DDI. Deep research: 0.53 is the *unadapted floor*, not a ceiling. The fix was **frozen features + probe** → **0.789** (V–VI), at the published ceiling.
5. **Generalization** — frozen-probe **beats fine-tuning on every task** (9-class 0.660, 25-class 0.573).
6. **Rigor catch** — found ~33% **train-test duplicate leakage** in the 25-class external (feature-NN audit); dropped, re-reported. 9-class external was degenerate (99.8% nevi → use OOF CV).
7. **Trust + deploy + LLM** — calibration/conformal/abstention; deployable pipeline; **photo-grounded, confidence-gated Gemini interviewer**.
8. **Leaderboard push (now)** — frozen multi-backbone **ensemble** on DDI (+ PanDerm, TTA) toward a genuine DDI SOTA.

Key documents to read next: `README.md`, `PROJECT_REPORT.md`, `PAPER_DRAFT.md`, `logs/DDI_BREAKTHROUGH.md`, `logs/DARKSKIN_RESEARCH.md`, `LLM_PLAN.md`, `OVERHAUL_PLAN.md`.

---

# PART F — Suggested learning path (from scratch)
1. ViT + transfer learning + SSL (DINOv2) → understand the backbone.
2. Fine-tuning vs frozen-probe → the central lesson.
3. Class imbalance (focal / cb-focal / LDAM / logit-adjustment / samplers / cRT) → the collapse story.
4. Calibration + conformal + abstention → the trust layer.
5. Group-aware CV + external test + bootstrap CIs + leakage auditing → honest evaluation.
6. The DDI dark-skin study → domain shift + the frozen-probe fix + honest ceilings.
7. Ensembling + foundation models → the leaderboard push.
8. VLM + Bayesian fusion → the LLM interviewer.

Study any file by asking: *"walk me through `scripts/_probe_multiclass.py` line by line"* — this guide gives you the map to do that.
