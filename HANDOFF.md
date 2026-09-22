# MNKAI-III — Finalisation Sprint Handoff (for Claude Code)

**Owner:** Ujjwal Thakran (MBBS 2022, CMCH Ludhiana) · **Clinical co-author:** Head of Dermatology, CMCH
**Sprint length:** 15–20 working days · **Written:** 22 September 2026
**Goal:** finish, freeze and externally validate MNKAI-III v1.0; build and test a new algorithm (Fair Conformal Interviewing); prepare the 100-patient prospective study and all paper and presentation assets.

---

## 0. How to use this file (for Ujjwal)

1. Put `HANDOFF.md` (this file) and `CLAUDE.md` in the root folder of the MNKAI-III repo.
2. Open Claude Code in that folder and type:
   > Read HANDOFF.md. Do Phase 0 only, then stop and give me the report.
3. After each phase, read Claude Code's plain-language summary, answer its questions, then say "continue with Phase N".
4. Things only you (or the derm head) can do are listed in Section 3. Claude Code will stop and ask when it needs them.

**Honesty note for the whole sprint:** no single technique here is new. The contribution is the combination: the Indian and dark-skin focus, rigorous evaluation, the prospective study, and (if it works) the new algorithm in Phase 5. Every claim must be framed that way.

---

## 1. Context for Claude Code

### 1.1 Who you are working with
- **Ujjwal** is a medical student with basic coding knowledge. Explain things in plain language, not jargon. After every task, write a 5–10 line summary covering what you did, why, the result, what it means clinically, and what comes next.
- **The Head of Dermatology at CMCH** is co-author and provides the clinical reference standard. Anything clinical (label mappings, which diseases count as malignant or urgent, question wording) must be approved by them via Ujjwal.
- **An IEC-approved prospective test on 100 patients** at CMCH is planned. The model used there must be frozen before the first patient.

### 1.2 What MNKAI-III is (as last documented — VERIFY everything in Phase 0)
A skin-disease classifier for Indian and dark skin, built on the **frozen-probe** idea:
1. A frozen foundation model (DINOv2 ViT-B/14 at 518×518, via `forward_features`) turns each photo into a 768-number feature vector. Features are cached once (`scripts/_cache_features.py` → `.npz`).
2. A small class-balanced MLP head (scikit-learn) is trained on the cached features. It uses group-aware 5-fold cross-validation and per-class bias tuning (`scripts/_probe_multiclass.py`, `_optimize_thresholds.coordinate_ascent`).
3. A trust layer adds temperature scaling, RAPS conformal sets, abstention and per-Fitzpatrick reporting (`scripts/_trust_refresh_25class.py`, `evaluation/*.py`).
4. A deployable bundle, `models/clinical_25class.pkl`, holds the scaler, MLP, temperature, per-class bias and conformal cutoff `qhat`. It is built by `scripts/_save_deployable.py`.
5. At runtime:
   - `demo_app/pipeline.py` runs photo → features → calibrated differential → conformal set → abstention gate → optional LLM interview.
   - `demo_app/llm.py` handles the Gemini steps: `should_interview`, `propose_questions`, and `fuse_history` (a Bayesian update).
6. A research branch covers binary dark-skin cancer on DDI (`_probe_backbone.py`, 0.53 → 0.789; confirm which metric in Phase 0) and a six-backbone frozen ensemble (`_ddi_ensemble.py`: DINOv2, DINOv2-reg, DINOv2-L, EVA-02, ConvNeXt-V2, SigLIP).
7. Configs are YAML files with `_base_:` inheritance, loaded by `utils/config.py`.

**Previously reported numbers** (reproduce these before trusting them):
- 25-class macro-F1 ≈ 0.573.
- ECE 0.20 → 0.016 after temperature scaling.
- RAPS average set ≈ 4 classes.
- Referring 30% of cases keeps 77% accuracy on the rest.
- DDI dark-skin cancer: 0.53 → 0.789 (confirm the metric: AUROC or other).
- A near-duplicate audit found ~33% of an earlier external test set duplicated training images (fixed).
- Training pool ≈ 271,000 images.
- Some notes say 27 classes; the deployed model says 25. Phase 1 resolves this.

### 1.3 Sprint goals
1. Audit and reproduce the current system; lock the taxonomy.
2. Harden evaluation: confidence intervals, clinically serious errors, fairness by Fitzpatrick and Monk skin tone, error analysis, ablations.
3. Compare dermatology-specific foundation models as backbones for the frozen probe.
4. Build and test **Fair Conformal Interviewing (FCI)**, a new calibrated, fairness-aware question-asking algorithm (Phase 5).
5. Freeze **v1.0**, then run the sealed external validations once.
6. Prepare the prospective study kit and all paper and presentation assets.

### 1.4 Non-goals
- Do not ship or freeze the six-backbone ensemble as the clinical model. It is too heavy and stays a research result.
- Do not fine-tune large backbones in this sprint.
- Do not let an LLM produce diagnostic probabilities in the frozen study version.
- The Android/mobile student model is a stretch goal after freeze (Section 14), not part of the study.

---

## 2. Ground rules (non-negotiable)

1. **Plan, then code.** At the start of each phase, write `docs/plans/phase_N.md` listing tasks, files to touch, expected outputs and risks. Show it to Ujjwal before large changes.
2. **Inspect before assuming.** This file's description of the codebase comes from notes and may be outdated. Open files before editing, treat the repo as the truth over this document, and report discrepancies.
3. **Don't break what works.** Existing scripts must keep running. Add new modules, scripts or feature flags rather than rewriting. Never overwrite `models/clinical_25class.pkl`; new versions go in `models/v1.0/`.
4. **Raw data is read-only.** Raw downloads live in `data/raw/`. Everything derived goes to `data/processed/`, `features/` or `outputs/`.
5. **Sealed sets.** DermaCon-IN, the SCIN sealed test split, and all prospective-study data are sealed.
   - No predictions or metrics on them until Phase 7, and then exactly once, with the frozen model.
   - Preparation (download, indexing, hashing, de-duplication, label mapping) is allowed.
   - Log every access to a sealed set in `logs/sealed_access.log` (timestamp, git commit, script, purpose).
6. **Group-aware everywhere.** Split by patient or case, never by image. Assert disjointness in code.
7. **Reproducible.** Use fixed, recorded seeds. Save split files to `splits/`. Every result comes from a script plus a config; record the command, config path and git commit hash.
8. **One source of truth.** Every number in slides or the paper must come from a row in `RESULTS.md` produced by a script. Never hand-type numbers into tables.
9. **Smoke test first.** Run every new script on ≤200 images before a full run.
10. **Hardware.** The primary machine is an ASUS TUF F15 with an RTX 4050 laptop GPU (6 GB VRAM).
    - Use `torch.inference_mode()`, fp16/bf16 autocast, conservative batch sizes, and sharded, resumable feature caching.
    - Vast.ai A100 is overflow only. Ask Ujjwal before spending money.
    - TensorFlow GPU is not supported on native Windows in recent versions; if the machine is Windows, run TensorFlow models under WSL2.
11. **Privacy.** No patient-identifiable data or prospective-study images may go into git, logs, or any external API (including Gemini) unless Ujjwal confirms the IEC approval and consent cover it. Add data folders to `.gitignore`.
12. **Licences.** Record the licence and commercial-use status of every dataset and model in `docs/LICENSES.md`.
    - Several are non-commercial (e.g., PanDerm is CC-BY-NC-ND 4.0; DermaCon-IN is CC BY-NC-SA 4.0).
    - That is fine for the academic paper but not for a future commercial product. Flag this whenever a choice depends on it.
13. **No fabrication.** Never invent numbers, citations, dataset fields or model availability. If unsure, say so and check.
14. **Commit often.** Commit after each task with a clear message. Tag `v1.0-frozen` at freeze.
15. **Ask before:** deleting anything, downloads over 20 GB, spending money, changing frozen artefacts, changing the class taxonomy, or sending any data off the machine.

---

## 3. Checkpoints that need Ujjwal or the derm head

| ID | When | Who | What |
|---|---|---|---|
| H1 | End of Phase 0 | Ujjwal (+ derm head) | Decide the final taxonomy (25 or 27 classes) and confirm the superclass grouping. |
| H2 | Start of Phase 2 | Ujjwal | Get data and model access. SCIN is public. The others need action: see the notes below this table. |
| H3 | Phases 2–5 | Derm head | Review and sign off: (a) the label-mapping CSVs, (b) which classes are malignant or urgent, (c) the question-bank wording, (d) Hindi/Punjabi translations (by a native-speaking clinician). |
| H4 | Phase 6 | Ujjwal + derm head | Approve the v1.0 freeze and the model card. |
| H5 | Before the first patient | Ujjwal + derm head | CTRI registration, IEC paperwork and consent form (Claude Code only drafts these). If the model or questionnaire differs from what the IEC approved, check whether an amendment or notification is needed. |
| H6 | Any time | Ujjwal | Approve any paid compute (Vast.ai). |

**H2 access steps:**
- **DermaCon-IN:** download from Harvard Dataverse; accept terms or log in if asked.
- **Derm Foundation:** it is a gated Hugging Face model. Accept the Health AI Developer Foundations terms and create an HF access token.
- **PanDerm:** weights are linked from its README.

---

## 4. Phase 0 — Audit and reproduce (Days 1–2)

**Why:** everything later depends on knowing exactly what exists and that past numbers are real.

### Tasks
- **0.1 Environment.**
  - Record the OS (Windows / WSL / Linux) and the versions of Python, PyTorch, CUDA, timm, scikit-learn and numpy. Is the GPU visible?
  - Save this as `docs/ENVIRONMENT.md` and lock versions with `pip freeze > requirements-lock.txt` (or a conda export).
- **0.2 Repo map.** Produce a directory tree (depth 3) and list every script with a one-line purpose, its inputs and its outputs.
- **0.3 Artefact formats.** Document exactly:
  - the feature cache `.npz`: keys, shapes, dtypes, and how IDs, labels, splits and folds are stored;
  - `models/clinical_25class.pkl`: object types (scaler, MLP class and hyper-parameters, temperature, per-class bias vector, `qhat`, RAPS parameters such as `k_reg` and `lambda`, and α);
  - the step-by-step flow of `demo_app/pipeline.py`;
  - `demo_app/llm.py`: what `should_interview`, `propose_questions` and `fuse_history` take and return, and **where the likelihood ratios used by `fuse_history` come from** (a fixed table, or generated by Gemini?).
- **0.4 Data manifest.**
  - Summarise the ~271k training pool: sources; images and patients/cases per source; counts per class and per Fitzpatrick type; and how skin type was obtained for each source.
  - Check whether **SCIN, DermaCon-IN, DDI, Fitzpatrick17k and PAD-UFES-20** are in the training pool (by path or name first, then by hash if unclear). This decides which sets can serve as clean test sets.
- **0.5 Taxonomy.** Extract the exact class list used by the deployed model and by any 27-class config. List every file where 25 or 27 appears.
- **0.6 Reproduce.**
  - Re-run `_probe_multiclass.py` and `_trust_refresh_25class.py` from the existing caches and configs, and compare with Section 1.2 (tolerance ±0.01).
  - If anything doesn't match, investigate and report. Do not "fix" it silently.
- **0.7 Existing leakage-audit code.** Find the near-duplicate audit that caught the ~33% duplicates and document how it works, so Phase 2 can reuse it.
- **0.8 Create** `RESULTS.md` (template in Appendix B), a `docs/LICENSES.md` skeleton, and `docs/AUDIT.md` (everything above).

**Acceptance:** `docs/AUDIT.md` answers every question above, and the reproduction status is explicit (match, or mismatch plus explanation).

**CHECKPOINT → stop and report to Ujjwal (H1).**

---

## 5. Phase 1 — Lock the taxonomy and the results ledger (Days 2–3)

- **1.1** After H1, create `configs/taxonomy/mnkai_v1.yaml` with one entry per class:
  - `id`, `name`, `superclass`, `synonyms`;
  - `is_malignant` and `is_urgent` (filled in after H3).
  - Reuse the existing 7-superclass grouping from `_coarse_eval.py` if it exists.
- **1.2** Make scripts read the class list from this file, or add an assertion that each hard-coded list matches it.
- **1.3** Regenerate the current model's headline metrics under the locked taxonomy and log them in `RESULTS.md` as the **v0.9 baseline**.
- **1.4** Search docs, slides and configs for stale numbers (25 vs 27, old metrics) and list them in `docs/STALE_NUMBERS.md` so Ujjwal can fix his materials.

---

## 6. Phase 2 — Datasets: download, index, de-duplicate, label-map (Days 2–5)

### 6.1 Datasets

| Dataset | Role in this sprint | Size | Where to get it | Licence (verify) |
|---|---|---|---|---|
| **DermaCon-IN** (Madarkar et al., IIT Hyderabad; NeurIPS 2025 Datasets & Benchmarks) | **Sealed external test** on Indian outpatient photos | 5,450 clinical images from ~3,000 patients in South India. 245 diagnoses in a Rook's-based hierarchy (main class → subclass → disease), plus body-site and lesion-descriptor annotations. | Harvard Dataverse `doi:10.7910/DVN/W7OUZM` (2 zip files, ~4 GB). Code: `github.com/shan4035/DermaCon-IN-code`. | CC BY-NC-SA 4.0 (non-commercial) |
| **SCIN** (Google + Stanford) | **FCI development** (images plus real self-reported history); second fairness axis (Monk skin tone) | 10,408 images from 5,033 cases; US crowdsourced smartphone photos | Public GCS bucket `dx-scin-public-data`. Repo: `github.com/google-research-datasets/scin` (see `scin_demo.ipynb` and the Dataset Documentation). | Read the licence/terms in the repo |
| **DDI** (Stanford) | Dark-skin cancer benchmark (already in use) | 656 images | Stanford AIMI, under a data-use agreement (Ujjwal already has access) | Research use per the DUA |
| **PAD-UFES-20** (optional) | Second FCI replication on skin lesions | 2,298 images, 6 lesion diagnoses; history fields: itch, grew, hurt, changed, bleed, elevation | Mendeley Data | Verify (believed CC BY 4.0) |
| **Fitzpatrick17k** (optional) | Extra per-skin-type evaluation, if not already in the training pool | ~16.5k images, 114 conditions | `github.com/mattgroh/fitzpatrick17k` (URL list; many links are dead) | Verify |

Record every licence and its commercial-use status in `docs/LICENSES.md`.

### 6.2 Standard index format
For every dataset, create `data/processed/<dataset>/index.parquet` with these columns:

```
image_id, case_id, path, source_label, source_hierarchy,
mnkai_class (nullable), mnkai_superclass (nullable), mapping_type,
fst (1-6 or NA), fst_source, monk (1-10 or NA), monk_source,
age, sex, body_site, split, sha256, phash
```

`case_id` is the patient or case identifier used for all grouping.

### 6.3 SCIN — download and parse
1. **Download** anonymously (no account needed), either with the Python client `google.cloud.storage.Client.create_anonymous_client()` or with `gcloud storage cp -r gs://dx-scin-public-data/dataset data/raw/scin/`.
   - Confirm the exact paths from `scin_demo.ipynb`. Expected: `dataset/scin_cases.csv`, `dataset/scin_labels.csv`, `dataset/images/`.
2. **Document the schema.** Read the Dataset Documentation and list the **actual** column names in `docs/datasets/SCIN_SCHEMA.md`. Expected content (verify, don't assume):
   - self-reported symptoms, e.g. itching, burning, pain, bleeding, increasing size, darkening, bothersome appearance;
   - other symptoms, e.g. fever, chills, fatigue, joint pain, mouth sores, shortness of breath;
   - condition duration, textures, body parts, age group, sex;
   - dermatologist condition labels with confidences (a weighted differential);
   - dermatologist-estimated Fitzpatrick type (eFST);
   - layperson-estimated Monk skin tone (US and India labeller pools);
   - self-reported FST.
3. **Work at case level** (each case has 1–3 images).
4. **Ground truth** is the condition with the highest dermatologist weight.
   - Primary analysis: gradable cases whose top weight is ≥ 0.5, with sensitivity analyses at ≥ 0.3 and ≥ 0.7.
   - If fewer than 40% of cases survive the 0.5 threshold, make 0.3 the primary threshold and record why.
   - Record how many cases are dropped at each step.
5. **Skin-tone groups:**
   - Primary: dermatologist-estimated FST (mode across dermatologists; on a tie, take the darker type, and record this rule). Groups are I–II, III–IV and V–VI.
   - Secondary: Monk (India labellers), in bins 1–3, 4–6 and 7–10.
6. **Known issues:** handle those listed in the SCIN README (duplicate images, one missing image, gradable-but-unlabelled cases).

### 6.4 DermaCon-IN — download and prepare (no predictions!)
1. **Download.** Try the Dataverse "download all files" API for `doi:10.7910/DVN/W7OUZM`. If access needs a login or accepting terms, ask Ujjwal to download it in a browser (H2). Extract as described in the repo README.
2. **Index.**
   - Build the index and find the patient ID (for grouping), the diagnosis hierarchy, the body site, and any skin-tone field.
   - If there is no skin-tone label, do **not** invent one. Per-tone results on DermaCon-IN will then be "not available", optionally with a clearly labelled automated estimate reported as exploratory.
3. **What is allowed now:**
   - Allowed: hashing, de-duplication, label mapping, counts.
   - **Not allowed until Phase 7:** running any classifier on it.

### 6.5 De-duplication and leakage checks
Reuse the existing leakage-audit code where possible.
1. **Exact duplicates:** compare sha256 hashes.
2. **Near duplicates:**
   - Flag pairs by perceptual hash (pHash) Hamming distance ≤ 8 (tune this on known duplicate pairs).
   - Then flag pairs by embedding cosine similarity ≥ 0.95, using the current DINOv2 features. Use embeddings only — no labels, no predictions.
3. **Scope:** check within each dataset, and **between each evaluation set and the training pool**.
4. **Confirm and log:**
   - Build a local HTML gallery of candidate pairs for manual confirmation (never upload it).
   - Remove confirmed duplicates from evaluation sets and log the counts in `RESULTS.md`.

### 6.6 Label-mapping protocol (DermaCon-IN → MNKAI, SCIN → MNKAI)
Create `data/mappings/<dataset>_to_mnkai_v1.csv` with these columns:

```
source_dataset, source_label, source_hierarchy, n_images, n_cases,
proposed_mnkai_class, proposed_superclass,
match_type (exact | synonym | broader | narrower | none), rationale,
derm_decision (accept | reject | alternative), derm_alternative, derm_notes
```

Rules:
- Claude Code drafts proposals using names, synonyms and the hierarchy. The derm head decides (H3).
- `exact` or `synonym` → used for fine-grained metrics.
- `broader` or `narrower` → used for superclass-level metrics only.
- `none` → **out-of-taxonomy (OOT)**. OOT cases are kept for abstention/OOD analysis and never counted as in-taxonomy.
- After sign-off, copy the file to `data/mappings/frozen/` and record its sha256 in `RESULTS.md`.

### 6.7 Splits
- **SCIN** is split by case, stratified by (skin-tone group × superclass):
  - `sealed_test`: 25%, sealed until Phase 7.
  - `dev`: the other 75%, divided into `dev_train` 50%, `dev_tune` 25% and `dev_cal` 25% (percentages of dev).
  - Save the splits to `splits/scin_v1.csv` together with the seed.
  - If any skin-tone group has fewer than 100 cases in `dev_cal`, merge groups into FST I–III vs IV–VI and record the decision.
- **Image head and SCIN:** SCIN probabilities must come from a head that never saw those cases.
  - If SCIN is in the training pool (Phase 0.4), train the v1.0 head on the pool **minus SCIN** (and minus DermaCon-IN, if present).
  - Record this in `RESULTS.md`.

---

## 7. Phase 3 — Backbone comparison (Days 3–8)

**Why:** reviewers will ask why you didn't use a dermatology foundation model. Because the probe is frozen, testing a new backbone costs only one feature-extraction pass.

### 7.1 Backbone registry
Create `mnkai/backbones.py` (or match the repo's existing structure).
- Each entry records: `name, loader, preprocessing (resize / crop / normalisation), input_size, embedding_dim, licence, commercial_ok`.
- Use each model's **official** preprocessing and example code.

| Name | Source | Notes |
|---|---|---|
| `dinov2_vitb14_518` | Current backbone | Reuse the existing cache; don't recompute. |
| `dinov2_vitl14` | torch hub / timm | May already be cached from the ensemble work. |
| `panderm_base`, `panderm_large` | `github.com/SiyuanYan1/PanDerm`. Weights are linked from the README: `panderm_bb_data6_checkpoint-499.pth` and `panderm_ll_data6_checkpoint-499.pth`. | CC-BY-NC-ND 4.0 (non-commercial). Use the repo's feature-extraction notebook to get preprocessing right. |
| `derm_foundation` | Hugging Face `google/derm-foundation` (gated; HAI-DEF terms) | TensorFlow BiT-101x3 CNN; 448×448 PNG input; 6144-dim embedding. Follow the official quick-start notebook exactly, and use a separate environment if TensorFlow conflicts with PyTorch. It is slow on a laptop GPU: if the projected time is over 12 h, ask about Vast.ai (H6). |
| `dermfm_zero` (PanDerm-2) | `github.com/SiyuanYan1/DermFM-Zero` (released Feb 2026) | Include only if the weights and licence allow research use. |
| `dermino` | arXiv 2508.12190 | Include only if public weights exist; otherwise record "not available". |

**Contamination check:** read each foundation model's paper for its pretraining data. If it lists SCIN, DDI, Fitzpatrick17k or DermaCon-IN, flag that evaluation set as possibly contaminated for that backbone.

### 7.2 Feature caching
Generalise `_cache_features.py` into:

```
scripts/cache_features.py --backbone NAME --index PATH --out features/<dataset>/<backbone>/
```

- Write sharded fp16 `.npy` files (e.g., 10k images per shard) plus `ids.parquet`.
- Make it resumable (skip finished shards) and checksum each shard.
- **What to cache now:** the training pool, SCIN and DDI. **Not DermaCon-IN** — that happens in Phase 7, for the chosen backbone only.
- Smoke test on 200 images and report images/second and projected total time before launching full runs.
- **Saving time:** first compare all backbones on a stratified 50k subset of the training pool (same folds). Fully cache only the top two, plus DINOv2-B, which is already cached.

### 7.3 Probe comparison
For each backbone, and for the concatenation of DINOv2-B with the best derm backbone:
- Train the **same** head, with the **same** class balancing, bias tuning and **saved folds** as `_probe_multiclass.py`.
- Report, with 95% bootstrap CIs:
  - OOF macro-F1 (primary), balanced accuracy, top-3 accuracy, superclass accuracy, and ECE after temperature scaling;
  - per-FST macro-F1 and top-3;
  - the DDI binary-cancer metric (the same one the original result used), via `_probe_backbone.py`;
  - cost: parameters, embedding dim, ms/image on the RTX 4050, and the licence/commercial flag.

### 7.4 Pre-declared selection rule
Write this into `RESULTS.md` **before** running 7.3:

> Choose the single backbone with the highest OOF macro-F1. If a smaller, faster or commercially licensable backbone is within 0.01, choose that instead. Use a concatenation only if it beats the best single backbone by ≥ 0.02. External and sealed data are never used for selection.

Also report the paired bootstrap CI of the difference between the top two, and note whether the chosen backbone would block commercial use.

---

## 8. Phase 4 — Evaluation hardening (Days 4–9)

- **4.1 Confidence intervals.**
  - Every metric gets a patient/case-level bootstrap CI (2,000 resamples, percentile intervals).
  - Comparisons use a paired bootstrap; simple proportions use Wilson intervals.
  - Wrap the existing `evaluation/bootstrap.py`.
- **4.2 Clinically serious errors** (needs the H3 malignant/urgent flags):
  - malignant sensitivity: top-1, in the top-3, and inside the conformal set;
  - **malignant-miss rate:** true malignant, AND no malignant class in the conformal set, AND not referred;
  - the same metrics for urgent classes;
  - the safety rule "refer if any malignant class is in the conformal set": its extra referral rate and how many misses it prevents.
- **4.3 Fairness.**
  - Per FST group and per Monk group (where available), report: accuracy, top-3, macro-F1, ECE, conformal coverage and set size, and referral rate.
  - Disparity = max − min across groups, with a bootstrap CI.
- **4.4 Error analysis.**
  - Superclass confusion matrix and the top-20 confused class pairs.
  - Local HTML error galleries for the derm head (never uploaded).
  - A list of suspected label errors.
- **4.5 Ablation table.** Reuse existing results where they exist; re-run only the cheap ones.
  - frozen probe vs fine-tuning;
  - single backbone vs the six-backbone ensemble (DDI);
  - image-only vs image + age/sex/site;
  - image-only vs image + history (Phase 5 baselines B1 vs B3);
  - TTA on vs off;
  - per-class bias on vs off.
- **4.6 Robustness.** Re-run `evaluation/corruption.py` (blur, noise, JPEG compression, brightness) for the chosen backbone. Clinic phone photos vary a lot.

---

## 9. Phase 5 — New algorithm: Fair Conformal Interviewing (FCI) (Days 6–13)

### 9.1 Plain-language version (for Ujjwal)
**The problem.** A good dermatologist asks more questions when a photo is hard to read. Our image model is less certain on some skin tones. With standard conformal prediction, those patients get one of two bad outcomes: a less reliable answer, or a longer and less useful differential. You cannot fix both at once, and this has been proven (see 9.2).

**How FCI fixes it.** FCI does what a clinician does:
- It asks the most useful history question, one at a time, only while the differential is still too long.
- It gives each skin-tone group the question budget that group needs.
- The result: **every group ends up with an equally short differential that is equally guaranteed to contain the right answer.**

**The price.** Some groups answer a few extra questions. This is measured and reported openly as "burden".

### 9.2 Positioning (cite these; don't over-claim)
Already published:
- **Group-conditional (Mondrian) conformal prediction by skin tone:** Lu et al., AAAI 2022; Vovk et al. 2003; Romano et al. 2020 (equalized coverage).
- **Choosing questions by how much they shrink the conformal set:** Conformal Information Pursuit (C-IP), arXiv 2507.03279, tested on MediQ.
- **The trade-off we target:** "On the Burden of Achieving Fairness in Conformal Prediction", arXiv 2605.14260 (2026). It shows that exact group-wise coverage and equal expected set size cannot, in general, both be achieved with a fixed score distribution.
- **Uncertainty-directed history-taking with images, done end-to-end by an LLM:** multimodal AMIE (Nature Medicine 2026; arXiv 2505.04653).

**Our candidate contribution:**
- A calibrated image-classifier prior combined with structured history.
- **Per-group question budgets, tuned so that post-interview set sizes are equal across skin tones, while group-conditional coverage is guaranteed end-to-end.**
- **Burden** (questions asked and referral rate per group) reported as an explicit fairness cost.

**Before calling it novel:** do a targeted literature check on Semantic Scholar, arXiv and Google Scholar. Search "conformal" + ("feature acquisition" / "question asking" / "active") + ("fairness" / "group-conditional"). Record the search in `docs/NOVELTY_CHECK.md`.

### 9.3 Architecture decision
- In v1.0, **the LLM never produces probabilities.** FCI chooses questions and computes probabilities from data-estimated tables, which makes it calibratable, auditable and reproducible.
- Gemini is optional and has only two jobs:
  - phrasing or translating the chosen question;
  - mapping a free-text reply to one structured option, which the user then confirms by tapping a button.
- **For the prospective study:** use structured buttons only (no LLM, no image upload), unless the IEC approval explicitly covers Gemini.
- **Legacy flow:** if `fuse_history`'s likelihood ratios come from Gemini (Phase 0.3), keep it only as a `legacy_llm` demo mode, disabled in the frozen study config.

### 9.4 Components and maths

**Label space.** SCIN won't cover every MNKAI class.
- `K_scin` = the MNKAI classes with ≥ 30 mapped SCIN dev cases.
- Merge the remaining in-taxonomy cases into one `other_in_taxonomy` label.
- Exclude OOT cases and report the counts. (FCI's guarantee covers in-taxonomy cases only; say so in the paper.)
- If `|K_scin| < 6`, run FCI at superclass level instead.

**Image probabilities `p_img(k | x)`:**
1. Run the frozen image model (with the head trained without SCIN) on each image. Sum the fine-class probabilities into the `K_scin` bins.
2. Combine a case's 1–3 images by averaging their log-probabilities, then renormalise.
3. Fit a temperature `T_img` on `dev_train`, because the label space and domain have changed.
4. Optional prior-shift correction: multiply by `w_k = prior_scin_train(k) / prior_pool(k)` and renormalise. Keep it only if it lowers NLL on `dev_tune`.

**Free inputs** (known without asking): age group, sex, body site.

**Question bank `Q`:**
- Built from the SCIN history and symptom fields (schema from Section 6.3).
- Multi-select fields become separate yes/no questions. Duration is categorical, merged into 4–5 clinically sensible bins.
- Every question also offers "not sure", which gives no update.
- Store the bank in `fci/question_bank.yaml` with fields `id, scin_columns, options, wording_en, wording_hi_draft, wording_pa_draft, approved_by`.
- Translations stay drafts until a native-speaking clinician approves them (H3).

**Likelihood tables:** `L_q[a | k] = P(answer a to question q | disease k)`.
- Estimate them on `dev_train` with add-one (Dirichlet α = 1) smoothing.
- Free inputs get tables too.

**Posterior after history `h`** (tempered naive Bayes):

```
log p_h(k) = beta * log p_img(k | x) + gamma * SUM_{(q,a) in h} log L_q[a | k] + const
(then normalise over k)
```

- Fit `beta, gamma` on `dev_tune` by grid search over {0.25, 0.5, ..., 1.5}.
- The objective is NLL, averaged over random question subsets of length 0 to B_max.
- Tempering corrects the overconfidence of naive Bayes.

**Conformity score (LAC):** `s(k) = 1 - p_h(k)`.
- Implement APS as a sensitivity analysis.
- The main image-only pipeline keeps RAPS; state this difference in the paper.

**Groups:** the skin-tone group `g` must be known at test time. It is dermatologist-estimated in SCIN and clinician-recorded in the study.

### 9.5 The policy
**Parameters:**
- θ = (per-group budgets B_1…B_G, target set size `k_star`, minimum gain `delta`).
- Provisional per-group thresholds `tau_tilde[g]`. These are split-conformal quantiles of the image + free-input posteriors on `dev_tune`, used only for decisions.

```
FCI_POLICY(x, g, answer_oracle):
    h = free_inputs(x); asked = {}
    p = posterior(x, h)
    loop:
        C_tmp = { k : 1 - p[k] <= tau_tilde[g] }
        if |C_tmp| <= k_star or |asked| >= B[g] or no questions left: break
        for each unasked q:
            P(a | h) = SUM_k L_q[a|k] * p[k]
            ES(q)    = SUM_a P(a|h) * |C_tmp(posterior(x, h + (q,a)))|
        q_best = argmin_q ES(q)        # ties: lower expected entropy, then fixed order
        if |C_tmp| - ES(q_best) < delta: break
        a = answer_oracle(q_best); asked.add(q_best)
        if a != "not sure": h = h + (q_best, a)
        p = posterior(x, h)
    return p, len(asked)
```

**The SCIN oracle** returns the recorded field. Every SCIN case answered the whole questionnaire, so "asking" just means revealing the recorded answer; a blank field counts as "not sure".

**Speed trick (important):** the question chosen at each step depends only on `h` and `tau_tilde[g]`, not on `B[g]` or `k_star`.
- So compute each case's greedy question path once, up to B_max (default 6), storing the posterior, set size and ES at every step.
- Any (B_g, k_star, delta) is then just a stopping rule applied to the stored path, so the grid search takes seconds.

### 9.6 Final sets and the guarantee

**Final set:** `C(x) = { k : 1 - p_T(k) <= tau[g] }`.

**Computing `tau[g]`:**
- Run the **frozen** policy on every `dev_cal` case and compute the scores `S_i = 1 - p_T,i(y_i)`.
- For group g with n_g calibration cases, `tau[g]` is the ceil((n_g + 1)(1 − α))-th smallest `S_i` in that group.
- If that index exceeds n_g, set `tau[g] = 1` (the set contains all classes).
- α = 0.10 is primary; α = 0.05 is a sensitivity analysis.

**Empty sets:** if `C(x)` is empty, return `{argmax p_T}` and count how often this happens. This can only increase coverage.

**Referral** (reported separately from coverage): refer if `|C| > k_refer` (default 3), or, with the safety rule on, if any malignant class is in `C`.

**Proposition (for the paper).**
- *Statement:* suppose all policy parts (L, beta, gamma, T_img, tau_tilde, θ) are fixed using data disjoint from `dev_cal`, and calibration and test cases are exchangeable within each group. Then P(Y ∈ C(X) | G = g) ≥ 1 − α for every g.
- *Reason:* the whole image + interview pipeline is a fixed function of (image, answers), and Mondrian split-conformal validity holds for any fixed score function.
- *No contradiction with the impossibility result:* that result assumes a fixed score distribution, whereas FCI changes each group's score distribution by acquiring more information.

**Caveat to state:** the guarantee holds for SCIN-like data. For CMCH patients it must be checked empirically, in the prospective study.

### 9.7 Budget tuning (on `dev_tune` only)
**Grid:**
- B_g ∈ {0,…,6} per group;
- k_star ∈ {1, 2, 3};
- delta ∈ {0, 0.1, 0.25}.

**Evaluating each θ** (2-fold cross-fitting inside `dev_tune`):
- Compute `tau[g]` on one half with policy θ.
- Measure per-group coverage and mean set size on the other half.
- Swap the halves and average.

**Objective:**
- Minimise the mean number of questions, subject to:
  - (max_g mean|C|_g − min_g mean|C|_g) ≤ ε, with default ε = 0.25;
  - max_g mean|C|_g ≤ 3.0.
- If no θ satisfies both constraints, choose the one with the smallest gap and report that.

**Pareto frontier:** save the whole frontier (set-size gap vs mean questions, and vs worst-group questions) for a figure.

### 9.8 Baselines
All are Mondrian-calibrated on `dev_cal` unless noted.

| ID | Method |
|---|---|
| B0 | image only, **pooled** (marginal) conformal |
| B1 | image only, Mondrian |
| B2 | image + free inputs, Mondrian |
| B3 | image + free inputs + **all** questions (information ceiling, maximum burden) |
| B4 | fixed question order, common budget B |
| B5 | expected-entropy (information-gain) question selection, common budget B |
| B6 | C-IP-style set-size selection, common budget B (FCI without per-group budgets) |
| **FCI** | set-size selection + per-group budgets tuned for equal set sizes |
| B7 (optional) | legacy Gemini free-form interview with a simulated patient answering from SCIN's recorded fields. Only if the SCIN terms allow sending images to a third-party API and Ujjwal approves the API cost. |

For B4–B6, sweep B over {0,…,6}.

### 9.9 Metrics and outputs
Report per group and overall, with 95% CIs (case bootstrap) and the variability over 200 re-draws of the calibration set:
- coverage, mean set size, singleton rate, and empty-set rate (before the fix);
- mean and max number of questions (**burden**), and referral rate;
- top-1 and top-3 accuracy of `p_T`, NLL, ECE, and malignant-miss rate.

**Disparities:** coverage gap, set-size gap, and burden ratio (worst group's mean questions / best group's).

**Figures:**
- **F-A:** coverage by group × method, with the 1 − α line.
- **F-B:** mean set size by group × method.
- **F-C:** Pareto frontier (set-size gap vs extra questions).
- **F-D:** set size vs number of questions per group (B6 sweep).
- **F-E:** one worked example as a schematic (image → initial set → questions → final set). Show a real SCIN image only if its licence allows reproduction.

### 9.10 Simulation study (do this first, Days 6–7)
Write `fci/simulate.py`:
- K = 8 classes and G = 2 groups.
- Group 2's image probabilities are noisier (e.g., a larger logit-noise scale).
- M = 10 yes/no questions with known likelihood tables.
- Samples are exchangeable.

Verify over 1,000 repetitions:
1. Mondrian coverage is ≥ 1 − α per group (within Monte-Carlo error).
2. B0 (pooled) shows a coverage gap.
3. B1 shows a set-size gap.
4. FCI closes the set-size gap by giving group 2 more questions, with coverage intact.

This tests the code before it touches real data, and becomes a paper figure.

### 9.11 Code layout and tests
```
fci/
  taxonomy_bridge.py   # fine classes -> K_scin label space
  question_bank.py     # loads question_bank.yaml, answer spaces, SCIN column mapping
  likelihoods.py       # smoothed tables L_q[a|k]
  posterior.py         # tempered naive Bayes, temperature, prior shift
  conformal.py         # LAC/APS scores, Mondrian and pooled quantiles
  policy.py            # greedy path computation + stopping rules
  tuning.py            # cross-fitted grid search, Pareto frontier
  baselines.py         # B0-B7
  metrics.py           # coverage, size, burden, referral, disparities
  simulate.py          # synthetic data generator
scripts/fci_simulation.py
scripts/fci_run_scin.py
scripts/fci_report.py
configs/fci/*.yaml                 # inherit with _base_
tests/test_fci_conformal.py        # coverage holds on synthetic data
tests/test_fci_posterior.py        # sums to 1; an answer favouring k raises p(k)
tests/test_fci_policy.py           # budgets respected; deterministic; "not sure" handled
tests/test_fci_splits.py           # train/tune/cal/test disjoint by case_id
```

Use numpy vectorisation throughout; avoid Python loops over classes.

### 9.12 Optional replication
Repeat on PAD-UFES-20: 6 lesion classes, history fields itch/grew/hurt/changed/bleed/elevation, and a skin-type field (check its distribution). Report the results even if the skin-type range is narrow.

### 9.13 Integration into the app
- Add `interview_mode: fci` to the `demo_app` config. The flow is:
  1. photo → `p_img`;
  2. the clinician selects the skin type;
  3. FCI asks structured questions (buttons; optional Gemini phrasing/translation);
  4. the app shows the final set and the referral flag.
- Log every step locally: answers, intermediate sets, timings.

### 9.14 Go / no-go (Day 13)
- **Go:** on dev data, FCI reduces the set-size gap compared with B6 at a modest extra burden, with coverage intact.
- **No-go:** freeze without FCI, and report the negative result plus the simulation study in the paper.
- Either way, record the decision and the reasons in `RESULTS.md`.

---

## 10. Phase 6 — Freeze v1.0 (Days 13–14)

What goes into v1.0 depends on the Day-13 status:
- switch backbone only if Phase 3 has finished;
- include FCI only on a Go decision.

**Freeze checklist:**
- [ ] The final backbone is chosen by the pre-declared rule (Section 7.4).
- [ ] The head is retrained on the full training pool, excluding SCIN, DermaCon-IN and all prospective data.
- [ ] Temperature, per-class bias and RAPS `qhat` are recomputed on a held-out calibration split **from the training pool** (reuse the `_trust_refresh` logic) — never on external data.
- [ ] If FCI is Go: `fci_artifacts.pkl` holds the L tables, beta, gamma, T_img, prior weights, tau_tilde, tau, θ and the question-bank version.
- [ ] `models/v1.0/` is packaged with: `model.pkl`, `fci_artifacts.pkl` (if any), `config_frozen.yaml`, `taxonomy.yaml`, `label_maps/`, `question_bank.yaml`, `MODEL_CARD.md`, `SHA256SUMS.txt`.
- [ ] `MODEL_CARD.md` covers:
  - intended use (decision support, not autonomous diagnosis);
  - the classes;
  - training data and licences;
  - dev performance with CIs, and per-tone results;
  - limitations and known failure modes;
  - version hashes.
- [ ] The `demo_app` study config points to `models/v1.0/`, with `legacy_llm` off.
- [ ] Git tag `v1.0-frozen`.
- [ ] **CHECKPOINT H4.** If the index test differs from what the IEC approved, flag it for H5.

**Fallback (minimum viable freeze):** if time is short, freeze the current DINOv2 pipeline plus the Phase 4 hardening as v1.0, so the prospective study can start on time. FCI and new backbones then become "v1.1 / future work".

---

## 11. Phase 7 — Sealed external validation (Days 14–15, exactly once)

1. **Log the access** in `logs/sealed_access.log`.
2. **DermaCon-IN** (the mapped, de-duplicated subset), run through the frozen v1.0 image pipeline. Report:
   - top-1, top-3 and macro-F1 on exact/synonym-mapped classes;
   - superclass accuracy;
   - ECE (expect worse calibration under dataset shift; report it honestly);
   - RAPS coverage and set size;
   - the abstention curve;
   - per-tone results, if a skin-tone label exists;
   - on OOT cases: referral/abstention rate and OOD-gate (`_ood_gate.py`) performance;
   - malignant sensitivity, if applicable.
3. **SCIN sealed test:** the image-only pipeline and, if Go, FCI with all baselines.
4. **No model changes** after seeing these results. Label any further analysis "exploratory".
5. **Leakage paragraph for the paper:**
   - SCIN was collected from March to November 2023 and released in 2024. DermaCon-IN was released in 2025.
   - Neither can be in DINOv2's web-curated pretraining data, because the DINOv2 paper appeared in April 2023.
   - For derm-specific backbones, report the pretraining-data check from Section 7.1.

---

## 12. Phase 8 — Prospective study kit (Days 12–16; drafts for the derm head and IEC)

**Design principle:**
- In the study, the live app runs the frozen image pipeline and shows the conformal set.
- The **full questionnaire is recorded for every patient**.
- FCI and all baselines can then be evaluated offline on real Indian patients afterwards, without deploying an unvalidated algorithm live.

Create the following in `study/`.

**`protocol_draft.md`**
- *Objectives:*
  - Primary: top-3 accuracy of v1.0 against the reference standard.
  - Secondary: top-1 and superclass accuracy; conformal coverage and set size; referral appropriateness; malignant sensitivity; per-FST results; offline FCI results.
- *Design and setting:* prospective, consecutive, single-centre diagnostic accuracy study (STARD 2015) in the CMCH dermatology OPD.
- *Eligibility:* the IEC decides on minors and intimate sites.
- *Index test:* frozen v1.0, identified by its SHA256.
- *Reference standard:*
  - the derm head's diagnosis, blinded to the AI output;
  - plus KOH, biopsy or other tests when clinically done;
  - and a rule for resolving disagreements.
- *Blinding:* AI output is not shown to the treating clinician (except in the reader arm).
- *OOT handling, pre-specified two ways:*
  - (a) all patients, with OOT counted as a failure unless the system referred;
  - (b) in-taxonomy patients only.
- *Sample size:* with n = 100, the 95% CI half-width is about ±7.8% at 80% accuracy and about ±9.8% at 50%.
- *Also include:* data management, and CTRI registration before the first patient.

**`sap.md`** (statistical analysis plan, pre-specified)
- Wilson CIs; descriptive per-group analyses; missing-data rules.
- Reader arm: paired comparison (McNemar per reader; a mixed-effects model across readers).

**`crf_fields.csv`** (data dictionary)
- *Patient:* study_id, visit date, age, sex, clinician-assessed FST (+ optional Monk with a printed chart), body site.
- *History:* the **full question bank**, asked to everyone.
- *India-relevant extras*, recorded for future work and not used by the frozen FCI:
  - prior topical steroid use;
  - household members affected;
  - new medicines in the last 8 weeks;
  - occupation / sun exposure;
  - diabetes.
- *Photos:* IDs, count, device.
- *AI outputs:* top-5 with probabilities, conformal set, abstain flag, model hash.
- *Reference:* diagnosis (free text + mapped MNKAI class + confidence), investigations and results, final diagnosis.
- *Reader-arm fields.*
- If CMCH uses REDCap, also export a REDCap data-dictionary CSV.

**`photo_protocol.md`**
- 2–3 photos per lesion: an overview at 30–50 cm plus a close-up at 10–15 cm.
- Diffuse natural light; no filters, beauty mode or digital zoom; plain background.
- Crop identifying features unless the lesion is there.
- Use the same devices throughout.

**`consent_points.md`**
- What is collected, and that analysis runs locally.
- No third-party cloud (recommended for v1.0).
- Storage and retention; the right to withdraw.
- Notes on India's DPDP Act 2023 and any rules in force.

**`ctri_fields_draft.md`**
- Pre-filled CTRI fields for Ujjwal and the derm head to complete (H5).

**`reader_study.md`** (optional)
- 3–4 interns or non-dermatology doctors read all cases twice: unaided, then AI-aided (randomised order, or a washout of ≥ 2 weeks).
- Outcome: the change in top-1 and top-3 accuracy.

---

## 13. Phase 9 — Paper and presentation assets (Days 15–20)

**Build script:** `scripts/make_paper_assets.py` builds everything from `RESULTS.md` and the saved predictions.
- Tables go to `paper/tables/` (CSV + LaTeX).
- Figures go to `paper/figures/` (PNG + PDF, 300 dpi, colour-blind-safe palette).

**Tables:**

| ID | Content |
|---|---|
| T1 | Datasets and splits (with licences) |
| T2 | Backbone comparison |
| T3 | v1.0 dev performance, with CIs |
| T4 | Trust metrics per skin tone |
| T5 | FCI vs baselines |
| T6 | Ablations |
| T7 | External validation (DermaCon-IN, SCIN sealed) |

**Figures:**

| ID | Content |
|---|---|
| F1 | Pipeline diagram |
| F2 | Reliability diagrams |
| F3 | Coverage and set size per tone |
| F4 | FCI Pareto frontier |
| F5 | Set size vs number of questions |
| F6 | Superclass confusion matrix |
| F7 | Abstention curve |
| F8 | Simulation study |

**Drafts** in `paper/drafts/`, for Ujjwal to edit:
- Methods: data, model, trust layer, FCI, evaluation, statistics.
- Results skeleton.
- Limitations.

**Reporting checklists** (filled drafts pointing to the relevant sections):
- TRIPOD+AI (BMJ 2024);
- CLAIM 2024 update;
- STARD 2015 (for the prospective part);
- DECIDE-AI (if there is a reader arm).

**Presentation outline** in `paper/presentation_outline.md`, 10–12 slides:
1. The problem
2. Why Indian and dark skin
3. System overview
4. Trust layer
5. Results
6. Fairness
7. FCI
8. External validation
9. Prospective study plan
10. Limitations
11. Next steps

**References:** verify that **every** reference exists (title, authors, venue, year, ID) before it goes into the paper.

---

## 14. Stretch — Android/mobile student model (after freeze; not part of the study)

1. **Teacher and students.** The teacher is the frozen v1.0 image model. Student candidates: MobileNetV3-Large, EfficientNet-Lite0, MobileViT-S.
2. **Distillation.**
   - Train on cached teacher logits: KL divergence at temperature 2–4, plus cross-entropy.
   - 224–256 px input, strong augmentation, and the same group-aware folds.
3. **Export.**
   - Export via ONNX → ONNX Runtime Mobile, or via LiteRT (TFLite).
   - Apply INT8 post-training quantisation using a calibration subset.
   - Targets: under 25 MB, under 1 s on a mid-range phone, and a macro-F1 drop of at most 3 points.
4. **Re-validate.** Re-run temperature scaling, conformal calibration and per-tone fairness on the student, and report teacher vs student side by side.

---

## Appendix A — Proposed directory layout (adapt to the existing repo)

```
HANDOFF.md  CLAUDE.md  RESULTS.md  requirements-lock.txt
docs/        AUDIT.md ENVIRONMENT.md LICENSES.md NOVELTY_CHECK.md STALE_NUMBERS.md
             plans/ datasets/
configs/     taxonomy/ backbones/ fci/
data/        raw/ processed/ mappings/        (raw + processed gitignored)
features/    <dataset>/<backbone>/             (gitignored)
splits/
mnkai/       backbones.py
fci/
scripts/
models/      v1.0/
study/
paper/       tables/ figures/ drafts/
logs/        sealed_access.log
tests/
```

## Appendix B — `RESULTS.md` template

```
| id   | date       | commit  | script                        | config | data / split | metric   | value | 95% CI      | notes                  |
|------|------------|---------|-------------------------------|--------|--------------|----------|-------|-------------|------------------------|
| R001 | YYYY-MM-DD | <hash>  | scripts/_probe_multiclass.py  | <path> | pool OOF     | macro-F1 | 0.xxx | 0.xxx–0.xxx | v0.9 reproduction      |
```

Also keep a "Decisions" section: the taxonomy choice, the selection rule, the FCI go/no-go, group merges, and threshold choices — each with date and reason.

## Appendix C — Metric definitions

| Metric | Definition |
|---|---|
| Macro-F1 | Mean of per-class F1. Classes with no support in the evaluation set are excluded and listed. |
| Balanced accuracy | Mean per-class recall. |
| Top-k accuracy | The true class is among the k highest-probability classes. |
| ECE | Top-1 confidence, 15 equal-width bins. |
| NLL | Mean negative log-probability of the true class. |
| Coverage | Fraction of cases whose true class is in the prediction set. |
| Set size | Number of classes in the set. Also report the singleton rate. |
| Burden | Number of questions asked (mean and max). |
| Referral rate | Fraction of cases referred or abstained. |
| Malignant-miss rate | Defined in Phase 4.2. |
| Disparity | max − min across skin-tone groups. |
| CIs | Case-level bootstrap (2,000 resamples) for development metrics; Wilson intervals for study proportions. |

## Appendix D — References to verify and cite

**Backbones and datasets**
1. Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193.
2. Darcet et al. Vision Transformers Need Registers. ICLR 2024; arXiv:2309.16588.
3. Yan et al. A multimodal vision foundation model for clinical dermatology (PanDerm). Nature Medicine 2025; arXiv:2410.15038.
4. DermINO: Hybrid Pretraining for a Versatile Dermatology Foundation Model. arXiv:2508.12190.
5. DermFM-Zero (PanDerm-2). github.com/SiyuanYan1/DermFM-Zero (2026) — check for an accompanying paper.
6. Google Derm Foundation model card (Health AI Developer Foundations).
7. Daneshjou et al. Disparities in dermatology AI performance on a diverse, curated clinical image set (DDI). Science Advances 2022.
8. Groh et al. Evaluating deep neural networks trained on clinical images in dermatology with the Fitzpatrick 17k dataset. CVPR Workshops 2021.
9. Ward et al. Creating an empirical dermatology dataset through crowdsourcing with web search advertisements (SCIN). JAMA Network Open 2024; arXiv:2402.18545.
10. Madarkar et al. DermaCon-IN: A Multi-concept Annotated Dermatological Image Dataset of Indian Skin Disorders for Clinical AI Research. NeurIPS 2025 Datasets & Benchmarks; arXiv:2506.06099.
11. Pacheco et al. PAD-UFES-20. Data in Brief 2020.

**Conformal prediction and fairness**
12. Vovk, Lindsay, Nouretdinov, Gammerman. Mondrian Confidence Machine. Technical report, 2003.
13. Romano, Barber, Sabatti, Candès. With Malice Toward None: Assessing Uncertainty via Equalized Coverage. Harvard Data Science Review 2020.
14. Sadinle, Lei, Wasserman. Least Ambiguous Set-Valued Classifiers with Bounded Error Levels. JASA 2019.
15. Romano, Sesia, Candès. Classification with Valid and Adaptive Coverage (APS). NeurIPS 2020.
16. Angelopoulos et al. Uncertainty Sets for Image Classifiers using Conformal Prediction (RAPS). ICLR 2021.
17. Lu et al. Fair Conformal Predictors for Applications in Medical Imaging. AAAI 2022.
18. Conformal uncertainty quantification to evaluate predictive fairness of foundation AI model for skin lesion classes across patient demographics. Health Information Science and Systems 2025; arXiv:2503.23819.
19. Conformal Prediction Sets with Improved Conditional Coverage using Trust Scores. arXiv:2501.10139.
20. On the Burden of Achieving Fairness in Conformal Prediction. arXiv:2605.14260.
21. A Generic Framework for Conformal Fairness. arXiv:2505.16115.

**Interactive diagnosis**
22. Conformal Information Pursuit for Interactively Guiding Large Language Models. arXiv:2507.03279.
23. Li et al. MediQ: Question-Asking LLMs and a Benchmark for Reliable Interactive Clinical Reasoning. NeurIPS 2024.
24. Advancing conversational diagnostic AI with multimodal reasoning (multimodal AMIE). Nature Medicine 2026; arXiv:2505.04653.
25. Tu et al. Towards conversational diagnostic artificial intelligence (AMIE). Nature 2025; arXiv:2401.05654.
26. Zhou et al. SkinGPT-4. Nature Communications 2024.

**Calibration and distillation**
27. Guo et al. On Calibration of Modern Neural Networks. ICML 2017.
28. Hinton, Vinyals, Dean. Distilling the Knowledge in a Neural Network. 2015.

**Reporting guidelines**
29. Collins et al. TRIPOD+AI. BMJ 2024.
30. Tejani et al. CLAIM 2024 Update. Radiology: Artificial Intelligence 2024.
31. Bossuyt et al. STARD 2015. BMJ 2015.
32. Vasey et al. DECIDE-AI. Nature Medicine 2022.

## Appendix E — Day-by-day schedule

| Day | Focus |
|---|---|
| 1 | Phase 0 audit + reproduction → report (H1) |
| 2 | Phase 1 taxonomy + ledger; access requests (H2); start SCIN download; licences |
| 3 | Phase 2 indexing and de-duplication; draft label maps → derm head (H3); backbone registry + smoke tests |
| 4 | 50k-subset caching for derm backbones; Phase 4.1 CIs on v0.9 |
| 5 | Phases 4.2–4.4 on v0.9; DermaCon-IN prepared and sealed |
| 6 | FCI simulation study + unit tests |
| 7 | Probes per backbone; apply the selection rule; full caching of the winners |
| 8 | SCIN features + image probabilities (head without SCIN); FCI likelihoods, posterior, beta/gamma |
| 9 | Baselines B0–B3; Phase 4.5–4.6 ablations and robustness |
| 10 | FCI greedy paths, tuning, Pareto frontier; baselines B4–B6 |
| 11 | FCI dev results; PAD-UFES-20 replication (optional) |
| 12 | Study kit drafts → derm head; app `interview_mode: fci` + structured questionnaire |
| 13 | FCI go/no-go; freeze preparation; model card |
| 14 | **FREEZE v1.0** (H4); tag; begin sealed evaluation |
| 15 | Sealed external validation (DermaCon-IN, SCIN sealed test) |
| 16 | Paper tables and figures; study-kit revisions |
| 17 | Methods drafts; reporting checklists |
| 18 | Presentation outline; figure polish |
| 19–20 | Buffer; CTRI/IEC support (H5); final review with the derm head |

## Appendix F — Risks and fallbacks

| Risk | Fallback |
|---|---|
| Phase 0 reproduction mismatch | Stop and resolve it before anything else. |
| DermaCon-IN inaccessible, or few classes map | Run the external test at superclass level; lean on the prospective study. |
| Small SCIN–MNKAI overlap | Run FCI at superclass level; do the PAD-UFES-20 replication; rely on the simulation study. |
| Derm backbones too slow on the laptop | Use the 50k-subset comparison only; use Vast.ai (H6); or skip. |
| Too few dark-skin cases in SCIN calibration | Merge skin-tone groups and report the merge. |
| FCI no-go | Report the negative result plus the simulation; freeze without FCI. |
| Time overrun | Protect the minimum viable path, in order: audit → taxonomy/ledger → evaluation hardening → study kit → freeze → external validation. New backbones and FCI join v1.0 only if they are ready by Day 13. |
