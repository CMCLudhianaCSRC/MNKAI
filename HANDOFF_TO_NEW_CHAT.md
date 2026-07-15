# MNKAI-III — Handoff to a New Claude Chat

This document has two parts:
- **PART A** — a copy-paste prompt to start a new Claude Code chat
- **PART B** — the full changes report (reference; the new chat can read this file directly)

The repo is at `C:\Users\ujjwa\Desktop\MNKAIII`. The master plan is at
`C:\Users\ujjwa\.claude\plans\serene-stargazing-rabin.md`.

---

# PART A — COPY-PASTE THIS PROMPT INTO THE NEW CHAT

```
You are continuing work on MNKAI-III, a clinical-grade dermatology AI for Indian
deployment (DINOv2-B/14 + multimodal cross-attention, 271k-image curated corpus).
Repo root: C:\Users\ujjwa\Desktop\MNKAIII. Python venv: .\.venv\Scripts\python.exe.

FIRST, read these two files in full for complete context:
  1. C:\Users\ujjwa\.claude\plans\serene-stargazing-rabin.md   (master plan)
  2. C:\Users\ujjwa\Desktop\MNKAIII\HANDOFF_TO_NEW_CHAT.md     (this handoff — PART B)

CURRENT STATE: Fold 0 is trained (macroF1 0.6378, AUROC 0.9524, acc 85.3%).
Phase 1c post-hoc is partly done. Folds 1-4 training code + config + 3 helper
scripts are already implemented and tested. The bugs (shades_of_gray_p mismatch,
CorruptedDataset blank-image fallback) are already fixed.

YOUR JOB — implement and run the remaining pipeline, in this order:

PHASE 1c (finish post-hoc on fold 0, ~2h):
  - Resume robustness sweep, run grad-cam, hard-neg mining, screening thresholds,
    TTA inference. Commands are in the master plan Section 4.

PHASE 1d (SSL pretrain, ~6-8h): run scripts/_ssl_pretrain.py per master plan Sec 5.

PHASE 1e (folds 1-4, ~36-40h): run training with
  configs/phase1_dinov2_mnkai3_folds1-4.yaml for folds 1,2,3,4 + per-fold
  post-hoc. SMOKE-TEST 1 epoch first (verify per-class focal_gamma, class boost,
  KD teacher load all appear in the log). Master plan Section 6.

PHASE 1f (ensemble + SWA, ~3-4h) — INCLUDES NEW CODE (novelty N7):
  - SWA + 5-fold logit-averaged ensemble + TTA.
  - Add an --uncertainty-weighted mode to scripts/_ensemble_predict.py: weight
    each fold per-sample by inverse epistemic uncertainty (variance/disagreement
    of the 5 folds' softmax outputs; Lakshminarayanan 2017). Report naive vs
    uncertainty-weighted. Master plan Section 7.

PHASE 3 (Indian 27-class + hierarchical taxonomy, novelty N4) — NEEDS NEW CODE:
  - Build configs/phase3_indian.yaml (27 classes, KD from Phase-1 ensemble,
    source_weights boosting DermaCon-IN/MPOX, lr 1e-4, 30-50 epochs).
  - Add a hierarchical-taxonomy loss (2-level: Level-1 {neoplastic,
    inflammatory, infectious, pigmentary, other}; Level-2 = specific condition).
    Loss = L_level1 + L_level2_given_level1.
  - Define the taxonomy tree in src/mnkai/data/manifest.py.
  - Verify DermNet folder -> Indian-class mapping in manifest.py.
  - Then train. Master plan Section 10.

PHASE 4d (novelty analyses, ~2-3h, NO training) — NEW CODE, do before the 26th:
  - N5 counterfactual fairness: scripts/_counterfactual_fairness.py — hold image
    fixed, perturb the metadata token (FST I->VI, sex, age), measure argmax
    stability. Per-class + overall "counterfactual stability" metric. Cites
    Kusner et al. 2017.
  - N6 class-conditional (Mondrian) conformal: extend
    src/mnkai/calibration/conformal.py with a per-class conformal quantile mode
    so coverage holds PER CLASS, not just marginally (Romano/Sesia/Candes 2020,
    Vovk). Report marginal vs class-conditional coverage. Master plan Section 13A.

PHASE 1g (concept-bottleneck heads, novelty N3) — NEW CODE, post-presentation:
  Auxiliary heads predicting Derm7pt 7-point checklist concepts. Master plan Sec 8.

PHASE 1h (image-quality gate) — NEW CODE, post-presentation. Master plan Sec 9.

PHASE 4 (publication apparatus) — NEW CODE: scripts/_dermatologist_agreement.py,
  scripts/_clinical_impact.py, datasheet, model card. Master plan Sec 11-13.

The 11 novelty claims (N1-N11) are enumerated in master plan Section 1A — read
that section; it is the publication story the code must support.

CONSTRAINTS:
  - The user has a college presentation on 2026-05-26. Prioritise Phase 1c->1f
    and Phase 3 so the core pipeline + figures are ready. Phases 1g/1h/4 are
    post-presentation.
  - GPU is RTX 4050 8GB. Each fold ~8-10h. Use background processes + checkpoints.
  - CRITICAL: any new inference/eval script that loads images MUST build AugConfig
    with shades_of_gray_p read from cfg.augmentation.shades_of_gray_p (NOT the
    1.0 default) — mismatch crashes accuracy 85%->37%. See PART B bug #1.
  - All training checkpoints use the hardened format (model+optim+sched+scaler+
    ema+epoch+best_macro_f1) for crash-safe resume after Windows reboots.
  - Do NOT use the Derm1M dataset — rejected for licensing (CC BY-NC) + the
    project's no-scraping rule.

Start by reading the two files above, then give me a short status confirmation
and begin Phase 1c.
```

---

# PART B — FULL CHANGES REPORT

## B.1 Project identity

- **MNKAI-III** — clinical-grade dermatology AI, Indian (Fitzpatrick III–VI) deployment
- Author: Ujjwal Thakran, MBBS 2022, Christian Medical College & Hospital, Ludhiana
- Academic proposal title: *"Development of a CNN for Skin Disease Classification; Fine-Tuned on Indian Datasets (Fitzpatrick III–VI)"*
- Note: the proposal describes ConvNeXt-Nano; the actual build is **DINOv2-B/14** — far more advanced. Proposal needs updating (see `ACADEMIC_DOC_UPDATED.md`).

## B.2 Architecture & data (already built)

- Backbone: DINOv2-B/14 ViT (`vit_base_patch14_dinov2.lvd142m`), img_size 224 override
- Fusion: cross-attention between image patch tokens and a tabular metadata token (age, sex, anatomical site, Fitzpatrick)
- Corpus: 271k images, 20+ trustworthy sources (ISIC, HAM10000, BCN20000, PAD-UFES-20, Derm7pt, Fitzpatrick17k, DermNet, MED-NODE, DermaCon-IN, MPOX, SCIN-derived). No synthetic, no scraping.
- Phase 1 = 9 classes (mel, nv, bcc, scc, df, bkl, ak, vasc, normal)
- Phase 3 = 27 classes (9 base + 18 Indian-relevant)
- Group-aware K-fold splits (zero patient leakage)

## B.3 Fold 0 result (DONE)

macroF1 0.6378 · macro AUROC 0.9524 · accuracy 85.29% · ECE 0.031 (after T=0.848)
Weak classes: mel sens@spec95 = 0.728, bkl sens@spec95 = 0.619.

## B.4 Code changes made this session

### Bug fixes
1. **`shades_of_gray_p` config-mismatch (CRITICAL)** — `_posthoc_calibrate.py`, `_robustness_sweep.py`, `_gradcam_explain.py`, `_hardneg_mining.py` were building `AugConfig()` with the default `shades_of_gray_p=1.0` while training used `0.0`. This silently applied Shades-of-Gray color constancy the model never saw, crashing accuracy 85%→37%. Fixed in all 4 — they now read `cfg.augmentation.shades_of_gray_p`. **Any new script that loads images must do the same.**
2. **`CorruptedDataset.__getitem__`** in `_robustness_sweep.py` returned a blank zero-image on first file-open failure. Fixed: replicated the parent `SkinDataset` 20-retry same-label fallback.
3. **Robustness sweep** had no incremental save. Fixed: writes `robustness.json` after every corruption + `num_workers=0` to avoid silent worker death.

### Feature additions (folds 1-4 upgrades) — all backward-compatible
- `src/mnkai/losses/focal.py`:
  - `FocalLoss` now accepts per-class gamma (length-C list) via a `gamma_vec` buffer.
  - `ClassBalancedFocal` now accepts a `class_boost` per-class multiplier (re-normalised).
  - `build_loss()` factory validates and passes both.
  - Verified: mel/nv loss-weight ratio = 47.8× with the configured boosts.
- `src/mnkai/training/sampler.py`:
  - `make_weighted_sampler()` now accepts `class_boost` dict (e.g. `{mel:1.5,bkl:1.3}`) and `sample_weights` ndarray (hard-negative weights).
- `src/mnkai/training/train.py`:
  - KD teacher: loads `kd_teacher_ckpt`, computes T²-scaled KL-divergence soft loss, blends `L = (1-α)·L_hard + α·L_kd` (Hinton 2015).
  - `hardneg_weights_csv` config option → loads per-row sample weights into the sampler.
  - `save_epoch_ckpts_after` → saves `ckpt_epoch_NN.pt` (EMA state) for SWA.

### New config
- `configs/phase1_dinov2_mnkai3_folds1-4.yaml` — extends fold-0 config; enables per-class focal-γ, class-boost loss + sampler, MixUp/CutMix, KD from fold-0, drop_path 0.15, per-epoch ckpts, SSL backbone init.

### New scripts
- `scripts/_screening_thresholds.py` — per-class threshold for sens ≥ 0.95/0.90/0.85.
- `scripts/_predict_tta.py` — 5-view TTA inference, logit-space averaging.
- `scripts/_ensemble_predict.py` — N-fold logit-averaged ensemble + optional TTA + metrics report.

All three sanity-checked (imports, forward/backward, config parse). Not yet run on real data.

## B.5 What still needs building (new code)

| Item | Novelty | Files to create/edit | Priority |
|---|---|---|---|
| Uncertainty-weighted ensembling | N7 | `--uncertainty-weighted` mode in `scripts/_ensemble_predict.py` | for the 26th |
| Phase 3 config | N1 | `configs/phase3_indian.yaml` | for the 26th |
| Hierarchical taxonomy loss | N4 | new loss in `src/mnkai/losses/`, tree in `manifest.py` | for the 26th |
| DermNet→Indian-class mapping verification | N1 | `src/mnkai/data/manifest.py` | for the 26th |
| Counterfactual fairness analysis | N5 | `scripts/_counterfactual_fairness.py` | for the 26th (cheap, no training) |
| Class-conditional (Mondrian) conformal | N6 | extend `src/mnkai/calibration/conformal.py` | for the 26th (cheap, no training) |
| Concept-bottleneck heads | N3 | `ConceptHead` in `backbones.py`, `configs/phase1g_concepts.yaml`, Derm7pt concept columns | post-presentation |
| Image-quality gate | — | `scripts/_train_iqc.py`, integration hook | post-presentation |
| Dermatologist agreement analysis | N10 | `scripts/_dermatologist_agreement.py` | post-presentation |
| Clinical-impact analysis | N10 | `scripts/_clinical_impact.py` | post-presentation |
| Datasheet + Model Card | N10 | `docs/datasheet.md`, `docs/model_card.md` | post-presentation |

The full novelty list (N1-N11) is in master plan Section 1A — the new chat must
read it; it is the publication story all this code supports.

## B.6 What still needs running (existing code)

Phase 1c.6-1c.10, Phase 1d SSL, Phase 1e folds 1-4, Phase 1f ensemble — all have
working code and commands in the master plan Sections 4-7.

## B.7 Verification checklist for the new chat

- After robustness resumes: `clean` accuracy ≈ 0.85 (proves SoG fix holds; if <0.6 abort).
- Folds 1-4 smoke test (1 epoch): log must show `per-class focal_gamma`, `per-class loss boost`, `KD teacher loaded`.
- Ensemble: 5-fold macroF1 should beat any single fold by ≥ 0.02.
- Phase 3: 27-class macroF1 0.45-0.55 is expected and is NOT a regression (harder problem).

## B.8 Hard rules

- No Derm1M dataset (CC BY-NC license + violates no-scraping rule).
- No synthetic data, no scraped data — trustworthy medical sources only.
- Never skip the `shades_of_gray_p` config read in image-loading scripts.
- Checkpoints stay in the hardened format for crash-safe resume.
- Leave the user's own `python pipeline.py` process (if running) untouched.
