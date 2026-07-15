# MNKAI-III  ·  Pre-Training Verification Report

_All data acquisition, deduplication, and quality steps are complete._
_This report shows class balance, source provenance, and split integrity for the next training run._

---

## 1. Dataset Pipeline Summary

| stage | rows | 8-class rows | classes covered |
|-------|------|--------------|------------------|
| raw master | 271,058 | 260,460 | mel:12792, nv:208814, bcc:11392, scc:2942, df:770, bkl:9273, ak:5388, vasc:1465, normal:7624 |
| after dedup | 258,982 | 248,386 | mel:11689, nv:203427, bcc:10963, scc:2867, df:735, bkl:8584, ak:4835, vasc:1354, normal:3932 |
| after QA filter | 258,760 | 248,164 | mel:11689, nv:203206, bcc:10963, scc:2867, df:735, bkl:8584, ak:4835, vasc:1354, normal:3931 |
| final splits | 248,164 | 248,164 | mel:11689, nv:203206, bcc:10963, scc:2867, df:735, bkl:8584, ak:4835, vasc:1354, normal:3931 |

## 2. 8-class Distribution Over Pipeline

| class | raw | dedup | clean | splits |
|-------|-----|-------|-------|--------|
| mel | 12,792 | 11,689 | 11,689 | 11,689 |
| nv | 208,814 | 203,427 | 203,206 | 203,206 |
| bcc | 11,392 | 10,963 | 10,963 | 10,963 |
| scc | 2,942 | 2,867 | 2,867 | 2,867 |
| df | 770 | 735 | 735 | 735 |
| bkl | 9,273 | 8,584 | 8,584 | 8,584 |
| ak | 5,388 | 4,835 | 4,835 | 4,835 |
| vasc | 1,465 | 1,354 | 1,354 | 1,354 |
| normal | 7,624 | 3,932 | 3,931 | 3,931 |

## 3. Source Provenance (final clean manifest)

| source | rows | 8-class share |
|--------|------|---------------|
| ISIC_2024 | 106,574 | mel:52, nv:106435, bcc:41, scc:11, df:3, bkl:21, ak:11 |
| ISIC_2024_MSK | 50,399 | mel:30, nv:50311, bcc:29, scc:14, bkl:12, ak:3 |
| ISIC_D | 28,203 | mel:7524, nv:7609, bcc:5816, scc:1584, df:340, bkl:3872, ak:1386, vasc:72 |
| ISIC_E_EXTRA | 26,042 | mel:1999, nv:20399, bcc:884, scc:1070, bkl:753, ak:881, vasc:56 |
| ISIC_E_LBL | 11,384 | mel:90, nv:11052, bcc:234, bkl:8 |
| ISIC_E_POOL | 9,109 | mel:823, nv:3518, bcc:2325, df:352, bkl:1876, vasc:215 |
| DERMNET_P1 | 3,836 | mel:538, bkl:1673, ak:1047, vasc:578 |
| MPOX_HEALTHY | 2,660 | normal:2660 |
| ISIC_E_SHARD | 2,515 | mel:9, nv:1783, bcc:715, bkl:1, vasc:7 |
| PAD-UFES-20 | 2,261 | mel:52, nv:228, bcc:834, scc:188, bkl:232, ak:727 |
| DERM7PT | 1,950 | mel:501, nv:1133, bcc:84, df:40, bkl:136, vasc:56 |
| SKINTYPE_NORMAL | 1,156 | normal:1156 |
| SKINDISEASE_P1 | 780 | ak:780 |
| ISIC_E_MSK | 636 | mel:1, nv:634, bcc:1 |
| MED-NODE | 170 | mel:70, nv:100 |
| BCN20000_KAGGLE | 135 | vasc:135 |
| HAM10000_SUPP | 129 | vasc:129 |
| OUR_NORMAL_SKIN | 115 | normal:115 |
| ISIC_2019_KAGGLE | 102 | nv:4, vasc:98 |
| ISIC_MINORITY | 8 | vasc:8 |

## 4. Splits Integrity

- **External test** (held-out source domain): **51,035** rows
  - sources: ISIC_E_MSK, ISIC_2024_MSK
  - 8-class: mel:31, nv:50945, bcc:30, scc:14, bkl:12, ak:3
- **Train/val pool**: **197,129** rows over **5** folds

| fold | rows |
|------|------|
| 0 | 39,426 |
| 1 | 39,426 |
| 2 | 39,426 |
| 3 | 39,426 |
| 4 | 39,425 |

- **Group-leakage check** (group_id in >1 fold): **0** (PASS — zero leakage)

## 5. Class Balance Assessment

- min: **df** = 735
- max: **nv** = 203,206
- imbalance ratio (max/min): **276.5x**
- **3x rule** (no class > 3x smallest): **VIOLATES — needs sampler/cap (largest is 276.5x)**

## 6. Quality Audit

- **Near-duplicate groups**: 5,398 covering 14,851 IDs (9,453 dropped)
- **Quality-filter rejections**: 222
  - low_res_77x77: 45
  - low_res_79x79: 34
  - low_res_75x75: 28
  - low_res_73x73: 25
  - low_res_71x71: 20
  - tiny_file: 17
  - low_res_67x67: 13
  - low_res_69x69: 10
  - low_res_63x63: 10
  - low_res_65x65: 5
  - low_res_55x55: 4
  - low_res_59x59: 3
  - low_res_61x61: 3
  - low_res_57x57: 3
  - low_res_53x53: 1
  - low_res_80x51: 1
- **External test held out** (MSK domain): zero contamination by design

## 7. 100k Threshold

- **Total clean 8-class images**: 248,164
- **Train/val available**: 197,129
- **Threshold**: **MET** (>=100k)

---

## 8. Architecture & Training Plan

### Phase 1 — base 9-class model (mel/nv/bcc/scc/df/bkl/ak/vasc/normal)

**Backbone**: `vit_base_patch14_dinov2.lvd142m` (DINOv2-B/14, ImageNet-free SSL).
224 px input, fp16, batch=32, grad_accum=2 (effective 64). 86M params, ~3.0 GB VRAM verified.

**Multimodal fusion**: cross-attention between tabular metadata (age, sex, anatom site, Fitzpatrick, image_type) and image patch tokens. Pacheco 2021-style.

**Loss**: focal loss (γ=2) with per-class inverse-sqrt weights.
**Optimizer**: AdamW, LR 3e-4, weight decay 0.05, cosine schedule with 5-epoch warmup.
**EMA**: decay 0.9998 (free ~0.5-1% AUC).
**Sampler**: WeightedRandomSampler with sqrt-inverse-frequency.

**Uncertainty**: split-conformal prediction (Angelopoulos 2021), α=0.05 → 95% coverage guarantee.
10% of train_val held out as calibration set. Outputs prediction *sets* not point estimates.

### Phase 3 — Indian fine-tune (20-class)

Re-heads to 20-class on DermaCon-IN + DermNet + SkinDiseaseDataset Phase-3 subset.
Adds: vitiligo, tinea, acne, psoriasis, eczema, melasma, lichen-planus, pemphigus, steroid-damage, scabies, contact-dermatitis.
Knowledge distillation from Phase 1 teacher to prevent catastrophic forgetting.
DermaCon-IN held-out subjects = **PRIMARY** publication metric.

### Hardware
RTX 4050 6 GB. ~3.0 GB VRAM. 50 epochs ≈ 10-14 h wall-clock per fold.