# Can We Beat DDI AUROC 0.58 for Dark-Skin Cancer? Yes — By ~0.15–0.20 AUROC, and It's a Method Problem, Not a Data Ceiling

**Verdict up front:** Our 0.58 is **not** the ceiling. It is almost exactly the *unadapted, out-of-the-box* number that off-the-shelf models scored in the original DDI paper. Multiple published methods reach **0.73–0.90 AUROC on DDI, with Fitzpatrick V–VI matching or exceeding lighter skin (0.76–0.90)** — and they do it *without any new dark-skin dataset*, by fixing the backbone/representation. Our 5-fold CV stalling at 0.579 despite training on 80% of DDI is the tell: it points to a **domain-mismatched (dermoscopy/ImageNet) backbone**, not to DDI's cancers being unlearnable. The realistic target is **~0.73–0.78 on FST V–VI**. Do not chase 0.99 — that is a dermoscopy easy-case artifact.

## The ceiling question (most important)

The single most decisive fact: our 0.58 overall / 0.53 on V–VI reproduces the **zero-shot floor** from Daneshjou et al., *Science Advances* 2022 (the paper that created DDI). Their off-the-shelf numbers:

| Model (out-of-box) | DDI overall | FST I–II | FST V–VI |
|---|---|---|---|
| DeepDerm | 0.56 | 0.61 | **0.50** |
| ModelDerm | 0.65 | 0.64 | 0.55 |
| HAM10000 | 0.67 | 0.72 | 0.57 |

We are sitting *exactly* in this unadapted band. But the **same paper** then fine-tuned on DDI and the ceiling jumped, closing the skin-tone gap entirely:

| Model, fine-tuned on DDI | FST I–II | FST V–VI |
|---|---|---|
| DeepDerm → DDI | 0.73 | **0.76** |
| HAM10000 → DDI | 0.77 | **0.78** |

Independent follow-ups confirm the high ceiling on the *same* data:

| Method (year) | DDI result | FST V–VI | What it proves |
|---|---|---|---|
| **DDI-COCO fine-tune** (2024, Swin/EffNet/Inception) | AUROC **0.79–0.895** | **0.828 / 0.862 / 0.895 — higher than I–II** | Straight fine-tune with a good backbone clears 0.8 on dark skin |
| **FEDD** (MICCAI 2023, diffusion features + linear probe) | **81% malignancy acc at 10% of labels** | "fair across tones" | Wins in the *low-label* regime — our exact situation |
| **DermFM-Zero** (2026, foundation model) | melanoma AUROC **0.939** zero-shot | — | Foundation features are the lever |
| **DermDiff** (2025, synthetic augmentation *only*) | **0.58** | near chance | Synthetic data alone does **NOT** fix it — a warning |

**Conclusion:** 0.58 is the floor, not the ceiling. The realistically achievable target on FST V–VI is **~0.73–0.78** (with 0.8+ demonstrated by DDI-COCO). That is a **+0.15 to +0.20 AUROC** gain over where we are. The binding constraint is **representation/domain**, not data volume — proven by the fact that everyone reaching 0.9 reuses DDI/Fitz17k/PAD, adding no new dark-skin cancer set.

## New trainable datasets — ranked

Honest headline: **pickings are thin.** There is no large, public, biopsy-confirmed, *clinical-photo*, Fitzpatrick V–VI cancer dataset — DDI is essentially the only one, and we have it. The genuinely-new items below mostly add *mid-tone* or *paired-modality* clinical cancer signal, not V–VI. Register for the Stanford AIMI portal once (one Research Use Agreement unlocks DDI-2 + MRA-MIDAS).

| Dataset | Dark-skin cancer content | Modality | Access | Expected value | Pull now? |
|---|---|---|---|---|---|
| **MRA-MIDAS** (Stanford/Cleveland) | Biopsy-proven melanoma/BCC/SCC; but only ~16.8% FST III–VI, V–VI a small slice | **Paired clinical + dermoscopy** | Free RUA, Stanford AIMI portal | HIGH for clinical-photo domain + a clinical↔dermoscopy bridge; modest as a V–VI fix | **YES** |
| **DDI-2** (Stanford/Daneshjou, 2025) | 665 biopsy-proven clinical photos, self-identified **Asian** (FST III–IV heavy) | Clinical photo | Free RUA, same portal | MEDIUM: de-risks the mid-tone band + clean external test; **not** V–VI Black-skin | **YES** |
| **ASAN Test set** (Korea) | Clinical BCC/SCC/IEC/melanoma, Asian skin (FST III–IV); per-tumor AUROC BCC 0.96, mel 0.96 | Clinical photo | Public test split (~1,276 imgs); verify link | MEDIUM–HIGH: rare *clinical-photo* malignancy in pigmented skin | **YES** (verify license) |
| **HIBA** (Argentina, via ISIC) | 253 melanomas, **346 clinical**, Latino skin (FST III–IV) | Clinical + dermoscopy | ISIC collection 251, CC BY | LOW–MEDIUM: small but clean, easy fold-in | Optional |
| **DERM12345** (Türkiye) | Labeled **acral** melanoma subtypes (ALM/ANM) — over-represented in dark skin | **Dermoscopy only** | Harvard Dataverse, CC BY | LOW for our gap (wrong modality); acral-head/pretraining only | No |
| **SD-198 / SD-260** | Some malignant classes, **not biopsy-confirmed**, light-skewed | Clinical photo | Email maintainers | LOW: clinical-photo pretraining variety only | No |
| **SCIN cancer subset** (have it) | ~1.35% malignant, mostly differential (not biopsy) labels; good Monk/Fitz tones | Clinical photo | Have it | LOW: re-mine confirmed dark-tone malignant cases; fairness eval | Re-audit |
| **PASSION / DermaCon-IN / SCIN** (have) | ~ZERO cancer (inflammatory) | Clinical photo | Have it | Dark-skin **domain pretraining / SSL** only — NOT malignancy | Use for pretrain |
| **Pipsqueak** (2026) | 16 real dark-skin melanoma images (aug→80) | Clinical | Author request | **EVAL PROBE ONLY** — too tiny to train | Eval only |
| **African albino SCC series / RDTC Moshi teledermatology** | Real FST V–VI biopsy-confirmed cancer | Clinical/histopath | **Not a dataset** — papers only; needs IRB/collaboration | Aspirational; the only route to true V–VI, but no download | Outreach only |

**Bottom line on data:** the highest-value *downloadable* additions are **MRA-MIDAS + DDI-2 + ASAN** — all obtainable by an individual, all adding confirmed *clinical-photo* cancer, but all skewing III–IV rather than V–VI. They will strengthen the clinical-photo decision boundary broadly; they will not by themselves saturate the darkest end. Genuinely new V–VI cancer imagery essentially does not exist as public data.

## Methods to try (ranked by expected DDI gain)

### Tier 1 — proven to move the DDI dark-skin number (do first)

1. **Swap the dermoscopy/ImageNet backbone for a clinical-photo dermatology foundation model.** This is the single biggest lever and the direct explanation for our 0.99-vs-0.58 gap.
   - **PanDerm** (Nature Medicine 2025) — pretrained on 2M+ images incl. clinical photos; +9.0% over next-best on the clinical PAD dataset. Public ViT-B weights (`panderm_bb_data6_checkpoint-499.pth`) + fine-tune scripts. **License: CC-BY-NC-ND 4.0 (non-commercial)** — flag for a product.
   - **DermINO** (2025) — the one paper explicitly reporting *flat* performance across FST I–VI. Check repo for a checkpoint.
   - **DermFM-Zero / PanDerm-2** (Feb 2026) — melanoma AUROC 0.939 zero-shot.
   - Recipe: **linear probe or light fine-tune** the foundation backbone on DDI+PAD (+MIDAS/DDI-2). Expected: the largest jump, from 0.58 toward ~0.7+.
   - **Compute:** feature extraction + linear probe fits a **6GB laptop**; full fine-tune of a ViT-B wants a **rented GPU** (a few hours on one A100/3090).

2. **Reproduce the DDI-paper two-stage fine-tune as a controlled target.** Clinical-photo foundation → HAM/DeepDerm clinical → **DDI last**. This is literally the recipe that produced 0.76–0.78 on V–VI. **DDI-only 5-fold fine-tune** is your reproduction gate: if a proper clinical backbone can't clear ~0.73, the bug is in the pipeline (leakage, frozen features, or preprocessing), not the data. GPU for fine-tune; probe fits laptop.

3. **Shades-of-Gray color constancy on every clinical image (train + test).** Cheap, standard, removes the illumination/white-balance confounder that plagues clinical photos. Ready code exists. Normalizes *illumination*, not pigment — small but free gain, stacks with everything. **6GB laptop, trivial.**

### Tier 2 — real but smaller / gap-closing rather than ceiling-raising

4. **FEDD-style diffusion-feature + linear probe** — MICCAI-proven on DDI, best in the *low-label* regime we're in (81% acc at 10% labels), fair across tones. Good fit if foundation-model fine-tuning underdelivers. **Feature extraction on GPU once, probe on laptop.**
5. **Licensed diffusion synthetic augmentation of FST V–VI malignant lesions** ("From Majority to Minority", LDM+LoRA on Fitzpatrick17k: 57.7%→61.0% with zero real dark data; 62%→71% with limited). Single-digit lifts, mostly *gap-closing*. **Warning: DermDiff shows augmentation *alone* stays at 0.58** — use as a stacking add-on, never the primary fix. Diffusion training needs a **rented GPU**; check base-model license for a product.
6. **Adversarial skin-tone unlearning / group-adaptive BatchNorm** — drove the light/dark AUC gap 0.07→0.01, but demonstrated on *dermoscopy*, and it equalizes subgroups rather than lifting a near-chance model. Apply *after* Tier 1 to close residual V–VI gap. **GPU.**
7. **Diffusion-driven test-time adaptation** — projects each clinical test image back toward the source domain at inference; conceptually matched to our train-dermoscopy/test-clinical shift, no DDI labels needed. Promising but no DDI numbers published. **Spike; GPU.**

### Tier 3 — will NOT fix our specific gap (be honest)

- **Pure synthetic/GAN data augmentation as a primary fix** (DermDiff = 0.58).
- **Few-shot / metric learning** — plausible but *no published DDI dark-skin lift* exists; speculative.
- **Class reweighting / balanced-softmax / curriculum** — addresses imbalance, not the clinical-photo *feature* gap; we already saw double-balancing cause collapse.

## Recommended plan

**This is a beatable method gap, not a data ceiling. Target ~0.73–0.78 on FST V–VI (+0.15–0.20 AUROC). Do NOT expect or chase 0.99.**

**Step 0 — Diagnose the pipeline first (½ day, laptop).** Before adding anything, confirm no patient leakage across folds, that features aren't frozen-dermoscopy, and that clinical preprocessing exists. Our 0.579 CV-on-80%-of-DDI is a red flag for a pipeline bug given the paper hits 0.76 on the same data.

**Step 1 — Foundation backbone + linear probe (1–2 days; extract on GPU once, probe on laptop).** Extract **PanDerm** (and/or **DermINO**) features for DDI+PAD+Fitz-clean, fit a class-balanced logistic/linear probe, report per-Fitzpatrick AUROC with CIs. Add **Shades-of-Gray** preprocessing. *Realistic expected: 0.65–0.75 overall; this is the make-or-break experiment.*

**Step 2 — Two-stage fine-tune, DDI last (2–3 days, rented GPU).** Foundation → clinical (HAM/PAD) → DDI. Gate: **≥0.73 on a DDI-only 5-fold**. *Realistic expected V–VI: 0.73–0.78.*

**Step 3 — Fold in the new confirmed clinical-photo cancer (parallel).** Register Stanford AIMI; add **MRA-MIDAS** (paired) + **DDI-2** + **ASAN** to the fine-tune pool. *Expected: broadens mid-tone robustness; small additional V–VI lift.*

**Step 4 — Only if Steps 1–2 plateau below target:** layer **FEDD diffusion features** (low-label regime) and **licensed dark-skin synthetic augmentation** + **adversarial tone-unlearning** to close the residual gap. *Expected: gap-closing, not ceiling-raising.*

**Parallel, slow track:** pursue an **RDTC Moshi / South African tertiary-clinic data-use agreement** — the only realistic source of genuinely-new FST V–VI biopsy-confirmed clinical cancer. Not a download; a collaboration.

**Honest ceiling statement:** even a perfect method plateaus near **~0.78 on FST V–VI** on DDI today — that is the published high-water mark, set by the scarcity of dark-skin cancer *and* the intrinsic difficulty (mycosis fungoides, in-situ melanoma/SCC). So: **beatable to ~0.76–0.78, not to 0.9+ on the darkest skin, and not near 0.99.** Pair the AUROC push with **calibration + selective prediction/abstention on FST V–VI** so the deployed product is safe where it is uncertain.

## Sources

- DDI original + fine-tuning: https://www.science.org/doi/10.1126/sciadv.abq6147 · https://pmc.ncbi.nlm.nih.gov/articles/PMC9374341/ · https://arxiv.org/abs/2203.08807
- PanDerm: https://www.nature.com/articles/s41591-025-03747-y · https://github.com/SiyuanYan1/PanDerm · https://arxiv.org/html/2410.15038v3
- DermINO: https://arxiv.org/abs/2508.12190 · DermFM-Zero: https://arxiv.org/html/2602.10624v1
- DDI-COCO fine-tune: https://arxiv.org/html/2401.13280v1 · FEDD: https://arxiv.org/abs/2307.11654 · DermDiff: https://arxiv.org/html/2503.17536v1
- Equitable Skin (domain adaptation): https://arxiv.org/html/2409.00873v1 · Majority-to-Minority synth: https://arxiv.org/html/2406.18375v1 · Diffusion TTA: https://arxiv.org/abs/2405.11289
- MRA-MIDAS: https://aimi.stanford.edu/datasets/mra-midas-Multimodal-Image-Dataset-for-AI-based-Skin-Cancer · https://ai.nejm.org/doi/full/10.1056/AIdbp2400732
- DDI-2: https://daneshjoulab.github.io/ddi2-dataset/ · https://pubmed.ncbi.nlm.nih.gov/39481531/
- ASAN: https://www.sciencedirect.com/science/article/pii/S0022202X18301118 · HIBA: https://www.nature.com/articles/s41597-023-02630-0 · https://api.isic-archive.com/collections/251/ · DERM12345: https://www.nature.com/articles/s41597-024-04104-3
- Shades-of-Gray code: https://github.com/nickshawn/Shades_of_Gray-color_constancy_transformation · DDI-CoCo (contrast): https://arxiv.org/pdf/2401.13280
- Pipsqueak (eval probe): https://www.sciencedirect.com/science/article/abs/pii/S1877750326000761 · SA biopsy series: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7204328/