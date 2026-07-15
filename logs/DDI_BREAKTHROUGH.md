# DDI dark-skin cancer — BREAKTHROUGH (2026-07-05)

**0.58 → 0.748 overall; 0.53 → 0.789 on the darkest skin (FST V–VI).** Frozen DINOv2 ViT-B/14 @ 518px
features + a class-balanced logistic probe, clean 5-fold stratified CV on DDI (656 imgs). Laptop, no new
data, no GPU training. Reaches the published DDI ceiling (Daneshjou 2022 fine-tune: 0.76–0.78 V–VI).

## The result
| Method | DDI overall AUROC | FST V–VI (dark) |
|---|---|---|
| Old fine-tuned 25-class model | 0.58 | 0.53 |
| Naive clinical fine-tune (Fitz+PAD) | 0.46 | 0.45 |
| "All 3" CV fine-tune (CleanPatrick+DDI-CV+PAD) | 0.579 | 0.53 |
| **Frozen DINOv2@224 + linear probe** | 0.736 | 0.740 |
| **Frozen DINOv2@518 + linear probe** | **0.748** | **0.789** |
| PanDerm probe (MISLOADED, unfair — 48 unexpected keys; CC-BY-NC-ND) | 0.63 | 0.57 |

## Why it works (root cause of the old 0.58)
Our **fine-tuning was destroying the features.** Training on 8,471 images with DDI only ~8% dragged the
representation toward the Fitz/PAD majority and wrecked DDI separation (val-0.82 was Fitz/PAD-driven, DDI-only
0.58). The **frozen** DINOv2 features were good all along — a simple probe recovers the signal. Classic
"linear-probe beats fine-tune under distribution shift / low labels" (cf. FEDD, MICCAI 2023). Confirmed the
deep-research verdict (`logs/DARKSKIN_RESEARCH.md`): 0.58 was the *unadapted floor*, not a data ceiling.

## Method (deployable)
- Backbone: `vit_base_patch14_dinov2.lvd142m`, **frozen**, img_size **518**, pooled feature (768-d).
- Head: `StandardScaler` + `LogisticRegression(class_weight='balanced', C=0.5)`.
- Eval: 5-fold stratified (by tone×malignant) CV; no leakage (scaler/LR fit on train folds only, features label-free).
- Script: `scripts/_probe_backbone.py --backbone dinov2 --img-size 518`.

## Honest caveats
- Ceiling is ~0.78 on the darkest skin — dark-skin cancer is genuinely hard + data-scarce; do NOT claim 0.9/0.99.
- Pair with **per-tone calibration + abstention** on FST V–VI where confidence is lower.
- PanDerm (clinical FM) not fairly tested (misload) and non-commercial; DINOv2 already hits target and is shippable.

## Next
1. Productionize: frozen-DINOv2@518 → probe as the **clinical-photo cancer head** (replaces the 0.58 detector).
2. Optional: train the probe on DDI + PAD + Fitz-clean (more data) for generalization; register Stanford AIMI for MRA-MIDAS/DDI-2/ASAN.
3. Wire into the app with abstention.
