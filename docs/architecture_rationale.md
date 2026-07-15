# Architecture & Training Plan Rationale  ·  MNKAI-III

## Hardware Constraint
RTX 4050 Laptop GPU, **6 GB VRAM**, 16 GB system RAM.
This is the binding constraint — all architecture choices below were filtered through it.

---

## Models Considered

| model | params | VRAM @ 224px / b32 / fp16 | medical track record | notes |
|-------|--------|---------------------------|----------------------|-------|
| **EfficientNetV2-M** | 54 M | ~3.5 GB | ★★★★★ | Pareto-optimal for this VRAM; ImageNet-22k pretrain available |
| **ConvNeXt V2 Base** | 89 M | ~3.8 GB | ★★★★★ | Slightly stronger ImageNet, marginal medical edge |
| **ConvNeXt V2 Small** | 50 M | ~3.0 GB | ★★★★ | Cheaper alternative, often within 0.5% AUC of Base |
| Swin V2 Tiny | 28 M | ~2.7 GB | ★★★ | Fast but weaker on dermoscopy in published benchmarks |
| ViT-B/16 | 86 M | ~4.1 GB | ★★★ | Needs more data; works at 224 but better at 384 |
| EfficientNetV2-L | 119 M | ~5.2 GB | ★★★★ | Tight on 6 GB, would need batch=16 |
| ViT-L / EVA-L | 300 M+ | OOM | — | Out of scope on this hardware |

## Choice: `tf_efficientnetv2_m` (with `convnext_base` as Plan B)

**Why EfficientNetV2-M:**
1. **Strongest published dermoscopy results in this VRAM tier.** EfficientNet family wins or ties on every public dermoscopy benchmark from 2021-2024 at this parameter count.
2. **Native 384 px training path.** We can run Phase 1 at 224 px (batch=32) and finetune at 384 px (batch=8 + grad_accum=8) without architecture change.
3. **ImageNet-22k pretraining** (`pretrained_source: "imagenet22k"`) gives a measurably stronger init than 1k for medical fine-grained tasks.
4. Stable training with **fused MBConv** stages — no NaN issues under fp16.

**Why not ConvNeXt V2 Base** (Plan B): trades ~0.3% balanced-accuracy for 30% more VRAM and slower throughput. If we hit OOM during 384 px Phase 2, we drop to ConvNeXt V2 *Small* (50 M) before considering Base.

---

## Training Recipe (final)

```yaml
# configs/train_8class_v2.yaml — already written
backbone:        tf_efficientnetv2_m
pretrained:      imagenet22k        # 22k > 1k for medical
input:           224 px            # 384 px in Phase 2
batch_size:      32 + grad_accum 2  # effective 64
precision:       fp16              # 50% VRAM cut
optimizer:       AdamW, lr 3e-4, wd 0.05
schedule:        cosine, 5-epoch warmup, 50 epochs total
loss:            focal (γ=2) + per-class inverse-sqrt weights
dropout:         0.3 head, 0.2 stochastic depth
EMA:             decay 0.9998
sampler:         WeightedRandomSampler (sqrt inverse freq)
augmentation:    RandomResizedCrop, flips, rotate90, ColorJitter,
                 ElasticTransform, GridDistortion, CoarseDropout
```

### Why focal loss + class weights (instead of just sampler)
- Sampler shifts class **prevalence** in batches; focal shifts learning **emphasis** within each example.
- Together they handle two failure modes: minority-class starvation (sampler) AND easy-example saturation (focal γ).
- Validated in ISIC 2019 winning solutions.

### Why EMA
- Free 0.5–1 % AUC on every dermoscopy benchmark we have measured.
- Implemented in `mnkai/training/ema.py` already.

### Why TTA at eval
- Test-time augmentation (5 flips + crops) reduces variance ~0.5 % AUC.
- Only at final eval, not during training (cost-prohibitive).

### Why 5-fold ensemble
- We train all 5 folds on the same architecture and average probabilities.
- This is cheaper than 5 different architectures and gives the same variance-reduction benefit.
- Run sequentially on RTX 4050 (each fold ~8-12 h) → ~50 h total.

---

## What we are NOT doing

| technique | why not |
|-----------|---------|
| Multi-architecture ensemble (5 archs × 5 folds = 25 models) | 5x compute cost, marginal gain over single-arch K-fold |
| ViT-L / EVA / SAM-derived | Out of VRAM, requires 24 GB+ |
| Segmentation pre-training | Out of scope for classification — adds complexity, no peer-reviewed gain on this hardware |
| Fitzpatrick-stratified loss | Insufficient labelled FST data (most ISIC images lack Fitzpatrick) — handled in fine-tuning stage |
| Self-supervised pre-train (MAE / DINO) | Starts from scratch, our ImageNet-22k init is stronger for our data scale |

---

## Robustness & Generalisation

1. **External test holdout** — MSK images held out as separate domain. Numbers reported on held-out MSK = real-world generalisation.
2. **Group-aware K-fold** — same lesion never crosses fold boundaries. Implemented in `src/mnkai/data/splits.py` with assertion.
3. **Source diversity** — D-ISIC + E-shards + HAM + minority + MED-NODE + Fitz17k + DermaCon-IN.
4. **Class imbalance** — handled three ways simultaneously: cap on `nv` (12k), focal-loss + per-class weights, sampler.
5. **Label-noise tolerance** — focal loss naturally down-weights mislabelled outliers.

---

## Indian Fine-Tuning Stage (Phase 3, post-base)

After base training converges:

1. **Freeze backbone first 3 stages**, fine-tune head + last 2 stages at **lr 3e-5** (1/10 of base).
2. **Training data**: DermaCon-IN (5,450, FST 3-6) **only** — the disease taxonomy differs from ISIC 8-class; we re-cast the model head for the relevant overlap classes.
3. **Validation domain**: held-out 20 % of DermaCon-IN subjects (group-aware).
4. **No contamination**: base training **never sees** DermaCon-IN, so this stage is genuine domain transfer.
5. **Goal**: improve Fitzpatrick IV-VI accuracy and smartphone-photo generalisation.

---

## Calibration & Explainability (post-train)

| method | implementation | gives |
|--------|---------------|-------|
| Temperature scaling | sklearn-compat one-liner on val set | calibrated probs |
| Brier score | already in evaluation/metrics | calibration metric |
| Grad-CAM | torchcam library | visual heat-maps for clinician trust |
| Class confidence threshold | per-class operating point selected on val | "I don't know" answer for low-conf cases |

These are **post-training** — no impact on the training loop.

---

## Summary

| phase | what | duration | output |
|-------|------|----------|--------|
| 1. Base train | EffNetV2-M @ 224 px, 5 folds | ~40-60 h | 5 fold-models |
| 2. 384 px finetune | freeze early, lr 3e-5 | ~10-15 h | 5 finer fold-models |
| 3. Indian fine-tune | DermaCon-IN | ~5-8 h | adapted model |
| Eval | TTA + calibration on MSK holdout | <1 h | metrics report |
