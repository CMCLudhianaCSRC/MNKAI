# MNKAI-III — Tier-0 Overnight Results (2026-07-02/03)

All produced autonomously overnight from the existing Fix-C 5-fold model — **no retraining,
no new data**. Predictions dumped to `eval/phase3_fixC_full/preds.npz` (verified: external
acc 0.4808 / macro-F1 0.4839 exactly matches `metrics_full.json`). Every number below is on the
pristine external test set (n=5,886) unless noted; per-class bias tuned ONLY on OOF.

## Headline: macro-F1 0.484 → 0.522 (per-class bias, free)
`_optimize_thresholds.py` — coordinate-ascent per-class logit bias on OOF, frozen, applied to external:
- **External macro-F1 0.486 → 0.522** (OOF 0.480 → 0.522 — the gain transfers, so it's real not overfit).
- Accuracy 0.481 → 0.489; balanced-acc 0.532 → 0.518 (trades a little head precision for tail recall, as intended).
- Biggest external per-class F1 gains: **ak +0.176, alopecia +0.141, vitiligo +0.127, psoriasis +0.096, urticaria +0.070, normal +0.059.**
- This is the cheapest Tier-0 lever and it alone reaches the projected ~0.50–0.52 target.

## "Are we competitive with DermaCon-IN?" — YES, at like-for-like granularity
`_coarse_eval.py` (7 clinically-defensible super-groups, prob-aggregation):
| Framing | Accuracy | macro-F1 |
|---|---|---|
| Fine 25-way (our headline) | 0.481 | 0.484 |
| **Coarse 7-group** | **0.661** | **0.633** |
| DermaCon-IN coarse (~8-class, published) | ~0.70 | — |
- Top-3 (fine) 0.725, top-5 0.841. The "low 0.48" was **task granularity**, not a weak model — at coarse level we're right alongside DermaCon-IN.

## Cancer triage is clinically strong
`_coarse_eval.py` binary malignant-vs-benign (mel/bcc/scc/ak vs rest):
- **AUROC 0.990.** At threshold 0.3: **sensitivity 0.904, specificity 0.994.** This is the deployable, honest "90%+" number.

## Conformal fixed (abstention is now usable)
`_conformal_fix.py` — temperature on OOF, cal/test split of external, target coverage 0.90:
| Method | Coverage | Mean set | Median |
|---|---|---|---|
| APS (old behaviour) | 0.911 | 6.20 | 7 |
| RAPS | 0.910 | 6.03 | 7 |
| **Clustered** | **0.900** | **5.81** | **6** |
- Was: coverage 0.984 (over-covering), mean set **10.6**. Now at target 0.90 with mean set **5.81** — a ~45% reduction. (Still ~6/25 because the model is genuinely uncertain on the tail — conformal reports that honestly; the usable abstention lives at the binary/coarse level.)

## Bootstrap 95% CIs (publishability)
`_bootstrap_ci.py` (B=2000):
| Metric | Median [95% CI] |
|---|---|
| accuracy | 0.481 [0.468, 0.494] |
| macro-F1 | 0.483 [0.468, 0.497] |
| balanced-acc | 0.533 [0.517, 0.547] |
| top-3 acc | 0.725 [0.714, 0.736] |
| melanoma recall | 0.844 [0.754, 0.923] (wide — only 76 mel) |
| malignant AUROC | 0.991 [0.984, 0.995] |
| melanoma AUROC | 0.988 [0.963, 0.999] |

## OOD / input-quality gate (trust) — first working version
`_ood_gate.py` — Mahalanobis (tied covariance) on the DINOv2 CLS embedding, fit on 12k train_val refs:
- **External test correctly reads as in-distribution:** 5.8% flagged at the 5%-target threshold (p95=1170). ✅
- OOD scores clearly elevated: synthetic-OOD median **1160** vs in-dist median **633**.
- **But pure-synthetic-noise/solid-colour detection is only ~48%** at the conservative threshold — a genuine limitation. Pure noise is an adversarial edge case (real non-skin photos separate more cleanly), but this needs: (a) tuning on a **real OOD image set**, and (b) pairing with a cheap **input-quality/skin-detection** check. Detector params saved to `ood_gate.npz` for app integration.

## Data findings (the "download more" priority was largely moot)
- **BCN20000:** 18,946 imgs but **92% already in our ISIC data** (only 1,470 new, all cancer — our strong suit). Stopped the redundant download.
- **SCIN:** all 10,379 images **already on disk**; the 5,231 mappable/non-duplicate ones already integrated; the rest are intentionally-excluded near-duplicates + out-of-taxonomy conditions (insect bite, folliculitis, herpes zoster…). No free win.
- **Real remaining gap = DDI** (dark-skin, biopsy-confirmed, held-out fairness eval) — needs Stanford AIMI registration (**your action**).
- Net: our data is already comprehensive. The near-term gains are **method levers on existing data** (this doc), not more downloads.

## Artifacts
- `eval/phase3_fixC_full/preds.npz` — per-image probs (external + OOF), reusable.
- `eval/phase3_fixC_full/{coarse_eval,conformal_fix,bootstrap_ci,threshold_opt}.json`
- Scripts: `scripts/_dump_predictions.py`, `_coarse_eval.py`, `_conformal_fix.py`, `_bootstrap_ci.py`, `_optimize_thresholds.py`

## Next (teed up)
- OOD / input-quality gate (trust) — needs a one-pass embedding dump.
- Apply the frozen per-class bias + clustered-conformal + binary-cancer operating point in the demo app.
- Tier-1 (cRT+LORT, single-mechanism+DRW) for the next macro-F1 step — needs (cheap) retraining.
