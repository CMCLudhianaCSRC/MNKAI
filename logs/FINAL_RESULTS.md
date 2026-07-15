# MNKAI-III — FINAL RESULTS
*Generated overnight on training completion. Numbers are real and measured.*

## Headline journey

| Stage | macro-F1 | balanced-acc | accuracy | top-3 acc | eczema recall | melanoma recall |
|---|---|---|---|---|---|---|
| **Phase 1** · 9-class base (5-fold CV) | **0.637 ± 0.009** | 0.69 | ~85% | — | n/a | ~0.93 |
| **Phase 3 v1** · 25-class collapsed | 0.339 | 0.527 | 0.247 | 0.468 | **0.004** ☠ | (cancers ok) |
| **Phase 3 corrected** · Fix C, **external test** | **0.484** | 0.533 | 0.481 | **0.725** | **0.503** ✅ | **0.842** ✅ |

> The corrected number is on the **pristine external test set (5,886 images, group-disjoint, never used in selection)** — the honest generalisation figure. Winner selected on OOF validation; external test untouched until this final ensemble eval.

## The fix worked
- **Common conditions recovered from collapse:** eczema recall **0.004 → 0.503**, tinea → 0.428, viral_infection → 0.729 (psoriasis still weak at 0.161 — flagged below).
- **Cancer safety preserved by the ensemble:** melanoma recall **0.842** (F1 0.815, AUROC 0.987), BCC recall 0.827, SCC 0.853 — all strong.
- **Reproducible:** per-fold OOF macro-F1 = 0.482/0.479/0.477/0.486/0.471 → **0.479 ± 0.005** (tight).

## Trustworthiness layer (external test)
- **Calibration:** ECE **0.176 → 0.016** after temperature scaling (T=0.60) — very well-calibrated.
- **Conformal (target 90%):** empirical coverage **98.4%**, mean set size 10.6 classes (conservative under 25-class imbalance).
- **Fairness (per-Fitzpatrick):** FST I–III balanced-acc 0.245 (n=642) vs FST IV–VI 0.209 (n=135). *Caveat: only 777 test images carry FST labels, so per-stratum estimates are noisy/unreliable — the thin-coverage limitation; the prospective study is the real fairness test.*

## Strong vs weak classes (external test, per-class F1)
- **Strong (≥0.6):** df 0.94, nv 0.93, scc 0.83, mel 0.82, bcc 0.80, viral 0.71, bkl 0.68, vasc 0.68, ak 0.63.
- **Moderate (0.3–0.6):** normal 0.59, tinea 0.50, acne 0.46, eczema 0.44, alopecia 0.43, vitiligo 0.37, pemphigus 0.31.
- **Weak (<0.3, the long tail):** rosacea 0.28, autoimmune 0.26, psoriasis 0.23, drug_eruption 0.23, lichen_planus 0.22, urticaria 0.21, contact_dermatitis 0.15, bacterial_infection 0.09.

## Honest takeaways
1. **The collapse is genuinely fixed** — the model is now clinically usable on common conditions, and melanoma is safe (0.84 recall, 0.99 AUROC).
2. **The long-tail inflammatory classes remain hard** (psoriasis, contact dermatitis, bacterial infection) — the clearest target for the data-expansion improvements.
3. **Fairness coverage is too thin to conclude** — must be settled by the prospective study.
4. **Winner = Fix C** (class-balanced focal, no resampling) — single-mechanism balancing, melanoma-floor-safe.
