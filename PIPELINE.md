# MNKAI-III — Pipeline v2 (trust-first, publication-grade)

Supersedes v1. Folds in everything from the planning discussion: the honest accuracy ceiling, the
synthetic-data verdict, the trust layer (OOD/quality/abstention), the real data inventory, and the
publishability guardrails. See `IMPROVEMENT_ROADMAP.md` for the underlying research, `DATASETS.md` for
acquisition details.

**Golden rules (unchanged):**
- External test set stays **pristine** — touched once per *finalized* tier, never for tuning.
- After any model-touching step: re-fit temperature + conformal, re-confirm melanoma recall ≥ 0.90.
- **One** rebalancing mechanism at a time (stacking a sampler on a reweighted loss was the collapse).

---

## 0. Honest performance reality (read this before setting targets)

- **25-class top-1 will not legitimately reach 90%.** Comparable honest work: Google's 26-condition system top-1 ~56–63%; DermaCon-IN **coarse ~8-class** ~70%; SD-198 (198-class) ~50–67%. The 85–93% on HAM/ISIC is 7–8 classes, dermoscopy, and usually leaky splits. Our realistic 25-class ceiling with real data + ensembles is **~60–70% top-1 (macro-F1 ~0.55–0.66)**.
- **Why we look "lower" than DermaCon-IN on the same data:** we report **25 fine classes** (they headline **~8 coarse**), **macro-F1** (they report accuracy), and **patient-disjoint** eval (leaky splits score higher). Adding more *classes* and more *heterogeneous sources* makes top-1 drop even with more images — that's task difficulty, not a weaker model. **Report coarse-to-coarse and we're competitive.**
- **To honestly show a 90% number, change the question, not the data:** binary **malignant-vs-benign** triage (cancers already AUROC 0.97–0.998), **~8 coarse super-classes**, or **top-3 differential** (0.725 now → ~0.85–0.90 with data). Those 90%s are real.
- **Synthetic data: do NOT use it to inflate the headline.** A generator trained on 161 bacterial images cannot invent real disease diversity — best case +a few tail-F1 points, and any large top-1 jump would come from **leakage**, which is the exact thing that makes a paper unpublishable/retractable. If used at all: **train-only, leak-free, ablated, validated on real external data**, reported honestly. Keep the no-synthetic-in-eval stance — it's a credibility asset.

---

## 1. Data — what we already have vs the gaps

**Already downloaded (D:\AIFINAL\images, ~271k rows in `data/manifests/master.csv`):**
ISIC-2024 ~157k + other ISIC pools; **DermaCon-IN full 5,450**; Fitzpatrick17k 5,148; PAD-UFES-20 2,298; Derm7pt 1,974; **SCIN partial 5,231**; large Mpox/HFMD/measles/chickenpox sets; DermNet + SkinDisease Kaggle; HAM10000 metadata (+ images via ISIC archive). **So HAM/ISIC are already here — no re-download.**

**Gaps worth acquiring (see `DATASETS.md` for exact commands):**
| Dataset | Have | Why | Use |
|---|---|---|---|
| **BCN20000 full** (~19k dermoscopy) | 151 | strengthens cancer pathway | train (dedup vs ISIC) |
| **DDI** (~656, dark-skin, biopsy) | 0 | 2nd external + fairness power | **held-out eval only** |
| **SCIN full** (~10.4k) | 5,231 | inflammatory tail + skin tone | train |
| **CleanPatrick** (Fitz17k-C cleaned labels) | raw only | raw is 16–30% mislabeled | relabel existing |

**The honest caveat:** most big open sets are cancer/dermoscopy; our weak classes are **inflammatory/infectious clinical photos** (bacterial 8%, contact-derm 17%, psoriasis 16%). SCIN + CleanPatrick + any new Indian clinical set help the tail; more melanoma data does not.

---

## 2. Do we lose progress with a new model? — No.

Model-agnostic assets carry across every backbone swap: data + group-disjoint splits, external test, the recipe (collapse-fix, single-mechanism balancing, EMA, MixUp/CutMix), the fusion architecture, and the eval/calibration/conformal/fairness harness. A backbone swap = **re-run fine-tuning (hours, automated)**, not start over. Additive upgrades that lose nothing: LIFT adapters, Derm-Foundation 2nd stream, KD, thresholds, conformal, cascade, model soups.

**Weight transfer:** DINOv2 → DINOv2-registers (both ViT-B dim 768) → fusion+head are shape-compatible, warm-start + cRT-retrain (reuse `freeze_backbone_epochs` + head-reinit). ViT-L/DINOv3 (dim 1024) → only fusion input-proj + head reinit; everything else kept.

---

## 3. Model choice + GPU

**Backbone path:** ① DINOv2-with-registers ViT-B/14 (same 6 GB, cleaner patch tokens) → ② LIFT adapters → ③ Tier-3: continued SSL on own images / DINOv3 base.

| Work | 6 GB laptop? | Rent? |
|---|---|---|
| Tier 0 (trust layer, soups, conformal, thresholds, eval) | ✅ | No |
| Tier 1 (cRT+LORT, DRW, hierarchical, registers, LIFT, aug) | ✅ | Optional (speed) |
| Tier 2 (data expansion + retrain) | ✅ slower | 👍 days→hours |
| Tier 3 (SSL pretrain, ViT-L, res 384/512) | ⚠️ | ✅ **only place you need it** |

**Rent specs:** A100 40 GB (Colab Pro+) is the sweet spot; 24 GB (RTX 4090/L4/A10) is a big cheap jump for Tier 2. Colab times out + ephemeral disk → checkpoint to Drive each epoch; for multi-day SSL use a persistent VM (Lambda/RunPod/Vast.ai). **You can reach ~0.55 macro-F1 on the laptop; rental only matters at Tier 3.**

---

## 4. Execution — TRUST-FIRST ordering

### TIER 0 — Trust layer + near-free wins (laptop, ~1–2 weeks)  →  macro-F1 ~0.50–0.52, and a *trustworthy* app
1. **Greedy model soup / SWA** over the 5 Fix-C folds — zero training, more stable predictions, attacks the 0.637→0.484 drop. *(`_swa_average.py`)*
2. **OOD gate** — Mahalanobis/energy score on frozen DINOv2 embeddings → reject non-skin / unlike-training input ("I won't guess"). **The #1 trustworthiness fix.**
3. **Input-quality gate** — blur / exposure / lesion-centering checks → "retake the photo."
4. **Fix conformal** — RAPS + clustered conformal + per-class temperature → set size 10.6 → ~3–5 at 90% coverage, so **abstention actually fires** ("refer to dermatologist"). *(`_posthoc_calibrate.py`, `calibrate.py`)*
5. **Per-class thresholds on OOF** (temp-scaled) to maximize macro-F1. *(`_screening_thresholds.py`, `find_threshold.py`)*
6. **Lock melanoma / malignant operating point** at recall ≥ 0.90; report sens/spec/PPV/NPV + over-referral + ISIC pAUC@80%TPR.
7. **Coarse + binary + top-3 reporting** — add an 8-superclass mapping and a malignant-vs-benign readout; these are the honest "high-accuracy" numbers and the DermaCon-IN-comparable figures. *(`_stratified_eval.py`)*
8. **Robustness/corruption sweep** — prove test-retest stability. *(`_robustness_sweep.py`, `corruption_sweep.py`)*
9. **App**: top-3 differential as primary UI + abstention banner + OOD/quality rejection + cancer flag.
   - **Gate:** one external-test shot → record Tier-0 numbers with bootstrap 95% CIs.

### TIER 1 — Method levers, no new data (laptop, ~3–6 weeks)  →  ~0.51–0.54
10. **Single-mechanism + DRW** under the ablation driver (ONE of balanced-softmax / cb_focal; `sampler_power`≈0–0.3; deferred reweighting). *(A6)*
11. **Decoupled cRT + LORT** — freeze backbone+fusion, re-init + retrain head, class-balanced sampling + heavy-negative label smoothing. *(A1)*
12. **Hierarchical coarse→fine loss** — super-class head + weighted aux loss (also powers the coarse readout). *(A5)*
13. **Backbone swap → DINOv2-with-registers** (warm-start fusion+head, cRT-retrain). *(B3)*
14. **LIFT adapters** — freeze backbone, <1% adapters + cosine classifier + logit-adjusted loss. *(B2)*
15. **Derm augmentation** — Shades-of-Gray color constancy + hair overlay/DullRazor + TrivialAugment + EMA + tuned label smoothing. *(B2)*
16. **Cancer-gate cascade** — malignant-vs-benign gate → fine head; decouples melanoma safety from the tail. *(B7)*
17. **TTA** (≤4 label-preserving views, gated on OOF macro-F1 + melanoma recall + conformal coverage). *(`_predict_tta.py`)*
    - **Gate:** external-test shot → record Tier-1 numbers. Re-lock melanoma point after each change.

### TIER 2 — Data expansion (laptop OK, A100 faster)  →  ~0.55–0.62 (biggest real gain)
18. **Acquire gaps** (`DATASETS.md`): SCIN-full + CleanPatrick + BCN20000-full for train; **DDI held out** as 2nd external / dark-skin eval, never trained on. Dedup every new pull against existing via the phash cache. *(A4)*
19. **Taxonomy-map** into the 25 classes; document the mapping.
20. **Rebuild group-disjoint folds** (patient/lesion-level; keep HAM/BCN/ISIC lesion-disjoint).
21. **Retrain** the Tier-1 best recipe on the expanded set.
22. **Powered fairness** — group temp scaling FST IV–VI (shrunk toward global T) + group-DRO late fine-tune + ITA skin-tone backfill for unlabeled test images. *(A7 + B5)*
    - **Gate:** external-test + **DDI** eval → record Tier-2 numbers with CIs.

### TIER 3 — Architecture / SOTA push (rent A100, GPU-bound)  →  ~0.58–0.66
23. **PanDerm "oracle"** first — measure the derm-FM ceiling on our splits (research-only, never shipped).
24. **Continued SSL** (MAE/DINOv2) on own consented images, base DINOv3 ViT-B/16 → new backbone → re-run fine-tune. *(B3)*
25. **Frozen Google Derm-Foundation embedding** as a 2nd fusion token + optional KD from PanDerm features. *(B3/B6)*
26. **(Optional) MONET concept bottleneck** for interpretability + per-Fitzpatrick concept audit. *(B6)*
    - **Gate:** final external + DDI → record Tier-3 numbers.

---

## 5. Publishability guardrails (do these throughout, not at the end)
- **Bootstrap 95% CIs on every headline metric** — point estimates alone read as amateur.
- **Ablation table** isolating each contribution (collapse-fix, calibration, conformal, cRT, data) on the same splits — reviewers will demand it; the ablation driver already exists.
- **Two external sets** — the current external + **DDI** (dark-skin). Two beats one, and backs the fairness claim.
- **Baselines on our splits** — plain-CE, the collapsed model, a derm-FM oracle → makes each delta attributable.
- **No synthetic data in val/test, ever.** If synthetic is used in train, ablate it and prove leak-freeness.
- **Report the honest framing**: 25-way top-1 ~0.5, but coarse/binary/top-3 in the 0.7–0.9s, calibrated + abstaining. Honesty is the paper's selling point (the collapse-fix + leak-free eval + trust layer).

---

## 6. Recommended sequence
1. **Tier 0 in full** — front-load trust (OOD gate + fixed conformal + locked operating point) so the app is trustworthy and the numbers are credible *before* any retraining.
2. **A6 → A1+LORT → A5** one mechanism at a time under the ablation driver.
3. **Model soups → LIFT → registers swap**, then wire cascade + abstention + OOD into the app.
4. **Tier 2 data expansion** (SCIN-full, CleanPatrick, BCN-full; DDI held out) — the biggest real gain.
5. **Tier 3 SSL** last, GPU-rented, only after the PanDerm oracle says the ceiling is worth it.

**Trustworthy + publishable is reachable entirely on the laptop (Tiers 0–2). The rental decision only becomes real at Tier 3.**
