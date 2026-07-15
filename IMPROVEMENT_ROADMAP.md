# MNKAI-III — Improvement Roadmap & Path Toward State-of-the-Art

## Executive Summary

MNKAI-III is, honestly, a strong single-GPU research system that is **not yet** at published derm-FM SOTA — and under its hard constraints (6 GB VRAM, no synthetic, no scraping, Indian focus, single author) it should not pretend to be. The cancer pathway is already clinically credible (external melanoma AUROC 0.987 / recall 0.842; SCC/BCC AUROC 0.998), but the headline 25-class external macro-F1 of **0.484** is dominated by a data-starved long tail (bacterial_infection F1 0.09, contact_dermatitis 0.15, urticaria 0.21, psoriasis 0.23) and an underpowered fairness story (FST IV-VI n=135). The realistic truth is that **loss/classifier/threshold tricks alone will lift the 0.15–0.31 band but cannot fix the 0.09 floor — that is a *data* problem**, and the single biggest in-constraint lever is license-clean Indian/dark-skin data (DermaCon-IN, DDI). With method levers (Tier 0–1) external macro-F1 is realistically reachable to **~0.50–0.53**; with license-clean data expansion (Tier 2) to **~0.55–0.62**; an architecture/SSL push (Tier 3) plausibly adds a few more points but is GPU-time-bound, not free. The right framing for deployment is not an autonomous 25-way diagnoser but a **high-sensitivity cancer triage + selective-referral decision-support tool** — which is exactly the FDA-cleared DermaSensor precedent and the most defensible clinical claim available at this performance level.

---

## Part A — The 7 committed roadmap items (detailed plans)

Throughout: **OOF = out-of-fold CV predictions on group-disjoint folds**; the **external 5,886-image test set stays pristine (single-shot, touched once per finalized config)**; and **every change re-verifies the melanoma-recall floor ≥ 0.90 at the locked threshold before it is accepted.**

### A1 — Decoupled cRT head-retrain

| | |
|---|---|
| **Goal** | Recover medium/few-class F1 by rebalancing only the classifier, never the representation (the root-cause fix for the double-balancing collapse). |
| **Method** | Two-stage decoupling (Kang et al. 2020). Stage 1: train backbone+fusion on the natural long-tailed distribution (no resampling). Stage 2: freeze backbone+fusion, re-init + retrain only the head with class-balanced sampling. Add the **LORT** upgrade (2024): stage-2 head with heavy-negative label smoothing (δ≈0.98–0.99 spread over negatives) — beats plain cRT/LWS at ~zero cost. |
| **Repo touchpoints** | `freeze_backbone_epochs` + shape-aware **head-reinit** (already "cRT-ready"); stage-2 sampler via `sampler_power`→balanced; LORT = one-line label-smoothing change in the stage-2 loss; sweep with the **ablation driver**. |
| **Expected gain** | +2 to +4 macro-F1 from cRT, +1 to +1.5 more from LORT. Realistic target 0.484 → ~0.50–0.51 from this lever alone. |
| **Effort** | Low–med. cRT a few hours (infra exists); LORT one line. Cost is re-running CV + a single external check. |
| **Risk** | Head retrain can erode the locked melanoma operating point. Tail classes with ~no support still won't move. |
| **Validation** | Tune entirely on OOF macro-F1. After retrain, **re-tune the per-cancer threshold (A3) and re-confirm melanoma recall ≥ 0.90 on OOF** before the single external shot. |
| **Dependencies** | Stage-1 checkpoint. Feeds A2 (thresholds) and the cancer-gate cascade (B7). |

### A2 — Per-class thresholds tuned on OOF

| | |
|---|---|
| **Goal** | Stop arg-maxing; recover tail recall by trading head precision via per-class decision thresholds — the cheapest single point of macro-F1 in the whole plan. |
| **Method** | Sweep per-class thresholds (or per-class logit bias) on OOF predictions to maximize macro-F1; this is mathematically post-hoc logit adjustment. Cancers handled by A3 (recall-targeted), the rest tuned around the locked cancer points. |
| **Repo touchpoints** | The existing **per-class screening-threshold script**; apply on **temperature-scaled** probabilities; use group-disjoint OOF folds. |
| **Expected gain** | +2 to +5 macro-F1 with zero retraining, almost entirely tail/medium classes. Will **not** rescue bacterial 0.09 if genuinely unseparable; recovers the 0.15–0.31 band. |
| **Effort** | Low — hours, script exists. |
| **Risk** | Overfitting thresholds to a small OOF set. |
| **Validation** | Tune on OOF **only**; the external set is the single-shot check. Report external realized macro-F1 (expect a small drop vs OOF). |
| **Dependencies** | Best available trained model (post-A1). Pairs with A3. |

### A3 — Lock melanoma operating point at sensitivity ≥ 0.90

| | |
|---|---|
| **Goal** | Convert a vanity AUROC (0.987) into a defensible clinical claim: melanoma recall 0.842 → ≥ 0.90 (target 0.95), with the full trade-off reported. |
| **Method** | For each malignant class, sweep OOF probabilities to the lowest threshold achieving target recall; **freeze** it; report sensitivity/specificity/PPV/NPV + over-referral rate at that exact point. Adopt **ISIC pAUC≥80%TPR** as the primary cancer metric alongside full AUROC. |
| **Repo touchpoints** | Per-class **screening-threshold script**; **temperature scaling** for calibrated probs; reuse existing `threshold_table_primary.csv` / `calibration.json` reporting. |
| **Expected gain** | Melanoma recall 0.842 → 0.90–0.95 at a quantified, reported specificity cost (mirrors DermaSensor 95.5% sens / 20.7% spec). |
| **Effort** | Low — 1–2 days, no retraining. |
| **Risk** | Over-referral (low PPV) if specificity collapses — must be reported, not hidden. Threshold set on OOF shifts ~0.02–0.05 on external. |
| **Validation** | Set threshold on OOF; report the **external-realized** recall. This floor is a hard gate on every subsequent change. |
| **Dependencies** | Depends on whichever model is current; must be re-run after A1/B-tier changes. |

### A4 — License-clean data expansion (DermaCon-IN, DDI, SCIN-full)

| | |
|---|---|
| **Goal** | Attack the data-starved tail and the underpowered fairness eval at the source — the single biggest real lever. |
| **Method** | **DermaCon-IN** (5,450 South-Indian clinical photos, FST IV–VI, infectious/inflammatory-rich, CC BY-NC-SA) as primary tail training fuel; **SCIN** (10,408 consented, eFST/eMST) for inflammatory bulk; **PAD-UFES-20** (CC-BY, 21 metadata fields) for the metadata token + clinical-photo cancers; **DDI** (656 biopsy-proven, FST-balanced) **held out as fairness/cancer eval only, never trained on**. Map taxonomies into the 25 classes (MBBS author well-placed). Keep HAM/BCN/ISIC as a fixed cancer corpus with **lesion-level** disjoint splits. |
| **Repo touchpoints** | **Group-aware 5-fold CV** for patient/lesion-disjoint folds; merge into existing data pipeline; feed the hierarchy into A5. |
| **Expected gain** | Largest of any item: **+0.05 to +0.12 external macro-F1** if tail classes gain even 100–300 in-domain images each; converts FST IV–VI fairness from underpowered to powered. |
| **Effort** | Med — 1–2 weeks per dataset (download, taxonomy mapping, fold construction). |
| **Risk** | NC-SA licenses are research-only and viral (block commercial deployment). Taxonomy mapping needs documentation for reproducibility. **Leakage** if HAM/BCN/ISIC split by image not lesion. |
| **Validation** | Patient/lesion-disjoint folds; DDI strictly held out; re-run external test once after merge. |
| **Dependencies** | Enables A5 (hierarchy), A7 (powered fairness), and group-DRO (B5). |

### A5 — Hierarchical coarse→fine loss

| | |
|---|---|
| **Goal** | Rescue the clinically-clustered extreme tail by borrowing gradient from data-rich super-group siblings, and strengthen the abstention layer's "coarse-correct" fallback. |
| **Method** | Add an auxiliary coarse head (e.g. neoplastic / inflammatory-papulosquamous / infectious / bullous-autoimmune) above the 25-way head, with a weighted auxiliary loss. Reuse DermaCon-IN's 8-coarse/19-sub Rook's hierarchy as a ready-made taxonomy. |
| **Repo touchpoints** | New coarse head + weighted aux loss in the trainer; tune the loss weight via the **ablation driver**; coarse-correct rate feeds the conformal/abstention layer (already top-3 0.725). |
| **Expected gain** | +1 to +3 macro-F1 on fine tail classes; notably better coarse-correct rate. Helps bacterial more than pure loss-rebalancing (infectious super-group gradient). |
| **Effort** | Med — ~1 day + taxonomy validation. |
| **Risk** | A bad grouping can hurt; adds one loss-weight hyperparameter. |
| **Validation** | OOF macro-F1 + coarse-correct rate; melanoma floor re-checked. |
| **Dependencies** | Taxonomy best defined with A4's hierarchy in hand. |

### A6 — Single-mechanism ablation + DRW (deferred reweighting)

| | |
|---|---|
| **Goal** | Institutionalize the lesson from the collapse: **exactly one** rebalancing mechanism, applied **late**, never stacked. |
| **Method** | Ablate ONE of {balanced-softmax / LogitAdjustedCE / cb_focal} with `sampler_power`≈0–0.3 and **no** inverse-freq sampler. Add the **DRW schedule**: plain CE/LDAM for the first ~70–80% of epochs, then switch on class reweighting/balanced sampling. Tune logit-adjustment τ ∈ [0.5, 1.0]. |
| **Repo touchpoints** | **LogitAdjustedCE/balanced-softmax** (built), **cb_focal** (built), `sampler_power` (built), DRW = an epoch-threshold flag, swept via the **ablation driver**. |
| **Expected gain** | +1 to +3 from clean single-mechanism balanced-softmax; DRW adds ~+1–2 vs naive reweighting and is mainly a guard-rail against re-collapse. |
| **Effort** | Low — a few hours; loss + a schedule flag. |
| **Risk** | Over-correcting head classes (eczema recall down) — mitigated by τ tuning on OOF macro-F1, not accuracy. |
| **Validation** | OOF macro-F1 with per-class recall in the loop (watch eczema). Melanoma floor re-checked. |
| **Dependencies** | This is the disciplined harness under which A1/A5 are run; do first or jointly. |

### A7 — Fairness hardening (group temp scaling FST IV–VI + FST oversampling)

| | |
|---|---|
| **Goal** | Make the locked operating points and calibration hold *within* skin-tone groups, and tell an honest, statistically-defensible fairness story. |
| **Method** | (1) **Honest eval first**: report every FST subgroup metric with stratified BCa bootstrap 95% CIs + an exact/permutation test for the I-III vs IV-VI gap, and state the minimum detectable effect — at n=135 the 0.245 vs 0.209 gap is almost certainly non-significant. (2) **Group (Mondrian) temperature scaling** for FST IV–VI, **shrunk toward global T=0.60** to avoid overfitting n=135. (3) FST oversampling / group-conditional thresholds — but defer the train-time parts until A4 raises FST IV–VI n. |
| **Repo touchpoints** | **Per-Fitzpatrick fairness eval** (wrap in 1000-resample BCa + exact test); **temperature scaling** extended per-group; **Mondrian conformal** makes per-group consistent; per-class **threshold script** for group-conditional points. |
| **Expected gain** | No accuracy gain from the eval fix, but converts an indefensible claim into a credible one. Group temp scaling reduces FST IV–VI ECE; magnitude limited by n=135 (report as underpowered). |
| **Effort** | Low — ~1–2 days. |
| **Risk** | Free-fit group T overfits 135 images → shrinkage mandatory. Do not fabricate FST labels. |
| **Validation** | Bootstrap the per-group ECE; report CIs everywhere; DDI (A4) becomes the powered external dark-skin check. |
| **Dependencies** | Real power requires A4 (DermaCon-IN/DDI). |

---

## Part B — New improvements beyond the 7

### B1 — Datasets

| Item | What + why for MNKAI | Expected gain | Effort | Source |
|---|---|---|---|---|
| **PAD-UFES-20** | CC-BY (commercial-friendly!) smartphone set, 6 classes, **21 metadata fields** incl. Fitzpatrick → best match for the cross-attention metadata token; clinical-photo cancers harden A3 on non-dermoscopic input. | +0.01–0.03 on clinical-photo BCC/SCC/AK; strengthens fusion ablation | Low | Mendeley zr7vgbcyr2 |
| **DERM12345 / Derm7pt** | Derm7pt's **7-point-checklist concept labels** enable a concept-bottleneck/interpretability head (clinical-grade story); DERM12345 adds fine cancer subclasses. | Small F1; main value interpretability + cancer-subclass resolution | Low-med | nature s41597-024-04104-3; jeremykawahara/derm7pt |
| **MSLD v2.0 (Mpox/exanthems)** | Viral-exanthem coverage (Mpox, measles, HFMD, chickenpox) relevant to tropical Indian outpatient settings; useful as an infectious node / OOD flag. | Niche coverage, high public-health relevance | Low | Kaggle MSLD v2.0 |
| **Fitzpatrick17k-C / CleanPatrick (cleaned only)** | Broadest tail-class coverage (114 dx) to warm-start data-starved classes — **but only the 2025 cleaned variant** (raw is 16–30% contaminated). | +0.01–0.03 if cleaned; negative if raw | Med | arXiv 2505.11034 |
| **ISIC 2024 SLICE-3D** | 400k low-res non-dermoscopic crops — phone-like; best as an **SSL pretraining / screening-robustness** booster (subsample 50–100k on 6 GB), not a taxonomy contributor. | Cancer-screening robustness + calibration | High | nature s41597-024-03743-w |
| **Excluded — SD-198 / Dermnet-Kaggle / raw Fitzpatrick17k** | Tempting for broad tail coverage **but murky atlas-scraped provenance → fails the no-scrape / license-clean bar.** | — | — | Deprioritize |

### B2 — Training methods

| Item | What + why for MNKAI | Expected gain | Effort | Source |
|---|---|---|---|---|
| **Greedy model soups + SWA** | Weight-average the 5-fold/seed checkpoints you **already train** — zero inference cost, directly attacks the 0.637→0.484 internal→external drop. **Do first.** | +0.5–1.5 macro-F1 + robustness | Low | Wortsman et al. 2022 |
| **LIFT/LIFT+ recipe** | Heavy fine-tuning *hurts* the tail; instead **freeze backbone + <1% structured adapters (AdaptFormer/LoRA) + cosine classifier + logit-adjusted loss + DINOv2 feature-prototype init** (replaces text init). Almost exactly MNKAI's architecture; ideal for 6 GB (5–15 epochs). | +3–6 macro-F1, tail-weighted | Med | LIFT ICML 2024; LIFT+ 2025 |
| **Derm color-constancy + hair aug** | Shades-of-Gray normalization + hair overlay/DullRazor + TrivialAugment → attacks heterogeneous Indian camera/lighting drift (interacts with fairness). | +1–3, stronger OOD | Low | Barata; Müller & Hutter ICCV21 |
| **Class-aware curriculum aug (CUDA)** | Stronger augmentation for starved tail classes, lighter for head — manufactures variety with no new/synthetic images. | +1–2, tail-weighted | Med | Ahn et al. ICLR 2023 |
| **EMA + tuned label smoothing** | EMA stabilizes the noisy small-batch grads forced by 6 GB accum; sweep LS ε with tail recall + conformal coverage in the loop. | +0.2–0.7 + stability | Low | Müller et al. NeurIPS19 |
| **Guarded selective TTA** | ≤4 label-preserving views, **gated on OOF macro-F1 + melanoma recall + conformal coverage** — naive TTA has dropped medical classifiers 28–31 pts. | +0.3–1.0 if gated | Low | "TTA Hurts Medical" 2026 |
| **GPaCo contrastive (deprioritize)** | Could tighten confusable inflammatory clusters, but **batch-size-hungry → fights 6 GB**; only via memory queue + accum, modest gain on frozen backbone. | +1–2, conditional | Med-high | GPaCo TPAMI 2023 |

### B3 — Models & architectures

| Item | What + why for MNKAI | Expected gain | Effort | Source |
|---|---|---|---|---|
| **DINOv2-with-registers backbone swap** | `vit_base_patch14_reg4_dinov2` removes high-norm artifact tokens → **cleaner patch features, which directly benefit your patch-level cross-attention fusion.** Same 6 GB footprint. | +0.3–1.5 + cleaner attention | Low | Darcet et al. ICLR 2024 |
| **Continued DINOv2/MAE SSL on your own consented derm images** | The **only fully constraint-clean route to derm-FM features** (this is literally how PanDerm/DermINO were built) and the **only method shown to also improve skin-tone fairness**. Prefer lightweight MAE on 6 GB (DINO/iBOT's 2-crop+EMA-teacher is memory-heavy). | +2–6 macro-F1, tail + dark-skin weighted | High (GPU-time) | DermINO 2508.12190; dinov2 model card |
| **Google Derm Foundation as a frozen 2nd stream** | 6144-d embedding per image, **cached once**, cross-attended as another token → cheap heterogeneous ensemble, also improves calibration. | +1–4 + calibration | Low | google/derm-foundation (HAI-DEF) |
| **KD from a derm FM into your ViT-B** | Precompute PanDerm/DermINO **features offline** (sidesteps 6 GB teacher-at-train) and feature-distill into the deployable student — borrow tail/skin-tone knowledge at no inference cost. | +1–3, tail-weighted | Med-high | PanDerm Nat Med 2025 |
| **DINOv3 ViT-B/16** | Permissive **commercial** license — best base for the SSL above. But at 224px it's ~a wash vs DINOv2; only worth it as the SSL base or if you raise resolution. | ~0 to +2 @224 | Low | dinov3 lvd1689m |
| **Excluded as the shipped backbone — PanDerm / DermLIP** | Beat MONET at ViT-B size and would most help your tail, **but CC-BY-NC-ND → no commercial deployment, no derivatives.** Use as a **research-only oracle** to measure the derm-FM ceiling, never ship. | (oracle only) | Med | PanDerm; DermLIP_ViT-B-16 |

### B4 — Uncertainty & calibration

| Item | What + why for MNKAI | Expected gain | Effort | Source |
|---|---|---|---|---|
| **Clustered conformal (replace marginal APS)** | Pools score-similar classes so rare tail classes borrow calibration strength — **the direct fix for the 98.4%-coverage / 10.6-set-size pathology** (per-class Mondrian would explode at your tail n). | Set size 10.6 → ~3–5, coverage → ~90% | Low-med | Ding/Tibshirani NeurIPS 2023 |
| **RAPS base score** | Truncates APS's heavy set-size tail — the standard first fix, layer the cluster method on top. | Meaningful set-size drop | Low | Angelopoulos et al. ICLR 2021 |
| **soft-TACP rank regularization** | Transfers coverage from well-served head (cancers, eczema) to weak tail — ~50% head-tail coverage-gap cut at equal/smaller set size. | ~50% gap reduction | Low | arXiv 2508.11345 |
| **Per-class / vector / Dirichlet-ODIR scaling** | Single global T=0.60 can't fix 25 differing reliability curves; per-class scaling feeds cleaner scores to conformal + abstention. | classwise-ECE ↓ | Low | Kull et al. NeurIPS19; 2411.02988 |
| **Last-layer K-head ensemble on cached frozen features** | Full deep ensembles infeasible on 6 GB, but your **backbone is frozen** → ensemble only the cheap head for most of the epistemic-UQ benefit; strengthens abstention. | ECE ↓, better correct/wrong ranking | Med | Lakshminarayanan et al. 2017 |
| **Mahalanobis/RMD OOD gate on DINOv2 embeddings** | Flags "unlike anything I was calibrated on" (Indian devices/skin/unseen conditions) → routes to abstention, **protects the conformal exchangeability assumption** under shift. | Near-free OOD/shift detector | Med | Mahalanobis++ 2505.18032 |
| **Excluded as a deliverable — Evidential DL** | Single-pass and tempting, but 2024 literature reports **poorly-ranked, miscalibrated uncertainty** that lags the softmax+head-ensemble+conformal stack. Ablation only. | — | Med | AIM 2024 UQ review |

### B5 — Fairness (beyond A7)

| Item | What + why for MNKAI | Expected gain | Effort | Source |
|---|---|---|---|---|
| **Continuous ITA-based distribution-aware reweighting** | Compute ITA from CIELAB on lesion-masked images (no labels) → individual-level reweighting **and backfill skin tone for the ~85% of test images lacking Fitzpatrick**, partially fixing the underpowered-n problem itself. | Moderate + fairness-eval coverage ↑ | Med | arXiv 2512.08733 |
| **Group-DRO / subgroup-DRO (late fine-tune)** | Worst-group loss over FST groups — principled anti-trade-off, but **only after A4 raises FST IV–VI n** (DRO on n=135 is unstable); pairs with the DRW schedule. | Worst-group ↑ (conditional on data) | Med | Sagawa et al. group-DRO |
| **Cluster-Focal (attribute-free)** | Mitigates calibration bias **without needing FST labels** — works on the whole set, sidestepping your label-coverage gap. | Modest, label-free | Med | MICCAI 2023 2307.01738 |
| **Domain-independent per-FST heads** | Cheap head duplication on your cRT-ready head; dark-skin boundaries adapt without the encoder sacrificing light-skin. **n=135 head → heavy regularization / branch only the last linear layer.** | Modest, low-cost | Low-med | Wang et al. CVPR 2020 |
| **Excluded/deprioritized — adversarial debiasing; ViT fairness pruning (SkewPrune)** | Adversarial: unstable + tone is partly diagnostic + can't learn signal at n=135. SkewPrune: only ViT tone-pruning method but ~1% delta for weeks of work; **avoid the skin-tone-normalization-by-transformed-image parts → edges into synthetic.** | — | High | 2509.00745; 2509.22712 |

### B6 — Multimodal fusion

| Item | What + why for MNKAI | Expected gain | Effort | Source |
|---|---|---|---|---|
| **DINOv2-with-registers under the cross-attention** | (Also in B3) the artifact-token removal is *specifically* a fusion win: your cross-attention reads patch tokens, and artifact tokens are exactly the background noise that corrupts patch-level attention. | +0.3–1.5 + cleaner maps | Low | Darcet et al. ICLR 2024 |
| **Add Derm Foundation embedding as a 2nd fusion token** | Your cross-attention already ingests a metadata token; add the frozen 6144-d derm embedding as another token → heterogeneous late fusion at ~zero extra VRAM. | +1–4 | Low | google/derm-foundation |
| **PAD-UFES-20 21-field metadata** | Richer structured metadata (diameter, site, cancer history, FST) strengthens the metadata-fusion ablation evidence and deployment realism. | +0.01–0.03 + ablation evidence | Low | PAD-UFES-20 |
| **MONET concept token (audit + optional bottleneck)** | Frozen concept annotations audit per-Fitzpatrick concept coverage (your fairness eval is underpowered) and can feed a concept-bottleneck head for interpretability. | +1–2 if bottleneck; main value trust/audit | Low-med | suinleelab/monet |

### B7 — Clinical decision system

| Item | What + why for MNKAI | Expected gain | Effort | Source |
|---|---|---|---|---|
| **Two-stage cancer-gate cascade** | Stage-1 binary malignant-vs-benign gate at high sensitivity → routes to fine head. **Decouples the safety-critical melanoma sensitivity from the messy 25-way tail** (cancers already 0.97–0.998 AUROC). | Cleaner separately-tunable cancer recall; often lifts both | Med | PMC11183002; nature s41598-025-89230-7 |
| **Selective prediction / "refer to dermatologist"** | Abstain when conformal set size > k or margin low → turns the weak tail (F1 0.09) from a liability into a safe referral. You already have `coverage_risk_curve.png` — formalize it. Re-tune α so set sizes are usable (currently over-covering 98.4%). | Retained-set error ↓ ~45–50% | Low-med | nature s41598-026-40637-w |
| **Top-3 triage as primary UI** | External top-3 0.725 vs top-1 0.481 — present a ranked differential + cancer flag, **the cheapest way to make a 0.484 model clinically honest and useful.** | 0.481 → 0.725 usable, reframed honestly | Low | BJD 7564904 |
| **DermaSensor-style honest reporting** | Lead with sensitivity + NPV (rule-out), disclose PPV/over-referral; pAUC≥80%TPR as the cancer metric. Reframes MNKAI as a **regulator-credible primary-care triage aid**, the achievable claim. | Credibility, not metric | Low | npj s41746-024-01161-1 |
| **Group-conditional operating points** | Report per-FST sensitivity at the locked threshold (A3+A7) so the melanoma floor holds *within* skin tones, with wide CIs at n=135. | Prevents hidden dark-skin sensitivity gap | Low-med | arXiv 2503.23819 |
| **Human+AI reader-study framing (retrospective)** | Validate as *decision support* (AI flag changes clinician sens/spec), not AI-alone — the regulatory-evidence structure; full prospective study is out of laptop scope. | Repositions modest standalone F1 | Med | npj s41746-023-00872-1 |

---

## Part C — Phased execution plan

| Tier | Actions | Projected external macro-F1 | Projected melanoma sensitivity | Effort | Main risk |
|---|---|---|---|---|---|
| **Tier 0 — Near-free quick wins (days)** | A2 OOF thresholds · A3 lock melanoma ≥0.90 + pAUC≥80%TPR · B2 greedy model soups/SWA · B4 RAPS + clustered conformal + soft-TACP · B4 per-class temp scaling · A7 honest bootstrap-CI fairness eval · B7 top-3 UI + selective-prediction + DermaSensor framing | **~0.50–0.52** | **≥0.90** (locked & reported) | Low (1 week) | Threshold/temp overfit to small OOF; over-referral must be disclosed |
| **Tier 1 — Method levers (no new data)** | A1 cRT + LORT · A6 single-mechanism balanced-softmax + DRW · A5 hierarchical loss · B2 LIFT/LIFT+ adapters + cosine + prototype init · B2 color-constancy/hair/TrivialAugment + EMA + LS · B3 DINOv2-registers swap · B4 last-layer K-head ensemble + Mahalanobis gate · B7 cancer-gate cascade | **~0.51–0.54** | ≥0.90 (re-locked after each change) | Med (3–6 weeks) | Re-introducing collapse if >1 mechanism stacked; head retrain erodes melanoma point |
| **Tier 2 — License-clean data expansion** | A4 DermaCon-IN + SCIN + PAD-UFES-20 training; DDI held-out eval · re-map taxonomy into A5 hierarchy · B5 ITA backfill + group-DRO late fine-tune (now powered) · A7 powered FST IV–VI fairness on DDI · B1 cleaned Fitzpatrick17k-C warm-start | **~0.55–0.62** | ≥0.90, now validated **per-FST** | Med-high (4–8 weeks) | NC-SA blocks commercial ship; taxonomy-mapping errors; lesion-level leakage |
| **Tier 3 — Architecture / SOTA push** | B3 continued MAE/DINOv2 SSL on own consented images (DINOv3 commercial base) · B3 frozen Derm Foundation 2nd stream + KD from PanDerm/DermINO features · B6 MONET concept bottleneck · (research-only oracle: PanDerm/DermLIP ceiling measurement) | **~0.58–0.66** (high variance, GPU-time-bound) | ≥0.90, fairness-improved by SSL | High (months, GPU-time) | SSL underfits on small pool / catastrophic forgetting; gains not guaranteed |

**Recommended ordered sequence:**

1. **Tier 0 entire** (1 week, near-free, locks the melanoma floor and fixes the conformal/fairness-reporting credibility before any retraining).
2. **A6 → A1+LORT → A5** under the ablation driver, one mechanism at a time (the disciplined long-tail core).
3. **B2 model soups** (fold over the new checkpoints) → **B2 LIFT adapters** → **B3 DINOv2-registers swap**.
4. **B7 cancer-gate cascade + selective prediction** wired into the demo app.
5. **A4 DermaCon-IN + SCIN + PAD-UFES-20** (the biggest real gain) with DDI held out → re-run A5 hierarchy, A7 powered fairness, B5 group-DRO.
6. **Tier 3 SSL** as a parallel long-running background track on the GPU, plus the **Derm Foundation frozen 2nd stream** (cheap) and the **PanDerm oracle** (measures the ceiling so you know whether SSL is worth the GPU-weeks) — last, because it is GPU-time-bound and lowest certainty per hour.

Re-confirm the melanoma-recall floor and re-fit conformal/temperature **after every step that touches the model.** Touch the external 5,886-image test set **once per finalized tier**, never for tuning.

---

## Part D — Honest path to SOTA

**What "SOTA" even means here — and why headline numbers lie.** Most published derm-classification "SOTA" (HAM/ISIC leaderboards, many Fitzpatrick17k papers) uses **image-level splits that leak the same lesion across train/test** (HAM10000 has only ~7,470 unique lesions in 10,015 images; cross-challenge HAM/BCN/ISIC reuse the same lesions). MNKAI-III's **group-disjoint 5-fold CV + single-shot external test is *more* honest than those numbers**, so a literal macro-F1 comparison understates MNKAI's standing. The fair comparators are **derm foundation models on clean external splits** (PanDerm, DermLIP, DermINO) — and those are the real ceiling. The other honest caveat: macro-F1 0.484 on 25 imbalanced classes with a near-unseparable tail is a *different and harder* task than 7-class dermoscopy cancer, where MNKAI is already near-SOTA (melanoma AUROC 0.987).

**Realistic ceiling.**

| Scenario | Realistic external macro-F1 ceiling | Why |
|---|---|---|
| Methods only, no new data, 6 GB | **~0.52–0.54** | Thresholds/cRT/LIFT/soups recover the 0.15–0.31 band; bacterial 0.09 stays ~floored — it is a *data*, not a *loss*, problem. |
| + License-clean data (DermaCon-IN/SCIN/PAD-UFES) | **~0.58–0.63** | In-domain Indian FST IV–VI images are the only thing that moves the starved tail and powers fairness. |
| + SSL/derm-FM features, still 6 GB | **~0.62–0.66** | Continued SSL/KD/Derm-Foundation fusion add derm priors; gains real but GPU-time-bound and not guaranteed. |
| If the GPU constraint were lifted (bigger VRAM, ≥384–512px) | **+2–5 more, plus a real DINOv3 advantage** | DINOv3 only beats DINOv2 at ≥512px; contrastive/multi-expert methods become feasible; full-scale SSL gets affordable. The 6 GB cap is the binding limit on the *architecture* tier, not the data tier. |

**The 3 things that would move the needle most:**

1. **License-clean in-domain Indian / dark-skin data (DermaCon-IN + DDI).** The only lever proven in the literature to actually *close* the FST gap and the only thing that lifts the 0.09 tail floor. Single highest expected ΔF1 (+0.05–0.12) and the fairness story depends on it.
2. **Continued self-supervised pretraining on your own consented images.** The one fully constraint-clean route to derm-FM-level, commercially-shippable features, and uniquely it also improves skin-tone fairness — but it costs GPU-weeks, so measure the ceiling with the PanDerm oracle first.
3. **Reframing the product as high-sensitivity cancer triage + selective referral (cascade + abstention + top-3 + DermaSensor-style reporting).** This does not raise macro-F1 at all, but it is what makes a 0.484-macro-F1, 0.987-melanoma-AUROC system a *defensible, deployable, regulator-credible* clinical tool for Indian primary care — which is the actual goal.

---

## Sources

**Backbones / foundation models**
- PanDerm — Nature Medicine 2025: https://www.nature.com/articles/s41591-025-03747-y · https://github.com/SiyuanYan1/PanDerm
- DermLIP ViT-B/16 (ICCV 2025): https://huggingface.co/redlessone/DermLIP_ViT-B-16 · https://arxiv.org/html/2503.14911v1
- DermINO (continued derm SSL): https://arxiv.org/abs/2508.12190
- DINOv2 model card: https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
- DINOv3 ViT-B/16 (lvd1689m): https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m · https://github.com/facebookresearch/dinov3 · https://arxiv.org/pdf/2510.07191
- Vision Transformers Need Registers (Darcet et al., ICLR 2024): https://arxiv.org/abs/2309.16588
- Google Derm Foundation: https://huggingface.co/google/derm-foundation · https://developers.google.com/health-ai-developer-foundations/derm-foundation
- MONET: https://huggingface.co/suinleelab/monet · https://github.com/suinleelab/MONET

**Long-tailed recognition**
- Menon et al., Logit Adjustment (ICLR 2021): https://arxiv.org/abs/2007.07314
- Ren et al., Balanced Meta-Softmax (NeurIPS 2020)
- Kang et al., Decoupling Representation and Classifier (ICLR 2020): https://arxiv.org/pdf/1910.09217
- LORT / Logits Retargeting (2024): https://arxiv.org/html/2403.00250v1
- Cao et al., LDAM-DRW (NeurIPS 2019): https://arxiv.org/abs/1906.07413
- LIFT (ICML 2024): https://arxiv.org/pdf/2309.10019 · LIFT+ (2025): https://arxiv.org/abs/2504.13282 · https://github.com/shijxcs/LIFT
- GPaCo (TPAMI 2023): https://arxiv.org/abs/2209.12400
- SADE (NeurIPS 2022): https://arxiv.org/abs/2107.09249 · RIDE: https://arxiv.org/pdf/2010.01809
- Weight Balancing (CVPR 2022): https://github.com/ShadeAlsha/LTR-weight-balancing
- SuperDisco (super-class long-tail, 2023): https://arxiv.org/pdf/2304.00101
- Systematic Review on Long-Tailed Learning (2024): https://arxiv.org/abs/2408.00483

**Datasets**
- DermaCon-IN: https://arxiv.org/abs/2506.06099
- DDI (Science Advances 2022): https://www.science.org/doi/10.1126/sciadv.abq6147 · https://ddi-dataset.github.io/
- SCIN: https://github.com/google-research-datasets/scin · https://arxiv.org/abs/2402.18545
- PAD-UFES-20: https://data.mendeley.com/datasets/zr7vgbcyr2/1 · https://pmc.ncbi.nlm.nih.gov/articles/PMC7479321/
- ISIC challenge data: https://challenge.isic-archive.com/data/ · ISIC 2024 SLICE-3D: https://www.nature.com/articles/s41597-024-03743-w
- HAM10000 leakage analysis: https://www.nature.com/articles/s41597-025-04382-5
- Fitzpatrick17k-C / CleanPatrick (2025): https://arxiv.org/pdf/2505.11034
- Derm7pt: https://github.com/jeremykawahara/derm7pt · DERM12345: https://www.nature.com/articles/s41597-024-04104-3
- MSLD v2.0: https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20/data

**Calibration / conformal / uncertainty**
- Clustered conformal (Ding/Tibshirani, NeurIPS 2023): https://arxiv.org/abs/2306.09335
- TACP / soft-TACP (2025): https://arxiv.org/html/2508.11345v1 · https://arxiv.org/html/2507.06867
- RAPS (Angelopoulos et al., ICLR 2021): https://arxiv.org/pdf/2009.14193
- Dirichlet calibration (Kull et al., NeurIPS 2019) · Confidence Calibration with Many Classes (NeurIPS 2024): https://arxiv.org/pdf/2411.02988
- DATS group-aware temperature: https://arxiv.org/html/2509.21161
- Selective Classification (Geifman & El-Yaniv, NeurIPS 2017): https://arxiv.org/pdf/1705.08500 · https://arxiv.org/pdf/2508.07556
- Mahalanobis++ (2025): https://arxiv.org/pdf/2505.18032 · SNGP (Liu et al., NeurIPS 2020)
- Trustworthy clinical AI UQ review (AIM 2024): https://www.sciencedirect.com/science/article/pii/S0933365724000721

**Fairness**
- One Size Fits None (2025): https://arxiv.org/pdf/2506.14400 · Size-adaptive Fairness Testing: https://arxiv.org/pdf/2506.10586
- npj Digital Medicine fairness review (2024): https://www.nature.com/articles/s41746-024-01276-5
- Cluster-Focal (MICCAI 2023): https://arxiv.org/abs/2307.01738
- Sagawa et al. group-DRO · Distribution-Aware Reweighting (ITA, 2025): https://arxiv.org/abs/2512.08733
- Wang et al., Towards Fairness in Visual Recognition (CVPR 2020): https://arxiv.org/abs/1911.11834
- SkewPrune (ViT fairness pruning): https://arxiv.org/html/2509.00745v1 · Skin-tone normalization + channel pruning: https://arxiv.org/html/2509.22712

**Training methods**
- Model Soups (Wortsman et al., ICML 2022): https://arxiv.org/abs/2203.05482
- TrivialAugment (ICCV 2021): https://arxiv.org/abs/2103.10158 · RandAugment (NeurIPS 2020)
- When Does Label Smoothing Help? (Müller et al., NeurIPS 2019)
- CUDA curriculum augmentation (ICLR 2023)
- "TTA Hurts Medical Image Classification" (2026): https://arxiv.org/abs/2604.09697 · BayTTA: https://arxiv.org/abs/2406.17640
- KD from vision foundation models (medical): https://arxiv.org/abs/2503.06976

**Clinical decision systems**
- DermaSensor FDA De Novo / honest reporting: https://www.nature.com/articles/s41746-024-01161-1
- ISIC pAUC≥80%TPR metric: https://arxiv.org/pdf/2506.03420 · https://pmc.ncbi.nlm.nih.gov/articles/PMC12639164/
- Two-stage cascade: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11183002/ · https://www.nature.com/articles/s41598-025-89230-7
- Selective prediction / deferral (derm): https://www.nature.com/articles/s41598-026-40637-w · https://arxiv.org/pdf/2509.12573
- Top-3 differential / reader studies: https://academic.oup.com/bjd/article/191/1/125/7564904 · https://www.nature.com/articles/s41746-023-00872-1
- Group-conditional operating points: https://arxiv.org/pdf/2503.23819
