# MNKAI-III v2 — Overhaul Plan (final month · +Colab Pro+ A100)

The plan to close this version: improve **every** metric, strengthen the LLM, exhaust the model-architecture
question (incl. a from-scratch/SSL restart), finish the academic write-up, and only then productionize.

---

## 0. The through-line (the insight that reorganizes the whole plan)

The DDI breakthrough (0.58→0.748; dark-skin 0.53→**0.789**) taught us the single most important lesson of the
project: **in our data/compute regime, FROZEN foundation-model features + a light probe beat heavy fine-tuning.**
Fine-tuning on our mixed data *degraded* the representation.

So the overhaul reorganizes around one master lever: **get the best possible frozen features, then attach light
task heads.** That makes the whole month coherent:
1. **Benchmark every candidate backbone** via the cheap probe pipeline (DINOv2 @518 already wins DDI).
2. **Build a derm-specific foundation model** (MAE/CAE SSL on our own images, A100) to get *even better* frozen
   features — the UNI recipe, for dermatology.
3. **Re-derive every metric** on the winning backbone with light heads (probes / small MLPs / the fusion head).
4. Strengthen the LLM, write it up, productionize.

---

## 1. Status audit — what's done vs what's left

### ✅ Done (this version)
| Area | Result |
|---|---|
| 9-class base (Phase 1) | macro-F1 **0.637 ± 0.009** |
| 25-class collapse → fix (Fix C) | 0.339 → external **0.484** |
| Tier-0 per-class bias | 0.484 → **0.522** |
| Tier-1 cRT (5-fold) — best 25-class | **0.539**, melanoma recall **0.870** |
| Trust layer | ECE 0.21→**0.042** (T=2.2, deterministic R02); conformal 10.6→**5.81**; per-Fitzpatrick fairness |
| **DDI dark-skin cancer** | 0.58→**0.748** (V–VI 0.53→**0.789**) — frozen DINOv2@518 + probe |
| LLM Milestone-1 core | adaptive interview + Bayesian fusion (Gemini) **working** |
| Data | comprehensive: ISIC/HAM/BCN, Fitz(cleaned via CleanPatrick), PAD, SCIN, DermaCon-IN, DDI |
| Docs | RESEARCH_REPORT_*, DEEP_ANALYSIS, IMPROVEMENT_ROADMAP, PIPELINE, DARKSKIN_RESEARCH, DDI_BREAKTHROUGH |
| Demo app | 2-model switcher + decision-support layer |

### ⏳ Left / in progress
- Apply the **frozen-probe pattern to the 9-class and 25-class tasks** (only DDI-cancer has it so far).
- **Derm SSL foundation model** (MAE/CAE on A100) — the big untapped lever (all Tier-3).
- **Data expansion** — register Stanford AIMI → MRA-MIDAS + DDI-2 + ASAN.
- **LLM** — wire Milestone-1 into the app; build Milestone-2 (learned SCIN-history fusion) + enhancements.
- **Re-derive every metric** on the best backbone; refresh calibration/conformal/fairness/robustness.
- **Academic write-up** finalization.
- **Productionization** (after sign-off).

---

## 2. The 4-week plan (Colab Pro+ = A100 40GB)

### Week 1 — Backbone bake-off + consolidate (mostly laptop; A100 for feature dumps)
- **Probe-benchmark every backbone on DDI *and* the 25-class task** (cheap, we have the pipeline): DINOv2@224/@518,
  DINOv2-with-registers, EVA-02, ConvNeXt-V2, DINOv3, PanDerm (loaded *correctly* via its own model code),
  DermINO, Google Derm-Foundation, and UNI (as a curiosity — pathology, expected to transfer poorly). One table,
  frozen features + logistic/MLP probe, per-metric. **Pick the winning frozen backbone.**
- Re-run the **9-class and 25-class** heads as frozen-feature probes/MLPs at 518 — expect gains from the same
  effect that fixed DDI.
- **Register Stanford AIMI** (one Research Use Agreement) → queue MRA-MIDAS + DDI-2; grab ASAN.
- **Set up the Colab A100 pipeline**: upload the image corpus to Google Drive/GCS, checkpoint-to-Drive scaffolding,
  a reproducible SSL training notebook (so a disconnect never kills a run).

### Week 2 — Derm foundation model via SSL (A100) + LLM Milestone-2
- **Continued SSL on our ~270k derm images** to specialize the backbone → the biggest single lever for *every*
  metric. Recipe: **MAE** (simplest, A100-friendly, pixel reconstruction) starting from ViT-B/L weights, ~a few
  days on A100; **CAE** as an alternative if MAE underdelivers. Output: `derm-MAE` frozen backbone.
- In parallel (laptop/Gemini): **LLM Milestone-2** — learned image×history fusion trained/ablated on SCIN's
  structured symptom fields (the *measurable* gain), plus wire Milestone-1 into the app.

### Week 3 — Re-derive every metric on the winning backbone + data expansion
- Feature-extract with the winning backbone (existing FM or `derm-MAE`) at 518; re-fit **all** task heads:
  9-class, 25-class, binary-cancer, DDI, coarse. Report every metric with bootstrap CIs.
- Fold **MRA-MIDAS + DDI-2 + ASAN** into the cancer/DDI probe training for robustness + a powered mid-tone fairness read.
- Ensemble the top-2 frozen backbones (e.g. derm-MAE + DINOv2) if it helps.

### Week 4 — Trust refresh + LLM hardening + write-up draft
- Recompute **calibration (per-class temp) + conformal (clustered) + per-Fitzpatrick fairness (now DDI-2-powered)**
  on the new models; add **per-skin-tone abstention**.
- **LLM hardening**: confidence-gated + information-gain questions + RAG on derm criteria + red-flag escalation +
  multi-image/temporal + skin-tone-aware; upgrade reasoning to gemini-2.5-pro.
- **Draft the paper** (Section by section) from the refreshed numbers.

---

## 3. Every metric — the mechanism for improving it

| Metric | Now | Target | Lever |
|---|---|---|---|
| 9-class macro-F1 | 0.637 | ~0.68–0.72 | winning frozen backbone + probe + 518 + ensemble |
| 25-class macro-F1 | 0.539 | ~0.58–0.63 | frozen-probe pattern + derm-SSL + data + hierarchical head |
| Top-3 (25-class) | 0.725 | ~0.80 | better backbone |
| Internal cancer AUROC | 0.988 | hold | — |
| **DDI dark-skin (V–VI)** | 0.789 | ~0.80+ | derm-SSL features + MRA-MIDAS/DDI-2 + tone calibration |
| Melanoma recall | 0.870 | ≥0.90 locked | operating point on the better model |
| Calibration ECE | 0.042 | hold/↓ | per-class temp |
| Conformal set size | 5.81 | ↓ | tighter features → smaller sets |
| Fairness (V–VI) | underpowered | powered | DDI-2 + per-tone calibration/abstention |

*Targets, not promises — every one gets an honest CI and a "did it move?" gate.*

---

## 4. The LLM — how it gets strong
- **Milestone-2 (measurable):** learned image×history fusion on SCIN's symptom fields → the provable macro-F1 gain.
- **Reasoning:** gemini-2.5-pro for the interview/fusion; **RAG** on real derm criteria (7-point checklist, ABCD,
  Fitzpatrick-specific presentations) to cut hallucination.
- **Smarts:** confidence-gated questioning (only when the image model is unsure) · information-gain question
  selection · **red-flag escalation** · multi-image/temporal ("has the mole changed?") · skin-tone-aware phrasing.
- **Safety:** never overrides a cancer flag; abstains and defers; keeps the calibrated model as the decision-maker.

---

## 5. New-model / restart options (exhaust the architecture question)

Given the frozen-probe insight, "restart training" mostly means **get better frozen features**, not heavier fine-tuning.

1. **Derm SSL foundation model — MAE (primary) / CAE (backup).** Continued self-supervised pretraining on our own
   ~270k derm images → a dermatology-specialized backbone (the UNI recipe, for skin). *The highest-ceiling restart;
   needs the A100.* Expected to lift **every** metric because it improves the shared representation.
2. **Bigger/better off-the-shelf backbones as frozen extractors:** EVA-02, ConvNeXt-V2, DINOv3, DINOv2-registers,
   at 518. Cheap to test via the probe bake-off (Week 1).
3. **Derm-specific FMs as frozen extractors:** PanDerm (load correctly via its own code), DermINO, Google
   Derm-Foundation. (PanDerm/DermINO are non-commercial → research-oracle only.)
4. **Multi-backbone frozen ensemble** (e.g. derm-MAE + DINOv2 + Derm-Foundation) — cheap, usually +1–2 pts.
5. **Higher resolution end-to-end** (518, maybe 768 on A100) — already proven to help DDI.
6. **Dermatopathology branch (stretch):** if biopsy slides become available, **UNI** becomes directly usable and a
   histology head could add a confirmatory signal — a genuinely novel multi-scale (clinical photo + histology) angle.

**Decision rule:** the Week-1 bake-off + Week-2 derm-MAE tell us the best backbone; we build everything on the winner.

---

## 6. Academic write-up (after the numbers refresh)
Contributions to foreground (all real, all ours):
1. **Majority-class collapse — diagnosis + single-mechanism fix** (eczema 0.004→0.503).
2. **Leak-free evaluation** (patient-disjoint CV + single-shot external) — harder/honest protocol.
3. **A full trust layer** (calibration + conformal + per-Fitzpatrick fairness) that ~no competitor publishes.
4. **The DDI dark-skin result + its lesson:** frozen-features-probe > fine-tune under distribution shift — with the
   honest ceiling (~0.78 V–VI) and the data-scarcity map from `DARKSKIN_RESEARCH.md`.
5. (If done) **derm-MAE** foundation model + the backbone bake-off.
Deliverable: finalize `RESEARCH_REPORT_UPDATED.md` → submission-ready (target: a fair-ML / medical-imaging venue or
npj Digital Medicine-style journal). Includes the honest negative-then-positive DDI arc — reviewers value that.

## 7. Productionization (ONLY after your sign-off)
Frozen-backbone → task heads (cancer/DDI/25-class) + LLM interview + OOD gate + per-tone abstention, on-device
(ONNX/CoreML) where possible. Held until you're happy with every metric.

## 8. The three links — where each lands
- **MAE** → primary SSL recipe for `derm-MAE` (Week 2). Simple, A100-friendly, weights to warm-start.
- **CAE** → backup SSL recipe (better linear-probe; more complex — needs the tokenizer).
- **UNI** → *not* a derm backbone (pathology); used as (a) the design template validating the SSL plan, (b) a free
  probe-benchmark, (c) the enabler *if* we add a dermatopathology branch later.

---

### The one-paragraph version
We turn the DDI breakthrough into the project's method: **best frozen features + light heads.** Week 1 we bake off
every backbone (incl. the three links); Week 2 we build a dermatology MAE on the A100 for even better features and
build LLM Milestone-2; Week 3 we re-derive **every** metric on the winner + add MRA-MIDAS/DDI-2/ASAN; Week 4 we
refresh the trust layer, harden the LLM, and draft the paper. Then — and only then — we productionize.
