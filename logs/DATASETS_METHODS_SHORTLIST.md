# Datasets + Methods Shortlist (2026-07-06)

Consolidated from this session's web-verified research (`DARKSKIN_RESEARCH.md`, `IMPROVEMENT_ROADMAP.md`,
`DATA_INVENTORY.md`). We already hold ~370k images (~266k labeled). Table marks HAVE vs NEW + priority.
Re-verify exact access at pull time.

## The 20-dataset shortlist (light + dark)

| # | Dataset | Size | Skin tone | Modality | Adds / helps | Access | Have? | Priority |
|---|---|---|---|---|---|---|---|---|
| 1 | **DDI-2** (Daneshjou) | 665 | dark/Asian FST III–IV | clinical, biopsy | mid-tone clinical cancer + clean external | daneshjoulab.github.io/ddi2-dataset (RUA) | no | **PULL** |
| 2 | **MRA-MIDAS** (Stanford) | ~1,290 | ~17% FST III–VI | **clinical + dermoscopy paired** | clinical-photo cancer + modality bridge | Stanford AIMI portal (RUA) | no | **PULL** |
| 3 | **ASAN test set** (Korea) | ~1,276 | Asian FST III–IV | clinical | pigmented-skin clinical BCC/SCC/mel (AUROC ~0.96) | public test split (verify) | no | **PULL** |
| 4 | **PASSION** (Sub-Saharan Africa) | 4,901 | **dark FST V–VI** | clinical | dark-skin *inflammatory* (eczema/tinea/scabies/impetigo = Indian-relevant) + SSL | passionderm.github.io (non-comm) | no | **PULL** |
| 5 | **HIBA** (Argentina) | 253 mel + 346 clin | Latino FST III–IV | clinical + dermoscopy | clean clinical melanoma, easy fold-in | ISIC collection 251, CC-BY | no | **PULL** |
| 6 | **Fitzpatrick17k + CleanPatrick** | 16.5k | **FST I–VI (2.2k dark)** | clinical | dark-skin + fairness + 25-class | HF Digital-Dermatology/CleanPatrick | **yes (cleaned)** | use fully |
| 7 | **SCIN** (Google) | 10k | diverse (Monk) | clinical | diverse inflammatory + skin tone | GCS bucket | **yes** | use fully |
| 8 | **DDI** | 656 | dark-balanced | clinical, biopsy | dark-skin cancer eval | Stanford (have) | **yes** | eval anchor |
| 9 | **DermaCon-IN** | 5,450 | Indian | clinical | Indian 25-class | Harvard Dataverse | **yes** | use fully |
| 10 | **PAD-UFES-20** | 2,298 | Brazilian light | clinical | clinical cancer + rich metadata | Mendeley zr7vgbcyr2 | **yes** | use fully |
| 11 | **DERM12345** (Türkiye) | 12k | mixed | **dermoscopy** | **acral melanoma** (over-rep in dark skin) | Harvard Dataverse, CC-BY | no | optional |
| 12 | **PH2** | 200 | light | dermoscopy | small clean melanoma benchmark | public | no | optional |
| 13 | **Dermofit** | 1,300 | light | clinical | 10-class incl. cancer | Edinburgh (paid licence) | no | optional |
| 14 | **SD-198 / SD-260** | 6k / — | mixed | clinical | 198-class breadth (murky provenance) | email maintainers | no | optional/skip |
| 15 | **ISIC Archive (more)** | +up to ~500k | light | dermoscopy | pure cancer/9-class volume + SSL | ISIC public API | partial (~230k) | optional (SSL only) |
| 16 | **HAM10000 / BCN20000** | 10k / 19k | light | dermoscopy | standard cancer benchmarks | ISIC | **yes** | — |
| 17 | **Derm7pt** | 2k | light | dermo+clinical | 7-point checklist (interpretability) | github | **yes** | — |
| 18 | **MED-NODE** | 170 | light | clinical | tiny mel/nevus | public | **yes** | — |
| 19 | **Monkeypox / MSLD v2** | 41k | diverse | clinical | viral exanthems (tropical) | Kaggle | **yes** | — |
| 20 | **Pipsqueak** | 16 (→80 aug) | dark | clinical | 16 real dark-skin melanomas | author request | no | **EVAL probe only** |

## Light vs dark — the honest split of NEW additions
- **New light-skin:** DERM12345 (dermoscopy acral), PH2, Dermofit, more-ISIC → volume for cancer/9-class, **but we're already saturated there** — marginal.
- **New dark/diverse (the valuable part):** **DDI-2 (665) + MRA-MIDAS (~1.3k) + ASAN (~1.3k) + PASSION (4.9k) + HIBA (~600)** ≈ **~8–9k new diverse/dark clinical images.** Of these, only DDI-2/MRA-MIDAS/ASAN/HIBA carry *cancer*; PASSION is inflammatory (helps the Indian 25-class tail: eczema/tinea/scabies/impetigo).
- **Reality:** even pulling everything adds ~8–9k diverse images and <2k *dark-skin cancer* — confirming the wall. It broadens robustness + powers fairness; it does not 10× the dark-skin cancer signal.

## Training methods — ranked (beyond frozen-probe)
**Do BEFORE any SSL (cheap, laptop, extend Lever A):**
1. **Probe/head upgrades** — MLP / attentive-pooling / cosine head / kNN instead of plain linear. Often +1–3 pts, minutes.
2. **Multi-backbone frozen ensemble** — DINOv2 + (registers / EVA-02 / ConvNeXt-V2) features concatenated. +1–2 pts, cheap.
3. **Higher resolution** — 518→768 (chunked). Proven to help DDI.
4. **Multi-task heads on shared frozen features** — 9-class + 25-class + cancer jointly. Cheap, regularizing.
5. **Semi-supervised / pseudo-labeling** — self-train on the ~100k weakly/un-labeled (Mpox, DermNet) with the frozen probe. Medium.

**A100 (a few days each, higher ceiling):**
6. **Continued SSL — MAE (primary) / CAE** on our 370k → derm backbone (the big lever).
7. **LoRA / adapters (LIFT)** — the *safe* fine-tune: adapts features without the catastrophic degradation full fine-tuning caused.
8. **Feature distillation from a derm-FM** (PanDerm, offline) into our student.
9. **Test-time adaptation** — for the train-dermoscopy → test-clinical shift.

**Key insight:** methods 1–4 are near-free extensions of Lever A and may beat a full SSL run per hour spent. **LoRA/adapters** are the fine-tuning we *should* have used (they don't wreck frozen features).

## Verdict: proceed with Lever A NOW
- Every PULL-NOW dataset needs **registration/download that takes days–weeks** (Stanford AIMI RUA for DDI-2/MRA-MIDAS; PASSION/HIBA download). **Lever A needs none of them.**
- **Recommendation:** (1) **start Lever A today** — frozen-DINOv2@518 + probe/MLP across 9-class, 25-class, cancer on the 266k we have (immediate gains, laptop); (2) **in parallel**, register **Stanford AIMI** (unlocks DDI-2 + MRA-MIDAS) and grab **PASSION + HIBA** (no gate); (3) fold those in when they arrive + add methods 1–4. Don't wait on data to start.
