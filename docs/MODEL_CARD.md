# Model Card — `models/clinical_25class.pkl` (MNKAI-III clinical frozen-probe)

*Skeleton authored as part of the reproducibility package (R13). Numbers below are
the honest, leakage-corrected values recorded elsewhere in the repo; verify against
`PROJECT_REPORT.md` / `PAPER_DRAFT.md` before external use.*

> **Decision support, not diagnosis.** This is a research prototype. It does not
> provide a medical diagnosis and must not be used as an autonomous diagnostic
> device. All outputs require review by a qualified clinician.

## 1. Model overview

| Field | Value |
|---|---|
| Artifact | `models/clinical_25class.pkl` |
| Task | 25-class dermatological condition classification (single clinical/dermoscopic image) |
| Method | **Frozen-probe**: frozen self-supervised backbone + lightweight trained head. No backbone fine-tuning. |
| Backbone | `vit_base_patch14_dinov2.lvd142m` (DINOv2 ViT-B/14), **frozen**, native **518 px**, 768-d pooled features |
| Head | `StandardScaler` → `MLPClassifier((512,))` (scikit-learn) |
| Calibration | Temperature scaling (single global `T`) + per-class logit bias (coordinate-ascent on OOF) |
| Uncertainty | RAPS split-conformal prediction sets (`k_reg=2`, `lambda=0.05`, `alpha=0.10`, target 90% coverage) |
| Producer script | `scripts/_save_deployable.py` |
| Serving | `demo_app/pipeline.py` (`ClinicalPipeline.assess`) |

### Classes (25)
`mel, nv, bcc, scc, df, bkl, ak, vasc, normal, acne, alopecia, autoimmune_systemic,
bacterial_infection, contact_dermatitis, drug_eruption, eczema, lichen_planus,
pemphigus, psoriasis, rosacea, scabies, tinea, urticaria, viral_infection, vitiligo`

Malignant / referral-relevant classes (pooled for the safety gate):
`mel, bcc, scc, ak`.

## 2. Intended use

- **Intended:** research and educational decision support; triage assistance that
  surfaces a calibrated differential, a conformal candidate set, an abstention
  signal, and (when uncertain) history questions — always under clinician review.
- **Out of scope / not intended:** autonomous diagnosis; primary basis for treatment
  decisions; use on image types outside the training distribution (e.g. histopathology,
  non-skin images); populations or devices materially different from the evaluation data
  without local validation.

## 3. Training & evaluation data

~266k labeled, de-duplicated images spanning dermoscopy (ISIC-2024/2019, BCN20000)
and clinical photos (Fitzpatrick17k [CleanPatrick-cleaned], SCIN, DermaCon-IN,
PAD-UFES-20, DDI, Mpox/MSLD). Splits are group-aware (patient/lesion-disjoint) with a
source-domain external holdout. A feature-space duplicate audit removed ~33% train↔test
duplicates from the multi-source external set (cosine > 0.99); reported numbers are
post-dedup. See `PAPER_DRAFT.md` §2 and §6 for provenance and the leakage correction.

Dark-skin (Fitzpatrick IV–VI) coverage is ~15k images — the binding scarcity of the
project and the reason fairness subgroups are underpowered (see §5).

## 4. Performance (honest, leakage-corrected)

| Metric | Value |
|---|---|
| 25-class macro-F1 (deduped external, n≈3,813) | **0.573** [0.552–0.591] |
| 9-class macro-F1 (OOF CV) | 0.660 [0.654–0.665] |
| Calibration ECE (before → after temp-scale) | 0.21 → 0.042 (T=2.2, deterministic R02) |
| Conformal set size @ 90% target coverage | ~4 |
| Abstention: accuracy retained after referring ~30% | 0.766 |

**Dark-skin cancer (DDI, separate frozen multi-backbone ensemble — NOT this 25-class
head; see `logs/DDI_ENSEMBLE_RESULT.md`):** overall AUROC 0.826 [0.789, 0.862];
Fitzpatrick V–VI 0.872 [0.808, 0.929]. The single-DINOv2-B probe underlying the
deployable stack reports overall 0.748 / V–VI 0.789.

*All numbers are as recorded in the project reports at the time of writing; they are
non-deterministic across re-runs until the determinism re-run (R02) is completed, so
treat them as point estimates with the stated CIs.*

## 5. Subgroup performance & fairness

- Per-Fitzpatrick calibration and reporting are part of the trust layer.
- **Underpowered subgroups:** 25-class per-tone estimates are small-n and wide —
  e.g. FST I–III balanced-accuracy ≈ 0.264 (n≈642), FST IV–VI ≈ 0.454 (n≈135) in the
  external set. These are reported honestly, not suppressed; do not over-interpret
  point estimates at these sample sizes, and note multiple-comparison exposure across
  tone bins.
- The strong dark-skin result is the **DDI cancer ensemble**, a different model from
  this 25-class head; do not transfer its AUROC to the 25-class task.

## 6. Operating point & decision logic

From `demo_app/pipeline.py`:
- **Differential:** top-5 by temperature-scaled + per-class-bias-adjusted probability.
- **Conformal set:** RAPS set on the calibrated probabilities.
- **Uncertain** when conformal set size > 4 OR top-1 margin < 0.15.
- **Malignancy gate:** pooled probability over `{mel, bcc, scc, ak}` ≥ 0.30 → refer.
- Melanoma sensitivity is operated ≥ 0.90 with abstention (see abstention module).
- Recommendation ladder: refer (possible malignancy) → refer (model not confident) →
  decision-support suggestion.

## 7. Failure modes & limitations

- Out-of-distribution inputs (non-clinical photos, unusual devices, non-skin images)
  can yield confident-but-wrong outputs; the conformal/abstention gates mitigate but do
  not eliminate this.
- Dark-skin 25-class subgroups are underpowered; malignancy screening on dark skin
  relies on the separate DDI ensemble, itself data-limited (~0.78–0.87 AUROC).
- No biopsy confirmation and no prospective/clinical validation.
- Results are not yet deterministic across re-runs (R02 pending).
- Rare conditions outside the 25 classes are mapped to the nearest class or abstained,
  not recognized as novel.

## 8. Reproducibility

- Environment: `requirements.txt` (pinned ranges; freeze `requirements.lock.txt` on the
  training machine for exact reproduction).
- Rebuild the artifact: `python scripts/_save_deployable.py` (consumes
  `eval/probe/feats_25class_518.npz`).
- Ethics: EXIF/metadata stripped at ingestion (`src/mnkai/data/qc.py: strip_exif`).

## 9. Contact & governance

Single-researcher academic prototype. Any deployment beyond research requires local
validation on the target population and device, clinician oversight, and a documented
risk assessment. **Decision support, not diagnosis.**
