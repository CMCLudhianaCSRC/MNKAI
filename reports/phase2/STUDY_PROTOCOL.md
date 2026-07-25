# STUDY PROTOCOL

## Building a North Indian Dermatology Photograph Dataset with a Biopsy-Confirmed Skin Cancer Arm
### A prospective, two-site image data collection sub-study, with a retrospective pathology-archive arm

**Principal investigator:** Ujjwal Thakran, final-year MBBS, Christian Medical College Ludhiana
**Document:** Master study protocol, Version 1.0
**Prepared for:** Institutional Ethics Committee submission and the college approval file
**Duration:** Approximately 13 months (1 month setup, 12 months accrual, analysis thereafter)
**Sites:** Two (Christian Medical College Ludhiana and one collaborating North Indian centre, to be named in the site addendum)

Companion documents, submitted together with this protocol: `PHOTO_CAPTURE_SOP.md` (how photographs are taken and handled), `MOUNT_SPEC.md` (the camera spacer), `CRF_METADATA.md` (the case record form and the exact fields recorded), and `LABELLING_SOP.md` (how a diagnosis is assigned to an image).

---

## 1. Background and rationale

Skin disease is one of the commonest reasons patients attend an out-patient department in India, and dermatologists are concentrated in large cities. There is genuine interest in whether a photograph taken on an ordinary smartphone can help a non-specialist decide who needs urgent referral — but any such system can only be trusted if it has been tested on patients who look like ours. At present it has not been. The evidence gap is specific and can be stated plainly.

**There is no dermatology photograph dataset from North India.** The closest existing resource is **DermaCon-IN** (Nature *Scientific Data*; arXiv:2506.06099), a genuinely valuable South Indian collection of 5,450 smartphone images from 3,002 patients, dermatologist-labelled, with Fitzpatrick and Monk skin-tone recording. It establishes that this kind of collection is feasible in an Indian clinic using ordinary consumer phones. But it covers a different population and, importantly, **it has no biopsy-confirmed cancer arm** — the diagnoses are clinical.

**There is no Indian dark-skin skin-cancer benchmark of any kind.** The only substantial public collection of biopsy-confirmed clinical photographs across darker skin is **DDI** (Diverse Dermatology Images; Daneshjou et al., *Science Advances*, 2022) — 656 photographs, all biopsy-confirmed, deliberately balanced across skin tone, assembled retrospectively from the pathology archive of a **single United States academic clinic**. In the completed primary research (Section 2), our model's dark-skin cancer performance — area under the ROC curve **0.872, 95% CI 0.808 to 0.929, on Fitzpatrick V–VI** — rests entirely on that one American source. A result from one overseas clinic cannot be assumed to hold in a Punjabi or wider North Indian out-patient population.

**The general-dermatology result is underpowered on darker skin.** In the same completed work, the 25-condition classifier was tested externally, but that external test contained only about **135 darker-skin images spread across 25 conditions** — roughly five images per condition. Nothing reliable can be said per condition from five images. This is the single clearest weakness of the completed research, and it is a data problem, not a method problem. No amount of further computing will fix it.

**A workable precedent exists for how to fix it.** **PAD-UFES-20** (Brazil) collected smartphone images through a purpose-built capture app alongside 26 structured metadata fields, and biopsy-proved every cancer (about 58% of all samples were biopsied). DDI shows how a biopsy-confirmed arm can be assembled from an existing pathology archive rather than waiting for cases to walk in. DermaCon-IN shows that mixed consumer smartphones under ordinary clinic lighting are acceptable, and that device variation is best treated as clinical realism and simply recorded.

Our contribution is therefore the combination that does not yet exist anywhere: **North Indian patients + a biopsy-confirmed cancer arm + prospective validation of a model built before the data were collected.**

---

## 2. Objectives

**PRIMARY objective (already completed; stated here for context, not for approval).** The main research is a computer-vision study using modern image-analysis models to classify skin conditions from clinical photographs. It is complete and internally validated on public research datasets only, with no patient contact and no identifiable data. Its two headline findings are the dark-skin cancer result (AUROC 0.872, 95% CI 0.808–0.929, Fitzpatrick V–VI, from DDI) and a 25-condition classifier evaluated on an independent external test set.

**SECONDARY objective (the subject of this application).** To prospectively collect a consented, dermatologist-labelled photograph dataset of **3,000 to 5,000 images** from two North Indian centres over approximately one year, together with a retrospective, biopsy-confirmed skin-cancer arm drawn from the histopathology archive.

**How the sub-study strengthens the primary work.** It supplies the three things the completed research cannot supply for itself:

1. An **Indian test population** on which the existing model can be evaluated without retraining it — a true external validation.
2. Enough darker-skin images per condition to replace the ~5-per-condition estimate with something a reader can rely on.
3. An **Indian biopsy-confirmed cancer set**, so that the 0.872 dark-skin figure can be checked against Indian histopathology rather than a single American clinic.

The sub-study is a component of the main research project, not a separate project. Nothing in the primary objective depends on its outcome; the sub-study only tests and extends it.

---

## 3. Study design

A **two-track observational study**, with no intervention, no change to patient management, and no randomisation.

**Track A — Prospective arm (12 months, both sites).** Consecutive consenting patients attending the dermatology out-patient department are photographed at the time of their routine visit, with their diagnosis recorded as made by the treating dermatologist. This track supplies the common inflammatory and infectious conditions that dominate an Indian OPD.

**Track B — Retrospective pathology-archive arm (both sites).** Skin cancer is comparatively uncommon in brown skin, and the required number of malignant cases will not accrue prospectively within one year. We will therefore identify patients whose skin lesion was **biopsied and histopathologically confirmed**, and use the clinical photograph taken at the time of that encounter where one exists in the medical record, under a separate waiver-of-consent request (Section 12). DDI itself was constructed this way, and this is the only realistic route to a biopsy-confirmed malignant set within the study period.

**Reference standard.** Track A diagnoses are the treating dermatologist's clinical diagnosis, with a second independent dermatologist review for a random sample and for all lesional/suspicious cases (see `LABELLING_SOP.md`). Track B diagnoses are histopathology, which is the accepted gold standard.

**Data splitting (stated here so that the ethics committee understands what the data will be used for).** All analysis splits will be **patient-level** — images from one patient never appear in both a development and a test set. In addition, the **final three months of prospective accrual will be frozen untouched as a prospective test set** and not examined until analysis. This is a safeguard against self-deception, not a statistical nicety.

---

## 4. Participants

**Inclusion criteria**
- Patients of any age attending the dermatology OPD at a participating site with a visible skin, hair or nail condition.
- Written informed consent from the patient; for minors, written consent from a parent or guardian plus the child's assent where age-appropriate.
- For Track B, a histopathology report confirming the lesion diagnosis.

**Exclusion criteria**
- Refusal or withdrawal of consent, at any point and for any reason.
- Any situation where photography would delay or interfere with clinical care, including acutely unwell patients and emergencies.
- Lesions that cannot be photographed without capturing the face or another identifying feature, where the patient is not willing for that region to be imaged.
- Patients unable to give consent and without an available legally acceptable representative.

**Consent process.** Consent is taken by the treating clinician or a trained study coordinator, in the patient's own language (Punjabi, Hindi or English), **before the camera is brought out**. The patient is told what will be photographed, that images are stored under a study number and not their name, that no face will be photographed, that the images may be shared as a research dataset, and that declining will not affect their treatment in any way. Consent is documented on the IEC-approved form and filed securely. Withdrawal is honoured at any time, and the images are removed. No photograph is ever taken first and consented afterwards.

---

## 5. Sample size and justification

These are **planning estimates**, presented to justify the targets rather than as guarantees. They follow standard precision-based reasoning, and the arithmetic is given openly.

**Per-condition target: about 96 to 100 test-set positives per condition.** To estimate a sensitivity of about 0.80 with a 95% confidence interval of roughly ±0.08, the usual proportion formula gives

> n = 1.96² × 0.80 × 0.20 / 0.08² = 3.84 × 0.16 / 0.0064 ≈ **96**

so we set a working target of **at least 60–100 test images per condition**, with 96–100 as the number that supports a genuinely powered per-condition claim. Conditions that fall short will be reported as underpowered rather than quietly included.

**Malignant target: at least 170 biopsy-confirmed cancers.** DDI contains 171 malignant lesions. To be able to say that our Indian set is at least as informative as the benchmark the primary result rests on, we set **≥170** as the target for Track B, with **≥100** as the minimum below which the cancer arm should be described as preliminary.

**Skin-tone target: at least 150, ideally 300, images per skin-tone bin in the test set.** A fairness comparison is a comparison of two proportions. To detect a difference of about 10 percentage points between two skin-tone groups at 80% power and a 5% significance level requires roughly 150 to 300 observations per group depending on the baseline rate. Below 150 per bin, a real gap of that size will usually be missed, which is precisely what happened with the ~135 darker-skin images in the completed external test.

**Overall target: 3,000 to 5,000 images.** This follows from the per-condition and per-tone targets applied across the condition list in Section 6, and is consistent in scale with DermaCon-IN (5,450 images). Accrual of roughly 12 to 20 patients per week per site over 12 months, at 3 to 4 images per lesion, reaches this range with reasonable margin.

---

## 6. Condition list

The list reflects a realistic North Indian OPD case mix. Final label definitions are in `LABELLING_SOP.md`.

| Group | Conditions |
|---|---|
| Eczematous | Atopic dermatitis, contact dermatitis, seborrhoeic dermatitis, hand eczema |
| Fungal | Tinea corporis/cruris/pedis/faciei, onychomycosis, pityriasis versicolor, candidiasis |
| Papulosquamous | Psoriasis, lichen planus, pityriasis rosea |
| Pigmentary | Vitiligo, melasma, post-inflammatory hyperpigmentation |
| Acne and appendageal | Acne vulgaris, rosacea, alopecia areata |
| Urticaria and reactive | Acute and chronic urticaria, drug eruption, erythema multiforme |
| Infestation | Scabies, pediculosis |
| Infection (bacterial/viral) | Impetigo, cellulitis, folliculitis, herpes simplex/zoster, viral warts, molluscum |
| Hansen's disease | Leprosy across the spectrum (a clinically important and under-represented condition) |
| **Malignant (Track B, biopsy-confirmed)** | Basal cell carcinoma, squamous cell carcinoma, melanoma, actinic keratosis |
| Benign lesions | Melanocytic naevus, seborrhoeic keratosis, dermatofibroma, haemangioma, epidermal cyst |

---

## 7. Data collected

**Images.** Three photographs per lesion (context, close-up, dermoscopic), with a fourth angled view for raised or nodular lesions. Capture is by smartphone with a minimum specification of a **12 MP or better main camera with autofocus**, used on the **main lens only** in default photo mode (not the 48/108 MP high-resolution mode), stored at high JPEG/HEIF quality (≥90) with the original file preserved, and delivering at least 1,500 pixels across the lesion region with the lesion filling at least half the frame. A 3D-printed spacer fixes camera-to-skin distance, and a colour-reference card and adhesive ruler appear in every close-up. Full detail is in `PHOTO_CAPTURE_SOP.md` and `MOUNT_SPEC.md`. Following DermaCon-IN's precedent, phones are **not** required to be identical; the device model is recorded for every image.

**Metadata.** Recorded on the case record form defined in `CRF_METADATA.md`: study ID, site, date, age band, sex, anatomical site, lesion number, duration, symptoms, prior treatment, Fitzpatrick skin type and Monk tone, clinical diagnosis, diagnostic confidence, whether biopsied, histopathology result where applicable, device model, and whether a dermoscope was used. No name, no hospital number, no address, no phone number.

---

## 8. Outcomes and deliverables

1. **The dataset itself** — a de-identified, dermatologist-labelled North Indian dermatology image collection with a biopsy-confirmed cancer arm, intended for release as a citable research resource under a data-use agreement, subject to IEC approval of the release terms.
2. **External validation of the completed primary model on Indian patients**, reported honestly: overall performance, per-condition performance where powered, and performance by skin-tone bin.
3. **A prospective evaluation** on the frozen final-three-month test set.
4. Publications and academic presentation of both the dataset and the validation.

No clinical decision will be made using the model during this study. The model is not shown to treating clinicians and does not influence patient care.

---

## 9. Timeline

| Period | Activity |
|---|---|
| Month 0 | IEC submission and approval; site agreements; equipment purchase and assembly; staff training on `PHOTO_CAPTURE_SOP.md`; pilot of 30–50 images for quality check; approved storage provisioned |
| Months 1–9 | Prospective accrual, both sites; monthly quality audit; Track B pathology-archive retrieval runs in parallel |
| Months 10–12 | Continued accrual; **these three months are frozen as the prospective test set and not analysed** |
| Month 12 | Accrual closes; database lock; second-reader adjudication completed |
| Months 13+ | Analysis, external validation of the primary model, dataset curation and release preparation, manuscript |

---

## 10. Data management and privacy (DPDP Act 2023)

India's Digital Personal Data Protection Act, 2023 applies to this study, and photographs of skin disease are personal data.

- **Study IDs only.** Files are named `SITE-PATIENTID-LESIONNO-SHOTTYPE-YYYYMMDD`. The patient's name and hospital number never appear in a file name, a folder name or an image.
- **The identity ↔ study-ID key** is held separately by the study coordinator, access-controlled, and is not stored with the images.
- **No faces**, no tattoos, no birthmarks outside the lesion, no name bands, no jewellery, no background persons, and no documents bearing names are permitted in frame.
- **Encryption at rest and in transit**, on **India-resident, institution-approved storage**. Personal Google Drive, Dropbox, personal laptops and personal phones are prohibited, as is transfer of any study image via WhatsApp, Telegram or email.
- **Location tagging is disabled** on all study devices; images are transferred and deleted from the phone at the end of every clinic session, including the device trash.
- **Withdrawal** is actioned within the same working week, with the images deleted and the deletion logged.
- **Retention and access** follow institutional policy; any external data sharing occurs only in de-identified form, under a data-use agreement, and only with IEC approval.

---

## 11. Risks and mitigation

| Risk | Mitigation |
|---|---|
| Loss of privacy through a recognisable image | No faces or identifying marks in frame; study IDs only; encrypted India-resident storage; separate key file |
| Patient feels obliged to consent | Consent taken by trained staff, in the patient's language, with explicit statement that refusal does not affect treatment |
| Photography delays clinical care | Photographs taken after the consultation; abandoned immediately if care would be affected |
| Discomfort or embarrassment, especially at sensitive sites | Minimum necessary images, draping, chaperone present, site-specific consent |
| Poor image quality wasting patient participation | Bedside quality checklist in `PHOTO_CAPTURE_SOP.md`; retake immediately; monthly audit |
| Under-accrual of malignant cases | Track B pathology-archive arm run from Month 1; ≥100 treated as the minimum reportable threshold |
| Under-exposure of darker skin by automatic camera settings | Explicit exposure-correction step trained into the SOP and checked at the bedside |
| Device or infection-control lapse | Spacer, dermoscope plate and phone wiped between every patient |

Because the study involves photographs only, with no intervention, no specimen collection beyond what is clinically indicated, and no change to management, **the risk to participants is minimal and is confined to privacy**.

---

## 12. Ethics

- IEC approval is required at both sites **before any image is collected**. No collection begins on either site until that site's approval is in hand.
- Written informed consent is obtained from every Track A participant before photography, using the IEC-approved form and information sheet in the participant's own language.
- For the **Track B retrospective arm**, we request a **waiver of consent** for the use of already-existing clinical photographs and their linked histopathology reports, on the grounds that the data already exist, re-contact of patients biopsied in the past is often impracticable, no new procedure is involved, only de-identified data are used, and the research cannot be conducted otherwise. This request is made explicitly and is for the committee to decide; the prospective arm proceeds independently of it.
- The study follows the ICMR National Ethical Guidelines for Biomedical and Health Research Involving Human Participants and the Declaration of Helsinki.
- No incentive or payment is offered.
- The study is observational; no participant is denied or delayed any care.
- Results, including negative and underpowered results, will be reported honestly. Conditions or skin-tone bins that do not reach the targets in Section 5 will be reported as underpowered rather than presented as findings.

---

## 13. Budget

Per capture station, indicative Indian prices:

| Item | Indicative cost | Notes |
|---|---|---|
| Study smartphone (≥12 MP main camera, autofocus) | Rs 14,000–16,000 | One per station |
| 3D-printed spacer/mount | Rs 100 self-printed (filament ≈ Rs 30–100) / Rs 150–400 via a print service | See `MOUNT_SPEC.md` |
| Colour-reference card + adhesive rulers | Rs 100–200 | Consumable; rulers are single-use |
| Clip-on dermoscope | Rs 5,000–15,000 | **Lesion/cancer stations only**; essential for that arm |
| **Per lesion station (with dermoscope)** | **≈ Rs 19,200–31,600** | |
| **Per general station (no dermoscope)** | **≈ Rs 14,200–16,600** | |

One-time, optional: an entry-level 3D printer at **Rs 12,000–18,000** if mounts are printed in-house. This pays for itself only if many mounts or replacements are needed; otherwise a print service is more sensible.

Additional recurring costs: alcohol wipes, drapes, gloves, printing of consent forms and CRFs, and institutional secure-storage charges, all of which fall within routine departmental consumables.

---

## 14. Team and responsibilities

| Role | Responsibility |
|---|---|
| Principal investigator (Ujjwal Thakran) | Overall conduct, protocol adherence, analysis, reporting, IEC correspondence |
| Supervising faculty / guide | Scientific and clinical oversight; approval of the sub-study; adjudication of difficult labels |
| Site co-investigator (one per site) | Local conduct, consent oversight, local IEC liaison |
| Treating dermatologists | Clinical diagnosis, diagnostic confidence, decision to biopsy |
| Second-reader dermatologist | Independent review per `LABELLING_SOP.md` |
| Study coordinator (one per site) | Consent documentation, CRF completeness, study-ID key custody, daily image transfer and verification, withdrawal handling |
| Trained nursing/coordinator staff | Photography per `PHOTO_CAPTURE_SOP.md`, bedside quality checks, equipment hygiene |
| Data custodian | Approved storage, access control, encryption, backup, audit log |
| Pathology department liaison | Track B case identification and histopathology verification |

---

## 15. References

1. Daneshjou R, Vodrahalli K, Novoa RA, et al. Disparities in dermatology AI performance on a diverse, curated clinical image set (DDI). *Science Advances*. 2022. — 656 biopsy-confirmed clinical photographs, 171 malignant, single US academic clinic, skin-tone balanced, assembled from a pathology archive.
2. DermaCon-IN: a South Indian dermatology image dataset. *Scientific Data* (Nature); arXiv:2506.06099. — 5,450 smartphone images, 3,002 patients, dermatologist-labelled, Fitzpatrick and Monk tone, mixed consumer phones under ambient clinical light; no biopsy-confirmed cancer arm.
3. Pacheco AGC, et al. PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones. *Data in Brief*. 2020. — Purpose-built smartphone capture app, 26 metadata fields, all cancers biopsy-proven (~58% of samples biopsied).
4. Tschandl P, Rosendahl C, Kittler H. The HAM10000 dataset. *Scientific Data*. 2018. — Released at 600 × 450 pixels; cited here as evidence that resolution is not the limiting factor for usable dermatology images.
5. Groh M, et al. Evaluating deep neural networks trained on clinical images in dermatology with the Fitzpatrick 17k dataset. 2021.
6. Digital Personal Data Protection Act, 2023 (India).
7. ICMR National Ethical Guidelines for Biomedical and Health Research Involving Human Participants, 2017.
8. World Medical Association. Declaration of Helsinki.

**Companion study documents:** `PHOTO_CAPTURE_SOP.md`, `MOUNT_SPEC.md`, `CRF_METADATA.md`, `LABELLING_SOP.md`.

---

*Version 1.0. Any amendment to this protocol will be submitted to the Institutional Ethics Committee for approval before implementation.*
