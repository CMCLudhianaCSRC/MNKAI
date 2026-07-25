# LABELLING SOP
## North India Dermatology Photo Data Collection Study
**Standard Operating Procedure for turning an image into a trustworthy diagnosis label**

Version 1.0 | For dermatologists, pathology liaison staff and the study coordinator | Two sites | Approx. 1 year

---

## 1. Why labelling quality decides the value of this dataset

A photograph on its own is not data. It becomes data when a diagnosis is attached to it, and the dataset is only ever as good as those diagnoses. A perfectly captured image with a careless label is worse than no image at all, because it silently corrupts everything built on top of it.

The single most important idea in this document is this: **a clinically-labelled image and a biopsy-confirmed image are not the same grade of evidence, and must never be recorded as if they were.** A dermatologist writing "seborrhoeic keratosis" is making a judgement. A pathologist reporting the excised tissue is reporting a fact. Both belong in this dataset, but they carry different weight, and any honest analysis must be able to tell them apart. We therefore store the *basis* of every diagnosis alongside the diagnosis itself.

This matters most for our stated differentiator. The dark-skin skin-cancer result (AUROC 0.872, 95% CI 0.808–0.929 on Fitzpatrick V–VI) rests on DDI — 656 biopsy-confirmed images from one US academic clinic — and the 25-condition per-skin-tone reading rests on only about 135 darker-skin images in the external test set, roughly 5 per condition. DermaCon-IN (5,450 images, 3,002 patients, South India) is dermatologist-labelled but has **no biopsy-confirmed cancer arm** at all. Our claim to novelty is North India **plus** a biopsy-confirmed cancer arm. If our malignant labels are not genuinely histopathology-backed, we have no claim.

---

## 2. The label grades

Every image carries a field called **`diagnosis_basis`**. It is mandatory. One of four values:

| Grade | `diagnosis_basis` value | What it means | When used |
|---|---|---|---|
| **A** | `histopathology` | Tissue diagnosis from the pathology report. The reference standard. | **Mandatory for every malignant case.** No case may be counted as a confirmed cancer without it. |
| **B** | `dermoscopy_supported` | Clinical diagnosis where a dermoscopic image was also reviewed and supports it. | Pigmented and lesional cases where dermoscopy was performed but no biopsy was taken. |
| **C** | `clinical_consensus` | Clinical diagnosis agreed by two independent dermatologists (or settled by adjudication). | The default for common inflammatory and infectious conditions. |
| **D** | `provisional` | Uncertain, working, or awaiting further information. | Cases pending histopathology, atypical presentations, or where readers cannot agree even after adjudication. |

Rules:

1. **The grade is stored per image, not per patient and not per condition.** The same patient can contribute a grade A biopsied lesion and a grade C clinical lesion on the same day.
2. **A malignant label without grade A is not a confirmed cancer.** It stays `provisional` until the report returns, and does not count toward the ≥100 minimum / ≥170 target for biopsy-confirmed malignant cases (170 being the number needed to match or beat DDI's 171).
3. **Grades never silently upgrade.** A case moves from `provisional` to `histopathology` only through the reconciliation step in Section 5, performed by a named person, with a date.
4. Grade D cases are **kept, not deleted.** Uncertainty is real clinical information. They are excluded from any analysis requiring a confident label, and that exclusion is stated.

---

## 3. Independent double-reading

**Two dermatologists label every image set independently and blinded to each other.**

1. Each reader sees the full image set for the lesion (context, close-up, dermoscopic where present) and the de-identified metadata: age band, sex, site, duration, symptoms, skin tone.
2. Neither reader sees the other's entry, or any previous label on the case. Use separate reader accounts, or separate paper forms collected before either is opened.
3. **Both labels are recorded permanently** — `label_reader1`, `label_reader2`, with reader ID and date. Do not overwrite one with the other. Reader disagreement is a useful signal about which conditions are hard, and it is lost forever if you keep only the winner.
4. Each reader also records a confidence: `confident` / `probable` / `uncertain`.

**Agreement is measured and reported.** Compute **Cohen's kappa** between the two readers, overall and per condition group, and refresh it monthly. Report it in any publication; a dataset that does not report inter-rater agreement cannot be assessed by anyone else.

**What triggers adjudication:**

- The two readers give **different diagnoses** — always.
- The two readers agree on the diagnosis but **either one marks `uncertain`**.
- Either reader flags **benign vs malignant disagreement** — this goes to adjudication immediately and as a priority, regardless of anything else.
- Monthly kappa for a condition group falls **below 0.60** — that condition group is re-read in bulk and the label definitions are reviewed, because the problem is probably the definition and not the readers. *(0.60 is a working threshold recommended for this study, not an established standard.)*

---

## 4. Adjudication

1. A **third, senior dermatologist** reviews the case. They see both readers' labels and the images.
2. The adjudicated diagnosis is written to `label_final`, with `adjudicated = yes`, the adjudicator's ID and the date.
3. `label_reader1` and `label_reader2` remain in the record unchanged.
4. If the adjudicator also cannot settle it, `label_final` is left as the best available and `diagnosis_basis` becomes `provisional`. Forcing a confident label on an unresolvable case is the worse error.
5. Adjudication is for diagnosis only. It cannot upgrade `diagnosis_basis` to `histopathology` — only a pathology report can do that.

---

## 5. The biopsy linkage workflow

This is the part most teams get wrong, and it is the part our cancer arm depends on entirely. The failure mode is mundane: a lesion is biopsied, the study never learns the result, and the case sits forever as an unconfirmed clinical guess.

**Step 1 — At capture (clinic).**
The clinician records `biopsy_taken = yes` and writes the **pathology accession number** into the case record at the moment the specimen is labelled. Not later, not from memory. The accession number is the only thread connecting the photograph to the report.

**Step 2 — The case enters the PENDING HISTOPATHOLOGY queue.**
`diagnosis_basis` is set to `provisional`. The clinical differential is recorded as normal. The case appears on a standing pending list held by the coordinator, with study ID, accession number, date of biopsy, and site.

**Step 3 — The queue is worked, not watched.**
A **designated pathology liaison** (one named person per site, with a named backup) reviews the pending list **weekly**. Cases older than 4 weeks are chased with the pathology department. A pending list that is never chased is how cancer arms fail.

**Step 4 — Reconciliation when the report returns.**
The liaison matches the accession number to the study ID, reads the report, and enters:
- `histopath_diagnosis` — verbatim diagnosis line from the report,
- `histopath_report_date`,
- `reconciled_by` and `reconciled_date`.

Only now does `diagnosis_basis` become `histopathology`, and only now may the case be counted toward the biopsy-confirmed cancer target.

**Step 5 — Failure cases. Handle them explicitly; do not improvise.**

| Situation | What to do |
|---|---|
| **Report never returns** (patient did not proceed, specimen lost, patient left the system) | After 12 weeks, set `histopath_status = not_available` with a free-text reason. The case **stays in the dataset** as grade B/C/D on its clinical label. It never counts as a confirmed cancer. Do not delete it and do not quietly relabel it. |
| **Accession number mistyped or unmatched** | Do not guess and do not match on patient name. Attempt matching on date + site + specimen type through the pathology department. If it still cannot be matched with certainty, set `histopath_status = unmatched`. An uncertain link is worse than no link. |
| **Biopsy result contradicts the clinical label** | **Keep both.** `label_final` becomes the histopathology diagnosis — histopathology always wins. The clinical labels stay in `label_reader1` / `label_reader2`. Set `discordant = yes`. |
| **Report is non-diagnostic / inadequate sample** | `histopath_status = non_diagnostic`. Treated exactly like "never returned" for counting purposes. |

**Discordant cases are the most scientifically interesting images in the dataset.** A lesion two dermatologists called benign that came back malignant on brown skin is precisely the case the primary research exists to catch. Flag them, count them, and describe them separately.

---

## 6. The retrospective archive arm

Skin cancer is less common in brown skin, and 170 biopsy-confirmed malignant cases will not accrue prospectively in one year at two hospitals. The second track pulls already-confirmed cases from the pathology archive. **DDI was built exactly this way** — retrospectively, from a pathology archive, 100% biopsy-confirmed.

1. **Start from pathology, not from photographs.** Search the archive for confirmed malignant and relevant benign diagnoses over the agreed date range.
2. **Find the matching clinical photographs** in the hospital record. Many cases will have none; those are not usable and the drop-out count is recorded.
3. **Apply the image-quality gate (Section 7) unchanged.** Archive photographs were not taken to this SOP and many will fail. That is expected — but the standard does not bend.
4. **Apply the same de-identification unchanged:** study IDs only, no faces, no identifying marks, identity↔study-ID key held separately under access control.
5. **Apply the same metadata schema unchanged**, so the two arms are structurally identical.
6. **Record the arm.** Every record carries `source_arm = prospective` or `source_arm = retrospective`, plus `capture_to_sop = yes/no`. The arms are then analysed separately and together, so any difference between them is visible rather than hidden.
7. Retrospective cases are grade A by construction, but still go through double-reading of the *clinical* label so reader behaviour can be compared across arms.

Ethics-committee approval must explicitly cover the retrospective arm and archival photograph use. It is a separate permission from the prospective one.

---

## 7. Quality control

**Image-quality gate, applied before labelling.** An image that cannot be graded must not be labelled — a guess on an ungradable image is a fabricated data point. Reject and record the reason. DermaCon-IN excluded images for poor quality, obstructions, metadata gaps, and ambiguous diagnosis; we use the same categories:

| `exclusion_reason` | Meaning |
|---|---|
| `blurred` | Not sharp on the lesion at 100% |
| `glare` / `shadow` | Lesion surface obscured by light or shadow |
| `underexposed` | Detail crushed — the common failure on darker skin |
| `obstructed` | Hair, dressing, ruler, ointment or clothing over the lesion |
| `framing` | Lesion cropped, off-centre, or too small in frame |
| `metadata_incomplete` | Missing site, consent record, skin tone, or device |
| `identifiable` | Face, tattoo, name band or other identifier in frame |
| `ambiguous_diagnosis` | Unresolvable even after adjudication |

Excluded images are **retained in a separate quarantine store with the reason**, not deleted. Exclusion counts by reason, site, month and skin tone are reported monthly. A rising exclusion rate at one site is a training problem, and it is only visible if you count.

**Periodic re-reads.** Every month, a **random 10%** of already-labelled cases is re-read blind by a reader who did not label them originally, and agreement is computed. If it drops, labelling has drifted and the criteria need re-briefing. *(10% is a recommendation for this study, chosen to be sustainable, not a published standard.)*

**Patient-level integrity.** All splits are patient-level — every image from one patient stays on one side — with the final ~3 months of prospective accrual frozen as a prospective test set. Never enter the same patient under two study IDs; if it happens, report it so the records can be merged.

---

## 8. Blinding and bias

**Labellers must never see the model's prediction while labelling.** This is not optional.

If a reader sees the model output first, their label drifts toward it. The dataset then stops being an independent test and becomes a record of the model agreeing with itself. Every performance number measured on it is inflated, and the inflation is invisible and unrecoverable. This is the fastest way to destroy a year of collection.

Practically:
- The labelling interface shows images and clinical metadata only. No model output, no confidence score, no heatmap.
- Model evaluation runs **after** `label_final` is locked, never before.
- Anyone who has seen model predictions for a case does not label or adjudicate that case.
- Readers are also blinded to each other (Section 3) and, where practical, to the treating clinician's written impression.

---

## 9. Dataset release

A de-identified public release is the point of the exercise, but it is **subject to ethics-committee approval and must not be assumed**. Consent language and the IEC application must cover it from the outset — retrofitting consent for release is not possible.

A release would contain: de-identified images (study IDs only, no faces or identifying marks); `label_final` with `diagnosis_basis`; `adjudicated` and `discordant` flags; agreement statistics including Cohen's kappa; skin tone (Fitzpatrick and Monk, as DermaCon-IN records both); age band rather than exact age; anatomical site; device model; `source_arm`; and the capture SOP version.

It would **not** contain: identifiable images, hospital numbers, accession numbers, dates finer than month, free-text clinical notes, or the identity↔study-ID key.

Licence: a research-use licence with an attribution requirement and a no-re-identification clause is the norm for clinical dermatology datasets, and is the recommended starting position for discussion with the institution and the IEC. Storage and any release infrastructure must remain India-resident, institution-approved, and encrypted at rest and in transit under the DPDP Act 2023.

---

*Queries on grades, adjudication or the pending histopathology queue: contact the study coordinator. When in doubt about a label, record the uncertainty — do not resolve it by guessing.*
