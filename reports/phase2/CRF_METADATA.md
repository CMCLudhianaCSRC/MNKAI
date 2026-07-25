# CASE REPORT FORM AND IMAGE METADATA SCHEMA
## North India Dermatology Photo Data Collection Study

Version 1.0 | Companion document to the Photo Capture SOP | Two sites | Approx. 1 year | Target 3,000–5,000 consented images

---

## 1. Why the structured form matters as much as the photograph

A photograph on its own cannot answer the questions this study exists to answer. The gap we are filling is not "more skin pictures" — it is **who** the pictures are of and **how the diagnosis was established**. The dark-skin skin-cancer result from the primary research (AUROC 0.872, 95% CI 0.808–0.929 on Fitzpatrick V–VI) rests on DDI: 656 biopsy-confirmed images from a single US academic clinic. The 25-condition read by skin tone is weaker still — only about 135 darker-skin images in the external test spread across 25 conditions, roughly 5 per condition. You cannot make a fair claim about how a tool behaves in brown skin from 5 images. The fix is to record skin tone, age, sex, site and diagnosis basis on **every single case**, so results can be split by subgroup afterwards. A blank field cannot be split on later: a case with a perfect photograph and an empty skin-tone field is close to useless for the question we are asking.

Second, the tool's history-taking layer asks for exactly these items — duration, symptoms, evolution, prior treatment, site. If the form and the tool use different field names or answer options, the two cannot be joined. The names below are therefore fixed and shared by the app and the analysis pipeline.

Third, precedent. PAD-UFES-20 (Brazil) captured 26 metadata fields alongside smartphone images and biopsied roughly 58% of samples; DermaCon-IN (South India, 5,450 images from 3,002 patients) recorded both Fitzpatrick type and Monk skin tone. Reviewers will compare us to these datasets, so matching their metadata depth is the cheapest credibility available to us.

---

## 2. The field table

One CRF record is completed **per lesion**, not per patient and not per image. A patient with three lesions generates three records sharing one `patient_study_id`. Images attach to the record by filename (SOP Section 10).

### 2.1 Identification and capture

| Field | Type | Allowed values / units | Req. | Why collected |
|---|---|---|---|---|
| `study_id` | string | `SITE-PATIENTID-LESIONNO`, e.g. `CMC-0142-L1` | Yes | Primary key. Links form to images without using any identifier. |
| `patient_study_id` | string | `SITE-PATIENTID`, e.g. `CMC-0142` | Yes | Enables **patient-level splitting** — the same patient must never appear in both training and test data. |
| `site` | enum | `CMC_LUDHIANA`, `SITE2` (final codes set by coordinator) | Yes | Site is a confounder; also allows leave-one-site-out checking. |
| `capture_date` | date | `YYYY-MM-DD` | Yes | Defines the temporal hold-out — the final ~3 months of accrual is frozen as a prospective test set. |
| `device_model` | string | Free text from a fixed picklist, e.g. `Redmi Note 13 Pro` | Yes | Mixed consumer phones are deliberate (as in DermaCon-IN). Recording the device lets us test whether performance depends on the camera. |
| `image_type` | enum | `context`, `closeup`, `dermoscopic`, `angle2` | Yes (per image) | Different image types are analysed separately. Stored per image inside `images[]`. |
| `dermoscope_used` | boolean | `true`, `false` | Yes | Dermoscopic and clinical images cannot be pooled. |
| `photographer_id` | string | Staff study code, e.g. `PH03` | Yes | Quality auditing; identifies retraining needs. Never a name. |

### 2.2 Lesion location

| Field | Type | Allowed values / units | Req. | Why collected |
|---|---|---|---|---|
| `anatomical_site` | enum | Controlled vocabulary, Section 4.1 | Yes | Body site strongly changes the differential and is a known shortcut the model can learn; must be recorded to test for it. |
| `laterality` | enum | `left`, `right`, `midline`, `bilateral`, `not_applicable` | Yes | Needed for follow-up and for de-duplicating the same lesion across visits. |
| `lesion_size_mm` | integer | 1–500 mm, longest diameter, from the in-frame ruler | Yes | Size is diagnostically meaningful and validates that the ruler was used. |

### 2.3 Patient characteristics

| Field | Type | Allowed values / units | Req. | Why collected |
|---|---|---|---|---|
| `age_years` | integer | 0–120 | Yes | Subgroup analysis; age alters the pre-test probability of malignancy. |
| `sex` | enum | `male`, `female`, `other`, `not_disclosed` | Yes | Fairness reporting. |
| `fitzpatrick` | enum | `I`, `II`, `III`, `IV`, `V`, `VI` | Yes | **The core stratification variable.** Target ≥150, ideally ≥300, images per skin-tone bin in the test set. |
| `monk_tone` | integer | 1–10 | Yes | DermaCon-IN recorded both Fitzpatrick and Monk; we recommend the same. Monk is more reliable than Fitzpatrick in brown skin, where Fitzpatrick was never designed to discriminate well. |
| `tone_assessor` | enum | `clinician`, `patient_self_report` | Yes | Self-reported and clinician-assigned Fitzpatrick disagree often; the analysis must know which it has. |
| `occupation_sun_exposure` | enum | `indoor`, `mixed`, `outdoor_most_of_day`, `unknown` | Optional | Cumulative sun exposure matters for the cancer arm; a coarse three-level field is realistic to collect in a busy clinic. |

### 2.4 History

| Field | Type | Allowed values / units | Req. | Why collected |
|---|---|---|---|---|
| `duration_value` + `duration_unit` | integer + enum | value ≥ 0; unit `days`, `weeks`, `months`, `years` | Yes | Duration separates acute from chronic disease; the tool's history layer asks for it. |
| `symptoms` | multi-select | Section 4.2 | Yes | Itch, pain and bleeding change the differential sharply. |
| `evolution` | enum | Section 4.3 | Yes | A changing or growing lesion is the single most important malignancy trigger in history. |
| `prior_treatment` | enum + short text | `none`, `topical_steroid`, `topical_other`, `oral_medication`, `traditional_remedy`, `unknown`; plus ≤100 chars naming the agent | Yes | Prior steroid or traditional remedy use visibly alters the lesion. Treated disease looks different, and the tool must not be blamed for it. |
| `comorbid_diabetes` | enum | `yes`, `no`, `unknown` | Yes | Diabetes changes the pattern of skin infection. |
| `immunosuppression` | enum | `yes`, `no`, `unknown` | Yes | Immunosuppression changes both presentation and cancer risk. |
| `hiv_status` | enum | `positive`, `negative`, `not_recorded` | **Committee decision** | Clinically relevant but highly sensitive. *Recommendation:* record only where already documented in routine care, never test for the study, default to `not_recorded`. **Whether this field is collected at all is a decision for the Institutional Ethics Committee; drop it entirely if the committee is not satisfied.** |
| `family_history_relevant` | enum | `yes`, `no`, `unknown`, `not_applicable` | Optional | Applies to a limited set (atopy, psoriasis, vitiligo, melanoma). Left `not_applicable` otherwise. |

### 2.5 Diagnosis and labelling

| Field | Type | Allowed values / units | Req. | Why collected |
|---|---|---|---|---|
| `provisional_diagnosis` | string | Study condition list term | Yes | The clinical working diagnosis at the bedside. |
| `differential_diagnoses` | array of string | 0–3 study terms | Optional | Records genuine uncertainty; lets us report where clinicians themselves disagree. |
| `clinician_confidence` | integer | 1–5 (1 = very unsure, 5 = certain) | Yes | Allows a clean comparison of tool and clinician on the cases clinicians find hard. |
| `biopsy_taken` | enum | `yes`, `no`, `pending` | Yes | Defines the biopsy-confirmed cancer arm (target ≥100 malignant, ≥170 to match or beat DDI's 171). |
| `pathology_accession_number` | string | Site-issued accession, **never the hospital MRN** | If `biopsy_taken = yes` | The only way to reconcile image and report. Treated as an identifier and held in the restricted key file, not in the released dataset. |
| `histopathology_diagnosis` | string | Pathology report wording, mapped to a study term | If biopsy resulted | Ground truth for the cancer arm. |
| `final_diagnosis` | string | Study condition list term | Yes | The label used for training and evaluation. |
| `diagnosis_basis` | enum | Section 4.4 | Yes | **Label quality is not uniform.** Analyses must be repeatable restricted to histopathology-confirmed cases only. |
| `labeller_1_id`, `labeller_2_id` | string | Staff study codes, e.g. `DR07` | Yes | Two independent labels per case. |
| `label_agreement` | enum | `agree`, `disagree` | Yes | Supports an inter-rater agreement statistic for the paper. |
| `adjudicator_id` | string | Staff study code | If `disagree` | Third reader resolves. |
| `adjudication_outcome` | enum | `labeller_1_upheld`, `labeller_2_upheld`, `third_label`, `unresolvable_excluded` | If `disagree` | Documents how disagreements were settled; `unresolvable_excluded` removes the case from the labelled set but keeps the audit trail. |
| `consent_signed` | boolean | `true` only | Yes | Record cannot be saved as `false`. Ethical gate. |
| `consent_withdrawn` | boolean | `true`, `false` | Yes | Set `true` on withdrawal; triggers same-day deletion of images. |

---

## 3. Worked example

A plausible completed record: a 54-year-old farmer with a non-healing plaque on the sun-exposed forearm, biopsied.

| Field | Value |
|---|---|
| `study_id` | `CMC-0142-L1` |
| `patient_study_id` | `CMC-0142` |
| `site` | `CMC_LUDHIANA` |
| `capture_date` | `2026-07-25` |
| `device_model` | `Redmi Note 13 Pro` |
| `dermoscope_used` | `true` |
| `photographer_id` | `PH03` |
| `anatomical_site` | `forearm` |
| `laterality` | `left` |
| `lesion_size_mm` | `22` |
| `age_years` | `54` |
| `sex` | `male` |
| `fitzpatrick` | `V` |
| `monk_tone` | `8` |
| `tone_assessor` | `clinician` |
| `occupation_sun_exposure` | `outdoor_most_of_day` |
| `duration_value` / `duration_unit` | `8` / `months` |
| `symptoms` | `["bleeding", "pain"]` |
| `evolution` | `growing` |
| `prior_treatment` | `topical_steroid` — "unnamed cream from local pharmacy" |
| `comorbid_diabetes` | `yes` |
| `immunosuppression` | `no` |
| `hiv_status` | `not_recorded` |
| `family_history_relevant` | `not_applicable` |
| `provisional_diagnosis` | `squamous_cell_carcinoma` |
| `differential_diagnoses` | `["basal_cell_carcinoma", "chronic_ulcer"]` |
| `clinician_confidence` | `3` |
| `biopsy_taken` | `yes` |
| `pathology_accession_number` | `HP-2026-04871` (restricted) |
| `histopathology_diagnosis` | `well_differentiated_squamous_cell_carcinoma` |
| `final_diagnosis` | `squamous_cell_carcinoma` |
| `diagnosis_basis` | `histopathology` |
| `labeller_1_id` / `labeller_2_id` | `DR07` / `DR11` |
| `label_agreement` | `disagree` |
| `adjudicator_id` / `adjudication_outcome` | `DR02` / `labeller_1_upheld` |
| `consent_signed` / `consent_withdrawn` | `true` / `false` |
| Images | `CMC-0142-L1-CONTEXT-20260725`, `CMC-0142-L1-CLOSEUP-20260725`, `CMC-0142-L1-DERM-20260725` |

---

## 4. Controlled vocabularies

### 4.1 `anatomical_site`

`scalp`, `forehead`, `periorbital`, `nose`, `cheek`, `perioral`, `lip`, `chin`, `ear`, `neck`, `chest`, `abdomen`, `back`, `buttock`, `axilla`, `groin`, `genital`, `perianal`, `shoulder`, `upper_arm`, `elbow`, `forearm`, `wrist`, `hand_dorsum`, `palm`, `finger`, `nail_finger`, `hip`, `thigh`, `knee`, `lower_leg`, `ankle`, `foot_dorsum`, `sole`, `toe`, `nail_toe`, `oral_mucosa`, `generalised`, `other`

`other` requires a one-word free-text note. `generalised` is used only when the eruption has no single site.

### 4.2 `symptoms` (multi-select; `none` is exclusive)

`itch`, `pain`, `burning`, `bleeding`, `discharge`, `numbness`, `none`

### 4.3 `evolution` (single-select)

`new`, `growing`, `changing_colour`, `changing_shape`, `stable`, `resolving`, `recurrent`, `unknown`

### 4.4 `diagnosis_basis` (single-select, in ascending order of certainty)

`clinical` — bedside diagnosis only.
`dermoscopy` — clinical plus dermoscopic assessment.
`laboratory` — supported by KOH, culture, Tzanck or similar, but not histology.
`histopathology` — biopsy-confirmed. **Only these cases count toward the biopsy-confirmed cancer arm.**

---

## 5. Data-quality rules

**Blocking fields — the record cannot be saved if any is missing or invalid:**
`study_id`, `patient_study_id`, `site`, `capture_date`, `device_model`, `anatomical_site`, `laterality`, `age_years`, `sex`, `fitzpatrick`, `monk_tone`, `duration_value`, `duration_unit`, `symptoms`, `evolution`, `provisional_diagnosis`, `clinician_confidence`, `biopsy_taken`, `consent_signed = true`, and at least one image of `image_type = closeup`.

**Ranges and formats:** `age_years` 0–120 (integer); `lesion_size_mm` 1–500; `monk_tone` 1–10; `clinician_confidence` 1–5; `capture_date` must be a valid date not in the future and not before study start; `study_id` must match `^[A-Z]{2,4}-\d{4}-L\d{1,2}$`.

**Conditional rules:** if `biopsy_taken = yes` then `pathology_accession_number` is required and the record stays open until `histopathology_diagnosis` is entered; if `biopsy_taken = pending` the record is flagged for follow-up at 30 days; `diagnosis_basis = histopathology` is only permitted when a histopathology diagnosis exists; if `label_agreement = disagree` then `adjudicator_id` and `adjudication_outcome` are required; `symptoms = ["none"]` cannot be combined with any other symptom.

**How "unknown" is encoded.** There is exactly one convention: the literal string `unknown` where the vocabulary allows it, `not_recorded` for `hiv_status`, `not_applicable` where the question does not apply, and `null` only for genuinely optional fields that were skipped. **Never** use blanks, `0`, `-1`, `NA`, `N/A`, `nil` or a dash. `unknown` means *asked and not established*; `null` means *not asked*. These are different things and the analysis treats them differently.

**Audit:** every record carries `entered_by`, `entered_at`, and an append-only edit log. Values are never silently overwritten.

---

## 6. JSON schema (field names are binding)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "NorthIndiaDermCRF",
  "type": "object",
  "required": [
    "study_id", "patient_study_id", "site", "capture_date", "device_model",
    "anatomical_site", "laterality", "age_years", "sex", "fitzpatrick",
    "monk_tone", "tone_assessor", "duration_value", "duration_unit",
    "symptoms", "evolution", "provisional_diagnosis", "clinician_confidence",
    "biopsy_taken", "consent_signed", "images"
  ],
  "properties": {
    "study_id": { "type": "string", "pattern": "^[A-Z]{2,4}-\\d{4}-L\\d{1,2}$" },
    "patient_study_id": { "type": "string", "pattern": "^[A-Z]{2,4}-\\d{4}$" },
    "site": { "enum": ["CMC_LUDHIANA", "SITE2"] },
    "capture_date": { "type": "string", "format": "date" },
    "device_model": { "type": "string", "maxLength": 60 },
    "dermoscope_used": { "type": "boolean" },
    "photographer_id": { "type": "string", "maxLength": 8 },

    "anatomical_site": { "enum": ["scalp","forehead","periorbital","nose","cheek","perioral","lip","chin","ear","neck","chest","abdomen","back","buttock","axilla","groin","genital","perianal","shoulder","upper_arm","elbow","forearm","wrist","hand_dorsum","palm","finger","nail_finger","hip","thigh","knee","lower_leg","ankle","foot_dorsum","sole","toe","nail_toe","oral_mucosa","generalised","other"] },
    "anatomical_site_other": { "type": ["string", "null"], "maxLength": 20 },
    "laterality": { "enum": ["left","right","midline","bilateral","not_applicable"] },
    "lesion_size_mm": { "type": ["integer", "null"], "minimum": 1, "maximum": 500 },

    "age_years": { "type": "integer", "minimum": 0, "maximum": 120 },
    "sex": { "enum": ["male","female","other","not_disclosed"] },
    "fitzpatrick": { "enum": ["I","II","III","IV","V","VI"] },
    "monk_tone": { "type": "integer", "minimum": 1, "maximum": 10 },
    "tone_assessor": { "enum": ["clinician","patient_self_report"] },
    "occupation_sun_exposure": { "enum": ["indoor","mixed","outdoor_most_of_day","unknown", null] },

    "duration_value": { "type": "integer", "minimum": 0 },
    "duration_unit": { "enum": ["days","weeks","months","years"] },
    "symptoms": { "type": "array", "items": { "enum": ["itch","pain","burning","bleeding","discharge","numbness","none"] }, "minItems": 1, "uniqueItems": true },
    "evolution": { "enum": ["new","growing","changing_colour","changing_shape","stable","resolving","recurrent","unknown"] },
    "prior_treatment": { "enum": ["none","topical_steroid","topical_other","oral_medication","traditional_remedy","unknown"] },
    "prior_treatment_detail": { "type": ["string", "null"], "maxLength": 100 },
    "comorbid_diabetes": { "enum": ["yes","no","unknown"] },
    "immunosuppression": { "enum": ["yes","no","unknown"] },
    "hiv_status": { "enum": ["positive","negative","not_recorded"], "description": "Collected only if approved by the Institutional Ethics Committee." },
    "family_history_relevant": { "enum": ["yes","no","unknown","not_applicable", null] },

    "provisional_diagnosis": { "type": "string" },
    "differential_diagnoses": { "type": "array", "items": { "type": "string" }, "maxItems": 3 },
    "clinician_confidence": { "type": "integer", "minimum": 1, "maximum": 5 },
    "biopsy_taken": { "enum": ["yes","no","pending"] },
    "pathology_accession_number": { "type": ["string", "null"], "maxLength": 30 },
    "histopathology_diagnosis": { "type": ["string", "null"] },
    "final_diagnosis": { "type": ["string", "null"] },
    "diagnosis_basis": { "enum": ["clinical","dermoscopy","laboratory","histopathology", null] },

    "labeller_1_id": { "type": ["string", "null"], "maxLength": 8 },
    "labeller_2_id": { "type": ["string", "null"], "maxLength": 8 },
    "label_agreement": { "enum": ["agree","disagree", null] },
    "adjudicator_id": { "type": ["string", "null"], "maxLength": 8 },
    "adjudication_outcome": { "enum": ["labeller_1_upheld","labeller_2_upheld","third_label","unresolvable_excluded", null] },

    "consent_signed": { "const": true },
    "consent_withdrawn": { "type": "boolean", "default": false },

    "images": {
      "type": "array", "minItems": 1,
      "items": {
        "type": "object",
        "required": ["filename", "image_type"],
        "properties": {
          "filename": { "type": "string" },
          "image_type": { "enum": ["context","closeup","dermoscopic","angle2"] },
          "colour_card_present": { "type": "boolean" },
          "ruler_present": { "type": "boolean" },
          "qc_pass": { "type": "boolean" }
        }
      }
    },

    "entered_by": { "type": "string", "maxLength": 8 },
    "entered_at": { "type": "string", "format": "date-time" }
  },
  "additionalProperties": false
}
```

---

## 7. De-identification — what must never be typed into this form

The form is designed so that **no free-text field ever needs to contain an identifier**. Free text exists in only three places (`anatomical_site_other`, `prior_treatment_detail`, and the pathology diagnosis text), and all three are length-capped deliberately.

**Never enter, in any field:** patient names or initials; hospital registration or MRN numbers; ward or bed numbers; phone numbers; addresses, village or locality names; Aadhaar or any government ID; email addresses; dates of birth (record `age_years` only); admission or discharge dates; referring doctor names; or any nickname or description that would identify a person to a colleague ("the sarpanch's wife", "the nurse from ward 4").

`pathology_accession_number` is the one field that is an identifier by nature. It is stored in the restricted key file alongside the identity↔study-ID mapping, held under access control by the coordinator, and is **removed before any dataset is shared or released**.

Minimise free text generally. Every free-text box is a route by which identifying information leaks into an otherwise clean dataset, and free text cannot be reliably screened at scale. If the picklist does not fit, choose the closest term and raise it with the coordinator so the vocabulary can be extended — do not write a sentence.

Under India's DPDP Act 2023 these records are personal data. They are stored encrypted at rest and in transit, on India-resident, institution-approved infrastructure, never on personal drives or messaging apps, and never in a spreadsheet emailed between staff.

---

*This document is fixed for the study duration. Any change to a field name, allowed value or controlled vocabulary requires a protocol amendment and re-approval, because changing a field name mid-study silently breaks every analysis that uses it.*
