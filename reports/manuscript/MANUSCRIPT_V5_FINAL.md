%%DOCTYPE%%RESEARCH PROJECT
%%TITLE%%Prospective clinical validation of an artificial intelligence system for the assessment of skin disease and skin cancer in Indian patients
%%CANDIDATE%%UJJWAL THAKRAN|MBBS Student Batch 2022,|Christian Medical College & Hospital,|Ludhiana, Punjab-141008
%%SUPERVISOR%%DR. ABHILASHA P WILLIAMS|Professor & Head,|Department of Dermatology,|Christian Medical College and Hospital,|Ludhiana, Punjab-141008
%%COSUPERVISOR%%DR. [CO-SUPERVISOR NAME]|[Designation],|Department of Dermatology,|Christian Medical College and Hospital,|Ludhiana, Punjab-141008

## CHECKLIST OF COMPONENTS

Prepared against the departmental Checklist for Assessment of Research Proposal. The location of each item is given below for the Supervisor and the Head of Department.

| Sr. No. | Criteria / guideline | Section |
| 1 | Introduction | Introduction |
| 2 | Aims | Aim (Aim 1, Aim 2) |
| 3 | Objectives, primary / secondary / tertiary | Objectives, numbered under each Aim |
| 4a | Study design | Material and Methods A |
| 4b | Study population, inclusion, exclusion, sample size | Material and Methods B, C |
| 4c | Methodology | Material and Methods D, E, F |
| 4d | Statistics | Material and Methods G |
| 5 | Impact on current knowledge | Impact of the Proposed Research |
| 6 | References, style, in-text citation, recency | References (superscript, numbered) |
| 7 | Abstract 200–300 words, 5–6 keywords | Abstract |
| 8 | Title | Title page |
| 9 | Proforma | Proforma |
| 10 | Consent form with Hindi and Punjabi | Informed Consent Form, three languages |
| 11 | Patient information sheet with Hindi and Punjabi | Patient Information Sheet, three languages |
| 12 | Competing interests; similar study in 3 years | Competing Interests |
| 13 | Financial implications | Financial Implications |
| 14 | Researcher role and academic impact | Role of the Researcher |

## ABSTRACT

**Introduction:** Computer systems that read photographs of skin lesions are being proposed to help decide which patients need a dermatologist. Almost all of them were built from photographs of light-skinned patients. When three published systems were tested on a skin-tone-balanced set in which every diagnosis was confirmed by biopsy, their ability to separate cancer from non-cancer on the darkest skin fell to between 0.50 and 0.57, where 0.50 is chance. There is no dermatological image collection from North India, so no such system has been tested on patients from this region.

**Aim and Objectives:** To test, in patients attending this hospital, whether a system developed by the investigator agrees with the diagnosis of a dermatologist; and to build a de-identified collection of dermatological photographs from this department.

**Material and Methods:** Prospective observational study in the Department of Dermatology over two years. One hundred consecutive consenting patients aged 18 years and above will be enrolled for the first aim. Sample size was calculated as N = Z²p(1−p)/d², taking p = 0.85 and d = 7%. Photographs and structured clinical data will be captured on a purpose-built application which records consent first. The system uses six frozen foundation models whose feature vectors feed a small trained classifier; their outputs are combined by ensembling, the probability is calibrated, and low-confidence cases are refused by an abstention rule. The dermatologist's diagnosis, entered before the system is run and while blinded to it, is the reference standard, and is replaced by histopathology wherever a biopsy is done. Primary outcome is top-3 agreement, reported with an exact binomial 95% confidence interval and Cohen's kappa. The system will run in parallel and will not affect treatment.

**Keywords:** Artificial intelligence, dermatology, skin cancer, skin of colour, Fitzpatrick skin type, diagnostic accuracy

## INTRODUCTION

Skin complaints are among the commonest reasons for an outpatient visit in India, and there are few dermatologists for the population. Much of this work falls to doctors who are not dermatologists. A tool that indicates which patients need specialist review would therefore be useful in outpatient departments, screening camps and teleconsultation.

Computer systems that read photographs have been proposed for this. Several have reported accuracy close to that of dermatologists under test conditions.^[1,2] Two problems limit what those reports mean for our patients.

The first is the material these systems were built from. The large public image collections used to build them were assembled in Europe, North America and Australia, and contain mostly light-skinned patients.^[1] Groh and colleagues labelled a large public collection by Fitzpatrick skin type and found that accuracy varied with skin tone, with darker types both fewer in number and less accurately classified.^[3]

Daneshjou and colleagues measured the clinical effect of this. They collected 656 photographs from 570 patients, balanced across skin tones, with every diagnosis confirmed by biopsy, and tested three published systems on it without adapting them.^[4] All three did worse on the darkest skin. Measured as area under the receiver operating characteristic curve, where 0.50 is chance and 1.00 is perfect, one fell from 0.61 on Fitzpatrick I–II to 0.50 on V–VI, the second from 0.64 to 0.55, and the third from 0.72 to 0.57.^[4] After retraining on photographs from that collection, performance on the darkest skin reached about 0.76 and 0.78.^[4]

This matters here for a second reason. Melanoma in Indian patients arises more often on the palms, soles and nails than on sun-exposed skin, is often mistaken at first for a benign lesion or a bruise, and is usually found late.^[5,6] A tool that is least reliable on brown skin, and at the sites where diagnosis is already hardest, is of little use and may falsely reassure.

The second problem is what these systems report. Most give a diagnosis and nothing else. They do not say how confident they are, do not flag cases outside the material they were built from, and do not report whether accuracy differs between groups of patients.^[7] A clinician then has no way to judge when the output should be ignored.

Work in India has begun to address the shortage of local data. DermaCon-IN, from South India, collected 5,450 photographs from 3,002 patients using ordinary smartphones under clinic lighting, labelled by dermatologists.^[8] A similar collection in Brazil used a purpose-built smartphone application and recorded 26 clinical fields alongside each photograph.^[9] Neither includes a biopsy-confirmed cancer component, and neither covers North India.

The investigator has built a system for this setting using public image collections and a personal computer. It differs from those above in three ways, described in Material and Methods: the underlying models are used unchanged rather than retrained, which gave better results when the available dermatological material was limited;^[10] the outputs of six such models are combined;^[11] and the system states its confidence, gives a short list rather than one answer,^[12] and refuses cases it is not confident about.

In testing on public collections, including the biopsy-confirmed set above, this system did not lose accuracy on darker skin. Those results are given later in this document as the reason for the present application. They were obtained on photographs taken in other countries and say nothing about patients attending this hospital. Photographs taken in a busy outpatient department differ in lighting, framing and artefact, and the diseases seen differ. Testing the system on real patients is the step that has not been taken, and that is what this application seeks permission for.

## AIM

**Aim 1.** To test whether an artificial intelligence system for the assessment of skin disease agrees with the diagnosis of a dermatologist in patients attending this hospital.

**Aim 2.** To build a de-identified collection of dermatological photographs, with clinical data and histopathology where available, from patients attending this department.

## OBJECTIVES

**Under Aim 1**

1. To measure the agreement between the system's ranked list of three diagnoses and the dermatologist's diagnosis in 100 consecutive consenting patients (primary objective).
2. To measure the sensitivity and specificity of the system for identifying lesions that need biopsy or referral, against histopathology where a biopsy is done.
3. To compare the system's accuracy across Fitzpatrick skin types IV, V and VI.
4. To measure how often the system refuses to answer, and its accuracy on the cases it does answer.

**Under Aim 2**

5. To collect, over 24 months, de-identified photographs with clinical data and histopathology where available.
6. To record the range and frequency of skin conditions seen in this department, and the distribution of Fitzpatrick skin types.
7. To measure the proportion of photographs rejected as technically inadequate, and the reasons.

## MATERIAL AND METHODS

### A. Study design

Prospective observational study in the Department of Dermatology, Christian Medical College and Hospital, Ludhiana, over two years from the date of approval.

Both aims run at the same time in the same department. Aim 1 uses the first 100 consecutive consenting patients. Aim 2 continues for the full 24 months and includes those 100 patients.

The system runs in parallel with clinical care and does not affect it. Its output is not shown to the treating doctor before the clinical diagnosis is written, is not entered in the patient record, and is not used to decide investigation, referral or treatment. No patient's management changes because of this study.

%%FIG:fig1_study_flow.png|Figure 1. Overall flow of the study.%%

%%FIG:fig2_visit_sequence.png|Figure 2. Sequence at a single outpatient visit.%%

### B. Study population

**Inclusion criteria**

1. Age 18 years or above, attending the Dermatology outpatient department with a skin lesion or eruption.
2. Willing and able to give written informed consent.
3. Lesion at a site that can be photographed without including the face or any identifying feature.

**Exclusion criteria**

1. Age under 18 years.
2. Unwilling or unable to give consent.
3. Lesions on the face, or at any site that cannot be photographed without an identifying feature.
4. Lesions of the genitalia or other intimate sites.
5. Actively bleeding, heavily crusted, dressed or surgically altered lesions where the morphology cannot be seen.
6. Critically ill patients, or those in whom photography would cause distress or delay urgent treatment.
7. Photographs still technically inadequate after two attempts; recorded as screening failures.

### C. Sample size

The primary outcome is top-3 agreement: the proportion of patients in whom the dermatologist's diagnosis appears somewhere in the system's list of three most likely diagnoses.

This is a single proportion, so the sample size is obtained from the standard formula for estimating a proportion with a stated precision, N = Z²p(1−p)/d². This is the formula used in the department's earlier studies and given in standard texts of medical statistics.^[13] Z is 1.96, the value of the standard normal distribution that gives 95% confidence. The term p(1−p) is the variability of a proportion, which is largest when p is near 0.5 and smaller as p approaches 1. Dividing by d² fixes how wide the confidence interval will be.

Two values must be assumed. The first, p, is how often the system is expected to agree. On public image collections it agreed in 88.7% of cases. A lower value of 0.85 is used here because photographs taken in a working outpatient department are of more variable quality than those in curated research collections, and the diseases seen differ. The second, d, is the margin of error we are willing to accept, set at 7%. This means the study will be able to state the agreement to within about ±7%, which is close enough to be useful and is achievable in one department.

N = (1.96)² × 0.85 × 0.15 / (0.07)² = 99.96, that is 100 patients.

Ten per cent is added for photographs that cannot be used and for patients who withdraw, so 110 patients will be approached to obtain 100 that can be analysed. A tighter margin of ±5% would need 196 patients and a looser margin of ±10% only 49; 100 was chosen as the smallest number giving a useful precision within the workload of a single department.

One hundred consecutive outpatients will contain only a few biopsy-confirmed cancers, so objective 2 will be underpowered in this group alone. Cancers accumulating over the full 24 months of Aim 2 will be used for that purpose, and any analysis of cancer will be reported as exploratory.

### D. The system being tested

The system takes a photograph and returns a ranked list of diagnoses across 25 conditions, a probability that the lesion is malignant, and a recommendation on referral. The terms below are the ones used in this field and are explained again in Appendix A.

It is built on six *foundation models*: large image-reading programs trained by other groups on many millions of general photographs. The usual practice is *fine-tuning*, retraining the whole program on the medical images of interest. In development, fine-tuning made results worse, because the amount of dermatological material available is small next to the size of these programs and retraining damages the general visual knowledge they already hold.^[10] The six programs are therefore left unchanged, or *frozen*, and only a small classifier is trained on the *feature vectors* they produce, each a list of several hundred numbers describing the image.

The six were built by different groups using different methods, so their errors are largely independent. Their estimates are averaged, which is called *ensembling*.^[11] Averaging independent readers is more reliable than any one of them, as in a clinical meeting where several specialists see the same case.

A probability from such a system is not trustworthy by default: it may state 90% and be right 60% of the time. Correcting this is *calibration*, and the residual gap is the *expected calibration error*.^[14] Instead of one answer the system gives a short list built by *conformal prediction*, sized so the correct diagnosis falls inside it on 90% of occasions.^[12] Cases where the output is least certain are refused, which is *selective prediction* or *abstention*.

The system runs on one laptop, needs no internet connection, and no photograph leaves the institution.

%%FIG:fig3_system.png|Figure 3. How the system produces its output.%%

### E. Photography and the data collection application

Photographs will be taken on an ordinary smartphone of at least 12 megapixels under normal clinic lighting, using the main camera lens. No dermatoscope, spacer or other attachment will be used. This follows DermaCon-IN, which used ordinary smartphones under clinic lighting and treated the resulting variation as a fair reflection of clinical conditions.^[8] Two photographs will be taken of each lesion: one showing the site with surrounding skin, and one close-up in which the lesion fills at least half the frame. The device model is recorded with every photograph.

Data will be captured on an application written for this study, which enforces the order of steps. It shows the patient information sheet and records consent before the camera can be opened; then collects the clinical fields from fixed lists rather than free text; then guides the photograph and checks it for blur and exposure before saving, prompting a retake if needed; then stores the record under the study number and transfers it to the institutional server the same day. No name or face is stored, and nothing is sent through a messaging application or a personal cloud account.

%%FIG:fig4_app.png|Figure 4. Screens of the data collection application.%%

### F. Reference standard

The treating dermatologist's diagnosis is the reference standard for Aim 1. It is entered on the proforma before the system is run, and the dermatologist does not see the system output.

Where a biopsy is done as part of routine care, histopathology replaces the clinical diagnosis as the reference standard for that case. No biopsy will be done for this study; biopsy is decided by the treating dermatologist on clinical grounds alone. Every case records how its diagnosis was reached: histopathology, dermoscopy-supported clinical, clinical, or provisional.

%%FIG:fig5_histopath.png|Figure 5. Linking a biopsy result to the study record.%%

### G. Statistical analysis

Data will be entered in Microsoft Excel and summarised using frequency distribution and descriptive analysis.

Top-3 agreement and top-1 agreement will be given as proportions with exact binomial 95% confidence intervals. Agreement beyond chance between the system and the dermatologist will be measured with Cohen's kappa.

For referral, sensitivity, specificity, positive and negative predictive value will be calculated against the reference standard with 95% confidence intervals, and the area under the receiver operating characteristic curve reported.

Accuracy across Fitzpatrick types will be compared with the chi-square test, or Fisher's exact test where expected counts are small. Normality of continuous data will be checked by statistical test and by plot. A p-value below 0.05 will be taken as significant. Because several subgroup comparisons are planned, subgroup results will be reported as exploratory.

Analysis will use IBM SPSS version 26.0 (IBM Corp., Armonk, NY).

## IMPACT OF THE PROPOSED RESEARCH

Published systems lose accuracy on darker skin, in one case falling to chance.^[4] Whether any system works on Indian patients is unknown, because it has not been tested. This study would provide that test for a North Indian population.

There is no dermatological image collection from North India, and none anywhere that combines Indian clinical photographs with biopsy-confirmed cancers. Collections from South India and Brazil have become widely used research resources.^[8,9] A collection from this department would be useful to later work here and elsewhere, whatever the result for the system tested.

Most published evaluations report accuracy only. This study also asks whether the system's stated confidence is truthful, whether it correctly refuses cases it should not judge, and whether accuracy differs by skin type. These determine whether such a system can be used safely, and are rarely reported.

Given the number of dermatologists available in this region, a properly tested triage aid that runs on an ordinary laptop with no recurring cost would have practical use. A negative result would be equally useful: evidence that the system does not work well enough on these patients would be a needed correction to current enthusiasm, and would be published as such.

## COMPETING INTERESTS

The principal investigator has no competing interest, financial, academic or otherwise. The system was built by the principal investigator without commercial funding, using public image collections and a personal computer. No patent has been filed, no commercial arrangement exists or is planned, and no company has any part in the study or interest in its result. Neither the supervisor nor the co-supervisor has any competing interest.

The principal investigator built the system being tested and therefore has an interest in a favourable result. Two features of the design address this. The dermatologist records the reference diagnosis first and while blinded to the system output, so the comparator cannot be influenced. The primary outcome, sample size and analysis plan are fixed in this protocol before any data are collected, so the analysis cannot be changed after the results are seen. A negative result will be reported as fully as a positive one.

**Similar work in the last three years:** to the knowledge of the investigator and the supervisor, no study of artificial intelligence applied to dermatological images, and no dermatological image collection, has been carried out in this department or elsewhere in Christian Medical College and Hospital, Ludhiana, in the preceding three years. This proposal does not duplicate any ongoing or recent work in the institution.

## FINANCIAL IMPLICATIONS

**Does the study add to the cost of care?** No. It adds no investigation, procedure, medicine or extra visit. Every clinical decision, including any decision to biopsy, is taken by the treating dermatologist exactly as it would be otherwise, and any resulting cost is part of routine care. Participation adds only the five to ten minutes needed for consent and photographs. No patient will be charged anything for this study.

**Does it involve personal expense to the student?** Yes, and this will be met by the investigator. No funding is sought from the institution or any outside body.

| Item | Purpose | Approximate cost (INR) |
| Smartphone, 12 megapixel or above | Photography | 14,000–16,000 |
| Development of the data collection application | Consent, clinical data, photographs | Nil (written by the investigator) |
| Stationery and printing of proformas and information sheets | Data recording | 1,000–2,000 |
| **Total** | | **15,000–18,000** |

The laptop on which the system runs is the investigator's own and costs nothing further. There is no subscription, cloud service or licence fee. Removing the dermatoscope and camera attachment from the earlier plan reduced the cost by about ₹5,000–15,000, and follows the method used by DermaCon-IN.^[8]

## ROLE OF THE RESEARCHER

**Principal investigator (Ujjwal Thakran).** Conceived the study, built the system being tested, wrote the data collection application, and wrote this protocol. During the study he will take consent, photograph lesions, maintain the study database and the histopathology register, carry out the analysis, and write the resulting papers.

Data recording is part of these duties but is not the main role. The design of the evaluation, the building and testing of the system, the interpretation of results and their preparation for publication rest with the principal investigator.

**Supervisor and co-supervisor.** Clinical oversight; recording or adjudicating the reference diagnoses; supervision of the study in the department; review of the analysis and of the papers.

**Academic and learning value to the student.** The project gives training in matters not covered in the undergraduate course: designing and running a prospective diagnostic accuracy study; calculating and defending a sample size; the ethics of research involving photographs of patients, including consent, de-identification and the Digital Personal Data Protection Act, 2023;^[15] applying statistics to diagnostic data; and preparing work for peer review. It also gives clinical learning, since every case is seen alongside a dermatologist's diagnosis, and, in those going to biopsy, alongside the histopathology.

## PRELIMINARY RESULTS

These results were obtained on public image collections while developing the system. They are given as the reason for this application. They are not clinical validation and no clinical claim is made from them. Calibration of the system is not final and these figures may change.

Per-condition results use the F1 score, the harmonic mean of precision and sensitivity, from 0 to 1; it penalises a system that looks accurate only by favouring common diagnoses. Macro-averaged F1 is the mean of the per-condition F1 scores, counting every condition equally. For 25 conditions, chance alone would give about 0.04. These terms are explained in Appendix A.

### Nine-category model

Tested on 197,137 photographs from the International Skin Imaging Collaboration archive by five-fold cross-validation, in which the material is split into five parts and each is tested by a system built on the other four.

%%TABLE_9CLASS%%

Macro-averaged F1 was 0.660 (95% CI 0.654–0.665), against 0.637 for retraining the whole program.

### Twenty-five condition model

Tested on 25 conditions chosen to match Indian outpatient practice, using 33,250 photographs to build the system and a separate set, never used in building it, to test.

%%TABLE_25CLASS%%

Macro-averaged F1 on the test set was 0.573 (95% CI 0.552–0.591), against 0.527 (95% CI 0.510–0.543) for retraining the whole program; the confidence intervals do not overlap. Top-1 agreement was 0.730 and top-3 agreement 0.887. Top-3 agreement is the basis of the sample size above.

One limitation is clear in this table. Results are good for common inflammatory and infectious conditions, with F1 of 0.879 for viral infection, 0.766 for vitiligo and 0.692 for tinea, but poor for the cancers, with sensitivity of 0.408 for melanoma, 0.413 for basal cell carcinoma and 0.453 for squamous cell carcinoma. **The 25-condition model is therefore not adequate on its own for detecting cancer and is not offered as such.** Cancer detection is done by a separate model, given below, and any lesion that model or the clinician finds suspicious goes for specialist assessment and biopsy as usual. Whether this division works in practice is one of the things this study will examine.

About 35% of the photographs in the test set were found to be near-duplicates of photographs used in building the system, because the same atlas images appear in more than one public collection. They were identified by comparing the images numerically, removed, and all figures recalculated. The figures above are after that correction; before it they were about 0.05 higher and were wrong.

### Detection of cancer

Tested on the collection of 656 photographs in which every diagnosis was confirmed by biopsy.^[4] Of these, 171 (26.1%) were malignant.

| Measure | Single model | Six models combined | 95% CI |
| Area under the curve, all skin tones | 0.748 | 0.826 | 0.789–0.862 |
| Sensitivity at 80% specificity | — | 0.696 | — |
| Sensitivity at 90% specificity | — | 0.538 | — |
| Specificity at 90% sensitivity | — | 0.538 | — |

Combining the six models raised the area under the curve by 0.078 (95% CI 0.049–0.107, p<0.001) over one model. Repeating the whole analysis with five different random splits gave a mean of 0.830 (standard deviation 0.009).

### Accuracy by skin tone

Measured within each skin-tone group of the same collection and set against the three published systems tested on the same material.^[4]

| Fitzpatrick group | DeepDerm | ModelDerm | HAM10000 model | This system | 95% CI |
| I–II (lightest) | 0.61 | 0.64 | 0.72 | 0.782 | 0.701–0.856 |
| III–IV | — | — | — | 0.830 | 0.771–0.882 |
| V–VI (darkest) | 0.50 | 0.55 | 0.57 | 0.872 | 0.808–0.929 |
| Lightest minus darkest | +0.11 | +0.09 | +0.15 | −0.09 | — |

Each published system did 0.09 to 0.15 worse on the darkest skin than the lightest. This system did not. On the darkest skin, sensitivity at 80% specificity was 0.750.

This needs stating carefully. The confidence intervals for the lightest and darkest groups overlap, and the darkest group holds only 48 malignant lesions, so the conclusion is that no significant difference by skin tone was shown, not that accuracy is better on darker skin. The three published systems were also tested without any exposure to that collection, while this system was built and tested within it by cross-validation, so the comparison is about whether a gap between skin tones exists under each approach, not a direct comparison of accuracy.

### Safety behaviour

After calibration the gap between stated confidence and observed accuracy fell from 0.21 to 0.042. The short list held the correct diagnosis on 89.2% of occasions against a target of 90%, with a mean of 3.4 diagnoses out of 25. Refusing the 30% of cases the system was least sure about raised accuracy on the rest from 0.637 to 0.766.

### Summary

On public material the system does not lose accuracy on darker skin, states its confidence honestly and refuses cases it is unsure of. Whether this holds for patients at this hospital is unknown, and is what this study will find out.

## PROFORMA

The same proforma is used for both aims. Patients in Aim 1 complete every section. Patients contributing only to Aim 2 complete all sections except the system output.

### Section 1 — Screening and consent

1. The treating dermatologist identifies a patient meeting the criteria.
2. The patient information sheet is shown in the application in English, Hindi or Punjabi.
3. Written consent is recorded in the application. The camera cannot be opened until this is done.
4. A study number is allotted in sequence. The patient is referred to by this number only.

### Section 2 — Data recording format

**PATIENT INFORMATION**

Study number: ....................................  Date: ....................................

Age: ............  Sex: Male / Female / Other

Fitzpatrick skin type: I / II / III / IV / V / VI

Occupation: ....................................  Outdoor work: Yes / No

Comorbidities: ....................................

Immunosuppression: Yes / No    Diabetes: Yes / No

**LESION**

Site: Head-neck / Anterior trunk / Posterior trunk / Upper limb / Lower limb / Palms-soles / Nail / Scalp

Side: Right / Left / Midline / Bilateral

Duration: ............ days / weeks / months / years

Symptoms: Itch / Pain / Burning / Bleeding / None

Change: New / Growing / Changing colour / Static / Resolving

Treatment already taken: ....................................

Relevant family history: ....................................

**CLINICAL ASSESSMENT — entered before the system is run**

Clinical diagnosis: ....................................

Differential: 1. ................ 2. ................ 3. ................

Confidence: Low / Moderate / High

Biopsy indicated: Yes / No    Histopathology number: ....................................

**PHOTOGRAPHS**

Site photograph: Yes / No    Close-up: Yes / No    Device: ....................................

Quality accepted: Yes / No    If retaken, reason: ....................................

**SYSTEM OUTPUT — entered after the clinical assessment (Aim 1 only)**

Ranked diagnoses: 1. ................ 2. ................ 3. ................

Probability of malignancy: ............

System refused to answer: Yes / No

**FINAL DIAGNOSIS**

Final diagnosis: ....................................

Basis: Histopathology / Dermoscopy-supported clinical / Clinical / Provisional

**FOR AIM 2 ONLY**

Second dermatologist's diagnosis: ....................................

Agreement with first: Yes / No    If no, adjudicated diagnosis: ....................................

Included in repository: Yes / No    If no, reason: ....................................

### Section 3 — Photography

1. Room lighting is kept the same between patients; mixed daylight and tube light is avoided where possible.
2. The lesion is exposed and any medicine, powder or cosmetic cleaned off. Jewellery and clothing that would identify the patient are removed from view.
3. The main camera lens is used. The ultra-wide lens and digital zoom are not used. No attachment is fitted.
4. The camera is left in its default photograph mode. High-resolution modes, filters and beautification are turned off.
5. A site photograph is taken first, showing enough surrounding skin to place the lesion.
6. A close-up is then taken with the lesion filling at least half the frame.
7. The application checks each photograph for blur and exposure and prompts a retake if needed.
8. In patients with darker skin the camera often under-exposes the lesion. The preview is checked and exposure raised if the lesion looks unnaturally dark.
9. Photographs are transferred to the institutional server the same day and deleted from the device. Nothing is sent through a messaging application, kept on a personal device, or uploaded to an outside service.

### Section 4 — Data handling and confidentiality

1. Photographs are stored under the study number only. No name, hospital number, telephone number or address is stored with the image or in any free-text field.
2. The list linking study numbers to patients is kept separately in a password-protected file, available only to the principal investigator and the supervisor, and used only to retrieve histopathology reports.
3. Photographs are stored encrypted on an institutional computer inside the institution. No image is kept on a personal cloud service.
4. Faces are not photographed. Any identifying mark such as a tattoo is kept out of view or masked before storage.
5. Handling of data follows the Digital Personal Data Protection Act, 2023,^[15] and the directions of the Institutional Ethics Committee.
6. Data will be kept for the period the Institutional Ethics Committee specifies, after which photographs will be destroyed unless separate approval to keep them has been given.
7. Any release of the collection for research outside this institution will need a separate application to the Institutional Ethics Committee and will not happen under this approval.

### Section 5 — Histopathology

1. Where a biopsy is done on clinical grounds, the histopathology number is written on the proforma at the visit.
2. The case goes into a pending register.
3. When the report comes, the diagnosis is entered against the study number and the basis changed to histopathology.
4. Where a report cannot be traced, the case keeps its clinical diagnosis and the basis is unchanged. These cases are reported separately.
5. Where histopathology differs from the clinical diagnosis, both are kept, histopathology becomes the reference standard, and the difference is recorded.

### Section 6 — Labelling for Aim 2

1. Every photograph carries a diagnosis recorded by a dermatologist.
2. Where possible a second dermatologist records a diagnosis independently, blinded to the first, and agreement is reported with Cohen's kappa.
3. Where the two differ, a third senior dermatologist decides, and this is recorded.
4. No dermatologist recording a diagnosis is shown the system output, since that would bias the reference standard and destroy the independence of the collection.
5. Technically inadequate photographs are excluded and the reason recorded.

## APPENDIX A — EXPLANATION OF TERMS

Terms used in this proposal, in the order a reader is likely to meet them.

**Artificial intelligence (in this context).** A computer program that has been shown many photographs with their correct diagnoses and has learned associations between appearance and diagnosis. Given a new photograph it returns the diagnoses it considers most likely.

**Fitzpatrick skin type.** A scale from I to VI describing constitutive skin colour and response to sun. Most Indian patients are type IV or V.

**Area under the receiver operating characteristic curve (AUC).** A measure of how well a test separates two groups, here malignant from benign. It is the probability that a randomly chosen malignant lesion is scored higher than a randomly chosen benign one. 0.50 means chance, 1.00 means perfect separation. Values around 0.70–0.80 are usually called acceptable and 0.80–0.90 good. AUC does not depend on the cut-off chosen.

**Sensitivity.** Of all patients who truly have the condition, the proportion the test identifies. A sensitivity of 0.75 means three of every four cases are picked up.

**Specificity.** Of all patients who do not have the condition, the proportion the test correctly clears.

**Sensitivity at 80% specificity.** Sensitivity and specificity trade off against each other as the cut-off moves. Fixing specificity at 80% and reading off the sensitivity allows two tests to be compared at the same operating point.

**Top-1 agreement.** The proportion of cases in which the system's single most likely diagnosis is the same as the dermatologist's.

**Top-3 agreement.** The proportion of cases in which the dermatologist's diagnosis appears anywhere in the system's three most likely diagnoses. This is the primary outcome here because a triage tool is useful if it puts the right answer in a short list, even when it does not rank it first.

**Precision.** Of the cases the system labelled as a given condition, the proportion that truly had it. Called positive predictive value in clinical work.

**F1 score.** The harmonic mean of precision and sensitivity, from 0 to 1. It is used instead of plain accuracy because a system can look accurate simply by always choosing common diagnoses; F1 penalises that.

**Macro-averaged F1.** The average of the F1 scores of all conditions, each counting equally regardless of how common it is. For 25 conditions, chance alone gives about 0.04.

**Accuracy.** The proportion of all cases classified correctly. Misleading when conditions are unequally common, which is why F1 is reported here as well.

**95% confidence interval.** The range within which the true value is expected to lie 95% of the time if the study were repeated. A wide interval means the estimate is imprecise, usually because the number of cases is small.

**p-value.** The probability of seeing a difference at least as large as the one observed if there were truly no difference. Below 0.05 is conventionally called significant.

**Cohen's kappa.** Agreement between two raters after allowing for the agreement expected by chance alone. 0 is chance agreement and 1 is perfect.

**Cross-validation.** A way of testing on limited data. The material is divided into equal parts; each part is tested by a system built on the remaining parts, and the results are combined.

**Foundation model.** A large image-reading program trained by another group on many millions of general photographs, used here as a starting point.

**Fine-tuning.** Retraining the whole of such a program on the images of interest. It can reduce accuracy when the available material is small.

**Frozen model.** A program used exactly as obtained, without retraining.

**Feature vector.** A list of several hundred numbers a model produces to describe an image, used as the input to a small classifier.

**Ensembling.** Combining the outputs of several models, usually by averaging. Independent errors partly cancel, so the combination is more reliable than any one model.

**Calibration and expected calibration error.** Calibration adjusts a system's stated probability so it matches observed accuracy. The expected calibration error is the average gap between the two; smaller is better.

**Conformal prediction.** A method that returns a set of diagnoses rather than one, sized so the correct answer falls inside the set on a stated proportion of occasions, here 90%.

**Selective prediction (abstention).** A rule allowing the system to refuse cases it is least certain about, so those go to a specialist instead of receiving a guess.

## INFORMED CONSENT FORM

Study number: ....................................  Date: ....................................

I, .................................................... do hereby consent to participate in the study named, "PROSPECTIVE CLINICAL VALIDATION OF AN ARTIFICIAL INTELLIGENCE SYSTEM FOR THE ASSESSMENT OF SKIN DISEASE AND SKIN CANCER IN INDIAN PATIENTS".

1. I understand that I can freely withdraw from this study at any time.
2. I have had a chance to ask questions and all of my questions have been answered to my satisfaction.
3. I have been given and I understand the information concerning the nature, purpose and duration of the study and the procedures involved, including any known or expected inconvenience.
4. I understand that photographs will be taken of the affected area of my skin, that my face will not be photographed, and that my name will not be stored with the photographs.
5. I understand that the computer program being studied will not be used to decide my treatment, and that my treatment will be the same whether or not I take part.
6. I understand that the photographs may be kept for future research and may be shared with other researchers in a form in which I cannot be identified, and that any such sharing needs separate approval from the Institutional Ethics Committee.
7. My medical data remains confidential and I authorize that they be looked at only by persons involved in the research.
8. By signing this form, I give my free and informed consent to take part in this study.

(Signature / thumb impression of the participant)

Name of the participant: ....................................

Phone number: ....................................

Date: ....................................    Place: ....................................

Signature of investigator: ....................................

Date: ....................................

## सूचित सहमति प्रपत्र

अध्ययन क्रमांक: ....................................  दिनांक: ....................................

मैं, .................................................... इस अध्ययन में भाग लेने के लिए सहमति देता / देती हूँ, जिसका नाम है "भारतीय रोगियों में त्वचा रोग एवं त्वचा कैंसर के आकलन हेतु कृत्रिम बुद्धिमत्ता प्रणाली का संभाव्य नैदानिक सत्यापन"।

1. मैं समझता / समझती हूँ कि मैं किसी भी समय स्वेच्छा से इस अध्ययन से हट सकता / सकती हूँ।
2. मुझे प्रश्न पूछने का अवसर मिला है और मेरे सभी प्रश्नों के उत्तर मेरी संतुष्टि के अनुसार दिए गए हैं।
3. मुझे अध्ययन की प्रकृति, उद्देश्य एवं अवधि तथा इसमें सम्मिलित प्रक्रियाओं की जानकारी दी गई है और मैं उसे समझता / समझती हूँ, जिसमें कोई भी ज्ञात या संभावित असुविधा भी सम्मिलित है।
4. मैं समझता / समझती हूँ कि मेरी त्वचा के प्रभावित भाग की तस्वीरें ली जाएँगी, मेरे चेहरे की तस्वीर नहीं ली जाएगी, तथा तस्वीरों के साथ मेरा नाम संग्रहीत नहीं किया जाएगा।
5. मैं समझता / समझती हूँ कि जिस कंप्यूटर प्रोग्राम का अध्ययन किया जा रहा है, उसका उपयोग मेरे उपचार का निर्णय लेने के लिए नहीं किया जाएगा, और मेरा उपचार वैसा ही रहेगा चाहे मैं भाग लूँ अथवा न लूँ।
6. मैं समझता / समझती हूँ कि ये तस्वीरें भविष्य के शोध हेतु रखी जा सकती हैं तथा अन्य शोधकर्ताओं के साथ ऐसे रूप में साझा की जा सकती हैं जिसमें मेरी पहचान न हो सके, और ऐसी किसी भी साझेदारी हेतु संस्थागत आचार समिति से पृथक अनुमति आवश्यक होगी।
7. मेरा चिकित्सीय विवरण गोपनीय रहेगा और मैं अधिकृत करता / करती हूँ कि इसे केवल शोध से जुड़े व्यक्ति ही देखें।
8. इस प्रपत्र पर हस्ताक्षर करके मैं इस अध्ययन में भाग लेने हेतु अपनी स्वतंत्र एवं सूचित सहमति देता / देती हूँ।

(प्रतिभागी के हस्ताक्षर / अंगूठे का निशान)

प्रतिभागी का नाम: ....................................

दूरभाष क्रमांक: ....................................

दिनांक: ....................................    स्थान: ....................................

अन्वेषक के हस्ताक्षर: ....................................

दिनांक: ....................................

## ਸੂਚਿਤ ਸਹਿਮਤੀ ਫਾਰਮ

ਅਧਿਐਨ ਨੰਬਰ: ....................................  ਮਿਤੀ: ....................................

ਮੈਂ, .................................................... ਇਸ ਅਧਿਐਨ ਵਿੱਚ ਭਾਗ ਲੈਣ ਲਈ ਸਹਿਮਤੀ ਦਿੰਦਾ / ਦਿੰਦੀ ਹਾਂ, ਜਿਸਦਾ ਨਾਮ ਹੈ "ਭਾਰਤੀ ਮਰੀਜ਼ਾਂ ਵਿੱਚ ਚਮੜੀ ਦੇ ਰੋਗਾਂ ਅਤੇ ਚਮੜੀ ਦੇ ਕੈਂਸਰ ਦੇ ਮੁਲਾਂਕਣ ਲਈ ਆਰਟੀਫੀਸ਼ੀਅਲ ਇੰਟੈਲੀਜੈਂਸ ਪ੍ਰਣਾਲੀ ਦੀ ਸੰਭਾਵੀ ਕਲੀਨਿਕਲ ਪੁਸ਼ਟੀ"।

1. ਮੈਂ ਸਮਝਦਾ / ਸਮਝਦੀ ਹਾਂ ਕਿ ਮੈਂ ਕਿਸੇ ਵੀ ਸਮੇਂ ਆਪਣੀ ਮਰਜ਼ੀ ਨਾਲ ਇਸ ਅਧਿਐਨ ਤੋਂ ਹਟ ਸਕਦਾ / ਸਕਦੀ ਹਾਂ।
2. ਮੈਨੂੰ ਸਵਾਲ ਪੁੱਛਣ ਦਾ ਮੌਕਾ ਮਿਲਿਆ ਹੈ ਅਤੇ ਮੇਰੇ ਸਾਰੇ ਸਵਾਲਾਂ ਦੇ ਜਵਾਬ ਮੇਰੀ ਤਸੱਲੀ ਅਨੁਸਾਰ ਦਿੱਤੇ ਗਏ ਹਨ।
3. ਮੈਨੂੰ ਅਧਿਐਨ ਦੀ ਪ੍ਰਕਿਰਤੀ, ਉਦੇਸ਼ ਅਤੇ ਮਿਆਦ ਬਾਰੇ ਅਤੇ ਇਸ ਵਿੱਚ ਸ਼ਾਮਲ ਪ੍ਰਕਿਰਿਆਵਾਂ ਬਾਰੇ ਜਾਣਕਾਰੀ ਦਿੱਤੀ ਗਈ ਹੈ ਅਤੇ ਮੈਂ ਉਸਨੂੰ ਸਮਝਦਾ / ਸਮਝਦੀ ਹਾਂ, ਜਿਸ ਵਿੱਚ ਕੋਈ ਵੀ ਜਾਣੀ ਜਾਂ ਸੰਭਾਵੀ ਅਸੁਵਿਧਾ ਵੀ ਸ਼ਾਮਲ ਹੈ।
4. ਮੈਂ ਸਮਝਦਾ / ਸਮਝਦੀ ਹਾਂ ਕਿ ਮੇਰੀ ਚਮੜੀ ਦੇ ਪ੍ਰਭਾਵਿਤ ਹਿੱਸੇ ਦੀਆਂ ਤਸਵੀਰਾਂ ਲਈਆਂ ਜਾਣਗੀਆਂ, ਮੇਰੇ ਚਿਹਰੇ ਦੀ ਤਸਵੀਰ ਨਹੀਂ ਲਈ ਜਾਵੇਗੀ, ਅਤੇ ਤਸਵੀਰਾਂ ਨਾਲ ਮੇਰਾ ਨਾਮ ਸਟੋਰ ਨਹੀਂ ਕੀਤਾ ਜਾਵੇਗਾ।
5. ਮੈਂ ਸਮਝਦਾ / ਸਮਝਦੀ ਹਾਂ ਕਿ ਜਿਸ ਕੰਪਿਊਟਰ ਪ੍ਰੋਗਰਾਮ ਦਾ ਅਧਿਐਨ ਕੀਤਾ ਜਾ ਰਿਹਾ ਹੈ, ਉਸਦੀ ਵਰਤੋਂ ਮੇਰੇ ਇਲਾਜ ਦਾ ਫ਼ੈਸਲਾ ਕਰਨ ਲਈ ਨਹੀਂ ਕੀਤੀ ਜਾਵੇਗੀ, ਅਤੇ ਮੇਰਾ ਇਲਾਜ ਉਹੀ ਰਹੇਗਾ ਭਾਵੇਂ ਮੈਂ ਭਾਗ ਲਵਾਂ ਜਾਂ ਨਾ ਲਵਾਂ।
6. ਮੈਂ ਸਮਝਦਾ / ਸਮਝਦੀ ਹਾਂ ਕਿ ਇਹ ਤਸਵੀਰਾਂ ਭਵਿੱਖ ਦੀ ਖੋਜ ਲਈ ਰੱਖੀਆਂ ਜਾ ਸਕਦੀਆਂ ਹਨ ਅਤੇ ਹੋਰ ਖੋਜਕਰਤਾਵਾਂ ਨਾਲ ਅਜਿਹੇ ਰੂਪ ਵਿੱਚ ਸਾਂਝੀਆਂ ਕੀਤੀਆਂ ਜਾ ਸਕਦੀਆਂ ਹਨ ਜਿਸ ਵਿੱਚ ਮੇਰੀ ਪਛਾਣ ਨਾ ਹੋ ਸਕੇ, ਅਤੇ ਅਜਿਹੀ ਕਿਸੇ ਵੀ ਸਾਂਝ ਲਈ ਸੰਸਥਾਗਤ ਨੈਤਿਕਤਾ ਕਮੇਟੀ ਤੋਂ ਵੱਖਰੀ ਮਨਜ਼ੂਰੀ ਜ਼ਰੂਰੀ ਹੋਵੇਗੀ।
7. ਮੇਰਾ ਮੈਡੀਕਲ ਵੇਰਵਾ ਗੁਪਤ ਰਹੇਗਾ ਅਤੇ ਮੈਂ ਅਧਿਕਾਰ ਦਿੰਦਾ / ਦਿੰਦੀ ਹਾਂ ਕਿ ਇਸਨੂੰ ਸਿਰਫ਼ ਖੋਜ ਨਾਲ ਜੁੜੇ ਵਿਅਕਤੀ ਹੀ ਵੇਖਣ।
8. ਇਸ ਫਾਰਮ 'ਤੇ ਦਸਤਖ਼ਤ ਕਰਕੇ ਮੈਂ ਇਸ ਅਧਿਐਨ ਵਿੱਚ ਭਾਗ ਲੈਣ ਲਈ ਆਪਣੀ ਆਜ਼ਾਦ ਅਤੇ ਸੂਚਿਤ ਸਹਿਮਤੀ ਦਿੰਦਾ / ਦਿੰਦੀ ਹਾਂ।

(ਭਾਗੀਦਾਰ ਦੇ ਦਸਤਖ਼ਤ / ਅੰਗੂਠੇ ਦਾ ਨਿਸ਼ਾਨ)

ਭਾਗੀਦਾਰ ਦਾ ਨਾਮ: ....................................

ਫ਼ੋਨ ਨੰਬਰ: ....................................

ਮਿਤੀ: ....................................    ਸਥਾਨ: ....................................

ਅਨਵੇਸ਼ਕ ਦੇ ਦਸਤਖ਼ਤ: ....................................

ਮਿਤੀ: ....................................

## PATIENT INFORMATION SHEET

For "PROSPECTIVE CLINICAL VALIDATION OF AN ARTIFICIAL INTELLIGENCE SYSTEM FOR THE ASSESSMENT OF SKIN DISEASE AND SKIN CANCER IN INDIAN PATIENTS"

**Introduction.** This form gives you information so you can decide whether to take part in this study. Any questions will be answered by the researcher or by the contacts below. Once you have read this and asked your questions, you can decide. If you agree, you will be asked to sign the consent form.

**Name and contact of researcher:** Ujjwal Thakran, MBBS student 2022 batch, Christian Medical College and Hospital, Ludhiana. Mobile: 9999209888

**What type of study is this?** Research project.

**Is there any funding involved?** No.

**What is the purpose of the study?** To find out whether a computer program that looks at photographs of skin conditions gives the same answer as a skin specialist, in patients coming to the skin outpatient department of this hospital. Nearly all such programs were made from photographs of light-skinned patients in other countries, and it is not known whether they work on Indian skin. We also want to build a collection of skin photographs from this hospital so this and later research can be checked on patients from our own region.

**Why have I been chosen?** Because you are eighteen or above and have come to the skin outpatient department with a skin problem.

**Do I have to take part?** No. Taking part is voluntary. If you agree and later change your mind, you can withdraw at any time without giving a reason.

**What will happen if I take part?** Photographs will be taken of the affected area of your skin with a mobile phone camera, and you will be asked some questions about your skin problem. This takes about five to ten minutes. Nothing will be injected and no test will be done on you for this study. If your doctor decides a small piece of skin is needed for testing, that is part of your normal treatment and would have been decided anyway.

**Will my treatment be affected if I refuse?** No. Your treatment will be the same either way.

**Will the computer program decide my treatment?** No. The program is only being tested. Your doctor will decide your diagnosis and treatment in the usual way, and will do so before the program is used. The program's answer will not be shown to your doctor before that, will not be written in your file, and will not be used for any decision about your care.

**What are the benefits and risks?** There is no direct benefit to you. There is no physical risk, as only photographs are taken. The main risk is to your privacy, and the steps taken to protect it are below.

**Will my information be kept confidential?** Yes. Photographs are stored under a number, not your name. Your face will not be photographed. Your name, hospital number and phone number will not be stored with the photographs.

**Will my identity be kept anonymous?** Yes. If the photographs are used later or shared with other researchers, they will be shared only in a form in which you cannot be identified, and only after separate permission from the Institutional Ethics Committee.

**Will I have to pay anything?** No. There is no cost to you of any kind.

**Contact for further information:** Mr. Ujjwal Thakran, MBBS student 2022 batch, Christian Medical College and Hospital, Ludhiana. Mobile: 9999209888

**For complaints:** Dr. Jitender Mahindroo, Chairman, Institutional Ethics Committee, Christian Medical College and Hospital, Ludhiana.

## रोगी सूचना पत्र

"भारतीय रोगियों में त्वचा रोग एवं त्वचा कैंसर के आकलन हेतु कृत्रिम बुद्धिमत्ता प्रणाली का संभाव्य नैदानिक सत्यापन" अध्ययन हेतु

**परिचय।** इस पत्र का उद्देश्य आपको जानकारी देना है, ताकि आप निर्णय ले सकें कि आप इस अध्ययन में भाग लेना चाहते हैं अथवा नहीं। आपके किसी भी प्रश्न का उत्तर शोधकर्ता अथवा नीचे दिए गए संपर्क व्यक्तियों द्वारा दिया जाएगा। जानकारी समझ लेने और अपने प्रश्न पूछ लेने के पश्चात आप निर्णय ले सकते हैं। सहमत होने पर आपसे सहमति प्रपत्र भरने को कहा जाएगा।

**शोधकर्ता का नाम एवं संपर्क:** उज्ज्वल ठाकरान, एमबीबीएस विद्यार्थी, 2022 बैच, क्रिश्चियन मेडिकल कॉलेज एवं अस्पताल, लुधियाना। मोबाइल: 9999209888

**यह किस प्रकार का अध्ययन है?** शोध परियोजना।

**क्या इसमें कोई वित्तीय सहायता है?** नहीं।

**इस शोध का उद्देश्य क्या है?** यह जानना कि त्वचा रोगों की तस्वीरें देखने वाला एक कंप्यूटर प्रोग्राम इस अस्पताल के त्वचा रोग बाह्य रोगी विभाग में आने वाले रोगियों में त्वचा विशेषज्ञ के समान उत्तर देता है अथवा नहीं। ऐसे लगभग सभी प्रोग्राम अन्य देशों के गोरी त्वचा वाले रोगियों की तस्वीरों से बनाए गए हैं, और यह ज्ञात नहीं है कि वे भारतीय त्वचा पर कार्य करते हैं या नहीं। हम इस अस्पताल से त्वचा की तस्वीरों का संग्रह भी बनाना चाहते हैं, ताकि यह तथा भविष्य का शोध हमारे अपने क्षेत्र के रोगियों पर जाँचा जा सके।

**मुझे क्यों चुना गया है?** क्योंकि आपकी आयु अठारह वर्ष या उससे अधिक है और आप त्वचा रोग बाह्य रोगी विभाग में त्वचा की समस्या लेकर आए हैं।

**क्या भाग लेना आवश्यक है?** नहीं। भाग लेना स्वैच्छिक है। सहमत होने के बाद भी आप बिना कोई कारण बताए किसी भी समय हट सकते हैं।

**भाग लेने पर क्या होगा?** मोबाइल फोन के कैमरे से आपकी त्वचा के प्रभावित भाग की तस्वीरें ली जाएँगी और आपकी त्वचा की समस्या के विषय में कुछ प्रश्न पूछे जाएँगे। इसमें लगभग पाँच से दस मिनट लगेंगे। कोई इंजेक्शन नहीं लगाया जाएगा और इस अध्ययन हेतु आप पर कोई जाँच नहीं की जाएगी। यदि आपके चिकित्सक जाँच हेतु त्वचा का छोटा टुकड़ा लेने का निर्णय लेते हैं, तो वह आपके सामान्य उपचार का भाग है।

**मना करने पर क्या मेरा उपचार प्रभावित होगा?** नहीं। आपका उपचार वैसा ही रहेगा।

**क्या कंप्यूटर प्रोग्राम मेरे उपचार का निर्णय करेगा?** नहीं। प्रोग्राम की केवल जाँच की जा रही है। आपके चिकित्सक सामान्य रूप से ही निदान एवं उपचार निर्धारित करेंगे, और प्रोग्राम चलाने से पूर्व करेंगे। प्रोग्राम का उत्तर उससे पहले चिकित्सक को नहीं दिखाया जाएगा, फाइल में नहीं लिखा जाएगा, और आपकी देखभाल के किसी निर्णय हेतु प्रयोग नहीं किया जाएगा।

**इसमें क्या लाभ एवं जोखिम हैं?** आपको कोई प्रत्यक्ष लाभ नहीं है। कोई शारीरिक जोखिम नहीं है, क्योंकि केवल तस्वीरें ली जाएँगी। मुख्य जोखिम निजता से संबंधित है, जिसकी सुरक्षा के उपाय नीचे दिए गए हैं।

**क्या मेरी जानकारी गोपनीय रहेगी?** हाँ। तस्वीरें आपके नाम से नहीं, एक क्रमांक से संग्रहीत होंगी। चेहरे की तस्वीर नहीं ली जाएगी। नाम, अस्पताल क्रमांक एवं दूरभाष क्रमांक तस्वीरों के साथ संग्रहीत नहीं किए जाएँगे।

**क्या मेरी पहचान गुप्त रहेगी?** हाँ। भविष्य में प्रयोग अथवा साझा किए जाने पर तस्वीरें केवल ऐसे रूप में साझा होंगी जिसमें पहचान न हो सके, और संस्थागत आचार समिति से पृथक अनुमति के पश्चात ही।

**क्या मुझे कुछ भुगतान करना होगा?** नहीं। किसी भी प्रकार का कोई शुल्क नहीं है।

**अधिक जानकारी हेतु संपर्क:** श्री उज्ज्वल ठाकरान, एमबीबीएस विद्यार्थी 2022 बैच, क्रिश्चियन मेडिकल कॉलेज एवं अस्पताल, लुधियाना। मोबाइल: 9999209888

**शिकायत हेतु संपर्क:** डॉ. जितेंद्र महिंद्रू, अध्यक्ष, संस्थागत आचार समिति, क्रिश्चियन मेडिकल कॉलेज एवं अस्पताल, लुधियाना।

## ਮਰੀਜ਼ ਜਾਣਕਾਰੀ ਪੱਤਰ

"ਭਾਰਤੀ ਮਰੀਜ਼ਾਂ ਵਿੱਚ ਚਮੜੀ ਦੇ ਰੋਗਾਂ ਅਤੇ ਚਮੜੀ ਦੇ ਕੈਂਸਰ ਦੇ ਮੁਲਾਂਕਣ ਲਈ ਆਰਟੀਫੀਸ਼ੀਅਲ ਇੰਟੈਲੀਜੈਂਸ ਪ੍ਰਣਾਲੀ ਦੀ ਸੰਭਾਵੀ ਕਲੀਨਿਕਲ ਪੁਸ਼ਟੀ" ਅਧਿਐਨ ਲਈ

**ਜਾਣ-ਪਛਾਣ।** ਇਸ ਪੱਤਰ ਦਾ ਉਦੇਸ਼ ਤੁਹਾਨੂੰ ਜਾਣਕਾਰੀ ਦੇਣਾ ਹੈ, ਤਾਂ ਜੋ ਤੁਸੀਂ ਫ਼ੈਸਲਾ ਕਰ ਸਕੋ ਕਿ ਇਸ ਅਧਿਐਨ ਵਿੱਚ ਭਾਗ ਲੈਣਾ ਹੈ ਜਾਂ ਨਹੀਂ। ਤੁਹਾਡੇ ਕਿਸੇ ਵੀ ਸਵਾਲ ਦਾ ਜਵਾਬ ਖੋਜਕਰਤਾ ਜਾਂ ਹੇਠਾਂ ਦਿੱਤੇ ਸੰਪਰਕਾਂ ਦੁਆਰਾ ਦਿੱਤਾ ਜਾਵੇਗਾ। ਜਾਣਕਾਰੀ ਸਮਝ ਲੈਣ ਅਤੇ ਸਵਾਲ ਪੁੱਛ ਲੈਣ ਤੋਂ ਬਾਅਦ ਤੁਸੀਂ ਫ਼ੈਸਲਾ ਕਰ ਸਕਦੇ ਹੋ। ਸਹਿਮਤ ਹੋਣ 'ਤੇ ਤੁਹਾਨੂੰ ਸਹਿਮਤੀ ਫਾਰਮ ਭਰਨ ਲਈ ਕਿਹਾ ਜਾਵੇਗਾ।

**ਖੋਜਕਰਤਾ ਦਾ ਨਾਮ ਅਤੇ ਸੰਪਰਕ:** ਉੱਜਵਲ ਠਾਕਰਾਨ, ਐਮਬੀਬੀਐਸ ਵਿਦਿਆਰਥੀ, 2022 ਬੈਚ, ਕ੍ਰਿਸ਼ਚੀਅਨ ਮੈਡੀਕਲ ਕਾਲਜ ਅਤੇ ਹਸਪਤਾਲ, ਲੁਧਿਆਣਾ। ਮੋਬਾਈਲ: 9999209888

**ਇਹ ਕਿਸ ਪ੍ਰਕਾਰ ਦਾ ਅਧਿਐਨ ਹੈ?** ਖੋਜ ਪ੍ਰੋਜੈਕਟ।

**ਕੀ ਇਸ ਵਿੱਚ ਕੋਈ ਵਿੱਤੀ ਸਹਾਇਤਾ ਹੈ?** ਨਹੀਂ।

**ਇਸ ਖੋਜ ਦਾ ਉਦੇਸ਼ ਕੀ ਹੈ?** ਇਹ ਜਾਣਨਾ ਕਿ ਚਮੜੀ ਦੇ ਰੋਗਾਂ ਦੀਆਂ ਤਸਵੀਰਾਂ ਵੇਖਣ ਵਾਲਾ ਕੰਪਿਊਟਰ ਪ੍ਰੋਗਰਾਮ ਇਸ ਹਸਪਤਾਲ ਦੇ ਚਮੜੀ ਰੋਗ ਬਾਹਰੀ ਮਰੀਜ਼ ਵਿਭਾਗ ਵਿੱਚ ਆਉਣ ਵਾਲੇ ਮਰੀਜ਼ਾਂ ਵਿੱਚ ਚਮੜੀ ਦੇ ਮਾਹਿਰ ਵਾਂਗ ਜਵਾਬ ਦਿੰਦਾ ਹੈ ਜਾਂ ਨਹੀਂ। ਅਜਿਹੇ ਲਗਭਗ ਸਾਰੇ ਪ੍ਰੋਗਰਾਮ ਹੋਰ ਦੇਸ਼ਾਂ ਦੇ ਗੋਰੀ ਚਮੜੀ ਵਾਲੇ ਮਰੀਜ਼ਾਂ ਦੀਆਂ ਤਸਵੀਰਾਂ ਤੋਂ ਬਣੇ ਹਨ, ਅਤੇ ਇਹ ਪਤਾ ਨਹੀਂ ਕਿ ਉਹ ਭਾਰਤੀ ਚਮੜੀ 'ਤੇ ਕੰਮ ਕਰਦੇ ਹਨ ਜਾਂ ਨਹੀਂ। ਅਸੀਂ ਇਸ ਹਸਪਤਾਲ ਤੋਂ ਚਮੜੀ ਦੀਆਂ ਤਸਵੀਰਾਂ ਦਾ ਸੰਗ੍ਰਹਿ ਵੀ ਬਣਾਉਣਾ ਚਾਹੁੰਦੇ ਹਾਂ।

**ਮੈਨੂੰ ਕਿਉਂ ਚੁਣਿਆ ਗਿਆ ਹੈ?** ਕਿਉਂਕਿ ਤੁਹਾਡੀ ਉਮਰ ਅਠਾਰਾਂ ਸਾਲ ਜਾਂ ਵੱਧ ਹੈ ਅਤੇ ਤੁਸੀਂ ਚਮੜੀ ਰੋਗ ਬਾਹਰੀ ਮਰੀਜ਼ ਵਿਭਾਗ ਵਿੱਚ ਚਮੜੀ ਦੀ ਸਮੱਸਿਆ ਲੈ ਕੇ ਆਏ ਹੋ।

**ਕੀ ਭਾਗ ਲੈਣਾ ਜ਼ਰੂਰੀ ਹੈ?** ਨਹੀਂ। ਭਾਗ ਲੈਣਾ ਸਵੈ-ਇੱਛਤ ਹੈ। ਸਹਿਮਤ ਹੋਣ ਤੋਂ ਬਾਅਦ ਵੀ ਤੁਸੀਂ ਬਿਨਾਂ ਕਾਰਨ ਦੱਸੇ ਕਿਸੇ ਵੀ ਸਮੇਂ ਹਟ ਸਕਦੇ ਹੋ।

**ਭਾਗ ਲੈਣ 'ਤੇ ਕੀ ਹੋਵੇਗਾ?** ਮੋਬਾਈਲ ਫ਼ੋਨ ਦੇ ਕੈਮਰੇ ਨਾਲ ਤੁਹਾਡੀ ਚਮੜੀ ਦੇ ਪ੍ਰਭਾਵਿਤ ਹਿੱਸੇ ਦੀਆਂ ਤਸਵੀਰਾਂ ਲਈਆਂ ਜਾਣਗੀਆਂ ਅਤੇ ਕੁਝ ਸਵਾਲ ਪੁੱਛੇ ਜਾਣਗੇ। ਇਸ ਵਿੱਚ ਪੰਜ ਤੋਂ ਦਸ ਮਿੰਟ ਲੱਗਣਗੇ। ਕੋਈ ਟੀਕਾ ਨਹੀਂ ਲਗਾਇਆ ਜਾਵੇਗਾ ਅਤੇ ਇਸ ਅਧਿਐਨ ਲਈ ਕੋਈ ਜਾਂਚ ਨਹੀਂ ਕੀਤੀ ਜਾਵੇਗੀ। ਜੇ ਤੁਹਾਡੇ ਡਾਕਟਰ ਜਾਂਚ ਲਈ ਚਮੜੀ ਦਾ ਛੋਟਾ ਟੁਕੜਾ ਲੈਣ ਦਾ ਫ਼ੈਸਲਾ ਕਰਦੇ ਹਨ, ਤਾਂ ਉਹ ਤੁਹਾਡੇ ਆਮ ਇਲਾਜ ਦਾ ਹਿੱਸਾ ਹੈ।

**ਇਨਕਾਰ ਕਰਨ 'ਤੇ ਕੀ ਮੇਰਾ ਇਲਾਜ ਪ੍ਰਭਾਵਿਤ ਹੋਵੇਗਾ?** ਨਹੀਂ। ਤੁਹਾਡਾ ਇਲਾਜ ਉਹੀ ਰਹੇਗਾ।

**ਕੀ ਕੰਪਿਊਟਰ ਪ੍ਰੋਗਰਾਮ ਮੇਰੇ ਇਲਾਜ ਦਾ ਫ਼ੈਸਲਾ ਕਰੇਗਾ?** ਨਹੀਂ। ਪ੍ਰੋਗਰਾਮ ਦੀ ਸਿਰਫ਼ ਜਾਂਚ ਹੋ ਰਹੀ ਹੈ। ਤੁਹਾਡੇ ਡਾਕਟਰ ਆਮ ਤਰੀਕੇ ਨਾਲ ਹੀ ਨਿਦਾਨ ਅਤੇ ਇਲਾਜ ਤੈਅ ਕਰਨਗੇ, ਅਤੇ ਪ੍ਰੋਗਰਾਮ ਚਲਾਉਣ ਤੋਂ ਪਹਿਲਾਂ ਕਰਨਗੇ। ਪ੍ਰੋਗਰਾਮ ਦਾ ਜਵਾਬ ਉਸ ਤੋਂ ਪਹਿਲਾਂ ਡਾਕਟਰ ਨੂੰ ਨਹੀਂ ਵਿਖਾਇਆ ਜਾਵੇਗਾ, ਫਾਈਲ ਵਿੱਚ ਨਹੀਂ ਲਿਖਿਆ ਜਾਵੇਗਾ, ਅਤੇ ਤੁਹਾਡੀ ਦੇਖਭਾਲ ਦੇ ਕਿਸੇ ਫ਼ੈਸਲੇ ਲਈ ਨਹੀਂ ਵਰਤਿਆ ਜਾਵੇਗਾ।

**ਇਸ ਵਿੱਚ ਕੀ ਲਾਭ ਅਤੇ ਜੋਖਮ ਹਨ?** ਤੁਹਾਨੂੰ ਕੋਈ ਸਿੱਧਾ ਲਾਭ ਨਹੀਂ ਹੈ। ਕੋਈ ਸਰੀਰਕ ਜੋਖਮ ਨਹੀਂ ਹੈ, ਕਿਉਂਕਿ ਸਿਰਫ਼ ਤਸਵੀਰਾਂ ਲਈਆਂ ਜਾਣਗੀਆਂ। ਮੁੱਖ ਜੋਖਮ ਨਿੱਜਤਾ ਨਾਲ ਸਬੰਧਤ ਹੈ, ਜਿਸਦੀ ਸੁਰੱਖਿਆ ਦੇ ਉਪਾਅ ਹੇਠਾਂ ਦਿੱਤੇ ਹਨ।

**ਕੀ ਮੇਰੀ ਜਾਣਕਾਰੀ ਗੁਪਤ ਰਹੇਗੀ?** ਹਾਂ। ਤਸਵੀਰਾਂ ਤੁਹਾਡੇ ਨਾਮ ਨਾਲ ਨਹੀਂ, ਇੱਕ ਨੰਬਰ ਨਾਲ ਸਟੋਰ ਹੋਣਗੀਆਂ। ਚਿਹਰੇ ਦੀ ਤਸਵੀਰ ਨਹੀਂ ਲਈ ਜਾਵੇਗੀ। ਨਾਮ, ਹਸਪਤਾਲ ਨੰਬਰ ਅਤੇ ਫ਼ੋਨ ਨੰਬਰ ਤਸਵੀਰਾਂ ਨਾਲ ਸਟੋਰ ਨਹੀਂ ਹੋਣਗੇ।

**ਕੀ ਮੇਰੀ ਪਛਾਣ ਗੁਪਤ ਰਹੇਗੀ?** ਹਾਂ। ਭਵਿੱਖ ਵਿੱਚ ਵਰਤੋਂ ਜਾਂ ਸਾਂਝ ਦੀ ਸੂਰਤ ਵਿੱਚ ਤਸਵੀਰਾਂ ਸਿਰਫ਼ ਅਜਿਹੇ ਰੂਪ ਵਿੱਚ ਸਾਂਝੀਆਂ ਹੋਣਗੀਆਂ ਜਿਸ ਵਿੱਚ ਪਛਾਣ ਨਾ ਹੋ ਸਕੇ, ਅਤੇ ਸੰਸਥਾਗਤ ਨੈਤਿਕਤਾ ਕਮੇਟੀ ਦੀ ਵੱਖਰੀ ਮਨਜ਼ੂਰੀ ਤੋਂ ਬਾਅਦ ਹੀ।

**ਕੀ ਮੈਨੂੰ ਕੁਝ ਭੁਗਤਾਨ ਕਰਨਾ ਪਵੇਗਾ?** ਨਹੀਂ। ਕਿਸੇ ਵੀ ਪ੍ਰਕਾਰ ਦਾ ਕੋਈ ਖ਼ਰਚਾ ਨਹੀਂ ਹੈ।

**ਹੋਰ ਜਾਣਕਾਰੀ ਲਈ ਸੰਪਰਕ:** ਸ੍ਰੀ ਉੱਜਵਲ ਠਾਕਰਾਨ, ਐਮਬੀਬੀਐਸ ਵਿਦਿਆਰਥੀ 2022 ਬੈਚ, ਕ੍ਰਿਸ਼ਚੀਅਨ ਮੈਡੀਕਲ ਕਾਲਜ ਅਤੇ ਹਸਪਤਾਲ, ਲੁਧਿਆਣਾ। ਮੋਬਾਈਲ: 9999209888

**ਸ਼ਿਕਾਇਤ ਲਈ ਸੰਪਰਕ:** ਡਾ. ਜਿਤੇਂਦਰ ਮਹਿੰਦਰੂ, ਚੇਅਰਮੈਨ, ਸੰਸਥਾਗਤ ਨੈਤਿਕਤਾ ਕਮੇਟੀ, ਕ੍ਰਿਸ਼ਚੀਅਨ ਮੈਡੀਕਲ ਕਾਲਜ ਅਤੇ ਹਸਪਤਾਲ, ਲੁਧਿਆਣਾ।

## REFERENCES

1. Tschandl P, Rosendahl C, Kittler H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci Data. 2018;5:180161.
2. Esteva A, Kuprel B, Novoa RA, Ko J, Swetter SM, Blau HM, et al. Dermatologist-level classification of skin cancer with deep neural networks. Nature. 2017;542(7639):115-118.
3. Groh M, Harris C, Soenksen L, Lau F, Han R, Kim A, et al. Evaluating deep neural networks trained on clinical images in dermatology with the Fitzpatrick 17k dataset. Proc IEEE/CVF Conf Comput Vis Pattern Recognit Workshops. 2021;1820-1828.
4. Daneshjou R, Vodrahalli K, Novoa RA, Jenkins M, Liang W, Rotemberg V, et al. Disparities in dermatology AI performance on a diverse, curated clinical image set. Sci Adv. 2022;8(31):eabq6147.
5. Gupta AK, Bharadwaj M, Mehrotra R. Skin cancer concerns in people of colour: risk factors and prevention. Asian Pac J Cancer Prev. 2016;17(12):5257-5264.
6. Agarwala S, Sharma S, Bhattacharyya J. Melanoma in the Indian subcontinent: an overview. Indian J Dermatol Venereol Leprol. 2020;86(5):483-492.
7. Daneshjou R, Smith MP, Sun MD, Rotemberg V, Zou J. Lack of transparency and potential bias in artificial intelligence data sets and algorithms: a scoping review. JAMA Dermatol. 2021;157(11):1362-1369.
8. Madarkar SS, Madarkar M, Venkatesh M, Bansal D, Prakash T, Mopuri KR, et al. DermaCon-IN: a multi-concept annotated dermatological image dataset of Indian skin disorders for clinical AI research. arXiv:2506.06099. 2025.
9. Pacheco AGC, Lima GR, Salomao AS, Krohling B, Biral IP, de Angelo GG, et al. PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones. Data Brief. 2020;32:106221.
10. Kumar A, Raghunathan A, Jones R, Ma T, Liang P. Fine-tuning can distort pretrained features and underperform out-of-distribution. Int Conf Learn Represent (ICLR). 2022.
11. Oquab M, Darcet T, Moutakanni T, Vo H, Szafraniec M, Khalidov V, et al. DINOv2: learning robust visual features without supervision. Trans Mach Learn Res. 2024.
12. Angelopoulos AN, Bates S, Malik J, Jordan MI. Uncertainty sets for image classifiers using conformal prediction. Int Conf Learn Represent (ICLR). 2021.
13. Daniel WW, Cross CL. Biostatistics: a foundation for analysis in the health sciences. 11th ed. Hoboken (NJ): Wiley; 2018.
14. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. Proc 34th Int Conf Mach Learn. 2017;70:1321-1330.
15. Ministry of Electronics and Information Technology, Government of India. The Digital Personal Data Protection Act, 2023. New Delhi: Government of India; 2023.
