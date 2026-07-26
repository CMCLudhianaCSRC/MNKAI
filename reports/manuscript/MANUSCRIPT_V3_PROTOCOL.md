%%DOCTYPE%%RESEARCH PROJECT
%%TITLE%%PROSPECTIVE CLINICAL VALIDATION OF AN ARTIFICIAL INTELLIGENCE SYSTEM FOR THE ASSESSMENT OF SKIN CONDITIONS AND SKIN CANCER ACROSS SKIN TONES, AND ESTABLISHMENT OF A NORTH INDIAN DERMATOLOGICAL IMAGE REPOSITORY
%%CANDIDATE%%UJJWAL THAKRAN|MBBS Student Batch 2022,|Christian Medical College & Hospital,|Ludhiana, Punjab-141008
%%SUPERVISOR%%DR. [SUPERVISOR NAME]|[Designation],|Department of Dermatology,|Christian Medical College and Hospital,|Ludhiana, Punjab-141008
%%COSUPERVISOR%%DR. [CO-SUPERVISOR NAME]|Professor & Head,|Department of Dermatology,|Christian Medical College and Hospital,|Ludhiana, Punjab-141008

## ABSTRACT:

**Introduction:** Artificial intelligence (AI) systems that interpret photographs of skin lesions are increasingly proposed as an aid to clinical triage. Nearly all such systems have been developed using photographs of light-skinned patients. When three published systems were independently evaluated on a skin-tone-balanced, biopsy-confirmed image set, their ability to separate malignant from benign lesions on the darkest skin fell to between 0.50 and 0.57 on a scale where 0.50 represents chance. No dermatological image repository exists from North India, and consequently no such system has ever been tested on the population of this region. The investigator has developed an AI system which, in preliminary testing on publicly available image sets, does not show this loss of performance on darker skin. That preliminary work is presented here as justification; it does not constitute clinical validation.

**Aim and Objectives:** To prospectively evaluate the diagnostic performance of this AI system against the diagnosis of a qualified dermatologist in patients attending the Dermatology outpatient department of this institution, and concurrently to establish a de-identified repository of dermatological photographs from this population.

**Material & Methods:** A prospective observational study will be conducted in the Department of Dermatology over a period of two years. One hundred consecutive consenting patients aged 18 years and above will be enrolled for the validation objective. For each patient, standardised photographs of the lesion will be obtained using a smartphone camera with a fixed-distance attachment, together with structured clinical details. The dermatologist's diagnosis, recorded before and independently of the AI output, will serve as the reference standard; histopathological diagnosis will serve as the reference standard wherever a biopsy is clinically indicated. The AI system's ranked differential diagnosis will be compared against this reference. The photographs and accompanying de-identified data collected over the full two-year period will constitute the image repository. The AI system will not be used to influence patient management at any stage.

**Keywords:** Artificial intelligence, dermatology, skin cancer, skin of colour, Fitzpatrick skin type, diagnostic accuracy, clinical validation

## INTRODUCTION

Diseases of the skin constitute one of the largest categories of outpatient presentation in India. Estimates of the dermatologist-to-population ratio in the country remain of the order of one dermatologist for every fifty thousand to one lakh persons, and the distribution is heavily weighted towards metropolitan centres. A substantial proportion of patients with skin complaints are therefore assessed in the first instance by a physician who is not a dermatologist, or are not assessed at all. Any tool capable of indicating reliably which patients require specialist referral, and with what urgency, would therefore have considerable practical value in screening camps, peripheral health centres and teleconsultation services.

Artificial intelligence, and specifically the branch of it concerned with the interpretation of images, has been proposed for precisely this purpose. In broad terms, such a system is a computer program that has been shown a very large number of photographs together with their correct diagnoses, and which learns statistical associations between the appearance of a lesion and its diagnosis. Presented with a new photograph, it returns a ranked list of the diagnoses it considers most probable. Several such systems have been reported to achieve accuracy comparable to that of dermatologists under experimental conditions.

These reports must, however, be interpreted with caution, for two distinct reasons.

The first concerns the populations from which the developmental photographs were drawn. The large publicly available image collections on which almost all such systems are built — principally the International Skin Imaging Collaboration archive and the HAM10000 collection — were assembled in Europe, North America and Australia [1]. They consist predominantly of light-skinned patients, and a large proportion of the images are dermatoscopic, that is, obtained through a magnifying instrument applied to the skin with a coupling fluid, rather than being ordinary clinical photographs of the kind that would be taken in an Indian outpatient department. Groh and colleagues annotated a large public collection by Fitzpatrick skin type and demonstrated that performance varied systematically with skin tone, with darker skin types being underrepresented and less accurately classified [2].

The clinical consequence of this imbalance was quantified directly by Daneshjou and colleagues, who assembled the Diverse Dermatology Images (DDI) set [3]. This consists of 656 photographs obtained from 570 patients, deliberately balanced across skin-tone groups, in which every single diagnosis was confirmed by histopathology rather than by clinical impression. Three previously published AI systems were then evaluated on this set without any adaptation to it. All three performed substantially worse on the darkest skin than on the lightest. Expressed as the area under the receiver operating characteristic curve — a measure of a test's ability to separate two groups, in which 0.50 indicates performance no better than the toss of a coin and 1.00 indicates perfect separation — the first system fell from 0.61 on Fitzpatrick types I–II to 0.50 on types V–VI, the second from 0.64 to 0.55, and the third from 0.72 to 0.57 [3]. In other words, on the darkest skin one of these systems performed no better than guessing. When the same systems were subsequently retrained using photographs drawn from the collection itself, performance on the darkest skin improved to approximately 0.76 and 0.78 [3].

This deficiency is of particular importance in the Indian context, and not merely because most Indian patients have Fitzpatrick type IV or V skin. Cutaneous malignancy in brown skin differs in its presentation. Melanoma in Indian and other pigmented populations arises disproportionately on the palms, soles and nail apparatus rather than on chronically sun-exposed sites, is frequently mistaken initially for a benign pigmented lesion or a traumatic haemorrhage, and is consequently diagnosed at a more advanced stage with a correspondingly poorer prognosis [4, 5]. A screening aid that performs least reliably in exactly the population and at exactly the sites where diagnosis is already most difficult offers little benefit and carries a real risk of false reassurance.

The second reason for caution concerns the manner in which such systems present their conclusions. The majority of published systems return a diagnosis and nothing further. They do not indicate the degree of confidence attaching to that diagnosis; they do not identify those cases which lie outside the range of material on which they were developed and which they are therefore unfit to judge; and they do not report whether their performance differs between identifiable groups of patients. Daneshjou and colleagues, in a separate review of the field, drew attention to the general absence of transparency regarding the composition of developmental data sets and the consequent potential for undetected bias [6]. A clinician presented with an unqualified diagnosis from such a system has no rational basis on which to decide when it should be disregarded, which is a serious impediment to safe use.

Against this background the investigator has, over the preceding academic year, developed an AI system intended specifically for this setting. Three design decisions distinguish it from the systems described above and are described in the Material and Methods section: the underlying image-interpretation programs are used unaltered rather than being retrained, which was found empirically to give better results when the quantity of available dermatological material is limited; the outputs of several such programs are combined, in the manner of a clinical meeting at which several specialists consider the same case; and the system is required to state its confidence honestly, to provide a short list of possibilities rather than a single answer, and to decline to offer an opinion on those cases about which it is least certain, recommending specialist referral instead.

In preliminary testing on publicly available image collections, including the biopsy-confirmed DDI set described above, this system has not shown the loss of performance on darker skin that characterises previously published systems. Those preliminary findings are presented in this document. It must be stated at the outset that they are preliminary in a strict sense: they were obtained on photographs collected in other countries, principally the United States and Europe, and they say nothing whatever about how the system would perform on patients attending this hospital. Photographs taken in a research setting differ from those taken in a busy outpatient department in lighting, in framing, in the presence of artefact, and in the spectrum of disease encountered. The step that has not been taken, and which cannot be taken without institutional approval, is to test the system prospectively on real patients.

That is the permission sought in this application. Three permissions are requested: first, to evaluate the system prospectively against dermatologist diagnosis in one hundred consenting patients attending the Dermatology outpatient department of this institution; second, to collect and retain a de-identified repository of dermatological photographs from this department over a period of two years; and third, to publish the findings of both activities. To the investigator's knowledge no dermatological image repository from North India exists, and none anywhere combines Indian clinical photography with histopathologically confirmed malignancy. Comparable repositories have been established in South India and in Brazil and have become widely used international research resources [7, 11].

## AIMS and OBJECTIVES

**Aim:** To prospectively evaluate the diagnostic performance of an artificial intelligence system for the assessment of skin conditions in patients attending a tertiary care hospital in North India, and to establish a de-identified dermatological image repository from this population.

**Primary Objective:**
- To determine the agreement between the ranked differential diagnosis generated by the AI system and the diagnosis of a qualified dermatologist, in one hundred consecutive consenting patients presenting to the Dermatology outpatient department.

**Secondary Objectives:**
- To determine the sensitivity and specificity of the AI system for the identification of lesions requiring biopsy or specialist referral, using histopathological diagnosis as the reference standard wherever a biopsy is performed.
- To determine whether the diagnostic performance of the system differs across Fitzpatrick skin types IV, V and VI within this population.
- To determine the proportion of cases in which the system correctly declines to offer an opinion, and the accuracy of the system on those cases upon which it does offer an opinion.
- To establish, over a period of two years, a de-identified repository of dermatological photographs with accompanying structured clinical data and, where available, histopathological confirmation, from patients attending this department.

## STUDY DESIGN

This will be a prospective observational study conducted in the Department of Dermatology, Christian Medical College and Hospital, Ludhiana, over a period of two years from the date of institutional approval.

The study comprises two components conducted concurrently within the same department.

The **validation component** will enrol one hundred consecutive consenting patients aged eighteen years and above attending the Dermatology outpatient department. For each patient, standardised photographs of the lesion will be obtained together with structured clinical details, following which the AI system will be run on the photograph. The dermatologist's clinical diagnosis will be recorded before and independently of the AI output.

The **repository component** will continue over the full two-year period and will accumulate de-identified photographs and accompanying structured data from consenting patients attending the same department, including those enrolled in the validation component.

The AI system will operate strictly in parallel with, and will at no point influence, the clinical care of any participant. Its output will not be shown to the treating clinician before the clinical diagnosis has been recorded, will not be entered into the patient record, and will not be used to determine investigation, referral or treatment. No participant's management will differ in any respect from what it would have been had the study not been conducted.

## INCLUSION CRITERIA

- Patients aged 18 years or above attending the Dermatology outpatient department with a skin lesion or eruption
- Patients willing and able to provide written informed consent
- Lesion or eruption located at a site that can be photographed without exposure of the face or of any other identifying feature

## EXCLUSION CRITERIA

- Patients under 18 years of age
- Patients unwilling or unable to provide informed consent
- Lesions confined to the face, or to any site that cannot be photographed without capturing an identifying feature
- Lesions of the genitalia or of other intimate sites
- Actively bleeding, heavily crusted, dressed or surgically altered lesions in which the underlying morphology cannot be assessed
- Patients who are critically ill, or in whom photography would cause distress or delay in urgent management
- Photographs of inadequate technical quality after two attempts, which will be recorded as screening failures

## SAMPLE SIZE

The primary objective is to estimate the proportion of cases in which the diagnosis recorded by the dermatologist appears within the ranked list of three most probable diagnoses generated by the AI system, hereafter referred to as top-3 agreement.

In preliminary testing on publicly available image collections, the system achieved a top-3 agreement of 0.887. A more conservative value of 0.85 has been assumed for the present study, in recognition of the fact that photographs obtained in a routine outpatient setting are of more variable quality than those in curated research collections, and that the spectrum of disease will differ.

The minimum sample size is therefore estimated to be N=100, by using the formula N=Z²*p*(1-p)/d², where Z=1.96 is the standard normal deviate at the 95% confidence level, p=0.85 is the anticipated proportion of cases showing top-3 agreement, and d is the margin of error (7%).

N = (1.96)² × 0.85 × 0.15 / (0.07)² = 99.96 ≈ 100

Allowing approximately ten per cent for photographs of inadequate technical quality and for withdrawal of consent, one hundred and ten patients will be approached in order to obtain one hundred evaluable cases.

The corresponding sample sizes at other levels of precision are given below, and one hundred was selected as the smallest sample providing a margin of error within ten per cent while remaining feasible within the outpatient workload of a single department.

| Anticipated agreement (p) | Margin of error ±5% | ±7% | ±10% |
| 0.80 | 246 | 125 | 62 |
| 0.85 | 196 | **100** | 49 |
| 0.90 | 138 | 71 | 35 |

It is acknowledged that one hundred consecutive outpatients will contain only a small number of histopathologically confirmed malignancies, and that the secondary objective concerning malignancy will therefore be underpowered within the validation component considered alone. Accumulation of malignant cases over the full two-year repository period, together with cases identified from the departmental histopathology records, is intended to address this, and any analysis of malignancy will be presented as exploratory.

## STATISTICAL ANALYSIS

The data will be entered into a Microsoft Excel sheet. Data will be summarised using frequency distribution and descriptive analysis.

The primary outcome, top-3 agreement, will be expressed as a proportion with an exact binomial 95% confidence interval. Top-1 agreement will be reported in the same manner. Agreement between the AI system and the dermatologist beyond that expected by chance will be assessed using Cohen's kappa.

For the secondary objective concerning referral, sensitivity, specificity, positive and negative predictive values will be calculated against the reference standard with 95% confidence intervals, and the area under the receiver operating characteristic curve will be reported.

Comparison of performance across Fitzpatrick skin types will use the chi-square test or Fisher's exact test as appropriate to the expected cell counts. The normality of continuous data will be assessed through statistical tests and graphical methods. A p-value of 0.05 will be considered statistically significant. Given the number of subgroup comparisons planned, subgroup analyses will be interpreted as exploratory and this will be stated explicitly in any report.

All statistical analysis will be performed using IBM SPSS (Statistical Packages for Social Sciences, version 26.0. Armonk, NY: IBM Corp.).

## CONFLICT STATEMENT

The principal investigator has no conflict of interest financially, academically, or in any other way. The AI system evaluated in this study was developed by the principal investigator without commercial funding, using publicly available image collections and a personal computer. No patent has been filed and no commercial arrangement exists or is contemplated in respect of it. No funding is sought or involved in the conduct of this study.

## MATERIAL AND METHODS:

### A. The artificial intelligence system under evaluation

The system accepts a photograph of a skin lesion and returns a ranked list of probable diagnoses across twenty-five conditions, an estimate of the probability that the lesion is malignant, and a recommendation as to whether specialist referral is indicated. Its construction is described below in the terms conventionally used in this field, with each term explained.

**Foundation models used without alteration.** The system is built upon six publicly available *foundation models* — large image-interpretation programs developed by major research groups, each trained on many millions of general photographs. Conventionally, such a program is *fine-tuned*, meaning that the whole of it is retrained on the specific medical images of interest. In the course of development it was found that fine-tuning consistently reduced performance in this setting, because the quantity of dermatological material available is small relative to the size of these programs, and retraining degrades the general visual knowledge already present within them. This phenomenon has been described formally in the computer-science literature [8]. The approach adopted instead was to leave the six programs entirely unaltered — described as keeping them *frozen* — and to train only a small classifier on the numerical descriptions, or *feature vectors*, that they produce. A feature vector is a list of several hundred numbers summarising the visual content of an image.

**Ensembling.** The six frozen programs were developed by different groups using different methods, and consequently make substantially independent errors. Their individual estimates of the probability of malignancy are averaged, a procedure termed *ensembling*. The rationale is that of a clinical meeting at which several specialists trained at different institutions consider the same case: the consensus is more reliable than any individual opinion.

**Calibration.** A probability output by such a system is not automatically trustworthy; a system may state ninety per cent confidence yet be correct only sixty per cent of the time. The process of correcting this correspondence is termed *calibration*, and the residual discrepancy is quantified as the *expected calibration error*, being the average difference between stated confidence and observed accuracy.

**Prediction sets.** Rather than returning a single diagnosis, the system returns a short list constructed by a method termed *conformal prediction*, which is designed so that the correct diagnosis lies within the list on a prespecified proportion of occasions, here ninety per cent.

**Selective prediction.** The system identifies those cases in which its own output is least certain and declines to offer an opinion upon them, recommending specialist referral instead. This is termed *selective prediction* or *abstention*.

The entire system operates on a single laptop computer and requires neither an internet connection nor institutional computing infrastructure. No patient photograph is transmitted outside the institution at any stage of the validation component.

### B. Reference standard

The diagnosis recorded by the treating dermatologist will serve as the reference standard for the primary objective. This diagnosis will be recorded on the study proforma before, and independently of, any output of the AI system, and the dermatologist will remain blinded to that output throughout.

Where a biopsy is performed as part of routine clinical care, the histopathological diagnosis will supersede the clinical diagnosis as the reference standard for that case. No biopsy will be performed for the purposes of this study; biopsy will be undertaken solely upon clinical indication as determined by the treating dermatologist.

The evidence grade of every case will be recorded explicitly as histopathological, dermoscopy-supported clinical, clinical, or provisional, and results will be reported separately by grade.

### C. Photographic method

Photographs will be obtained using a smartphone camera of at least twelve megapixels fitted with a three-dimensionally printed spacer which fixes the distance between the camera and the skin, thereby ensuring that scale and focus are consistent between operators and between visits. A colour reference card and an adhesive scale will be included within the field of the photograph to permit correction of colour and measurement of lesion size. A clip-on dermatoscope will be used in addition for pigmented and suspicious lesions.

Three photographs will be obtained of each lesion: a regional photograph demonstrating the anatomical site, a close-up photograph at the fixed distance, and, where indicated, a dermatoscopic photograph. The full photographic procedure is set out in the Data Collection Protocol below.

## PRELIMINARY RESULTS

The findings presented in this section were obtained on publicly available image collections during the developmental phase and are presented solely as justification for the study proposed. They do not constitute clinical validation and no clinical claim is made upon them. Final calibration of the system has not been completed and these figures are expected to be refined.

The measure used throughout for individual conditions is the F1 score, being the harmonic mean of precision and sensitivity, which ranges from 0 to 1 and which penalises a system that achieves apparent accuracy merely by favouring common diagnoses. Sensitivity carries its usual meaning, being the proportion of cases of a given condition that the system correctly identified. Macro-averaged F1 denotes the mean of the per-condition F1 scores, in which every condition contributes equally irrespective of its frequency; for twenty-five conditions a value of approximately 0.04 would be expected by chance alone.

### Nine-category dermatoscopic model

The system was first evaluated on a nine-category task derived from the International Skin Imaging Collaboration archive, comprising 197,137 photographs assessed by five-fold cross-validation, in which the material is divided into five parts and each part is assessed in turn by a system developed on the remaining four. Per-category performance is given in Table 1.

%%TABLE_9CLASS%%

The macro-averaged F1 score was 0.660 (95% CI 0.654–0.665), compared with 0.637 for the conventional approach of retraining the whole program.

### Twenty-five condition clinical model

The system was then evaluated on twenty-five conditions selected to reflect the spectrum encountered in Indian outpatient practice, using 33,250 photographs for development and a separate set of photographs, never used during development, for testing. Per-condition performance on the independent test set is given in Table 2.

%%TABLE_25CLASS%%

The macro-averaged F1 score on the independent test set was 0.573 (95% CI 0.552–0.591), against 0.527 (95% CI 0.510–0.543) for the conventional retraining approach; the confidence intervals do not overlap. Top-1 agreement was 0.730 and top-3 agreement 0.887, the latter being the basis of the sample size calculation above.

During this evaluation it was found that approximately 35% of the photographs in the test set were near-identical duplicates of photographs already used in development, arising because the same atlas images appear in more than one public collection. These were identified by comparing the numerical descriptions of the images, removed, and all figures recalculated. The figures reported here are after that correction; before it, they were approximately 0.05 higher and were incorrect.

### Detection of malignancy

Performance in separating malignant from benign lesions was assessed on the DDI collection of 656 photographs, in which every diagnosis was confirmed by histopathology. Of these, 171 (26.1%) were malignant. Results are given in Table 3.

| Measure | Single program | Six programs combined | 95% CI |
| Area under the curve, all skin tones | 0.748 | 0.826 | 0.789–0.862 |
| Sensitivity at 80% specificity | — | 0.696 | — |
| Sensitivity at 90% specificity | — | 0.538 | — |
| Specificity at 90% sensitivity | — | 0.538 | — |

Combining the six programs improved the area under the curve by 0.078 (95% CI 0.049–0.107, p<0.001) over a single program. The result was stable when the entire analysis was repeated with five different random divisions of the material (mean 0.830, standard deviation 0.009).

### Performance by skin tone

Performance was then examined separately within each Fitzpatrick skin-tone group of the same collection, and compared with the three previously published systems evaluated by Daneshjou and colleagues on the identical material [3]. Results are given in Table 4.

| Fitzpatrick group | DeepDerm [3] | ModelDerm [3] | HAM10000 model [3] | Present system | 95% CI |
| I–II (lightest) | 0.61 | 0.64 | 0.72 | 0.782 | 0.701–0.856 |
| III–IV | — | — | — | 0.830 | 0.771–0.882 |
| V–VI (darkest) | 0.50 | 0.55 | 0.57 | 0.872 | 0.808–0.929 |
| Difference, lightest minus darkest | +0.11 | +0.09 | +0.15 | −0.09 | — |

Each of the three published systems performed between 0.09 and 0.15 worse on the darkest skin than on the lightest. The present system did not show this pattern. On the darkest skin, sensitivity at 80% specificity was 0.750.

This finding must be stated with precision. The confidence intervals for the lightest and darkest groups overlap, and the darkest group contains only 48 malignant lesions; the defensible conclusion is therefore that no significant difference by skin tone was demonstrated, and not that performance is superior on darker skin. Furthermore, the three published systems were evaluated without any exposure to this collection, whereas the present system was developed and assessed within it by cross-validation; the comparison therefore concerns the presence or absence of a gap between skin tones under each approach, rather than being a direct comparison of accuracy.

### Behaviour of the safety features

After calibration, the discrepancy between stated confidence and observed accuracy fell from 0.21 to 0.042. The short list of possible diagnoses contained the correct diagnosis on 89.2% of occasions against a target of 90%, and contained a mean of 3.4 diagnoses out of 25. When the system declined to offer an opinion on the 30% of cases about which it was least certain, accuracy on the remaining cases rose from 0.637 to 0.766.

### Summary of preliminary findings

The system does not, on publicly available material, show the loss of performance on darker skin that characterises previously published systems, and it states its confidence honestly and declines to answer when uncertain. Whether any of this holds for patients attending this hospital is unknown, and is the question this study is designed to answer.

## DATA COLLECTION PROTOCOL

### Section 1 — Consent and screening

1. The patient is identified in the outpatient queue by the treating dermatologist as having a skin lesion or eruption meeting the inclusion criteria.
2. The Patient Information Sheet is provided and explained in the language the patient understands. The patient is given the opportunity to ask questions.
3. Written informed consent is obtained on the Informed Consent Form before any photograph is taken. No photograph is taken before consent.
4. A study identification number is allocated in sequence. From this point forward the patient is referred to only by this number.

### Section 2 — Clinical proforma

**PATIENT INFORMATION**

Study ID: ................................................  Date: ................................................

Age: ................  Gender: Male / Female / Other

Fitzpatrick skin type: I / II / III / IV / V / VI

Occupation: ................................................  Outdoor exposure: Yes / No

Comorbidities (if any): ................................................

Immunosuppression: Yes / No  Diabetes mellitus: Yes / No

**LESION INFORMATION**

Anatomical site: Head-neck / Anterior trunk / Posterior trunk / Upper limb / Lower limb / Palms-soles / Nail / Scalp

Laterality: Right / Left / Midline / Bilateral

Duration: ................ (days / weeks / months / years)

Symptoms: Itch / Pain / Burning / Bleeding / None

Evolution: New / Increasing in size / Changing in colour / Static / Resolving

Previous treatment taken: ................................................

Family history relevant to the presentation: ................................................

**CLINICAL ASSESSMENT (recorded before the AI system is run)**

Provisional clinical diagnosis: ................................................

Differential diagnoses: 1. ................ 2. ................ 3. ................

Clinician's confidence: Low / Moderate / High

Biopsy indicated: Yes / No   If yes, histopathology number: ................................................

**PHOTOGRAPHS OBTAINED**

Regional photograph: Yes / No   Close-up photograph: Yes / No   Dermatoscopic photograph: Yes / No / Not applicable

Device used: ................................................

**AI SYSTEM OUTPUT (recorded after the clinical assessment has been entered)**

Ranked diagnoses: 1. ................ 2. ................ 3. ................

Estimated probability of malignancy: ................

System declined to offer an opinion: Yes / No

**FINAL DIAGNOSIS**

Final diagnosis: ................................................

Basis of final diagnosis: Histopathology / Dermoscopy-supported clinical / Clinical / Provisional

### Section 3 — Photographic procedure

1. The examination room lighting is kept consistent between patients. Mixed daylight and tube light is avoided where possible.
2. The lesion is exposed and cleaned of any applied medication, powder or cosmetic. Jewellery and clothing that would identify the patient are removed from the field.
3. The smartphone is fitted with the printed spacer. The main camera lens is used; the ultra-wide lens and any digital zoom are not used.
4. The camera is set to its default photograph mode. High-resolution modes, filters and beautification settings are disabled. Flash is not used for the regional photograph.
5. A regional photograph is taken first, demonstrating the anatomical site with sufficient surrounding skin to establish context.
6. The colour reference card and adhesive scale are placed adjacent to, but not overlapping, the lesion.
7. The spacer is placed in gentle contact with the skin, taking care not to blanch it, since pressure alters the colour of the lesion. A close-up photograph is taken with the lesion occupying at least half of the field.
8. For pigmented, ulcerated or otherwise suspicious lesions, a dermatoscopic photograph is taken in addition.
9. Each photograph is inspected immediately. A photograph is retaken if it is out of focus, if there is glare over the lesion, if a shadow crosses the lesion, or if any identifying feature is visible.
10. In darker-skinned patients the automatic exposure of the camera frequently under-exposes the lesion. The preview is checked and exposure increased manually where the lesion appears unnaturally dark.
11. Photographs are transferred the same day to the designated institutional computer and deleted from the device. Photographs are not transmitted through any messaging application, are not stored on any personal device, and are not uploaded to any external service.

### Section 4 — Data handling and confidentiality

1. Photographs are stored under the study identification number only. No name, hospital registration number, telephone number or address is stored with the image or in any free-text field of the proforma.
2. The document linking study identification numbers to patient identity is maintained separately, in a password-protected file accessible only to the principal investigator and the supervisor, and is required solely for the retrieval of histopathology reports.
3. Photographs are stored on an institutional computer with encryption at rest, physically located within the institution. No image is stored on a personal cloud service.
4. Faces are not photographed. Any incidental identifying feature such as a tattoo is excluded from the field or masked before storage.
5. All handling of data will comply with the Digital Personal Data Protection Act, 2023, and with the guidance of the Institutional Ethics Committee.
6. Data will be retained for the period specified by the Institutional Ethics Committee, following which photographs will be destroyed unless a separate approval for continued retention or release has been obtained.
7. Any future release of the repository for research use beyond this institution will be the subject of a separate application to the Institutional Ethics Committee, and will not proceed under the present approval.

### Section 5 — Histopathology linkage

1. Where a biopsy is performed on clinical indication, the histopathology number is recorded on the proforma at the time of the visit.
2. The case is entered in a pending register.
3. When the report becomes available, the histopathological diagnosis is entered against the study identification number and the evidence grade of the case is amended to histopathological.
4. Where a report cannot be retrieved, the case is retained with its clinical diagnosis and the evidence grade is left unchanged. Such cases are reported separately.
5. Where the histopathological diagnosis differs from the clinical diagnosis, both are retained; the histopathological diagnosis becomes the reference standard and the discordance is recorded.

### Section 6 — Labelling for the repository component

1. Every photograph in the repository will carry a diagnosis recorded by a qualified dermatologist.
2. Where feasible, a second dermatologist will record a diagnosis independently and blinded to the first, and agreement between them will be reported using Cohen's kappa.
3. Where the two differ, a third senior dermatologist will adjudicate and the adjudicated diagnosis will be recorded as such.
4. No dermatologist recording a diagnosis will be shown the output of the AI system, since knowledge of it would bias the reference standard and would destroy the independence of the repository.
5. Photographs of inadequate technical quality will be excluded from the repository and the reason for exclusion recorded.

## INFORMED CONSENT FORM

Study ID: ................................................  Date: ................................................

I, ................................................ do hereby consent to participate in the study named, "PROSPECTIVE CLINICAL VALIDATION OF AN ARTIFICIAL INTELLIGENCE SYSTEM FOR THE ASSESSMENT OF SKIN CONDITIONS AND SKIN CANCER ACROSS SKIN TONES, AND ESTABLISHMENT OF A NORTH INDIAN DERMATOLOGICAL IMAGE REPOSITORY".

- I understand that I can freely withdraw from this study at any time.
- I have had a chance to ask questions and all of my questions have been answered to my satisfaction.
- I have been given and I understand the information on the survey concerning its nature, purpose, and duration as well as the procedures involved in the study, including any known or expected inconvenience.
- I understand that photographs will be taken of the affected area of my skin, that my face will not be photographed, and that my name will not be stored with the photographs.
- I understand that the computer program being studied will not be used to decide my treatment, and that my treatment will be exactly the same whether or not I take part.
- I understand that the photographs may be kept for future research and may be shared with other researchers in a form in which I cannot be identified, and that any such sharing requires separate approval from the Institutional Ethics Committee.
- My medical data remains confidential and I authorize that they be looked at only by persons involved in the research.
- By signing this form, I give my free and informed consent to take part in this study.

(Signature / thumb impression of the participant)

Name of the participant: ................................................

Phone no: ................................................

Date: ................................................  Place: ................................................

Signature of investigator: ................................................

Date: ................................................

## PATIENT INFORMATION SHEET

For "PROSPECTIVE CLINICAL VALIDATION OF AN ARTIFICIAL INTELLIGENCE SYSTEM FOR THE ASSESSMENT OF SKIN CONDITIONS AND SKIN CANCER ACROSS SKIN TONES, AND ESTABLISHMENT OF A NORTH INDIAN DERMATOLOGICAL IMAGE REPOSITORY"

### Introduction

The purpose of this form is to provide you with information, so you can decide whether to participate in this study. Any questions you may have will be answered by the researcher or by the other contact persons provided below. Once you are familiar with the information on this sheet and have asked any questions you may have, you can decide whether or not to participate. If you agree, you will be asked to fill in the consent form for this study.

**Research title:** Prospective clinical validation of an artificial intelligence system for the assessment of skin conditions and skin cancer across skin tones, and establishment of a North Indian dermatological image repository.

**Name and contact details of researcher:** Ujjwal Thakran, MBBS student 2022 batch, Christian Medical College and Hospital, Ludhiana. Mob: 9999209888

**What type of study is this?**
- Research project.

**Is there any funding involved?**
- No.

**What is the purpose of the research project?**
- To find out whether a computer program that looks at photographs of skin conditions gives the same answer as a skin specialist, in patients attending the skin outpatient department of this hospital. Almost all such programs have been made using photographs of light-skinned patients from other countries, and it is not known whether they work properly on Indian skin. We also wish to build a collection of skin photographs from this hospital, so that this and future research can be checked on patients from our own region.

**Why have I been chosen to participate in this research project?**
- You have been chosen because you are eighteen years of age or above and have come to the skin outpatient department with a skin problem.

**Do I have to take part in this research project?**
- Your participation in this study is voluntary. It is up to you to decide whether you want to participate or not. If you decide to participate, you are still free to withdraw at any time and without giving a reason.

**What happens if I take part in this research project?**
- Photographs will be taken of the affected area of your skin using a mobile phone camera, and some questions will be asked about your skin problem. This will take about five to ten minutes. Nothing will be injected, and no test will be done on you for the purposes of this study. If your doctor decides that a small piece of skin needs to be taken for testing, that decision is part of your normal treatment and would have been made whether or not you took part in this study.

**Will my treatment be withheld if I refuse to take part in this study project?**
- No. Your treatment will be exactly the same whether or not you take part.

**Will the computer program decide my treatment?**
- No. The computer program is only being tested. Your doctor will decide your diagnosis and treatment in the usual way, and will do so before the computer program is used. The result given by the computer program will not be shown to your doctor before that, will not be written in your file, and will not be used to decide anything about your care.

**What are the benefits and risks involved in taking part in this research project?**
- There is no direct benefit to you. There is no physical risk, as only photographs will be taken. The main risk is to your privacy, and the steps taken to protect it are described below.

**Will the information provided by me be kept confidential?**
- Yes. Your photographs will be stored under a number, not your name. Your face will not be photographed. Your name, hospital number and phone number will not be stored with the photographs.

**Will my identity be kept anonymous?**
- Yes. The photographs will be stored in a form in which you cannot be identified. If the photographs are used for research in the future or shared with other researchers, they will be shared only in this anonymous form, and only after separate permission has been taken from the Institutional Ethics Committee.

**Contact for further information**
- Mr. Ujjwal Thakran, MBBS student 2022 batch, Christian Medical College and Hospital, Ludhiana. Mob: 9999209888

**For complaints contact:**
- Dr. Jitender Mahindroo, Chairman, Institutional Ethics Committee, Christian Medical College and Hospital, Ludhiana.

## REFERENCES:

1. Tschandl P, Rosendahl C, Kittler H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci Data. 2018;5:180161.
2. Groh M, Harris C, Soenksen L, Lau F, Han R, Kim A, et al. Evaluating deep neural networks trained on clinical images in dermatology with the Fitzpatrick 17k dataset. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops; 2021. p. 1820-1828.
3. Daneshjou R, Vodrahalli K, Novoa RA, Jenkins M, Liang W, Rotemberg V, et al. Disparities in dermatology AI performance on a diverse, curated clinical image set. Sci Adv. 2022;8(31):eabq6147.
4. Gupta AK, Bharadwaj M, Mehrotra R. Skin cancer concerns in people of colour: risk factors and prevention. Asian Pac J Cancer Prev. 2016;17(12):5257-5264.
5. Agarwala S, Sharma S, Bhattacharyya J. Melanoma in the Indian subcontinent: an overview. Indian J Dermatol Venereol Leprol. 2020;86(5):483-492.
6. Daneshjou R, Smith MP, Sun MD, Rotemberg V, Zou J. Lack of transparency and potential bias in artificial intelligence data sets and algorithms: a scoping review. JAMA Dermatol. 2021;157(11):1362-1369.
7. Pacheco AGC, Lima GR, Salomao AS, Krohling B, Biral IP, de Angelo GG, et al. PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones. Data Brief. 2020;32:106221.
8. Kumar A, Raghunathan A, Jones R, Ma T, Liang P. Fine-tuning can distort pretrained features and underperform out-of-distribution. In: International Conference on Learning Representations (ICLR); 2022.
9. Oquab M, Darcet T, Moutakanni T, Vo H, Szafraniec M, Khalidov V, et al. DINOv2: learning robust visual features without supervision. Trans Mach Learn Res. 2024.
10. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. In: Proceedings of the 34th International Conference on Machine Learning; 2017. p. 1321-1330.
11. Angelopoulos AN, Bates S, Malik J, Jordan MI. Uncertainty sets for image classifiers using conformal prediction. In: International Conference on Learning Representations (ICLR); 2021.
12. Ministry of Electronics and Information Technology, Government of India. The Digital Personal Data Protection Act, 2023. New Delhi: Government of India; 2023.
