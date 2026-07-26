%%DOCTYPE%%RESEARCH PROJECT
%%TITLE%%Prospective clinical validation of an artificial intelligence system for the assessment of skin disease and skin cancer in Indian patients
%%CANDIDATE%%UJJWAL THAKRAN|MBBS Student Batch 2022,|Christian Medical College & Hospital,|Ludhiana, Punjab-141008
%%SUPERVISOR%%DR. [SUPERVISOR NAME]|[Designation],|Department of Dermatology,|Christian Medical College and Hospital,|Ludhiana, Punjab-141008
%%COSUPERVISOR%%DR. [CO-SUPERVISOR NAME]|Professor & Head,|Department of Dermatology,|Christian Medical College and Hospital,|Ludhiana, Punjab-141008

## CHECKLIST OF COMPONENTS

This proposal has been prepared against the departmental Checklist for Assessment of Research Proposal. The location of each required component is given below for the convenience of the Supervisor and the Head of Department in completing the checklist.

| Sr. No. | Criteria / guideline | Section in this document |
| 1 | Introduction (2–3 pages, focused) | Introduction |
| 2 | Aims (broad aim of study) | Aim |
| 3 | Objectives (primary / secondary / tertiary) | Objectives |
| 4a | Study design — clear and appropriate | Material and Methods, A |
| 4b | Study population; inclusion; exclusion; sample size with calculation | Material and Methods, B |
| 4c | Description of methodology | Material and Methods, C–E |
| 4d | Description of statistics | Material and Methods, F |
| 5 | Impact — does the research add to current knowledge? | Impact of the Proposed Research |
| 6 | References in Harvard style; correct in-text citation; recent (within 10 years) | References |
| 7 | Abstract (200–300 words) with 5–6 keywords | Abstract |
| 8 | Title — appropriate length and reflective of the research | Title page |
| 9 | Proforma — clear and specific data recording format | Proforma |
| 10 | Consent form with Hindi and Punjabi translations | Informed Consent Form (English, Hindi, Punjabi) |
| 11 | Patient information sheet with Hindi and Punjabi translations | Patient Information Sheet (English, Hindi, Punjabi) |
| 12 | Competing interests clarified; similar study in department/CMC in last 3 years | Competing Interests |
| 13 | Financial implications — added cost of care; personal expense to student | Financial Implications |
| 14 | Researcher role clear (not primarily data recording); academic impact | Role of the Researcher |

## ABSTRACT

**Background:** Artificial intelligence systems that interpret photographs of skin lesions are increasingly proposed as an aid to clinical triage, but almost all have been developed using photographs of light-skinned patients. When three published systems were independently evaluated on a skin-tone-balanced, biopsy-confirmed image set, their ability to separate malignant from benign lesions on the darkest skin fell to between 0.50 and 0.57, where 0.50 represents chance (Daneshjou et al., 2022). No dermatological image repository exists from North India, and consequently no such system has ever been tested in this population.

**Aim and Objectives:** To prospectively evaluate the diagnostic performance of an artificial intelligence system developed by the investigator against the diagnosis of a qualified dermatologist in patients attending this institution, and concurrently to establish a de-identified dermatological image repository from this population.

**Material and Methods:** A prospective observational study will be conducted in the Department of Dermatology over two years. One hundred consecutive consenting patients aged eighteen years and above will be enrolled for the validation objective; the sample size was calculated using N=Z²p(1−p)/d² with p=0.85 and d=7%. Standardised photographs will be obtained using a smartphone camera fitted with a fixed-distance attachment, together with structured clinical data. The dermatologist's diagnosis, recorded before and independently of the system output, will serve as the reference standard, superseded by histopathology wherever biopsy is clinically indicated. The system will operate strictly in parallel and will not influence patient management. In preliminary testing on publicly available image sets the system achieved an area under the curve of 0.872 (95% CI 0.808–0.929) on the darkest skin types; these findings are preliminary and require clinical validation.

**Keywords:** Artificial intelligence, dermatology, skin cancer, skin of colour, Fitzpatrick skin type, diagnostic accuracy

## INTRODUCTION

Diseases of the skin constitute one of the largest categories of outpatient presentation in India. The dermatologist-to-population ratio in the country remains of the order of one dermatologist for every fifty thousand to one lakh persons, and the distribution is heavily weighted towards metropolitan centres. A substantial proportion of patients with skin complaints are therefore assessed in the first instance by a physician who is not a dermatologist, or are not assessed at all. Any tool capable of indicating reliably which patients require specialist referral, and with what urgency, would therefore have considerable practical value in screening camps, peripheral health centres and teleconsultation services.

Artificial intelligence, and specifically the branch of it concerned with the interpretation of images, has been proposed for precisely this purpose. In broad terms, such a system is a computer program that has been shown a very large number of photographs together with their correct diagnoses, and which learns statistical associations between the appearance of a lesion and its diagnosis. Presented with a new photograph, it returns a ranked list of the diagnoses it considers most probable. Several such systems have been reported to achieve accuracy comparable to that of dermatologists under experimental conditions. These reports must, however, be interpreted with caution, for two distinct reasons.

The first concerns the populations from which the developmental photographs were drawn. The large publicly available image collections on which almost all such systems are built were assembled in Europe, North America and Australia (Tschandl, Rosendahl and Kittler, 2018). They consist predominantly of light-skinned patients, and a large proportion of the images are dermatoscopic, that is, obtained through a magnifying instrument applied to the skin, rather than being ordinary clinical photographs of the kind taken in an Indian outpatient department. Groh et al. (2021) annotated a large public collection by Fitzpatrick skin type and demonstrated that performance varied systematically with skin tone, darker skin types being both underrepresented and less accurately classified.

The clinical consequence of this imbalance was quantified directly by Daneshjou et al. (2022), who assembled the Diverse Dermatology Images set. This consists of 656 photographs obtained from 570 patients, deliberately balanced across skin-tone groups, in which every diagnosis was confirmed by histopathology rather than by clinical impression. Three previously published systems were evaluated on this set without any adaptation to it. All three performed substantially worse on the darkest skin than on the lightest. Expressed as the area under the receiver operating characteristic curve — a measure of a test's ability to separate two groups, in which 0.50 indicates performance no better than the toss of a coin and 1.00 indicates perfect separation — the first fell from 0.61 on Fitzpatrick types I–II to 0.50 on types V–VI, the second from 0.64 to 0.55, and the third from 0.72 to 0.57. On the darkest skin, one of these systems performed no better than guessing. When the same systems were subsequently retrained using photographs drawn from that collection, performance on the darkest skin improved to approximately 0.76 and 0.78.

This deficiency is of particular importance in the Indian context, and not merely because most Indian patients have Fitzpatrick type IV or V skin. Cutaneous malignancy in brown skin differs in its presentation. Melanoma in Indian and other pigmented populations arises disproportionately on the palms, soles and nail apparatus rather than on chronically sun-exposed sites, is frequently mistaken initially for a benign pigmented lesion or for traumatic haemorrhage, and is consequently diagnosed at a more advanced stage with a correspondingly poorer prognosis (Gupta, Bharadwaj and Mehrotra, 2016; Agarwala, Sharma and Bhattacharyya, 2020). A screening aid that performs least reliably in exactly the population, and at exactly the sites, where diagnosis is already most difficult offers little benefit and carries a real risk of false reassurance.

The second reason for caution concerns the manner in which such systems present their conclusions. The majority return a diagnosis and nothing further. They do not indicate the confidence attaching to that diagnosis; they do not identify cases lying outside the range of material on which they were developed and which they are therefore unfit to judge; and they do not report whether performance differs between identifiable groups of patients. Daneshjou et al. (2021), in a separate review, drew attention to the general absence of transparency regarding the composition of developmental data sets and the consequent potential for undetected bias. A clinician presented with an unqualified diagnosis has no rational basis on which to decide when it should be disregarded, which is a serious impediment to safe use.

Against this background the investigator has, over the preceding academic year, developed an artificial intelligence system intended specifically for this setting, using publicly available image collections and a personal computer. Its construction is described in the Material and Methods section. In preliminary testing, including on the biopsy-confirmed set described above, the system has not shown the loss of performance on darker skin that characterises previously published systems. Those preliminary findings are presented in this proposal as justification for the study.

It must be stated plainly that they are preliminary in a strict sense. They were obtained on photographs collected in other countries, and they say nothing about how the system would perform on patients attending this hospital. Photographs taken in a research setting differ from those taken in a busy outpatient department in lighting, framing, artefact, and in the spectrum of disease encountered. The step that has not been taken, and which cannot be taken without institutional approval, is to test the system prospectively on real patients. That is the permission sought in this application.

## AIM

To evaluate prospectively the diagnostic performance of an artificial intelligence system for the assessment of skin disease and skin cancer in patients attending a tertiary care hospital in North India, and to establish a de-identified dermatological image repository from this population.

## OBJECTIVES

**Primary objective**

- To determine the agreement between the ranked differential diagnosis generated by the artificial intelligence system and the diagnosis recorded by a qualified dermatologist, in one hundred consecutive consenting patients presenting to the Dermatology outpatient department.

**Secondary objectives**

- To determine the sensitivity and specificity of the system for the identification of lesions requiring biopsy or specialist referral, using histopathological diagnosis as the reference standard wherever a biopsy is performed.
- To determine whether the diagnostic performance of the system differs across Fitzpatrick skin types IV, V and VI within this population.
- To determine the proportion of cases in which the system correctly declines to offer an opinion, and the accuracy of the system on those cases upon which it does offer an opinion.

**Tertiary objectives**

- To establish, over a period of two years, a de-identified repository of dermatological photographs with accompanying structured clinical data and, where available, histopathological confirmation, from patients attending this department.
- To describe the spectrum and relative frequency of dermatological conditions presenting to this department, and the distribution of Fitzpatrick skin types within it.
- To assess the technical adequacy of standardised smartphone photography as performed in a routine outpatient setting, and to record the proportion and causes of photographs rejected as ungradable.

## MATERIAL AND METHODS

### A. Study design

This will be a prospective observational study conducted in the Department of Dermatology, Christian Medical College and Hospital, Ludhiana, over a period of two years from the date of institutional approval.

The study comprises two components conducted concurrently within the same department. The **validation component** will enrol one hundred consecutive consenting patients for comparison of the system output against dermatologist diagnosis. The **repository component** will continue over the full two-year period, accumulating de-identified photographs and structured data from consenting patients attending the same department, including those enrolled in the validation component.

The artificial intelligence system will operate strictly in parallel with, and will at no point influence, the clinical care of any participant. Its output will not be shown to the treating clinician before the clinical diagnosis has been recorded, will not be entered into the patient record, and will not be used to determine investigation, referral or treatment. No participant's management will differ in any respect from what it would have been had the study not been conducted.

### B. Study population

**Inclusion criteria**

- Patients aged 18 years or above attending the Dermatology outpatient department with a skin lesion or eruption
- Patients willing and able to provide written informed consent
- Lesion or eruption located at a site that can be photographed without exposure of the face or of any other identifying feature

**Exclusion criteria**

- Patients under 18 years of age
- Patients unwilling or unable to provide informed consent
- Lesions confined to the face, or to any site that cannot be photographed without capturing an identifying feature
- Lesions of the genitalia or of other intimate sites
- Actively bleeding, heavily crusted, dressed or surgically altered lesions in which the underlying morphology cannot be assessed
- Patients who are critically ill, or in whom photography would cause distress or delay in urgent management
- Photographs of inadequate technical quality after two attempts, which will be recorded as screening failures

**Sample size**

The primary objective is to estimate the proportion of cases in which the diagnosis recorded by the dermatologist appears within the ranked list of three most probable diagnoses generated by the system, hereafter referred to as top-3 agreement.

In preliminary testing on publicly available image collections the system achieved a top-3 agreement of 0.887. A more conservative value of 0.85 has been assumed for the present study, in recognition of the fact that photographs obtained in a routine outpatient setting are of more variable quality than those in curated research collections, and that the spectrum of disease will differ.

The minimum sample size is therefore estimated to be N=100, by using the formula N=Z²*p*(1-p)/d², where Z=1.96 is the standard normal deviate at the 95% confidence level, p=0.85 is the anticipated proportion of cases showing top-3 agreement, and d is the margin of error (7%).

N = (1.96)² × 0.85 × 0.15 / (0.07)² = 99.96 ≈ 100

Allowing approximately ten per cent for photographs of inadequate technical quality and for withdrawal of consent, one hundred and ten patients will be approached in order to obtain one hundred evaluable cases. The corresponding sample sizes at other levels of precision are given below; one hundred was selected as the smallest sample providing a margin of error within ten per cent while remaining feasible within the outpatient workload of a single department.

| Anticipated agreement (p) | Margin of error ±5% | ±7% | ±10% |
| 0.80 | 246 | 125 | 62 |
| 0.85 | 196 | **100** | 49 |
| 0.90 | 138 | 71 | 35 |

It is acknowledged that one hundred consecutive outpatients will contain only a small number of histopathologically confirmed malignancies, and that the secondary objective concerning malignancy will therefore be underpowered within the validation component considered alone. Accumulation of malignant cases over the full two-year repository period is intended to address this, and any analysis of malignancy will be presented as exploratory.

### C. The artificial intelligence system under evaluation

The system accepts a photograph of a skin lesion and returns a ranked list of probable diagnoses across twenty-five conditions, an estimate of the probability that the lesion is malignant, and a recommendation as to whether specialist referral is indicated. Its construction is described below in the terms conventionally used in this field, each of which is explained on first use.

**Foundation models used without alteration.** The system is built upon six publicly available *foundation models* — large image-interpretation programs developed by major research groups, each trained on many millions of general photographs. Conventionally such a program is *fine-tuned*, meaning that the whole of it is retrained on the specific medical images of interest. During development it was found that fine-tuning consistently reduced performance in this setting, because the quantity of dermatological material available is small relative to the size of these programs, and retraining degrades the general visual knowledge already present within them; this phenomenon has been described formally by Kumar et al. (2022). The approach adopted instead was to leave the six programs entirely unaltered — described as keeping them *frozen* — and to train only a small classifier upon the numerical descriptions, or *feature vectors*, that they produce. A feature vector is a list of several hundred numbers summarising the visual content of an image.

**Ensembling.** The six frozen programs were developed by different groups using different methods and consequently make substantially independent errors. Their individual estimates of the probability of malignancy are averaged, a procedure termed *ensembling*. The rationale is that of a clinical meeting at which several specialists trained at different institutions consider the same case: the consensus is more reliable than any individual opinion.

**Calibration.** A probability output by such a system is not automatically trustworthy; a system may state ninety per cent confidence yet be correct only sixty per cent of the time. The process of correcting this correspondence is termed *calibration* (Guo et al., 2017), and the residual discrepancy is quantified as the *expected calibration error*, being the average difference between stated confidence and observed accuracy.

**Prediction sets.** Rather than returning a single diagnosis, the system returns a short list constructed by a method termed *conformal prediction* (Angelopoulos et al., 2021), designed so that the correct diagnosis lies within the list on a prespecified proportion of occasions, here ninety per cent.

**Selective prediction.** The system identifies those cases in which its own output is least certain and declines to offer an opinion upon them, recommending specialist referral instead. This is termed *selective prediction* or *abstention*.

The entire system operates on a single laptop computer and requires neither an internet connection nor institutional computing infrastructure. No patient photograph will be transmitted outside the institution at any stage.

### D. Reference standard

The diagnosis recorded by the treating dermatologist will serve as the reference standard for the primary objective. This diagnosis will be recorded on the study proforma before, and independently of, any output of the system, and the dermatologist will remain blinded to that output throughout.

Where a biopsy is performed as part of routine clinical care, the histopathological diagnosis will supersede the clinical diagnosis as the reference standard for that case. No biopsy will be performed for the purposes of this study; biopsy will be undertaken solely upon clinical indication as determined by the treating dermatologist. The evidence grade of every case will be recorded explicitly as histopathological, dermoscopy-supported clinical, clinical, or provisional, and results will be reported separately by grade.

### E. Photographic method

Photographs will be obtained using a smartphone camera of at least twelve megapixels fitted with a three-dimensionally printed spacer which fixes the distance between the camera and the skin, thereby ensuring that scale and focus are consistent between operators and between visits. A colour reference card and an adhesive scale will be included within the field to permit correction of colour and measurement of lesion size. A clip-on dermatoscope will be used in addition for pigmented and suspicious lesions.

Three photographs will be obtained of each lesion: a regional photograph demonstrating the anatomical site, a close-up photograph at the fixed distance, and, where indicated, a dermatoscopic photograph. The full procedure is set out in the Proforma section below.

### F. Statistical analysis

The data will be entered into a Microsoft Excel sheet. Data will be summarised using frequency distribution and descriptive analysis.

The primary outcome, top-3 agreement, will be expressed as a proportion with an exact binomial 95% confidence interval. Top-1 agreement will be reported in the same manner. Agreement between the system and the dermatologist beyond that expected by chance will be assessed using Cohen's kappa.

For the secondary objective concerning referral, sensitivity, specificity, positive and negative predictive values will be calculated against the reference standard with 95% confidence intervals, and the area under the receiver operating characteristic curve will be reported.

Comparison of performance across Fitzpatrick skin types will use the chi-square test or Fisher's exact test as appropriate to the expected cell counts. The normality of continuous data will be assessed through statistical tests and graphical methods. A p-value of 0.05 will be considered statistically significant. Given the number of subgroup comparisons planned, subgroup analyses will be interpreted as exploratory and this will be stated explicitly in any report.

All statistical analysis will be performed using IBM SPSS (Statistical Packages for Social Sciences, version 26.0. Armonk, NY: IBM Corp.).

## IMPACT OF THE PROPOSED RESEARCH

The proposed research adds to current knowledge in four distinct respects.

**It addresses a documented and clinically important gap.** It is established that published dermatological artificial intelligence systems lose accuracy on darker skin, falling in one instance to the level of chance (Daneshjou et al., 2022). It is not established whether any system performs adequately on Indian patients, because no such evaluation has been published. This study would provide the first prospective evaluation of such a system in a North Indian population.

**It would create a resource that does not presently exist.** To the investigator's knowledge there is no dermatological image repository from North India, and none anywhere combining Indian clinical photography with histopathological confirmation of malignancy. Comparable repositories established in South India and in Brazil have become widely used international research resources (Pacheco et al., 2020). A repository from this department would be of value to future research in this institution and beyond, independently of the performance of the system evaluated here.

**It contributes methodologically.** Most published evaluations of such systems report accuracy alone. This study additionally evaluates whether the system's stated confidence is truthful, whether it correctly identifies the cases it should not judge, and whether its performance differs by skin type. These properties determine whether a system can be used safely, and are rarely reported.

**It is directly relevant to service delivery in this region.** Given the dermatologist-to-population ratio in Punjab and the surrounding states, an accurately validated triage aid that operates on an ordinary laptop without recurring cost would have practical application in screening camps and peripheral centres. Equally, a negative result would be of value: a finding that the system does not perform adequately on this population would be an important corrective to the current enthusiasm for such tools, and would be published as such.

## COMPETING INTERESTS

The principal investigator has no competing interest, financial, academic or of any other nature. The artificial intelligence system evaluated in this study was developed by the principal investigator without commercial funding, using publicly available image collections and a personal computer. No patent has been filed in respect of it, no commercial arrangement exists or is contemplated, and no company or external body has any involvement in the study or any interest in its outcome. Neither the supervisor nor the co-supervisor has any competing interest.

It is acknowledged that the principal investigator developed the system under evaluation and therefore has a natural interest in a favourable result. Two safeguards address this directly. First, the reference standard is recorded by the treating dermatologist before the system output is generated and while blinded to it, so the comparator cannot be influenced by the investigator. Second, the primary outcome, the sample size, and the analysis plan are fixed in this protocol before any data are collected, so the analysis cannot be adjusted after the results are seen. The investigator undertakes to report a negative result with the same completeness as a positive one.

**Similar studies in the department or at this institution in the last three years:** to the best of the knowledge of the investigator and the supervisor, no study of artificial intelligence applied to dermatological images, and no study establishing a dermatological image repository, has been conducted in the Department of Dermatology or elsewhere in Christian Medical College and Hospital, Ludhiana, within the preceding three years. This proposal does not duplicate any ongoing or recently completed work in the institution.

## FINANCIAL IMPLICATIONS

**Does the study add to the pre-existing cost of care?**

No. The study adds no investigation, no procedure, no medication and no additional hospital visit. Every clinical decision, including any decision to perform a biopsy, is taken by the treating dermatologist on clinical grounds exactly as it would have been in the absence of the study, and any resulting cost is part of routine care and is unaffected by participation. Participation adds only the time required to obtain consent and photographs, approximately five to ten minutes, at no charge to the patient. No participant will be charged any amount whatsoever in connection with this study.

**Does the study entail personal expense to the student researcher?**

The study requires modest expenditure, which the investigator will meet personally. No funding is sought from the institution or from any external body. The anticipated expenditure is set out below.

| Item | Purpose | Approximate cost (INR) |
| Smartphone (≥12 megapixel, one unit) | Standardised photography | 14,000–16,000 |
| Three-dimensionally printed spacer | Fixed camera-to-skin distance | 100–400 |
| Colour reference card and adhesive scales | Colour correction and lesion measurement | 100–200 |
| Clip-on dermatoscope | Dermatoscopic photography of suspicious lesions | 5,000–15,000 |
| Stationery, printing of proformas and information sheets | Data recording | 1,000–2,000 |
| **Total** | | **20,200–33,600** |

The computer on which the system operates is the investigator's existing personal laptop and entails no additional expenditure. No recurring cost, subscription, cloud service or licence fee is involved. Should the department already possess a dermatatoscope suitable for attachment to a smartphone, the largest single item would not be required.

## ROLE OF THE RESEARCHER

**Role of the principal investigator.** The principal investigator conceived the study, designed and constructed the artificial intelligence system under evaluation, and has written this protocol. During the study he will be responsible for obtaining informed consent, for standardised photography, for maintaining the study database and the histopathology linkage register, for conducting the statistical analysis, and for preparing the resulting manuscripts. He will additionally be responsible for the technical operation of the system, including the preparation of the analysis pipeline and the verification of its outputs.

Data recording is a necessary part of these duties but is not the primary role. The intellectual work of the project — the design of the evaluation, the construction and validation of the system, the interpretation of the results and their preparation for publication — rests with the principal investigator.

**Role of the supervisor and co-supervisor.** The supervisor and co-supervisor will provide clinical oversight, will record or adjudicate the reference-standard diagnoses, will supervise the conduct of the study within the department, and will review the analysis and the resulting manuscripts.

**Academic and learning impact on the student researcher.** The project provides the investigator with structured training in several areas not otherwise covered in the undergraduate curriculum: the design and conduct of a prospective diagnostic accuracy study; the calculation and justification of sample size; the ethical conduct of research involving photography of patients, including consent, de-identification and data protection under the Digital Personal Data Protection Act, 2023; the application of statistical methods to diagnostic data; and the preparation of research for peer review. It also provides direct clinical learning in dermatological diagnosis, since every case enrolled is examined alongside the reference-standard diagnosis recorded by a dermatologist, and in the correlation of clinical appearance with histopathological outcome in those cases proceeding to biopsy.

## PRELIMINARY RESULTS

The findings in this section were obtained on publicly available image collections during the developmental phase and are presented solely as justification for the study proposed. They do not constitute clinical validation and no clinical claim is made upon them. Final calibration of the system has not been completed and these figures are expected to be refined.

The measure used for individual conditions is the F1 score, being the harmonic mean of precision and sensitivity, which ranges from 0 to 1 and which penalises a system that achieves apparent accuracy merely by favouring common diagnoses. Sensitivity carries its usual meaning. Macro-averaged F1 denotes the mean of the per-condition F1 scores, in which every condition contributes equally irrespective of frequency; for twenty-five conditions a value of approximately 0.04 would be expected by chance alone.

### Nine-category dermatoscopic model

The system was first evaluated on a nine-category task derived from the International Skin Imaging Collaboration archive, comprising 197,137 photographs assessed by five-fold cross-validation, in which the material is divided into five parts and each part assessed in turn by a system developed on the remaining four. Per-category performance is given in Table 1.

%%TABLE_9CLASS%%

The macro-averaged F1 score was 0.660 (95% CI 0.654–0.665), compared with 0.637 for the conventional approach of retraining the whole program.

### Twenty-five condition clinical model

The system was then evaluated on twenty-five conditions selected to reflect the spectrum encountered in Indian outpatient practice, using 33,250 photographs for development and a separate set never used during development for testing. Per-condition performance on the independent test set is given in Table 2.

%%TABLE_25CLASS%%

The macro-averaged F1 score on the independent test set was 0.573 (95% CI 0.552–0.591), against 0.527 (95% CI 0.510–0.543) for the conventional retraining approach; the confidence intervals do not overlap. Top-1 agreement was 0.730 and top-3 agreement 0.887, the latter being the basis of the sample size calculation above.

Attention is drawn to an important limitation evident in Table 2. Performance is good for the common inflammatory and infectious conditions — viral infection 0.879, vitiligo 0.766, tinea 0.692 — but is substantially weaker for the malignant categories, with sensitivities of 0.408 for melanoma, 0.413 for basal cell carcinoma and 0.453 for squamous cell carcinoma. **The twenty-five-condition model is therefore not, by itself, an adequate instrument for the detection of malignancy, and it is not proposed as one.** Detection of malignancy is performed by a separate dedicated model, reported in the following section, and any lesion which that model or the treating clinician regards as suspicious proceeds to specialist assessment and biopsy in the ordinary way. This separation of function is a deliberate design decision and is one of the matters the proposed prospective study is intended to examine.

During this evaluation it was found that approximately 35% of the photographs in the test set were near-identical duplicates of photographs already used in development, arising because the same atlas images appear in more than one public collection. These were identified by comparing the numerical descriptions of the images, removed, and all figures recalculated. The figures reported here are after that correction; before it they were approximately 0.05 higher and were incorrect.

### Detection of malignancy

Performance in separating malignant from benign lesions was assessed on the Diverse Dermatology Images collection of 656 photographs, in which every diagnosis was confirmed by histopathology. Of these, 171 (26.1%) were malignant. Results are given in Table 3.

| Measure | Single program | Six programs combined | 95% CI |
| Area under the curve, all skin tones | 0.748 | 0.826 | 0.789–0.862 |
| Sensitivity at 80% specificity | — | 0.696 | — |
| Sensitivity at 90% specificity | — | 0.538 | — |
| Specificity at 90% sensitivity | — | 0.538 | — |

Combining the six programs improved the area under the curve by 0.078 (95% CI 0.049–0.107, p<0.001) over a single program. The result was stable when the entire analysis was repeated with five different random divisions of the material (mean 0.830, standard deviation 0.009).

### Performance by skin tone

Performance was then examined separately within each Fitzpatrick skin-tone group of the same collection and compared with the three previously published systems evaluated by Daneshjou et al. (2022) on identical material. Results are given in Table 4.

| Fitzpatrick group | DeepDerm | ModelDerm | HAM10000 model | Present system | 95% CI |
| I–II (lightest) | 0.61 | 0.64 | 0.72 | 0.782 | 0.701–0.856 |
| III–IV | — | — | — | 0.830 | 0.771–0.882 |
| V–VI (darkest) | 0.50 | 0.55 | 0.57 | 0.872 | 0.808–0.929 |
| Difference, lightest minus darkest | +0.11 | +0.09 | +0.15 | −0.09 | — |

Each of the three published systems performed between 0.09 and 0.15 worse on the darkest skin than on the lightest. The present system did not show this pattern. On the darkest skin, sensitivity at 80% specificity was 0.750.

This finding must be stated with precision. The confidence intervals for the lightest and darkest groups overlap, and the darkest group contains only 48 malignant lesions; the defensible conclusion is therefore that no significant difference by skin tone was demonstrated, and not that performance is superior on darker skin. Furthermore, the three published systems were evaluated without any exposure to that collection, whereas the present system was developed and assessed within it by cross-validation; the comparison therefore concerns the presence or absence of a gap between skin tones under each approach, rather than being a direct comparison of accuracy.

### Behaviour of the safety features

After calibration, the discrepancy between stated confidence and observed accuracy fell from 0.21 to 0.042. The short list of possible diagnoses contained the correct diagnosis on 89.2% of occasions against a target of 90%, and contained a mean of 3.4 diagnoses out of 25. When the system declined to offer an opinion on the 30% of cases about which it was least certain, accuracy on the remaining cases rose from 0.637 to 0.766.

### Summary of preliminary findings

The system does not, on publicly available material, show the loss of performance on darker skin that characterises previously published systems, and it states its confidence honestly and declines to answer when uncertain. Whether any of this holds for patients attending this hospital is unknown, and is the question this study is designed to answer.

## PROFORMA

### Section 1 — Consent and screening

1. The patient is identified in the outpatient queue by the treating dermatologist as having a skin lesion or eruption meeting the inclusion criteria.
2. The Patient Information Sheet is provided and explained in the language the patient understands. The patient is given the opportunity to ask questions.
3. Written informed consent is obtained on the Informed Consent Form before any photograph is taken. No photograph is taken before consent.
4. A study identification number is allocated in sequence. From this point forward the patient is referred to only by this number.

### Section 2 — Data recording format

**PATIENT INFORMATION**

Study ID: ................................................  Date: ................................................

Age: ................  Gender: Male / Female / Other

Fitzpatrick skin type: I / II / III / IV / V / VI

Occupation: ................................................  Outdoor exposure: Yes / No

Comorbidities (if any): ................................................

Immunosuppression: Yes / No    Diabetes mellitus: Yes / No

**LESION INFORMATION**

Anatomical site: Head-neck / Anterior trunk / Posterior trunk / Upper limb / Lower limb / Palms-soles / Nail / Scalp

Laterality: Right / Left / Midline / Bilateral

Duration: ................ (days / weeks / months / years)

Symptoms: Itch / Pain / Burning / Bleeding / None

Evolution: New / Increasing in size / Changing in colour / Static / Resolving

Previous treatment taken: ................................................

Relevant family history: ................................................

**CLINICAL ASSESSMENT (recorded before the system is run)**

Provisional clinical diagnosis: ................................................

Differential diagnoses: 1. ................ 2. ................ 3. ................

Clinician's confidence: Low / Moderate / High

Biopsy indicated: Yes / No    If yes, histopathology number: ................................

**PHOTOGRAPHS OBTAINED**

Regional: Yes / No    Close-up: Yes / No    Dermatoscopic: Yes / No / Not applicable

Device used: ................................................

**SYSTEM OUTPUT (recorded after the clinical assessment has been entered)**

Ranked diagnoses: 1. ................ 2. ................ 3. ................

Estimated probability of malignancy: ................

System declined to offer an opinion: Yes / No

**FINAL DIAGNOSIS**

Final diagnosis: ................................................

Basis: Histopathology / Dermoscopy-supported clinical / Clinical / Provisional

### Section 3 — Photographic procedure

1. Examination room lighting is kept consistent between patients; mixed daylight and tube light is avoided where possible.
2. The lesion is exposed and cleaned of applied medication, powder or cosmetic. Jewellery and clothing that would identify the patient are removed from the field.
3. The smartphone is fitted with the printed spacer. The main camera lens is used; the ultra-wide lens and digital zoom are not used.
4. The camera is set to its default photograph mode. High-resolution modes, filters and beautification settings are disabled. Flash is not used for the regional photograph.
5. A regional photograph is taken first, demonstrating the anatomical site with sufficient surrounding skin to establish context.
6. The colour reference card and adhesive scale are placed adjacent to, but not overlapping, the lesion.
7. The spacer is placed in gentle contact with the skin, taking care not to blanch it, since pressure alters lesion colour. A close-up photograph is taken with the lesion occupying at least half of the field.
8. For pigmented, ulcerated or otherwise suspicious lesions a dermatoscopic photograph is taken in addition.
9. Each photograph is inspected immediately and retaken if out of focus, if there is glare over the lesion, if a shadow crosses the lesion, or if any identifying feature is visible.
10. In darker-skinned patients the automatic exposure of the camera frequently under-exposes the lesion. The preview is checked and exposure increased manually where the lesion appears unnaturally dark.
11. Photographs are transferred the same day to the designated institutional computer and deleted from the device. Photographs are not transmitted through any messaging application, are not stored on any personal device, and are not uploaded to any external service.

### Section 4 — Data handling and confidentiality

1. Photographs are stored under the study identification number only. No name, hospital registration number, telephone number or address is stored with the image or in any free-text field.
2. The document linking study identification numbers to patient identity is maintained separately in a password-protected file accessible only to the principal investigator and the supervisor, and is required solely for retrieval of histopathology reports.
3. Photographs are stored on an institutional computer with encryption at rest, physically located within the institution. No image is stored on a personal cloud service.
4. Faces are not photographed. Any incidental identifying feature such as a tattoo is excluded from the field or masked before storage.
5. All handling of data will comply with the Digital Personal Data Protection Act, 2023, and with the guidance of the Institutional Ethics Committee.
6. Data will be retained for the period specified by the Institutional Ethics Committee, after which photographs will be destroyed unless separate approval for continued retention has been obtained.
7. Any future release of the repository for research use beyond this institution will be the subject of a separate application to the Institutional Ethics Committee and will not proceed under the present approval.

### Section 5 — Histopathology linkage

1. Where a biopsy is performed on clinical indication, the histopathology number is recorded on the proforma at the time of the visit.
2. The case is entered in a pending register.
3. When the report becomes available, the histopathological diagnosis is entered against the study identification number and the evidence grade amended to histopathological.
4. Where a report cannot be retrieved, the case is retained with its clinical diagnosis and the evidence grade left unchanged. Such cases are reported separately.
5. Where the histopathological diagnosis differs from the clinical diagnosis, both are retained; histopathology becomes the reference standard and the discordance is recorded.

### Section 6 — Labelling for the repository component

1. Every photograph in the repository will carry a diagnosis recorded by a qualified dermatologist.
2. Where feasible, a second dermatologist will record a diagnosis independently and blinded to the first, and agreement will be reported using Cohen's kappa.
3. Where the two differ, a third senior dermatologist will adjudicate and the adjudicated diagnosis recorded as such.
4. No dermatologist recording a diagnosis will be shown the output of the system, since knowledge of it would bias the reference standard and destroy the independence of the repository.
5. Photographs of inadequate technical quality will be excluded and the reason recorded.

## INFORMED CONSENT FORM

Study ID: ................................................  Date: ................................................

I, ................................................ do hereby consent to participate in the study named, "PROSPECTIVE CLINICAL VALIDATION OF AN ARTIFICIAL INTELLIGENCE SYSTEM FOR THE ASSESSMENT OF SKIN DISEASE AND SKIN CANCER IN INDIAN PATIENTS".

- I understand that I can freely withdraw from this study at any time.
- I have had a chance to ask questions and all of my questions have been answered to my satisfaction.
- I have been given and I understand the information concerning the nature, purpose and duration of the study as well as the procedures involved, including any known or expected inconvenience.
- I understand that photographs will be taken of the affected area of my skin, that my face will not be photographed, and that my name will not be stored with the photographs.
- I understand that the computer program being studied will not be used to decide my treatment, and that my treatment will be exactly the same whether or not I take part.
- I understand that the photographs may be kept for future research and may be shared with other researchers in a form in which I cannot be identified, and that any such sharing requires separate approval from the Institutional Ethics Committee.
- My medical data remains confidential and I authorize that they be looked at only by persons involved in the research.
- By signing this form, I give my free and informed consent to take part in this study.

(Signature / thumb impression of the participant)

Name of the participant: ................................................

Phone no: ................................................

Date: ................................................    Place: ................................................

Signature of investigator: ................................................

Date: ................................................

## सूचित सहमति प्रपत्र

अध्ययन क्रमांक: ................................................  दिनांक: ................................................

मैं, ................................................ इस अध्ययन में भाग लेने के लिए सहमति देता / देती हूँ, जिसका नाम है "भारतीय रोगियों में त्वचा रोग एवं त्वचा कैंसर के आकलन हेतु कृत्रिम बुद्धिमत्ता प्रणाली का संभाव्य नैदानिक सत्यापन"।

- मैं समझता / समझती हूँ कि मैं किसी भी समय स्वतंत्र रूप से इस अध्ययन से हट सकता / सकती हूँ।
- मुझे प्रश्न पूछने का अवसर मिला है और मेरे सभी प्रश्नों के उत्तर मेरी संतुष्टि के अनुसार दिए गए हैं।
- मुझे अध्ययन की प्रकृति, उद्देश्य एवं अवधि तथा इसमें सम्मिलित प्रक्रियाओं की जानकारी दी गई है और मैं उसे समझता / समझती हूँ, जिसमें कोई भी ज्ञात या संभावित असुविधा भी सम्मिलित है।
- मैं समझता / समझती हूँ कि मेरी त्वचा के प्रभावित भाग की तस्वीरें ली जाएँगी, मेरे चेहरे की तस्वीर नहीं ली जाएगी, तथा तस्वीरों के साथ मेरा नाम संग्रहीत नहीं किया जाएगा।
- मैं समझता / समझती हूँ कि जिस कंप्यूटर प्रोग्राम का अध्ययन किया जा रहा है, उसका उपयोग मेरे उपचार का निर्णय लेने के लिए नहीं किया जाएगा, और मेरा उपचार वैसा ही रहेगा चाहे मैं भाग लूँ अथवा न लूँ।
- मैं समझता / समझती हूँ कि ये तस्वीरें भविष्य के शोध हेतु रखी जा सकती हैं तथा अन्य शोधकर्ताओं के साथ ऐसे रूप में साझा की जा सकती हैं जिसमें मेरी पहचान न हो सके, और ऐसी किसी भी साझेदारी के लिए संस्थागत आचार समिति से पृथक अनुमति आवश्यक होगी।
- मेरा चिकित्सीय विवरण गोपनीय रहेगा और मैं अधिकृत करता / करती हूँ कि इसे केवल शोध से जुड़े व्यक्ति ही देखें।
- इस प्रपत्र पर हस्ताक्षर करके मैं इस अध्ययन में भाग लेने हेतु अपनी स्वतंत्र एवं सूचित सहमति देता / देती हूँ।

(प्रतिभागी के हस्ताक्षर / अंगूठे का निशान)

प्रतिभागी का नाम: ................................................

दूरभाष क्रमांक: ................................................

दिनांक: ................................................    स्थान: ................................................

अन्वेषक के हस्ताक्षर: ................................................

दिनांक: ................................................

## ਸੂਚਿਤ ਸਹਿਮਤੀ ਫਾਰਮ

ਅਧਿਐਨ ਨੰਬਰ: ................................................  ਮਿਤੀ: ................................................

ਮੈਂ, ................................................ ਇਸ ਅਧਿਐਨ ਵਿੱਚ ਭਾਗ ਲੈਣ ਲਈ ਸਹਿਮਤੀ ਦਿੰਦਾ / ਦਿੰਦੀ ਹਾਂ, ਜਿਸਦਾ ਨਾਮ ਹੈ "ਭਾਰਤੀ ਮਰੀਜ਼ਾਂ ਵਿੱਚ ਚਮੜੀ ਦੇ ਰੋਗਾਂ ਅਤੇ ਚਮੜੀ ਦੇ ਕੈਂਸਰ ਦੇ ਮੁਲਾਂਕਣ ਲਈ ਆਰਟੀਫੀਸ਼ੀਅਲ ਇੰਟੈਲੀਜੈਂਸ ਪ੍ਰਣਾਲੀ ਦੀ ਸੰਭਾਵੀ ਕਲੀਨਿਕਲ ਪੁਸ਼ਟੀ"।

- ਮੈਂ ਸਮਝਦਾ / ਸਮਝਦੀ ਹਾਂ ਕਿ ਮੈਂ ਕਿਸੇ ਵੀ ਸਮੇਂ ਆਜ਼ਾਦ ਤੌਰ 'ਤੇ ਇਸ ਅਧਿਐਨ ਤੋਂ ਹਟ ਸਕਦਾ / ਸਕਦੀ ਹਾਂ।
- ਮੈਨੂੰ ਸਵਾਲ ਪੁੱਛਣ ਦਾ ਮੌਕਾ ਮਿਲਿਆ ਹੈ ਅਤੇ ਮੇਰੇ ਸਾਰੇ ਸਵਾਲਾਂ ਦੇ ਜਵਾਬ ਮੇਰੀ ਤਸੱਲੀ ਅਨੁਸਾਰ ਦਿੱਤੇ ਗਏ ਹਨ।
- ਮੈਨੂੰ ਅਧਿਐਨ ਦੀ ਪ੍ਰਕਿਰਤੀ, ਉਦੇਸ਼ ਅਤੇ ਮਿਆਦ ਬਾਰੇ ਅਤੇ ਇਸ ਵਿੱਚ ਸ਼ਾਮਲ ਪ੍ਰਕਿਰਿਆਵਾਂ ਬਾਰੇ ਜਾਣਕਾਰੀ ਦਿੱਤੀ ਗਈ ਹੈ ਅਤੇ ਮੈਂ ਉਸਨੂੰ ਸਮਝਦਾ / ਸਮਝਦੀ ਹਾਂ, ਜਿਸ ਵਿੱਚ ਕੋਈ ਵੀ ਜਾਣੀ ਜਾਂ ਸੰਭਾਵੀ ਅਸੁਵਿਧਾ ਵੀ ਸ਼ਾਮਲ ਹੈ।
- ਮੈਂ ਸਮਝਦਾ / ਸਮਝਦੀ ਹਾਂ ਕਿ ਮੇਰੀ ਚਮੜੀ ਦੇ ਪ੍ਰਭਾਵਿਤ ਹਿੱਸੇ ਦੀਆਂ ਤਸਵੀਰਾਂ ਲਈਆਂ ਜਾਣਗੀਆਂ, ਮੇਰੇ ਚਿਹਰੇ ਦੀ ਤਸਵੀਰ ਨਹੀਂ ਲਈ ਜਾਵੇਗੀ, ਅਤੇ ਤਸਵੀਰਾਂ ਨਾਲ ਮੇਰਾ ਨਾਮ ਸਟੋਰ ਨਹੀਂ ਕੀਤਾ ਜਾਵੇਗਾ।
- ਮੈਂ ਸਮਝਦਾ / ਸਮਝਦੀ ਹਾਂ ਕਿ ਜਿਸ ਕੰਪਿਊਟਰ ਪ੍ਰੋਗਰਾਮ ਦਾ ਅਧਿਐਨ ਕੀਤਾ ਜਾ ਰਿਹਾ ਹੈ, ਉਸਦੀ ਵਰਤੋਂ ਮੇਰੇ ਇਲਾਜ ਦਾ ਫ਼ੈਸਲਾ ਕਰਨ ਲਈ ਨਹੀਂ ਕੀਤੀ ਜਾਵੇਗੀ, ਅਤੇ ਮੇਰਾ ਇਲਾਜ ਉਹੀ ਰਹੇਗਾ ਭਾਵੇਂ ਮੈਂ ਭਾਗ ਲਵਾਂ ਜਾਂ ਨਾ ਲਵਾਂ।
- ਮੈਂ ਸਮਝਦਾ / ਸਮਝਦੀ ਹਾਂ ਕਿ ਇਹ ਤਸਵੀਰਾਂ ਭਵਿੱਖ ਦੀ ਖੋਜ ਲਈ ਰੱਖੀਆਂ ਜਾ ਸਕਦੀਆਂ ਹਨ ਅਤੇ ਹੋਰ ਖੋਜਕਰਤਾਵਾਂ ਨਾਲ ਅਜਿਹੇ ਰੂਪ ਵਿੱਚ ਸਾਂਝੀਆਂ ਕੀਤੀਆਂ ਜਾ ਸਕਦੀਆਂ ਹਨ ਜਿਸ ਵਿੱਚ ਮੇਰੀ ਪਛਾਣ ਨਾ ਹੋ ਸਕੇ, ਅਤੇ ਅਜਿਹੀ ਕਿਸੇ ਵੀ ਸਾਂਝ ਲਈ ਸੰਸਥਾਗਤ ਨੈਤਿਕਤਾ ਕਮੇਟੀ ਤੋਂ ਵੱਖਰੀ ਮਨਜ਼ੂਰੀ ਜ਼ਰੂਰੀ ਹੋਵੇਗੀ।
- ਮੇਰਾ ਮੈਡੀਕਲ ਵੇਰਵਾ ਗੁਪਤ ਰਹੇਗਾ ਅਤੇ ਮੈਂ ਅਧਿਕਾਰ ਦਿੰਦਾ / ਦਿੰਦੀ ਹਾਂ ਕਿ ਇਸਨੂੰ ਸਿਰਫ਼ ਖੋਜ ਨਾਲ ਜੁੜੇ ਵਿਅਕਤੀ ਹੀ ਵੇਖਣ।
- ਇਸ ਫਾਰਮ 'ਤੇ ਦਸਤਖ਼ਤ ਕਰਕੇ ਮੈਂ ਇਸ ਅਧਿਐਨ ਵਿੱਚ ਭਾਗ ਲੈਣ ਲਈ ਆਪਣੀ ਆਜ਼ਾਦ ਅਤੇ ਸੂਚਿਤ ਸਹਿਮਤੀ ਦਿੰਦਾ / ਦਿੰਦੀ ਹਾਂ।

(ਭਾਗੀਦਾਰ ਦੇ ਦਸਤਖ਼ਤ / ਅੰਗੂਠੇ ਦਾ ਨਿਸ਼ਾਨ)

ਭਾਗੀਦਾਰ ਦਾ ਨਾਮ: ................................................

ਫ਼ੋਨ ਨੰਬਰ: ................................................

ਮਿਤੀ: ................................................    ਸਥਾਨ: ................................................

ਅਨਵੇਸ਼ਕ ਦੇ ਦਸਤਖ਼ਤ: ................................................

ਮਿਤੀ: ................................................

## PATIENT INFORMATION SHEET

For "PROSPECTIVE CLINICAL VALIDATION OF AN ARTIFICIAL INTELLIGENCE SYSTEM FOR THE ASSESSMENT OF SKIN DISEASE AND SKIN CANCER IN INDIAN PATIENTS"

### Introduction

The purpose of this form is to provide you with information, so you can decide whether to participate in this study. Any questions you may have will be answered by the researcher or by the other contact persons provided below. Once you are familiar with the information on this sheet and have asked any questions you may have, you can decide whether or not to participate. If you agree, you will be asked to fill in the consent form for this study.

**Name and contact details of researcher:** Ujjwal Thakran, MBBS student 2022 batch, Christian Medical College and Hospital, Ludhiana. Mob: 9999209888

**What type of study is this?** — Research project.

**Is there any funding involved?** — No.

**What is the purpose of the research project?** — To find out whether a computer program that looks at photographs of skin conditions gives the same answer as a skin specialist, in patients attending the skin outpatient department of this hospital. Almost all such programs have been made using photographs of light-skinned patients from other countries, and it is not known whether they work properly on Indian skin. We also wish to build a collection of skin photographs from this hospital, so that this and future research can be checked on patients from our own region.

**Why have I been chosen to participate?** — Because you are eighteen years of age or above and have come to the skin outpatient department with a skin problem.

**Do I have to take part?** — Your participation is voluntary. It is up to you to decide whether you want to participate or not. If you decide to participate, you are still free to withdraw at any time and without giving a reason.

**What happens if I take part?** — Photographs will be taken of the affected area of your skin using a mobile phone camera, and some questions will be asked about your skin problem. This will take about five to ten minutes. Nothing will be injected, and no test will be done on you for the purposes of this study. If your doctor decides that a small piece of skin needs to be taken for testing, that decision is part of your normal treatment and would have been made whether or not you took part.

**Will my treatment be withheld if I refuse to take part?** — No. Your treatment will be exactly the same whether or not you take part.

**Will the computer program decide my treatment?** — No. The program is only being tested. Your doctor will decide your diagnosis and treatment in the usual way, and will do so before the program is used. The result given by the program will not be shown to your doctor before that, will not be written in your file, and will not be used to decide anything about your care.

**What are the benefits and risks?** — There is no direct benefit to you. There is no physical risk, as only photographs will be taken. The main risk is to your privacy, and the steps taken to protect it are described below.

**Will my information be kept confidential?** — Yes. Your photographs will be stored under a number, not your name. Your face will not be photographed. Your name, hospital number and phone number will not be stored with the photographs.

**Will my identity be kept anonymous?** — Yes. If the photographs are used for research in future or shared with other researchers, they will be shared only in a form in which you cannot be identified, and only after separate permission from the Institutional Ethics Committee.

**Will I have to pay anything?** — No. There is no cost to you of any kind for taking part in this study.

**Contact for further information** — Mr. Ujjwal Thakran, MBBS student 2022 batch, Christian Medical College and Hospital, Ludhiana. Mob: 9999209888

**For complaints contact** — Dr. Jitender Mahindroo, Chairman, Institutional Ethics Committee, Christian Medical College and Hospital, Ludhiana.

## रोगी सूचना पत्र

"भारतीय रोगियों में त्वचा रोग एवं त्वचा कैंसर के आकलन हेतु कृत्रिम बुद्धिमत्ता प्रणाली का संभाव्य नैदानिक सत्यापन" अध्ययन हेतु

### परिचय

इस पत्र का उद्देश्य आपको जानकारी देना है, ताकि आप निर्णय ले सकें कि आप इस अध्ययन में भाग लेना चाहते हैं अथवा नहीं। आपके किसी भी प्रश्न का उत्तर शोधकर्ता अथवा नीचे दिए गए संपर्क व्यक्तियों द्वारा दिया जाएगा। इस पत्र की जानकारी समझ लेने और अपने प्रश्न पूछ लेने के पश्चात आप निर्णय ले सकते हैं। यदि आप सहमत हों, तो आपसे इस अध्ययन का सहमति प्रपत्र भरने को कहा जाएगा।

**शोधकर्ता का नाम एवं संपर्क:** उज्ज्वल ठाकरान, एमबीबीएस विद्यार्थी, 2022 बैच, क्रिश्चियन मेडिकल कॉलेज एवं अस्पताल, लुधियाना। मोबाइल: 9999209888

**यह किस प्रकार का अध्ययन है?** — शोध परियोजना।

**क्या इसमें कोई वित्तीय सहायता सम्मिलित है?** — नहीं।

**इस शोध का उद्देश्य क्या है?** — यह जानना कि त्वचा रोगों की तस्वीरें देखने वाला एक कंप्यूटर प्रोग्राम इस अस्पताल की त्वचा रोग बाह्य रोगी विभाग में आने वाले रोगियों में त्वचा विशेषज्ञ के समान उत्तर देता है अथवा नहीं। ऐसे लगभग सभी प्रोग्राम अन्य देशों के गोरी त्वचा वाले रोगियों की तस्वीरों से बनाए गए हैं, और यह ज्ञात नहीं है कि वे भारतीय त्वचा पर ठीक से कार्य करते हैं या नहीं। हम इस अस्पताल से त्वचा की तस्वीरों का एक संग्रह भी बनाना चाहते हैं, ताकि यह तथा भविष्य का शोध हमारे अपने क्षेत्र के रोगियों पर जाँचा जा सके।

**मुझे इस शोध हेतु क्यों चुना गया है?** — क्योंकि आपकी आयु अठारह वर्ष या उससे अधिक है और आप त्वचा रोग बाह्य रोगी विभाग में त्वचा की समस्या लेकर आए हैं।

**क्या मेरा भाग लेना आवश्यक है?** — आपका भाग लेना पूर्णतः स्वैच्छिक है। भाग लेना अथवा न लेना आपका निर्णय है। यदि आप भाग लेने का निर्णय लेते हैं, तब भी आप बिना कोई कारण बताए किसी भी समय हट सकते हैं।

**यदि मैं भाग लूँ तो क्या होगा?** — मोबाइल फोन के कैमरे से आपकी त्वचा के प्रभावित भाग की तस्वीरें ली जाएँगी और आपकी त्वचा की समस्या के विषय में कुछ प्रश्न पूछे जाएँगे। इसमें लगभग पाँच से दस मिनट लगेंगे। कोई इंजेक्शन नहीं लगाया जाएगा और इस अध्ययन के लिए आप पर कोई जाँच नहीं की जाएगी। यदि आपके चिकित्सक यह निर्णय लेते हैं कि जाँच हेतु त्वचा का छोटा टुकड़ा लेना आवश्यक है, तो वह निर्णय आपके सामान्य उपचार का भाग है और वह इस अध्ययन में भाग लेने अथवा न लेने से अप्रभावित रहता।

**यदि मैं भाग लेने से मना कर दूँ तो क्या मेरा उपचार रोक दिया जाएगा?** — नहीं। आपका उपचार वैसा ही रहेगा चाहे आप भाग लें अथवा न लें।

**क्या कंप्यूटर प्रोग्राम मेरे उपचार का निर्णय करेगा?** — नहीं। प्रोग्राम की केवल जाँच की जा रही है। आपके चिकित्सक सामान्य रूप से ही आपका निदान एवं उपचार निर्धारित करेंगे, और वे ऐसा प्रोग्राम चलाने से पूर्व करेंगे। प्रोग्राम का परिणाम उससे पहले आपके चिकित्सक को नहीं दिखाया जाएगा, आपकी फाइल में नहीं लिखा जाएगा, और आपकी देखभाल के विषय में किसी निर्णय हेतु प्रयोग नहीं किया जाएगा।

**इसमें क्या लाभ एवं जोखिम हैं?** — आपको कोई प्रत्यक्ष लाभ नहीं है। कोई शारीरिक जोखिम नहीं है, क्योंकि केवल तस्वीरें ली जाएँगी। मुख्य जोखिम आपकी निजता से संबंधित है, और उसकी सुरक्षा हेतु किए गए उपाय नीचे दिए गए हैं।

**क्या मेरी जानकारी गोपनीय रखी जाएगी?** — हाँ। आपकी तस्वीरें आपके नाम से नहीं, एक क्रमांक से संग्रहीत की जाएँगी। आपके चेहरे की तस्वीर नहीं ली जाएगी। आपका नाम, अस्पताल क्रमांक एवं दूरभाष क्रमांक तस्वीरों के साथ संग्रहीत नहीं किया जाएगा।

**क्या मेरी पहचान गुप्त रखी जाएगी?** — हाँ। यदि भविष्य में इन तस्वीरों का शोध हेतु प्रयोग हो अथवा अन्य शोधकर्ताओं के साथ साझा किया जाए, तो वे केवल ऐसे रूप में साझा की जाएँगी जिसमें आपकी पहचान न हो सके, और संस्थागत आचार समिति से पृथक अनुमति लेने के पश्चात ही।

**क्या मुझे कुछ भुगतान करना होगा?** — नहीं। इस अध्ययन में भाग लेने हेतु आपको किसी भी प्रकार का कोई शुल्क नहीं देना होगा।

**अधिक जानकारी हेतु संपर्क** — श्री उज्ज्वल ठाकरान, एमबीबीएस विद्यार्थी 2022 बैच, क्रिश्चियन मेडिकल कॉलेज एवं अस्पताल, लुधियाना। मोबाइल: 9999209888

**शिकायत हेतु संपर्क** — डॉ. जितेंद्र महिंद्रू, अध्यक्ष, संस्थागत आचार समिति, क्रिश्चियन मेडिकल कॉलेज एवं अस्पताल, लुधियाना।

## ਮਰੀਜ਼ ਜਾਣਕਾਰੀ ਪੱਤਰ

"ਭਾਰਤੀ ਮਰੀਜ਼ਾਂ ਵਿੱਚ ਚਮੜੀ ਦੇ ਰੋਗਾਂ ਅਤੇ ਚਮੜੀ ਦੇ ਕੈਂਸਰ ਦੇ ਮੁਲਾਂਕਣ ਲਈ ਆਰਟੀਫੀਸ਼ੀਅਲ ਇੰਟੈਲੀਜੈਂਸ ਪ੍ਰਣਾਲੀ ਦੀ ਸੰਭਾਵੀ ਕਲੀਨਿਕਲ ਪੁਸ਼ਟੀ" ਅਧਿਐਨ ਲਈ

### ਜਾਣ-ਪਛਾਣ

ਇਸ ਪੱਤਰ ਦਾ ਉਦੇਸ਼ ਤੁਹਾਨੂੰ ਜਾਣਕਾਰੀ ਦੇਣਾ ਹੈ, ਤਾਂ ਜੋ ਤੁਸੀਂ ਫ਼ੈਸਲਾ ਕਰ ਸਕੋ ਕਿ ਤੁਸੀਂ ਇਸ ਅਧਿਐਨ ਵਿੱਚ ਭਾਗ ਲੈਣਾ ਚਾਹੁੰਦੇ ਹੋ ਜਾਂ ਨਹੀਂ। ਤੁਹਾਡੇ ਕਿਸੇ ਵੀ ਸਵਾਲ ਦਾ ਜਵਾਬ ਖੋਜਕਰਤਾ ਜਾਂ ਹੇਠਾਂ ਦਿੱਤੇ ਸੰਪਰਕ ਵਿਅਕਤੀਆਂ ਦੁਆਰਾ ਦਿੱਤਾ ਜਾਵੇਗਾ। ਇਸ ਪੱਤਰ ਦੀ ਜਾਣਕਾਰੀ ਸਮਝ ਲੈਣ ਅਤੇ ਆਪਣੇ ਸਵਾਲ ਪੁੱਛ ਲੈਣ ਤੋਂ ਬਾਅਦ ਤੁਸੀਂ ਫ਼ੈਸਲਾ ਕਰ ਸਕਦੇ ਹੋ। ਜੇ ਤੁਸੀਂ ਸਹਿਮਤ ਹੋ, ਤਾਂ ਤੁਹਾਨੂੰ ਇਸ ਅਧਿਐਨ ਦਾ ਸਹਿਮਤੀ ਫਾਰਮ ਭਰਨ ਲਈ ਕਿਹਾ ਜਾਵੇਗਾ।

**ਖੋਜਕਰਤਾ ਦਾ ਨਾਮ ਅਤੇ ਸੰਪਰਕ:** ਉੱਜਵਲ ਠਾਕਰਾਨ, ਐਮਬੀਬੀਐਸ ਵਿਦਿਆਰਥੀ, 2022 ਬੈਚ, ਕ੍ਰਿਸ਼ਚੀਅਨ ਮੈਡੀਕਲ ਕਾਲਜ ਅਤੇ ਹਸਪਤਾਲ, ਲੁਧਿਆਣਾ। ਮੋਬਾਈਲ: 9999209888

**ਇਹ ਕਿਸ ਪ੍ਰਕਾਰ ਦਾ ਅਧਿਐਨ ਹੈ?** — ਖੋਜ ਪ੍ਰੋਜੈਕਟ।

**ਕੀ ਇਸ ਵਿੱਚ ਕੋਈ ਵਿੱਤੀ ਸਹਾਇਤਾ ਸ਼ਾਮਲ ਹੈ?** — ਨਹੀਂ।

**ਇਸ ਖੋਜ ਦਾ ਉਦੇਸ਼ ਕੀ ਹੈ?** — ਇਹ ਜਾਣਨਾ ਕਿ ਚਮੜੀ ਦੇ ਰੋਗਾਂ ਦੀਆਂ ਤਸਵੀਰਾਂ ਵੇਖਣ ਵਾਲਾ ਇੱਕ ਕੰਪਿਊਟਰ ਪ੍ਰੋਗਰਾਮ ਇਸ ਹਸਪਤਾਲ ਦੇ ਚਮੜੀ ਰੋਗ ਬਾਹਰੀ ਮਰੀਜ਼ ਵਿਭਾਗ ਵਿੱਚ ਆਉਣ ਵਾਲੇ ਮਰੀਜ਼ਾਂ ਵਿੱਚ ਚਮੜੀ ਦੇ ਮਾਹਿਰ ਵਾਂਗ ਜਵਾਬ ਦਿੰਦਾ ਹੈ ਜਾਂ ਨਹੀਂ। ਅਜਿਹੇ ਲਗਭਗ ਸਾਰੇ ਪ੍ਰੋਗਰਾਮ ਹੋਰ ਦੇਸ਼ਾਂ ਦੇ ਗੋਰੀ ਚਮੜੀ ਵਾਲੇ ਮਰੀਜ਼ਾਂ ਦੀਆਂ ਤਸਵੀਰਾਂ ਤੋਂ ਬਣਾਏ ਗਏ ਹਨ, ਅਤੇ ਇਹ ਪਤਾ ਨਹੀਂ ਕਿ ਉਹ ਭਾਰਤੀ ਚਮੜੀ 'ਤੇ ਠੀਕ ਕੰਮ ਕਰਦੇ ਹਨ ਜਾਂ ਨਹੀਂ। ਅਸੀਂ ਇਸ ਹਸਪਤਾਲ ਤੋਂ ਚਮੜੀ ਦੀਆਂ ਤਸਵੀਰਾਂ ਦਾ ਸੰਗ੍ਰਹਿ ਵੀ ਬਣਾਉਣਾ ਚਾਹੁੰਦੇ ਹਾਂ, ਤਾਂ ਜੋ ਇਹ ਅਤੇ ਭਵਿੱਖ ਦੀ ਖੋਜ ਸਾਡੇ ਆਪਣੇ ਖੇਤਰ ਦੇ ਮਰੀਜ਼ਾਂ 'ਤੇ ਜਾਂਚੀ ਜਾ ਸਕੇ।

**ਮੈਨੂੰ ਇਸ ਖੋਜ ਲਈ ਕਿਉਂ ਚੁਣਿਆ ਗਿਆ ਹੈ?** — ਕਿਉਂਕਿ ਤੁਹਾਡੀ ਉਮਰ ਅਠਾਰਾਂ ਸਾਲ ਜਾਂ ਵੱਧ ਹੈ ਅਤੇ ਤੁਸੀਂ ਚਮੜੀ ਰੋਗ ਬਾਹਰੀ ਮਰੀਜ਼ ਵਿਭਾਗ ਵਿੱਚ ਚਮੜੀ ਦੀ ਸਮੱਸਿਆ ਲੈ ਕੇ ਆਏ ਹੋ।

**ਕੀ ਮੇਰਾ ਭਾਗ ਲੈਣਾ ਜ਼ਰੂਰੀ ਹੈ?** — ਤੁਹਾਡਾ ਭਾਗ ਲੈਣਾ ਪੂਰੀ ਤਰ੍ਹਾਂ ਸਵੈ-ਇੱਛਤ ਹੈ। ਭਾਗ ਲੈਣਾ ਜਾਂ ਨਾ ਲੈਣਾ ਤੁਹਾਡਾ ਫ਼ੈਸਲਾ ਹੈ। ਜੇ ਤੁਸੀਂ ਭਾਗ ਲੈਣ ਦਾ ਫ਼ੈਸਲਾ ਕਰਦੇ ਹੋ, ਤਾਂ ਵੀ ਤੁਸੀਂ ਬਿਨਾਂ ਕੋਈ ਕਾਰਨ ਦੱਸੇ ਕਿਸੇ ਵੀ ਸਮੇਂ ਹਟ ਸਕਦੇ ਹੋ।

**ਜੇ ਮੈਂ ਭਾਗ ਲਵਾਂ ਤਾਂ ਕੀ ਹੋਵੇਗਾ?** — ਮੋਬਾਈਲ ਫ਼ੋਨ ਦੇ ਕੈਮਰੇ ਨਾਲ ਤੁਹਾਡੀ ਚਮੜੀ ਦੇ ਪ੍ਰਭਾਵਿਤ ਹਿੱਸੇ ਦੀਆਂ ਤਸਵੀਰਾਂ ਲਈਆਂ ਜਾਣਗੀਆਂ ਅਤੇ ਤੁਹਾਡੀ ਚਮੜੀ ਦੀ ਸਮੱਸਿਆ ਬਾਰੇ ਕੁਝ ਸਵਾਲ ਪੁੱਛੇ ਜਾਣਗੇ। ਇਸ ਵਿੱਚ ਲਗਭਗ ਪੰਜ ਤੋਂ ਦਸ ਮਿੰਟ ਲੱਗਣਗੇ। ਕੋਈ ਟੀਕਾ ਨਹੀਂ ਲਗਾਇਆ ਜਾਵੇਗਾ ਅਤੇ ਇਸ ਅਧਿਐਨ ਲਈ ਤੁਹਾਡੇ 'ਤੇ ਕੋਈ ਜਾਂਚ ਨਹੀਂ ਕੀਤੀ ਜਾਵੇਗੀ। ਜੇ ਤੁਹਾਡੇ ਡਾਕਟਰ ਇਹ ਫ਼ੈਸਲਾ ਕਰਦੇ ਹਨ ਕਿ ਜਾਂਚ ਲਈ ਚਮੜੀ ਦਾ ਛੋਟਾ ਟੁਕੜਾ ਲੈਣਾ ਜ਼ਰੂਰੀ ਹੈ, ਤਾਂ ਉਹ ਫ਼ੈਸਲਾ ਤੁਹਾਡੇ ਆਮ ਇਲਾਜ ਦਾ ਹਿੱਸਾ ਹੈ।

**ਜੇ ਮੈਂ ਭਾਗ ਲੈਣ ਤੋਂ ਇਨਕਾਰ ਕਰਾਂ ਤਾਂ ਕੀ ਮੇਰਾ ਇਲਾਜ ਰੋਕ ਦਿੱਤਾ ਜਾਵੇਗਾ?** — ਨਹੀਂ। ਤੁਹਾਡਾ ਇਲਾਜ ਉਹੀ ਰਹੇਗਾ ਭਾਵੇਂ ਤੁਸੀਂ ਭਾਗ ਲਵੋ ਜਾਂ ਨਾ ਲਵੋ।

**ਕੀ ਕੰਪਿਊਟਰ ਪ੍ਰੋਗਰਾਮ ਮੇਰੇ ਇਲਾਜ ਦਾ ਫ਼ੈਸਲਾ ਕਰੇਗਾ?** — ਨਹੀਂ। ਪ੍ਰੋਗਰਾਮ ਦੀ ਸਿਰਫ਼ ਜਾਂਚ ਕੀਤੀ ਜਾ ਰਹੀ ਹੈ। ਤੁਹਾਡੇ ਡਾਕਟਰ ਆਮ ਤਰੀਕੇ ਨਾਲ ਹੀ ਤੁਹਾਡਾ ਨਿਦਾਨ ਅਤੇ ਇਲਾਜ ਤੈਅ ਕਰਨਗੇ, ਅਤੇ ਉਹ ਪ੍ਰੋਗਰਾਮ ਚਲਾਉਣ ਤੋਂ ਪਹਿਲਾਂ ਅਜਿਹਾ ਕਰਨਗੇ। ਪ੍ਰੋਗਰਾਮ ਦਾ ਨਤੀਜਾ ਉਸ ਤੋਂ ਪਹਿਲਾਂ ਤੁਹਾਡੇ ਡਾਕਟਰ ਨੂੰ ਨਹੀਂ ਵਿਖਾਇਆ ਜਾਵੇਗਾ, ਤੁਹਾਡੀ ਫਾਈਲ ਵਿੱਚ ਨਹੀਂ ਲਿਖਿਆ ਜਾਵੇਗਾ, ਅਤੇ ਤੁਹਾਡੀ ਦੇਖਭਾਲ ਬਾਰੇ ਕਿਸੇ ਫ਼ੈਸਲੇ ਲਈ ਨਹੀਂ ਵਰਤਿਆ ਜਾਵੇਗਾ।

**ਇਸ ਵਿੱਚ ਕੀ ਲਾਭ ਅਤੇ ਜੋਖਮ ਹਨ?** — ਤੁਹਾਨੂੰ ਕੋਈ ਸਿੱਧਾ ਲਾਭ ਨਹੀਂ ਹੈ। ਕੋਈ ਸਰੀਰਕ ਜੋਖਮ ਨਹੀਂ ਹੈ, ਕਿਉਂਕਿ ਸਿਰਫ਼ ਤਸਵੀਰਾਂ ਲਈਆਂ ਜਾਣਗੀਆਂ। ਮੁੱਖ ਜੋਖਮ ਤੁਹਾਡੀ ਨਿੱਜਤਾ ਨਾਲ ਸਬੰਧਤ ਹੈ, ਅਤੇ ਉਸਦੀ ਸੁਰੱਖਿਆ ਲਈ ਕੀਤੇ ਗਏ ਉਪਾਅ ਹੇਠਾਂ ਦਿੱਤੇ ਗਏ ਹਨ।

**ਕੀ ਮੇਰੀ ਜਾਣਕਾਰੀ ਗੁਪਤ ਰੱਖੀ ਜਾਵੇਗੀ?** — ਹਾਂ। ਤੁਹਾਡੀਆਂ ਤਸਵੀਰਾਂ ਤੁਹਾਡੇ ਨਾਮ ਨਾਲ ਨਹੀਂ, ਇੱਕ ਨੰਬਰ ਨਾਲ ਸਟੋਰ ਕੀਤੀਆਂ ਜਾਣਗੀਆਂ। ਤੁਹਾਡੇ ਚਿਹਰੇ ਦੀ ਤਸਵੀਰ ਨਹੀਂ ਲਈ ਜਾਵੇਗੀ। ਤੁਹਾਡਾ ਨਾਮ, ਹਸਪਤਾਲ ਨੰਬਰ ਅਤੇ ਫ਼ੋਨ ਨੰਬਰ ਤਸਵੀਰਾਂ ਨਾਲ ਸਟੋਰ ਨਹੀਂ ਕੀਤਾ ਜਾਵੇਗਾ।

**ਕੀ ਮੇਰੀ ਪਛਾਣ ਗੁਪਤ ਰੱਖੀ ਜਾਵੇਗੀ?** — ਹਾਂ। ਜੇ ਭਵਿੱਖ ਵਿੱਚ ਇਹ ਤਸਵੀਰਾਂ ਖੋਜ ਲਈ ਵਰਤੀਆਂ ਜਾਣ ਜਾਂ ਹੋਰ ਖੋਜਕਰਤਾਵਾਂ ਨਾਲ ਸਾਂਝੀਆਂ ਕੀਤੀਆਂ ਜਾਣ, ਤਾਂ ਉਹ ਸਿਰਫ਼ ਅਜਿਹੇ ਰੂਪ ਵਿੱਚ ਸਾਂਝੀਆਂ ਕੀਤੀਆਂ ਜਾਣਗੀਆਂ ਜਿਸ ਵਿੱਚ ਤੁਹਾਡੀ ਪਛਾਣ ਨਾ ਹੋ ਸਕੇ, ਅਤੇ ਸੰਸਥਾਗਤ ਨੈਤਿਕਤਾ ਕਮੇਟੀ ਤੋਂ ਵੱਖਰੀ ਮਨਜ਼ੂਰੀ ਲੈਣ ਤੋਂ ਬਾਅਦ ਹੀ।

**ਕੀ ਮੈਨੂੰ ਕੁਝ ਭੁਗਤਾਨ ਕਰਨਾ ਪਵੇਗਾ?** — ਨਹੀਂ। ਇਸ ਅਧਿਐਨ ਵਿੱਚ ਭਾਗ ਲੈਣ ਲਈ ਤੁਹਾਨੂੰ ਕਿਸੇ ਵੀ ਪ੍ਰਕਾਰ ਦਾ ਕੋਈ ਖ਼ਰਚਾ ਨਹੀਂ ਦੇਣਾ ਪਵੇਗਾ।

**ਹੋਰ ਜਾਣਕਾਰੀ ਲਈ ਸੰਪਰਕ** — ਸ੍ਰੀ ਉੱਜਵਲ ਠਾਕਰਾਨ, ਐਮਬੀਬੀਐਸ ਵਿਦਿਆਰਥੀ 2022 ਬੈਚ, ਕ੍ਰਿਸ਼ਚੀਅਨ ਮੈਡੀਕਲ ਕਾਲਜ ਅਤੇ ਹਸਪਤਾਲ, ਲੁਧਿਆਣਾ। ਮੋਬਾਈਲ: 9999209888

**ਸ਼ਿਕਾਇਤ ਲਈ ਸੰਪਰਕ** — ਡਾ. ਜਿਤੇਂਦਰ ਮਹਿੰਦਰੂ, ਚੇਅਰਮੈਨ, ਸੰਸਥਾਗਤ ਨੈਤਿਕਤਾ ਕਮੇਟੀ, ਕ੍ਰਿਸ਼ਚੀਅਨ ਮੈਡੀਕਲ ਕਾਲਜ ਅਤੇ ਹਸਪਤਾਲ, ਲੁਧਿਆਣਾ।

## REFERENCES

Agarwala, S., Sharma, S. and Bhattacharyya, J. (2020) 'Melanoma in the Indian subcontinent: an overview', *Indian Journal of Dermatology, Venereology and Leprology*, 86(5), pp. 483–492.

Angelopoulos, A.N., Bates, S., Malik, J. and Jordan, M.I. (2021) 'Uncertainty sets for image classifiers using conformal prediction', *International Conference on Learning Representations (ICLR)*.

Daneshjou, R., Smith, M.P., Sun, M.D., Rotemberg, V. and Zou, J. (2021) 'Lack of transparency and potential bias in artificial intelligence data sets and algorithms: a scoping review', *JAMA Dermatology*, 157(11), pp. 1362–1369.

Daneshjou, R., Vodrahalli, K., Novoa, R.A., Jenkins, M., Liang, W., Rotemberg, V., Ko, J., Swetter, S.M., Kiyohara, E.M., Zhu, Y., Chiou, G., Gupta, S., Zou, J. and Chen, J.Y. (2022) 'Disparities in dermatology AI performance on a diverse, curated clinical image set', *Science Advances*, 8(31), eabq6147.

Groh, M., Harris, C., Soenksen, L., Lau, F., Han, R., Kim, A., Koochek, A. and Badri, O. (2021) 'Evaluating deep neural networks trained on clinical images in dermatology with the Fitzpatrick 17k dataset', *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops*, pp. 1820–1828.

Guo, C., Pleiss, G., Sun, Y. and Weinberger, K.Q. (2017) 'On calibration of modern neural networks', *Proceedings of the 34th International Conference on Machine Learning*, 70, pp. 1321–1330.

Gupta, A.K., Bharadwaj, M. and Mehrotra, R. (2016) 'Skin cancer concerns in people of colour: risk factors and prevention', *Asian Pacific Journal of Cancer Prevention*, 17(12), pp. 5257–5264.

Kumar, A., Raghunathan, A., Jones, R., Ma, T. and Liang, P. (2022) 'Fine-tuning can distort pretrained features and underperform out-of-distribution', *International Conference on Learning Representations (ICLR)*.

Ministry of Electronics and Information Technology, Government of India (2023) *The Digital Personal Data Protection Act, 2023*. New Delhi: Government of India.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M. and Khalidov, V. (2024) 'DINOv2: learning robust visual features without supervision', *Transactions on Machine Learning Research*.

Pacheco, A.G.C., Lima, G.R., Salomao, A.S., Krohling, B., Biral, I.P. and de Angelo, G.G. (2020) 'PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones', *Data in Brief*, 32, 106221.

Tschandl, P., Rosendahl, C. and Kittler, H. (2018) 'The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions', *Scientific Data*, 5, 180161.

Ward, A., Li, J., Wang, J., Lakshminarasimhan, S., Carrick, A. and Campana, B. (2024) 'Creating an empirical dermatology dataset through crowdsourcing with web search advertisements', *JAMA Network Open*, 7(11), e2446615.
