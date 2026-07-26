%%TITLE%%Artificial intelligence for the assessment of skin conditions and skin cancer across skin tones: an interim report
%%AUTHORS%%Ujjwal Thakran, [Co-author], [Co-author]
%%AFFIL%%Ujjwal Thakran | Department of Dermatology, Christian Medical College & Hospital, Ludhiana, Punjab, India
%%AFFIL%%[Co-author] | Department of Dermatology, Christian Medical College & Hospital, Ludhiana, Punjab, India
%%AFFIL%%[Co-author] | Department of Dermatology, Christian Medical College & Hospital, Ludhiana, Punjab, India
%%CORR%%Ujjwal Thakran, Department of Dermatology, Christian Medical College & Hospital, Ludhiana, Punjab, India
%%NOTE%%Interim report. Data collection for the prospective arm has not commenced; final calibration and validation are ongoing. Figures presented here are interim and are expected to be refined.

## Abstract

**Introduction:** Computer programs that read photographs of skin lesions are being proposed as a way of helping doctors decide which patients need a specialist. Almost all such programs, however, have been developed using photographs of light-skinned patients. When three published programs were independently tested on a collection of brown and dark skin, their ability to separate cancer from non-cancer fell close to chance. This is a particular problem in India, where most patients have brown skin and where there are very few dermatologists for the population. The present work asked whether a program could be built that performs equally well across skin tones, that states honestly when it is unsure, and that runs on ordinary hardware available in a district hospital.

**Methods:** About 266,000 labelled photographs were assembled from eight publicly available research collections, of which approximately 15,000 were of darker skin. Rather than training a large program from the beginning, several existing publicly available image-reading programs were used unchanged, and only a small decision layer was trained on top of them. The ability to distinguish cancer from non-cancer was tested on an international reference collection of 656 photographs in which every diagnosis had been confirmed by biopsy and in which skin tones were deliberately balanced. Ability to name the condition was tested separately across 25 common skin conditions. Accuracy was reported with 95% confidence intervals; comparisons used paired resampling tests.

**Results:** On the biopsy-confirmed collection, the program separated cancer from non-cancer with an area under the curve of 0.826 (95% CI 0.789-0.862) overall and 0.872 (95% CI 0.808-0.929) on the darkest skin, compared with 0.50-0.57 on the darkest skin for three previously published programs. At a setting that correctly cleared 80% of non-cancerous lesions, sensitivity for cancer was 0.750 on the darkest skin. Performance across the three skin-tone groups was 0.782, 0.830 and 0.872, with overlapping confidence intervals, indicating no significant difference by skin tone. Across 25 conditions the mean per-condition performance score was 0.573 (95% CI 0.552-0.591). When asked to hand over its 30% least certain cases to a doctor, accuracy on the remainder rose to 0.766. During testing, approximately 35% of the test photographs were found to be near-duplicates of photographs already seen during development; these were identified and removed, and all figures reported here are after their removal.

**Keywords:** Artificial intelligence, dermatology, skin cancer, skin of colour, Fitzpatrick skin type, decision support

## Introduction

Skin complaints are among the commonest reasons for an outpatient consultation in India, and the number of trained dermatologists per head of population is small. A program able to look at a photograph of a lesion and indicate which patients require specialist review therefore has obvious appeal, particularly for screening camps, peripheral centres and teleconsultation.

The evidence supporting such programs is, however, unbalanced. The large public photograph collections on which they are developed were assembled largely in Europe, North America and Australia, and consist mostly of light-skinned patients, frequently photographed through a dermatoscope rather than with an ordinary camera [1, 2]. Programs developed on such material cannot be assumed to work on Indian patients.

This concern is not theoretical. Daneshjou and colleagues assembled a collection of 656 photographs, deliberately balanced for skin tone, in which every diagnosis was confirmed by biopsy, and tested three previously published programs on it [3]. All three performed substantially worse on the darkest skin than on the lightest. Expressed as area under the receiver operating characteristic curve, one program fell from 0.61 on light skin to 0.50 on dark skin, a value equivalent to guessing; the second fell from 0.64 to 0.55, and the third from 0.72 to 0.57. When those same programs were subsequently adapted using photographs from the collection itself, performance on the darkest skin improved to approximately 0.76 and 0.78 [3].

The clinical importance of this gap is amplified by the way skin cancer presents in brown skin. Melanoma in Indian and other brown-skinned populations arises more often on the palms, soles and nail beds than on sun-exposed sites, is less likely to be detected early, and carries a correspondingly poorer prognosis [4, 5]. A screening aid that performs least well precisely where detection is already most difficult is of limited value.

A second and less-discussed difficulty concerns trust rather than accuracy. Most published programs return a diagnosis and nothing else. They do not indicate how confident they are, do not identify the cases they are unfit to judge, and do not report whether they perform less well in particular groups of patients [6]. A clinician cannot use such a program safely, because there is no way to know when to disregard it.

Groh and colleagues drew attention to the same imbalance in a large public collection annotated by skin type, and reported performance that varied with skin tone [2]. Similar findings have prompted several groups to develop dermatology-specific image programs. Most of these require substantial computing resources, and several are released under licences that restrict clinical or commercial use.

Against this background, the present work asked three questions. First, whether a program could be constructed that separates cancer from non-cancer on brown and dark skin as well as it does on light skin. Second, whether such a program could be made to state honestly when it is uncertain, and to recommend specialist referral rather than guess. Third, whether this could be achieved on ordinary computing hardware, without the servers and specialised equipment assumed by most published work. This report presents interim findings; collection of local photographic data and final calibration of the program are ongoing and are described under Future Work.

## Material and Methods

### Study design

This was a retrospective computational study using publicly available, anonymised research photograph collections. No patients were recruited, no identifiable patient data were handled, and no clinical intervention was undertaken. A prospective photographic collection study has been separately approved and had not commenced at the time of this report.

### Photograph collections used

Approximately 266,000 labelled photographs were assembled from eight publicly available research collections, comprising both dermatoscopic images and ordinary clinical photographs. Of these, approximately 15,000 were of Fitzpatrick skin types IV to VI. This scarcity of darker-skinned material is the central limitation of the field and of this work.

Two separate tasks were defined. The first was the naming of the condition among 25 common diagnoses, using 33,250 photographs for development and a separate set of 5,886 photographs, never used during development, for testing. The second was a nine-category dermatoscopic task using 197,137 photographs for development.

For the assessment of cancer detection across skin tones, the reference collection of Daneshjou and colleagues was used [3]. This consists of 656 photographs obtained from 570 patients at a single United States academic centre. Every diagnosis was confirmed by histopathology. Of the 656 lesions, 171 (26.1%) were malignant. The darkest skin-tone group comprised 207 photographs, of which 48 were malignant.

### How the program was constructed

A conventional approach would take a large image-reading program and retrain it in its entirety on dermatological photographs. In this work, that approach was found to reduce rather than improve performance, because the quantity of dermatological material available is small relative to the size of such programs, and retraining degrades the general visual knowledge they already possess.

The approach adopted instead was to leave several existing, publicly available image-reading programs entirely unchanged, and to train only a small decision layer on top of the descriptions they produce. Six such programs were used, developed by different research groups using different methods, so that their individual errors are largely independent. Their opinions were then averaged. The principle is that of a clinical meeting in which several specialists trained at different institutions review the same case: the consensus is more reliable than any individual opinion.

Because none of the six underlying programs was altered, the entire procedure was completed on a single laptop computer with 6 GB of graphics memory. The photographs were processed once, and all subsequent analysis was performed on the stored descriptions.

### Safety behaviours

Three behaviours were added so that the program's output could be used safely.

First, its stated confidence was corrected so that it corresponds to observed accuracy: when the program reports 70% certainty, it should be correct on approximately 70% of such occasions.

Second, instead of a single diagnosis the program returns a short list of possible diagnoses, constructed so that the correct diagnosis is contained within it on approximately 90% of occasions.

Third, the program identifies the cases about which it is least certain and recommends that these be referred to a dermatologist rather than issuing an opinion.

### Statistical analysis

Discrimination between malignant and non-malignant lesions was expressed as the area under the receiver operating characteristic curve (AUC), for which 0.5 indicates performance no better than chance and 1.0 indicates perfect separation. Performance in naming conditions across 25 categories was expressed as the mean of the per-condition performance scores, which weights uncommon conditions equally with common ones; a value of approximately 0.04 would be expected by chance alone.

All estimates are reported with 95% confidence intervals obtained by resampling within skin-tone groups. Comparisons between the combined program and a single program were made using a paired resampling test on identical cases, with a one-sided p-value. Stability was assessed by repeating the entire analysis with five different random divisions of the data. Analyses of the reference collection used fivefold cross-validation, in which the collection is divided into five parts and each part is assessed in turn by a program developed on the remaining four.

### Detection of duplicated photographs

During analysis it became apparent that the initial results were implausibly favourable. Comparison of the numerical descriptions of the test photographs with those used in development showed that approximately 35% of the test photographs were near-identical duplicates of photographs already seen during development, arising from the same atlas images appearing in more than one public collection. All such photographs were removed and every figure was recalculated. All results reported here are after this correction. The sensitivity of the findings to the threshold used to define a duplicate was examined and is reported below.

## Results

### Detection of skin cancer

On the biopsy-confirmed reference collection, the combined program separated malignant from non-malignant lesions with an AUC of 0.826 (95% CI 0.789-0.862) overall. A single program used alone achieved 0.748 (95% CI 0.704-0.791). The improvement of 0.078 (95% CI 0.049-0.107) was statistically significant (p<0.001). Results were stable across five repetitions (0.830, standard deviation 0.009) (Table 1).

### Performance by skin tone

Performance across the three skin-tone groups was 0.782 (95% CI 0.701-0.856) for Fitzpatrick types I-II, 0.830 (95% CI 0.771-0.882) for types III-IV, and 0.872 (95% CI 0.808-0.929) for types V-VI. The confidence intervals of the lightest and darkest groups overlap, and the difference between them was therefore not statistically significant (Table 1).

On the darkest skin the combined program improved upon the single program by 0.083 (95% CI 0.034-0.133, p=0.001), and this result was likewise stable across repetitions (0.874, standard deviation 0.007).

### Clinical operating characteristics

At a threshold at which 80% of non-malignant lesions were correctly cleared, sensitivity for malignancy was 0.696 overall and 0.750 in the darkest skin-tone group. At a stricter threshold clearing 90% of non-malignant lesions, sensitivity was 0.538 overall and 0.625 in the darkest group (Table 2).

Across the range of thresholds a clinician might reasonably adopt, use of the program conferred greater net benefit than either referring every patient or referring none.

### Naming the condition

Across 25 conditions, the mean per-condition performance score on the independent test set was 0.573 (95% CI 0.552-0.591), against a value of approximately 0.04 expected by chance. The conventional approach of retraining the entire program achieved 0.527 (95% CI 0.510-0.543) on the same photographs; the confidence intervals do not overlap. On the nine-category dermatoscopic task the corresponding values were 0.660 (95% CI 0.654-0.665) and 0.637 (Table 3).

### Effect of the duplicate-removal threshold

The threshold used to define a duplicate photograph was varied between 0.95 and 0.995. The mean per-condition score varied smoothly between 0.545 and 0.574 across this range. At the most stringent threshold, which removed almost half of the test photographs, performance remained above that of the conventional retraining approach. The reported figure is therefore not an artefact of the particular threshold selected.

### Safety behaviours

After correction, the discrepancy between stated confidence and observed accuracy fell from 0.21 to 0.042. The short list of possible diagnoses contained the correct diagnosis on 89.2% of occasions against a target of 90%, and contained a mean of 3.4 diagnoses out of 25. When the program referred its 30% least certain cases, accuracy on the cases it retained rose to 0.766 (Table 4).

### Reduction to four programs

Of the six programs combined, two contributed almost nothing. Using only four reduced computation by approximately one third while performance on the darkest skin fell from 0.872 to 0.870.

## Discussion

The principal interim finding of this work is that performance was comparable across skin tones. This contrasts with the three previously published programs assessed by Daneshjou and colleagues on the same collection, each of which performed between 0.09 and 0.15 worse on the darkest skin than on the lightest [3]. In the present work the darkest group was not disadvantaged.

It is important to state this claim carefully. The confidence intervals for the lightest and darkest groups overlap, and the darkest group contains only 48 malignant lesions. The defensible conclusion is therefore that no significant difference by skin tone was demonstrated, and not that performance was superior on darker skin. It should further be noted that the previously published programs were assessed without any exposure to this collection, whereas the present program was developed and assessed within it using cross-validation. The comparison therefore speaks to the presence or absence of a gap between skin tones under each approach, rather than constituting a direct comparison of accuracy.

Compared with the same programs after they had been adapted using the collection, which reached approximately 0.76 and 0.78 on the darkest skin [3], the present interim figure of 0.872 is higher, although the confidence interval is wide and the assessment methods differ.

The second finding of interest is that leaving the underlying programs unchanged produced better results than retraining them. This is consistent with work outside medicine showing that retraining a large program on a small quantity of data can distort the general knowledge it has already acquired, and that a simple decision layer added on top may generalise better [7]. The present work appears to be a clinical demonstration of that phenomenon rather than a new observation. A control analysis further indicated that the advantage arises from leaving the programs unchanged and from combining several of them, rather than from any particular method by which an individual program was originally developed.

The practical significance of a sensitivity of 0.750 at 80% specificity on darker skin should be judged against current practice rather than against perfection. Approximately three of every four malignant lesions were identified while three-quarters of benign lesions were correctly cleared. One in four cancers was missed at that threshold, which is why the program is presented as an aid to triage and not as a diagnostic test, and why the referral behaviour described above is integral to its intended use.

The 25-condition results should be interpreted more cautiously. The program performs far better than chance but does not approach specialist accuracy, and is best regarded as an aid to triage and a source of a second opinion. Of greater importance, the ability of the present work to assess performance in darker-skinned patients for these 25 conditions is weak: the independent test set contained only approximately 135 photographs of darker skin distributed across 25 conditions, roughly five per condition. No reliable conclusion about performance by skin tone for this task can be drawn from so small a sample, and none is offered.

The discovery that approximately 35% of the test photographs were duplicates of development photographs deserves emphasis, because it is unlikely to be peculiar to this work. Public collections in dermatology draw on overlapping atlas sources, and a test set assembled from several of them may contain photographs already seen. Had this not been detected, the results reported here would have been approximately 0.05 higher and would have been incorrect. Verification for duplicated photographs is recommended as a routine step whenever material from more than one public collection is combined.

### Limitations

This is an interim report and several limitations must be stated. Assessment of cancer detection rests entirely on a single collection of 656 photographs from one centre in the United States; there is no comparable collection from India, and the findings cannot be assumed to transfer to Indian patients. The darkest skin-tone group contains 48 malignant lesions, so the corresponding confidence interval is wide. Although the collection contains 656 photographs from 570 patients, patient identifiers are not released, so it is not possible to ensure that photographs from the same patient were kept together during analysis; a small number of such photographs will have been separated, which may render the figures slightly optimistic. The method of assessment used here differs from that used in the original report of the collection, so comparisons are not exact. Performance has not been compared against that of dermatologists, and the program has not been used prospectively in any clinic. Final calibration has not been completed and the figures presented are expected to be refined.

### Future work

A prospective photographic collection study has been approved and will address the principal limitation directly. It is intended to collect 3,000 to 5,000 consented photographs across two hospitals in North India over approximately one year, each labelled by dermatologists, with histopathological confirmation for all suspected malignancies. To our knowledge no dermatological photograph collection from North India exists, and none anywhere combines Indian clinical photographs with a biopsy-confirmed cancer component. Such a collection would permit the present findings to be confirmed or refuted in the population for which the program is intended, and would constitute a resource of independent value.

## Conclusion

In this interim analysis, a program constructed by combining several unchanged, publicly available image-reading programs distinguished malignant from non-malignant lesions with an area under the curve of 0.872 on the darkest skin, without the reduction in performance on darker skin reported for previously published programs. The program corrects its stated confidence, offers a short list rather than a single answer, and refers the cases about which it is least certain. All of this was achieved on ordinary computing hardware. These findings are interim; final calibration and prospective assessment in Indian patients are required before any clinical use, and the program is intended to support and not to replace clinical judgement.

## Tables

**Table 1:** Discrimination between malignant and non-malignant lesions, by skin tone (656 biopsy-confirmed photographs)

| Group | Single program | Combined program | 95% CI (combined) |
| Overall | 0.748 | 0.826 | 0.789-0.862 |
| Fitzpatrick I-II | 0.732 | 0.782 | 0.701-0.856 |
| Fitzpatrick III-IV | 0.739 | 0.830 | 0.771-0.882 |
| Fitzpatrick V-VI | 0.789 | 0.872 | 0.808-0.929 |

**Table 2:** Operating characteristics of the combined program

| Threshold | Overall | Fitzpatrick V-VI |
| Sensitivity at 80% specificity | 0.696 | 0.750 |
| Sensitivity at 90% specificity | 0.538 | 0.625 |
| Specificity at 90% sensitivity | 0.538 | 0.635 |

**Table 3:** Naming the condition, mean per-condition score

| Task | Conventional retraining | Present approach | 95% CI |
| 25 conditions (independent test) | 0.527 | 0.573 | 0.552-0.591 |
| 9 categories (cross-validation) | 0.637 | 0.660 | 0.654-0.665 |

**Table 4:** Safety behaviours

| Measure | Before | After |
| Discrepancy between stated confidence and accuracy | 0.21 | 0.042 |
| Number of diagnoses in the short list (of 25) | 10.6 | 3.4 |
| Accuracy after referring the least certain 30% | 0.637 | 0.766 |

## References

1. Tschandl P, Rosendahl C, Kittler H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci Data. 2018;5:180161.
2. Groh M, Harris C, Soenksen L, Lau F, Han R, Kim A, et al. Evaluating deep neural networks trained on clinical images in dermatology with the Fitzpatrick 17k dataset. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops; 2021. p. 1820-1828.
3. Daneshjou R, Vodrahalli K, Novoa RA, Jenkins M, Liang W, Rotemberg V, et al. Disparities in dermatology AI performance on a diverse, curated clinical image set. Sci Adv. 2022;8(31):eabq6147.
4. Gupta AK, Bharadwaj M, Mehrotra R. Skin cancer concerns in people of colour: risk factors and prevention. Asian Pac J Cancer Prev. 2016;17(12):5257-5264.
5. Agarwala S, Sharma S, Bhattacharyya J. Melanoma in the Indian subcontinent: an overview. Indian J Dermatol Venereol Leprol. 2020;86(5):483-492.
6. Daneshjou R, Smith MP, Sun MD, Rotemberg V, Zou J. Lack of transparency and potential bias in artificial intelligence data sets and algorithms: a scoping review. JAMA Dermatol. 2021;157(11):1362-1369.
7. Kumar A, Raghunathan A, Jones R, Ma T, Liang P. Fine-tuning can distort pretrained features and underperform out-of-distribution. In: International Conference on Learning Representations (ICLR); 2022.
8. Oquab M, Darcet T, Moutakanni T, Vo H, Szafraniec M, Khalidov V, et al. DINOv2: learning robust visual features without supervision. Trans Mach Learn Res. 2024.
9. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. In: Proceedings of the 34th International Conference on Machine Learning; 2017. p. 1321-1330.
10. Angelopoulos AN, Bates S, Malik J, Jordan MI. Uncertainty sets for image classifiers using conformal prediction. In: International Conference on Learning Representations (ICLR); 2021.
11. Pacheco AGC, Lima GR, Salomao AS, Krohling B, Biral IP, de Angelo GG, et al. PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones. Data Brief. 2020;32:106221.
12. Ward A, Li J, Wang J, Lakshminarasimhan S, Carrick A, Campana B, et al. Creating an empirical dermatology dataset through crowdsourcing with web search advertisements. JAMA Netw Open. 2024;7(11):e2446615.
