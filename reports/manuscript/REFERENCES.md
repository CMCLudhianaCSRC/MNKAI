# MNKAI-III Manuscript — References

Real, correctly-attributed references for the claims in `MANUSCRIPT.md`. Details the author should confirm before submission are flagged **TODO-verify** (page numbers, exact venue strings, arXiv IDs). Nothing here is fabricated; where a fact is uncertain it is marked, not invented.

---

### Foundation models / backbones

1. **Oquab, M., Darcet, T., Moutakanni, T., et al.** (2023/2024). *DINOv2: Learning Robust Visual Features without Supervision.* Transactions on Machine Learning Research (TMLR). arXiv:2304.07193. — Frozen DINOv2 ViT-B/14 backbone; primary representation.

2. **Fang, Y., Sun, Q., Wang, X., et al.** (2023). *EVA-02: A Visual Representation for Neon Genesis.* arXiv:2303.11331. **TODO-verify** final venue (Image and Vision Computing, 2024). — EVA-02-B ensemble member.

3. **Woo, S., Debnath, S., Hu, R., et al.** (2023). *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders.* IEEE/CVF CVPR 2023. arXiv:2301.00808. — ConvNeXt-V2-B ensemble member.

4. **Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L.** (2023). *Sigmoid Loss for Language Image Pre-Training (SigLIP).* IEEE/CVF ICCV 2023. arXiv:2303.15343. — SigLIP-B ensemble member.

### The core phenomenon (LP-FT)

5. **Kumar, A., Raghunathan, A., Jones, R., Ma, T., Liang, P.** (2022). *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution.* International Conference on Learning Representations (ICLR) 2022. arXiv:2202.10054. — The central cited phenomenon; we frame our result as a clinical demonstration of it, not a new finding.

6. **FEDD / linear-probing-in-dermatology.** **TODO-verify** exact citation. The internal notes reference "FEDD, MICCAI 2023" for linear-probe-beats-fine-tune on DDI in the low-label regime. Confirm authors/title (likely *FEDD: Fair Efficient Diffusion-based Dermatology* or a linear-probing derm paper) and MICCAI 2023 before citing; drop if it cannot be verified.

### Dark-skin dermatology / fairness / benchmarks

7. **Daneshjou, R., Vodrahalli, K., Novoa, R. A., et al.** (2022). *Disparities in dermatology AI performance on a diverse, curated clinical image set (Diverse Dermatology Images, DDI).* Science Advances, 8(31), eabq6147. — DDI benchmark (656 biopsy-confirmed images), the dark-skin performance gap, and the published-model AUROC ceilings (~0.56–0.67 off-the-shelf; adaptation ~0.73–0.90). **TODO-verify** exact AUROC figures cited in text against the paper.

8. **Groh, M., Harris, C., Soenksen, L., et al.** (2021). *Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset.* IEEE/CVF CVPR Workshops (ISIC) 2021. arXiv:2104.09957. — Fitzpatrick17k; tone-dependent performance.

9. **Tschandl, P., Rosendahl, C., Kittler, H.** (2018). *The HAM10000 dataset: a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data, 5, 180161. — HAM10000; comparator training source in DDI baselines.

10. **Codella, N. C. F., Gutman, D., Celebi, M. E., et al.** (2018/2019). *Skin Lesion Analysis Toward Melanoma Detection (ISIC Challenge).* arXiv:1902.03368 (2019) / arXiv:1710.05006 (2018). — ISIC dermoscopy benchmarks used in the training corpus. **TODO-verify** which ISIC year(s) to cite for ISIC-2019/2024.

11. **Combalia, M., Codella, N. C. F., Rotemberg, V., et al.** (2019). *BCN20000: Dermoscopic Lesions in the Wild.* arXiv:1908.02288. — BCN20000 training source.

12. **Pacheco, A. G. C., Lima, G. R., Salomão, A. S., et al.** (2020). *PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones.* Data in Brief, 32, 106221. — PAD-UFES-20 clinical-photo source.

### Domain-specific dermatology foundation models (Related Work; not used as primary backbone)

13. **PanDerm.** **TODO-verify** citation (dermatology foundation model). Mentioned only to explain exclusion (mis-loaded checkpoint, non-commercial license). Confirm authors/venue or drop.

14. **SCIN dataset — Ward, A., et al.** (Google, 2024). *SCIN: Skin Condition Image Network.* **TODO-verify** exact citation/venue. — Clinical-photo training source.

15. **DermaCon-IN.** **TODO-verify** citation. — Indian dermatology dataset in the training mix. Confirm provenance/citation before inclusion.

### Trust stack — calibration, conformal, selective prediction

16. **Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q.** (2017). *On Calibration of Modern Neural Networks.* International Conference on Machine Learning (ICML) 2017. arXiv:1706.04599. — Temperature scaling.

17. **Angelopoulos, A. N., Bates, S., Malik, J., Jordan, M. I.** (2021). *Uncertainty Sets for Image Classifiers using Conformal Prediction (RAPS).* ICLR 2021. arXiv:2009.14193. — RAPS regularized adaptive prediction sets.

18. **Romano, Y., Sesia, M., Candès, E. J.** (2020). *Classification with Valid and Adaptive Coverage (APS).* Advances in Neural Information Processing Systems (NeurIPS) 2020. arXiv:2006.02544. — APS, the basis RAPS regularizes.

19. **Vovk, V., Gammerman, A., Shafer, G.** (2005). *Algorithmic Learning in a Random World.* Springer. — Conformal prediction foundations; Mondrian/group-conditional coverage (future work).

20. **Angelopoulos, A. N., Bates, S.** (2023). *Conformal Prediction: A Gentle Introduction.* Foundations and Trends in Machine Learning, 16(4), 494–591. — Tutorial reference for the conformal framework.

21. **Geifman, Y., El-Yaniv, R.** (2017). *Selective Classification for Deep Neural Networks.* NeurIPS 2017. arXiv:1705.08500. — Selective prediction / classification with a reject option (abstention curve).

### Long-tailed learning (fine-tuning comparator)

22. **Cui, Y., Jia, M., Lin, T.-Y., Song, Y., Belongie, S.** (2019). *Class-Balanced Loss Based on Effective Number of Samples.* IEEE/CVF CVPR 2019. arXiv:1901.05555. — Class-balanced (focal) loss; used in the collapse diagnosis and fine-tune baseline.

23. **Cao, K., Wei, C., Gaidon, A., Aréchiga, N., Ma, T.** (2019). *Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss (LDAM-DRW).* NeurIPS 2019. arXiv:1906.07413. — Deferred re-weighting; long-tail comparator context.

24. **Kang, B., Xie, S., Rohrbach, M., et al.** (2020). *Decoupling Representation and Classifier for Long-Tailed Recognition (cRT).* ICLR 2020. arXiv:1910.09217. — Classifier re-training (cRT), our matched fine-tuning baseline.

25. **Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P.** (2017). *Focal Loss for Dense Object Detection.* IEEE ICCV 2017. arXiv:1708.02002. — Focal loss origin.

### Methodology anchors (leakage, ensembling)

26. **CleanPatrick / Fitzpatrick17k label cleaning.** **TODO-verify** citation for the CleanPatrick cleaning procedure applied to Fitzpatrick17k. Confirm authors/venue or describe as an in-house cleaning step if no citation exists.

27. **Dietterich, T. G.** (2000). *Ensemble Methods in Machine Learning.* Multiple Classifier Systems, LNCS 1857, 1–15. — General reference for score-averaging ensembles and variance reduction via model diversity.

---

## Notes for the author

- **Highest-priority verifications:** refs 6 (FEDD/MICCAI), 13 (PanDerm), 14–15 (SCIN, DermaCon-IN), 26 (CleanPatrick). These are the ones most likely to be mis-attributed; verify or drop before submission.
- Confirm the DDI published-AUROC ranges (ref 7) as they appear in the manuscript text and in supplementary Table 6.
- npj Digital Medicine uses a numbered (Vancouver-style) reference format; convert this list to journal style at submission and add DOIs.
- Every reference here maps to at least one in-text claim; none is decorative.
