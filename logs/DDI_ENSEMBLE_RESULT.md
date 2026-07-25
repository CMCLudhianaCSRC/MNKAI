# DDI dark-skin cancer — frozen multi-backbone ENSEMBLE (final result)

**The #1 leaderboard lever worked.** Score-averaging six *frozen* foundation-model probes lifts the DDI
malignant-vs-benign AUROC well past the single-backbone baseline, with the largest gains on dark skin.

## Method
DDI = 656 biopsy-confirmed images → image-level **5-fold StratifiedKFold**
(DDI's 656 images come from 570 patients, but the public metadata carries no patient identifier, so patient-grouped CV is not constructible; ~80-90 images share a patient, a residual source of optimism we disclose. Note this is NOT the original paper's protocol - Daneshjou et al. 2022 used a randomized 60/20/20 split.)
Each backbone: frozen features → per-fold `StandardScaler` → `LogisticRegression(class_weight="balanced", C=0.5)`
→ OOF P(malignant). **Ensemble = the mean of the 6 backbones' OOF scores** (score-average beat feature-concat).
Backbones: DINOv2-B/reg/L @518, EVA-02-B @448, ConvNeXt-V2-B @384, SigLIP-B @384. (`scripts/_ddi_ensemble.py`)

## Result (with bootstrap 95% CIs — `scripts/_ddi_ensemble_ci.py`)

| Model | Overall AUROC | **FST V–VI (dark)** | FST III–IV | FST I–II |
|---|---|---|---|---|
| Single DINOv2-B @518 (baseline) | 0.748 [0.704, 0.791] | 0.789 [0.713, 0.859] | 0.739 | 0.732 |
| DINOv2-Large @518 (best single) | 0.791 | 0.818 | 0.781 | 0.787 |
| Ensemble — concat features | 0.808 | 0.834 | 0.816 | 0.777 |
| **Ensemble — score-average** | **0.826 [0.789, 0.862]** | **0.872 [0.808, 0.929]** | 0.830 [0.771, 0.882] | 0.782 [0.701, 0.856] |

## Significance & stability
- **Paired bootstrap ΔAUROC (ensemble − baseline):** overall **+0.078** [0.049, 0.107], p < 0.001;
  **V–VI +0.083** [0.034, 0.133], p = 0.0005; III–IV +0.091, p < 0.001; I–III +0.050, p = 0.03.
  The ensemble is significantly better overall and in every darker-tone bin.
- **Multi-seed CV stability (fold seeds 0–4):** overall 0.830 ± 0.009; V–VI **0.874 ± 0.007** — the point
  estimate is not a lucky-fold artifact.

## Reading
- Dark-skin (V–VI) AUROC **0.789 → 0.872** is a real, significant gain and sits at/above the published DDI
  ceiling. This is the headline clinical result.
- Consistent with the project thesis: **all six members are FROZEN** — no fine-tuning. The gain comes from
  *architectural diversity* across frozen SSL/contrastive backbones, not from adapting any one of them.
- Honest caveats for the paper: DDI is small (656; per-tone Ns modest → wide V–VI CI), single-source, USA-clinic;
  AUROC is threshold-free — an operating-point/decision-curve + score-calibration analysis is still owed (roadmap R14).

Artifacts: `eval/probe/ddi_ensemble.json`, `eval/probe/ddi_ensemble_ci.json`, `eval/probe/ddi_backbones/*.npy` (cached features).
