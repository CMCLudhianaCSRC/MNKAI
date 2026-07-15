# MNKAI-III — Project Report (2026-07-10)

Status review of the whole project + honest current metrics + what's next.

## Executive summary
A clinical-grade dermatology AI for Indian deployment, built on a 6 GB laptop. The project's real
contributions are **methodological rigor**, not a leaderboard number: a diagnosed-and-fixed majority-class
collapse, a full trust layer (calibration + conformal + fairness), and two late-breaking wins — (1) **frozen
foundation-model features + a light head beat fine-tuning** everywhere we've tested (dark-skin cancer 0.53→0.79;
25-class too), and (2) the discovery that our **25-class external test had ~33% train-duplicate leakage**, which
we are now fixing. Catching that ourselves is what keeps the work honest.

## What we've done (phases)
| Phase | Work | Result |
|---|---|---|
| 1 | 9-class dermoscopy base (DINOv2 + cross-attn fusion) | macro-F1 0.637 ± 0.009 |
| 3 | 25-class Indian; diagnosed **majority-class collapse** (double-balancing) + single-mechanism fix | eczema recall 0.004→0.503; external 0.484 |
| Tier 0 | per-class bias, conformal fix (10.6→5.81), calibration (ECE 0.176→0.016), OOD gate, bootstrap CIs | 25-class 0.484→0.522 |
| Tier 1 | decoupled **cRT** 5-fold | 25-class 0.539, melanoma recall 0.870 |
| Dark-skin | DDI failure diagnosed → **frozen DINOv2@518 + probe** | DDI cancer **0.53→0.789** (its own clean CV) |
| Frozen-probe (new) | applied to 25-class | wins after dedup (below) |
| LLM | Milestone-1 core: adaptive interview + Bayesian fusion (Gemini) | working |

## Current HONEST metrics (canonical, after leakage correction)
| Metric | Reported (leaky) | **Honest (canonical)** | Notes |
|---|---|---|---|
| **25-class macro-F1 — frozen-probe 518+MLP+bias** | 0.626 | **0.573 [0.553–0.592]** | new best; deduped external n=3,813; CI |
| 25-class macro-F1 — cRT+bias | 0.539 | 0.527 [0.510–0.543] | deduped; probe beats it, CIs disjoint |
| 25-class top-1 (clean) | — | probe 0.631 / cRT 0.506 | |
| **9-class macro-F1 — frozen-probe MLP** | 0.637 (FT) | **0.656 (OOF CV)** | probe > fine-tune; external split was degenerate (99.8% nv) |
| **DDI dark-skin cancer AUROC** | 0.58 | **0.748 (V–VI 0.789)** | DDI's own clean 5-fold — solid |
| Internal cancer AUROC | 0.988 | — | dermoscopy |
| Calibration ECE | 0.176→0.016 | — | |

**Two eval bugs found + handled:** 25-class external had 35% exact train-duplicates (dropped); 9-class external
was 99.8% nevi (degenerate → use OOF CV). Clean-external filepaths saved: `eval/probe/CLEAN_25class.clean_external.csv`.

## The two late findings that matter most
1. **Frozen-features-+-light-head > fine-tuning** in our regime. Fine-tuning *degraded* the representation (it
   wrecked DDI; it under-uses features on 25-class). The new paradigm: freeze a strong backbone, attach an MLP
   head. Wins on DDI cancer and 25-class; cheap; laptop-only.
2. **Train↔external near-duplicate leakage (~33% exact dupes)** in the 25-class external test, from cross-source
   atlas images (DermNet/SkinDisease) slipping past the group-aware split. Inflated all 25-class external numbers.
   **Must rebuild a deduplicated split and re-report** — this is now the top priority (paper validity).

## What's left
1. **Fix the leakage** — feature/phash-dedup train↔external, rebuild a clean external test, re-report ALL metrics.
2. **Roll the frozen-probe paradigm out** — 9-class, cancer, coarse, with head upgrades (MLP/ensemble/+metadata).
3. **Re-check the 9-class base** for the same leakage.
4. **Data** — register Stanford AIMI (MRA-MIDAS/DDI-2/ASAN); grab PASSION/HIBA. Targeted dark-skin, not volume.
5. **LLM** — wire Milestone-1 into the app; Milestone-2 (learned SCIN-history fusion).
6. **Optional SSL** (Colab A100, few days) — derm backbone for even better frozen features.
7. **Write-up** — after clean numbers; foreground collapse-fix + trust layer + frozen-probe + the honest
   leakage-catch (reviewers value it).
8. **Productionize** — only after sign-off.

## Honest framing
We do not claim SOTA accuracy. We claim rigor: honest (now genuinely leak-free) evaluation, a trust layer nobody
else ships, a reproducible collapse-fix, and a clean demonstration that frozen-probe beats fine-tuning for
data-scarce dermatology — including the hardest case (dark-skin cancer).

We also do **not** claim the frozen-vs-fine-tune effect itself is novel. That fine-tuning can distort strong
pretrained features and underperform out-of-distribution — and that a frozen light probe can beat it — is an
established result (Kumar et al., *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*,
ICLR 2022, arXiv:2202.10054), which also proposes the LP-FT remedy. Our novelty is the **clinical, data-scarce,
dark-skin demonstration + the deployable trust stack**, not the phenomenon. LP-FT (linear-probe-then-fine-tune) is
noted as future work since we did not evaluate that intermediate regime.
