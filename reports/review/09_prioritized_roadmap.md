# MNKAI-III — Prioritized Roadmap to Completion (Agent 4)

Three phases by resource requirement. Every item maps to a recommendation ID in `04_consolidated_recommendations.md`. The organizing principle: **Phase 1 makes the existing numbers honest, reproducible, and defensible without new compute; Phase 2 needs one modest GPU pass; Phase 3 needs real resources (fine-tuning runs, external data, a reader study).**

---

## Phase 1 — Now, on the laptop (static edits + CPU sklearn on cached features)

No fresh feature extraction, no fine-tuning, no forbidden paths. Two tracks run in parallel: **1A static edits** apply immediately; **1B analysis** is authored now and executed once the running ensemble job frees the machine.

### 1A. Static edits — apply immediately (the auto-apply set)
- **R09** — Cite Kumar et al. 2022 (LP-FT); reframe novelty as clinical dark-skin demonstration + deployable trust stack.
- **R11** — Guard the manifest↔splits merge (normalize paths, assert row retention, log drops).
- **R13 (scaffolding)** — Restore/pin `requirements.txt`; strip hardcoded `D:/AIFINAL` paths; author the `clinical_25class.pkl` model-card skeleton.
- **R18** — Backbone kwarg-by-kwarg retry + re-raise (kills the silent 518→default downgrade).
- **R22** — Static hygiene: dead ternary, per-call import, ISIC `[:0]` dead code, `blur_score` + `threshold_for_target_error` docstrings.

### 1B. Analysis & controlled re-runs — author now, run when compute frees
- **R02** — Add `random_state=0` everywhere + seed before the OOF loop; single controlled re-run of `_save_deployable.py`; re-record 0.573 / ECE 0.016 / conformal set-size. *(Do this first — it is the precondition for trustworthy versions of every number below.)*
- **R01** — Verify DDI image→patient multiplicity; if needed, switch to `StratifiedGroupKFold` and reconcile 0.748/0.789.
- **R03** — Stratified bootstrap 95% CIs on every DDI AUROC, multiple seeds; retire "reaches the ceiling" language.
- **R04** — DeLong / paired-bootstrap ΔAUROC tests on all comparative claims.
- **R05** — Move config selection + conformal `(k_reg, λ)` selection off the evaluation split (OOF/held-out selection, evaluate once).
- **R06** — Frozen-feature duplicate audit across OOF folds and the 9-class folds; re-report 0.5948, tuned bias, 0.656.
- **R07** — Powered per-subgroup reporting (weak 25-class per-tone numbers surfaced with n / positives / multiple-comparisons note).
- **R12** — Dedup-threshold sensitivity curve (0.95–0.995).
- **R10** — Fix the zero-image silent fallback (fail loud, no zero-image, no index substitution, transform-sized placeholder, seeded RNG).
- **R15** — Resolve `exanthema` alias collision + add a collision check.
- **R16** — `GroupTemperatureScaler`: route small subgroups through the fitted global T.

**Outcome of Phase 1:** every headline number is deterministic, carries a CI and a significance test, is free of selection-on-test and cross-fold leakage, and is backed by honest per-subgroup reporting — plus a reproducibility package and the correct prior-art positioning. This is the minimum bar for submission.

---

## Phase 2 — Modest effort (one GPU feature-extraction pass; no full fine-tuning)

Runs after the ensemble job completes; each item is a single forward pass over cached images or a light analysis.

- **R08** — Supervised-ImageNet frozen-probe baseline; complete the 2×2 {SSL, supervised-IN} × {frozen, fine-tune}. **De-confounds the central claim** — the single highest-value Phase-2 item.
- **R14** — DDI operating-point, decision-curve / net-benefit, and temperature-scaling of the DDI cancer score.
- **R19** — Decide and document the focal-loss `no_grad` behavior (differentiate or declare deviation).
- **R20** — Seed mixup/cutmix RNG for full training determinism.
- **R21** — Parameterize corruption-sweep mean/std from `AugConfig`.

**Outcome of Phase 2:** the "freezing wins" claim is causally isolated from "DINOv2 is just a better representation," the DDI screen has clinical operating-point evidence, and training is fully reproducible.

---

## Phase 3 — Needs real resources (fine-tuning runs, external data, clinical study)

Not laptop-feasible; scoped for when compute or collaborators are available.

- **R17** — Protocol-matched end-to-end fine-tune vs frozen-probe on identical DDI-only group-disjoint folds at matched resolution. Converts the flagship 0.58→0.75 comparison from apples-to-oranges into a controlled head-to-head.
- **External dark-skin / Indian clinical validation** (2nd site) — retires the "DDI is the only public dark-skin clinical cancer set" ceiling and the data-limited ~0.78 cap.
- **Dermatologist reader study** — the decisive evidence for the decision-support / trust-stack contribution; no algorithmic substitute.

**Outcome of Phase 3:** the causal freezing claim is fully controlled, generalization is demonstrated beyond DDI, and the decision-support value is validated with clinicians — the requirements for a high-impact clinical-AI venue.

---

## One-paragraph critical path

Do **R02 (determinism)** first — nothing downstream is trustworthy until the numbers stop moving. Then **R01 (group-CV)**, **R03 (CIs)**, and **R04 (significance tests)** in parallel: these three decide whether the headline dark-skin result survives scrutiny. **R05/R06** remove the two remaining optimistic biases (selection-on-test, cross-fold leakage). Ship the Phase-1 static edits alongside. That package is submittable to a solid venue; **R08** (Phase 2) and **R17 + external validation + reader study** (Phase 3) are what lift it toward a high-impact one.
