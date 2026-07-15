# Agent 8 — Innovation & Inventor Report (MNKAI-III)

*Inventor-mode, not reviewer-mode. Every idea below builds on a REAL strength that already exists in the codebase and is laptop-feasible (numpy on cached `.npz` features, sklearn heads, no GPU training, no touching the running ensemble/feature-extraction jobs).*

## 0. What the real strengths are (the raw material)

Reading the actual code, MNKAI-III's genuine, defensible assets are:

1. **The frozen-probe finding** (`_probe_multiclass.py`, `_probe_backbone.py`, `logs/DDI_BREAKTHROUGH.md`): a frozen DINOv2@518 + light head beats fine-tuning on every task because fine-tuning *destroys* the representation in the data-scarce, multi-domain regime. This is a real scientific claim, not a leaderboard trick.
2. **The frozen-feature leakage audit** (§6 of `PAPER_DRAFT.md`): cosine-NN in feature space found 33% train-test duplicates that the group-aware split missed. This is currently a one-off catch, not a protocol.
3. **The trust layer**, but as *four disconnected heuristics*: temperature scaling (`_trust_refresh_25class.py`), RAPS/clustered conformal (`_conformal_fix.py`), Mahalanobis OOD (`_ood_gate.py`), and abstention margin/set-size (`pipeline.py`). They all live in the SAME frozen-feature space but never talk to each other.
4. **The photo-grounded, confidence-gated LLM interviewer** (`llm.py`): `should_interview` gate + `propose_questions(image_b64=)` + Bayesian `fuse_history`. The LLM emits per-class log-LRs in [-2,2] that are **completely uncalibrated and unbounded in influence** — the one place a text channel can silently move a cancer decision.
5. **The dark-skin focus** with an honest, data-limited ~0.78 V–VI ceiling.

The unifying insight for this report: **everything already happens in one 768-d frozen-feature space, but the project treats that space as five separate tools. The inventions below exploit the geometry of that single space to get guarantees, fairness, and safety that none of the individual heuristics can give alone.**

---

## 1. Novel algorithms & methodologies

### IDEA 1 — Fitzpatrick-Mondrian Conformal with finite-sample subgroup coverage (the field-defining move)

**Problem it solves.** The current conformal (`_conformal_fix.py`) gives *marginal* coverage — averaged over all skin tones. But the entire premise of the project is dark-skin fairness, and §8 admits the dark subgroups are underpowered (V–VI n≈135). Marginal coverage can be 90% overall while dark skin gets 70% coverage and nobody notices. A fairness paper that reports fairness metrics but whose uncertainty guarantee is not itself fair is internally inconsistent.

**The invention.** Group-conditional (Mondrian) conformal prediction *stratified by Fitzpatrick*, with a **hierarchical shrinkage twist** that makes it work at n≈135. Naive per-group conformal at n=135 gives a jittery quantile; pure pooling gives an unfair one. Instead:
- Fit a pooled RAPS quantile `q_pool` and per-tone quantiles `q_g`.
- Shrink: `q_g* = w_g·q_g + (1−w_g)·q_pool`, where `w_g` grows with subgroup n (empirical-Bayes / James-Stein style).
- Certify with the **finite-sample worst-case correction**: report the group-conditional coverage lower bound at the realized `n_g` (Vovk's β-binomial bound), so the paper can state *"for every Fitzpatrick group, coverage ≥ 1−α−ε_g with ε_g explicit at n_g."*

**Why novel.** Mondrian conformal (Vovk) and clustered conformal (Ding 2023, already cited in the code) exist, but *hierarchical-shrinkage Mondrian conformal calibrated on a protected attribute with an explicit small-sample coverage certificate per subgroup* is not standard, and nobody has done it for skin tone. It converts the project's biggest weakness (underpowered dark subgroups) into a headline guarantee.

**Laptop-feasible.** Pure numpy on the existing `preds.npz` + a Fitzpatrick column. Zero GPU. Could be run tonight from cached artifacts.

**Validation experiment.** Recompute coverage per tone before/after, plus a coverage-vs-n curve; bootstrap the per-group coverage CI.

---

### IDEA 2 — `FrozenAudit`: a null-calibrated, severity-graded leakage-audit protocol (reusable standalone contribution)

**Problem it solves.** The 33% leakage catch used an arbitrary cosine > 0.99 threshold. That threshold is unprincipled: how do you know 0.99 isn't dropping legitimate look-alikes, or missing near-dupes at 0.97? For the finding to be a *reusable contribution* (the task explicitly asks for this), the threshold must be calibrated, not guessed.

**The invention.** A three-part protocol packaged as a tool:
1. **Null calibration of the duplicate threshold.** Estimate the within-dataset distribution of nearest-neighbor cosine among images *known* to be distinct (same source, different patients), then set the duplicate threshold at a controlled false-positive rate against that null — turning "0.99" into "the cosine at which P(distinct pair exceeds it) < 0.1%."
2. **Severity grading.** Exact (>0.995) / near-duplicate (augmentation or re-scan, 0.97–0.995) / semantic twin (same lesion, different view). Each grade gets a different treatment (drop / down-weight / group-merge).
3. **Leakage-adjusted performance interval.** Instead of a single de-leaked number, report the *band* [de-leaked, leaked] with the fraction removed — an honesty interval that any multi-source medical paper should carry.

**Why novel.** Duplicate detection exists (SSIM, pHash, embedding-NN), but a **null-calibrated, severity-graded, foundation-feature audit that outputs a leakage-adjusted performance interval as a standard reporting object** is a genuine methodological contribution with a clean name and a reusable script. This is the most *citable* single artifact the project could ship.

**Laptop-feasible.** Cosine on cached features + a same-source/different-patient sampler for the null. No GPU.

**Patentable?** Weak as a patent (it's a QA methodology), stronger as an open protocol/standard. Flagged non-patentable.

---

### IDEA 3 — Tone-Conditional Feature-Space Alignment (TCFA): break the 0.78 dark-skin ceiling WITHOUT new data or backbone training

**Problem it solves.** The DDI V–VI result (0.789) is stated as a *data ceiling*. But the frozen-probe insight says fine-tuning fails because it moves the features; it does NOT say the frozen features are optimally *used*. Dark-skin images sit in a shifted region of DINOv2 feature space relative to the (majority light-skin) probe-training distribution. The probe is applied to mis-aligned features.

**The invention.** A per-Fitzpatrick **affine whitening / CORAL alignment in frozen-feature space** applied at inference — no backbone gradient step:
- Estimate the per-tone feature mean/covariance from *unlabeled* dark-skin images.
- Learn a per-tone affine map `A_g, b_g` that aligns tone-g feature statistics to the probe's training-feature statistics (CORAL / whitening-recoloring), or a lightweight optimal-transport map.
- At inference, transform `f → A_g f + b_g` before the frozen head.

This is domain adaptation done entirely on frozen features (cheap, label-free for the alignment part), which is exactly the regime the project already proved is right. It is the natural next move *implied* by the frozen-probe thesis but not yet taken.

**Why novel.** CORAL/feature whitening exist; applying **per-skin-tone feature alignment on a frozen foundation backbone as the fairness-adaptation mechanism for dark-skin cancer** is new and directly tests whether 0.78 is a data ceiling or a *feature-usage* ceiling. Either result is publishable: a gain breaks the ceiling; a null result *strengthens* the "genuinely data-limited" claim with a much stronger negative control than "we tried synthetic data."

**Laptop-feasible.** numpy/sklearn on the cached DDI feature `.npz`, per-fold. No GPU, no backbone training.

**Patentable?** Yes — a concrete inference-time, tone-conditional feature-space transform for a frozen medical classifier is a specific, novel, useful method.

---

### IDEA 4 — Unified frozen-feature "epistemic substrate": one geometry drives OOD + abstention + question selection

**Problem it solves.** OOD (Mahalanobis distance), abstention (conformal set size + margin), and the LLM interview gate are three separate, hand-tuned thresholds (`>4`, `<0.15`, p95). They are all functions of the *same* point in DINOv2 space, yet they're computed and tuned independently, and none informs the others.

**The invention.** A single per-input **epistemic coordinate** in frozen space that factorizes uncertainty into two geometrically-defined axes and routes every downstream decision:
- **Aleatoric axis** = local label disagreement among the k nearest labeled training features (kNN entropy in frozen space).
- **Epistemic/OOD axis** = distance to the training manifold (the existing Mahalanobis score).
- A 2-D gate: high OOD → refuse ("not a lesion I know"); low OOD + high local entropy → *interview* (history can help); low OOD + low entropy → decide. This replaces three thresholds with one interpretable map.

Crucially, this makes **question selection information-theoretic** (Idea 5 below) instead of blind.

**Why novel.** Decomposing uncertainty into kNN-aleatoric vs Mahalanobis-epistemic *in a foundation-feature space* and using that 2-D coordinate to jointly route refusal / abstention / human-vs-LLM escalation is a novel unification. It also yields a free interpretability feature: the k nearest training lesions are a **retrieval-based second opinion** ("your case looks most like these 5 confirmed cases").

**Laptop-feasible.** kNN + existing Mahalanobis on cached features. No GPU.

**Patentable?** Yes — a system that decomposes model uncertainty into two feature-geometry axes to route between autonomous decision, human referral, and adaptive LLM interview.

---

### IDEA 5 — Expected-Information-Gain question selection for the LLM interviewer (photo-grounded → posterior-grounded)

**Problem it solves.** `propose_questions` currently asks the LLM to pick "the most discriminating questions" purely from the differential text + photo. The image model's *posterior over the conformal set* — the thing we actually want to collapse — never enters the selection objective. The LLM guesses at discriminativeness; it doesn't optimize it.

**The invention.** Turn question selection into greedy **Bayesian experimental design**: for each candidate question, the LLM predicts the answer-conditional log-LR vector (it already does this in `fuse_history`); select the question maximizing expected reduction in posterior entropy over the *current conformal set* (not all 25 classes — only the live differential). This closes the loop: the classifier's uncertainty defines the objective, the LLM supplies the likelihood model, and the selected question provably targets the actual ambiguity.

**Why novel.** Expected-information-gain / active-inference question selection is known in dialog systems; **grounding the information-gain objective in a conformal prediction set from a frozen-feature image classifier, with the VLM as the answer-likelihood oracle** is a new synthesis and a clean safety story (questions can only be chosen to reduce *certified* set ambiguity).

**Laptop-feasible.** The scoring loop is tiny; needs the LLM API for the likelihood predictions (respect the no-network constraint — design + offline simulation now, live later). Can be validated offline with synthetic answer distributions.

**Patentable?** Yes — a method for selecting patient-history questions by maximizing expected information gain over a conformal prediction set produced by a frozen-feature classifier.

---

### IDEA 6 — Certified LLM-evidence fusion: the text channel can never override a cancer flag beyond a calibrated, bounded margin

**Problem it solves.** In `fuse_history`, the posterior is `log(prior) + LR`, LR ∈ [-2,2] straight from the LLM, *uncalibrated*. A hallucinated reassuring history ("it's been stable for years") can currently drop a malignant class's posterior by up to e² ≈ 7.4×. The system's safety claim ("never override a cancer") is enforced only by a prompt, not by math.

**The invention.** Two mechanisms:
1. **Calibrated LR shrinkage via self-consistency.** Sample the LLM's LR vector a few times; shrink each LR toward 0 in proportion to its variance (inconsistent evidence ⇒ weak evidence). This is conformal-flavored reliability weighting of a language model's numeric output.
2. **A certified safety clamp.** Cap the *net* downward log-LR on any malignant class so that if the image channel's calibrated malignant probability exceeds a threshold, no amount of history can push it below the referral line. State it as a theorem: melanoma referral sensitivity is preserved ≥ 0.90 regardless of the text channel.

**Why novel.** LLM-as-evidence Bayesian fusion is emerging, but **self-consistency-calibrated, formally-bounded LLM evidence with a certified floor on a safety-critical class** is a new and very publishable safety mechanism for multimodal medical AI. It converts a prompt-based promise into a proof.

**Laptop-feasible.** The clamp is pure arithmetic (works now, offline). The self-consistency part needs the API (design now).

**Patentable?** Yes — strongly. A method for bounding a language model's influence on a safety-critical classifier output with a calibrated, certified margin.

---

## 2. Additional novel ideas (shorter)

### IDEA 7 — Prevalence-adaptive deployment recalibration (interdisciplinary: epidemiology × conformal)
The per-class logit bias is optimized once on the training mix. Real clinics have different disease prevalence (a Mumbai STI clinic ≠ a rural teledermatology line). Invention: a **one-parameter-per-class prior-shift** applied to the frozen head at deployment using only local base rates (no images, no labels, no retraining) — Saerens/prior-shift correction wired into the deployable `.pkl`. Turns the frozen head into a *site-configurable* classifier. Feasible (arithmetic on logits). Patentable as a concrete deployment method.

### IDEA 8 — Equalized-risk selective prediction (fairness-constrained abstention)
The abstention curve (refer 30% → 77% retained accuracy) is computed marginally. Invention: choose the abstention threshold *per Fitzpatrick group* so that **post-referral accuracy is equalized across tones** (equal opportunity under abstention), rather than referring a uniform 30% that leaves dark skin worse-served. Feasible on existing preds. Pairs with Idea 1.

### IDEA 9 — Retrieval-augmented, leakage-graph-aware honest CIs
The same cosine-NN duplicate graph (Idea 2) defines connected components of near-identical images. Invention: **cluster-bootstrap** the confidence intervals over those components instead of over rows, so CIs are not inflated by residual near-duplicates. A rigor upgrade that also uses the leakage machinery for a second purpose. Feasible.

### IDEA 10 — Feature-space active-learning coreset for the binding constraint (dark-skin labels)
Dark-skin labeled data is *the* bottleneck. Invention: use frozen features + the duplicate graph to run **tone-stratified facility-location coreset selection** over unlabeled dark-skin images, ranking which few hundred images, if labeled, would most improve the probe — a concrete, budgeted data-acquisition plan derived entirely from frozen geometry. Feasible offline; turns "we need more data" into "label exactly these 200."

---

## 3. Interdisciplinary angles
- **Epidemiology / Bayesian nowcasting** (Idea 7): local prevalence as a deployable prior; ties derm-AI to public-health surveillance.
- **Econometrics / decision theory**: set the abstention and referral thresholds by expected clinical utility (cost of missed melanoma vs cost of a referral), making the operating point a health-economics object, not a hyperparameter.
- **Statistical genetics-style shrinkage** (James-Stein / empirical Bayes) imported into conformal calibration for small subgroups (Idea 1).
- **Optimal transport** for the tone-conditional feature alignment (Idea 3).
- **Optimal experimental design / active inference** for the interview (Idea 5).

---

## 4. Validation experiments that would substantially strengthen the work
(All laptop-feasible from cached `.npz` / existing `preds.npz`; none touch the running ensemble or feature extraction.)
1. **Per-tone conformal coverage table** (Idea 1) — before/after Fitzpatrick-Mondrian; coverage-vs-n curve; the single most convincing new figure.
2. **Leakage-threshold sensitivity sweep** (Idea 2) — macro-F1 vs cosine cut 0.95→0.999, with the null-calibrated FP-rate marked; report the leakage-adjusted interval.
3. **TCFA ablation on DDI folds** (Idea 3) — frozen probe vs +per-tone alignment, per fold, per FST stratum. Ceiling-breaker or strong negative control.
4. **kNN vs Mahalanobis OOD/abstention comparison** (Idea 4) — does the 2-D epistemic map dominate the single thresholds on synthetic OOD + held-out?
5. **Certified-clamp stress test** (Idea 6) — adversarially reassuring histories on confirmed malignant cases; show sensitivity is provably preserved with the clamp and NOT without it (a dramatic before/after safety result).
6. **Cluster-bootstrap CIs** (Idea 9) — show how much the naive row-bootstrap CIs were over-optimistic.
7. **Prevalence-shift robustness** (Idea 7) — simulate 3 clinic prevalence profiles, show prior-shift recovers accuracy with zero retraining.

---

## 5. What turns this from a solid paper into a field-defining contribution

The current paper is *"we found frozen-probe beats fine-tuning, and we were rigorous."* Solid, but its trust layer and fairness story are a **checklist of separately-applied standard tools**. The field-defining reframing:

> **A single frozen foundation-feature geometry, turned into a fairness-CERTIFIED decision-support instrument** — where prediction, uncertainty set, OOD refusal, human-vs-LLM escalation, the interview questions, and the bound on the LLM's influence are all derived from that one geometry and all carry an explicit, finite-sample, *per-skin-tone* guarantee.

Concretely: ship **Idea 1 (per-Fitzpatrick certified coverage) + Idea 4 (unified epistemic substrate) + Idea 6 (certified LLM clamp)** as one framework. The claim becomes not "we did calibration and conformal and fairness" but **"for every skin tone, we can state a guaranteed coverage and a guaranteed melanoma-referral sensitivity that no channel — image, text, or subgroup size — can violate."** That is a *certificate*, not a metric. In a field where dark-skin AI repeatedly fails silently, a per-tone guarantee that provably cannot be silently violated is the contribution people cite. The frozen-probe finding is the engine; the fairness certificate is what makes it matter.

The second-order field-defining asset is **`FrozenAudit` (Idea 2)** released as a standalone, null-calibrated leakage-audit standard — the kind of tool that gets adopted across the whole multi-source medical-imaging community independent of this paper's specific numbers.
