# MNKAI-III — Plan: A Trained, Photo-Grounded Clinical Interviewer

Goal: a model that **looks at the lesion photo**, and *when the image classifier is unsure*, asks the few most
discriminating history questions, then fuses the answers with the image to sharpen the diagnosis — safely, as
decision support. Not prompt-engineering alone: a genuinely **trained vision-language model (VLM)**.

## Design principles
- **Confidence-gated:** the interviewer only fires when the image probe is uncertain (large conformal set / low
  margin). Confident cases → answer directly. (We now have the conformal + abstention machinery for this gate.)
- **Photo-grounded:** questions depend on what the lesion *looks like* (scaly → ask itch/chronicity; pigmented →
  ask change/ABCDE). This requires the model to see the image, not just the class list.
- **Image model stays the anchor:** the LLM proposes questions + weighs history as evidence; the calibrated
  classifier makes the decision. The LLM never overrides a cancer flag.
- **Safe:** red-flag escalation, abstain/refer, "not a diagnosis" framing.

## Three-version roadmap (ship value early, train properly after)
| Version | What | Trains? | Effort | Needs |
|---|---|---|---|---|
| **v0 (have)** | Gemini reads the *differential* → questions → Bayesian fusion | no | done | Gemini key |
| **v1 (next, days)** | Send the **photo** to Gemini-2.5 (it's a VLM) → *photo-grounded* questions, gated on low confidence | no | ~2–4 days | Gemini key |
| **v2 (trained, weeks, A100)** | A **fine-tuned compact VLM** specialized for the derm interview — local, no API, distilled | **yes** | ~3–4 weeks | Colab A100 |

v1 is the fast product win and generates the dialogues we'll distill for v2.

## v2 architecture (aligns with our frozen-probe thesis)
Keep the vision encoder **frozen** (the pattern that already won), train only a small bridge + adapter:
```
photo ──► [FROZEN DINOv2/derm-backbone] ──► image features ──► [Projector (trainable)] ──► image tokens
                                                                                              │
image-classifier differential + conversation history ──► text tokens ──────────────► [ Small LLM ] ──► question OR fused differential
                                                                                        (LoRA-tuned; Qwen2.5-1.5B/3B or Phi-3-mini)
```
- Only the **projector + LoRA adapters** train (~1–2% of params) → fits an A100, fast, and preserves the strong
  frozen vision features.
- Output modes: (a) the next best question, (b) the refined differential once enough is known.

## Training data (the real bottleneck — two sources, combined)
1. **SCIN-grounded supervision (real):** SCIN has image + structured symptom fields (itch, duration, spread,
   body-site, systemic symptoms) + weighted dermatologist differential. Convert each case → a dialogue: image →
   model asks the discriminating questions → the symptom values are the answers → the differential is the target.
   ~5–10k *real* grounded interview examples.
2. **Teacher distillation (scale):** for each labeled derm image (with its true diagnosis), have a strong teacher
   VLM (Gemini-2.5-pro) generate a short 2–3-turn photo-grounded interview + rationale + fused diagnosis. Distil
   into the student. Scales to tens of thousands. Filter teacher outputs against the ground-truth label.
3. **Gating labels:** train on ambiguous cases (image-alone conformal set large) so the model learns *when* to ask
   vs answer.

## Training recipe (stages)
1. **Align** (freeze vision + LLM): train the projector on image → caption/differential (grounds the image tokens).
2. **Instruction-tune** (LoRA on LLM): the interview dialogues — ask discriminating questions, then fuse.
3. **Preference-tune (optional, DPO):** prefer questions with higher information-gain / that flip the differential;
   penalize unsafe or non-discriminating questions.
4. **Fusion head:** keep the calibrated Bayesian update (image prior × history log-LR) so the final probability is
   calibrated — the LLM outputs evidence weights, not the final label.

## Confidence gate + fusion (unchanged, proven)
- Gate: invoke the interviewer iff image-probe conformal set size > k OR top-1 margin < τ (from the trust layer).
- Fuse: posterior ∝ image_prior × Π(history likelihood ratios); image model remains the decision-maker.
- Stop: when the conformal set shrinks below k or after N questions → output differential + referral flag.

## Safety
Red-flag detector (bleeding/rapid-growth/systemic → urgent referral) always overrides; never contradicts a
cancer flag; abstains and defers when uncertain; every output framed as decision-support.

## Evaluation (does it actually help?)
- **Primary:** macro-F1 / top-1 on held-out SCIN cases, **image-only vs image+interview** — the measurable lift.
- **Ablations:** photo-grounded vs differential-only questions; confidence-gated vs always-ask.
- **Quality:** clinician rating of question relevance + safety; hallucination rate; over-referral rate.
- **Efficiency:** average #questions to a confident answer.

## Compute + timeline (Colab A100)
- v1: no training, ~2–4 days (prompting + gating + app wiring), generates distillation dialogues.
- v2: data build ~1 wk; align+LoRA instruction-tune ~a few days on one A100; eval+DPO ~1 wk. **~3–4 weeks total.**
- Backbone frozen → cheap; only projector + LoRA train. A 1.5–3B LLM keeps inference laptop-deployable (ONNX/GGUF).

## Honest notes
- A strong medical VLM is ambitious; **v1 delivers most of the product value now**, and de-risks v2 by producing
  training dialogues. Commit to the v2 training run only after v1 shows the interview lifts accuracy.
- The interview's real payoff is exactly on the **weak tail** (contact-derm, drug-eruption, urticaria, scabies) and
  **low-confidence dark-skin** cases — where history is decisive and the image alone is unreliable. This directly
  complements the frozen-probe classifier's weak spots.
```
Ship order: v1 (photo-grounded Gemini + gate + app) → collect dialogues → v2 (distilled compact VLM).
```
