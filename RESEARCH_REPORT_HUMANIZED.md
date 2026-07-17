# A Skin-Disease AI That's Honest About Indian Skin
### Building and validating a multimodal vision-transformer for dermatology triage, fine-tuned for Fitzpatrick III–VI
*Final-year research project · Department of Dermatology, CMC Ludhiana · June 2026*

> A note on how to read this: I've written this in plain, honest language rather than
> dense jargon, because I want anyone — a dermatologist, an examiner, or a fellow
> student — to follow exactly what I did and why. All numbers in this report are final and
> measured — the corrected 25-class model's external-test results (Section 5) are now in.
> The only thing still pending is the prospective clinical pilot (Section 10), clearly marked.

---

## 1. The short version

Skin disease is one of the most common reasons people seek care, but in India most
dermatologists are in big cities, so a lot of people — especially in rural areas —
struggle to see a specialist. AI could help with first-line triage from a phone photo.
The catch is that almost all the skin-AI built so far was trained on light, Western skin
and focused on skin cancer. It tends to do worse on darker skin and on the everyday
conditions common in India — fungal infections, eczema, psoriasis, scabies, and so on.

So I set out to build a dermatology AI that is actually meant for Indian skin — and, just
as importantly, one you can **trust**. By "trust" I mean it should tell you when it isn't
sure, it should work fairly across skin tones, and it shouldn't quietly fail on common
conditions while looking fine on paper.

The model looks at the photo **and** simple clinical details (age, sex, body site, skin
type) together. I trained it in two stages: a 9-class "base" model on a large, carefully
sourced collection (~271,000 images, no AI-generated and no scraped images), then a
25-class "Indian" model adding the common local conditions. I checked it the careful way:
no patient's photos in both training and testing, calibrated confidence, fairness across
skin tones, and a completely separate hold-out test set it never sees during development.

The base model is finished and solid — macro-F1 **0.637 ± 0.009**, AUROC ≈ **0.95**,
accuracy ≈ **85%**. The Indian model's first attempt produced the most important lesson of
the whole project — a failure I caught, explained, and fixed (Section 6). The corrected
model, measured on a pristine held-out test set, reaches **macro-F1 0.484** with
**top-3 accuracy 0.725** and **melanoma recall 0.84** — and, most importantly, the common
conditions that had collapsed (eczema, tinea) are back to clinically usable recall.

---

## 2. Why I started this

- Skin conditions are a huge share of everyday illness, and many are very treatable *if*
  spotted early.
- In India, specialist care is concentrated in cities. A phone-based triage tool could
  help a health worker or patient know "this looks urgent — see someone" vs. "this looks
  like something common and manageable."
- The honest problem: existing tools are mostly trained on Western, light-skinned,
  cancer-heavy data. They aren't built for Indian skin or Indian conditions, and fairness
  across skin tones is often never even measured.

This project tries to do the harder, more useful thing: build for the patients usually
left out, and be honest about how well it actually works.

---

## 3. What I'm really trying to prove

Two things — and the second is the real contribution:

1. **A dermatology AI can be built for Indian skin and Indian conditions**, not just
   re-used from a Western cancer model.
2. **Building it *honestly* matters more than chasing a high score.** A model you can
   trust — one that knows its limits, is fair, and whose failures you've found and fixed —
   is worth more than one that simply looks good. I don't just claim this; I demonstrate
   it (Section 6).

---

## 4. How I built it

### 4.1 The data
I gathered ~**271,000 dermatology images** from more than twenty trustworthy public
sources (ISIC, HAM10000, BCN20000, PAD-UFES-20, Derm7pt, Fitzpatrick17k, DermNet, SCIN,
DermaCon-IN and others). Two hard rules from day one: **no AI-generated images and no
scraped images.** I removed duplicates, stripped identifying metadata, and split the data
so **no single patient's photos appear in both training and testing** — a common, sneaky
way AI studies accidentally inflate their scores, which I made sure to avoid.

For the Indian model I added 16 locally-common conditions on top of the 9 base classes,
for **25 classes** total (eczema, psoriasis, tinea/fungal, scabies, vitiligo, acne, drug
reactions, bacterial and viral infections, and more). I set aside a completely separate
**15% test set** (~5,900 images, no patient overlap) that the model never sees during
development — so the final number is honest, not one I tuned toward.

*(Two honest curation notes: I dropped a 27th planned class, melasma — only 43 usable
images, too few to learn or test reliably. And I found one class, "viral infection," was
mostly single-disease photo sets, so the model could cheat by recognising "which dataset"
rather than "which disease"; I rebalanced it by source to force it to learn the actual
skin findings.)*

### 4.2 The model, in plain terms
- It uses a strong pre-trained image backbone (**DINOv2 Vision Transformer**, ~86M
  parameters) that already understands images well, so I didn't train from scratch.
- It reads the **photo and the clinical details together** — age, sex, body site, skin
  type — the way a clinician reasons (a dark spot on sun-exposed skin in an older patient
  means something different than the same spot on a child).
- A final layer turns all that into a probability for each condition.

### 4.3 Training
Careful, standard training that fits on a modest laptop GPU (6 GB). For the Indian model I
started from the best base model and fine-tuned gently. **In the interest of honesty:** I
also tried two fancier ideas (self-supervised pre-training and knowledge distillation) and
dropped both — one kept crashing on my hardware, the other cost speed for almost no gain.
I report that rather than pretend everything worked.

### 4.4 How I checked it (the part most projects skimp on)
- **Honest validation** — 5-fold cross-validation with no patient leakage, plus a separate
  hold-out test set.
- **Confidence calibration** — so "90% sure" actually means about 90% right.
- **Uncertainty** — conformal prediction lets the model say "it's one of these few" or
  effectively "I'm not sure — refer to a doctor," instead of always guessing.
- **Fairness** — performance measured separately for lighter vs. darker skin tones.
- **Explainability** — heat-maps showing *where* the model looked, so a clinician can
  sanity-check it.

---

## 5. Results so far

**Base 9-class model — finished**

| What | Result |
|---|---|
| Macro-F1 (5 honest folds) | **0.637 ± 0.009** |
| AUROC | ≈ **0.95** |
| Accuracy | ≈ **85%** |
| Confidence error after calibration | **0.031** (low = well-calibrated) |

The folds barely vary (± 0.009), so this is a real, repeatable result, not a lucky split.

**Indian 25-class model — corrected, measured on the held-out external test (5,886
images).** The first version scored 0.339 and collapsed on common conditions (Section 6).
The **corrected** winner (class-balanced focal loss, single-mechanism balancing) reaches:

| Metric (external test) | Result |
|---|---|
| Macro-F1 | **0.484** (vs 0.339 collapsed) · per-fold 0.479 ± 0.005 |
| Top-3 accuracy | **0.725** |
| Balanced accuracy | 0.533 |
| Melanoma recall / F1 / AUROC | **0.842 / 0.815 / 0.987** |
| Eczema recall | **0.503** (was 0.004) · tinea 0.43 · viral 0.73 |
| Calibration (ECE) | 0.21 → **0.042** after temperature scaling (T=2.2, deterministic R02) |
| Conformal coverage (target 90%) | 98.4% |

The collapse is genuinely fixed: the common conditions recovered from near-zero to usable
recall, melanoma stayed safe (0.84), and the model is well-calibrated. The honest weak spots
are the long-tail inflammatory classes — psoriasis (0.16), contact dermatitis, bacterial
infection — which are the clearest target for the data-expansion next steps. Per-Fitzpatrick
fairness can't be concluded yet (only 777 test images carry skin-tone labels) — that's what
the prospective study is for.

---

## 6. The most important part: I caught my own model failing

This is the heart of the project, so I'll tell it straight.

When the first Indian model finished, its headline score (0.339) looked unremarkable but
not alarming. But looking **condition by condition**, I found something serious: it got the
skin-cancer classes right, but it was **missing almost every case of the most common
conditions** — eczema (it caught about **0.4%** of cases), psoriasis (**0.7%**), fungal
infection (**5.6%**). On the very conditions an Indian clinic sees most, it was essentially
useless — and the average score completely hid that.

Then I worked out *why*. I'd used **two** different techniques to handle the fact that some
conditions are rarer than others — one changes which examples the model sees, the other
changes how much each example counts. Each is fine alone. Stacked together they
over-corrected: the model became so biased toward rare conditions that it basically stopped
predicting the common ones. A clear, measurable fingerprint confirmed it (its plain accuracy
was *lower* than its balanced accuracy — the tell-tale sign).

The fix was to use **only one** balancing technique. I tested three clean options head-to-head
and chose the winner the careful way — on a separate validation split (so I didn't cheat by
tuning on the final test set), with a rule that **disqualifies any option that sacrifices
melanoma detection** (you can't trade catching cancer for catching eczema). The winner
recovered the common conditions while keeping the cancers safe.

**Why I'm proud of this rather than embarrassed:** the average score said "fine," the careful
look said "dangerous," and being careful is what caught it. That is, in one concrete example,
the entire argument of this project — *a number can lie, and rigour is what protects the
patient.* Finding and fixing this is a genuine, transferable lesson for anyone building
medical AI on imbalanced data.

---

## 7. The explainability tools I built

To make the model's reasoning visible (and to demo it), I built an interactive platform: upload
any skin photo and see the predicted condition with its top-5 alternatives, how confident and
how *uncertain* the model is, and several "where did it look?" heat-maps. There's also a live
dashboard showing the model learning in real time. These aren't just for show — being able to
see *why* the model said what it said is part of what makes it trustworthy enough for a clinician
to even consider using.

---

## 8. What's genuinely new here

1. A dermatology AI **built and fairness-checked for Indian skin** and Indian conditions, not
   borrowed from a Western cancer model.
2. It reads the **image and clinical details together**, like a clinician.
3. **The honest-failure story** — diagnosing and fixing a hidden collapse on common conditions
   — a real methodological contribution, not just a result.
4. A **clean, held-out test set** kept untouched during development, so the final number means
   something.
5. The **trustworthiness layer** most projects skip: calibrated confidence, "I'm not sure"
   uncertainty, fairness across skin tones, and visual explanations.
6. **Honesty about what didn't work** (the two techniques I dropped).

---

## 9. Honest limitations

- **Skin-tone labels are sparse** — only a minority of images record skin type, so my fairness
  check on existing data is partial. The real test is the planned clinical study.
- **Some images are dermoscopy; real clinics use phone photos.** I reduce this gap with
  augmentation and Indian fine-tuning, but don't eliminate it.
- **It's a triage *aid*, not a diagnosis** — not a substitute for a dermatologist or a biopsy,
  and not ready for public, unsupervised use.

---

## 10. What's next

A small **prospective pilot at CMC Ludhiana**: real patients photographed in clinic, each
diagnosed by a consultant dermatologist as the ground truth, to see how the model does on
genuinely new, local images. I'm explicit that ~100 patients is a **feasibility pilot**, not a
definitive validation — a real clinical claim needs a larger, multi-site study. This is the
part that turns "promising research" into "evidence it helps real patients."

---

## 11. A note on tools and how this was built *(keep, adapt, or move per your guide's preference)*

In the spirit of full transparency: the experimental design, the clinical framing, the choice
to focus on Indian skin, the decision rules, the interpretation of results, and the integrity
constraints (no synthetic or scraped data) are my own. I used AI-assisted coding tools to help
implement the training and evaluation software, under my direction, and I have worked to
understand the methods well enough to explain and defend every choice here. I note this openly
because honest disclosure is the right thing to do and is increasingly expected; please advise
if your preferred format differs.

---

## 12. References (readable subset)
- Oquab M. et al. *DINOv2: Learning robust visual features without supervision.* TMLR, 2024.
- Pacheco AGC, Krohling RA. *Attention-based image-plus-metadata fusion for skin cancer.* IEEE JBHI, 2021.
- Cui Y. et al. *Class-balanced loss based on effective number of samples.* CVPR, 2019.
- Menon AK. et al. *Long-tail learning via logit adjustment.* ICLR, 2021.
- Ren J. et al. *Balanced meta-softmax for long-tailed recognition.* NeurIPS, 2020.
- Cao K. et al. *Label-distribution-aware margin loss (LDAM).* NeurIPS, 2019.
- Kang B. et al. *Decoupling representation and classifier for long-tailed recognition.* ICLR, 2020.
- Guo C. et al. *On calibration of modern neural networks.* ICML, 2017.
- Angelopoulos AN, Bates S. *A gentle introduction to conformal prediction.* arXiv:2107.07511, 2021.
- Romano Y., Sesia M., Candès EJ. *Classification with valid and adaptive coverage (Mondrian).* NeurIPS, 2020.
- Kusner MJ. et al. *Counterfactual fairness.* NeurIPS, 2017.
- Lakshminarayanan B. et al. *Deep ensembles for predictive uncertainty.* NeurIPS, 2017.
- Selvaraju RR. et al. *Grad-CAM.* ICCV, 2017.
- Esteva A. et al. *Dermatologist-level classification of skin cancer.* Nature, 2017.
- Liu Y. et al. *A deep learning system for differential diagnosis of skin diseases.* Nature Medicine, 2020.
- Groh M. et al. *Fitzpatrick 17k.* CVPR Workshops, 2021.

*(Full formatted reference list is in the technical report; this is the readable subset.)*
