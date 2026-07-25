# MNKAI-III — Plain-Language Summary

**An artificial-intelligence tool that reads photographs of skin conditions, and knows when to send the patient to a dermatologist.**

*Prepared for clinical and institutional review. This summary avoids technical language. Every figure in it matches the full technical manuscript.*

Ujjwal Thakran · Christian Medical College, Ludhiana

---

## 1. The problem

Skin disease is one of the commonest reasons a patient walks into an Indian outpatient clinic, and there are very few dermatologists per head of population. Software that can look at a photograph and suggest what a lesion might be is therefore attractive — especially for triage, where the real question is often simply *"does this person need a specialist, and how soon?"*

There is a serious catch. Almost all of the world's skin-image collections come from Western hospitals and contain mostly **light skin**. When independent researchers tested three published skin-cancer programs on brown and dark skin, the results were poor. On a scale where 0.5 is the same as tossing a coin and 1.0 is perfect, the three programs scored **0.50, 0.55 and 0.57** on the darkest skin types. One of them was, literally, no better than guessing.

That is the population we serve. A tool that works on Northern European skin and fails on Punjabi skin is not a useful tool — it is a liability. Cancers on brown skin are also easy to miss for other reasons: melanoma in Indian patients often appears on the palms, soles and nail beds rather than sun-exposed areas, and is frequently found late.

There is a second problem, which is about trust rather than accuracy. Most medical AI gives you an answer and nothing else. It does not tell you how confident it is, it does not say when it is out of its depth, and it does not report whether it performs worse for some patients than others. A clinician cannot safely use a tool that is always confident.

---

## 2. What we built

A decision-support tool that a doctor can run on an ordinary laptop. Given a photograph of a skin lesion, it does four things:

1. **Suggests the likely conditions** — a short ranked list, not a single guess, across 25 common skin conditions.
2. **Flags possible cancer** — a separate, deliberately cautious estimate of whether the lesion may be malignant.
3. **Says when it is unsure** — and recommends referral to a dermatologist instead of guessing.
4. **Asks questions when it is unsure** — the tool can look at the photograph and ask the patient relevant history questions ("has it changed shape recently?", "does it bleed?"), then revise its opinion in light of the answers.

Two points about how it is built matter for approval, and neither is technical.

**It runs on ordinary hardware.** The whole system runs on a single laptop. There is no expensive server, no cloud subscription, and no ongoing computing cost. This was a deliberate design constraint: a tool that requires a data centre cannot be deployed in the settings that need it most.

**We did not build the "vision" from scratch.** We took several large, publicly available image-understanding programs — already trained by major research laboratories on enormous collections of general photographs — and left them **completely unchanged**. On top of them we trained a small, lightweight decision layer using medical images. This turned out to work better than the conventional approach of retraining the whole program on our own data, which actually damaged its performance when medical data was limited. Doing less worked better than doing more.

A useful way to picture the tool: it consults **six different expert readers at once** and takes their consensus. Each of the six was trained differently, so each makes different mistakes; averaging their opinions cancels out much of the individual error. It is the same logic as a tumour board, where specialists trained at different institutions review the same case and the consensus is more reliable than any one opinion.

---

## 3. How it was tested

**The training material.** We assembled about **266,000 labelled skin images** from eight public research collections. Of these, roughly **15,000** were of darker skin — and that scarcity is the central difficulty in this entire field.

**The cancer test.** For the crucial question — can it detect skin cancer on dark skin? — we used an internationally recognised benchmark called **DDI**: 656 photographs from a United States clinic in which **every single diagnosis was confirmed by biopsy**. This is the gold standard: not a doctor's impression, but a pathologist's examination of the actual tissue. It is also, at present, essentially the only public collection of its kind anywhere in the world.

**What "tested honestly" means.** This deserves emphasis, because it is where a great deal of published medical AI quietly goes wrong. Partway through, we discovered that roughly a third of our test images were near-duplicates of images the system had already been shown during training. The tool was, in effect, being examined on questions it had already seen. We built a method to detect this, removed the affected images, and **re-reported our own headline result downwards** — from about 0.63 to 0.57 on the 25-condition task. Nobody required us to do this. We also report confidence ranges on every figure rather than single flattering numbers, and we ran a control experiment which showed that one of our early explanations for *why* the method worked was wrong, and corrected the claim.

---

## 4. What we found

### Skin cancer on dark skin — the headline result

| | Score (0.5 = coin toss, 1.0 = perfect) |
|---|---|
| Three published programs (tested by independent researchers) | 0.50, 0.55, 0.57 |
| **Our tool, on the darkest skin types** | **0.872** (realistic range 0.81 – 0.93) |

In practical terms: at a setting where the tool correctly clears **80%** of harmless lesions, it catches about **three out of every four cancers** on dark skin. That is a genuinely useful triage performance, and it is at the level of the best results published for this benchmark.

One honest qualification: the published programs were tested "cold," having never seen this collection, whereas our tool was trained and tested within it using a standard rotation method. So the fair statement is *"our approach reaches the best achievable level on this benchmark,"* not *"we beat those programs by that margin."*

### Fairness — the gap disappears

Every published program tested on this benchmark performs **worse on dark skin than light skin**, by 0.09 to 0.15 points. Our tool does not show that penalty: it scored 0.78 on the lightest skin, 0.83 in the middle range and 0.87 on the darkest.

We are deliberately careful here. Because these groups are small, the statistical ranges overlap, so the correct claim is that performance was **comparable across skin tones with no significant disparity** — *not* that the tool is "better on dark skin." Overstating this would be exactly the kind of claim we have tried to avoid.

### The 25 common conditions

On the broader task of naming which of 25 skin conditions is present, the tool is right far more often than chance but is **not** at specialist level. It is best understood as a triage and second-opinion aid. We must also state plainly that our ability to measure its performance *specifically on dark skin* for these 25 conditions is weak: the test set contained only about **135 darker-skin images spread across 25 conditions** — roughly five per condition. That is not enough to make a reliable claim, and we do not make one. **This specific weakness is the reason for the proposed second study.**

---

## 5. Why it can be trusted

Four safety behaviours are built in — and most published dermatology AI reports none of them.

| Behaviour | What it means in practice |
|---|---|
| **Honest confidence** | When the tool says it is 70% sure, it is right about 70% of the time. Its confidence was badly overstated before correction and is now well-calibrated. |
| **A short list, not one answer** | Instead of a single guess, it gives a small list that contains the correct diagnosis about 90% of the time — typically 3 to 4 conditions out of 25. |
| **Knowing when to refer** | If the least-confident 30% of cases are handed to a dermatologist, accuracy on the remainder rises substantially. The tool defers rather than guessing. |
| **Recognising the unfamiliar** | It flags images that are unlike anything it was trained on, rather than forcing them into a category. |

There is also a hard safety rule in the question-asking feature: the patient's answers can **raise** concern about cancer, but can never **clear** a cancer flag. History can escalate; it cannot reassure.

---

## 6. What this is *not*

We want the committee to see the limits as clearly as the results.

- **It is decision support, not diagnosis.** It assists a clinician; it does not replace one, and it makes no autonomous decision.
- **It has not been tested on our own patients.** All validation so far is on existing research collections. It has never been used prospectively in a clinic.
- **The dark-skin cancer evidence is thin and foreign.** It rests on 656 images from a single American clinic, and the darkest-skin subgroup contains only 48 confirmed cancers. The result is real but the sample is small, and it may not transfer to Indian patients.
- **There has been no comparison against dermatologists**, and no clinical trial.
- **The question-asking feature has not been formally evaluated.** It works and is demonstrable, but we report no performance figure for it and make no clinical claim about it.

---

## 7. What we propose next (the secondary objective)

The limitations above share a single root cause: **there is no dermatology image collection from North India, and none anywhere that pairs Indian clinical photographs with biopsy-confirmed cancers.** We cannot prove the tool works for our patients because the evidence to test it does not exist.

We therefore propose to create it — as a **secondary objective** alongside the completed technical research.

**What it involves.** Approximately **3,000–5,000 photographs**, with patient consent, from **two hospitals over about one year**, each labelled by dermatologists, with **biopsy confirmation for every suspected cancer**. Photography uses ordinary smartphones with a simple 3D-printed spacer to keep the distance consistent, a colour card for accuracy, and a clip-on dermoscope at lesion stations. No costly infrastructure is required.

**Two arms, for a practical reason.** Common conditions will be collected prospectively over the year. Confirmed cancers will additionally be drawn from the **existing pathology archive**, because skin cancer is comparatively uncommon in brown skin and enough confirmed cases will simply not accumulate in twelve months. The international benchmark we are measured against was built the same way.

**Safeguards.** Ethics committee approval before any collection begins; consent taken before any photograph; no faces or identifying features; study numbers instead of names; encrypted storage on institution-approved systems within India, in line with the Digital Personal Data Protection Act 2023.

**Why it is worth doing.** This has been done in South India and in Brazil, and both collections became widely used international resources. Neither includes a biopsy-confirmed cancer arm, and neither covers North India. This would be **the first of its kind**, publishable as a resource in its own right, and it would let us state — or honestly refute — that this tool works for Indian patients.

---

## 8. What we are asking for

1. **Institutional Ethics Committee approval** for the collection study, including a consent waiver for the retrospective pathology-archive component.
2. **Access to the dermatology outpatient clinic** for consented photography during routine care.
3. **Access to the pathology archive** to identify biopsy-confirmed cases.
4. **Clinical supervision** — dermatologist involvement in labelling and in adjudicating disagreements.

The technical work is complete and cost nothing beyond a laptop. What we are requesting is the clinical access and governance needed to test it properly on the patients it was built for.

---

*Full technical detail, statistical methods and all confidence ranges are in the accompanying manuscript. Study procedures are specified in the companion documents: study protocol, photography standard operating procedure, equipment specification, data form, and labelling procedure.*
