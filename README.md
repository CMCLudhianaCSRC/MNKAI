# MNKAI-III — Rigorous, Data-Scarce Dermatology AI for Indian Skin

A clinical **decision-support** system for 25 dermatological conditions, built entirely on a 6 GB laptop.
The emphasis is **honesty and rigor** — leak-free evaluation, a full trust layer, and a simple recipe that
outperforms fine-tuning — not a leaderboard number. Research prototype; **not a diagnosis / not a medical device.**

## Headline (honest, leak-free numbers)
| Task | Fine-tuned baseline | **Final method (frozen-probe)** | Test |
|---|---|---|---|
| 25-class Indian macro-F1 | 0.527 | **0.573 [0.55–0.59]** | de-duplicated external |
| 9-class base macro-F1 | 0.637 | **0.656** | OOF CV |
| **DDI dark-skin cancer AUROC** | 0.53 | **0.748 (Fitzpatrick V–VI 0.789)** | DDI 5-fold CV |
| **DDI dark-skin cancer AUROC — frozen ensemble** | 0.53 | **0.826 [0.79–0.86] (V–VI 0.872 [0.81–0.93])** | DDI 5-fold CV, 6 frozen backbones, ΔAUROC vs single p<0.001 |
| Calibration (ECE) | 0.20 | **0.016** | after temperature scaling |
| Conformal set @90% coverage | ~10.6 | **~4** | usable prediction sets |
| Selective prediction | — | **refer 30% → 77% retained accuracy** | abstention |

## The core idea
**Freeze a strong self-supervised backbone (DINOv2 @518); attach a light MLP head.** In our data-scarce,
multi-domain regime, end-to-end fine-tuning *degrades* the representation — it collapsed dark-skin cancer to
chance and under-uses features on the 25-class. Frozen-features-+-probe wins on **every task**, cheaply, on
the laptop. (Discovered while fixing the DDI dark-skin failure: 0.53→0.79, at the published ceiling.)

## System architecture
```
photo -> FROZEN DINOv2@518 -> MLP probe -> temperature -> per-class bias -> calibrated differential
                                                                                 |
                                                     +-- conformal set / margin --+
                                                     |                            v
                                 confident? -- yes --> decision support (top-k + cancer flag)
                                     | no (abstention gate)
                                     v
                           PHOTO-GROUNDED LLM interview (Gemini v1) -> Bayesian fusion -> refined differential
```
- Classifier: `demo_app/pipeline.py` (loads `models/clinical_25class.pkl`).
- Trust: temperature calibration + RAPS conformal + abstention.
- LLM interviewer: `demo_app/llm.py` — sees the photo, asks discriminating history questions **only when the
  classifier is uncertain**, Bayesian-fuses answers (image model stays the decision-maker; never overrides cancer).

## Run it
```bash
# one image, full clinical output (differential + conformal set + abstention + gated interview)
python demo_app/pipeline.py --image path/to/lesion.jpg
# demo app (2-model + decision support UI)
PYTHONPATH=src python demo_app/server.py --port 8800
```
LLM interviewer needs a Gemini API key in `secrets/gemini_key.txt` (free tier, Google AI Studio).

## Reproduce the science
```bash
python scripts/_cache_features.py --config configs/phase3_indian_25class.yaml --img-size 518 --out eval/probe/feats_25class_518.npz
python scripts/_probe_multiclass.py --feats eval/probe/feats_25class_518.npz --head mlp   # 25-class
python scripts/_probe_backbone.py --backbone dinov2 --img-size 518 --out eval/probe/ddi.json  # DDI dark-skin cancer
python scripts/_trust_refresh_25class.py   # calibration + conformal + abstention
python scripts/_save_deployable.py         # writes models/clinical_25class.pkl
```

## What we found (and fixed) — the honest record
- **Majority-class collapse** from double-balancing -> single-mechanism fix (eczema recall 0.004->0.503).
- **~33% train-duplicate leakage** in the 25-class external test (cross-source atlas images) -> detected via
  frozen-feature nearest-neighbour, dropped, re-reported. The 9-class external was 99.8% nevi (degenerate -> OOF CV).
- **Dark-skin cancer ceiling** ~ 0.78 — data-limited (DDI is essentially the only public dark-skin clinical cancer
  set), not method-limited. Paired with per-tone abstention.

## Key documents
- `PAPER_DRAFT.md` — submission draft. `PROJECT_REPORT.md` — status + honest metrics.
- `logs/DDI_BREAKTHROUGH.md`, `logs/DARKSKIN_RESEARCH.md` — the dark-skin study.
- `LLM_PLAN.md` — the interviewer (v1 shipped; v2 trained-VLM plan). `OVERHAUL_PLAN.md`, `logs/DATA_INVENTORY.md`.

## Scope done on the laptop vs deferred
- **Done (laptop):** all models, trust layer, DDI breakthrough, LLM v1 (photo-grounded, gated), deployable pipeline, paper.
- **Deferred (would need an A100, not pursued):** SSL derm-backbone pretraining; the LLM v2 trained compact VLM.
  v1 delivers the "asks questions about the photo when unsure" behaviour today.

## Honest limitations
Decision support, not diagnosis. Dark-skin cancer ~0.78 (data-scarce). Fairness subgroups underpowered
(external FST IV–VI small; DDI is the powered dark-skin read). No biopsy confirmation / prospective trial.
Melanoma handled at a locked high-sensitivity operating point with abstention.
