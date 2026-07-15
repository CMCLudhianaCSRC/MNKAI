# MNKAI-III — Explainability & Diagnostics Demo

A presentation-grade web app that wraps the **current best model**
(Phase-1 9-class, fold-2, macro-F1 0.647) — used *exactly as trained, no
retraining* — and exposes deep interpretability for any uploaded dermatology image.

## Run
```bash
# from repo root (model loads on startup ~10-20s)
PYTHONPATH=src .venv/Scripts/python.exe demo_app/server.py --port 8800
# open http://localhost:8800   (also reachable on your LAN IP for phone/projector)
```
Force CPU/GPU: `DEMO_DEVICE=cpu|cuda`. Swap the model: `DEMO_CKPT=... DEMO_CONFIG=...`
(e.g. point at a corrected Phase-3 25-class checkpoint later).

Optional — light up the latent-space explorer (needs a pass over training imgs):
```bash
PYTHONPATH=src .venv/Scripts/python.exe demo_app/_compute_prototypes.py --per-class 60
```

## Features
- **Identical preprocessing** to training (`build_valid` + `encode_tabular`); optional
  age/sex/site/Fitzpatrick metadata feeds the multimodal model.
- **Prediction dashboard**: predicted class, top-5, softmax chart, confidence ring,
  inference time, full model/architecture/image metadata.
- **Explainability**: Grad-CAM, Grad-CAM++, Saliency, ViT **attention rollout**,
  and on-demand **Score-CAM**.
- **Layer activation explorer** + an **information-flow** network graph.
- **"How the model saw it"** — an animated, play/pause, slider-scrubbable pipeline
  from raw pixels → edges/texture → early/mid/high-level features → decision.
- **Advanced analytics**: entropy, normalised entropy, top1−top2 margin, Gini
  impurity, effective #classes, and **MC-dropout epistemic uncertainty**.
- **Embedding / latent-space explorer**: cosine similarity to class centroids and a
  PCA scatter of the training set with your image overlaid (after prototypes built).
- **Presentation mode** (fullscreen), **PDF export** (print), drag-and-drop, responsive.

## Architecture (modular — add methods easily)
```
demo_app/
  server.py        stdlib HTTP API (no extra deps): /api/analyze, /api/scorecam, ...
  inference.py     model load + identical preprocessing + predict + MC-dropout + embedding
  explain.py       grad-cam / grad-cam++ / saliency / attention rollout / score-cam / feature maps / pipeline
  analytics.py     entropy / margin / uncertainty / calibration / latent similarity
  _compute_prototypes.py   builds latent prototypes for the embedding explorer
  static/          index.html · styles.css · app.js  (Chart.js via CDN)
```
Add a new explainability method = one function in `explain.py` + one tab in `app.js`.

> Research/education demo — **not** a clinical diagnostic device.
