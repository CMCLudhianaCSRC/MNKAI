# Agent 1C — Code Review: `demo_app/` + Cross-Cutting Concerns

Scope: `demo_app/{pipeline.py, llm.py, inference.py, server.py, analytics.py, explain.py, _compute_prototypes.py}` read in full, plus repo-wide assessment of architecture, testing, docs, reproducibility, security/robustness, and the malignant-class safety logic. Static review only — no code executed, no heavy compute, no secrets read.

## Executive summary

The app layer is well-written for a research demo: clean modular split (`inference` / `explain` / `analytics` / `server`), faithful reuse of the training preprocessing, a thoughtful trust layer (temperature + RAPS conformal + abstention gate + MC-dropout), and a genuinely careful LLM interviewer that is gated on classifier uncertainty and instructed never to override cancer. Good hygiene: `secrets/` is gitignored and untracked (the Gemini key is **not** committed), and `mc_dropout()`'s `model.train()` is actually safe because the head/tabular MLP uses only `Dropout` + `LayerNorm` (no `BatchNorm`), matching the in-code comment.

However there are two **serious security issues in the demo server** (an unauthenticated file-read path traversal that can exfiltrate `secrets/gemini_key.txt`, and default `0.0.0.0` binding with wildcard CORS and no auth), a **safety-logic concern** (the malignancy referral gate thresholds on macro-F1-tuned biased probabilities rather than the calibrated distribution), and a **architecture/consistency gap**: the shipped web app runs the *fine-tuned* checkpoints that the project's central finding says are inferior, while the scientifically-preferred *frozen-probe + conformal + LLM* path (`pipeline.py`/`llm.py`) is CLI-only and not wired into the server. Testing of the app layer is nonexistent, and the reproducible-environment story regressed (deleted `requirements.txt`, lower-bound-only deps).

---

## Findings

### F1 — HIGH (security): Unauthenticated path traversal leaks the Gemini API key and arbitrary files
`demo_app/server.py:138-143`
```python
if self.path.startswith("/static/"):
    f = STATIC / self.path[len("/static/"):]
    if f.is_file():
        ... return self._send(f.read_bytes(), ...)
```
`self.path` is the raw request target — this custom handler does **no** `os.path.normpath`/`resolve`/containment check (unlike `SimpleHTTPRequestHandler.translate_path`). A request such as `GET /static/../../secrets/gemini_key.txt` resolves to `demo_app/static/../../secrets/gemini_key.txt` = `<repo>/secrets/gemini_key.txt`; `is_file()` is true and the bytes are returned. This exfiltrates the LLM API key and, more generally, **any file readable by the process** (checkpoints, data manifests, other secrets). Because the server binds `0.0.0.0` by default (F2), this is reachable from the LAN. Fix: resolve the candidate and enforce it is inside `STATIC`, e.g. `f = (STATIC / rel).resolve(); if not str(f).startswith(str(STATIC.resolve())): 404`, and reject `..` segments.

### F2 — HIGH (security): Server binds all interfaces, no auth, wildcard CORS; patient photos exposed
`demo_app/server.py:131` (`Access-Control-Allow-Origin: *`), `:175` (`--host default "0.0.0.0"`), README lines advertising "also reachable on your LAN IP". The analyze/scorecam endpoints accept arbitrary base64 images and run model inference with **no authentication** and CORS open to every origin. For a clinical decision-support demo that ingests lesion photos (PHI), anyone on the network can submit images, harvest predictions, and (via F1) read files. Fix: default `--host 127.0.0.1`, drop the `*` CORS header (or restrict to the served origin), and document that LAN exposure requires an explicit opt-in plus auth.

### F3 — MEDIUM (safety / logic): Malignancy referral gate thresholds on F1-tuned biased probabilities, not the calibrated distribution
`demo_app/pipeline.py:64-71,80`
```python
cal = softmax(logit / art["temperature"])                 # calibrated (ECE ~0.016)
adj = softmax(log_softmax(np.log(np.clip(cal,1e-9,1))) + art["bias"])  # macro-F1-tuned
...
mal = float(sum(adj[i] for i,c in enumerate(classes) if c in MALIGNANT))
"recommendation": ("Refer ... (possible malignancy)" if mal >= 0.30 else ...)
```
The per-class `bias` is tuned to maximise macro-F1 (per README/`_optimize_thresholds`), which deliberately re-weights classes and **decalibrates** the distribution. The `0.30` malignancy safety threshold is then applied to these decalibrated numbers rather than to the temperature-calibrated `cal` (whose whole purpose is trustworthy probabilities). If the F1 bias down-weights a malignant class, the safety gate can under-fire. A safety/risk threshold should read the calibrated `cal`; the biased `adj` is appropriate only for the point-label decision. Same design issue in `analytics.decision_support` (`analytics.py:83,92`), where `probs` already include the F1 bias for `indian25` (applied in `inference.predict`, `inference.py:235-236`). Not auto-applyable — changes clinical trigger behavior; needs the team's decision.

### F4 — MEDIUM (architecture): Shipped web app uses the inferior fine-tuned models; the frozen-probe method is CLI-only
`demo_app/server.py:22-25` imports `inference`/`explain`/`analytics` and serves the fine-tuned checkpoints (`base9` Phase-1 fold-2, `indian25` Phase-3 Fix-C). It does **not** import `pipeline.py` or `llm.py`. The project's central scientific finding is that a *frozen* backbone + probe + conformal + LLM interview beats end-to-end fine-tuning, and that path lives only in `pipeline.py` (invokable via `python demo_app/pipeline.py --image ...`). So the interactive demo people will actually run embodies the approach the paper argues *against*, and the conformal/LLM trust layer is unreachable from the UI. This is a modularity/consistency defect: two parallel decision-support stacks with no shared core. Recommend unifying on the `pipeline.py` frozen-probe path (or clearly labeling the server as the "legacy fine-tuned" demo).

### F5 — MEDIUM (duplication / maintainability): Two divergent malignancy/abstention implementations that can drift
The malignant class set and the safety thresholds are defined twice with different content:
- `pipeline.py:29` `MALIGNANT = {"mel","bcc","scc","ak","melanoma","basal_cell_carcinoma","squamous_cell_carcinoma"}` (the three long-form names never match — the deployed class vocabulary is short codes only, per `configs/phase3_indian_25class.yaml`, so they are dead entries).
- `analytics.py:48` `MALIGNANT = {"mel","bcc","scc","ak"}`.
- Abstention/malignancy logic also diverges: `pipeline.py` uses `mal>=0.30` OR (`|conformal|>4` OR `margin<0.15`); `analytics.decision_support` uses a different composite (`mel_p>=0.10`, `malignant_top>=0.30`, `top_p<0.45`, `margin<0.12`, `mc_std>=0.10`). Two safety-critical definitions in two files invite silent drift. Consolidate into one shared module with one canonical `MALIGNANT` set and one gate.

### F6 — MEDIUM (security / privacy): Patient lesion photo is sent to Google's Gemini API
`demo_app/llm.py:43-51` inlines `image_b64` into the Gemini request. For a tool framed as clinical decision support, the image (potential PHI) leaves the device to a third-party cloud with no consent prompt, no de-identification, and no toggle surfaced to the user beyond "no key = disabled". This should be an explicit, documented, opt-in behavior. (The key handling itself is fine: read from env or `secrets/gemini_key.txt`, sent as a header, never logged.)

### F7 — MEDIUM (robustness): No request-size or decompression-bomb limits on the server
`demo_app/server.py:160-161` reads `n = int(Content-Length)` then `self.rfile.read(n)` with no cap — a large declared body is buffered whole into memory (DoS/OOM). `inference.decode_dataurl` (`inference.py:296-298`) does `Image.open(base64...)` with no `Image.MAX_IMAGE_PIXELS` guard, so a crafted small file can decompress to a huge bitmap. Add a max body size, a max base64 length, and a Pillow pixel cap.

### F8 — LOW (testing): Zero tests for the entire app layer, including safety-critical logic
The suite (`tests/`, 9 files, 670 lines) covers `src/mnkai` (losses, calibration, bootstrap, splits, schedulers, abstention, qc). **No test imports `demo_app`** at all (grep confirmed). The RAPS set construction (`pipeline._raps_set`), the abstention gate, the `MALIGNANT` pooling, and `analytics.decision_support` — the pieces that gate a "refer for possible cancer" message — are entirely untested. At minimum add unit tests pinning: RAPS set size vs `qhat`, the malignancy gate firing/not-firing at boundary probs, and `decision_support` level transitions.

### F9 — LOW (reproducibility): Environment is not pinned; `requirements.txt` was removed
`git status` shows `D requirements.txt` (deleted); only `requirements_legacy.txt` remains. `pyproject.toml` specifies lower bounds only (`torch>=2.4`, `timm>=1.0.11`, …) with no lockfile/hashes. A ViT + timm + calibration stack is version-sensitive; "clone and run" is not reproducible. Restore a pinned `requirements.txt` (or commit a `uv.lock`/`pip freeze`), and make the README's `.venv/Scripts/python.exe` (Windows-only) invocation portable.

### F10 — LOW (robustness / dead feature): Fragile Gemini response parsing; advertised Bayesian fusion is never wired in
`demo_app/llm.py:52` does `r["candidates"][0]["content"]["parts"][0]["text"]` with no guard — when Gemini returns a safety block or an API error, `candidates` may be empty/absent → `KeyError`/`IndexError` (caught only because `pipeline.assess` wraps the call). More notably, `fuse_history()` (the "Bayesian fusion" the README/`pipeline.py` docstring describe as part of the flow) is only exercised in `llm.py`'s `__main__`; neither `pipeline.py` nor `server.py` ever calls it, so the refined posterior is a documented-but-unshipped feature. Either wire it in or adjust the docs.

### F11 — LOW (documentation accuracy): `demo_app/README.md` describes a different system than the code ships
`demo_app/README.md` states the app wraps "the current best model (Phase-1 9-class, fold-2, macro-F1 0.647)" and mentions Chart.js "via CDN", while the root `README.md`/`STUDY_GUIDE` present the 25-class frozen-probe as the method and the default registry still exposes `base9`. Combined with F4/F10, the app docs overstate integration (pipeline fusion, "the current best model"). Reconcile the two READMEs and state plainly which stack the server runs.

---

## Notes / non-issues verified
- `mc_dropout` `model.train()` is **safe** — the head (`backbones.py:160-172`) and tabular MLP (`:139-143`) use only `Dropout`/`LayerNorm`/`Linear`; no `BatchNorm` running-stats corruption. Comment is accurate.
- `secrets/` is gitignored and `git ls-files secrets/` is empty — the API key is not in version control. Good.
- `pipeline._raps_set` logic (searchsorted + regularization penalty, `max(1, min(k, C))`) is correct at both extremes.
- `torch.load(..., weights_only=False)` (`inference.py:181`) is a code-exec vector only for untrusted checkpoints; here checkpoints are local/trusted — informational, prefer `weights_only=True` where possible.
