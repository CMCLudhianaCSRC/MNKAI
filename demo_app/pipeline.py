"""MNKAI-III clinical pipeline (deployable, laptop-only).

photo -> FROZEN DINOv2@518 feature -> saved probe (scaler+MLP) -> temperature -> per-class
bias -> calibrated differential -> RAPS conformal set -> abstention gate -> (if uncertain)
photo-grounded LLM interview. The calibrated classifier is the decision-maker; the LLM only
asks history questions when the classifier is unsure. Decision support, not diagnosis.

    from demo_app.pipeline import ClinicalPipeline
    out = ClinicalPipeline().assess(pil_image, meta={"age":58})
"""
from __future__ import annotations
import sys, pickle, base64, io
from pathlib import Path
from functools import lru_cache
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "demo_app"))
try:
    import llm
except Exception:
    llm = None

MALIGNANT = {"mel", "bcc", "scc", "ak", "melanoma", "basal_cell_carcinoma", "squamous_cell_carcinoma"}


@lru_cache(maxsize=1)
def _load():
    # timm is imported lazily so the pure helpers in this module (`_raps_set`,
    # `_calibrated_adjust`) can be imported/tested without the backbone stack.
    import timm
    from timm.data import resolve_data_config, create_transform
    with open(ROOT / "models" / "clinical_25class.pkl", "rb") as f:
        art = pickle.load(f)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bb = timm.create_model(art["backbone"], pretrained=True, num_classes=0, img_size=art["img_size"]).to(dev).eval()
    tfm = create_transform(**resolve_data_config({"input_size": (3, art["img_size"], art["img_size"])}, model=bb))
    return art, bb, tfm, dev


def _raps_set(prob, qhat, lam, k_reg):
    order = np.argsort(-prob); srt = prob[order]; csum = np.cumsum(srt)
    pen = lam * np.maximum(0, np.arange(1, len(prob) + 1) - k_reg)
    k = int(np.searchsorted(csum + pen, qhat) + 1)
    return order[:max(1, min(k, len(prob)))]


def _calibrated_adjust(p, temperature, bias):
    """Temperature-scale a probability vector, then apply a per-class logit bias.

    Extracted verbatim from `ClinicalPipeline.assess` (behaviour-identical) so the
    transform is unit-testable. Returns (cal, adj):
      cal : temperature-scaled calibrated probabilities
      adj : cal re-expressed in log space + per-class `bias`, re-softmaxed (the
            point-decision distribution).
    """
    from scipy.special import log_softmax, softmax
    logit = np.log(np.clip(p, 1e-9, 1))
    cal = softmax(logit / temperature)
    adj = softmax(log_softmax(np.log(np.clip(cal, 1e-9, 1))) + bias)
    return cal, adj


class ClinicalPipeline:
    def __init__(self):
        self.art, self.bb, self.tfm, self.dev = _load()

    @torch.no_grad()
    def _feat(self, pil):
        x = self.tfm(pil.convert("RGB")).unsqueeze(0).to(self.dev)
        return self.bb(x).float().cpu().numpy()

    def assess(self, pil: Image.Image, meta: dict | None = None, image_b64: str | None = None,
               do_interview: bool = True) -> dict:
        art = self.art; classes = art["classes"]
        f = self.art["scaler"].transform(self._feat(pil))
        p = self.art["head"].predict_proba(f)[0]
        cal, adj = _calibrated_adjust(p, art["temperature"], art["bias"])  # calibrate + per-class bias
        order = np.argsort(-adj)
        top = [{"name": classes[i], "prob": round(float(adj[i]), 4)} for i in order[:5]]
        cset = _raps_set(cal, art["conformal"]["qhat"], art["conformal"]["lambda"], art["conformal"]["k_reg"])
        margin = float(np.sort(adj)[::-1][0] - np.sort(adj)[::-1][1])
        # cancer safety: pooled malignant probability
        mal = float(sum(adj[i] for i, c in enumerate(classes) if c in MALIGNANT))
        uncertain = (len(cset) > 4) or (margin < 0.15)
        out = {
            "differential": top,
            "conformal_set": [classes[i] for i in cset],
            "conformal_set_size": int(len(cset)),
            "top1_margin": round(margin, 3),
            "malignant_probability": round(mal, 3),
            "uncertain": bool(uncertain),
            "recommendation": ("Refer to a dermatologist (possible malignancy)" if mal >= 0.30 else
                               "Refer to a dermatologist — model not confident" if uncertain else
                               "Decision support: likely " + top[0]["name"]),
            "disclaimer": "Research prototype — decision support, not a diagnosis.",
        }
        # photo-grounded interview only when uncertain (the abstention gate)
        if do_interview and uncertain and llm and llm.available():
            try:
                out["interview_questions"] = llm.propose_questions(top, meta, k=3, image_b64=image_b64)
            except Exception as e:
                out["interview_error"] = str(e)[:120]
        return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--image", required=True); a = ap.parse_args()
    pil = Image.open(a.image); b64 = base64.b64encode(open(a.image, "rb").read()).decode()
    import json; print(json.dumps(ClinicalPipeline().assess(pil, {"age": 55}, image_b64=b64), indent=2))
