"""MNKAI-III demo — inference core.

Loads the current best model (Phase-1 fold-2, 9-class) EXACTLY as trained,
reusing the training preprocessing (build_valid transform + encode_tabular) so
the demo is faithful to the pipeline. Image-only uploads get a neutral
"unknown" clinical-metadata vector; the UI may optionally supply age/sex/site/
Fitzpatrick which are encoded identically to training.

Model path / device are overridable via env:
  DEMO_CONFIG   (default configs/phase1_dinov2_mnkai3.yaml)
  DEMO_CKPT     (default checkpoints/phase1_dinov2_fold2/best.pt)
  DEMO_DEVICE   (auto | cpu | cuda)
"""
from __future__ import annotations
import os, sys, time, io, base64, json
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mnkai.utils.config import load_config
from mnkai.models.backbones import ModelConfig, build_model
from mnkai.data.datasets import TabularSpec, encode_tabular
from mnkai.data.transforms import AugConfig, build_valid

# Selectable models. Each entry is lazy-loaded on first use.
MODELS = {
    "base9": {
        "config": str(ROOT / "configs/phase1_dinov2_mnkai3.yaml"),
        "ckpt":   str(ROOT / "checkpoints/phase1_dinov2_fold2/best.pt"),
        "label":  "Base · 9-class lesion",
        "version": "Phase-1 · 9-class · fold-2 (macro-F1 0.647)",
    },
    "indian25": {
        "config": str(ROOT / "configs/phase3_indian_25class.yaml"),
        "ckpt":   str(ROOT / "checkpoints/phase3_fixC/fold3/best.pt"),
        "label":  "Corrected · 25-class Indian",
        "version": "Phase-3 corrected · 25-class · Fix C + per-class bias (macro-F1 0.484→0.52)",
        # Tier-0 per-class logit bias (tuned on OOF) — lifts macro-F1 0.484→0.52.
        "bias_json": str(ROOT / "eval/phase3_fixC_full/threshold_opt.json"),
    },
}
DEFAULT_MODEL = os.environ.get("DEMO_MODEL", "base9")
# Back-compat: a custom DEMO_CKPT/DEMO_CONFIG still works as its own entry.
if os.environ.get("DEMO_CKPT"):
    MODELS["custom"] = {"config": os.environ.get("DEMO_CONFIG", MODELS["base9"]["config"]),
                        "ckpt": os.environ["DEMO_CKPT"], "label": "Custom model", "version": "custom checkpoint"}
    DEFAULT_MODEL = "custom"

# Clinical metadata for a polished, presentation-ready dashboard.
CLASS_INFO = {
    "mel":    {"name": "Melanoma",                 "group": "Malignant",     "risk": "high",
               "desc": "The most lethal skin cancer; arises from melanocytes. Early detection is critical."},
    "nv":     {"name": "Melanocytic nevus",        "group": "Benign",        "risk": "low",
               "desc": "A common mole. Usually harmless but monitored for change (ABCDE)."},
    "bcc":    {"name": "Basal cell carcinoma",     "group": "Malignant",     "risk": "moderate",
               "desc": "Most common skin cancer; locally invasive, rarely metastasises."},
    "scc":    {"name": "Squamous cell carcinoma",  "group": "Malignant",     "risk": "high",
               "desc": "Second-most-common skin cancer; can metastasise if untreated."},
    "df":     {"name": "Dermatofibroma",           "group": "Benign",        "risk": "low",
               "desc": "A benign fibrous skin nodule, often on the legs."},
    "bkl":    {"name": "Benign keratosis",         "group": "Benign",        "risk": "low",
               "desc": "Seborrhoeic keratosis / solar lentigo / lichen-planus-like keratosis."},
    "ak":     {"name": "Actinic keratosis",        "group": "Pre-malignant", "risk": "moderate",
               "desc": "Sun-induced; a precursor that can progress to squamous cell carcinoma."},
    "vasc":   {"name": "Vascular lesion",          "group": "Benign",        "risk": "low",
               "desc": "Angiomas, haemangiomas, pyogenic granulomas, haemorrhage."},
    "normal": {"name": "Normal skin",              "group": "Benign",        "risk": "low",
               "desc": "Healthy skin with no lesion of concern."},
    # --- Phase-3 Indian conditions (25-class model) ---
    "acne":                {"name": "Acne",                  "group": "Inflammatory", "risk": "low",
               "desc": "Comedones, papules, pustules of the pilosebaceous unit."},
    "alopecia":            {"name": "Alopecia",              "group": "Other",        "risk": "low",
               "desc": "Hair loss — patterned, areata, or scarring."},
    "autoimmune_systemic": {"name": "Systemic autoimmune",   "group": "Autoimmune",   "risk": "moderate",
               "desc": "Cutaneous signs of systemic autoimmune disease (e.g. lupus)."},
    "bacterial_infection": {"name": "Bacterial infection",   "group": "Infectious",   "risk": "moderate",
               "desc": "Impetigo, cellulitis, folliculitis and related."},
    "contact_dermatitis":  {"name": "Contact dermatitis",    "group": "Inflammatory", "risk": "low",
               "desc": "Irritant or allergic eczematous reaction to contact."},
    "drug_eruption":       {"name": "Drug eruption",         "group": "Inflammatory", "risk": "moderate",
               "desc": "Cutaneous reaction to a medication."},
    "eczema":              {"name": "Eczema",                "group": "Inflammatory", "risk": "low",
               "desc": "Atopic/other dermatitis — itchy, inflamed, scaly skin."},
    "lichen_planus":       {"name": "Lichen planus",         "group": "Inflammatory", "risk": "low",
               "desc": "Pruritic violaceous papules; mucosa may be involved."},
    "pemphigus":           {"name": "Pemphigus",             "group": "Autoimmune",   "risk": "high",
               "desc": "Autoimmune blistering disease; can be serious."},
    "psoriasis":           {"name": "Psoriasis",             "group": "Inflammatory", "risk": "low",
               "desc": "Well-demarcated erythematous plaques with silvery scale."},
    "rosacea":             {"name": "Rosacea",               "group": "Inflammatory", "risk": "low",
               "desc": "Central-face erythema, telangiectasia, papulopustules."},
    "scabies":             {"name": "Scabies",               "group": "Infectious",   "risk": "low",
               "desc": "Mite infestation — intensely itchy burrows."},
    "tinea":               {"name": "Tinea (fungal)",        "group": "Infectious",   "risk": "low",
               "desc": "Dermatophyte infection — annular scaly plaques."},
    "urticaria":           {"name": "Urticaria",             "group": "Inflammatory", "risk": "low",
               "desc": "Transient itchy weals (hives)."},
    "viral_infection":     {"name": "Viral infection",       "group": "Infectious",   "risk": "moderate",
               "desc": "Viral exanthem / pox-type lesions (mpox, HFMD, varicella…)."},
    "vitiligo":            {"name": "Vitiligo",              "group": "Other",        "risk": "low",
               "desc": "Depigmented macules from melanocyte loss."},
}


def _device() -> torch.device:
    d = os.environ.get("DEMO_DEVICE", "auto").lower()
    if d == "cpu":
        return torch.device("cpu")
    if d == "cuda" or (d == "auto" and torch.cuda.is_available()):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


class Engine:
    """Singleton holding the model + preprocessing. Lazy-loaded on first use."""

    def __init__(self, key: str = "base9"):
        spec = MODELS.get(key) or MODELS["base9"]
        self.key = key if key in MODELS else "base9"
        self.config_path = spec["config"]; self.ckpt_path = spec["ckpt"]
        self.version = spec["version"]; self.label = spec["label"]
        self.device = _device()
        self.cfg = load_config(self.config_path)
        self.classes = [c.lower() for c in self.cfg["classes"]["names"]]
        tabc = self.cfg.get("tabular", {}) or {}
        self.tab_spec = TabularSpec(
            use_sex=tabc.get("use_sex", True), use_age=tabc.get("use_age", True),
            use_site=tabc.get("use_site", True), use_fitzpatrick=tabc.get("use_fitzpatrick", True),
        ) if tabc.get("enabled", False) else None
        aug = AugConfig(
            img_size=int(self.cfg["data"]["img_size"]),
            mean=tuple(self.cfg["data"].get("normalize_mean", [0.485, 0.456, 0.406])),
            std=tuple(self.cfg["data"].get("normalize_std", [0.229, 0.224, 0.225])),
            shades_of_gray_p=float(self.cfg.get("augmentation", {}).get("shades_of_gray_p", 0.0)),
            hair_removal=False)
        self.aug = aug
        self.transform = build_valid(aug)
        self.mean = np.array(aug.mean, dtype=np.float32)
        self.std = np.array(aug.std, dtype=np.float32)
        self.img_size = int(self.cfg["data"]["img_size"])
        self._model = None
        # Tier-0 per-class logit bias (optional): tuned on OOF to maximise macro-F1
        # (lifts the 25-class model 0.484 -> ~0.52). Aligned to self.classes by name.
        self.bias = None
        _bj = spec.get("bias_json")
        if _bj and Path(_bj).is_file():
            try:
                _d = json.loads(Path(_bj).read_text())
                _bcls = [str(c).lower() for c in _d.get("classes", [])]
                _bvec = _d.get("bias", [])
                if len(_bcls) == len(_bvec) == len(self.classes):
                    _m = {c: float(b) for c, b in zip(_bcls, _bvec)}
                    self.bias = np.array([_m.get(c, 0.0) for c in self.classes], dtype=np.float32)
            except Exception:
                self.bias = None

    # -- model -------------------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            mcfg = ModelConfig(
                name=self.cfg["train"]["model_name"], pretrained=False, num_classes=len(self.classes),
                drop_rate=float(self.cfg["train"].get("drop_rate", 0.10)),
                drop_path_rate=float(self.cfg["train"].get("drop_path_rate", 0.10)),
                head_dropout=float(self.cfg["train"].get("head_dropout", 0.30)),
                tabular_dim=(self.tab_spec.dim if self.tab_spec else 0),
                tabular_hidden=int(self.cfg["train"].get("tabular_hidden", 128)),
                head_hidden=int(self.cfg["train"].get("head_hidden", 0)),
                img_size=self.img_size, fusion=str(self.cfg["train"].get("fusion", "concat")),
                cross_attn_heads=int(self.cfg["train"].get("cross_attn_heads", 4)),
                cross_attn_dim=int(self.cfg["train"].get("cross_attn_dim", 256)))
            m = build_model(mcfg).to(self.device).eval()
            obj = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
            sd = obj.get("ema_state") or obj.get("model_state") or obj
            missing, unexpected = m.load_state_dict(sd, strict=False)
            self._ckpt_meta = {"epoch": obj.get("epoch"), "best_macro_f1": obj.get("best_macro_f1"),
                               "missing": len(missing), "unexpected": len(unexpected)}
            self._model = m
        return self._model

    def model_info(self) -> dict:
        _ = self.model  # ensure loaded
        n_params = sum(p.numel() for p in self._model.parameters())
        return {
            "architecture": self.cfg["train"]["model_name"],
            "params_millions": round(n_params / 1e6, 1),
            "fusion": self.cfg["train"].get("fusion"),
            "img_size": self.img_size,
            "num_classes": len(self.classes),
            "classes": self.classes,
            "device": str(self.device),
            "checkpoint": Path(self.ckpt_path).as_posix(),
            "ckpt_epoch": self._ckpt_meta.get("epoch"),
            "ckpt_macro_f1": round(float(self._ckpt_meta.get("best_macro_f1") or 0), 4),
            "version": f"MNKAI-III · {self.version}",
            "model_key": self.key,
            "model_label": self.label,
        }

    # -- preprocessing -----------------------------------------------------
    def preprocess(self, pil_img: Image.Image, meta: dict | None = None):
        """Returns (img_tensor[1,C,H,W], tabular_tensor[1,D]|None, processed_uint8 HxWx3)."""
        pil_img = pil_img.convert("RGB")
        img_t = self.transform(pil_img).unsqueeze(0).to(self.device)
        tab_t = None
        if self.tab_spec is not None:
            row = pd.Series({
                "sex": (meta or {}).get("sex"),
                "age_approx": (meta or {}).get("age"),
                "anatom_site_general": (meta or {}).get("site"),
                "fitzpatrick": (meta or {}).get("fitzpatrick"),
            })
            tab = encode_tabular(row, self.tab_spec)
            tab_t = torch.from_numpy(tab).unsqueeze(0).to(self.device)
        # de-normalise for display
        arr = img_t[0].detach().cpu().numpy().transpose(1, 2, 0)
        disp = np.clip(arr * self.std + self.mean, 0, 1)
        disp_u8 = (disp * 255).astype(np.uint8)
        return img_t, tab_t, disp_u8

    # -- prediction --------------------------------------------------------
    @torch.no_grad()
    def predict(self, img_t, tab_t):
        t0 = time.time()
        logits = self.model(img_t, tabular=tab_t) if tab_t is not None else self.model(img_t)
        dt = (time.time() - t0) * 1000
        if self.bias is not None:                    # Tier-0 per-class bias (best decision rule)
            logits = logits + torch.from_numpy(self.bias).to(logits.device)
        probs = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        return logits, probs, dt

    @torch.no_grad()
    def mc_dropout(self, img_t, tab_t, n=15):
        """Epistemic uncertainty via MC-dropout: enable dropout at test, sample n."""
        was_training = self.model.training
        self.model.train()  # activates dropout
        # but keep BN/LN in eval — DINOv2 ViT uses LayerNorm (no running stats), safe
        samples = []
        for _ in range(n):
            lg = self.model(img_t, tabular=tab_t) if tab_t is not None else self.model(img_t)
            samples.append(F.softmax(lg, dim=-1)[0].detach().cpu().numpy())
        if not was_training:
            self.model.eval()
        S = np.stack(samples)               # (n, C)
        return S.mean(0), S.std(0)

    @torch.no_grad()
    def embedding(self, img_t):
        """Pooled image feature (the model's latent representation)."""
        bb = self.model.backbone
        feats = bb.forward_features(img_t)
        if feats.dim() == 3:
            pooled = feats[:, 0]
        elif feats.dim() == 4:
            pooled = feats.mean(dim=(2, 3))
        else:
            pooled = feats
        return pooled[0].detach().cpu().numpy()


@lru_cache(maxsize=4)
def engine(key: str = None) -> Engine:
    return Engine(key or DEFAULT_MODEL)


def list_models() -> list:
    """Registry metadata for the model selector (no weights loaded)."""
    out = []
    for k, spec in MODELS.items():
        out.append({
            "key": k, "label": spec["label"], "version": spec["version"],
            "available": Path(spec["ckpt"]).exists(),
            "default": (k == DEFAULT_MODEL),
        })
    return out


# -- small image utils --------------------------------------------------------
def to_b64_png(arr_u8: np.ndarray) -> str:
    if arr_u8.ndim == 2:
        im = Image.fromarray(arr_u8, mode="L")
    else:
        im = Image.fromarray(arr_u8, mode="RGB")
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def decode_dataurl(data_url: str) -> Image.Image:
    b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    return Image.open(io.BytesIO(base64.b64decode(b64)))
