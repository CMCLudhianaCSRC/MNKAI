"""MNKAI-III demo — advanced analytics: uncertainty, calibration, embedding."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROTO = ROOT / "demo_app" / "prototypes.npz"   # built by _compute_prototypes.py


def entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1)
    return float(-(p * np.log(p)).sum())


def normalized_entropy(probs: np.ndarray) -> float:
    return float(entropy(probs) / np.log(len(probs)))


def margin(probs: np.ndarray) -> float:
    s = np.sort(probs)[::-1]
    return float(s[0] - s[1])


def gini(probs: np.ndarray) -> float:
    return float(1.0 - np.sum(probs ** 2))


def uncertainty_summary(probs, mc_mean=None, mc_std=None) -> dict:
    out = {
        "confidence": round(float(probs.max()), 4),
        "entropy": round(entropy(probs), 4),
        "normalized_entropy": round(normalized_entropy(probs), 4),
        "margin_top1_top2": round(margin(probs), 4),
        "gini_impurity": round(gini(probs), 4),
        "effective_classes": round(float(np.exp(entropy(probs))), 2),
    }
    if mc_std is not None:
        out["mc_dropout_predictive_std"] = round(float(mc_std.max()), 4)
        out["mc_dropout_mean_std"] = round(float(mc_std.mean()), 4)
        out["epistemic_uncertainty"] = round(float(mc_std.mean() / (mc_mean.mean() + 1e-8)), 4)
    # qualitative flag
    ne = out["normalized_entropy"]
    out["confidence_band"] = ("high" if ne < 0.35 else "moderate" if ne < 0.6 else "low")
    return out


MALIGNANT = {"mel", "bcc", "scc", "ak"}          # treat AK as pre-malignant
_FST_DARK = {"4", "5", "6", "iv", "v", "vi"}


def decision_support(classes, probs, unc, meta=None, class_info=None, model_key="base9") -> dict:
    """Turn raw probabilities + uncertainty into a clinical recommendation.

    Implements the DEEP_ANALYSIS guidance: (1) conformal-style abstention when the
    model is uncertain -> 'refer to a dermatologist'; (2) a melanoma-safety override
    that fires even when melanoma is not the top-1; (3) a fairness caution for
    Fitzpatrick IV-VI (fairness is underpowered, n=777).
    """
    meta = meta or {}; class_info = class_info or {}
    probs = np.asarray(probs, dtype=float)
    order = probs.argsort()[::-1]
    top_cls = classes[int(order[0])]; top_p = float(probs[int(order[0])])
    pmap = {c: float(probs[i]) for i, c in enumerate(classes)}

    band = unc.get("confidence_band", "moderate")
    mc_std = unc.get("mc_dropout_predictive_std")
    margin12 = unc.get("margin_top1_top2", 1.0)

    reasons = []
    abstain = False
    if band == "low":
        abstain = True; reasons.append("high predictive entropy (spread-out probabilities)")
    if top_p < 0.45:
        abstain = True; reasons.append(f"low top-1 confidence ({top_p:.0%})")
    if margin12 < 0.12:
        abstain = True; reasons.append("top-1 and top-2 are too close to separate")
    if mc_std is not None and mc_std >= 0.10:
        abstain = True; reasons.append(f"unstable under MC-dropout (σ={mc_std:.2f})")

    # --- melanoma / malignancy safety net -------------------------------------
    mel_p = pmap.get("mel", 0.0)
    malignant_top = sum(pmap.get(c, 0.0) for c in MALIGNANT)
    mel_flag = None
    if "mel" in classes and mel_p >= 0.10 and top_cls != "mel":
        mel_flag = (f"Melanoma is in the differential ({mel_p:.0%}) but not the top prediction — "
                    "it cannot be confidently ruled out.")
    elif top_cls == "mel":
        mel_flag = "Top prediction is melanoma — treat as urgent until excluded in person."

    # --- overall recommendation -----------------------------------------------
    urgent = (top_cls in MALIGNANT) or (mel_p >= 0.10) or (malignant_top >= 0.30)
    if urgent:
        level = "urgent_referral"
        headline = "In-person dermatologist review recommended (possible malignancy)"
    elif abstain:
        level = "refer_to_dermatologist"
        headline = "Refer to a dermatologist — the model is not confident enough to assist"
    else:
        level = "decision_support_ok"
        headline = "Model is reasonably confident — use only as decision support"

    # --- fairness caution ------------------------------------------------------
    fst = str(meta.get("fitzpatrick") or meta.get("skin_tone") or "").strip().lower()
    fairness = None
    if fst:
        dark = any(tok in _FST_DARK for tok in fst.replace("fst", "").split())
        if dark or fst in _FST_DARK:
            fairness = ("Fitzpatrick IV–VI: our fairness evaluation is underpowered (n=777, melanoma "
                        "not skin-tone-stratified) — interpret this result with extra caution.")

    return {
        "level": level,                       # decision_support_ok | refer_to_dermatologist | urgent_referral
        "headline": headline,
        "abstain": bool(abstain or urgent),
        "reasons": reasons,
        "melanoma_safety": mel_flag,
        "malignant_probability": round(float(malignant_top), 4),
        "fairness_caution": fairness,
        "disclaimer": ("Research prototype — not a medical device. Does not provide a diagnosis. "
                       "Always confirm with a qualified clinician."),
    }


def load_prototypes():
    """Return dict with 2D coords of reference embeddings + class labels, or None."""
    if not PROTO.exists():
        return None
    try:
        z = np.load(PROTO, allow_pickle=True)
        return {"coords": z["coords"], "labels": z["labels"], "classes": list(z["classes"]),
                "centroids": z["centroids"], "method": str(z["method"])}
    except Exception:
        return None


def project_uploaded(emb: np.ndarray):
    """Project an uploaded embedding into the prototype 2D space (if available).
    Uses the saved PCA components when present; else returns None."""
    if not PROTO.exists():
        return None
    try:
        z = np.load(PROTO, allow_pickle=True)
        if "pca_mean" in z and "pca_comp" in z:
            x = (emb - z["pca_mean"]) @ z["pca_comp"].T
            return [float(x[0]), float(x[1])]
    except Exception:
        return None
    return None


def nearest_class_in_latent(emb: np.ndarray):
    """Cosine similarity of the uploaded embedding to each class centroid (if available)."""
    z = load_prototypes()
    if z is None or "centroids" not in z:
        return None
    cents = z["centroids"]; classes = z["classes"]
    e = emb / (np.linalg.norm(emb) + 1e-8)
    sims = []
    for c, cen in zip(classes, cents):
        cn = cen / (np.linalg.norm(cen) + 1e-8)
        sims.append({"class": c, "similarity": round(float(np.dot(e, cn)), 4)})
    return sorted(sims, key=lambda d: d["similarity"], reverse=True)
