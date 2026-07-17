"""DDI CLINICAL READOUT on the frozen 6-backbone score-average ensemble (CACHED features).

Reproduces the exact OOF P(malignant) of scripts/_ddi_ensemble_ci.py (6 backbones, 5-fold
StratifiedKFold seed 0, class-balanced LR), then reports what a dermatology reviewer wants
BEYOND AUROC:

  (a) operating-point sensitivity/specificity: Se @ fixed Sp 0.80 & 0.90, and Sp @ fixed
      Se 0.90 -- overall AND for FST V-VI (dark skin).
  (b) decision-curve / net-benefit analysis: NB(pt) = TP/n - FP/n * pt/(1-pt) across
      threshold probabilities pt in 0.05..0.50, versus treat-all and treat-none.
  (c) temperature calibration of the ensemble malignant score (cross-fit T so ECE is honest,
      not fit-and-eval-on-same-data), reporting ECE before/after.

    python scripts/_ddi_clinical.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "eval" / "probe" / "ddi_backbones"
DDI = Path("D:/AIFINAL/images/ddi")
TONE = {12: "FST I-II", 34: "FST III-IV", 56: "FST V-VI"}
MEMBERS = ["dinov2_b", "dinov2_reg", "eva02_b", "convnextv2_b", "siglip_b", "dinov2_l"]
EPS = 1e-7


def oof_scores(X, y, strat, seed=0):
    """5-fold stratified OOF P(malignant) -- identical recipe to _ddi_ensemble_ci.py."""
    s = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, strat):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5, random_state=0)
        clf.fit(sc.transform(X[tr]), y[tr])
        s[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return s


# ---------- (a) operating points ----------
def se_at_spec(y, score, spec_target):
    """Max sensitivity achievable while specificity >= spec_target."""
    fpr, tpr, thr = roc_curve(y, score)
    spec = 1.0 - fpr
    ok = spec >= spec_target
    if not ok.any():
        return None
    i = np.argmax(np.where(ok, tpr, -1))
    return {"sensitivity": float(tpr[i]), "specificity": float(spec[i]), "threshold": float(thr[i])}


def sp_at_sens(y, score, sens_target):
    """Max specificity achievable while sensitivity >= sens_target."""
    fpr, tpr, thr = roc_curve(y, score)
    spec = 1.0 - fpr
    ok = tpr >= sens_target
    if not ok.any():
        return None
    i = np.argmax(np.where(ok, spec, -1))
    return {"specificity": float(spec[i]), "sensitivity": float(tpr[i]), "threshold": float(thr[i])}


def operating_points(y, score):
    return {
        "se_at_spec_0.80": se_at_spec(y, score, 0.80),
        "se_at_spec_0.90": se_at_spec(y, score, 0.90),
        "sp_at_sens_0.90": sp_at_sens(y, score, 0.90),
        "n": int(len(y)), "n_malignant": int(y.sum()), "prevalence": float(y.mean()),
    }


# ---------- (b) decision curve / net benefit ----------
def net_benefit(y, prob, pts):
    n = len(y)
    prev = float(y.mean())
    rows = []
    for pt in pts:
        pred = prob >= pt
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        w = pt / (1.0 - pt)
        nb_model = tp / n - fp / n * w
        nb_all = prev - (1.0 - prev) * w
        rows.append({
            "pt": round(float(pt), 3),
            "nb_model": float(nb_model),
            "nb_treat_all": float(nb_all),
            "nb_treat_none": 0.0,
            "delta_vs_best_default": float(nb_model - max(nb_all, 0.0)),
        })
    return rows


# ---------- (c) temperature calibration + ECE ----------
def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _nll(T, z, y):
    p = np.clip(_sigmoid(z / T), EPS, 1 - EPS)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_temperature(z, y):
    r = minimize_scalar(_nll, args=(z, y), bounds=(0.05, 20.0), method="bounded")
    return float(r.x)


def ece(y, prob, n_bins=10):
    """Equal-width 10-bin expected calibration error (confidence = P(malignant))."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    n = len(y)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (prob > lo) & (prob <= hi) if lo > 0 else (prob >= lo) & (prob <= hi)
        if not m.any():
            continue
        conf = prob[m].mean()
        acc = y[m].mean()
        e += (m.sum() / n) * abs(acc - conf)
    return float(e)


def crossfit_calibration(y, score, seed=0):
    """Honest ECE: cross-fit T on 5 folds of the OOF scores (fit T on 4 folds, apply to the
    held-out fold), so no sample's calibrated prob was used to fit its own T."""
    z = _logit(score)
    cal = np.zeros_like(score)
    Ts = []
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in skf.split(z.reshape(-1, 1), y):
        T = fit_temperature(z[tr], y[tr])
        Ts.append(T)
        cal[te] = _sigmoid(z[te] / T)
    T_full = fit_temperature(z, y)  # single T reported for deployment
    return cal, float(np.mean(Ts)), T_full


def main():
    meta = pd.read_csv(DDI / "ddi_metadata.csv")
    y = meta["malignant"].astype(int).to_numpy()
    tone = meta["skin_tone"].to_numpy()
    strat = (pd.Series(tone).astype(str) + "_" + pd.Series(y).astype(str)).to_numpy()
    feats = {m: np.load(CACHE / f"{m}.npy") for m in MEMBERS}

    # score-average ensemble OOF P(malignant), seed 0 (matches ensemble_ci point estimate)
    per_member = {m: oof_scores(feats[m], y, strat, seed=0) for m in MEMBERS}
    ens = np.column_stack([per_member[m] for m in MEMBERS]).mean(1)

    vvi = tone == 56  # FST V-VI subgroup

    # (c) calibration first, so the decision curve can use calibrated probabilities
    cal, T_mean_cv, T_full = crossfit_calibration(y, ens, seed=0)
    ece_before = ece(y, ens)
    ece_after = ece(y, cal)
    ece_before_vvi = ece(y[vvi], ens[vvi])
    ece_after_vvi = ece(y[vvi], cal[vvi])

    pts = [round(p, 3) for p in np.arange(0.05, 0.501, 0.05)]

    result = {
        "protocol": "DDI 656 biopsy-confirmed images, image-level 5-fold StratifiedKFold (seed 0). "
                    "Score = mean of 6 frozen-backbone class-balanced-LR OOF P(malignant). "
                    "Operating points read off the empirical ROC; net benefit and ECE use the "
                    "cross-fit temperature-calibrated probability.",
        "n": int(len(y)), "n_malignant": int(y.sum()), "prevalence": float(y.mean()),
        "n_vvi": int(vvi.sum()), "n_malignant_vvi": int(y[vvi].sum()),

        "operating_points": {
            "overall": operating_points(y, ens),
            "FST V-VI": operating_points(y[vvi], ens[vvi]),
        },

        "decision_curve": {
            "note": "NB = TP/n - FP/n * pt/(1-pt); delta_vs_best_default = NB_model - max(NB_treat_all,0).",
            "overall": net_benefit(y, cal, pts),
            "FST V-VI": net_benefit(y[vvi], cal[vvi], pts),
        },

        "calibration": {
            "method": "temperature scaling on ensemble logit; T fit by NLL, cross-fit over 5 folds "
                      "for an honest ECE (10-bin equal-width).",
            "temperature_cv_mean": round(T_mean_cv, 4),
            "temperature_full_fit": round(T_full, 4),
            "ece_overall_before": round(ece_before, 4),
            "ece_overall_after": round(ece_after, 4),
            "ece_vvi_before": round(ece_before_vvi, 4),
            "ece_vvi_after": round(ece_after_vvi, 4),
        },
    }

    out = ROOT / "eval" / "probe" / "ddi_clinical.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
