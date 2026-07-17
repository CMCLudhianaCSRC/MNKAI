"""DDI ensemble — statistical hardening (R03 + R04) on CACHED features (no D:, no re-extract).

For the frozen multi-backbone ensemble that scored 0.826 overall / 0.872 V-VI, add:
  1. Stratified bootstrap 95% CIs on AUROC (overall + per Fitzpatrick tone).
  2. Multi-seed CV stability (fold-randomisation seeds 0..4).
  3. Paired bootstrap Delta-AUROC (ensemble - single-DINOv2 baseline) with 95% CI + one-sided p.

DDI is one biopsy-confirmed image per lesion; no patient IDs are released, so image-level
StratifiedKFold is the standard protocol (Daneshjou et al. 2022) — patient-grouped CV is N/A.

    python scripts/_ddi_ensemble_ci.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "eval" / "probe" / "ddi_backbones"
DDI = Path("D:/AIFINAL/images/ddi")
TONE = {12: "FST I-II", 34: "FST III-IV", 56: "FST V-VI"}
# ensemble members (same set as scripts/_ddi_ensemble.py)
MEMBERS = ["dinov2_b", "dinov2_reg", "eva02_b", "convnextv2_b", "siglip_b", "dinov2_l"]
BASELINE = "dinov2_b"          # the original single-backbone baseline (0.748 / 0.789)
N_BOOT = 4000
RNG = np.random.default_rng(12345)


def oof_scores(X, y, strat, seed=0):
    """Reproduce the 5-fold stratified OOF P(malignant) used by the ensemble script."""
    s = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, strat):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5, random_state=0)
        clf.fit(sc.transform(X[tr]), y[tr])
        s[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return s


def auroc_by_tone(score, y, tone):
    out = {"overall": float(roc_auc_score(y, score))}
    for k, nm in TONE.items():
        m = tone == k
        out[nm] = float(roc_auc_score(y[m], score[m])) if len(np.unique(y[m])) > 1 else None
    return out


def strat_boot_ci(score, y, tone, n=N_BOOT):
    """Stratified bootstrap (resample within tone x label cells) -> 95% CI for each AUROC."""
    cells = {}
    key = tone.astype(int) * 10 + y.astype(int)
    for c in np.unique(key):
        cells[c] = np.where(key == c)[0]
    keys = list(cells)
    boot = {k: [] for k in ["overall", *TONE.values()]}
    for _ in range(n):
        idx = np.concatenate([RNG.choice(cells[c], size=len(cells[c]), replace=True) for c in keys])
        bs, by, bt = score[idx], y[idx], tone[idx]
        boot["overall"].append(roc_auc_score(by, bs))
        for kk, nm in TONE.items():
            m = bt == kk
            if len(np.unique(by[m])) > 1:
                boot[nm].append(roc_auc_score(by[m], bs[m]))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items() if v}


def paired_delta(score_a, score_b, y, tone, n=N_BOOT):
    """Paired bootstrap Delta-AUROC = A - B on the SAME resampled indices (overall + per tone)."""
    key = tone.astype(int) * 10 + y.astype(int)
    cells = {c: np.where(key == c)[0] for c in np.unique(key)}
    keys = list(cells)
    out = {}
    for label, mask in [("overall", np.ones(len(y), bool))] + [(nm, tone == kk) for kk, nm in TONE.items()]:
        deltas = []
        for _ in range(n):
            idx = np.concatenate([RNG.choice(cells[c], size=len(cells[c]), replace=True) for c in keys])
            idx = idx[mask[idx]]
            if len(np.unique(y[idx])) < 2:
                continue
            deltas.append(roc_auc_score(y[idx], score_a[idx]) - roc_auc_score(y[idx], score_b[idx]))
        deltas = np.asarray(deltas)
        out[label] = {
            "delta_auroc": float(np.mean(deltas)),
            "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            "p_one_sided": float(np.mean(deltas <= 0)),  # H0: ensemble <= baseline
        }
    return out


def main():
    meta = pd.read_csv(DDI / "ddi_metadata.csv")
    y = meta["malignant"].astype(int).to_numpy()
    tone = meta["skin_tone"].to_numpy()
    strat = (pd.Series(tone).astype(str) + "_" + pd.Series(y).astype(str)).to_numpy()
    feats = {m: np.load(CACHE / f"{m}.npy") for m in MEMBERS}

    # point-estimate OOF scores (seed 0, matches the ensemble script)
    per_member = {m: oof_scores(feats[m], y, strat, seed=0) for m in MEMBERS}
    ens = np.column_stack([per_member[m] for m in MEMBERS]).mean(1)   # score-average ensemble
    base = per_member[BASELINE]

    result = {
        "protocol": "DDI 656 images, one biopsy-confirmed lesion each, no patient IDs -> image-level "
                    "5-fold StratifiedKFold (Daneshjou et al. 2022). Ensemble = mean of 6 frozen-backbone "
                    "class-balanced LR OOF P(malignant).",
        "point_estimate": {
            "ensemble_scoreavg": auroc_by_tone(ens, y, tone),
            "baseline_dinov2_b": auroc_by_tone(base, y, tone),
        },
        "bootstrap_ci95": {
            "ensemble_scoreavg": strat_boot_ci(ens, y, tone),
            "baseline_dinov2_b": strat_boot_ci(base, y, tone),
        },
        "delta_ensemble_minus_baseline": paired_delta(ens, base, y, tone),
        "multiseed_stability": {},
    }

    # multi-seed CV stability for the ensemble overall + V-VI
    for label in ["overall", "FST V-VI"]:
        vals = []
        for seed in range(5):
            pm = np.column_stack([oof_scores(feats[m], y, strat, seed=seed) for m in MEMBERS]).mean(1)
            a = auroc_by_tone(pm, y, tone)
            vals.append(a[label])
        result["multiseed_stability"][label] = {
            "seeds_0_4": [round(v, 4) for v in vals],
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
        }

    (ROOT / "eval" / "probe" / "ddi_ensemble_ci.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("\nsaved -> eval/probe/ddi_ensemble_ci.json")


if __name__ == "__main__":
    main()
