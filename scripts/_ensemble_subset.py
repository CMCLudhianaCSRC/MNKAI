"""ANALYSIS 4 — ENSEMBLE SUBSET EFFICIENCY (accuracy-vs-cost frontier).

For every non-empty subset of the 6 cached DDI backbones (63 subsets), evaluate the
score-average ensemble AUROC (overall + FST V-VI) using the SAME 5-fold stratified OOF
protocol as scripts/_ddi_ensemble_ci.py (seed 0). Then build the 'accuracy vs number of
models' frontier: for each ensemble size k=1..6, the BEST subset by V-VI AUROC and its
overall+V-VI AUROC. Identify the smallest subset that retains >= 0.86 V-VI AUROC
(deployment sweet spot).

Reuses the proven oof_scores() recipe: StandardScaler + class-balanced LogisticRegression
(C=0.5, max_iter=2000), StratifiedKFold(5, shuffle, random_state=0), strat = tone_label.

    ./.venv/Scripts/python.exe scripts/_ensemble_subset.py
"""
from __future__ import annotations
import json
from itertools import combinations
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
MEMBERS = ["dinov2_b", "dinov2_reg", "eva02_b", "convnextv2_b", "siglip_b", "dinov2_l"]
SWEET_THRESH = 0.86  # target V-VI AUROC to retain


def oof_scores(X, y, strat, seed=0):
    """5-fold stratified OOF P(malignant) — identical to _ddi_ensemble_ci.py."""
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


def main():
    meta = pd.read_csv(DDI / "ddi_metadata.csv")
    y = meta["malignant"].astype(int).to_numpy()
    tone = meta["skin_tone"].to_numpy()
    strat = (pd.Series(tone).astype(str) + "_" + pd.Series(y).astype(str)).to_numpy()
    feats = {m: np.load(CACHE / f"{m}.npy") for m in MEMBERS}
    n_img = len(y)

    # Per-backbone OOF scores computed ONCE; every subset ensemble = mean of member scores.
    per_member = {m: oof_scores(feats[m], y, strat, seed=0) for m in MEMBERS}

    # Evaluate all 63 non-empty subsets.
    all_subsets = []
    for k in range(1, len(MEMBERS) + 1):
        for combo in combinations(MEMBERS, k):
            ens = np.column_stack([per_member[m] for m in combo]).mean(1)
            a = auroc_by_tone(ens, y, tone)
            all_subsets.append({
                "k": k,
                "members": list(combo),
                "overall": round(a["overall"], 4),
                "FST V-VI": round(a["FST V-VI"], 4) if a["FST V-VI"] is not None else None,
                "FST I-II": round(a["FST I-II"], 4) if a["FST I-II"] is not None else None,
                "FST III-IV": round(a["FST III-IV"], 4) if a["FST III-IV"] is not None else None,
            })

    # Frontier: for each k, the best subset by V-VI AUROC (tie-break on overall).
    frontier = []
    for k in range(1, len(MEMBERS) + 1):
        cands = [s for s in all_subsets if s["k"] == k]
        best = max(cands, key=lambda s: (s["FST V-VI"], s["overall"]))
        # also record best-by-overall for reference
        best_ov = max(cands, key=lambda s: (s["overall"], s["FST V-VI"]))
        frontier.append({
            "k": k,
            "best_by_VVI": {"members": best["members"], "overall": best["overall"], "FST V-VI": best["FST V-VI"]},
            "best_by_overall": {"members": best_ov["members"], "overall": best_ov["overall"], "FST V-VI": best_ov["FST V-VI"]},
        })

    # Deployment sweet spot: smallest subset (fewest models) with V-VI >= threshold;
    # among equal-size candidates pick the highest V-VI.
    qualifying = [s for s in all_subsets if s["FST V-VI"] is not None and s["FST V-VI"] >= SWEET_THRESH]
    recommended = None
    if qualifying:
        min_k = min(s["k"] for s in qualifying)
        recommended = max([s for s in qualifying if s["k"] == min_k],
                          key=lambda s: (s["FST V-VI"], s["overall"]))

    full6 = next(s for s in all_subsets if s["k"] == 6)

    result = {
        "protocol": "DDI 656 images, image-level 5-fold StratifiedKFold (seed 0), score-average "
                    "ensemble of class-balanced LR OOF P(malignant) over cached frozen backbones. "
                    "Frontier = best subset per size k by FST V-VI AUROC.",
        "n_images": int(n_img),
        "n_malignant": int(y.sum()),
        "members": MEMBERS,
        "n_subsets_evaluated": len(all_subsets),
        "frontier": frontier,
        "full_ensemble_6": {"members": full6["members"], "overall": full6["overall"], "FST V-VI": full6["FST V-VI"]},
        "sweet_spot_threshold_VVI": SWEET_THRESH,
        "recommended_subset": ({
            "members": recommended["members"],
            "k": recommended["k"],
            "overall": recommended["overall"],
            "FST V-VI": recommended["FST V-VI"],
        } if recommended else None),
        "all_subsets": sorted(all_subsets, key=lambda s: (-(s["FST V-VI"] or 0), s["k"])),
    }

    out = ROOT / "eval" / "probe" / "ensemble_subset.json"
    out.write_text(json.dumps(result, indent=2))

    # console summary
    print("=== ENSEMBLE SUBSET EFFICIENCY FRONTIER (best per size, by V-VI) ===")
    print(f"{'k':>2}  {'overall':>8}  {'V-VI':>7}  members")
    for f in frontier:
        b = f["best_by_VVI"]
        print(f"{f['k']:>2}  {b['overall']:>8.4f}  {b['FST V-VI']:>7.4f}  {'+'.join(b['members'])}")
    print(f"\nfull 6-backbone : overall {full6['overall']:.4f}  V-VI {full6['FST V-VI']:.4f}")
    if recommended:
        print(f"\nRECOMMENDED (smallest subset >= {SWEET_THRESH} V-VI): k={recommended['k']}  "
              f"overall {recommended['overall']:.4f}  V-VI {recommended['FST V-VI']:.4f}")
        print(f"  members: {'+'.join(recommended['members'])}")
    else:
        print(f"\nNo subset reached V-VI >= {SWEET_THRESH}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
