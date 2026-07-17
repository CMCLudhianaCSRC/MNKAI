"""ANALYSIS 1 — statistical rigor on the two MAIN frozen-probe numbers.

Static / laptop-only: works from the CACHED feature .npz files, retrains NOTHING
on D:, touches no shipped model. For BOTH multi-class tasks it runs the standard
5-fold OOF MLP probe (recipe from MEMORY.md / scripts/_probe_multiclass.py) on the
train_val split using the frozen 'fold' column, then:

  * OOF macro-F1 POINT estimate.
  * Bootstrap 95% CI on OOF macro-F1 (>=2000 reps, per-true-class-STRATIFIED
    resampling so rare classes are never dropped from a replicate).
  * (25-class only) PAIRED bootstrap delta-macroF1 (518 - 224) on the SAME
    resampled indices -> 95% CI + one-sided p (H0: 518 <= 224).
  * frozen-probe vs fine-tune gap vs the known fine-tune point estimates.

    ./.venv/Scripts/python.exe scripts/_rigor_main_ci.py

Output: eval/probe/rigor_main_ci.json
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "eval" / "probe"
N_BOOT = 2000
RNG = np.random.default_rng(20260716)

# Known fine-tune point estimates (external, not a paired test) for gap reporting.
FINETUNE = {"25class": 0.527, "9class": 0.637}


def make_head():
    # Exact MEMORY.md recipe: MLP((512,), max_iter=300, early_stopping, alpha=1e-3, seed 0.
    return MLPClassifier(hidden_layer_sizes=(512,), max_iter=300, early_stopping=True,
                         alpha=1e-3, random_state=0)


def oof_preds(X, y, fold):
    """5-fold OOF hard predictions over folds 0..K on train_val."""
    pred = np.full(len(y), -1, dtype=np.int64)
    for fk in sorted(f for f in set(fold.tolist()) if f >= 0):
        tr, te = fold != fk, fold == fk
        sc = StandardScaler().fit(X[tr])
        clf = make_head().fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    assert (pred >= 0).all(), "some samples never predicted"
    return pred


def class_cells(y):
    return [np.where(y == c)[0] for c in np.unique(y)]


def strat_boot_indices(cells):
    """One per-true-class-stratified bootstrap resample -> concatenated index array."""
    return np.concatenate([RNG.choice(c, size=len(c), replace=True) for c in cells])


def boot_ci_macrof1(y, pred, cells, n=N_BOOT):
    vals = np.empty(n)
    for i in range(n):
        idx = strat_boot_indices(cells)
        vals[i] = f1_score(y[idx], pred[idx], average="macro")
    return {
        "point": float(f1_score(y, pred, average="macro")),
        "mean": float(vals.mean()),
        "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
        "n_boot": n,
    }


def paired_delta(y, pred_a, pred_b, cells, n=N_BOOT):
    """Paired bootstrap delta = macroF1(A) - macroF1(B) on SAME resampled indices."""
    d = np.empty(n)
    for i in range(n):
        idx = strat_boot_indices(cells)
        d[i] = (f1_score(y[idx], pred_a[idx], average="macro")
                - f1_score(y[idx], pred_b[idx], average="macro"))
    return {
        "delta_point": float(f1_score(y, pred_a, average="macro")
                             - f1_score(y, pred_b, average="macro")),
        "delta_mean": float(d.mean()),
        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "p_one_sided_A_le_B": float(np.mean(d <= 0)),  # H0: 518 <= 224
        "n_boot": n,
    }


def load_tv(name):
    z = np.load(PROBE / name, allow_pickle=True)
    tv = z["split"].astype(str) == "train_val"
    return z["features"][tv], z["y"][tv], z["fold"][tv], len(z["classes"])


def main():
    t0 = time.time()
    out = {"note": "5-fold OOF MLP frozen-probe (StandardScaler + MLP((512,), max_iter=300, "
                   "early_stopping, alpha=1e-3, seed 0)); train_val split, frozen 'fold' column. "
                   "CIs = per-true-class-stratified bootstrap on OOF (y_true,y_pred) pairs.",
           "n_boot": N_BOOT, "boot_seed": 20260716}

    # ---- 25-class @ 518 ----
    print("[25c/518] OOF probe ...", flush=True)
    X5, y5, f5, C5 = load_tv("feats_25class_518.npz")
    p518 = oof_preds(X5, y5, f5)
    cells25 = class_cells(y5)
    r518 = boot_ci_macrof1(y5, p518, cells25)
    r518.update(n_train_val=int(len(y5)), n_classes=int(C5),
                finetune_ref=FINETUNE["25class"],
                gap_vs_finetune=round(r518["point"] - FINETUNE["25class"], 4))
    print(f"  25c@518 OOF macro-F1 {r518['point']:.4f}  CI {r518['ci95']}", flush=True)

    # ---- 25-class @ 224 (for paired delta) ----
    print("[25c/224] OOF probe ...", flush=True)
    X2, y2, f2, _ = load_tv("feats_25class_224.npz")
    assert np.array_equal(y5, y2) and np.array_equal(f5, f2), "518/224 rows must align"
    p224 = oof_preds(X2, y2, f2)
    r224 = boot_ci_macrof1(y2, p224, cells25)
    print(f"  25c@224 OOF macro-F1 {r224['point']:.4f}  CI {r224['ci95']}", flush=True)

    print("[25c] paired 518-224 delta ...", flush=True)
    delta = paired_delta(y5, p518, p224, cells25)
    print(f"  delta {delta['delta_point']:+.4f}  CI {delta['ci95']}  p(518<=224)={delta['p_one_sided_A_le_B']:.4f}",
          flush=True)

    # ---- 9-class @ 224 ----
    print("[9c/224] OOF probe (large, be patient) ...", flush=True)
    X9, y9, f9, C9 = load_tv("feats_9class_224.npz")
    p9 = oof_preds(X9, y9, f9)
    cells9 = class_cells(y9)
    r9 = boot_ci_macrof1(y9, p9, cells9)
    r9.update(n_train_val=int(len(y9)), n_classes=int(C9),
              finetune_ref=FINETUNE["9class"],
              gap_vs_finetune=round(r9["point"] - FINETUNE["9class"], 4))
    print(f"  9c@224 OOF macro-F1 {r9['point']:.4f}  CI {r9['ci95']}", flush=True)

    out["task_25class_518"] = r518
    out["task_25class_224"] = r224
    out["paired_518_minus_224_25class"] = delta
    out["task_9class_224"] = r9
    out["elapsed_sec"] = round(time.time() - t0, 1)

    dst = PROBE / "rigor_main_ci.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {dst}  ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
