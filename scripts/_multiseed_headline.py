"""Multi-seed re-issue of the headline macro-F1 numbers (last laptop-feasible roadmap item).

Single-run numbers capture test-set sampling variance (via bootstrap CIs) but NOT
training variance. This re-runs the whole probe pipeline under several seeds and reports
mean +- SD, so the paper can state both.

Cached features only: no D:, no GPU, no re-extraction.
  25-class (33,250 train)  : seeds 0-4      -> feasible
  9-class  (197,137 train) : seeds 0-2      -> slow, fewer seeds by design

    python scripts/_multiseed_headline.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts._optimize_thresholds import coordinate_ascent


def load(npz):
    z = np.load(ROOT / npz, allow_pickle=True)
    return (z["features"].astype(np.float32), z["y"].astype(int),
            np.array([str(s) for s in z["split"]]), z["fold"].astype(int),
            [str(c) for c in z["classes"]])


def fit_pred(Xtr, ytr, Xte, seed):
    sc = StandardScaler().fit(Xtr)
    h = MLPClassifier((512,), max_iter=300, early_stopping=True, alpha=1e-3,
                      random_state=seed).fit(sc.transform(Xtr), ytr)
    return np.log(np.clip(h.predict_proba(sc.transform(Xte)), 1e-9, 1))


def oof(X, y, fold, tr, nC, seed):
    Xt, yt, ft = X[tr], y[tr], fold[tr]
    out = np.full((len(yt), nC), -30.0, dtype=np.float32)
    for f in np.unique(ft):
        a, b = ft != f, ft == f
        lg = fit_pred(Xt[a], yt[a], Xt[b], seed)
        cls = np.unique(yt[a])
        tmp = np.full((int(b.sum()), nC), -30.0, dtype=np.float32)
        tmp[:, cls] = lg
        out[b] = tmp
    return out, yt


def dup_mask(Xex, Xtr, thr=0.99, block=512):
    A = Xex / np.clip(np.linalg.norm(Xex, axis=1, keepdims=True), 1e-9, None)
    B = Xtr / np.clip(np.linalg.norm(Xtr, axis=1, keepdims=True), 1e-9, None)
    best = np.zeros(len(A), dtype=np.float32)
    for i in range(0, len(A), block):
        best[i:i+block] = (A[i:i+block] @ B.T).max(1)
    return best <= thr


def run(npz, seeds, with_external, label):
    X, y, split, fold, classes = load(npz)
    tr = split == "train_val"; nC = len(classes)
    ex = split == "external_test"
    keep = dup_mask(X[ex], X[tr]) if with_external else None
    oofs, exts = [], []
    for s in seeds:
        t0 = time.time()
        lg, yt = oof(X, y, fold, tr, nC, s)
        bias, _, _ = coordinate_ascent(lg, yt, nC)
        oofs.append(float(f1_score(yt, (lg + bias).argmax(1), average="macro")))
        if with_external:
            lge = fit_pred(X[tr], y[tr], X[ex], s)
            pe = (lge + bias).argmax(1)
            exts.append(float(f1_score(y[ex][keep], pe[keep], average="macro")))
        print(f"  [{label}] seed {s}: OOF {oofs[-1]:.4f}"
              + (f"  EXT(clean) {exts[-1]:.4f}" if with_external else "")
              + f"   ({time.time()-t0:.0f}s)", flush=True)
    out = {"seeds": list(seeds), "n_train_val": int(tr.sum()), "n_classes": nC,
           "oof_macro_f1": {"values": [round(v, 4) for v in oofs],
                            "mean": round(float(np.mean(oofs)), 4),
                            "sd": round(float(np.std(oofs, ddof=1)), 4) if len(oofs) > 1 else None}}
    if with_external:
        out["external_clean_macro_f1"] = {"values": [round(v, 4) for v in exts],
                                          "n": int(keep.sum()),
                                          "mean": round(float(np.mean(exts)), 4),
                                          "sd": round(float(np.std(exts, ddof=1)), 4) if len(exts) > 1 else None}
    return out


res = {"note": "Multi-seed re-issue: training variance of the headline macro-F1. "
               "Seed varies the MLP head init AND the per-class bias fit; folds and "
               "features are fixed. Complements the bootstrap CIs, which capture "
               "test-set sampling variance instead."}
print("[25-class] seeds 0-4 ...", flush=True)
res["task_25class_518"] = run("eval/probe/feats_25class_518.npz", range(5), True, "25c")
(ROOT / "eval/probe/multiseed_headline.json").write_text(json.dumps(res, indent=2))
print("\n[9-class] seeds 0-2 (slow) ...", flush=True)
res["task_9class_224"] = run("eval/probe/feats_9class_224.npz", range(3), False, "9c")
(ROOT / "eval/probe/multiseed_headline.json").write_text(json.dumps(res, indent=2))
print("\nsaved -> eval/probe/multiseed_headline.json")
