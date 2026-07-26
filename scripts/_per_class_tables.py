"""Per-class performance tables for the 9-class and 25-class tasks.

The summary artifacts only store macro-averages; the approval document needs a
per-condition breakdown. Computes precision / recall (sensitivity) / F1 / support
per class, on cached frozen features only (no D:, no GPU, no retraining).

  9-class  : out-of-fold cross-validation (its external split is class-degenerate)
  25-class : out-of-fold, plus the de-duplicated external test

    python scripts/_per_class_tables.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts._optimize_thresholds import coordinate_ascent

SEED = 0


def load(npz):
    z = np.load(ROOT / npz, allow_pickle=True)
    return (z["features"].astype(np.float32), z["y"].astype(int),
            np.array([str(s) for s in z["split"]]), z["fold"].astype(int),
            [str(c) for c in z["classes"]])


def head():
    return MLPClassifier((512,), max_iter=300, early_stopping=True, alpha=1e-3, random_state=SEED)


def fit_pred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    h = head().fit(sc.transform(Xtr), ytr)
    return np.log(np.clip(h.predict_proba(sc.transform(Xte)), 1e-9, 1))


def oof(X, y, fold, tr, nC):
    Xt, yt, ft = X[tr], y[tr], fold[tr]
    out = np.full((len(yt), nC), -30.0, dtype=np.float32)
    for f in np.unique(ft):
        a, b = ft != f, ft == f
        lg = fit_pred(Xt[a], yt[a], Xt[b])
        cls = np.unique(yt[a])
        tmp = np.full((int(b.sum()), nC), -30.0, dtype=np.float32)
        tmp[:, cls] = lg
        out[b] = tmp
    return out, yt


def table(y_true, y_pred, classes):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(len(classes)), zero_division=0)
    rows = [{"class": classes[i], "precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
             "f1": round(float(f[i]), 3), "support": int(s[i])} for i in range(len(classes))]
    rows.sort(key=lambda x: -x["support"])
    return rows


def dup_mask(Xex, Xtr, thr=0.99, block=512):
    A = Xex / np.clip(np.linalg.norm(Xex, axis=1, keepdims=True), 1e-9, None)
    B = Xtr / np.clip(np.linalg.norm(Xtr, axis=1, keepdims=True), 1e-9, None)
    best = np.zeros(len(A), dtype=np.float32)
    for i in range(0, len(A), block):
        best[i:i + block] = (A[i:i + block] @ B.T).max(1)
    return best <= thr


res = {}

# ---------------- 9-class (OOF) ----------------
print("[9-class] out-of-fold ...", flush=True)
X, y, split, fold, classes = load("eval/probe/feats_9class_224.npz")
tr = split == "train_val"
lg, yt = oof(X, y, fold, tr, len(classes))
bias, _, _ = coordinate_ascent(lg, yt, len(classes))
res["task_9class_oof"] = {"n": int(tr.sum()), "n_classes": len(classes),
                          "per_class": table(yt, (lg + bias).argmax(1), classes)}
for r in res["task_9class_oof"]["per_class"]:
    print(f"   {r['class']:<28} n={r['support']:>6}  sens={r['recall']:.3f}  F1={r['f1']:.3f}", flush=True)

# ---------------- 25-class (OOF + clean external) ----------------
print("\n[25-class] out-of-fold + de-duplicated external ...", flush=True)
X, y, split, fold, classes = load("eval/probe/feats_25class_518.npz")
tr, ex = split == "train_val", split == "external_test"
lg, yt = oof(X, y, fold, tr, len(classes))
bias, _, _ = coordinate_ascent(lg, yt, len(classes))
res["task_25class_oof"] = {"n": int(tr.sum()), "n_classes": len(classes),
                           "per_class": table(yt, (lg + bias).argmax(1), classes)}

lg_ex = fit_pred(X[tr], y[tr], X[ex])
keep = dup_mask(X[ex], X[tr], 0.99)
yex, pex = y[ex][keep], (lg_ex + bias).argmax(1)[keep]
res["task_25class_external_clean"] = {"n": int(keep.sum()), "n_classes": len(classes),
                                      "per_class": table(yex, pex, classes)}
for r in res["task_25class_external_clean"]["per_class"]:
    print(f"   {r['class']:<28} n={r['support']:>5}  sens={r['recall']:.3f}  F1={r['f1']:.3f}", flush=True)

(ROOT / "eval/probe/per_class_tables.json").write_text(json.dumps(res, indent=2))
print("\nsaved -> eval/probe/per_class_tables.json")
