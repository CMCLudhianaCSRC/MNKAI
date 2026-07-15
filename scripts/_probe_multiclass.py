"""Probe cached frozen features for a multi-class task (25-class / 9-class):
OOF 5-fold + external, class-balanced logistic (or MLP) head, + per-class bias,
coarse readout, and per-Fitzpatrick fairness. Compares to the fine-tuned baseline.

    python scripts/_probe_multiclass.py --feats eval/probe/feats_25class_224.npz \
        --head lr --out eval/probe/probe_25class_224.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.special import log_softmax
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from scripts._optimize_thresholds import coordinate_ascent
from scripts._coarse_eval import COARSE


def make_head(kind, C):
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(512,), max_iter=300, early_stopping=True, alpha=1e-3)
    return LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)


def topk(prob, y, k):
    tk = np.argsort(-prob, axis=1)[:, :k]
    return float(np.mean([y[i] in tk[i] for i in range(len(y))]))


def coarse_block(prob, y, classes):
    names = list(COARSE.keys()); cg = {g: i for i, g in enumerate(names)}
    c2g = {m: cg[g] for g, ms in COARSE.items() for m in ms}
    grp = np.zeros((len(classes), len(names)), np.float32)
    for j, c in enumerate(classes):
        if c in c2g: grp[j, c2g[c]] = 1.0
    cp = prob @ grp; cy = np.array([c2g.get(classes[t], 0) for t in y])
    return {"acc": round(accuracy_score(cy, cp.argmax(1)), 4), "macro_f1": round(f1_score(cy, cp.argmax(1), average="macro"), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True); ap.add_argument("--head", default="lr")
    ap.add_argument("--out", required=True); ap.add_argument("--baseline", type=float, default=0.539)
    a = ap.parse_args()
    z = np.load(a.feats, allow_pickle=True)
    X = z["features"]; y = z["y"]; split = z["split"].astype(str); fold = z["fold"]; fst = z["fst"]
    classes = [str(c) for c in z["classes"]]; C = len(classes)
    tvm = split == "train_val"; exm = split == "external_test"
    Xtv, ytv, foldtv = X[tvm], y[tvm], fold[tvm]
    Xex, yex, fstex = X[exm], y[exm], fst[exm]
    print(f"[probe] train_val={tvm.sum()} external={exm.sum()} classes={C} head={a.head}", flush=True)

    # OOF 5-fold on train_val
    oof_prob = np.zeros((len(ytv), C))
    for fk in sorted(set(foldtv[foldtv >= 0])):
        tr = foldtv != fk; te = foldtv == fk
        sc = StandardScaler().fit(Xtv[tr]); clf = make_head(a.head, C).fit(sc.transform(Xtv[tr]), ytv[tr])
        oof_prob[te] = clf.predict_proba(sc.transform(Xtv[te]))
    oof_log = log_softmax(np.log(np.clip(oof_prob, 1e-9, 1)), axis=1)

    # external: train on all train_val
    sc = StandardScaler().fit(Xtv); clf = make_head(a.head, C).fit(sc.transform(Xtv), ytv)
    ex_prob = clf.predict_proba(sc.transform(Xex)); ex_log = log_softmax(np.log(np.clip(ex_prob, 1e-9, 1)), axis=1)

    bias, _, _ = coordinate_ascent(oof_log, ytv, C, rounds=3)
    res = {"head": a.head, "n_train_val": int(tvm.sum()), "n_external": int(exm.sum()),
           "OOF_macro_f1": round(f1_score(ytv, oof_prob.argmax(1), average="macro"), 4),
           "EXT_macro_f1_raw": round(f1_score(yex, ex_prob.argmax(1), average="macro"), 4),
           "EXT_macro_f1_bias": round(f1_score(yex, (ex_log + bias).argmax(1), average="macro"), 4),
           "EXT_top1": round(accuracy_score(yex, ex_prob.argmax(1)), 4),
           "EXT_top3": round(topk(ex_prob, yex, 3), 4),
           "EXT_balacc": round(balanced_accuracy_score(yex, ex_prob.argmax(1)), 4),
           "coarse_ext": coarse_block(ex_prob, yex, classes),
           "baseline_finetuned": a.baseline}
    # per-tone fairness (external)
    ft = {}
    for lbl, m in [("FST_I-III", (fstex <= 3) & (fstex >= 1)), ("FST_IV-VI", fstex >= 4)]:
        if m.sum() > 20:
            ft[lbl] = {"n": int(m.sum()), "bal_acc": round(balanced_accuracy_score(yex[m], (ex_log[m] + bias).argmax(1)), 4)}
    res["fairness_ext"] = ft
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\n===== FROZEN-PROBE multi-class ({a.head}) =====")
    print(f"  OOF macro-F1      {res['OOF_macro_f1']}")
    print(f"  EXT macro-F1 raw  {res['EXT_macro_f1_raw']}   +bias {res['EXT_macro_f1_bias']}   (fine-tuned baseline {a.baseline})")
    print(f"  EXT top1 {res['EXT_top1']} | top3 {res['EXT_top3']} | coarse {res['coarse_ext']['macro_f1']}")
    print(f"  saved -> {a.out}\n")


if __name__ == "__main__":
    main()
