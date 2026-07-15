"""Coarse-taxonomy, binary-cancer, and top-k evaluation from dumped predictions.

Answers: "are we actually competitive with DermaCon-IN?" — by reporting the SAME
kind of numbers they headline (coarse super-class accuracy) instead of our harder
25-way fine macro-F1. Also reports the honest 'high-accuracy' framings:
  - coarse ~7 super-class accuracy + macro-F1  (proper prob-aggregation classifier)
  - binary malignant-vs-benign AUROC + sens/spec
  - fine-grained top-1 / top-3 / top-5

    python scripts/_coarse_eval.py --preds eval/phase3_fixC_full/preds.npz \
        --out eval/phase3_fixC_full/coarse_eval.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, roc_auc_score

# 25 fine classes -> 7 clinically-defensible coarse super-groups (Rook-style).
COARSE = {
    "neoplastic_malignant": ["mel", "bcc", "scc", "ak"],
    "neoplastic_benign":    ["nv", "bkl", "df", "vasc"],
    "inflammatory":         ["eczema", "psoriasis", "lichen_planus", "contact_dermatitis",
                             "drug_eruption", "rosacea", "acne", "urticaria"],
    "infectious":           ["bacterial_infection", "viral_infection", "tinea", "scabies"],
    "autoimmune_bullous":   ["pemphigus", "autoimmune_systemic"],
    "pigmentary_adnexal":   ["vitiligo", "alopecia"],
    "normal":               ["normal"],
}
MALIGNANT = ["mel", "bcc", "scc", "ak"]   # ak = pre-malignant


def topk_acc(prob, y, k):
    topk = np.argsort(-prob, axis=1)[:, :k]
    return float(np.mean([y[i] in topk[i] for i in range(len(y))]))


def coarse_block(prob, y, classes):
    cls_to_coarse = {}
    coarse_names = list(COARSE.keys())
    cgidx = {g: i for i, g in enumerate(coarse_names)}
    for g, members in COARSE.items():
        for m in members:
            cls_to_coarse[m] = cgidx[g]
    # group matrix [C, G]
    C, G = len(classes), len(coarse_names)
    grp = np.zeros((C, G), dtype=np.float32)
    for j, c in enumerate(classes):
        grp[j, cls_to_coarse[c]] = 1.0
    coarse_prob = prob @ grp              # aggregate fine probs into coarse groups
    coarse_pred = coarse_prob.argmax(1)
    coarse_y = np.array([cls_to_coarse[classes[t]] for t in y])
    return {
        "n_groups": G, "groups": coarse_names,
        "accuracy": round(accuracy_score(coarse_y, coarse_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(coarse_y, coarse_pred), 4),
        "macro_f1": round(f1_score(coarse_y, coarse_pred, average="macro"), 4),
    }


def binary_cancer(prob, y, classes):
    mal_idx = [classes.index(c) for c in MALIGNANT if c in classes]
    score = prob[:, mal_idx].sum(1)                 # P(malignant)
    is_mal = np.isin(y, mal_idx).astype(int)
    out = {"n_malignant": int(is_mal.sum()), "n_benign": int((1 - is_mal).sum())}
    try:
        out["auroc"] = round(roc_auc_score(is_mal, score), 4)
    except Exception:
        out["auroc"] = None
    # sens/spec at a few thresholds
    for thr in (0.3, 0.4, 0.5):
        pred = (score >= thr).astype(int)
        tp = int(((pred == 1) & (is_mal == 1)).sum()); fn = int(((pred == 0) & (is_mal == 1)).sum())
        tn = int(((pred == 0) & (is_mal == 0)).sum()); fp = int(((pred == 1) & (is_mal == 0)).sum())
        sens = tp / (tp + fn + 1e-9); spec = tn / (tn + fp + 1e-9)
        out[f"thr_{thr}"] = {"sensitivity": round(sens, 4), "specificity": round(spec, 4)}
    return out


def block(prob, y, classes, name):
    pred = prob.argmax(1)
    return {
        "split": name, "n": int(len(y)),
        "fine_top1_acc": round(accuracy_score(y, pred), 4),
        "fine_top3_acc": round(topk_acc(prob, y, 3), 4),
        "fine_top5_acc": round(topk_acc(prob, y, 5), 4),
        "fine_macro_f1": round(f1_score(y, pred, average="macro"), 4),
        "fine_balanced_acc": round(balanced_accuracy_score(y, pred), 4),
        "coarse": coarse_block(prob, y, classes),
        "binary_cancer": binary_cancer(prob, y, classes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    z = np.load(args.preds, allow_pickle=True)
    classes = [str(c) for c in z["classes"]]
    res = {"classes": classes, "coarse_scheme": {g: m for g, m in COARSE.items()}}
    res["external"] = block(z["test_prob_ens"], z["test_y"], classes, "external_test")
    res["oof"] = block(z["oof_prob"], z["oof_y"], classes, "oof")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))

    e = res["external"]
    print("\n===== MNKAI-III — coarse / binary / top-k (EXTERNAL TEST) =====")
    print(f"  fine 25-way : top1 {e['fine_top1_acc']:.3f} | top3 {e['fine_top3_acc']:.3f} | "
          f"top5 {e['fine_top5_acc']:.3f} | macroF1 {e['fine_macro_f1']:.3f}")
    print(f"  coarse {e['coarse']['n_groups']}-grp: acc {e['coarse']['accuracy']:.3f} | "
          f"macroF1 {e['coarse']['macro_f1']:.3f}  <-- compare to DermaCon-IN ~0.70 coarse acc")
    bc = e["binary_cancer"]
    print(f"  binary cancer: AUROC {bc['auroc']} | @0.3 sens {bc['thr_0.3']['sensitivity']:.3f} "
          f"spec {bc['thr_0.3']['specificity']:.3f}")
    print(f"  saved -> {args.out}\n")


if __name__ == "__main__":
    main()
