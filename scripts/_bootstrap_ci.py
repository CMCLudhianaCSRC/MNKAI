"""Bootstrap 95% CIs for the headline external-test metrics (publishability).

Resamples the external test set with replacement (B=2000) and reports 2.5/50/97.5
percentiles for: accuracy, macro-F1, balanced-acc, top-3 acc, and the melanoma /
malignant recall + AUROC. Point estimates alone read as amateur; CIs are expected.

    python scripts/_bootstrap_ci.py --preds eval/phase3_fixC_full/preds.npz \
        --B 2000 --out eval/phase3_fixC_full/bootstrap_ci.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, roc_auc_score

MALIGNANT = ["mel", "bcc", "scc", "ak"]


def topk_hit(prob, y, k):
    tk = np.argsort(-prob, axis=1)[:, :k]
    return np.array([y[i] in tk[i] for i in range(len(y))])


def ci(vals):
    v = np.array([x for x in vals if x is not None and not np.isnan(x)])
    return {"lo": round(float(np.percentile(v, 2.5)), 4),
            "med": round(float(np.percentile(v, 50)), 4),
            "hi": round(float(np.percentile(v, 97.5)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    z = np.load(args.preds, allow_pickle=True)
    classes = [str(c) for c in z["classes"]]
    prob = z["test_prob_ens"]; y = z["test_y"]; n = len(y)
    pred = prob.argmax(1)
    mal_idx = [classes.index(c) for c in MALIGNANT if c in classes]
    mel_i = classes.index("mel")
    mal_score = prob[:, mal_idx].sum(1)
    is_mal = np.isin(y, mal_idx).astype(int)
    hit3 = topk_hit(prob, y, 3)

    acc, mf1, bacc, t3, mel_rec, mal_auroc, mel_auroc = [], [], [], [], [], [], []
    for _ in range(args.B):
        bi = rng.integers(0, n, n)
        acc.append(accuracy_score(y[bi], pred[bi]))
        mf1.append(f1_score(y[bi], pred[bi], average="macro"))
        bacc.append(balanced_accuracy_score(y[bi], pred[bi]))
        t3.append(hit3[bi].mean())
        # melanoma recall
        m = (y[bi] == mel_i)
        mel_rec.append((pred[bi][m] == mel_i).mean() if m.sum() > 0 else np.nan)
        # AUROCs
        try: mal_auroc.append(roc_auc_score(is_mal[bi], mal_score[bi]))
        except Exception: mal_auroc.append(np.nan)
        try: mel_auroc.append(roc_auc_score((y[bi] == mel_i).astype(int), prob[bi, mel_i]))
        except Exception: mel_auroc.append(np.nan)

    res = {"B": args.B, "n": n,
           "point": {"accuracy": round(accuracy_score(y, pred), 4),
                     "macro_f1": round(f1_score(y, pred, average="macro"), 4),
                     "balanced_acc": round(balanced_accuracy_score(y, pred), 4),
                     "top3_acc": round(topk_hit(prob, y, 3).mean(), 4)},
           "ci95": {"accuracy": ci(acc), "macro_f1": ci(mf1), "balanced_acc": ci(bacc),
                    "top3_acc": ci(t3), "melanoma_recall": ci(mel_rec),
                    "malignant_auroc": ci(mal_auroc), "melanoma_auroc": ci(mel_auroc)}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("\n===== bootstrap 95% CIs (external test) =====")
    for k, d in res["ci95"].items():
        print(f"  {k:18s}: {d['med']:.3f}  [{d['lo']:.3f}, {d['hi']:.3f}]")
    print(f"  saved -> {args.out}\n")


if __name__ == "__main__":
    main()
