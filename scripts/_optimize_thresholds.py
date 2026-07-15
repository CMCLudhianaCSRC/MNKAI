"""Per-class logit-bias optimization (post-hoc, macro-F1) — Tier-0 A2.

Coordinate-ascent a per-class additive bias on the OOF predictions to maximize
OOF macro-F1, FREEZE it, then apply to the external test. This is mathematically
post-hoc logit adjustment; it trades head-class precision for tail recall and is
the cheapest single lever for macro-F1. Tuned ONLY on OOF; external stays honest.

    python scripts/_optimize_thresholds.py --preds eval/phase3_fixC_full/preds.npz \
        --out eval/phase3_fixC_full/threshold_opt.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.special import log_softmax
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score


def macro_f1(logscore, bias, y):
    pred = (logscore + bias).argmax(1)
    return f1_score(y, pred, average="macro")


def coordinate_ascent(logscore, y, C, rounds=4, grid=None):
    if grid is None:
        grid = np.linspace(-3.0, 3.0, 61)
    bias = np.zeros(C)
    best = macro_f1(logscore, bias, y)
    hist = [round(best, 4)]
    for r in range(rounds):
        # order classes by support ascending (help rare classes first)
        order = np.argsort(np.bincount(y, minlength=C))
        for c in order:
            base = bias.copy()
            best_b, best_f = bias[c], best
            for v in grid:
                base[c] = v
                f = macro_f1(logscore, base, y)
                if f > best_f:
                    best_f, best_b = f, v
            bias[c] = best_b
            best = best_f
        hist.append(round(best, 4))
    return bias, best, hist


def report(logscore, bias, y, classes):
    pred0 = logscore.argmax(1)
    pred1 = (logscore + bias).argmax(1)
    return {
        "macro_f1_before": round(f1_score(y, pred0, average="macro"), 4),
        "macro_f1_after": round(f1_score(y, pred1, average="macro"), 4),
        "acc_before": round(accuracy_score(y, pred0), 4),
        "acc_after": round(accuracy_score(y, pred1), 4),
        "balacc_before": round(balanced_accuracy_score(y, pred0), 4),
        "balacc_after": round(balanced_accuracy_score(y, pred1), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    z = np.load(args.preds, allow_pickle=True)
    classes = [str(c) for c in z["classes"]]; C = len(classes)

    oof_log = log_softmax(z["oof_logits"].astype(np.float64), axis=1)
    oof_y = z["oof_y"]
    test_log = log_softmax(z["test_logits"].mean(0).astype(np.float64), axis=1)
    test_y = z["test_y"]

    bias, oof_best, hist = coordinate_ascent(oof_log, oof_y, C)
    res = {"classes": classes, "bias": [round(float(b), 3) for b in bias],
           "oof_macro_f1_trajectory": hist,
           "OOF": report(oof_log, bias, oof_y, classes),
           "EXTERNAL": report(test_log, bias, test_y, classes)}

    # per-class F1 delta on external
    p0 = test_log.argmax(1); p1 = (test_log + bias).argmax(1)
    f0 = f1_score(test_y, p0, average=None, labels=range(C))
    f1v = f1_score(test_y, p1, average=None, labels=range(C))
    res["external_per_class_delta"] = sorted(
        [{"class": classes[i], "f1_before": round(float(f0[i]), 3),
          "f1_after": round(float(f1v[i]), 3), "delta": round(float(f1v[i] - f0[i]), 3)}
         for i in range(C)], key=lambda d: d["delta"], reverse=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    e = res["EXTERNAL"]
    print("\n===== per-class bias optimization (tuned on OOF, applied to EXTERNAL) =====")
    print(f"  OOF macro-F1 : {res['OOF']['macro_f1_before']:.4f} -> {res['OOF']['macro_f1_after']:.4f}")
    print(f"  EXT macro-F1 : {e['macro_f1_before']:.4f} -> {e['macro_f1_after']:.4f}  "
          f"(bal-acc {e['balacc_before']:.3f} -> {e['balacc_after']:.3f}, "
          f"acc {e['acc_before']:.3f} -> {e['acc_after']:.3f})")
    print("  biggest external per-class F1 gains:")
    for d in res["external_per_class_delta"][:6]:
        print(f"    {d['class']:22s} {d['f1_before']:.3f} -> {d['f1_after']:.3f}  ({d['delta']:+.3f})")
    print(f"  saved -> {args.out}\n")


if __name__ == "__main__":
    main()
