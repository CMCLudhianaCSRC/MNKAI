"""Canonical HONEST 25-class eval: de-duplicated external test.

The 25-class external had ~33% exact train-duplicates (cross-source atlas images).
We flag external images whose nearest train image (frozen-feature cosine) exceeds a
duplicate threshold, DROP them, and re-report BOTH the frozen-probe and cRT on the
clean held-out set, with bootstrap CIs. Saves the clean-external filepaths for reuse.

    python scripts/_clean_eval_25class.py --feats eval/probe/feats_25class_518.npz \
        --crt eval/phase3_crt_full/preds_5fold.npz --config configs/phase3_indian_25class.yaml \
        --dup-thr 0.99 --out eval/probe/CLEAN_25class.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import log_softmax
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from scripts._optimize_thresholds import coordinate_ascent
from mnkai.data.manifest import LabelNormalizer, normalize_manifest
from mnkai.utils.config import load_config


def boot_ci(y, pred, B=1000, seed=0):
    rng = np.random.default_rng(seed); n = len(y); vals = []
    for _ in range(B):
        bi = rng.integers(0, n, n); vals.append(f1_score(y[bi], pred[bi], average="macro"))
    return round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True); ap.add_argument("--crt", required=True)
    ap.add_argument("--config", required=True); ap.add_argument("--dup-thr", type=float, default=0.99)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    np.random.seed(0)
    z = np.load(a.feats, allow_pickle=True)
    X = z["features"].astype(np.float32); y = z["y"]; split = z["split"].astype(str); fold = z["fold"]; fst = z["fst"]
    classes = [str(c) for c in z["classes"]]; C = len(classes)
    tv = split == "train_val"; ex = split == "external_test"
    Xtv, ytv, foldtv = X[tv], y[tv], fold[tv]; Xex, yex, fstex = X[ex], y[ex], fst[ex]

    # external near-dup mask via cosine to nearest train image
    tvn = Xtv / (np.linalg.norm(Xtv, axis=1, keepdims=True) + 1e-8)
    exn = Xex / (np.linalg.norm(Xex, axis=1, keepdims=True) + 1e-8)
    mx = np.zeros(len(exn))
    for i in range(0, len(exn), 256): mx[i:i+256] = (exn[i:i+256] @ tvn.T).max(1)
    clean = mx <= a.dup_thr
    print(f"[clean] external {len(yex)} -> clean {clean.sum()} (dropped {(~clean).sum()} dups > {a.dup_thr})", flush=True)

    # frozen-probe: OOF (for bias) + external preds
    oof = np.zeros((len(ytv), C))
    for fk in sorted(set(foldtv[foldtv >= 0])):
        tr = foldtv != fk; te = foldtv == fk
        sc = StandardScaler().fit(Xtv[tr]); m = MLPClassifier((512,), max_iter=300, early_stopping=True, alpha=1e-3, random_state=0).fit(sc.transform(Xtv[tr]), ytv[tr])
        oof[te] = m.predict_proba(sc.transform(Xtv[te]))
    bias, _, _ = coordinate_ascent(log_softmax(np.log(np.clip(oof, 1e-9, 1)), 1), ytv, C, rounds=3)
    sc = StandardScaler().fit(Xtv); m = MLPClassifier((512,), max_iter=300, early_stopping=True, alpha=1e-3, random_state=0).fit(sc.transform(Xtv), ytv)
    pb_log = log_softmax(np.log(np.clip(m.predict_proba(sc.transform(Xex)), 1e-9, 1)), 1)
    pb = (pb_log + bias).argmax(1)

    # cRT: external preds + its own OOF bias
    cr = np.load(a.crt, allow_pickle=True)
    assert (cr["test_y"] == yex).mean() > 0.99, "cRT external misaligned"
    cr_oof_log = log_softmax(cr["oof_logits"].astype(np.float64), 1)
    cr_bias, _, _ = coordinate_ascent(cr_oof_log, cr["oof_y"], C, rounds=3)
    cr_ex_log = log_softmax(cr["test_logits"].mean(0).astype(np.float64), 1)
    crp = (cr_ex_log + cr_bias).argmax(1)

    def block(pred, mask, name):
        yy = yex[mask]; pp = pred[mask]
        lo, hi = boot_ci(yy, pp)
        return {"model": name, "n": int(mask.sum()), "macro_f1": round(f1_score(yy, pp, average="macro"), 4),
                "macro_f1_ci95": [lo, hi], "top1": round(accuracy_score(yy, pp), 4),
                "bal_acc": round(balanced_accuracy_score(yy, pp), 4)}

    full = np.ones(len(yex), bool)
    res = {"dup_threshold": a.dup_thr, "n_external_full": int(len(yex)), "n_external_clean": int(clean.sum()),
           "pct_duplicates": round(float((~clean).mean()), 3),
           "frozen_probe": {"FULL": block(pb, full, "frozen-probe 518+MLP+bias"), "CLEAN": block(pb, clean, "frozen-probe 518+MLP+bias")},
           "cRT": {"FULL": block(crp, full, "cRT+bias"), "CLEAN": block(crp, clean, "cRT+bias")}}
    # per-tone fairness on CLEAN (frozen-probe)
    tone = {}
    for lbl, mm in [("FST_I-III", clean & (fstex <= 3) & (fstex >= 1)), ("FST_IV-VI", clean & (fstex >= 4))]:
        if mm.sum() > 20:
            tone[lbl] = {"n": int(mm.sum()), "bal_acc": round(balanced_accuracy_score(yex[mm], pb[mm]), 4)}
    res["fairness_clean_probe"] = tone

    # save clean-external filepaths for reproducibility
    man = normalize_manifest(pd.read_csv(load_config(a.config)["data"]["manifest_csv"], low_memory=False),
                             LabelNormalizer(classes=classes))
    sp = pd.read_csv(load_config(a.config)["data"]["splits_csv"], low_memory=False)
    k = "filepath" if "filepath" in sp.columns else "isic_id"
    man = man.merge(sp[[k, "split", "fold"]], on=k, how="inner").reset_index(drop=True)
    exfp = man.loc[man["split"] == "external_test", "filepath"].values
    if len(exfp) == len(clean):
        pd.DataFrame({"filepath": exfp[clean]}).to_csv(str(Path(a.out).with_suffix(".clean_external.csv")), index=False)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=2))
    print("\n===== CANONICAL 25-class (de-duplicated external) =====")
    print(f"  duplicates dropped: {res['pct_duplicates']*100:.0f}%  (clean n={res['n_external_clean']})")
    for mdl in ("frozen_probe", "cRT"):
        c = res[mdl]["CLEAN"]; f = res[mdl]["FULL"]
        print(f"  {c['model']:26s}: CLEAN macro-F1 {c['macro_f1']} {c['macro_f1_ci95']}  (leaky FULL {f['macro_f1']}) | top1 {c['top1']}")
    print(f"  fairness (clean, probe): {tone}")
    print(f"  saved -> {a.out}\n")


if __name__ == "__main__":
    main()
