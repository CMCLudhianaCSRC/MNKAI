"""R05 - WINNER'S-CURSE-CORRECTED external evaluation (25-class frozen probe).

The previously-reported external macro-F1 (0.573) was obtained after choosing the
best config (img_size 518 vs 224, head MLP vs LR) partly by looking at external
performance -> optimistic model-selection bias. This script fixes that:

  1. For each of 4 configs {224,518} x {lr,mlp}: fit the probe recipe on train_val
     ONLY, compute 5-fold OOF macro-F1, and tune the per-class bias ONLY on OOF.
     This OOF macro-F1 is the SELECTION metric. External is NEVER looked at here.
  2. SELECT the single config with the best OOF macro-F1. Then, exactly once,
     evaluate THAT config on the untouched external_test split -> honest external
     macro-F1 (+ stratified bootstrap 95% CI).
  3. Conformal: hold out a calibration slice from train_val, select (k_reg,lambda)
     + qhat on it, FREEZE, evaluate coverage/set-size once on external.
  4. Write eval/probe/r05_winners_curse.json.

Recipe matches scripts/_probe_multiclass.py exactly (R02 random_state=0 applied).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
from scipy.special import log_softmax, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from scripts._optimize_thresholds import coordinate_ascent
from scripts._phase3_eval_full import fit_temperature
from scripts._conformal_fix import raps_scores, qhat_of, coverage_and_size

FEATS = {224: "eval/probe/feats_25class_224.npz", 518: "eval/probe/feats_25class_518.npz"}
CONFIGS = [(224, "lr"), (224, "mlp"), (518, "lr"), (518, "mlp")]
PREV_REPORTED = 0.573


def make_head(kind):
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(512,), max_iter=300, early_stopping=True,
                             alpha=1e-3, random_state=0)
    return LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5, random_state=0)


def load(img):
    z = np.load(FEATS[img], allow_pickle=True)
    X = z["features"].astype(np.float32); y = z["y"].astype(int)
    split = z["split"].astype(str); fold = z["fold"]
    classes = [str(c) for c in z["classes"]]
    tv = split == "train_val"; ex = split == "external_test"
    return (X[tv], y[tv], fold[tv].astype(int)), (X[ex], y[ex]), classes


def oof_probs(Xtv, ytv, foldtv, head, C):
    oof = np.zeros((len(ytv), C))
    for fk in sorted(set(foldtv[foldtv >= 0].tolist())):
        tr = foldtv != fk; te = foldtv == fk
        sc = StandardScaler().fit(Xtv[tr])
        clf = make_head(head).fit(sc.transform(Xtv[tr]), ytv[tr])
        oof[te] = clf.predict_proba(sc.transform(Xtv[te]))
    return oof


def strat_bootstrap_ci(y, pred, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    idx_by_c = {c: np.where(y == c)[0] for c in classes}
    stats = np.empty(B)
    for b in range(B):
        take = np.concatenate([rng.choice(idx_by_c[c], size=len(idx_by_c[c]), replace=True)
                               for c in classes])
        stats[b] = f1_score(y[take], pred[take], average="macro")
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi), float(stats.mean())


def main():
    np.random.seed(0)
    result = {"protocol": "winner's-curse-corrected: select config by OOF macro-F1 (train_val only), "
              "evaluate the single selected config once on external_test",
              "prev_reported_external_macro_f1": PREV_REPORTED, "configs": {}}

    # ---- STEP 1: OOF (selection) metric for every config; external NOT touched ----
    for img, head in CONFIGS:
        (Xtv, ytv, foldtv), _ext, classes = load(img)
        C = len(classes)
        oof = oof_probs(Xtv, ytv, foldtv, head, C)
        oof_raw_f1 = f1_score(ytv, oof.argmax(1), average="macro")
        oof_log = log_softmax(np.log(np.clip(oof, 1e-9, 1)), axis=1)
        bias, _, _ = coordinate_ascent(oof_log, ytv, C, rounds=3)
        oof_bias_f1 = f1_score(ytv, (oof_log + bias).argmax(1), average="macro")
        key = f"img{img}_{head}"
        result["configs"][key] = {"img_size": img, "head": head,
                                  "OOF_macro_f1_raw": round(float(oof_raw_f1), 4),
                                  "OOF_macro_f1_bias": round(float(oof_bias_f1), 4)}
        print(f"[oof] {key:12s} OOF macro-F1 raw={oof_raw_f1:.4f} +bias={oof_bias_f1:.4f}", flush=True)

    # ---- STEP 2: select by OOF macro-F1 (bias-adjusted = the deployed objective) ----
    sel_key = max(result["configs"], key=lambda k: result["configs"][k]["OOF_macro_f1_bias"])
    sel_key_raw = max(result["configs"], key=lambda k: result["configs"][k]["OOF_macro_f1_raw"])
    sel = result["configs"][sel_key]
    img, head = sel["img_size"], sel["head"]
    result["selection_metric"] = "OOF_macro_f1_bias"
    result["selected_config"] = sel_key
    result["selected_config_by_raw_oof"] = sel_key_raw
    result["selection_agrees_raw_vs_bias"] = (sel_key == sel_key_raw)
    print(f"\n[select] winner by OOF(+bias) = {sel_key}  (by raw OOF = {sel_key_raw})", flush=True)

    # refit winner OOF to recover its frozen bias, then evaluate ONCE on external
    (Xtv, ytv, foldtv), (Xex, yex), classes = load(img)
    C = len(classes)
    oof = oof_probs(Xtv, ytv, foldtv, head, C)
    oof_log = log_softmax(np.log(np.clip(oof, 1e-9, 1)), axis=1)
    bias, _, _ = coordinate_ascent(oof_log, ytv, C, rounds=3)   # frozen on OOF

    sc = StandardScaler().fit(Xtv)
    clf = make_head(head).fit(sc.transform(Xtv), ytv)
    ex_prob = clf.predict_proba(sc.transform(Xex))
    ex_log = log_softmax(np.log(np.clip(ex_prob, 1e-9, 1)), axis=1)

    pred_raw = ex_prob.argmax(1)
    pred_bias = (ex_log + bias).argmax(1)
    ext_raw = f1_score(yex, pred_raw, average="macro")
    ext_bias = f1_score(yex, pred_bias, average="macro")
    lo, hi, boot_mean = strat_bootstrap_ci(yex, pred_bias, B=1000, seed=0)

    result["honest_external"] = {
        "n_external": int(len(yex)),
        "macro_f1_raw": round(float(ext_raw), 4),
        "macro_f1_bias": round(float(ext_bias), 4),
        "macro_f1_bias_ci95": [round(lo, 4), round(hi, 4)],
        "bootstrap_mean": round(boot_mean, 4),
        "delta_vs_prev_reported": round(float(ext_bias) - PREV_REPORTED, 4),
    }
    print(f"[external] honest macro-F1 raw={ext_raw:.4f} +bias={ext_bias:.4f} "
          f"CI95=[{lo:.4f},{hi:.4f}] delta_vs_0.573={ext_bias-PREV_REPORTED:+.4f}", flush=True)

    # ---- STEP 3: conformal - calibrate/select on held-out train_val slice, freeze, eval once ----
    rng = np.random.default_rng(0)
    # stratified 20% calibration slice from train_val
    cal_mask = np.zeros(len(ytv), bool)
    for c in np.unique(ytv):
        ci = np.where(ytv == c)[0]
        k = max(1, int(round(0.20 * len(ci))))
        cal_mask[rng.choice(ci, size=k, replace=False)] = True
    fit_mask = ~cal_mask

    sc_c = StandardScaler().fit(Xtv[fit_mask])
    clf_c = make_head(head).fit(sc_c.transform(Xtv[fit_mask]), ytv[fit_mask])
    cal_prob = clf_c.predict_proba(sc_c.transform(Xtv[cal_mask]))
    ex_prob_c = clf_c.predict_proba(sc_c.transform(Xex))
    cal_log = np.log(np.clip(cal_prob, 1e-9, 1)); exc_log = np.log(np.clip(ex_prob_c, 1e-9, 1))
    ycal = ytv[cal_mask]

    T = fit_temperature(cal_log.astype(np.float32), ycal)
    cal_cal = softmax(cal_log / T, 1); ex_cal = softmax(exc_log / T, 1)

    # select (k_reg, lambda) minimizing mean set size s.t. coverage>=0.89 ON THE CAL SLICE
    np.random.seed(0)
    best = None
    for kr in (1, 2, 3):
        for lam in (0.01, 0.05, 0.1):
            s = raps_scores(cal_cal, ycal, lam, kr)
            qh = qhat_of(s, 0.10)
            cov, sz = coverage_and_size(cal_cal, ycal, qh, lam, kr)  # selection on cal slice
            if cov >= 0.89 and (best is None or sz.mean() < best[0]):
                best = (float(sz.mean()), float(cov), kr, lam, float(qh))
    conf = {"target_coverage": 0.90}
    if best is not None:
        _, cal_cov, kr, lam, qh = best
        # FROZEN params -> evaluate ONCE on external
        ex_cov, ex_sz = coverage_and_size(ex_cal, yex, qh, lam, kr)
        conf.update({"k_reg": kr, "lambda": lam, "qhat": round(qh, 4), "temperature": round(float(T), 3),
                     "cal_slice_coverage": round(cal_cov, 3),
                     "external_coverage": round(float(ex_cov), 3),
                     "external_mean_set": round(float(ex_sz.mean()), 2),
                     "external_median_set": int(np.median(ex_sz)),
                     "prev_reported_mean_set": 4.0,
                     "delta_mean_set_vs_prev": round(float(ex_sz.mean()) - 4.0, 2)})
        print(f"[conformal] frozen kr={kr} lam={lam} qhat={qh:.3f} -> external cov={ex_cov:.3f} "
              f"mean-set={ex_sz.mean():.2f} (prev ~4)", flush=True)
    result["conformal"] = conf

    result["old_vs_new"] = {"metric": "external macro-F1 (winner-curse-corrected)",
                            "old": str(PREV_REPORTED),
                            "new": str(result["honest_external"]["macro_f1_bias"])}

    out = ROOT / "eval/probe/r05_winners_curse.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
