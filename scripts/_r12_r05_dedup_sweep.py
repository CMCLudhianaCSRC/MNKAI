"""R12 (dedup-threshold sensitivity) + R05-deduped (winner's-curse on the CLEAN external).

Runs entirely on CACHED frozen features — no D:, no image reads, no GPU, no retraining.

R12  — is the 0.573 headline a knife-edge artifact of picking cosine 0.99?
       Sweep the near-duplicate cutoff 0.95..0.995 and report the corrected external
       macro-F1 at each, for both the frozen probe and (for context) how many rows drop.

R05d — the completed R05 scored the FULL external (n=5,886), which still contains the
       near-duplicate contamination, so its 0.629 reproduces the known inflated value.
       Here we redo the winner's-curse protocol properly: select the config on OOF only,
       then evaluate that single config ONCE on the DE-DUPLICATED external.

    python scripts/_r12_r05_dedup_sweep.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts._optimize_thresholds import coordinate_ascent

SEED = 0
CACHES = {224: ROOT / "eval/probe/feats_25class_224.npz",
          518: ROOT / "eval/probe/feats_25class_518.npz"}


def load(img_size):
    z = np.load(CACHES[img_size], allow_pickle=True)
    X, y = z["features"].astype(np.float32), z["y"].astype(int)
    split = np.array([str(s) for s in z["split"]])
    fold = z["fold"].astype(int)
    classes = [str(c) for c in z["classes"]]
    tr, ex = split == "train_val", split == "external_test"
    return X, y, fold, tr, ex, classes


def make_head(kind):
    if kind == "mlp":
        return MLPClassifier((512,), max_iter=300, early_stopping=True, alpha=1e-3, random_state=SEED)
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=SEED)


def fit_predict(Xtr, ytr, Xte, kind):
    sc = StandardScaler().fit(Xtr)
    h = make_head(kind).fit(sc.transform(Xtr), ytr)
    return np.log(np.clip(h.predict_proba(sc.transform(Xte)), 1e-9, 1)), h, sc


def oof_logits(X, y, fold, tr, kind, n_classes):
    """5-fold OOF log-probs on the train_val partition."""
    Xt, yt, ft = X[tr], y[tr], fold[tr]
    out = np.full((len(yt), n_classes), -np.inf, dtype=np.float32)
    for f in np.unique(ft):
        a, b = ft != f, ft == f
        lg, _, _ = fit_predict(Xt[a], yt[a], Xt[b], kind)
        # map the fold-model's classes back onto the global class index
        cls = np.unique(yt[a])
        tmp = np.full((b.sum(), n_classes), -30.0, dtype=np.float32)
        tmp[:, cls] = lg
        out[b] = tmp
    return out, yt


def dup_max_cos(Xex, Xtr, block=512):
    """Max cosine similarity of each external row to ANY train row (L2-normalised)."""
    A = Xex / np.clip(np.linalg.norm(Xex, axis=1, keepdims=True), 1e-9, None)
    B = Xtr / np.clip(np.linalg.norm(Xtr, axis=1, keepdims=True), 1e-9, None)
    best = np.zeros(len(A), dtype=np.float32)
    for i in range(0, len(A), block):
        best[i:i + block] = (A[i:i + block] @ B.T).max(1)
    return best


def main():
    res = {"note": "R12 dedup-threshold sensitivity + R05 re-run on the DE-DUPLICATED external. "
                   "Cached features only; seed 0; recipe matches scripts/_probe_multiclass.py.",
           "seed": SEED}

    # ---------------- select the config on OOF ONLY (winner's-curse protocol) --------------
    print("[oof] scoring 4 configs on train_val only (external unseen) ...", flush=True)
    sel = {}
    cache = {}
    for size in (224, 518):
        X, y, fold, tr, ex, classes = load(size)
        cache[size] = (X, y, fold, tr, ex, classes)
        nC = len(classes)
        for kind in ("lr", "mlp"):
            lg, yt = oof_logits(X, y, fold, tr, kind, nC)
            raw = f1_score(yt, lg.argmax(1), average="macro")
            bias = coordinate_ascent(lg, yt, nC)
            biased = f1_score(yt, (lg + bias).argmax(1), average="macro")
            sel[f"img{size}_{kind}"] = {"img_size": size, "head": kind,
                                        "OOF_macro_f1_raw": round(float(raw), 4),
                                        "OOF_macro_f1_bias": round(float(biased), 4)}
            print(f"  {size}_{kind}: OOF raw {raw:.4f} +bias {biased:.4f}", flush=True)
    best = max(sel, key=lambda k: sel[k]["OOF_macro_f1_bias"])
    res["configs_scored_on_oof"] = sel
    res["selected_config"] = best
    print(f"[oof] selected on OOF alone -> {best}", flush=True)

    # ---------------- fit the selected config once, then sweep the dedup threshold --------
    size, kind = sel[best]["img_size"], sel[best]["head"]
    X, y, fold, tr, ex, classes = cache[size]
    nC = len(classes)
    lg_oof, yt = oof_logits(X, y, fold, tr, kind, nC)
    bias = coordinate_ascent(lg_oof, yt, nC)            # bias tuned on OOF, never on external
    lg_ex, _, _ = fit_predict(X[tr], y[tr], X[ex], kind)
    yex = y[ex]
    print("[dup] computing nearest-train cosine for every external row ...", flush=True)
    mx = dup_max_cos(X[ex], X[tr])

    sweep = []
    for thr in [0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995]:
        keep = mx <= thr
        if keep.sum() < 50:
            continue
        f1_raw = f1_score(yex[keep], lg_ex[keep].argmax(1), average="macro")
        f1_bias = f1_score(yex[keep], (lg_ex[keep] + bias).argmax(1), average="macro")
        sweep.append({"dup_threshold": thr, "n_kept": int(keep.sum()),
                      "n_dropped": int((~keep).sum()),
                      "pct_dropped": round(100.0 * (~keep).mean(), 1),
                      "macro_f1_raw": round(float(f1_raw), 4),
                      "macro_f1_bias": round(float(f1_bias), 4)})
        print(f"  thr {thr}: keep {keep.sum():5d} ({100*(~keep).mean():4.1f}% dropped)  "
              f"macroF1 raw {f1_raw:.4f} +bias {f1_bias:.4f}", flush=True)
    res["r12_dedup_sensitivity"] = sweep

    # full external, for reference
    res["full_external"] = {
        "n": int(ex.sum()),
        "macro_f1_raw": round(float(f1_score(yex, lg_ex.argmax(1), average="macro")), 4),
        "macro_f1_bias": round(float(f1_score(yex, (lg_ex + bias).argmax(1), average="macro")), 4),
    }

    # ---------------- R05 on the DE-DUPLICATED external (the canonical 0.99 cut) ----------
    keep99 = mx <= 0.99
    f1b = f1_score(yex[keep99], (lg_ex[keep99] + bias).argmax(1), average="macro")
    # stratified bootstrap CI over the kept rows
    rng = np.random.default_rng(20260725)
    pred = (lg_ex + bias).argmax(1)
    idx_keep = np.where(keep99)[0]
    boots = []
    for _ in range(2000):
        bi = rng.choice(idx_keep, size=len(idx_keep), replace=True)
        boots.append(f1_score(yex[bi], pred[bi], average="macro"))
    res["r05_on_deduplicated_external"] = {
        "protocol": "config selected on OOF only; per-class bias tuned on OOF only; "
                    "evaluated ONCE on the de-duplicated external (cosine <= 0.99).",
        "selected_config": best,
        "n_external_clean": int(keep99.sum()),
        "macro_f1_bias": round(float(f1b), 4),
        "ci95": [round(float(np.percentile(boots, 2.5)), 4),
                 round(float(np.percentile(boots, 97.5)), 4)],
        "prev_reported_canonical": 0.573,
    }

    (ROOT / "eval/probe/r12_r05_dedup_sweep.json").write_text(json.dumps(res, indent=2))
    print("\n=== R05 on the DE-DUPLICATED external ===")
    r = res["r05_on_deduplicated_external"]
    print(f"  selected on OOF: {r['selected_config']}  |  n={r['n_external_clean']}")
    print(f"  macro-F1 {r['macro_f1_bias']}  95% CI {r['ci95']}   (previously reported 0.573)")
    print("saved -> eval/probe/r12_r05_dedup_sweep.json")


if __name__ == "__main__":
    main()
