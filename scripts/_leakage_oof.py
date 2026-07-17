"""ANALYSIS 2 - Near-duplicate leakage audit ACROSS the 5 OOF folds.

The prior leakage audit only checked train_val vs external_test. This one asks a
different question: within the OOF cross-validation (where the per-class decision
bias was tuned), does any validation row have a near-duplicate sitting in the
TRAINING partition of that same fold? If so, the model effectively "saw" that
lesion at fit time and its OOF prediction is optimistic.

Protocol (mirrors the OOF recipe in scripts/_probe_multiclass.py):
  * Restrict to split == 'train_val' (fold in {0,1,2,3,4}); fold == -1 is the
    external_test set and is excluded here (already audited separately).
  * For each fold k: query = rows with fold == k (this fold's validation set),
    reference = rows with fold != k inside train_val (what the fold-k model trains on).
  * L2-normalize features, compute per-query MAX cosine similarity to any reference
    row (chunked matmul, reusing qc_checks.max_cosine_to_reference).
  * Flag validation rows with max cosine > 0.98 and > 0.99.
  * For each flagged row, also record whether its nearest training neighbour has the
    SAME label -> those are the rows whose OOF prediction is most plausibly inflated.

Runs on CACHED features only. Never touches the image drive.

    ./.venv/Scripts/python.exe scripts/_leakage_oof.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from qc_checks import _l2_normalize  # noqa: E402

OUT = ROOT / "eval" / "probe" / "leakage_oof.json"
CACHES = [
    {"name": "25class_518", "path": ROOT / "eval" / "probe" / "feats_25class_518.npz",
     "headline_metric": "macro-F1 0.573 (25-class OOF)"},
    {"name": "9class_224", "path": ROOT / "eval" / "probe" / "feats_9class_224.npz",
     "headline_metric": "macro-F1 0.656 (9-class OOF)"},
]
THRESHOLDS = (0.98, 0.99)
CHUNK = 512  # keep the (chunk x n_ref) sim block bounded in RAM


def max_cos_and_arg(query: np.ndarray, reference: np.ndarray, chunk: int = CHUNK):
    """Per query row: max cosine to any reference row + the argmax reference index.

    Both matrices are L2-normalized once, then a chunked query @ reference.T gives
    cosine similarities without materialising the full n_q x n_r matrix.
    """
    q = _l2_normalize(query)
    r = _l2_normalize(reference)
    n = q.shape[0]
    best = np.empty(n, dtype=np.float32)
    arg = np.empty(n, dtype=np.int64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sims = q[s:e] @ r.T
        arg[s:e] = sims.argmax(axis=1)
        best[s:e] = sims[np.arange(e - s), arg[s:e]]
    return best, arg


def audit_cache(name: str, path: Path, headline_metric: str) -> dict:
    d = np.load(path, allow_pickle=True)
    feats = np.asarray(d["features"], dtype=np.float32)
    y = np.asarray(d["y"]).ravel()
    fold = np.asarray(d["fold"]).ravel().astype(int)

    tv = fold >= 0                      # train_val OOF rows (fold in 0..4); drop fold==-1 external
    idx_all = np.where(tv)[0]
    folds = sorted(np.unique(fold[tv]).tolist())
    n_oof = int(tv.sum())

    per_fold = []
    # accumulate global maxima and flag masks over the OOF rows (keyed by position in idx_all)
    global_best = np.zeros(n_oof, dtype=np.float32)
    global_same_label = np.zeros(n_oof, dtype=bool)
    pos_of = {int(g): i for i, g in enumerate(idx_all)}  # global row -> position in idx_all

    for k in folds:
        val_mask = fold == k
        ref_mask = tv & (fold != k)     # training partition for fold k
        val_idx = np.where(val_mask)[0]
        ref_idx = np.where(ref_mask)[0]
        best, arg = max_cos_and_arg(feats[val_idx], feats[ref_idx])
        nearest_global = ref_idx[arg]
        same_label = y[val_idx] == y[nearest_global]

        rec = {"fold": int(k), "n_val": int(val_idx.size), "n_train": int(ref_idx.size),
               "max_cosine": round(float(best.max()), 6),
               "median_cosine": round(float(np.median(best)), 6),
               "p99_cosine": round(float(np.percentile(best, 99)), 6)}
        for t in THRESHOLDS:
            fl = best > t
            key = f"n_gt_{str(t).replace('.', '')}"
            rec[key] = int(fl.sum())
            rec[key + "_same_label"] = int((fl & same_label).sum())
        per_fold.append(rec)

        # fold into the global OOF-position arrays
        for j, g in enumerate(val_idx):
            p = pos_of[int(g)]
            global_best[p] = best[j]
            global_same_label[p] = bool(same_label[j])

    summary = {"n_oof_rows": n_oof, "folds": folds,
               "max_cosine_any_fold": round(float(global_best.max()), 6)}
    for t in THRESHOLDS:
        fl = global_best > t
        tag = str(t).replace(".", "")
        n_fl = int(fl.sum())
        n_fl_same = int((fl & global_same_label).sum())
        summary[f"n_val_gt_{tag}"] = n_fl
        summary[f"frac_val_gt_{tag}"] = round(n_fl / n_oof, 6)
        summary[f"n_val_gt_{tag}_same_label"] = n_fl_same
        summary[f"frac_val_gt_{tag}_same_label"] = round(n_fl_same / n_oof, 6)

    return {"headline_metric": headline_metric, "per_fold": per_fold, "summary": summary}


def main():
    results = {
        "description": "Near-duplicate leakage audit ACROSS the 5 OOF folds (validation row vs "
                       "its own fold's training partition). L2-normalized cosine, thresholds "
                       "0.98/0.99. 'same_label' = nearest train neighbour shares the label, i.e. "
                       "the OOF prediction is plausibly optimistic.",
        "thresholds": list(THRESHOLDS),
        "caches": {},
    }
    for c in CACHES:
        print(f"[audit] {c['name']} ...", flush=True)
        r = audit_cache(c["name"], c["path"], c["headline_metric"])
        results["caches"][c["name"]] = r
        s = r["summary"]
        print(f"  n_oof={s['n_oof_rows']}  max_cos={s['max_cosine_any_fold']}  "
              f">0.98={s['n_val_gt_098']} ({s['frac_val_gt_098']:.4%}, "
              f"same-label {s['n_val_gt_098_same_label']})  "
              f">0.99={s['n_val_gt_099']} ({s['frac_val_gt_099']:.4%}, "
              f"same-label {s['n_val_gt_099_same_label']})", flush=True)

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
