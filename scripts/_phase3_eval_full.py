"""Phase-3 PUBLICATION-GRADE evaluation.

Two honest evaluations from the same 5 fold checkpoints:

  (1) OOF  — each fold's model on its OWN held-out val fold (in-distribution CV).
  (2) TEST — the 5-fold ENSEMBLE (mean of softmax probs) on the HELD-OUT
             external_test set that NONE of the folds ever saw. This is the
             headline generalization number a high-tier reviewer requires.

Reports, for both: accuracy, balanced accuracy, macro-F1, top-3 accuracy,
per-class precision/recall/F1/support, per-class one-vs-rest AUROC & PR-AUC.
For the external test additionally: temperature-scaled calibration (ECE + Brier
before/after), split-conformal APS sets (coverage + mean set size at alpha),
and per-class x per-Fitzpatrick recall (fairness, FST I-III vs IV-VI).

Usage (after the corrected folds finish):
  PYTHONPATH=src .venv/Scripts/python.exe scripts/_phase3_eval_full.py \
      --config configs/phase3_fixB_balsoftmax.yaml \
      --ckpt-fmt "checkpoints/phase3_fixB/fold{n}/best.pt" \
      --out eval/phase3_fixB
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, precision_recall_fscore_support,
    accuracy_score, roc_auc_score, average_precision_score, brier_score_loss,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from mnkai.data.datasets import SkinDataset, TabularSpec
from mnkai.data.manifest import LabelNormalizer, normalize_manifest
from mnkai.data.splits import fold_train_valid_indices
from mnkai.data.transforms import AugConfig, build_valid
from mnkai.models.backbones import ModelConfig, build_model
from mnkai.utils.config import load_config


# ----------------------------------------------------------------- helpers --
def expected_calibration_error(probs, labels, n_bins=15):
    conf = probs.max(1); pred = probs.argmax(1); acc = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            ece += abs(acc[m].mean() - conf[m].mean()) * m.mean()
    return float(ece)


def multiclass_brier(probs, labels, C):
    onehot = np.eye(C)[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def fit_temperature(logits, labels):
    """Grid + local search for the single temperature minimising NLL."""
    lg = torch.from_numpy(logits).double(); y = torch.from_numpy(labels).long()
    best_T, best_nll = 1.0, 1e9
    for T in np.concatenate([np.linspace(0.5, 5.0, 46), np.linspace(0.05, 0.5, 10)]):
        nll = F.cross_entropy(lg / float(T), y).item()
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T


def conformal_aps(cal_probs, cal_labels, test_probs, alpha=0.10):
    """Split-conformal Adaptive Prediction Sets (Romano/Angelopoulos 2020).
    Returns (coverage, mean_set_size) on test at target 1-alpha coverage."""
    def scores(probs, labels=None):
        order = np.argsort(-probs, axis=1)
        sorted_p = np.take_along_axis(probs, order, axis=1)
        cum = np.cumsum(sorted_p, axis=1)
        if labels is not None:  # calibration: score of the TRUE label
            s = np.empty(len(probs))
            for i in range(len(probs)):
                rank = np.where(order[i] == labels[i])[0][0]
                s[i] = cum[i, rank]
            return s
        return order, cum
    cal_s = scores(cal_probs, cal_labels)
    n = len(cal_s)
    qhat = np.quantile(cal_s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
    order, cum = scores(test_probs)
    sizes = (cum <= qhat).sum(1) + 1  # include first class that crosses qhat
    sizes = np.clip(sizes, 1, test_probs.shape[1])
    return qhat, sizes


def per_class_table(labels, preds, probs, classes):
    p, r, f1c, sup = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(classes))), zero_division=0)
    rows = []
    for i in range(len(classes)):
        m = labels == i
        try:
            auroc = roc_auc_score((labels == i).astype(int), probs[:, i]) if m.any() and (~m).any() else float("nan")
            prauc = average_precision_score((labels == i).astype(int), probs[:, i]) if m.any() else float("nan")
        except Exception:
            auroc = prauc = float("nan")
        rows.append({"class": classes[i], "f1": round(float(f1c[i]), 3),
                     "precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                     "auroc": round(float(auroc), 3), "pr_auc": round(float(prauc), 3),
                     "support": int(sup[i])})
    return sorted(rows, key=lambda d: d["f1"], reverse=True)


def headline(labels, preds, probs, classes):
    top3 = np.argsort(-probs, axis=1)[:, :3]
    return {
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, preds)), 4),
        "macro_f1": round(float(f1_score(labels, preds, average="macro", zero_division=0)), 4),
        "top3_accuracy": round(float(np.mean([labels[i] in top3[i] for i in range(len(labels))])), 4),
    }


# ------------------------------------------------------------------- main --
def build_model_for(cfg, classes, tab, device):
    mcfg = ModelConfig(
        name=cfg["train"]["model_name"], pretrained=False, num_classes=len(classes),
        drop_rate=float(cfg["train"].get("drop_rate", 0.10)),
        drop_path_rate=float(cfg["train"].get("drop_path_rate", 0.10)),
        head_dropout=float(cfg["train"].get("head_dropout", 0.30)),
        tabular_dim=(tab.dim if tab else 0),
        tabular_hidden=int(cfg["train"].get("tabular_hidden", 128)),
        head_hidden=int(cfg["train"].get("head_hidden", 0)),
        img_size=int(cfg["data"].get("img_size", 0)) or None,
        fusion=str(cfg["train"].get("fusion", "concat")),
        cross_attn_heads=int(cfg["train"].get("cross_attn_heads", 4)),
        cross_attn_dim=int(cfg["train"].get("cross_attn_dim", 256)),
    )
    return build_model(mcfg).to(device).eval()


@torch.no_grad()
def infer(model, df, classes, tab, tfm, device, bs=32):
    ds = SkinDataset(df, classes=classes, transform=tfm, images_dir=None, tabular=tab)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=False)
    out = []
    for b in dl:
        x = b["image"].to(device, non_blocking=True)
        tb = b.get("tabular");  tb = tb.to(device) if tb is not None else None
        l = model(x, tabular=tb) if tb is not None else model(x)
        out.append(l.float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt-fmt", required=True, help="e.g. checkpoints/phase3_fixB/fold{n}/best.pt")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.10, help="conformal miscoverage")
    ap.add_argument("--skip-external", action="store_true",
                    help="OOF only; do NOT touch the external test set (use during "
                         "model selection so the held-out set stays pristine).")
    args = ap.parse_args()
    folds = [int(x) for x in args.folds.split(",")]
    skip_ext = bool(args.skip_external)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    classes = list(cfg["classes"]["names"]); C = len(classes)
    cidx = {c: i for i, c in enumerate(classes)}
    tab_cfg = cfg.get("tabular", {}) or {}
    tab = TabularSpec(use_sex=tab_cfg.get("use_sex", True), use_age=tab_cfg.get("use_age", True),
                      use_site=tab_cfg.get("use_site", True),
                      use_fitzpatrick=tab_cfg.get("use_fitzpatrick", True)) if tab_cfg.get("enabled", False) else None

    norm = LabelNormalizer(classes=classes)
    manifest = normalize_manifest(pd.read_csv(cfg["data"]["manifest_csv"], low_memory=False), normalizer=norm)
    splits = pd.read_csv(cfg["data"]["splits_csv"], low_memory=False)
    manifest = manifest.merge(splits[["filepath", "split", "fold"]], on="filepath", how="inner")
    aug = AugConfig(img_size=int(cfg["data"]["img_size"]),
                    shades_of_gray_p=float(cfg.get("augmentation", {}).get("shades_of_gray_p", 0.0)),
                    hair_removal=False)
    tfm = build_valid(aug)

    test_df = manifest[manifest["split"] == "external_test"].reset_index(drop=True)
    test_y = test_df["label_name"].map(cidx).values.astype(int)
    test_fst = pd.to_numeric(test_df.get("fitzpatrick", pd.Series([np.nan]*len(test_df))), errors="coerce").values
    print(f"[eval] external_test: {len(test_df)} imgs (skip_external={skip_ext}); OOF folds: {folds}", flush=True)

    # ---- (1) OOF: each fold model on its own val fold ----
    oof_logits, oof_y = [], []
    # ---- (2) TEST: accumulate ensemble probs over folds (skipped during selection) ----
    test_prob_sum = np.zeros((len(test_df), C))
    for n in folds:
        ckpt = args.ckpt_fmt.format(n=n)
        cfg["train"]["fold"] = n
        model = build_model_for(cfg, classes, tab, device)
        obj = torch.load(ckpt, map_location=device, weights_only=False)
        sd = obj.get("ema_state") or obj.get("model_state") or obj
        model.load_state_dict(sd, strict=False)
        t0 = time.time()
        # OOF (in-distribution validation — always computed)
        _, vidx = fold_train_valid_indices(manifest, fold=n)
        vdf = manifest.iloc[vidx].reset_index(drop=True)
        lg = infer(model, vdf, classes, tab, tfm, device)
        oof_logits.append(lg); oof_y.append(vdf["label_name"].map(cidx).values.astype(int))
        if not skip_ext:                                   # external test ensemble
            tl = infer(model, test_df, classes, tab, tfm, device)
            test_prob_sum += F.softmax(torch.from_numpy(tl), dim=-1).numpy()
        print(f"[eval] fold {n}: oof {len(vdf)}"
              f"{'' if skip_ext else ' + test '+str(len(test_df))} in {(time.time()-t0)/60:.1f}m", flush=True)
        del model
        if device.type == "cuda": torch.cuda.empty_cache()

    oof_logits = np.concatenate(oof_logits); oof_y = np.concatenate(oof_y)
    oof_prob = F.softmax(torch.from_numpy(oof_logits), dim=-1).numpy(); oof_pred = oof_prob.argmax(1)

    out = {
        "config": args.config, "ckpt_fmt": args.ckpt_fmt, "n_folds": len(folds),
        "selection_only": skip_ext,
        "OOF_cv": {**headline(oof_y, oof_pred, oof_prob, classes),
                   "n": int(len(oof_y)), "per_class": per_class_table(oof_y, oof_pred, oof_prob, classes)},
        "EXTERNAL_TEST_ensemble": None,
    }

    if not skip_ext:
        test_prob = test_prob_sum / len(folds); test_pred = test_prob.argmax(1)
        # calibration (temperature scaling, split cal/test)
        rng = np.random.RandomState(0); perm = rng.permutation(len(test_df))
        cal_idx, ev_idx = perm[:len(perm)//2], perm[len(perm)//2:]
        ece_before = expected_calibration_error(test_prob[ev_idx], test_y[ev_idx])
        brier_before = multiclass_brier(test_prob[ev_idx], test_y[ev_idx], C)
        mean_logit = np.log(np.clip(test_prob, 1e-8, 1))
        T = fit_temperature(mean_logit[cal_idx], test_y[cal_idx])
        test_prob_ts = F.softmax(torch.from_numpy(mean_logit / T), dim=-1).numpy()
        ece_after = expected_calibration_error(test_prob_ts[ev_idx], test_y[ev_idx])
        brier_after = multiclass_brier(test_prob_ts[ev_idx], test_y[ev_idx], C)
        # conformal APS
        qhat, sizes = conformal_aps(test_prob_ts[cal_idx], test_y[cal_idx], test_prob_ts[ev_idx], alpha=args.alpha)
        order = np.argsort(-test_prob_ts[ev_idx], axis=1)
        covered = np.array([test_y[ev_idx][i] in order[i, :sizes[i]] for i in range(len(ev_idx))])
        # per-Fitzpatrick fairness
        fairness = {}
        for lo, hi, nm in [(1, 3, "FST_I-III"), (4, 6, "FST_IV-VI")]:
            m = (test_fst >= lo) & (test_fst <= hi)
            if m.sum() >= 30:
                fairness[nm] = {"n": int(m.sum()),
                                "balanced_acc": round(float(balanced_accuracy_score(test_y[m], test_pred[m])), 3),
                                "macro_f1": round(float(f1_score(test_y[m], test_pred[m], average="macro", zero_division=0)), 3)}
        out["EXTERNAL_TEST_ensemble"] = {**headline(test_y, test_pred, test_prob, classes),
                   "n": int(len(test_df)), "per_class": per_class_table(test_y, test_pred, test_prob, classes)}
        out["calibration_external"] = {"temperature": round(T, 3),
                   "ECE_before": round(ece_before, 4), "ECE_after": round(ece_after, 4),
                   "Brier_before": round(brier_before, 4), "Brier_after": round(brier_after, 4)}
        out["conformal_external"] = {"alpha": args.alpha, "target_coverage": round(1-args.alpha, 3),
                   "empirical_coverage": round(float(covered.mean()), 3),
                   "mean_set_size": round(float(sizes.mean()), 2), "qhat": round(float(qhat), 4)}
        out["fairness_external"] = fairness

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics_full.json").write_text(json.dumps(out, indent=2))

    print("\n===== PHASE-3 EVAL =====")
    print("OOF CV (selection):", out["OOF_cv"]["macro_f1"], "macroF1 |", out["OOF_cv"]["balanced_accuracy"], "bal-acc")
    if not skip_ext:
        e = out["EXTERNAL_TEST_ensemble"]
        print("EXTERNAL (ens)  :", e["macro_f1"], "macroF1 |", e["balanced_accuracy"], "bal-acc |", e["top3_accuracy"], "top3")
        print("\nExternal-test per-class F1 (sorted):")
        for d in e["per_class"]:
            print(f"  {d['class']:<22} F1={d['f1']:.3f} R={d['recall']:.3f} AUROC={d['auroc']:.3f} n={d['support']}")
    else:
        print("(external test untouched — selection mode)")
    print(f"\nsaved -> {outdir/'metrics_full.json'}")


if __name__ == "__main__":
    main()
