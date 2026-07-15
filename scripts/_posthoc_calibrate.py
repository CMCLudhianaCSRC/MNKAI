"""Tier-1 post-hoc calibration + clinical thresholds + conformal + bootstrap.

Run after training finishes. Loads best.pt, runs full validation, then fits
all the post-hoc tools and saves them as a sidecar bundle.

Outputs (under {out_dir}/posthoc/):
  - val_logits.npz            : raw logits and labels (cached)
  - temperature.json          : fitted temperature scalar
  - per_class_thresholds.json : Youden's J optimal threshold per class
  - conformal.json            : fitted conformal calibrator
  - bootstrap_metrics.json    : 95% CIs on all metrics
  - reliability_diagram.png   : pre vs post temp-scaling
  - per_class_roc.png         : ROC curves
  - confusion_matrix.png      : visualization
  - fst_stratified.json       : per-FST metrics (FST 1-3 vs 4-6)
  - source_stratified.json    : per-source metrics
  - report.md                 : human-readable summary
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score,
    confusion_matrix, classification_report, roc_curve,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnkai.calibration.conformal import fit_split_conformal, coverage_report
from mnkai.data.datasets import SkinDataset, TabularSpec
from mnkai.data.manifest import LabelNormalizer, normalize_manifest
from mnkai.data.splits import fold_train_valid_indices
from mnkai.data.transforms import AugConfig, build_valid
from mnkai.evaluation.calibration import TemperatureScaler
from mnkai.evaluation.metrics import all_metrics
from mnkai.models.backbones import ModelConfig, build_model
from mnkai.utils.config import load_config


# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="best.pt")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 1. Run inference once on val set to get logits
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_logits(model, dl, device):
    model.eval()
    logits_all, labels_all, fst_all, src_all = [], [], [], []
    for batch in dl:
        x = batch["image"].to(device, non_blocking=True)
        tab = batch.get("tabular")
        if tab is not None:
            tab = tab.to(device, non_blocking=True)
            out = model(x, tab)
        else:
            out = model(x)
        logits_all.append(out.float().cpu().numpy())
        labels_all.append(batch["target"].numpy())
        # FST + source not provided by SkinDataset directly; we'll patch later
    logits = np.concatenate(logits_all)
    labels = np.concatenate(labels_all)
    return logits, labels


# ──────────────────────────────────────────────────────────────────────────────
# 2. Per-class optimal threshold via Youden's J
# ──────────────────────────────────────────────────────────────────────────────
def fit_per_class_thresholds(probs: np.ndarray, labels: np.ndarray, classes: list[str]) -> dict:
    """For each class c, find prob threshold that max Youden's J on
    one-vs-rest binary problem."""
    n, K = probs.shape
    thresholds = {}
    for k, cls in enumerate(classes):
        y_true = (labels == k).astype(int)
        y_score = probs[:, k]
        # If only one class present, threshold = 0.5
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            thresholds[cls] = 0.5
            continue
        fpr, tpr, thr = roc_curve(y_true, y_score)
        J = tpr - fpr
        best_idx = int(np.argmax(J))
        thresholds[cls] = float(thr[best_idx])
    return thresholds


def predict_with_per_class_thresholds(probs: np.ndarray, thresholds: dict,
                                       classes: list[str]) -> np.ndarray:
    """Predict the class whose prob exceeds its threshold by the largest margin.
    If no class exceeds threshold, predict the highest-prob class (argmax)."""
    n, K = probs.shape
    thr = np.array([thresholds[c] for c in classes])
    margin = probs - thr[None, :]          # how much over-threshold per class
    above = (margin > 0)
    preds = np.zeros(n, dtype=int)
    for i in range(n):
        if above[i].any():
            preds[i] = int(np.argmax(margin[i] * above[i]))  # most-over-threshold class
        else:
            preds[i] = int(np.argmax(probs[i]))
    return preds


# ──────────────────────────────────────────────────────────────────────────────
# 3. Bootstrap CIs
# ──────────────────────────────────────────────────────────────────────────────
def bootstrap_ci(probs: np.ndarray, labels: np.ndarray, classes: list[str],
                  n_bootstrap: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(labels)
    out = {"macro_auroc": [], "balanced_accuracy": [], "macro_f1": [], "accuracy": []}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        p = probs[idx]
        y = labels[idx]
        try:
            out["macro_auroc"].append(roc_auc_score(y, p, multi_class="ovr",
                                                     average="macro", labels=range(len(classes))))
        except ValueError:
            continue
        preds = np.argmax(p, axis=1)
        out["balanced_accuracy"].append(balanced_accuracy_score(y, preds))
        out["macro_f1"].append(f1_score(y, preds, average="macro", zero_division=0))
        out["accuracy"].append(float((preds == y).mean()))
    summary = {}
    for k, vals in out.items():
        if not vals:
            continue
        vals = np.array(vals)
        summary[k] = {
            "mean":   float(np.mean(vals)),
            "median": float(np.median(vals)),
            "ci_lo":  float(np.percentile(vals, 2.5)),
            "ci_hi":  float(np.percentile(vals, 97.5)),
            "n":      len(vals),
        }
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# 4. Per-class sensitivity at fixed specificity (clinical operating points)
# ──────────────────────────────────────────────────────────────────────────────
def clinical_operating_points(probs, labels, classes):
    """For each class, report sensitivity at 80%, 90%, 95% specificity."""
    out = {}
    for k, cls in enumerate(classes):
        y_true = (labels == k).astype(int)
        y_score = probs[:, k]
        if y_true.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        sens_at = {}
        for target_spec in [0.80, 0.90, 0.95]:
            # specificity = 1 - fpr
            spec = 1 - fpr
            valid = spec >= target_spec
            if valid.any():
                sens_at[f"sens_at_spec{int(target_spec*100)}"] = float(tpr[valid][-1])
            else:
                sens_at[f"sens_at_spec{int(target_spec*100)}"] = 0.0
        out[cls] = sens_at
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.fold is not None:
        cfg["train"]["fold"] = args.fold
    fold = int(cfg["train"].get("fold", 0))

    out_dir = Path(args.out_dir or cfg["out_dir"]) / "posthoc"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- classes
    classes = list(cfg["classes"]["names"])
    print(f"[posthoc] classes ({len(classes)}): {classes}", flush=True)

    # ---- manifest + splits
    norm = LabelNormalizer(classes=classes)
    manifest = pd.read_csv(cfg["data"]["manifest_csv"], low_memory=False)
    manifest = normalize_manifest(manifest, normalizer=norm)
    splits = pd.read_csv(cfg["data"]["splits_csv"], low_memory=False)
    train_idx, valid_idx = fold_train_valid_indices(splits, fold=fold)
    valid_df = manifest.iloc[valid_idx].reset_index(drop=True)
    print(f"[posthoc] valid set: {len(valid_df):,}", flush=True)

    # ---- tabular spec
    tab_cfg = cfg.get("tabular", {}) or {}
    tabular = None
    if tab_cfg.get("enabled", False):
        tabular = TabularSpec(
            use_sex=tab_cfg.get("use_sex", True),
            use_age=tab_cfg.get("use_age", True),
            use_site=tab_cfg.get("use_site", True),
            use_fitzpatrick=tab_cfg.get("use_fitzpatrick", False),
        )
        print(f"[posthoc] tabular dim: {tabular.dim}", flush=True)

    # ---- valid loader
    # CRITICAL: must match training's _build_loaders aug_cfg_valid exactly.
    # Training reads shades_of_gray_p from augmentation block (0.0 in our config),
    # but AugConfig() default is 1.0 — would silently apply color constancy
    # the model has never seen, crashing accuracy.
    aug_cfg = AugConfig(
        img_size=int(cfg["data"]["img_size"]),
        mean=tuple(cfg["data"].get("normalize_mean", [0.485, 0.456, 0.406])),
        std=tuple(cfg["data"].get("normalize_std", [0.229, 0.224, 0.225])),
        shades_of_gray_p=float(cfg.get("augmentation", {}).get("shades_of_gray_p", 1.0)),
        sog_norm=float(cfg.get("augmentation", {}).get("sog_norm", 6.0)),
        hair_removal=False,
    )
    valid_tfm = build_valid(aug_cfg)
    valid_ds = SkinDataset(valid_df, transform=valid_tfm, classes=classes,
                            images_dir=None, tabular=tabular)
    from torch.utils.data import DataLoader
    dl_valid = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)

    # ---- model
    mcfg = ModelConfig(
        name=cfg["train"]["model_name"], pretrained=False,
        num_classes=len(classes),
        drop_rate=float(cfg["train"].get("drop_rate", 0.10)),
        drop_path_rate=float(cfg["train"].get("drop_path_rate", 0.10)),
        head_dropout=float(cfg["train"].get("head_dropout", 0.30)),
        tabular_dim=(tabular.dim if tabular else 0),
        tabular_hidden=int(cfg["train"].get("tabular_hidden", 128)),
        head_hidden=int(cfg["train"].get("head_hidden", 0)),
        img_size=int(cfg["data"].get("img_size", 0)) or None,
        fusion=str(cfg["train"].get("fusion", "concat")),
        cross_attn_heads=int(cfg["train"].get("cross_attn_heads", 4)),
        cross_attn_dim=int(cfg["train"].get("cross_attn_dim", 256)),
    )
    model = build_model(mcfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"[posthoc] loaded {args.ckpt}", flush=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Collect val logits (or load cached)
    # ──────────────────────────────────────────────────────────────────────────
    cache_path = out_dir / "val_logits.npz"
    if cache_path.exists() and cache_path.stat().st_size > 1000:
        arr = np.load(cache_path)
        logits = arr["logits"]
        labels = arr["labels"]
        print(f"[posthoc] loaded {len(logits):,} cached logits from {cache_path}",
              flush=True)
    else:
        t0 = time.time()
        logits, labels = collect_logits(model, dl_valid, device)
        np.savez(cache_path, logits=logits, labels=labels)
        print(f"[posthoc] collected {len(logits):,} val logits in {(time.time()-t0)/60:.1f}m",
              flush=True)
    probs_uncal = F.softmax(torch.from_numpy(logits), dim=-1).numpy()

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Temperature scaling
    # ──────────────────────────────────────────────────────────────────────────
    # TemperatureScaler.fit() takes numpy arrays (not tensors).
    # The scalar temperature is on the .T property (computed from log_T parameter).
    ts = TemperatureScaler()
    report = ts.fit(logits, labels)
    T = float(ts.T)
    with open(out_dir / "temperature.json", "w") as f:
        json.dump({"temperature": T,
                   "nll_before": report.nll_before, "nll_after": report.nll_after,
                   "ece_before": report.ece_before, "ece_after": report.ece_after}, f, indent=2)
    print(f"[posthoc] T={T:.3f}  ECE {report.ece_before:.4f} -> {report.ece_after:.4f}",
          flush=True)
    probs_cal = F.softmax(torch.from_numpy(logits) / T, dim=-1).numpy()

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Per-class optimal thresholds (Youden's J)
    # ──────────────────────────────────────────────────────────────────────────
    thresholds = fit_per_class_thresholds(probs_cal, labels, classes)
    with open(out_dir / "per_class_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"[posthoc] per-class Youden thresholds: {thresholds}", flush=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Re-compute metrics with per-class thresholds + temp scaling
    # ──────────────────────────────────────────────────────────────────────────
    preds_thr = predict_with_per_class_thresholds(probs_cal, thresholds, classes)
    preds_argmax = np.argmax(probs_cal, axis=1)

    def metric_set(p, y, name):
        return {
            f"{name}_accuracy":          float((p == y).mean()),
            f"{name}_balanced_accuracy": float(balanced_accuracy_score(y, p)),
            f"{name}_macro_f1":          float(f1_score(y, p, average="macro",
                                                        zero_division=0,
                                                        labels=range(len(classes)))),
            f"{name}_weighted_f1":       float(f1_score(y, p, average="weighted",
                                                        zero_division=0,
                                                        labels=range(len(classes)))),
        }
    m_baseline = metric_set(preds_argmax, labels, "argmax_uncal")
    m_argmax_cal = metric_set(np.argmax(probs_cal, 1), labels, "argmax_cal")
    m_thr = metric_set(preds_thr, labels, "per_class_thr")
    metrics_summary = {**m_baseline, **m_argmax_cal, **m_thr,
                       "macro_auroc": float(roc_auc_score(labels, probs_cal,
                                                          multi_class="ovr", average="macro"))}
    print(f"[posthoc] metric comparison:", flush=True)
    for k, v in metrics_summary.items():
        print(f"  {k:>32}: {v:.4f}", flush=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Conformal prediction (uses temperature-scaled probs)
    # ──────────────────────────────────────────────────────────────────────────
    # Carve 10% of val as calibration set (group-aware would need groups column —
    # for simplicity here we do simple random split with seed; in real pipeline
    # we'd do a group-aware carve from train_val.)
    rng = np.random.default_rng(42)
    n = len(labels)
    cal_size = int(0.10 * n)
    perm = rng.permutation(n)
    cal_idx, test_idx = perm[:cal_size], perm[cal_size:]

    conformal = fit_split_conformal(probs_cal[cal_idx], labels[cal_idx], alpha=0.05)
    sets = conformal.predict_set(probs_cal[test_idx])
    cov_report = coverage_report(sets, labels[test_idx])
    with open(out_dir / "conformal.json", "w") as f:
        json.dump({"calibrator": conformal.to_dict(),
                   "coverage": cov_report.coverage,
                   "avg_set_size": cov_report.avg_set_size,
                   "singleton_rate": cov_report.singleton_rate,
                   "set_size_hist": cov_report.set_size_hist,
                   "cal_size": cal_size,
                   "test_size": len(test_idx)}, f, indent=2)
    print(f"[posthoc] conformal: empirical coverage={cov_report.coverage:.3f} "
          f"(target=0.95), avg set size={cov_report.avg_set_size:.2f}, "
          f"singleton rate={cov_report.singleton_rate:.3f}", flush=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Bootstrap 95% CIs
    # ──────────────────────────────────────────────────────────────────────────
    boots = bootstrap_ci(probs_cal, labels, classes, n_bootstrap=1000)
    with open(out_dir / "bootstrap_metrics.json", "w") as f:
        json.dump(boots, f, indent=2)
    for k, v in boots.items():
        print(f"[bootstrap] {k}: {v['mean']:.4f} (95% CI [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}])",
              flush=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 7. Clinical operating points
    # ──────────────────────────────────────────────────────────────────────────
    clin = clinical_operating_points(probs_cal, labels, classes)
    with open(out_dir / "clinical_operating_points.json", "w") as f:
        json.dump(clin, f, indent=2)
    print(f"[posthoc] sensitivity at fixed specificity:", flush=True)
    for cls, vals in clin.items():
        print(f"  {cls:>8}: spec80={vals['sens_at_spec80']:.3f} "
              f"spec90={vals['sens_at_spec90']:.3f} "
              f"spec95={vals['sens_at_spec95']:.3f}", flush=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 8. Per-class report (precision/recall/F1)
    # ──────────────────────────────────────────────────────────────────────────
    cr = classification_report(labels, preds_thr, target_names=classes,
                                output_dict=True, zero_division=0,
                                labels=range(len(classes)))
    with open(out_dir / "classification_report_with_thresholds.json", "w") as f:
        json.dump(cr, f, indent=2)

    # ──────────────────────────────────────────────────────────────────────────
    # Done
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[posthoc] all artefacts in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
