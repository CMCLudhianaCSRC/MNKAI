"""Dump per-image logits/probs for the Fix-C 5-fold ensemble, for BOTH the
external test set and the OOF validation, so downstream analyses (coarse-collapse
comparison, RAPS/clustered conformal, per-class thresholds, validated model soup)
can run offline from arrays instead of re-running inference each time.

Reuses the exact inference path from _phase3_eval_full.py (same model build, same
transforms, same fold->valid indexing), so numbers match metrics_full.json.

    python scripts/_dump_predictions.py \
        --config configs/phase3_fixC_cbfocal_only.yaml \
        --ckpt-fmt checkpoints/phase3_fixC/fold{n}/best.pt \
        --out eval/phase3_fixC_full/preds.npz
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mnkai.data.manifest import LabelNormalizer, normalize_manifest
from mnkai.data.datasets import TabularSpec
from mnkai.data.splits import fold_train_valid_indices
from mnkai.data.transforms import AugConfig, build_valid
from mnkai.utils.config import load_config
# reuse the exact model build + inference from the eval script
from scripts._phase3_eval_full import build_model_for, infer


def _col(df, name):
    return df[name].values if name in df.columns else np.array([None] * len(df), dtype=object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt-fmt", required=True)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    folds = [int(x) for x in args.folds.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[dump] device={device}", flush=True)
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
    test_fst = pd.to_numeric(_col(test_df, "fitzpatrick"), errors="coerce")
    N = len(test_df)
    print(f"[dump] external_test={N}, folds={folds}", flush=True)

    test_logits = np.zeros((len(folds), N, C), dtype=np.float32)
    oof_logits, oof_y, oof_fold, oof_fst, oof_fp = [], [], [], [], []
    for i, n in enumerate(folds):
        ckpt = args.ckpt_fmt.format(n=n)
        cfg["train"]["fold"] = n
        model = build_model_for(cfg, classes, tab, device)
        obj = torch.load(ckpt, map_location=device, weights_only=False)
        sd = obj.get("ema_state") or obj.get("model_state") or obj
        model.load_state_dict(sd, strict=False)
        t0 = time.time()
        _, vidx = fold_train_valid_indices(manifest, fold=n)
        vdf = manifest.iloc[vidx].reset_index(drop=True)
        vlg = infer(model, vdf, classes, tab, tfm, device)
        oof_logits.append(vlg)
        oof_y.append(vdf["label_name"].map(cidx).values.astype(int))
        oof_fold.append(np.full(len(vdf), n, dtype=int))
        oof_fst.append(pd.to_numeric(_col(vdf, "fitzpatrick"), errors="coerce"))
        oof_fp.append(_col(vdf, "filepath").astype(str))
        test_logits[i] = infer(model, test_df, classes, tab, tfm, device)
        print(f"[dump] fold {n}: oof {len(vdf)} + test {N} in {(time.time()-t0)/60:.1f}m", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof_logits = np.concatenate(oof_logits).astype(np.float32)
    oof_y = np.concatenate(oof_y)
    oof_fold = np.concatenate(oof_fold)
    oof_fst = np.concatenate(oof_fst).astype(np.float32)
    oof_fp = np.concatenate(oof_fp)
    oof_prob = F.softmax(torch.from_numpy(oof_logits), dim=-1).numpy().astype(np.float32)

    test_prob_ens = F.softmax(torch.from_numpy(test_logits), dim=-1).numpy().mean(0).astype(np.float32)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        classes=np.array(classes),
        # external test
        test_prob_ens=test_prob_ens, test_logits=test_logits, test_y=test_y,
        test_fst=test_fst.astype(np.float32), test_fp=_col(test_df, "filepath").astype(str),
        # OOF
        oof_logits=oof_logits, oof_prob=oof_prob, oof_y=oof_y,
        oof_fold=oof_fold, oof_fst=oof_fst, oof_fp=oof_fp,
    )
    # sanity
    from sklearn.metrics import f1_score, accuracy_score
    tp = test_prob_ens.argmax(1); op = oof_prob.argmax(1)
    print(f"[dump] external: acc={accuracy_score(test_y,tp):.4f} macroF1={f1_score(test_y,tp,average='macro'):.4f}", flush=True)
    print(f"[dump] OOF:      acc={accuracy_score(oof_y,op):.4f} macroF1={f1_score(oof_y,op,average='macro'):.4f}", flush=True)
    print(f"[dump] saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
