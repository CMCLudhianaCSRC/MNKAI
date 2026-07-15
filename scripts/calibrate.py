"""Fit a temperature scaler on a held-out half of the validation fold.

Saves a JSON with the temperature(s) and the before/after NLL/ECE.

    python -m scripts.calibrate \
        --predictions checkpoints/run0/eval_valid_fold0/predictions.npz \
        --out checkpoints/run0/temperature.json \
        --per-group fitzpatrick   # optional
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from mnkai.evaluation.calibration import GroupTemperatureScaler, TemperatureScaler
from mnkai.utils.logging import setup_logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True,
                    help="path to predictions.npz produced by scripts.evaluate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None,
                    help="optional: CSV used to look up subgroup (for --per-group)")
    ap.add_argument("--per-group", default=None,
                    help="column name for per-group calibration (e.g. fitzpatrick)")
    ap.add_argument("--logits-from-probs", action="store_true",
                    help="if predictions store probs only, recover logits via log()")
    args = ap.parse_args()

    setup_logging("logs", "calibrate.log", "INFO")
    z = np.load(args.predictions, allow_pickle=True)
    y_true = z["y_true"].astype(int)
    y_prob = z["y_prob"].astype(float)
    if "logits" in z.files:
        logits = z["logits"].astype(float)
    elif args.logits_from_probs:
        logits = np.log(np.clip(y_prob, 1e-12, 1.0))
    else:
        # fallback: use logit(prob) treating each class as independent
        logits = np.log(np.clip(y_prob, 1e-12, 1.0))

    # split predictions 50/50: half to fit, half to evaluate
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y_true))
    fit, ev = perm[: len(perm) // 2], perm[len(perm) // 2 :]

    if args.per_group and args.manifest:
        df = pd.read_csv(args.manifest)
        if args.per_group not in df.columns:
            raise SystemExit(f"--per-group '{args.per_group}' not in {args.manifest}")
        # the predictions npz stores filepaths in same order as y_true
        if "filepaths" not in z.files:
            raise SystemExit("predictions.npz missing 'filepaths'; cannot align with manifest")
        paths = z["filepaths"]
        path_to_grp = df.set_index("filepath")[args.per_group].astype(str).to_dict()
        groups = [str(path_to_grp.get(p, "missing")) for p in paths]
        keys = sorted({g for g in groups if g != "missing"})
        scaler = GroupTemperatureScaler(group_keys=keys)
        report = scaler.fit(logits[fit], y_true[fit], [groups[i] for i in fit])
    else:
        scaler = TemperatureScaler()
        report = scaler.fit(logits[fit], y_true[fit])

    out = {
        "n_fit": int(len(fit)),
        "n_eval": int(len(ev)),
        "report": {
            "nll_before": report.nll_before,
            "nll_after": report.nll_after,
            "ece_before": report.ece_before,
            "ece_after": report.ece_after,
            "temperature": report.temperature,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    torch.save(scaler.state_dict(), args.out.replace(".json", ".pt"))
    logger.info(f"[calibrate] {report}")


if __name__ == "__main__":
    main()
