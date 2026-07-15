"""Find clinically-useful operating thresholds.

  - Sensitivity at 95% specificity (PPT promise) for the primary class.
  - Risk-coverage abstention threshold.

    python -m scripts.find_threshold \
        --predictions checkpoints/run0/eval_valid_fold0/predictions.npz \
        --classes mel,nv,bcc,scc,df,bkl,ak,vasc \
        --primary mel --target-spec 0.95 --target-error 0.05 \
        --out checkpoints/run0/operating_points.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger

from mnkai.evaluation.abstention import (
    risk_coverage_curve,
    threshold_for_target_error,
)
from mnkai.evaluation.threshold import (
    operating_point_at_specificity,
    threshold_table,
    youden_j_threshold,
)
from mnkai.utils.logging import setup_logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--classes", required=True, help="comma-separated canonical class names")
    ap.add_argument("--primary", default="mel")
    ap.add_argument("--target-spec", type=float, default=0.95)
    ap.add_argument("--target-error", type=float, default=0.05)
    ap.add_argument("--min-coverage", type=float, default=0.60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    setup_logging("logs", "find_threshold.log", "INFO")
    classes = [c.strip().lower() for c in args.classes.split(",")]
    primary = args.primary.strip().lower()
    pi = classes.index(primary)

    z = np.load(args.predictions, allow_pickle=True)
    y = z["y_true"].astype(int)
    prob = z["y_prob"].astype(float)
    pos = (y == pi).astype(int)
    scores = prob[:, pi]

    out = {"classes": classes, "primary": primary,
           "n_total": int(len(y)), "n_primary_positive": int(pos.sum())}

    # operating point at target spec
    op = operating_point_at_specificity(scores, pos, target_specificity=args.target_spec)
    out["sens_at_spec"] = op.as_dict()

    # Youden's J as a balance reference
    yj = youden_j_threshold(scores, pos)
    out["youden_j"] = yj.as_dict()

    # risk-coverage abstention
    cp = threshold_for_target_error(prob, y, target_error=args.target_error,
                                    min_coverage=args.min_coverage)
    out["abstention"] = {
        "threshold": cp.threshold, "coverage": cp.coverage,
        "error_rate": cp.error_rate, "n_kept": cp.n_kept, "n_total": cp.n_total,
        "target_error": args.target_error, "min_coverage": args.min_coverage,
    }
    out["risk_coverage_curve"] = [
        {"threshold": c.threshold, "coverage": c.coverage, "error_rate": c.error_rate}
        for c in risk_coverage_curve(prob, y, n_thresholds=51)
    ]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    logger.info(f"[threshold] {primary} sens@spec={args.target_spec} -> "
                f"thr={op.threshold:.3f} sens={op.sensitivity:.3f} ppv={op.ppv:.3f}")
    logger.info(f"[threshold] abstain @ err<={args.target_error} thr={cp.threshold:.3f} "
                f"coverage={cp.coverage:.3f}")


if __name__ == "__main__":
    main()
