"""Run a config across all K folds sequentially, then aggregate.

    python -m scripts.run_5fold --config configs/convnext_tiny.yaml --k 5
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from mnkai.utils.config import load_config
from mnkai.utils.logging import setup_logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--start-fold", type=int, default=0)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()

    setup_logging("logs", "run_5fold.log", "INFO")
    cfg = load_config(args.config)
    base_out = Path(args.out_root or cfg.get("out_dir", "checkpoints/run0"))
    base_out.mkdir(parents=True, exist_ok=True)

    fold_metrics: list[dict] = []
    for fold in range(args.start_fold, args.k):
        out_dir = base_out.parent / f"{base_out.name.replace('_fold0','')}_fold{fold}"
        logger.info(f"[5fold] training fold {fold} -> {out_dir}")
        cmd = [sys.executable, "-m", "scripts.train",
               "--config", args.config,
               "--fold", str(fold),
               "--out-dir", str(out_dir)]
        rc = subprocess.call(cmd)
        if rc != 0:
            logger.error(f"[5fold] fold {fold} failed rc={rc}; aborting")
            sys.exit(rc)
        # collect best metrics
        m_path = out_dir / "best_metrics.json"
        if m_path.exists():
            with open(m_path, encoding="utf-8") as f:
                m = json.load(f)
            fold_metrics.append({"fold": fold, **m})

    if not fold_metrics:
        logger.warning("[5fold] no per-fold metrics collected")
        return

    rows: list[dict] = []
    for m in fold_metrics:
        rows.append({
            "fold": m["fold"],
            "macro_f1": m.get("macro_f1"),
            "balanced_accuracy": m.get("balanced_accuracy"),
            "accuracy": m.get("accuracy"),
            "macro_auroc_ovr": m.get("macro_auroc_ovr"),
            "ece": m.get("ece"),
        })
    df = pd.DataFrame(rows)
    summary = {
        col: {"mean": float(df[col].mean()), "std": float(df[col].std(ddof=0))}
        for col in df.columns if col != "fold"
    }
    out_csv = base_out.parent / f"{base_out.name.replace('_fold0','')}_summary.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"[5fold] per-fold:\n{df.to_string(index=False)}")
    logger.info(f"[5fold] summary (mean ± std):\n{json.dumps(summary, indent=2)}")
    out_json = base_out.parent / f"{base_out.name.replace('_fold0','')}_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"per_fold": rows, "summary": summary}, f, indent=2, default=float)


if __name__ == "__main__":
    main()
