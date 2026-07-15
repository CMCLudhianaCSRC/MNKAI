"""Train a single fold from a YAML config.

Usage:
    python -m scripts.train --config configs/convnext_tiny.yaml --fold 0
"""
from __future__ import annotations

import argparse

from mnkai.training.train import train
from mnkai.utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=None, help="override train.fold")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--resume-from", default=None,
                    help="path to a hardened checkpoint (last.pt) to resume from "
                         "— restores model/optim/scheduler/EMA/epoch counter")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.fold is not None:
        cfg.setdefault("train", {})["fold"] = int(args.fold)
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir
    if args.resume_from is not None:
        cfg.setdefault("train", {})["resume_from"] = args.resume_from
    train(cfg)


if __name__ == "__main__":
    main()
