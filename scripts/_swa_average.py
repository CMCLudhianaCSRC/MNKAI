"""Stochastic Weight Averaging (SWA) post-training.

Averages the last N best-checkpoints (or the last N epoch checkpoints if we
saved them) to produce a smoother, often-better-generalizing model.

Reference: Izmailov et al. 2018, "Averaging Weights Leads to Wider Optima and
Better Generalization".

We use a simpler variant: average the EMA-state from the last K best checkpoints
that were saved during training. Most of our gain in Phase 1 came from EMA
already, so this is a "free" extra ensemble.
"""
from __future__ import annotations
import argparse, sys, copy
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                     help="List of checkpoint paths to average")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if not args.ckpts:
        print("No checkpoints provided"); return

    avg_state = None
    n = len(args.ckpts)
    template = None
    for ck_path in args.ckpts:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        sd = ck.get("ema_state") or ck.get("model_state") or ck.get("state_dict") or ck
        if avg_state is None:
            avg_state = {k: v.clone().float() / n if torch.is_tensor(v) else v
                          for k, v in sd.items()}
            template = ck
        else:
            for k, v in sd.items():
                if torch.is_tensor(v) and k in avg_state and torch.is_tensor(avg_state[k]):
                    avg_state[k] += v.float() / n
        print(f"  averaged {ck_path}")

    # Build output checkpoint with averaged weights, copy other fields from template
    out_ckpt = {
        "epoch":       template.get("epoch", 0),
        "model_state": avg_state,
        "classes":     template.get("classes"),
        "config":      template.get("config"),
        "swa_average_of": args.ckpts,
        "metrics":     {"note": "SWA-averaged model, re-validate"},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, args.output)
    print(f"\nSWA average of {n} checkpoints saved to {args.output}")


if __name__ == "__main__":
    main()
