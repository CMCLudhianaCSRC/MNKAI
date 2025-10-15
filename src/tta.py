# src/tta.py
from __future__ import annotations
import torch

def logits_tta(model, x, device, do_tta: bool):
    with torch.no_grad():
        logits = model(x)
        if not do_tta:
            return logits
        flips = ([3], [2], [2, 3])  # H, W, HW
        bag = [logits]
        for dims in flips:
            bag.append(model(torch.flip(x, dims=dims)))
        return torch.stack(bag, 0).mean(0)
