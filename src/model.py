# src/model.py
from typing import Optional
import torch
from torch import nn
import timm
from loguru import logger

def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    drop_rate: float = 0.0,
    compile_model: bool = False,
) -> nn.Module:
    """
    Thin wrapper around timm.create_model with a clean classifier head.
    """
    logger.info(f"[model] Building {model_name} (num_classes={num_classes}, pretrained={pretrained})")
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        global_pool="avg",
    )
    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # PyTorch 2.x
            logger.info("[model] torch.compile applied")
        except Exception as e:
            logger.warning(f"[model] torch.compile failed: {e}")
    return model
