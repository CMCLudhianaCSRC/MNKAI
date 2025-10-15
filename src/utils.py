# src/utils.py
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Seed function suitable for torch DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        logger.info(f"Using CUDA GPU: {name} (capability {cap})")
        return torch.device("cuda")
    logger.warning("CUDA not available — falling back to CPU")
    return torch.device("cpu")


def configure_torch_backends(allow_tf32: bool = True, compile_mode: bool = False) -> None:
    if allow_tf32 and torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("[torch] TF32 enabled for Ampere/Ada GPUs.")
        except Exception:
            pass
    if compile_mode:
        logger.info("[torch] torch.compile requested — will attempt after model creation.")


def time_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_mkdir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_logging(log_dir: str | Path, filename: str = "train.log", level: str = "INFO") -> None:
    """Configure loguru to log both to stdout and to a rotating file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stdout,
        level=level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )
    logger.add(
        log_dir / filename,
        rotation="10 MB",
        retention=5,
        enqueue=True,
        level=level.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    )
