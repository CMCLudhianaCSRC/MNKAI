from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_dir: str | Path | None = None, log_file: str = "train.log", level: str = "INFO") -> None:
    logger.remove()
    logger.add(sys.stderr, level=level,
               format="<green>{time:HH:mm:ss}</green> <level>{level: <7}</level> {message}")
    if log_dir is not None:
        p = Path(log_dir)
        p.mkdir(parents=True, exist_ok=True)
        logger.add(p / log_file, level="DEBUG", rotation="20 MB", retention=5,
                   format="{time:YYYY-MM-DD HH:mm:ss} {level: <7} {name}:{line} {message}")
