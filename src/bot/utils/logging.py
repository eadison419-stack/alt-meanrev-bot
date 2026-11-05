"""Explicit logging setup. Callers must never log credentials or full settings."""

import sys
from pathlib import Path

from loguru import logger


def configure_logging(level: str = "INFO", log_path: Path | None = None) -> None:
    logger.remove()
    options = dict(level=level, backtrace=False, diagnose=False)
    logger.add(sys.stderr, **options)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_path, encoding="utf-8", **options)
