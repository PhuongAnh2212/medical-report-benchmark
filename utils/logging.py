"""
utils/logging.py

Centralized logging setup for the benchmark. All modules should call
`get_logger(__name__)` instead of using `print`, so that verbosity is
configurable in one place (configs/default.yaml -> logging section) and
logs can optionally be persisted to disk for later inspection (e.g. on
Kaggle, where stdout from a background run can be lost).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_CONFIGURED = False


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_filename: str = "run.log",
) -> None:
    """Configure the root logger once for the whole process.

    Safe to call multiple times; only the first call takes effect.

    Args:
        level: Logging level name, e.g. "DEBUG", "INFO", "WARNING".
        log_dir: If provided, also write logs to `log_dir/log_filename`.
        log_filename: Filename to use when log_dir is provided.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path / log_filename))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a module-level logger, configuring defaults if not already done.

    Args:
        name: Usually `__name__` of the calling module.

    Returns:
        A configured `logging.Logger` instance.
    """
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
