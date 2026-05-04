"""Logger estruturado compartilhado entre módulos."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Retorna um ``logging.Logger`` configurado e idempotente por ``name``."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
