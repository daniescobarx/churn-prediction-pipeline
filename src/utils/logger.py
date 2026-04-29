"""Logger estruturado reutilizável em todos os módulos do projeto."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Configura e retorna um logger estruturado.

    Idempotente: chamadas subsequentes com o mesmo ``name`` reutilizam
    o logger já configurado e não duplicam handlers.

    Args:
        name: nome do módulo chamador (tipicamente ``__name__``).

    Returns:
        Instância de :class:`logging.Logger` pronta para uso.
    """
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
