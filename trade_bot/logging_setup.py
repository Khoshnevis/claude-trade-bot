"""Structured logging configuration."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler


def setup_logging(level: str = "INFO", file: str | None = "trade_bot.log") -> logging.Logger:
    logger = logging.getLogger("trade_bot")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if file:
        fh = RotatingFileHandler(file, maxBytes=10 * 1024 * 1024, backupCount=5)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
