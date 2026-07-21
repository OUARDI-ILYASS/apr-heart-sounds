"""Structured logging: pretty on the console, complete in the log file.

Each phase writes its own log to reports/<phase>/run.log, so a failed run
leaves a full record behind even when the console scrollback is gone.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_FMT_CONSOLE = "%(asctime)s | %(levelname)-7s | %(message)s"
_FMT_FILE = "%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d | %(message)s"
_DATEFMT = "%H:%M:%S"


class _ColourFormatter(logging.Formatter):
    COLOURS = {
        "DEBUG": "\033[90m", "INFO": "\033[0m", "WARNING": "\033[33m",
        "ERROR": "\033[31m", "CRITICAL": "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if sys.stderr.isatty():
            return f"{self.COLOURS.get(record.levelname, '')}{text}{self.RESET}"
        return text


def get_logger(name: str, log_file: Optional[str | Path] = None,
               level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(_ColourFormatter(_FMT_CONSOLE, datefmt=_DATEFMT))
    logger.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FMT_FILE))
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    return logger


def banner(logger: logging.Logger, title: str, width: int = 78) -> None:
    logger.info("=" * width)
    logger.info(title.center(width))
    logger.info("=" * width)


def section(logger: logging.Logger, title: str, width: int = 78) -> None:
    logger.info("")
    logger.info(f"--- {title} " + "-" * max(0, width - len(title) - 5))
