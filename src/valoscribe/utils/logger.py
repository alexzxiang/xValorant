from __future__ import annotations
import logging
import sys
from typing import Optional


_DEFAULT_FMT = "%(asctime)s [%(levelname)s] (%(name)s): %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    log_path: Optional[str] = None,
    *,
    fmt: str = _DEFAULT_FMT,
    datefmt: str = _DEFAULT_DATEFMT,
) -> None:
    """
    Configure root logging once per process.
    - Console handler to stdout
    - Optional file handler (append mode)
    Calling this multiple times is safe; extra handlers won't be duplicated.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # dedupe if already configured
    if any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        return

    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root.addHandler(stream)

    if log_path:
        fileh = logging.FileHandler(log_path, encoding="utf-8")
        fileh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        root.addHandler(fileh)


def get_logger(name: str) -> logging.Logger:
    """
    Module-level helper.
    Usage:
        from valoscribe.utils.logger import get_logger
        log = get_logger(__name__)
        log.info("hello")
    """
    return logging.getLogger(name)