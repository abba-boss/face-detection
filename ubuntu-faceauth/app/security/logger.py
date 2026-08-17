"""
Centralised logging for Ubuntu FaceAuth.

Rules:
  - Never log raw embeddings.
  - Never log biometric pixel data.
  - Structured log lines make grep-based auditing easy.

Singleton behaviour
-------------------
get_logger(name) returns the same Logger instance on repeated calls
so handlers are never duplicated.  The one exception is when a caller
provides a log_file that the cached instance doesn't yet have — in
that case the file handler is added to the existing logger.  This
handles the common pattern where modules are imported (creating a
console-only logger) before main() has had a chance to set up the
log file path.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str,
               log_file: Optional[Path] = None,
               level: str = "INFO") -> logging.Logger:
    """
    Return a named logger.

    - First call: creates the logger with a console handler and
      an optional file handler.
    - Subsequent calls with the same name: return the cached
      instance.  If log_file is provided and the logger does not
      yet have a FileHandler, the file handler is added now.
    """
    if name in _loggers:
        logger = _loggers[name]
        # Add file handler retroactively if not already present
        if log_file and not _has_file_handler(logger):
            _add_file_handler(logger, log_file)
        return logger

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — always present
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler — optional
    if log_file:
        _add_file_handler(logger, log_file)

    _loggers[name] = logger
    return logger


# ── Helpers ──────────────────────────────────────────────────────────────

def _has_file_handler(logger: logging.Logger) -> bool:
    return any(isinstance(h, logging.FileHandler) for h in logger.handlers)


def _add_file_handler(logger: logging.Logger, log_file: Path) -> None:
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
