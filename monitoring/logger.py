"""Structured (JSON-lines) logging setup.

Console output is human-readable; file output is one JSON object per line with
timestamp, level, logger, message, and any ``extra`` context fields.
"""

from __future__ import annotations

import logging
from typing import Any


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise ``record`` (including ``extra`` fields) to JSON."""
        raise NotImplementedError


def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    json_file: bool = True,
) -> None:
    """Configure root logging with console and (optionally) rotating JSON file handlers."""
    raise NotImplementedError


def get_logger(name: str) -> logging.Logger:
    """Return a module logger under the ``regime_trader`` namespace."""
    raise NotImplementedError


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Log a named structured event with arbitrary key/value context."""
    raise NotImplementedError
