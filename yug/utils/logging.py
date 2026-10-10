"""Structured logging for YUG.

Chief Implementation Directive Section 15 requires structured logs
capable of reconstructing what happened. This is implemented on top of
Python's standard-library ``logging`` module with a custom JSON
formatter, rather than a third-party structured-logging library
(e.g. structlog): the stdlib already provides everything Phase 0 needs
-- named loggers, levels, handlers, file output -- so an additional
dependency is not justified yet. If a future phase needs capabilities
the stdlib genuinely lacks (e.g. async-safe context binding at scale),
that will be evaluated then, against a concrete requirement.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CONFIGURED = False

# Reserved LogRecord attributes that must not be treated as "extra" fields
# when flattening a record to JSON.
_RESERVED_RECORD_ATTRS = frozenset(
    logging.LogRecord(
        name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None
    ).__dict__.keys()
) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Formats each log record as a single-line JSON object.

    Fields: timestamp (ISO 8601, UTC), level, logger name, message, and
    any caller-supplied ``extra`` fields. Exception info, if present, is
    included as a formatted string.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_RECORD_ATTRS
        }
        if extras:
            payload["extra"] = extras

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(
    level: str = "INFO",
    log_dir: str | Path | None = None,
    stream: bool = True,
) -> None:
    """Configure the root ``yug`` logger with structured JSON output.

    Args:
        level: Standard logging level name (e.g. "DEBUG", "INFO").
        log_dir: If given, a ``yug.log`` file is created there in
            addition to (or instead of) stream output. The directory is
            created if it does not exist.
        stream: Whether to also log to stderr.

    Safe to call more than once; each call replaces prior handlers on
    the ``yug`` logger rather than stacking duplicates.
    """
    global _CONFIGURED

    root = logging.getLogger("yug")
    root.setLevel(level.upper())
    root.handlers.clear()

    formatter = JsonFormatter()

    if stream:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    if log_dir is not None:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir_path / "yug.log")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``yug``.

    If :func:`configure_logging` has not been called yet, applies a
    minimal default configuration (INFO level, stderr only) so that
    log calls before explicit setup do not raise or silently vanish.
    """
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(f"yug.{name}" if not name.startswith("yug.") else name)
