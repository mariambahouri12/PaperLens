"""
Structured logging configuration for PaperLens.

Console:
    Human-readable logs for local development and execution.

File (optional):
    One JSON record per line, suitable for log shipping and aggregation.

All application loggers should use the ``paperlens.<layer>`` namespace.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                timezone.utc,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        context: dict[str, Any] = {}

        for key, value in record.__dict__.items():
            if key.startswith("ctx_"):
                context[key[4:]] = value

        if context:
            payload["context"] = context

        return json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )


def configure_logging(
    level: str = "INFO",
    log_file: Path | None = None,
) -> None:
    """
    Configure PaperLens logging.

    Calling this function multiple times replaces the previous PaperLens
    handlers instead of creating duplicate log entries.
    """
    logger = logging.getLogger("paperlens")

    logger.setLevel(level.upper())
    logger.handlers.clear()
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | "
            "%(name)s | %(message)s"
        )
    )
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            log_file,
            encoding="utf-8",
        )
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
