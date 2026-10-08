#logging/structured_logger.py
"""
Structured logging.

Console: one human-readable line per record, for local runs.
File (optional): one JSON record per line, so logs can be shipped to any
aggregator. All modules log under "paperlens.<layer>".
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created, timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Extra context: logger.info("...", extra={"ctx_document_id": "abc"})
        for key, value in record.__dict__.items():
            if key.startswith("ctx_"):
                payload[key[4:]] = value

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    root = logging.getLogger("paperlens")
    root.setLevel(level.upper())
    root.handlers.clear()
    root.propagate = False

    # A Windows console (cp1252) cannot print every character found in
    # papers: replace instead of letting logging print a traceback.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(JSONFormatter())
        root.addHandler(file_handler)