"""Structured (JSON-lines) logging shared across the application."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """Minimal structured formatter; keeps logs greppable and machine-readable."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("event", "analysis_id", "model", "duration_ms", "error"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


_CONFIGURED = False


def configure_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger("ecolens")
    root.setLevel(level)
    root.propagate = False

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(JsonFormatter())
    root.addHandler(stream)

    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(JsonFormatter())
            root.addHandler(file_handler)
        except OSError:
            # Logging must never crash the application.
            pass
    _CONFIGURED = True


def get_logger(name: str = "ecolens") -> logging.Logger:
    if not name.startswith("ecolens"):
        name = f"ecolens.{name}"
    return logging.getLogger(name)
