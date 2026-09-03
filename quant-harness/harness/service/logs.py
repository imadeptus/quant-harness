"""JSON logging to stdout — one line per record, standard library only (12-factor XI)."""
from __future__ import annotations

import datetime as _dt
import json
import logging
import sys
from typing import Any, Dict

LOGGER_NAME = "qh.api"


class JsonFormatter(logging.Formatter):
    """Serialise a record as a single JSON object. Structured fields are passed
    via ``logger.info(msg, extra={"fields": {...}})`` and merged at top level."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class StdoutHandler(logging.Handler):
    """Writes to whatever ``sys.stdout`` is *at emit time*, so stream redirection
    (pytest capsys, process supervisors) is honoured."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            sys.stdout.write(self.format(record) + "\n")
            sys.stdout.flush()
        except (OSError, ValueError):  # a closed/broken stdout must never crash a request
            self.handleError(record)


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Idempotent: attaches exactly one JSON stdout handler to the API logger."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    if not any(isinstance(h, StdoutHandler) for h in logger.handlers):
        handler = StdoutHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger
