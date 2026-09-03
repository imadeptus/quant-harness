"""Compatibility shim — JSON logging moved to ``harness.service.logs`` in 0.3.1."""
from __future__ import annotations

from harness.service.logs import LOGGER_NAME, JsonFormatter, StdoutHandler, configure_logging

__all__ = ["LOGGER_NAME", "JsonFormatter", "StdoutHandler", "configure_logging"]
