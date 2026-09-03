"""Compatibility shim — the report renderer moved to ``harness.service.report`` in 0.3.1."""
from __future__ import annotations

from harness.service.report import CALIBRATION_NOTE, DISCLAIMER, render_report

__all__ = ["CALIBRATION_NOTE", "DISCLAIMER", "render_report"]
