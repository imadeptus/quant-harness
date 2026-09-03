"""Hosted verdict API for quant-harness.

"Upload returns, get a calibrated verdict": a thin FastAPI layer over
``harness.runner.run_cpcv_returns`` (CPCV + Deflated Sharpe + PBO + mechanical
PASS/KILL). Built for AI agents and humans alike; per-call payment hooks are
STUBS — see ``api/README.md``.

Entry points:
    api.app:app            module-level ASGI app (configured from the env)
    api.app.create_app     factory taking an explicit ``api.settings.Settings``
"""
from __future__ import annotations

__all__ = ["__version__"]

try:  # the installed distribution is the source of truth for the version
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    __version__ = _pkg_version("quant-harness")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0"
