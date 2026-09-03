"""Compatibility shims for the hosted verdict API.

Since 0.3.1 the service lives inside the installable package as
``harness.service`` (app, models, payments, report, settings, logs, and the
Vercel entry ``harness.service.vercel``). The modules in this directory only
re-export it so that ``uvicorn api.app:app``, the Dockerfile of earlier releases
and ``from api.settings import Settings`` keep working.

Entry points:
    harness.service.app:app                  module-level ASGI app (configured from the env)
    harness.service.app.create_app           factory taking an explicit ``Settings``
    harness.service.vercel.create_vercel_app Vercel Python function (web/api/judge/index.py)
"""
from __future__ import annotations

__all__ = ["__version__"]

try:  # the installed distribution is the source of truth for the version
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    __version__ = _pkg_version("quant-harness")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0"
