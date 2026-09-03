"""Hosted verdict service — quant-harness as an HTTP judge (FastAPI).

"Upload returns, get a calibrated verdict": a thin ASGI layer over
``harness.audit.audit_returns`` (CPCV + Deflated Sharpe + PBO + the mechanical
PASS/KILL verdict), the same function behind the ``qh-audit`` CLI. Built for AI
agents and humans alike. Requires the ``api`` extra::

    pip install "quant-harness[api]"

Entry points:
    harness.service.app:app                  module-level ASGI app configured from the env
                                             (``uvicorn harness.service.app:app``; Docker)
    harness.service.app.create_app           factory taking an explicit ``Settings``
    harness.service.vercel.create_vercel_app Vercel Python function entry (mounted at
                                             /api/judge, ``web/api/judge/index.py``)

Env: ``QH_INTERNAL_SECRET`` (X-Internal-Secret gate), ``QH_ROOT_PATH`` (mount prefix),
``QH_MAX_*`` limits, ``QH_API_KEYS``; see ``api/README.md``. The ``api/*`` modules of
the source tree are compatibility shims re-exporting this package.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # typed names for mypy; resolved lazily at runtime (no fastapi import here)
    from .app import create_app
    from .settings import Settings

__all__ = ["create_app", "Settings"]


def __getattr__(name: str) -> Any:
    """Lazy exports (PEP 562): ``Settings`` needs no fastapi; ``create_app`` does."""
    if name == "Settings":
        from .settings import Settings
        return Settings
    if name == "create_app":
        from .app import create_app
        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
