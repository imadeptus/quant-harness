"""Vercel Python function entry for the judge.

The product (``web/``, Next.js) ships the judge as one Python function,
``web/api/judge/index.py``, served at ``/api/judge`` (``vercel.json`` rewrites
``/api/judge/:path*`` to it). That file is two lines::

    from harness.service.vercel import create_vercel_app
    app = create_vercel_app()          # `app`: the name Vercel loads

This module builds the ``Settings`` for that deployment from the environment:

- ``QH_ROOT_PATH`` defaults to ``/api/judge`` (``DEFAULT_ROOT_PATH``);
- the Node layer's variable names are bridged when the ``QH_*`` ones are unset,
  so one Vercel env config drives both halves: ``JUDGE_INTERNAL_SECRET`` ->
  ``QH_INTERNAL_SECRET``, ``MAX_CONFIGS`` / ``MAX_PERIODS`` / ``MAX_BODY_MB`` ->
  ``QH_MAX_CONFIGS`` / ``QH_MAX_PERIODS`` / ``QH_MAX_BODY_BYTES``, ``APP_URL`` ->
  ``QH_PUBLIC_URL``;
- ``VERCEL_ENV=production`` without a secret is a hard error at import time — an
  open judge would otherwise be reachable by anyone at ``/api/judge/v1/verdict``.

Nothing here mutates ``os.environ``.
"""
from __future__ import annotations

import dataclasses
import os
from typing import Dict, Optional

from fastapi import FastAPI

from .app import create_app
from .logs import configure_logging
from .settings import EnvMap, Settings, _env_int, _env_str

DEFAULT_ROOT_PATH = "/api/judge"

# Node-layer name -> judge name, for the integer limits (same unit).
_BRIDGED_INTS: Dict[str, str] = {"MAX_CONFIGS": "QH_MAX_CONFIGS", "MAX_PERIODS": "QH_MAX_PERIODS"}


def _bridged_env(env: EnvMap) -> Dict[str, str]:
    """A copy of ``env`` with the product's variable names mapped onto ``QH_*``
    whenever the ``QH_*`` variant is unset (``QH_*`` always wins)."""
    out: Dict[str, str] = dict(env)
    if _env_str(env, "QH_ROOT_PATH") is None:
        out["QH_ROOT_PATH"] = DEFAULT_ROOT_PATH
    if _env_str(env, "QH_INTERNAL_SECRET") is None and _env_str(env, "JUDGE_INTERNAL_SECRET"):
        out["QH_INTERNAL_SECRET"] = env["JUDGE_INTERNAL_SECRET"]
    if _env_str(env, "QH_PUBLIC_URL") is None and _env_str(env, "APP_URL"):
        out["QH_PUBLIC_URL"] = env["APP_URL"]
    for node_name, qh_name in _BRIDGED_INTS.items():
        if _env_str(env, qh_name) is None and _env_str(env, node_name) is not None:
            out[qh_name] = str(_env_int(env, node_name, 0))   # validates: int, positive
    if _env_str(env, "QH_MAX_BODY_BYTES") is None and _env_str(env, "MAX_BODY_MB") is not None:
        out["QH_MAX_BODY_BYTES"] = str(_env_int(env, "MAX_BODY_MB", 0) * 1024 * 1024)
    return out


def vercel_settings(env: Optional[EnvMap] = None) -> Settings:
    """``Settings`` for the Vercel deployment (see the module docstring).

    Raises ``RuntimeError`` when ``VERCEL_ENV=production`` and no internal secret
    is configured; logs a warning (and serves open) in any other environment.
    """
    e: EnvMap = os.environ if env is None else env
    settings = Settings.from_env(_bridged_env(e))
    if settings.internal_secret is None:
        vercel_env = _env_str(e, "VERCEL_ENV")
        message = ("no internal secret configured: set JUDGE_INTERNAL_SECRET (or "
                   "QH_INTERNAL_SECRET) so only the product's Node layer can call the judge")
        if vercel_env == "production":
            raise RuntimeError(f"refusing to start an open judge in production — {message}")
        configure_logging(settings.log_level).warning(
            message, extra={"fields": {"vercel_env": vercel_env, "root_path": settings.root_path}})
    return dataclasses.replace(settings)


def create_vercel_app(env: Optional[EnvMap] = None) -> FastAPI:
    """The ASGI app Vercel serves at ``/api/judge`` (``web/api/judge/index.py``)."""
    return create_app(vercel_settings(env))
