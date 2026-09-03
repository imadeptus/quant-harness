"""Compatibility shim — the service moved to ``harness.service.app`` in 0.3.1.

``uvicorn api.app:app`` and ``from api.app import create_app`` keep working; new
code should import ``harness.service.app`` directly.
"""
from __future__ import annotations

from harness.service.app import (DEFAULT_START, HEALTH_PATH, INTERNAL_SECRET_HEADER,
                                 AuthGateMiddleware, BodySizeLimitMiddleware,
                                 InternalSecretMiddleware, RequestLogMiddleware,
                                 api_version, app, compute_verdict, create_app, route_path)

__all__ = ["DEFAULT_START", "HEALTH_PATH", "INTERNAL_SECRET_HEADER", "AuthGateMiddleware",
           "BodySizeLimitMiddleware", "InternalSecretMiddleware", "RequestLogMiddleware",
           "api_version", "app", "compute_verdict", "create_app", "route_path"]
