"""Hosted verdict API — upload returns, get a calibrated PASS/KILL verdict.

    uvicorn api.app:app --host 0.0.0.0 --port 8000

Request flow for ``POST /v1/verdict``: body-size limit (413, from Content-Length)
-> API key (401) -> payment gate (402, stub) -> schema validation (422) -> matrix
limits (413) -> judge (``harness.audit.audit_returns``, the same function behind
``qh-audit``) -> JSON. 401/402 are decided from the headers alone, in ASGI
middleware, before any of the body is read or parsed. One JSON log line per
request goes to stdout.
"""
from __future__ import annotations

import json
import secrets
import time
from math import isfinite
from typing import (Any, Awaitable, Callable, Dict, List, MutableMapping, Optional,
                    Sequence, Tuple)

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from starlette.exceptions import HTTPException as StarletteHTTPException

from harness import __version__ as harness_version
from harness.audit import AuditInputError, audit_returns

from .logs import configure_logging
from .models import CostSensitivityRow, Metrics, VerdictRequest, VerdictResponse
from .payments import (STUB_ASSUMPTION, STUB_HEADER, STUB_HEADER_VALUE, PaymentGate,
                       build_gate)
from .report import DISCLAIMER, render_report
from .settings import Settings

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

DEFAULT_START = "2020-01-01T00:00:00Z"


def api_version() -> str:
    """One version string for /healthz, every response and every qh-audit report:
    ``harness.__version__`` (pinned to pyproject by tests/test_version.py)."""
    return harness_version


# ---- middleware (pure ASGI, so they also see the final status code) -------------

class BodySizeLimitMiddleware:
    """Reject bodies above ``max_bytes`` with 413 — via Content-Length up front and,
    for chunked uploads, by counting bytes as they stream in."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        detail = (f"request body exceeds QH_MAX_BODY_BYTES={self.max_bytes}; "
                  "send fewer configs/periods or raise the limit")
        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            response = JSONResponse(status_code=413, content={"detail": detail, "error": detail})
            await response(scope, receive, send)
            return

        seen = 0

        async def limited_receive() -> MutableMapping[str, Any]:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    raise HTTPException(status_code=413, detail=detail)
            return message

        await self.app(scope, limited_receive, send)


class RequestLogMiddleware:
    """One JSON line per request: path, status, ms, n_configs, n_periods, verdict."""

    def __init__(self, app: ASGIApp, logger: Any) -> None:
        self.app = app
        self.logger = logger

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        t0 = time.perf_counter()
        status: Optional[int] = None

        async def send_wrapper(message: MutableMapping[str, Any]) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            state = scope.get("state") or {}
            self.logger.info("request", extra={"fields": {
                "path": scope.get("path"),
                "method": scope.get("method"),
                "status": status if status is not None else 500,
                "ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "n_configs": state.get("n_configs"),
                "n_periods": state.get("n_periods"),
                "verdict": state.get("verdict"),
            }})


def _key_accepted(supplied: Optional[str], api_keys: Tuple[str, ...]) -> bool:
    given = (supplied or "").encode()
    return any(secrets.compare_digest(given, k.encode()) for k in api_keys)


def _with_response_header(send: Send, name: str, value: str) -> Send:
    async def wrapped(message: MutableMapping[str, Any]) -> None:
        if message["type"] == "http.response.start":
            headers = list(message.get("headers") or [])
            headers.append((name.lower().encode(), value.encode()))
            message = {**message, "headers": headers}
        await send(message)
    return wrapped


class AuthGateMiddleware:
    """401 (API key) and 402 (payment gate) for ``/v1/*``, decided from the headers
    alone — before a single byte of the body is read, so an unauthenticated client
    cannot make the server parse up to QH_MAX_BODY_BYTES of JSON. A request served
    through a stub gate is marked with ``X-QH-Payment: unverified-stub`` and a
    ``PAYMENT:`` assumption line."""

    PROTECTED_PREFIX = "/v1/"

    def __init__(self, app: ASGIApp, api_keys: Tuple[str, ...], gate: PaymentGate) -> None:
        self.app = app
        self.api_keys = api_keys
        self.gate = gate

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not str(scope.get("path", "")).startswith(self.PROTECTED_PREFIX):
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive)      # headers and path only; body untouched
        if self.api_keys and not _key_accepted(request.headers.get("X-API-Key"), self.api_keys):
            detail = "missing or invalid X-API-Key"
            response: Response = JSONResponse(status_code=401,
                                              content={"detail": detail, "error": detail},
                                              headers={"WWW-Authenticate": "ApiKey"})
            await response(scope, receive, send)
            return
        challenge = self.gate.check(request)
        if challenge is not None:
            await challenge(scope, receive, send)
            return
        if self.gate.is_stub:
            scope.setdefault("state", {})["payment"] = STUB_HEADER_VALUE
            send = _with_response_header(send, STUB_HEADER, STUB_HEADER_VALUE)
        await self.app(scope, receive, send)


# ---- the judge call --------------------------------------------------------------

def _finite_or_none(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if isfinite(v) else None


def _start_timestamp(req: VerdictRequest) -> pd.Timestamp:
    ts = pd.Timestamp(req.start) if req.start is not None else pd.Timestamp(DEFAULT_START)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _sensitivity_rows(report: Dict[str, Any]) -> Optional[List[CostSensitivityRow]]:
    table = report["cost_sensitivity"]
    if not table["available"]:
        return None
    return [CostSensitivityRow(
        multiplier=float(row["multiplier"]), costs_bps=float(row["costs_bps"]),
        oos_sharpe_annualized=_finite_or_none(row["oos_sharpe_annualized"]),
        worst_path_sharpe_annualized=_finite_or_none(row["worst_path_sharpe_annualized"]),
        oos_max_drawdown=_finite_or_none(row["oos_max_drawdown"]),
        deflated_sharpe_ratio=_finite_or_none(row["deflated_sharpe_ratio"]),
        verdict=row["verdict"]) for row in table["rows"]]


def compute_verdict(req: VerdictRequest, version: str,
                    extra_assumptions: Optional[Sequence[str]] = None) -> VerdictResponse:
    """Pure function: validated request -> response.

    Delegates the whole evaluation to ``harness.audit.audit_returns`` — the same
    function behind ``qh-audit`` — so the HTTP verdict can never drift from the CLI
    one. Raises HTTPException(422) when the audit cannot evaluate the data (e.g. a
    zero-variance series, CPCV groups too small for the data, or a date index that
    cannot be built from ``start``/``freq``). ``extra_assumptions`` are request-level
    lines (e.g. the stub-payment marker) appended to the response and the report.
    """
    R = req.returns_matrix()
    n_cfg, T = R.shape
    try:
        index = pd.date_range(_start_timestamp(req), periods=T,
                              freq=pd.Timedelta(seconds=req.freq_seconds))
    except (OverflowError, ValueError, pd.errors.OutOfBoundsDatetime,
            pd.errors.OutOfBoundsTimedelta) as exc:
        raise HTTPException(status_code=422,
                            detail=f"cannot build a date index from start/freq: {exc}") from exc
    try:
        report = audit_returns(
            R, req.trades_matrix(), index, n_trials=req.n_trials,
            thresholds=req.thresholds.to_thresholds(), cpcv=req.cpcv.to_config(),
            costs_bps=req.costs_bps, assume_trades_per_bar=req.assume_trades_per_bar,
            title="verdict API request")
    except AuditInputError as exc:
        raise HTTPException(status_code=422,
                            detail=f"judge cannot evaluate this input: {exc}") from exc

    judge = report["judge"]
    n_eff = int(report["n_trials_effective"])

    # The audit's own assumptions/warnings first, then the request-level ones.
    assumptions: List[str] = list(report["assumptions"])
    assumptions += [f"WARNING: {w}" for w in report["warnings"]]
    if req.costs_bps is not None:
        assumptions.append(
            f"costs_bps={req.costs_bps}: returns treated as gross; {req.costs_bps} bps x trades "
            "subtracted on every bar before judging.")
    else:
        assumptions.append("returns treated as already net of fees, slippage and funding "
                           "(no costs_bps supplied).")
    if n_eff > n_cfg:
        assumptions.append(
            f"n_trials={n_eff} > uploaded configs ({n_cfg}): DSR deflated by n_trials; "
            f"trial-Sharpe variance source: {judge['trial_variance_source']}.")
    if n_cfg < 2:
        assumptions.append("PBO not computed: it needs at least 2 uploaded configs.")
    if req.costs_bps is not None and report["cost_sensitivity"]["reason"]:
        assumptions.append(f"cost sensitivity not computed: {report['cost_sensitivity']['reason']}.")
    assumptions.append(f"freq={req.freq}: annualization factor {judge['ann_factor']} periods/year "
                       "inferred from bar spacing; CPCV verdict uses the median across paths.")
    assumptions += list(extra_assumptions or ())

    metrics = Metrics(
        oos_sharpe_annualized=_finite_or_none(judge["oos_sharpe_annualized"]),
        worst_path_sharpe_annualized=_finite_or_none(judge["worst_path_sharpe_annualized"]),
        oos_max_drawdown=_finite_or_none(judge["oos_max_drawdown"]),
        psr_vs_zero=_finite_or_none(judge["psr_vs_zero"]),
        deflated_sharpe_ratio=_finite_or_none(judge["deflated_sharpe_ratio"]),
        pbo=_finite_or_none(report["pbo"]),
        n_paths=int(judge["n_paths"]),
        n_configs_tried=n_eff,
        oos_bars=int(judge["oos_bars"]),
        approx_oos_trades=int(judge["approx_oos_trades"]),
        ann_factor=float(judge["ann_factor"]),
    )
    checks = {c["key"]: bool(c["ok"]) for c in report["checks"]}
    thresholds: Dict[str, float] = dict(report["thresholds"])   # asdict(Thresholds)
    report_md = render_report(report["verdict"], checks, metrics.model_dump(), thresholds,
                              assumptions, version)
    return VerdictResponse(
        verdict=report["verdict"], checks=checks, metrics=metrics, thresholds=thresholds,
        cpcv={k: int(v) for k, v in report["cpcv"].items()}, assumptions=assumptions,
        cost_sensitivity=_sensitivity_rows(report), report_md=report_md,
        disclaimer=DISCLAIMER, version=version,
    )


# ---- app factory -----------------------------------------------------------------

def create_app(settings: Settings) -> FastAPI:
    logger = configure_logging(settings.log_level)
    gate: PaymentGate = build_gate(settings, logger)
    if gate.is_stub:
        logger.warning("payment gate %s is a STUB: 402 challenge only, payment is never "
                       "verified (QH_ALLOW_STUB_PAYMENT_GATE=1)", gate.name,
                       extra={"fields": {"gate": gate.name}})
    version = api_version()

    app = FastAPI(
        title="quant-harness verdict API",
        version=version,
        description=("Upload per-bar returns, get a mechanical PASS/KILL verdict backed by "
                     "Combinatorial Purged CV, the Deflated Sharpe Ratio and PBO. " + DISCLAIMER),
    )
    app.state.settings = settings
    app.state.payment_gate = gate

    # add_middleware() wraps outermost-last: request log (outer) -> body-size limit
    # (413 from Content-Length) -> auth/payment (401/402, headers only) -> app.
    app.add_middleware(AuthGateMiddleware, api_keys=settings.api_keys, gate=gate)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(RequestLogMiddleware, logger=logger)

    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def require_api_key(key: Optional[str] = Security(api_key_header)) -> None:
        """Declares the security scheme for /docs; the middleware has already
        enforced it by the time this runs."""
        if settings.api_keys and not _key_accepted(key, settings.api_keys):
            raise HTTPException(status_code=401, detail="missing or invalid X-API-Key",
                                headers={"WWW-Authenticate": "ApiKey"})

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> Response:
        # Never echo `input` back: for a returns matrix that is the whole request body.
        detail = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")}
                  for e in exc.errors()]
        summary = "; ".join(f"{'.'.join(str(x) for x in d['loc'])}: {d['msg']}" for d in detail)
        return JSONResponse(status_code=422, content={"detail": detail, "error": summary})

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> Response:
        error = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail, "error": error},
                            headers=exc.headers)

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok", "version": version}

    @app.post("/v1/verdict", response_model=VerdictResponse,
              dependencies=[Depends(require_api_key)],
              responses={401: {"description": "missing/invalid API key"},
                         402: {"description": "payment required (gate challenge)"},
                         413: {"description": "matrix or body over the configured limits"},
                         422: {"description": "input the judge cannot evaluate honestly"}})
    def verdict(req: VerdictRequest, request: Request) -> VerdictResponse:
        # Sync on purpose: the judge is CPU-bound numpy; FastAPI runs it in the threadpool
        # so the event loop keeps serving /healthz.
        n_cfg, n_periods = req.n_configs, req.n_periods
        request.state.n_configs = n_cfg
        request.state.n_periods = n_periods
        if n_cfg > settings.max_configs:
            raise HTTPException(status_code=413, detail=(
                f"{n_cfg} configs exceed QH_MAX_CONFIGS={settings.max_configs}"))
        if n_periods > settings.max_periods:
            raise HTTPException(status_code=413, detail=(
                f"{n_periods} periods exceed QH_MAX_PERIODS={settings.max_periods}"))
        extra: List[str] = []
        if request.scope.get("state", {}).get("payment") == STUB_HEADER_VALUE:
            extra.append(STUB_ASSUMPTION)
        response = compute_verdict(req, version, extra_assumptions=extra)
        request.state.verdict = response.verdict
        return response

    return app


app = create_app(Settings.from_env())
