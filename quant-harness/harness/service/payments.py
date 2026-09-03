"""Payment gates — pluggable pre-checks on ``POST /v1/verdict``.

Only ``NoopGate`` is real. ``X402Gate`` and ``NowPaymentsGate`` are **STUBS**:
they emit a correct-looking 402 challenge when the payment header is absent and
let the request through when it is present, WITHOUT verifying anything. No
network call is made, no signature is checked, no money moves. Turning them
into real gates is documented in api/README.md, section
"Payments: what is real and what is a stub".
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Protocol

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from .settings import Settings

README_PAYMENTS = "api/README.md#payments-what-is-real-and-what-is-a-stub"
STUB_NOTE = ("STUB: this gate only issues the 402 challenge; it does not verify payment. "
             f"See {README_PAYMENTS}.")
# Every response served through a stub gate carries this header and assumption line,
# so neither the client nor the operator can mistake it for a paid, verified call.
STUB_HEADER = "X-QH-Payment"
STUB_HEADER_VALUE = "unverified-stub"
STUB_ASSUMPTION = ("PAYMENT: served through a stub payment gate — the payment header was "
                   "present but NOT verified (no facilitator / provider lookup).")


class PaymentGate(Protocol):
    """Return ``None`` to let the request through, or a ``Response`` (normally a
    402) to short-circuit it."""

    name: str
    is_stub: bool

    def check(self, request: Request) -> Optional[Response]: ...


class NoopGate:
    """Free access. The default and the only fully real gate."""

    name = "noop"
    is_stub = False

    def check(self, request: Request) -> Optional[Response]:
        return None


def usdc_atomic_units(price_usdc: float) -> str:
    """USDC has 6 decimals; x402 quotes ``maxAmountRequired`` in atomic units as a string."""
    return str(int(round(price_usdc * 1_000_000)))


class X402Gate:
    """x402-style challenge (HTTP 402 + ``accepts`` list) for the ``X-PAYMENT`` header.

    TODO(stub): verification is NOT implemented. A real gate must forward the
    ``X-PAYMENT`` payload to an x402 facilitator (``/verify`` then ``/settle``)
    and only serve the verdict once settlement is confirmed. See README_PAYMENTS.
    """

    name = "x402"
    HEADER = "X-PAYMENT"
    is_stub = True

    def __init__(self, pay_to: str, price_usdc: float, network: str, asset: str,
                 public_url: str, logger: logging.Logger) -> None:
        self.pay_to = pay_to
        self.price_usdc = price_usdc
        self.network = network
        self.asset = asset
        self.public_url = public_url
        self._log = logger

    def requirements(self, request: Request) -> Dict[str, Any]:
        return {
            "scheme": "exact",
            "network": self.network,
            "maxAmountRequired": usdc_atomic_units(self.price_usdc),
            "resource": f"{self.public_url}{request.url.path}",
            "description": (f"quant-harness calibrated PASS/KILL verdict on uploaded returns "
                            f"({self.price_usdc} USDC per call)"),
            "mimeType": "application/json",
            "payTo": self.pay_to,
            "maxTimeoutSeconds": 60,
            "asset": self.asset,
            "extra": {"name": "USDC", "version": "2"},
        }

    def check(self, request: Request) -> Optional[Response]:
        if request.headers.get(self.HEADER) is None:
            return JSONResponse(status_code=402, content={
                "x402Version": 1,
                "error": f"{self.HEADER} header is required",
                "accepts": [self.requirements(request)],
                "stub": True,
                "stub_note": STUB_NOTE,
            })
        # TODO(stub): verify + settle via facilitator before letting the request through.
        self._log.warning("x402 payment header present; verification not implemented "
                          "(stub) - request allowed",
                          extra={"fields": {"gate": self.name, "path": request.url.path}})
        return None


class NowPaymentsGate:
    """NOWPayments-style gate for the ``X-Payment-Id`` header.

    TODO(stub): verification is NOT implemented. A real gate must look the id up
    via ``GET https://api.nowpayments.io/v1/payment/{id}`` (or trust a signed IPN
    callback) and require status ``finished`` before serving. See README_PAYMENTS.
    """

    name = "nowpayments"
    HEADER = "X-Payment-Id"
    is_stub = True

    def __init__(self, api_key: str, price_usdc: Optional[float], logger: logging.Logger) -> None:
        self._api_key = api_key          # never logged, never echoed
        self.price_usdc = price_usdc
        self._log = logger

    def check(self, request: Request) -> Optional[Response]:
        if request.headers.get(self.HEADER) is None:
            price = f"{self.price_usdc} USDC" if self.price_usdc is not None else "the listed price"
            return JSONResponse(status_code=402, content={
                "error": f"{self.HEADER} header is required",
                "provider": "nowpayments",
                "price_usdc": self.price_usdc,
                "instructions": (f"Create a NOWPayments payment for {price} (pay_currency usdc), "
                                 f"wait for status 'finished', then retry this request with "
                                 f"header {self.HEADER}: <payment_id>."),
                "stub": True,
                "stub_note": STUB_NOTE,
            })
        # TODO(stub): GET /v1/payment/{id} with the API key and require status == "finished".
        self._log.warning("NOWPayments payment id present; verification not implemented "
                          "(stub) - request allowed",
                          extra={"fields": {"gate": self.name, "path": request.url.path}})
        return None


def build_gate(settings: Settings, logger: logging.Logger) -> PaymentGate:
    """Select the gate from ``settings.payment_gate``; fail fast on a half-configured one."""
    if settings.payment_gate == "noop":
        return NoopGate()
    if not settings.allow_stub_payment_gate:
        raise ValueError(
            f"QH_PAYMENT_GATE={settings.payment_gate} is a stub: it issues the 402 challenge "
            f"but does not verify payment. Set QH_ALLOW_STUB_PAYMENT_GATE=1 to run it "
            f"knowingly (never in front of paid traffic); see {README_PAYMENTS}")
    if settings.payment_gate == "x402":
        if not settings.x402_pay_to or settings.x402_price_usdc is None:
            raise ValueError("QH_PAYMENT_GATE=x402 requires QH_X402_PAY_TO and QH_X402_PRICE_USDC")
        if settings.x402_price_usdc <= 0:
            raise ValueError("QH_X402_PRICE_USDC must be positive")
        return X402Gate(settings.x402_pay_to, settings.x402_price_usdc, settings.x402_network,
                        settings.x402_asset, settings.public_url, logger)
    if settings.payment_gate == "nowpayments":
        if not settings.nowpayments_api_key:
            raise ValueError("QH_PAYMENT_GATE=nowpayments requires NOWPAYMENTS_API_KEY")
        return NowPaymentsGate(settings.nowpayments_api_key, settings.nowpayments_price_usdc, logger)
    raise ValueError(f"unknown payment gate {settings.payment_gate!r}")
