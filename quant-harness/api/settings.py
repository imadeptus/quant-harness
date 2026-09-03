"""Runtime configuration — everything that differs between deployments comes
from environment variables (12-factor III).

Construct ``Settings(...)`` directly in tests; use ``Settings.from_env()`` in the
real process. Parsing fails fast with a clear ``ValueError`` on a malformed value.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

PAYMENT_GATES: Tuple[str, ...] = ("noop", "x402", "nowpayments")


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _env_float(name: str) -> Optional[float]:
    raw = _env_str(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str) -> bool:
    raw = (_env_str(name) or "").lower()
    return raw in {"1", "true", "yes", "on"}


def _env_csv(name: str) -> Tuple[str, ...]:
    raw = os.environ.get(name, "")
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    """All knobs of the API process. Field -> env var mapping is in ``from_env``."""

    # request limits
    max_configs: int = 200                      # QH_MAX_CONFIGS
    max_periods: int = 50_000                   # QH_MAX_PERIODS
    max_body_bytes: int = 64 * 1024 * 1024      # QH_MAX_BODY_BYTES
    # auth (empty tuple = open)
    api_keys: Tuple[str, ...] = ()              # QH_API_KEYS (comma-separated)
    # payments (stubs — see api/README.md)
    payment_gate: str = "noop"                  # QH_PAYMENT_GATE = noop|x402|nowpayments
    x402_pay_to: Optional[str] = None           # QH_X402_PAY_TO
    x402_price_usdc: Optional[float] = None     # QH_X402_PRICE_USDC
    x402_network: str = "base-sepolia"          # QH_X402_NETWORK
    x402_asset: str = "USDC"                    # QH_X402_ASSET (token contract on mainnet)
    nowpayments_api_key: Optional[str] = None   # NOWPAYMENTS_API_KEY
    nowpayments_price_usdc: Optional[float] = None  # QH_NOWPAYMENTS_PRICE_USDC
    # The paid gates are STUBS (challenge only, no verification). They refuse to
    # start unless the operator opts in explicitly, so a stub can never be mistaken
    # for a working paywall by accident.
    allow_stub_payment_gate: bool = False       # QH_ALLOW_STUB_PAYMENT_GATE=1
    # misc
    public_url: str = "http://localhost:8000"   # QH_PUBLIC_URL (x402 `resource`)
    log_level: str = "INFO"                     # QH_LOG_LEVEL

    def __post_init__(self) -> None:
        if self.payment_gate not in PAYMENT_GATES:
            raise ValueError(
                f"QH_PAYMENT_GATE must be one of {PAYMENT_GATES}, got {self.payment_gate!r}")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            max_configs=_env_int("QH_MAX_CONFIGS", cls.max_configs),
            max_periods=_env_int("QH_MAX_PERIODS", cls.max_periods),
            max_body_bytes=_env_int("QH_MAX_BODY_BYTES", cls.max_body_bytes),
            api_keys=_env_csv("QH_API_KEYS"),
            payment_gate=(_env_str("QH_PAYMENT_GATE", cls.payment_gate) or cls.payment_gate).lower(),
            x402_pay_to=_env_str("QH_X402_PAY_TO"),
            x402_price_usdc=_env_float("QH_X402_PRICE_USDC"),
            x402_network=_env_str("QH_X402_NETWORK", cls.x402_network) or cls.x402_network,
            x402_asset=_env_str("QH_X402_ASSET", cls.x402_asset) or cls.x402_asset,
            nowpayments_api_key=_env_str("NOWPAYMENTS_API_KEY"),
            nowpayments_price_usdc=_env_float("QH_NOWPAYMENTS_PRICE_USDC"),
            allow_stub_payment_gate=_env_bool("QH_ALLOW_STUB_PAYMENT_GATE"),
            public_url=(_env_str("QH_PUBLIC_URL", cls.public_url) or cls.public_url).rstrip("/"),
            log_level=(_env_str("QH_LOG_LEVEL", cls.log_level) or cls.log_level).upper(),
        )
