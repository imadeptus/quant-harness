"""Compatibility shim — payment gates moved to ``harness.service.payments`` in 0.3.1."""
from __future__ import annotations

from harness.service.payments import (README_PAYMENTS, STUB_ASSUMPTION, STUB_HEADER, STUB_HEADER_VALUE,
                                      STUB_NOTE, NoopGate, NowPaymentsGate, PaymentGate, X402Gate,
                                      build_gate, usdc_atomic_units)

__all__ = ["README_PAYMENTS", "STUB_ASSUMPTION", "STUB_HEADER", "STUB_HEADER_VALUE", "STUB_NOTE",
           "NoopGate", "NowPaymentsGate", "PaymentGate", "X402Gate", "build_gate", "usdc_atomic_units"]
