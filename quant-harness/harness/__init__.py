"""quant-harness — an honest backtest harness for trading strategies.

The point of this package is not to *find* a profitable strategy but to make it
hard to *fool yourself* into thinking you have. It enforces the guardrails that
separate a real edge from a data-mined artefact:

- **Leakage-safe by construction** — the position held during bar t is the signal
  from bar t-1 (`run_backtest`); families cannot peek at the future.
- **Honest out-of-sample** — rolling/anchored walk-forward and Combinatorial
  Purged CV with purge+embargo; the only equity curve you may believe is the
  concatenation of untouched test windows.
- **Realistic costs** — taker fee + slippage on every turnover, funding as a
  periodic cash flow.
- **Multiple-testing correction** — the Deflated Sharpe Ratio deflates the winning
  Sharpe by the number of configs tried; PBO measures whether the *search itself*
  overfits.
- **A mechanical verdict** — PASS/KILL against pre-registered `Thresholds`, so a
  result cannot be talked up after the fact.

The detector behind that verdict is itself calibrated and measured — see
`harness.calibration` and `reports/CALIBRATION.md`.

The same judge is available for *someone else's* returns as `audit_returns` /
the `qh-audit` CLI (CSV in, PASS/KILL + Markdown/JSON report out) — see `harness.audit`.

Quick start (no data download needed)::

    import numpy as np, pandas as pd
    from harness import run_cpcv_returns, Thresholds, CPCVConfig

    # 6 configs x 900 bars of pure noise -> the judge must KILL it
    R = np.random.default_rng(0).normal(0.0, 0.01, (6, 900))
    T = np.zeros_like(R); T[:, ::3] = 1               # a trade every 3 bars
    idx = pd.date_range("2024-01-01", periods=900, freq="1D", tz="UTC")
    rep = run_cpcv_returns(R, T, idx, [{"c": i} for i in range(6)],
                           CPCVConfig(n_groups=10, k_test=2), Thresholds())
    print(rep["verdict"])   # -> "KILL"
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .backtest import (BacktestResult, Costs, annualization_factor,
                       max_drawdown, run_backtest)
from .deflated_sharpe import (deflated_sharpe_ratio, expected_max_sharpe,
                              probabilistic_sharpe_ratio)
from .families import (all_families, build_signal, breakout_grid, carry_grid,
                       grid_for, mean_reversion_grid, momentum_grid)
from .netutil import NetworkError, get_json, post_json
from .pbo import pbo_cscv
from .runner import Thresholds, run, run_cpcv, run_cpcv_returns
from .walk_forward import (CPCVConfig, WalkForwardConfig, assemble_oos, n_folds,
                           walk_forward_folds)

if TYPE_CHECKING:  # typed names for mypy; at runtime they are resolved lazily below
    from .audit import AuditInputError, audit_returns, render_markdown

__version__ = "0.3.0"   # keep equal to [project].version in pyproject.toml (tests/test_version.py)

_AUDIT_EXPORTS = ("audit_returns", "render_markdown", "AuditInputError")


def __getattr__(name: str) -> Any:
    """Lazy `harness.audit` exports (PEP 562).

    `harness.audit` is also a runnable module (`python -m harness.audit`); importing
    it eagerly here would make runpy re-execute an already-imported module and emit
    a RuntimeWarning. Resolving the names on first access avoids that while keeping
    `from harness import audit_returns` working.
    """
    if name in _AUDIT_EXPORTS:
        from . import audit
        return getattr(audit, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    # judge / verdict
    "run", "run_cpcv", "run_cpcv_returns", "Thresholds",
    # strategy audit (the `qh-audit` CLI and the hosted API sit on top of this)
    "audit_returns", "render_markdown", "AuditInputError",
    # resilient HTTP client (retries, backoff, host rotation) used by the loaders
    "get_json", "post_json", "NetworkError",
    # backtest engine
    "run_backtest", "Costs", "BacktestResult", "max_drawdown", "annualization_factor",
    # splitting
    "WalkForwardConfig", "CPCVConfig", "walk_forward_folds", "assemble_oos", "n_folds",
    # statistics
    "deflated_sharpe_ratio", "probabilistic_sharpe_ratio", "expected_max_sharpe",
    "pbo_cscv",
    # signal families
    "build_signal", "grid_for", "all_families", "momentum_grid",
    "mean_reversion_grid", "breakout_grid", "carry_grid",
]
