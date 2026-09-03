"""Backtest engine — pandas, cost-aware, leakage-safe by construction.

Discipline enforced here (so families cannot cheat):
- Signal computed on bar t is executed at t+1 (shift by 1). No signal-on-close /
  trade-on-close look-ahead.
- Costs charged on every position CHANGE: taker fee + slippage per unit turnover.
- Funding charged/credited on the held position at each funding timestamp.
Returns a per-bar strategy return series and a trade count.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Costs:
    taker_fee: float = 0.0005     # per side, fraction (5 bps)
    slippage: float = 0.0002      # per side, fraction (2 bps)
    apply_funding: bool = True


@dataclass
class BacktestResult:
    returns: pd.Series            # per-bar net strategy returns
    n_trades: int
    gross_sharpe_periodic: float
    net_sharpe_periodic: float


def run_backtest(df: pd.DataFrame, target_pos: pd.Series, costs: Costs) -> BacktestResult:
    """df needs columns: close, funding. target_pos in {-1,0,+1} on df.index.

    The position actually HELD during bar t is the signal from t-1 (one-bar
    delay). Market return of bar t is close_t/close_{t-1} - 1, earned by the
    position held during t.
    """
    close = df["close"].astype(float)
    mkt_ret = close.pct_change().fillna(0.0)

    held = target_pos.shift(1).fillna(0.0)          # <-- the anti-look-ahead line
    gross = held * mkt_ret

    # Turnover = |change in position| between consecutive bars; cost per unit.
    turnover = held.diff().abs().fillna(held.abs())
    per_side_cost = costs.taker_fee + costs.slippage
    cost = turnover * per_side_cost

    # Funding: a long pays positive funding, a short receives it. Charged on the
    # position held at each funding timestamp (nonzero funding rows).
    funding_pnl = pd.Series(0.0, index=df.index)
    if costs.apply_funding and "funding" in df.columns:
        funding_pnl = -held * df["funding"].fillna(0.0)

    net = gross - cost + funding_pnl
    n_trades = int((turnover > 1e-9).sum())

    def _sharpe(x: pd.Series) -> float:
        s = x.std(ddof=1)
        return float(x.mean() / s) if s > 0 else 0.0

    return BacktestResult(
        returns=net,
        n_trades=n_trades,
        gross_sharpe_periodic=_sharpe(gross),
        net_sharpe_periodic=_sharpe(net),
    )


def max_drawdown(returns: pd.Series) -> float:
    """Max drawdown of the compounded equity curve, as a fraction in [-1, 0].

    A drawdown can never be worse than total ruin (-100%): if any per-bar net
    return is <= -100% the account is wiped, so we saturate at -1.0 rather than
    letting (1+r).cumprod() flip sign and report a nonsense magnitude (the old
    code returned ~-6.4e10 on heavy synthetic edges). The curve is accumulated
    in log space so long, strongly-compounding series cannot overflow float64.
    On sane returns (equity stays > 0) this equals the naive equity/peak - 1.
    """
    r = np.asarray(returns.fillna(0.0), dtype=float)
    if r.size == 0:
        return 0.0
    growth = 1.0 + r
    if np.any(growth <= 0.0):
        return -1.0                              # ruin: lost everything (or more)
    log_equity = np.cumsum(np.log(growth))
    running_peak = np.maximum.accumulate(log_equity)
    dd = np.expm1(log_equity - running_peak)     # in (-1, 0] by construction
    return float(dd.min())


def annualization_factor(index: pd.DatetimeIndex) -> float:
    """Periods per year, inferred from the median bar spacing."""
    if len(index) < 3:
        return 1.0
    # Use total_seconds() on the diffs — robust to tz-aware indices, unlike
    # .view('int64') whose unit is version-dependent.
    deltas = index.to_series().diff().dropna().dt.total_seconds().to_numpy()
    dt = float(np.median(deltas))  # seconds per bar
    if dt <= 0:
        return 1.0
    return (365.0 * 24 * 3600) / dt
