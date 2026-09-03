#!/usr/bin/env python3
"""quant-harness quickstart — runs end to end with NO data download.

    python examples/quickstart.py

Two demos, both on synthetic data with known ground truth:

  A. The verdict.  Feed the judge a matrix of pure-noise "strategies" and a matrix
     with a genuine edge; watch it KILL the first and PASS the second. This is the
     core of the harness — a mechanical PASS/KILL you cannot talk your way past.

  B. The engine.   Run one synthetic instrument through the leakage-safe backtest
     (signal on bar t is executed at t+1), with realistic costs, and read the
     honest metrics: gross vs net Sharpe, turnover, max drawdown.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from harness import (Costs, CPCVConfig, Thresholds, annualization_factor,
                     build_signal, deflated_sharpe_ratio, max_drawdown,
                     run_backtest, run_cpcv_returns)

BARS = 912
IDX = pd.date_range("2024-01-01", periods=BARS, freq="1D", tz="UTC")
ANN = annualization_factor(IDX)  # ~365 for daily bars


def _matrix(mu: float, seed: int, n_configs: int = 6, vol: float = 0.01) -> np.ndarray:
    """n_configs independent per-bar return series with per-bar mean `mu`."""
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(mu, vol, BARS) for _ in range(n_configs)])


def _judge(R: np.ndarray) -> dict:
    trades = np.zeros_like(R)
    trades[:, ::3] = 1  # a trade every 3 bars -> clears the min-trades gate
    grid = [{"config": i} for i in range(R.shape[0])]
    return run_cpcv_returns(R, trades, IDX, grid,
                            CPCVConfig(n_groups=10, k_test=2, purge=1, embargo=5),
                            Thresholds())


def demo_a_the_verdict() -> None:
    print("=" * 68)
    print("DEMO A — the PASS/KILL verdict (6 configs, CPCV, Deflated Sharpe)")
    print("=" * 68)

    noise = _judge(_matrix(mu=0.0, seed=1))
    # A true edge of annualised Sharpe ~3: mu = 3/sqrt(ANN) * vol.
    edge_mu = 3.0 / np.sqrt(ANN) * 0.01
    edge = _judge(_matrix(mu=edge_mu, seed=2))

    for name, rep in [("pure noise ", noise), ("real edge  ", edge)]:
        print(f"  {name}: verdict={rep['verdict']:<4}  "
              f"OOS Sharpe(ann)={rep['oos_sharpe_annualized']:+.2f}  "
              f"DSR={rep['deflated_sharpe_ratio']:.3f}  "
              f"maxDD={rep['oos_max_drawdown']:.2f}")
    assert noise["verdict"] == "KILL" and edge["verdict"] == "PASS"
    print("  -> noise is KILLed, the genuine edge PASSes. As it must.\n")


def demo_b_the_engine() -> None:
    print("=" * 68)
    print("DEMO B — the leakage-safe engine on one synthetic instrument")
    print("=" * 68)

    # A price with persistent trends, so time-series momentum has something to
    # find. Drift follows a slow AR(1) so regimes last (illustrative, not a claim).
    rng = np.random.default_rng(7)
    drift = np.zeros(BARS)
    for t in range(1, BARS):
        drift[t] = 0.97 * drift[t - 1] + rng.normal(0.0, 0.0008)
    logret = drift + rng.normal(0.0, 0.01, BARS)
    close = 100.0 * np.exp(np.cumsum(logret))
    df = pd.DataFrame({"close": close, "funding": 0.0}, index=IDX)

    signal = build_signal(df, {"family": "momentum", "lookback": 24,
                               "threshold": 0.0, "allow_short": True})
    costs = Costs(taker_fee=0.0005, slippage=0.0002, apply_funding=True)
    res = run_backtest(df, signal, costs)

    gross_ann = res.gross_sharpe_periodic * np.sqrt(ANN)
    net_ann = res.net_sharpe_periodic * np.sqrt(ANN)
    print(f"  momentum(lookback=24) on a trending synthetic price:")
    print(f"    trades           : {res.n_trades}")
    print(f"    gross Sharpe(ann): {gross_ann:+.2f}")
    print(f"    net   Sharpe(ann): {net_ann:+.2f}   (after taker+slippage)")
    print(f"    max drawdown     : {max_drawdown(res.returns):.2%}")
    print(f"    single-path DSR  : {deflated_sharpe_ratio(res.returns.values, 1, 1/BARS):.3f}")
    print("  -> costs move gross->net; on ONE in-sample path this is NOT a verdict.")
    print("     A real verdict needs the walk-forward judge in Demo A.\n")


if __name__ == "__main__":
    demo_a_the_verdict()
    demo_b_the_engine()
    print("Done. See reports/CALIBRATION.md for how the judge itself was validated.")
