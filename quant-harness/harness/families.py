"""Strategy families: parameterized signal generators.

A "family" maps parameters -> a target-position series in {-1, 0, +1}, computed
using ONLY information available at the close of each bar. The backtest engine
enforces the one-bar execution delay; families must never look forward.

Each family carries an economic hypothesis (why an edge could exist):
- momentum      : time-series momentum persists (Liu & Tsyvinski, RFS 2021).
- mean_reversion: short-horizon overreaction reverts.
- breakout      : range breakouts continue (Donchian).
- carry         : funding on perps pays the side the crowd is short of.

`build_signal(df, params)` takes the FULL frame so funding-based families can
read the funding column; price families use only df['close'].
"""
from __future__ import annotations

from itertools import product
from typing import Dict, Iterator, List

import numpy as np
import pandas as pd


# ---------- price families ----------

def momentum_signal(close: pd.Series, lookback: int, threshold: float,
                    allow_short: bool = True) -> pd.Series:
    logret = np.log(close).diff()
    mom = logret.rolling(lookback).sum()
    pos = pd.Series(0.0, index=close.index)
    pos[mom > threshold] = 1.0
    if allow_short:
        pos[mom < -threshold] = -1.0
    return pos.fillna(0.0)


def mean_reversion_signal(close: pd.Series, lookback: int, entry_z: float,
                          allow_short: bool = True) -> pd.Series:
    mean = close.rolling(lookback).mean()
    std = close.rolling(lookback).std(ddof=0)
    z = (close - mean) / std.replace(0, np.nan)
    pos = pd.Series(0.0, index=close.index)
    pos[z < -entry_z] = 1.0            # price below band -> expect reversion up
    if allow_short:
        pos[z > entry_z] = -1.0
    return pos.fillna(0.0)


def breakout_signal(close: pd.Series, lookback: int, allow_short: bool = True) -> pd.Series:
    # Donchian on PRIOR window (shifted) so the current bar is excluded -> no peek.
    upper = close.rolling(lookback).max().shift(1)
    lower = close.rolling(lookback).min().shift(1)
    pos = pd.Series(0.0, index=close.index)
    pos[close > upper] = 1.0
    if allow_short:
        pos[close < lower] = -1.0
    return pos.fillna(0.0)


# ---------- funding family ----------

def carry_signal(df: pd.DataFrame, threshold: float, hold: int = 1) -> pd.Series:
    """Funding carry: take the side that RECEIVES funding, using LAST-KNOWN rate.

    If funding > 0 longs pay shorts -> be short to receive it; if < 0 -> be long.
    Uses funding shifted by 1 (last realized) to avoid using the rate that is only
    known at the funding instant. Not delta-hedged, so it still carries price risk
    (documented caveat) — a first-order proxy for the carry factor, not the full
    market-neutral trade.
    """
    if "funding" not in df.columns:
        return pd.Series(0.0, index=df.index)
    f = df["funding"].shift(1).fillna(0.0)
    # forward-fill the last known funding between 8h stamps so the position holds
    f_known = f.replace(0.0, np.nan).ffill().fillna(0.0)
    pos = pd.Series(0.0, index=df.index)
    pos[f_known > threshold] = -1.0
    pos[f_known < -threshold] = 1.0
    if hold > 1:
        pos = pos.rolling(hold, min_periods=1).mean().apply(np.sign)
    return pos.fillna(0.0)


# ---------- grids (each combo = one trial; total N feeds the Deflated Sharpe) ----------

def momentum_grid(lookbacks=(6, 12, 24, 48, 96, 168, 336),
                  thresholds=(0.0, 0.0025, 0.005, 0.01, 0.02, 0.04),
                  allow_short=(True, False)) -> Iterator[Dict]:
    for lb, th, sh in product(lookbacks, thresholds, allow_short):
        yield {"family": "momentum", "lookback": lb, "threshold": th, "allow_short": sh}


def mean_reversion_grid(lookbacks=(12, 24, 48, 96, 168),
                        entry_z=(1.0, 1.5, 2.0, 2.5, 3.0),
                        allow_short=(True, False)) -> Iterator[Dict]:
    for lb, z, sh in product(lookbacks, entry_z, allow_short):
        yield {"family": "mean_reversion", "lookback": lb, "entry_z": z, "allow_short": sh}


def breakout_grid(lookbacks=(12, 24, 48, 96, 168, 336),
                  allow_short=(True, False)) -> Iterator[Dict]:
    for lb, sh in product(lookbacks, allow_short):
        yield {"family": "breakout", "lookback": lb, "allow_short": sh}


def carry_grid(thresholds=(0.0, 0.00005, 0.0001, 0.0002, 0.0005),
               hold=(1, 3, 6)) -> Iterator[Dict]:
    for th, h in product(thresholds, hold):
        yield {"family": "carry", "threshold": th, "hold": h}


FAMILY_GRIDS = {
    "momentum": momentum_grid,
    "mean_reversion": mean_reversion_grid,
    "breakout": breakout_grid,
    "carry": carry_grid,
}


def build_signal(df: pd.DataFrame, params: Dict) -> pd.Series:
    fam = params["family"]
    close = df["close"]
    if fam == "momentum":
        return momentum_signal(close, params["lookback"], params["threshold"],
                               params.get("allow_short", True))
    if fam == "mean_reversion":
        return mean_reversion_signal(close, params["lookback"], params["entry_z"],
                                     params.get("allow_short", True))
    if fam == "breakout":
        return breakout_signal(close, params["lookback"], params.get("allow_short", True))
    if fam == "carry":
        return carry_signal(df, params["threshold"], params.get("hold", 1))
    raise ValueError(f"unknown family: {fam}")


def grid_for(family: str) -> List[Dict]:
    return list(FAMILY_GRIDS[family]())


def all_families() -> List[str]:
    return list(FAMILY_GRIDS.keys())
