"""Межбиржевые funding-пары Binance↔Bybit (спека 0003).

Пара по символу: w > 0 = лонг Binance / шорт Bybit, w < 0 — наоборот.
Ценовые ноги одного underlying почти гасятся; PnL = венью-базис + funding-спред
− издержки четырёх ног. Веса бара t держатся с бара t+1 (анти-look-ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import Costs
from .xs import XSBacktestResult


def basis_weights(spread: pd.DataFrame, member: pd.DataFrame, window: int,
                  theta: float, rebalance: int) -> pd.DataFrame:
    """w_i = −sign(скользящий средний спред) / n_active при |спред| > theta.

    Знак минус: спред = funding_binance − funding_bybit; положительный спред
    собираем шортом дорогой (Binance) ноги. Равновзвешенно по активным парам,
    гросс 1.0; между ребалансами веса заморожены."""
    m = spread.rolling(window).mean().values
    mem = member.reindex_like(spread).fillna(False).values
    W = np.zeros_like(m, dtype=float)
    last = np.zeros(m.shape[1])
    for t in range(m.shape[0]):
        if t % rebalance == 0:
            row = m[t]
            active = mem[t] & np.isfinite(row) & (np.abs(row) > theta)
            n = int(active.sum())
            wrow = np.zeros(m.shape[1])
            if n > 0:
                wrow[active] = -np.sign(row[active]) / n
            last = wrow
        W[t] = last
    return pd.DataFrame(W, index=spread.index, columns=spread.columns)


def run_basis_backtest(ret_bin: pd.DataFrame, ret_byb: pd.DataFrame,
                       fund_bin: pd.DataFrame, fund_byb: pd.DataFrame,
                       weights: pd.DataFrame, costs: Costs) -> XSBacktestResult:
    held = weights.shift(1).fillna(0.0)
    price = (held * (ret_bin.fillna(0.0) - ret_byb.fillna(0.0))).sum(axis=1)
    spread = fund_bin.fillna(0.0) - fund_byb.fillna(0.0)
    funding = (-(held * spread)).sum(axis=1)  # лонг-нога платит свой funding
    dw = held.diff()
    dw.iloc[0] = held.iloc[0]
    turnover = dw.abs().sum(axis=1)
    # каждая единица веса пары исполняется на ОБЕИХ биржах
    cost = turnover * (costs.taker_fee + costs.slippage) * 2
    net = price + funding - cost
    events = (dw.abs() > 1e-9).sum(axis=1) * 2  # 2 венью-ноги на пару
    return XSBacktestResult(returns=net, n_trades=int(events.sum()),
                            turnover=turnover, trade_events=events)
