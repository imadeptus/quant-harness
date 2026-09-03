"""Cross-sectional семейство (спека 0001): point-in-time universe, ранговые
веса и портфельный движок. Судья (runner/DSR/CPCV/PBO) не здесь — он общий.

Дисциплина та же, что в backtest.py: веса бара t считаются из данных до t
включительно, а ДЕРЖИТСЯ позиция с бара t+1 (shift внутри run_xs_backtest).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import Costs


def monthly_universe(qv: pd.DataFrame, top_k: int, vol_window: int,
                     min_history: int, skip_top: int = 0,
                     min_median_qv: float | None = None) -> pd.DataFrame:
    """Point-in-time членство: на первом баре каждого месяца — топ-K по
    медианному quote_volume за прошлые vol_window баров, строго до бара
    ребаланса. Свежие листинги (первый валидный бар внутри панели) должны
    накопить min_history баров; колонки, валидные с самого начала панели,
    считаются «давно листнутыми» — их реальная дата листинга раньше наших данных.
    """
    member = pd.DataFrame(False, index=qv.index, columns=qv.columns)
    # Календарные месяцы по UTC; naive-конверсия убирает warning про tz-drop.
    idx = qv.index.tz_convert("UTC").tz_localize(None) if qv.index.tz is not None else qv.index
    months = idx.to_period("M")
    notna = qv.notna()
    first_valid = {c: (int(notna[c].values.argmax()) if notna[c].any() else len(qv))
                   for c in qv.columns}
    for month in months.unique():
        pos = int((months == month).argmax())
        start = pos - vol_window
        if start < 0:
            continue  # до первого полного окна объёмов членства нет
        window = qv.iloc[start:pos]
        eligible = []
        for c in qv.columns:
            fv = first_valid[c]
            if fv > 0 and pos - fv < min_history:
                continue  # свежий листинг, истории мало
            med = window[c].median()
            if not np.isfinite(med):
                continue
            if min_median_qv is not None and med < min_median_qv:
                continue  # пол ликвидности: ниже — slippage-модель фикция
            eligible.append((float(med), c))
        ranked = sorted(eligible, reverse=True)
        # band: skip_top самых ликвидных пропускаем (спека 0002 — хвост)
        top = [c for _, c in ranked[skip_top:skip_top + top_k]]
        member.loc[(months == month), top] = True
    return member


def _quantile_count(n_members: int, quantile: float) -> int:
    return max(1, int(np.floor(n_members * quantile)))


def _ranked_weights(signal: pd.DataFrame, member: pd.DataFrame, quantile: float,
                    rebalance: int, invert: bool) -> pd.DataFrame:
    """Общий ранговый конструктор: +0.5 на верхний квантиль сигнала, −0.5 на
    нижний (invert=True меняет стороны — для carry), равновзвешенно внутри
    стороны, пересчёт раз в rebalance баров, между ребалансами веса заморожены."""
    sig = signal.values
    mem = member.reindex_like(signal).fillna(False).values
    W = np.zeros_like(sig, dtype=float)
    last = np.zeros(sig.shape[1])
    for t in range(sig.shape[0]):
        if t % rebalance == 0:
            row = sig[t].copy()
            row[~mem[t]] = np.nan
            valid = np.isfinite(row)
            n_valid = int(valid.sum())
            wrow = np.zeros(sig.shape[1])
            if n_valid >= 2:
                k = _quantile_count(n_valid, quantile)
                order = np.argsort(row)[:n_valid]  # валидные по возрастанию
                lo, hi = order[:k], order[-k:]
                long_side, short_side = (lo, hi) if invert else (hi, lo)
                wrow[long_side] = +0.5 / k
                wrow[short_side] = -0.5 / k
            last = wrow
        W[t] = last
    return pd.DataFrame(W, index=signal.index, columns=signal.columns)


def xs_momentum_weights(closes: pd.DataFrame, member: pd.DataFrame, lookback: int,
                        skip: int, quantile: float, rebalance: int) -> pd.DataFrame:
    """H1: лонг обгонявших корзину, шорт отстававших (прошлая относительная
    доходность за lookback баров, минус последние skip баров против реверса)."""
    past = closes.shift(skip) / closes.shift(skip + lookback) - 1.0
    return _ranked_weights(past, member, quantile, rebalance, invert=False)


def xs_carry_weights(funding: pd.DataFrame, member: pd.DataFrame, window: int,
                     quantile: float, rebalance: int) -> pd.DataFrame:
    """H2: лонг дешёвого funding (получаем), шорт дорогого (получаем спред)."""
    trailing = funding.rolling(window).mean()
    return _ranked_weights(trailing, member, quantile, rebalance, invert=True)


@dataclass
class XSBacktestResult:
    returns: pd.Series      # per-bar net portfolio returns
    n_trades: int           # число изменений позиции по ногам
    turnover: pd.Series     # Σ|Δw_i| на бар
    trade_events: pd.Series  # число изменений позиций (ног) на бар — для судьи


def run_xs_backtest(closes: pd.DataFrame, funding: pd.DataFrame,
                    weights: pd.DataFrame, costs: Costs) -> XSBacktestResult:
    """Портфель: r_p = Σ held_i · r_i, издержки на Σ|Δheld_i|, funding по ногам.
    held = weights.shift(1) — анти-look-ahead как в run_backtest."""
    held = weights.shift(1).fillna(0.0)
    # Пропуски цен (делистинг/дырка) -> нулевая доходность бара, не NaN-дыра.
    rets = closes.pct_change().fillna(0.0)
    gross = (held * rets).sum(axis=1)
    dw = held.diff()
    dw.iloc[0] = held.iloc[0]
    turnover = dw.abs().sum(axis=1)
    cost = turnover * (costs.taker_fee + costs.slippage)
    if costs.apply_funding:
        fpnl = -(held * funding.reindex_like(closes).fillna(0.0)).sum(axis=1)
    else:
        fpnl = pd.Series(0.0, index=closes.index)
    net = gross - cost + fpnl
    events = (dw.abs() > 1e-9).sum(axis=1)
    return XSBacktestResult(returns=net, n_trades=int(events.sum()),
                            turnover=turnover, trade_events=events)
