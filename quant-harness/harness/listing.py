"""Пост-листинговый event-study движок (спека 0004).

Сигнал — возраст контракта (дни с первого бара), а не его цена. Корзина молодых
∈ [e_start, h] дней против хеджа из зрелых (>mature_age), дельта-нейтрально.
Веса бара t держатся с t+1 (анти-look-ahead в run_listing_backtest).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .xs import XSBacktestResult


def age_in_days(closes: pd.DataFrame) -> pd.DataFrame:
    """Возраст каждого символа в днях = (t − первый валидный бар). До листинга NaN.
    Календарные сутки берём из индекса (сетка 1d)."""
    age = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    for c in closes.columns:
        valid = closes[c].notna().values
        if not valid.any():
            continue
        first = int(valid.argmax())
        deltas = (closes.index - closes.index[first]).days.to_numpy()
        col = np.full(len(closes), np.nan)
        col[first:] = deltas[first:]
        age[c] = col
    return age


def young_basket_weights(closes: pd.DataFrame, qv: pd.DataFrame, e_start: int,
                         h: int, direction: int, mature_age: int = 180,
                         min_median_qv: float = 0.0, qv_window: int = 30) -> pd.DataFrame:
    """direction=+1: лонг молодых (возраст ∈ [e_start,h]) гросс +1, шорт зрелых
    (>mature_age) гросс −1. direction=−1 меняет стороны. Пол ликвидности по
    трейлинг-медиане qv. Если молодых нет — обе ноги пусты (нейтральность)."""
    age = age_in_days(closes).values
    med_qv = qv.rolling(qv_window, min_periods=3).median().values
    liq_ok = med_qv >= min_median_qv
    W = np.zeros_like(age, dtype=float)
    for t in range(age.shape[0]):
        a = age[t]
        young = (a >= e_start) & (a <= h) & liq_ok[t]
        mature = (a > mature_age) & liq_ok[t]
        ny, nm = int(np.nansum(young)), int(np.nansum(mature))
        if ny == 0 or nm == 0:
            continue  # нейтральность требует обе ноги
        W[t, np.nan_to_num(young).astype(bool)] = direction * 1.0 / ny
        W[t, np.nan_to_num(mature).astype(bool)] = -direction * 1.0 / nm
    return pd.DataFrame(W, index=closes.index, columns=closes.columns)


def young_vs_btc_weights(closes: pd.DataFrame, qv: pd.DataFrame, e_start: int,
                         h: int, direction: int, hedge: str = "BTCUSDT",
                         rebalance: int = 7, mature_age: int = 180,
                         min_median_qv: float = 0.0, qv_window: int = 30) -> pd.DataFrame:
    """Спека 0005: корзина молодых ∈ [e_start,h] против ОДНОГО хедж-инструмента
    (шорт `hedge` равным ноционалом). Ребаланс раз в `rebalance` баров; между —
    веса заморожены. Хедж держит постоянный вес −direction, пока корзина непуста,
    → оборот хеджа ≈ 0 (лечит патологию 0004)."""
    if hedge not in closes.columns:
        raise ValueError(f"hedge instrument {hedge} not in panel")
    age = age_in_days(closes).values
    med_qv = qv.rolling(qv_window, min_periods=3).median().values
    liq_ok = med_qv >= min_median_qv
    hedge_i = closes.columns.get_loc(hedge)
    W = np.zeros_like(age, dtype=float)
    last = np.zeros(age.shape[1])
    for t in range(age.shape[0]):
        if t % rebalance == 0:
            a = age[t]
            young = (a >= e_start) & (a <= h) & liq_ok[t]
            young[hedge_i] = False  # хедж-инструмент не может быть «молодым»
            ny = int(np.nansum(young))
            wrow = np.zeros(age.shape[1])
            if ny > 0:
                wrow[np.nan_to_num(young).astype(bool)] = direction * 1.0 / ny
                wrow[hedge_i] = -direction * 1.0
            last = wrow
        W[t] = last
    return pd.DataFrame(W, index=closes.index, columns=closes.columns)


def run_listing_backtest(closes: pd.DataFrame, funding: pd.DataFrame,
                         weights: pd.DataFrame, young: pd.DataFrame,
                         taker: float, slip_young: float,
                         slip_mature: float) -> XSBacktestResult:
    """Портфель с ПО-НОГОВЫМИ издержками: молодая нога дороже (slip_young),
    зрелый хедж дешевле (slip_mature). held = weights.shift(1)."""
    held = weights.shift(1).fillna(0.0)
    rets = closes.pct_change().fillna(0.0)
    gross = (held * rets).sum(axis=1)
    dw = held.diff()
    dw.iloc[0] = held.iloc[0]
    # per-column издержка: taker + (молодая нога -> slip_young, иначе slip_mature)
    slip = np.where(young.shift(1).fillna(False).values, slip_young, slip_mature)
    per_col_cost = taker + slip
    cost = (dw.abs().values * per_col_cost).sum(axis=1)
    fpnl = -(held * funding.reindex_like(closes).fillna(0.0)).sum(axis=1)
    net = gross - pd.Series(cost, index=closes.index) + fpnl
    events = (dw.abs() > 1e-9).sum(axis=1)
    turnover = dw.abs().sum(axis=1)
    return XSBacktestResult(returns=net, n_trades=int(events.sum()),
                            turnover=turnover, trade_events=events)
