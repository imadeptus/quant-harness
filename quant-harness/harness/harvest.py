"""Cash-and-carry funding harvest (спека 0008): доходность по данным + стресс.

НЕ предсказание — дельта-нейтральная позиция собирает funding. Допущение:
спот трекает перп идеально → ценовые ноги сокращаются → дневной P&L = funding −
издержки. Остаточный базис-риск НЕ в данных → отдельные стресс-функции.

ГЛАВНОЕ: гладкий Sharpe этой кривой ИЛЛЮЗОРНО высок; итог смотреть как
доходность ПРОТИВ стресса (basis_stress / funding_flip_stress).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def harvest_returns(funding: pd.DataFrame, member: pd.DataFrame, theta: float,
                    trail: int = 7, taker: float = 0.0005, slip: float = 0.0002,
                    periods_per_year: int = 365, basis: pd.DataFrame = None) -> dict:
    """Дневной net-return сбора funding: держим монету, если трейлинг-среднее
    funding (по t−1) > θ; равновзвешенно; издержки на вход/выход ОБЕИХ ног.

    basis — опц. DataFrame per-coin дневного базис-return (spot_ret − perp_ret).
    Если задан, включается в P&L (реальный базис-драг вместо допущения «спот=перп»).
    Возвращает dict с returns (Series), turnover, cost, и агрегатами."""
    fund = funding.where(member.reindex_like(funding).fillna(False))
    trailing = fund.rolling(trail, min_periods=max(2, trail // 2)).mean()
    # решение по t−1 (никакого look-ahead на сегодняшний funding); bool-массивы,
    # чтобы ~ было логическим отрицанием, а не побитовой инверсией int.
    hold = (trailing.shift(1) > theta).fillna(False).to_numpy(dtype=bool)
    prev = np.vstack([np.zeros((1, hold.shape[1]), dtype=bool), hold[:-1]])
    n_hold = hold.sum(axis=1)

    fund_vals = funding.fillna(0.0).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        recv = np.where(n_hold > 0, (hold * fund_vals).sum(axis=1) / n_hold, 0.0)
        entered = (hold & ~prev).sum(axis=1)
        exited = (~hold & prev).sum(axis=1)
        turnover = np.where(n_hold > 0, np.minimum((entered + exited) / n_hold, 1.0), 0.0)
    # реальный базис-P&L: средний по удерживаемым базис-return (позиция t = hold t)
    if basis is not None:
        b_vals = basis.reindex(index=funding.index, columns=funding.columns).fillna(0.0).to_numpy()
        with np.errstate(invalid="ignore"):
            basis_pnl = np.where(n_hold > 0, (hold * b_vals).sum(axis=1) / n_hold, 0.0)
        recv = recv + basis_pnl

    recv = pd.Series(recv, index=funding.index)
    turnover = pd.Series(turnover, index=funding.index)
    n_hold = pd.Series(n_hold, index=funding.index)
    # издержки: оборот × 2 ноги (спот+перп) × (taker+slip) на сторону
    cost = turnover * 2.0 * (taker + slip)

    returns = (recv - cost).fillna(0.0)
    ann_yield = float(returns.mean() * periods_per_year)
    vol = float(returns.std(ddof=1))
    sharpe = float(returns.mean() / vol * np.sqrt(periods_per_year)) if vol > 0 else 0.0
    equity = (1.0 + returns).cumprod()
    dd = float((equity / equity.cummax() - 1.0).min())

    return {
        "returns": returns, "turnover": turnover, "cost": cost,
        "annualized_yield": ann_yield, "sharpe_illusory": sharpe,
        "max_drawdown_from_data": dd,
        "avg_positions": float(n_hold.mean()),
        "avg_daily_return": float(returns.mean()),
        "periods_per_year": periods_per_year,
    }


def basis_stress(res: dict, gap: float) -> dict:
    """Разовый неблагоприятный разъезд базиса `gap` на дельта-нейтральном
    портфеле = убыток ~gap на задействованный ноционал (ценовые ноги перестают
    сокращаться на величину gap). Сколько дней доходности это стирает."""
    daily = res["avg_daily_return"]
    days_lost = float(gap / daily) if daily > 0 else float("inf")
    return {"gap": gap, "stress_loss": gap, "days_of_yield_lost": days_lost,
            "months_of_yield_lost": days_lost / 30.0 if np.isfinite(days_lost) else float("inf")}


def funding_flip_stress(res: dict, flip_rate: float, days: int) -> dict:
    """Сценарий: funding уходит в flip_rate (<0) на `days` дней — платим вместо
    сбора. Итоговый убыток за период."""
    loss = float(flip_rate * days)
    return {"flip_rate": flip_rate, "days": days, "flip_loss": loss}
