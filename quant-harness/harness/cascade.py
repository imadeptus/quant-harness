"""Каскад-реверсал прокси (спека 0007): fade объёмо-подтверждённого шок-бара.

Ликвидационный каскад наблюдаем в цене как резкое однобарное движение со
всплеском объёма. Сигнал бара t: если бар t дал |ret| > k·σ И объём > m·медиана,
открываем позицию ПРОТИВ движения (fade овершута) и держим hold баров.
Веса держатся с t+1 (анти-look-ahead в run_xs_backtest), поэтому доход самого
шок-бара не наш — берём последующий откуп.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cascade_reversal_weights(closes: pd.DataFrame, volume: pd.DataFrame,
                             member: pd.DataFrame, k: float, m: float, hold: int,
                             vol_window: int = 30, direction: int = +1) -> pd.DataFrame:
    """Дельта-нейтральные веса на объёмо-подтверждённый шок-бар.

    direction=+1 (fade, спека 0007): лонг залитых вниз, шорт загнанных вверх.
    direction=−1 (follow, спека 0009): по направлению шока.
    k — порог шока в сигмах ret; m — множитель всплеска объёма над медианой;
    hold — сколько баров держать. Равновзвешенно, гросс 1.0."""
    ret = closes.pct_change()
    sigma = ret.rolling(vol_window, min_periods=max(5, vol_window // 3)).std()
    med_vol = volume.rolling(vol_window, min_periods=max(5, vol_window // 3)).median()
    mem = member.reindex_like(closes).fillna(False).values

    r = ret.values
    s = sigma.values
    vspike = np.divide(volume.values, med_vol.values,
                       out=np.zeros_like(r), where=med_vol.values > 0)
    with np.errstate(invalid="ignore"):
        shock_down = (r < -k * s) & (vspike > m) & mem
        shock_up = (r > k * s) & (vspike > m) & mem
    shock_down = np.nan_to_num(shock_down).astype(bool)
    shock_up = np.nan_to_num(shock_up).astype(bool)

    raw = np.zeros_like(r)          # желаемый знак позиции на баре срабатывания
    raw[shock_down] = +1.0 * direction   # +1 fade (лонг после падения), -1 follow
    raw[shock_up] = -1.0 * direction
    # держим hold баров: для КАЖДОГО имени берём самый свежий ненулевой шок в окне
    # [t-hold+1, t] — forward-fill ненулевых с лимитом (hold-1). Векторно и без
    # потери перекрывающихся шоков разных имён (баг break-логики).
    signal = pd.DataFrame(raw, index=closes.index, columns=closes.columns)
    signal = signal.where(signal != 0)
    if hold > 1:
        signal = signal.ffill(limit=hold - 1)
    held_sign = signal.fillna(0.0).values

    W = np.zeros_like(r)
    for t in range(closes.shape[0]):
        longs = held_sign[t] > 0
        shorts = held_sign[t] < 0
        nl, ns = int(longs.sum()), int(shorts.sum())
        if nl > 0:
            W[t, longs] = 0.5 / nl if ns > 0 else 1.0 / nl
        if ns > 0:
            W[t, shorts] = -0.5 / ns if nl > 0 else -1.0 / ns
    return pd.DataFrame(W, index=closes.index, columns=closes.columns)
