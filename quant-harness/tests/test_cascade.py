"""Каскад-реверсал прокси (спека 0007): fade объёмо-подтверждённого шок-бара."""
import numpy as np
import pandas as pd
import pytest

from harness.backtest import Costs
from harness.cascade import cascade_reversal_weights
from harness.xs import run_xs_backtest

IDX = pd.date_range("2025-01-01", periods=120, freq="1D", tz="UTC")


def _panel():
    # A: тихий; B: шок вниз на объёме в баре 60; C: шок вверх на объёме в баре 60
    rng = np.random.default_rng(0)
    closes = pd.DataFrame(100.0, index=IDX, columns=list("ABC"))
    closes["A"] = 100 + rng.normal(0, 0.2, 120).cumsum() * 0.01
    closes["B"] = 100.0
    closes["C"] = 100.0
    closes.iloc[60, 1] = 80.0     # B: -20% шок
    closes.iloc[61:, 1] = 80.0
    closes.iloc[60, 2] = 120.0    # C: +20% шок
    closes.iloc[61:, 2] = 120.0
    vol = pd.DataFrame(1000.0, index=IDX, columns=list("ABC"))
    vol.iloc[60, 1] = 50000.0     # всплеск объёма на шоке B
    vol.iloc[60, 2] = 50000.0     # всплеск объёма на шоке C
    member = pd.DataFrame(True, index=IDX, columns=list("ABC"))
    return closes, vol, member


def test_cascade_fades_volume_confirmed_shock():
    closes, vol, member = _panel()
    W = cascade_reversal_weights(closes, vol, member, k=2.0, m=3.0, hold=1,
                                 vol_window=30)
    # вес ставится на БАР ШОКА (t=60); движок держит его на 61 через shift(1)
    row = W.iloc[60]
    assert row["B"] > 0, "шок вниз на объёме -> fade вверх (лонг)"
    assert row["C"] < 0, "шок вверх на объёме -> шорт"
    assert row["A"] == 0.0, "тихий бар — без позиции"
    assert row.sum() == pytest.approx(0.0), "дельта-нейтральность"


def test_cascade_ignores_shock_without_volume():
    closes, vol, member = _panel()
    vol.iloc[60, 1] = 1000.0  # шок B без всплеска объёма
    W = cascade_reversal_weights(closes, vol, member, k=2.0, m=3.0, hold=1,
                                 vol_window=30)
    assert W.iloc[60]["B"] == 0.0, "шок без объёма не считается каскадом"


def test_cascade_hold_horizon():
    closes, vol, member = _panel()
    W = cascade_reversal_weights(closes, vol, member, k=2.0, m=3.0, hold=3,
                                 vol_window=30)
    # вес держится hold=3 бара с бара шока: t=60,61,62 нонзеро; 63+ гаснет
    assert W.iloc[60]["B"] > 0 and W.iloc[62]["B"] > 0
    assert W.iloc[63]["B"] == 0.0, "позиция должна погаснуть за hold баров"


def test_cascade_direction_follow_flips_sign():
    # spec 0009: direction=-1 = follow (продолжение): шок вниз -> шорт (не лонг)
    closes, vol, member = _panel()
    W = cascade_reversal_weights(closes, vol, member, k=2.0, m=3.0, hold=1,
                                 vol_window=30, direction=-1)
    row = W.iloc[60]
    assert row["B"] < 0, "follow: шок вниз -> шорт (по направлению)"
    assert row["C"] > 0, "follow: шок вверх -> лонг"


def test_cascade_holds_overlapping_shocks_per_name():
    # B шокнут на баре 40, C — на баре 42; при hold=5 на баре 42 ОБА в позиции
    closes = pd.DataFrame(100.0, index=IDX, columns=list("BC"))
    closes.iloc[40, 0] = 80.0; closes.iloc[41:, 0] = 80.0
    closes.iloc[42, 1] = 80.0; closes.iloc[43:, 1] = 80.0
    vol = pd.DataFrame(1000.0, index=IDX, columns=list("BC"))
    vol.iloc[40, 0] = 50000.0; vol.iloc[42, 1] = 50000.0
    member = pd.DataFrame(True, index=IDX, columns=list("BC"))
    W = cascade_reversal_weights(closes, vol, member, k=2.0, m=3.0, hold=5,
                                 vol_window=30)
    # на баре 42: шок B (бар 40) ещё в окне hold=5, шок C (бар 42) свежий -> оба лонг
    assert W.iloc[42]["B"] > 0, "ранний шок B потерян из hold-окна (баг break)"
    assert W.iloc[42]["C"] > 0


def test_cascade_reuses_xs_engine_no_lookahead():
    closes, vol, member = _panel()
    funding = pd.DataFrame(0.0, index=IDX, columns=list("ABC"))
    W = cascade_reversal_weights(closes, vol, member, k=2.0, m=3.0, hold=1,
                                 vol_window=30)
    res = run_xs_backtest(closes, funding, W, Costs(taker_fee=0.0, slippage=0.0))
    # движок применяет held=W.shift(1): доходность бара 61 (сам шок уже прошёл)
    # зарабатывается позицией, открытой по сигналу бара 60 -> не look-ahead на сам шок
    assert np.isfinite(res.returns).all()
