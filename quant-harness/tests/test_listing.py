"""Пост-листинговый движок (спека 0004): возраст, корзина молодых+хедж, издержки."""
import numpy as np
import pandas as pd
import pytest

from harness.backtest import Costs
from harness.listing import (age_in_days, young_basket_weights,
                             young_vs_btc_weights, run_listing_backtest)

IDX = pd.date_range("2025-01-01", periods=200, freq="1D", tz="UTC")


def _panel():
    # A листнут с бара 0; B — с бара 100; C — с бара 190 (совсем молодой)
    closes = pd.DataFrame(index=IDX, columns=list("ABC"), dtype=float)
    closes["A"] = 100.0
    closes.loc[IDX[100:], "B"] = 50.0
    closes.loc[IDX[190:], "C"] = 10.0
    return closes


def test_age_in_days_counts_from_first_valid_bar():
    age = age_in_days(_panel())
    assert np.isnan(age.loc[IDX[99], "B"]), "до листинга возраст не определён"
    assert age.loc[IDX[100], "B"] == 0
    assert age.loc[IDX[110], "B"] == 10
    assert age.loc[IDX[150], "A"] == 150


def test_young_basket_delta_neutral_and_direction():
    closes = _panel()
    qv = pd.DataFrame(1e9, index=IDX, columns=list("ABC"))  # ликвидность не режет
    # drift(+1): молодые ∈ [1,14] лонг гросс +1, зрелые (>180д) шорт -1
    t = IDX[195]  # A стар (195д); B возраст 95 (не молод, не зрел>180); C возраст 5 (молод)
    W = young_basket_weights(closes, qv, e_start=1, h=14, direction=+1,
                             mature_age=180, min_median_qv=0.0)
    row = W.loc[t]
    assert row["C"] == pytest.approx(1.0), "единственный молодой = весь лонг-гросс"
    assert row["A"] == pytest.approx(-1.0), "единственный зрелый = весь хедж-шорт"
    assert row["B"] == 0.0
    assert row.sum() == pytest.approx(0.0), "дельта-нейтральность"


def test_young_basket_reversal_flips_sign():
    closes = _panel()
    qv = pd.DataFrame(1e9, index=IDX, columns=list("ABC"))
    W = young_basket_weights(closes, qv, e_start=1, h=14, direction=-1,
                             mature_age=180, min_median_qv=0.0)
    row = W.loc[IDX[195]]
    assert row["C"] == pytest.approx(-1.0) and row["A"] == pytest.approx(1.0)


def test_young_basket_liquidity_floor_excludes():
    closes = _panel()
    qv = pd.DataFrame(1e9, index=IDX, columns=list("ABC"))
    qv["C"] = 1.0  # молодой, но неликвидный
    W = young_basket_weights(closes, qv, e_start=1, h=14, direction=+1,
                             mature_age=180, min_median_qv=1000.0)
    # C отсеян полом -> нет молодых -> нет и хеджа (нейтральность требует обе ноги)
    assert (W.loc[IDX[195]] == 0.0).all()


def test_young_vs_btc_single_instrument_hedge():
    closes = _panel().copy()
    closes["BTCUSDT"] = 200.0  # добавляем хедж-инструмент, он всегда «зрелый»
    qv = pd.DataFrame(1e9, index=IDX, columns=closes.columns)
    # drift: молодые лонг, шорт ровно BTC (не корзину зрелых)
    W = young_vs_btc_weights(closes, qv, e_start=1, h=14, direction=+1,
                             hedge="BTCUSDT", rebalance=1, min_median_qv=0.0)
    row = W.loc[IDX[195]]  # C молод (возраст 5)
    assert row["C"] == pytest.approx(1.0), "молодой = весь лонг-гросс"
    assert row["BTCUSDT"] == pytest.approx(-1.0), "хедж — ровно BTC, вес −1"
    assert row["A"] == 0.0, "зрелые кроме BTC в хедж НЕ входят"
    assert row.sum() == pytest.approx(0.0)


def test_young_vs_btc_hedge_turnover_near_zero():
    # хедж-нога BTC должна держать постоянный вес, пока корзина молодых непуста
    closes = _panel().copy()
    closes["BTCUSDT"] = 200.0
    qv = pd.DataFrame(1e9, index=IDX, columns=closes.columns)
    W = young_vs_btc_weights(closes, qv, e_start=1, h=30, direction=+1,
                             hedge="BTCUSDT", rebalance=1, min_median_qv=0.0)
    btc = W["BTCUSDT"]
    active = btc[btc != 0]
    # среди активных баров вес BTC постоянен -1 -> оборот хеджа ноль
    assert np.allclose(active.values, -1.0)


def test_young_vs_btc_weekly_rebalance_freezes_between():
    closes = _panel().copy()
    closes["BTCUSDT"] = 200.0
    qv = pd.DataFrame(1e9, index=IDX, columns=closes.columns)
    W = young_vs_btc_weights(closes, qv, e_start=1, h=30, direction=+1,
                             hedge="BTCUSDT", rebalance=7, min_median_qv=0.0)
    changes = (W.diff().abs().sum(axis=1) > 1e-12)
    # изменения весов только на барах, кратных 7 (после старта)
    change_bars = [i for i in range(150, 200) if changes.iloc[i]]
    assert all(i % 7 == 0 for i in change_bars), f"веса меняются вне ребаланса: {change_bars}"


def test_listing_backtest_per_leg_slippage_and_lookahead():
    idx = IDX[:10]
    closes = pd.DataFrame(100.0, index=idx, columns=["Y", "M"])
    closes.iloc[5, 0] = 110.0  # скачок Y на баре 5
    closes.iloc[6:, 0] = 110.0
    funding = pd.DataFrame(0.0, index=idx, columns=["Y", "M"])
    W = pd.DataFrame(0.0, index=idx, columns=["Y", "M"])
    W.iloc[5:, 0] = 1.0    # лонг молодого Y
    W.iloc[5:, 1] = -1.0   # шорт зрелого M
    young = pd.DataFrame(False, index=idx, columns=["Y", "M"])
    young.iloc[5:, 0] = True  # Y — молодой, M — хедж
    res = run_listing_backtest(closes, funding, W, young,
                               taker=0.0, slip_young=0.010, slip_mature=0.002)
    # held = W.shift: доходность бара 5 (скачок Y) НЕ наша — вес включён на 5
    assert res.returns.iloc[5] == pytest.approx(0.0), "look-ahead"
    # бар 6: вход обеих ног. оборот Y=1 (slip 10bps+0), M=1 (slip 2bps+0)
    assert res.returns.iloc[6] == pytest.approx(-(0.010 + 0.002))
    assert res.n_trades == 2
