"""XS-семейство (спека 0001): point-in-time universe, веса, портфельный движок."""
import numpy as np
import pandas as pd
import pytest

from harness.backtest import Costs
from harness.xs import monthly_universe, run_xs_backtest, xs_carry_weights, xs_momentum_weights

IDX = pd.date_range("2025-01-01", periods=900, freq="4h", tz="UTC")  # ~150 дней


def _qv_panel():
    qv = pd.DataFrame(index=IDX, columns=list("ABCD"), dtype=float)
    qv["A"] = 1000.0
    qv["B"] = np.where(IDX < pd.Timestamp("2025-03-01", tz="UTC"), 900.0, 10.0)
    qv["C"] = np.where(IDX < pd.Timestamp("2025-03-01", tz="UTC"), 10.0, 900.0)
    qv["D"] = np.nan  # листинг 15 апреля
    qv.loc[IDX >= pd.Timestamp("2025-04-15", tz="UTC"), "D"] = 2000.0
    return qv


def test_universe_point_in_time_topk():
    # окно объёма 180 баров (~30д), мин. история 540 баров (~90д), топ-2
    m = monthly_universe(_qv_panel(), top_k=2, vol_window=180, min_history=540)
    t_feb = pd.Timestamp("2025-02-10", tz="UTC")
    t_apr = pd.Timestamp("2025-04-10", tz="UTC")
    t_may = pd.Timestamp("2025-05-10", tz="UTC")
    assert m.loc[t_feb, "A"] and m.loc[t_feb, "B"] and not m.loc[t_feb, "C"]
    # объём B упал 1 марта -> к апрельскому ребалансу окно это видит
    assert m.loc[t_apr, "A"] and m.loc[t_apr, "C"] and not m.loc[t_apr, "B"]
    # D огромный по объёму, но истории < 90д -> исключён
    assert not m.loc[t_may, "D"]
    # состав постоянен внутри месяца
    apr = m.loc["2025-04"]
    assert (apr.nunique() <= 1).all()


def test_universe_no_lookahead_before_first_window():
    m = monthly_universe(_qv_panel(), top_k=2, vol_window=180, min_history=540)
    assert not m.loc["2025-01"].any().any(), "членство до первого полного окна"


def _flat_member(cols=list("ABCD")):
    return pd.DataFrame(True, index=IDX, columns=cols)


def test_xs_momentum_weights_neutral_and_ranked():
    closes = pd.DataFrame(100.0, index=IDX, columns=list("ABCD"))
    # скачок внутри lookback-окна: на t0 прошлые 6-баровые доходности различаются
    t0 = 300
    for sym, g in {"A": 1.10, "B": 1.01, "C": 0.99, "D": 0.90}.items():
        closes.iloc[t0 - 5:, closes.columns.get_loc(sym)] = 100.0 * g
    w = xs_momentum_weights(closes, _flat_member(), lookback=6, skip=0,
                            quantile=0.25, rebalance=1)
    row = w.iloc[t0]
    assert row["A"] == pytest.approx(0.5) and row["D"] == pytest.approx(-0.5)
    assert row["B"] == 0.0 and row["C"] == 0.0
    assert abs(row.sum()) < 1e-12 and row.abs().sum() == pytest.approx(1.0)


def test_xs_momentum_rebalance_cadence():
    rng = np.random.default_rng(0)
    closes = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (len(IDX), 4)), axis=0),
        index=IDX, columns=list("ABCD"))
    w = xs_momentum_weights(closes, _flat_member(), lookback=42, skip=6,
                            quantile=0.25, rebalance=42)
    changes = (w.diff().abs().sum(axis=1) > 1e-12)
    assert changes.iloc[200:].sum() <= len(IDX) // 42 + 1, "веса меняются чаще ребаланса"


def test_xs_carry_weights_long_cheap_short_expensive():
    funding = pd.DataFrame(0.0, index=IDX, columns=list("ABCD"))
    funding["A"] = -0.0002   # отрицательный funding -> лонг получает
    funding["D"] = 0.0005    # дорогой funding -> шорт получает
    w = xs_carry_weights(funding, _flat_member(), window=18, quantile=0.25, rebalance=1)
    row = w.iloc[300]
    assert row["A"] == pytest.approx(0.5), "самый дешёвый funding должен быть лонгом"
    assert row["D"] == pytest.approx(-0.5), "самый дорогой funding должен быть шортом"


def test_xs_backtest_no_lookahead_and_funding_sign():
    closes = pd.DataFrame(100.0, index=IDX[:10], columns=["A", "B"])
    closes.iloc[5, 0] = 110.0  # скачок A на баре 5
    closes.iloc[6:, 0] = 110.0
    funding = pd.DataFrame(0.0, index=IDX[:10], columns=["A", "B"])
    W = pd.DataFrame(0.0, index=IDX[:10], columns=["A", "B"])
    W.iloc[5:, 0] = 0.5  # вес включён на баре 5 — доходность бара 5 НЕ наша
    res = run_xs_backtest(closes, funding, W, Costs(taker_fee=0.0, slippage=0.0))
    assert res.returns.iloc[5] == pytest.approx(0.0), "look-ahead: вес бара t заработал на баре t"
    assert res.returns.iloc[6] == pytest.approx(0.0), "цена уже не движется после скачка"

    # знак funding: лонг платит положительную ставку
    funding2 = pd.DataFrame(0.001, index=IDX[:10], columns=["A", "B"])
    res2 = run_xs_backtest(closes * 0 + 100.0, funding2, W, Costs(taker_fee=0.0, slippage=0.0))
    assert res2.returns.iloc[6] == pytest.approx(-0.5 * 0.001)


def test_xs_backtest_costs_on_turnover():
    closes = pd.DataFrame(100.0, index=IDX[:10], columns=["A", "B"])
    funding = pd.DataFrame(0.0, index=IDX[:10], columns=["A", "B"])
    W = pd.DataFrame(0.0, index=IDX[:10], columns=["A", "B"])
    W.iloc[3:, 0] = 0.5
    W.iloc[3:, 1] = -0.5
    res = run_xs_backtest(closes, funding, W, Costs(taker_fee=0.0005, slippage=0.0002))
    # вход двумя ногами на баре 4 (held = W.shift): оборот 1.0 * 7 bps
    assert res.returns.iloc[4] == pytest.approx(-1.0 * 0.0007)
    assert res.n_trades == 2
    assert res.trade_events.iloc[4] == 2 and res.trade_events.sum() == res.n_trades
