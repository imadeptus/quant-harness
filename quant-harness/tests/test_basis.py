"""Движок межбиржевых funding-пар (спека 0003)."""
import numpy as np
import pandas as pd
import pytest

from harness.backtest import Costs
from harness.basis import basis_weights, run_basis_backtest

IDX = pd.date_range("2025-01-01", periods=100, freq="1D", tz="UTC")


def test_basis_weights_receive_positive_spread():
    spread = pd.DataFrame({"A": 0.001, "B": 0.0}, index=IDX)
    member = pd.DataFrame(True, index=IDX, columns=["A", "B"])
    W = basis_weights(spread, member, window=3, theta=0.0005, rebalance=1)
    row = W.iloc[10]
    # спред binance-bybit > 0 -> ПОЛУЧАТЬ его = шорт дорогой (binance) ноги: w < 0
    assert row["A"] == pytest.approx(-1.0), "единственная активная пара забирает весь гросс"
    assert row["B"] == 0.0, "|спред| ниже порога — пара неактивна"


def test_basis_backtest_price_neutral_and_funding_sign():
    rng = np.random.default_rng(1)
    ret = pd.DataFrame(rng.normal(0, 0.02, (100, 1)), index=IDX, columns=["A"])
    f_bin = pd.DataFrame(0.0003, index=IDX, columns=["A"])
    f_byb = pd.DataFrame(0.0, index=IDX, columns=["A"])
    W = pd.DataFrame(-1.0, index=IDX, columns=["A"])  # шорт binance / лонг bybit
    res = run_basis_backtest(ret, ret.copy(), f_bin, f_byb, W,
                             Costs(taker_fee=0.0, slippage=0.0))
    # цены обеих ног идентичны -> ценовой PnL ноль, остаётся чистый спред
    assert res.returns.iloc[10] == pytest.approx(0.0003)


def test_basis_costs_four_legs_and_trades():
    ret = pd.DataFrame(0.0, index=IDX[:10], columns=["A"])
    f = pd.DataFrame(0.0, index=IDX[:10], columns=["A"])
    W = pd.DataFrame(0.0, index=IDX[:10], columns=["A"])
    W.iloc[3:] = -1.0
    res = run_basis_backtest(ret, ret.copy(), f, f.copy(), W,
                             Costs(taker_fee=0.0005, slippage=0.0002))
    # вход пары на баре 4: оборот 1.0 на ОБЕИХ биржах -> 2 x 7 bps
    assert res.returns.iloc[4] == pytest.approx(-2 * 0.0007)
    assert res.n_trades == 2, "вход пары = 2 венью-ноги"
