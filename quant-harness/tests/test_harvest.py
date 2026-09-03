"""Funding harvest (спека 0008): доходность по данным + стресс хвоста."""
import numpy as np
import pandas as pd
import pytest

from harness.harvest import harvest_returns, basis_stress, funding_flip_stress

IDX = pd.date_range("2025-01-01", periods=100, freq="1D", tz="UTC")


def test_harvest_collects_positive_funding_minus_costs():
    # A: стабильно положительный funding 0.001/день; B: ноль
    funding = pd.DataFrame({"A": 0.001, "B": 0.0}, index=IDX)
    member = pd.DataFrame(True, index=IDX, columns=["A", "B"])
    res = harvest_returns(funding, member, theta=0.0002, trail=7,
                          taker=0.0005, slip=0.0002)
    r = res["returns"]
    # после разогрева держим только A (funding>θ); дневной ~ 0.001 минус амортизация входа
    assert r.iloc[40] == pytest.approx(0.001, abs=1e-6), "должны собирать funding A"
    assert res["avg_positions"] > 0
    assert res["annualized_yield"] > 0


def test_harvest_excludes_below_threshold():
    # funding 0.0001 < θ 0.0002 -> не держим, доходность ~0
    funding = pd.DataFrame({"A": 0.0001}, index=IDX)
    member = pd.DataFrame(True, index=IDX, columns=["A"])
    res = harvest_returns(funding, member, theta=0.0002, trail=7,
                          taker=0.0005, slip=0.0002)
    assert res["returns"].iloc[40:].abs().sum() == pytest.approx(0.0, abs=1e-9)


def test_harvest_charges_entry_cost_both_legs():
    # вход в A на первом баре после разогрева: издержка = оборот × 2 ноги × 7bps
    funding = pd.DataFrame({"A": 0.001}, index=IDX)
    member = pd.DataFrame(True, index=IDX, columns=["A"])
    res = harvest_returns(funding, member, theta=0.0002, trail=7,
                          taker=0.0005, slip=0.0002)
    # на баре входа net = funding - 2*(taker+slip); найдём первый бар с позицией
    entry = res["turnover"] > 0
    first = entry.idxmax()
    assert res["cost"].loc[first] == pytest.approx(1.0 * 2 * 0.0007, abs=1e-9)


def test_basis_stress_wipes_months_of_yield():
    funding = pd.DataFrame({"A": 0.001, "B": 0.001}, index=IDX)
    member = pd.DataFrame(True, index=IDX, columns=["A", "B"])
    res = harvest_returns(funding, member, theta=0.0002, trail=7,
                          taker=0.0005, slip=0.0002)
    st = basis_stress(res, gap=0.05)
    # gap 5% на дельта-нейтраль портфель = разовый убыток ~5% на задействованный ноционал
    assert st["stress_loss"] == pytest.approx(0.05, abs=1e-9)
    # сколько дней доходности стирает
    assert st["days_of_yield_lost"] > 0


def test_harvest_includes_real_basis_pnl():
    # реальный базис: держим A, спот дрейфует против перпа -> базис-драг в P&L
    funding = pd.DataFrame({"A": 0.001}, index=IDX)
    member = pd.DataFrame(True, index=IDX, columns=["A"])
    # basis[A] = spot_ret - perp_ret; отрицательный дрейф -0.0003/день
    basis = pd.DataFrame({"A": -0.0003}, index=IDX)
    base = harvest_returns(funding, member, theta=0.0002, trail=7,
                           taker=0.0005, slip=0.0002)
    withb = harvest_returns(funding, member, theta=0.0002, trail=7,
                            taker=0.0005, slip=0.0002, basis=basis)
    # с базис-драгом доходность НИЖЕ на ~0.0003/день по удерживаемым барам
    assert withb["annualized_yield"] < base["annualized_yield"]
    # на баре, где держим A, разница ровно базис
    held_bar = 40
    assert (base["returns"].iloc[held_bar] - withb["returns"].iloc[held_bar]) == pytest.approx(0.0003, abs=1e-6)


def test_funding_flip_stress_is_negative():
    funding = pd.DataFrame({"A": 0.001}, index=IDX)
    member = pd.DataFrame(True, index=IDX, columns=["A"])
    res = harvest_returns(funding, member, theta=0.0002, trail=7,
                          taker=0.0005, slip=0.0002)
    fl = funding_flip_stress(res, flip_rate=-0.0002, days=30)
    assert fl["flip_loss"] < 0, "месяц отрицательного funding = убыток"
