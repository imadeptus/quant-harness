"""Paper-трекер funding harvest (ядро учёта): базис-P&L по реальным ценам обеих ног."""
import pytest

from harness.paper import new_state, tick

COSTS = {"taker": 0.0005, "slip": 0.0002}
NOTIONAL = 1000.0


def test_position_marks_basis_pnl_and_funding():
    st = new_state(capital=10000.0, notional=NOTIONAL)
    # открыли DOGE вручную (в универсе)
    st = tick(st, prices={"DOGE": {"spot": 0.10, "perp": 0.10}},
              funding_since={"DOGE": 0.0}, target=["DOGE"], costs=COSTS, ts="t0")
    assert "DOGE" in st["positions"]
    # следующий тик: спот +10%, перп +5% -> базис 5%; funding 0.001
    st = tick(st, prices={"DOGE": {"spot": 0.11, "perp": 0.105}},
              funding_since={"DOGE": 0.001}, target=["DOGE"], costs=COSTS, ts="t1")
    # базис-P&L = notional*(spot_ret - perp_ret) = 1000*(0.10-0.05)=50 ; funding=1000*0.001=1
    assert st["cum_basis_pnl"] == pytest.approx(50.0, abs=1e-6)
    assert st["cum_funding"] == pytest.approx(1.0, abs=1e-6)


def test_entry_and_exit_charge_both_legs():
    st = new_state(capital=10000.0, notional=NOTIONAL)
    st = tick(st, prices={"WIF": {"spot": 1.0, "perp": 1.0}},
              funding_since={"WIF": 0.0}, target=["WIF"], costs=COSTS, ts="t0")
    # вход: оборот notional на ОБЕИХ биржах -> 2*(taker+slip)*notional
    assert st["cum_costs"] == pytest.approx(NOTIONAL * 2 * 0.0007, abs=1e-6)
    entry_cost = st["cum_costs"]
    # выход (target пуст): ещё раз обе ноги
    st = tick(st, prices={"WIF": {"spot": 1.0, "perp": 1.0}},
              funding_since={"WIF": 0.0}, target=[], costs=COSTS, ts="t1")
    assert "WIF" not in st["positions"]
    assert st["cum_costs"] == pytest.approx(2 * entry_cost, abs=1e-6)


def test_equity_is_capital_plus_net_pnl():
    st = new_state(capital=10000.0, notional=NOTIONAL)
    st = tick(st, prices={"DOGE": {"spot": 0.10, "perp": 0.10}},
              funding_since={"DOGE": 0.0}, target=["DOGE"], costs=COSTS, ts="t0")
    st = tick(st, prices={"DOGE": {"spot": 0.11, "perp": 0.105}},
              funding_since={"DOGE": 0.001}, target=["DOGE"], costs=COSTS, ts="t1")
    expected = 10000.0 + st["cum_funding"] + st["cum_basis_pnl"] - st["cum_costs"]
    assert st["equity"] == pytest.approx(expected, abs=1e-6)


def test_flags_basis_divergence_risk():
    st = new_state(capital=10000.0, notional=NOTIONAL)
    st = tick(st, prices={"X": {"spot": 1.0, "perp": 1.0}},
              funding_since={"X": 0.0}, target=["X"], costs=COSTS, ts="t0")
    # большой разъезд базиса на тике -> флаг риска
    st = tick(st, prices={"X": {"spot": 1.06, "perp": 1.0}},
              funding_since={"X": 0.0}, target=["X"], costs=COSTS, ts="t1",
              basis_alert=0.05)
    assert any("X" in w for w in st["ticks"][-1]["warnings"])
