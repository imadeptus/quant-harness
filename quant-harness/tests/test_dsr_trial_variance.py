"""Точная trial-variance для DSR: дисперсия IS-Sharpe по ВСЕМ конфигурациям.

Дискриминирующее свойство против старой прокси (variance победителей по фолдам):
грид из N одинаковых конфигов — это N раз один и тот же trial, их дисперсия 0.
Прокси по фолдам дала бы здесь ненулевое число (фолды-то разные).
"""
from harness.backtest import Costs
from harness.data import make_synthetic
from harness.runner import Thresholds, run
from harness.walk_forward import WalkForwardConfig


def _run(grid):
    bars = make_synthetic(n=2000, seed=1, edge=0.0)
    wf = WalkForwardConfig(train_size=500, test_size=200, embargo=10)
    return run(bars.df, grid, wf, Costs(apply_funding=False), Thresholds())


def _mom(lookback):
    return {"family": "momentum", "lookback": lookback,
            "threshold": 0.02, "allow_short": True}


def test_trial_variance_zero_for_identical_configs():
    rep = _run([_mom(24) for _ in range(5)])
    assert rep["trial_sharpe_variance"] <= 1e-8


def test_trial_variance_positive_for_diverse_grid():
    rep = _run([_mom(lb) for lb in (6, 24, 96, 336)])
    assert rep["trial_sharpe_variance"] > 1e-8
