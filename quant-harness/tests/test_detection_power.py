"""Позитивный контроль судьи: он ОБЯЗАН уметь выдать PASS на хорошей стратегии
и KILL на шуме. Без этого все KILL-вердикты бессмысленны (заклинивший null-автомат).

Это тот тест, отсутствие которого чуть не оставило открытым вопрос «а не зря ли
мы получаем сплошные KILL» — теперь гарантия детекционной силы зафиксирована.
"""
import numpy as np
import pandas as pd

from harness.runner import Thresholds, run_cpcv_returns
from harness.walk_forward import CPCVConfig

IDX = pd.date_range("2024-01-01", periods=912, freq="1D", tz="UTC")
CPCV = CPCVConfig(n_groups=10, k_test=2, purge=1, embargo=5)


def _judge(mu, vol, n_cfg=6, seed0=0, trade_every=3):
    R = np.vstack([np.random.default_rng(seed0 + s).normal(mu, vol, 912)
                   for s in range(n_cfg)])
    T = np.zeros_like(R)
    T[:, ::trade_every] = 1
    return run_cpcv_returns(R, T, IDX, [{"c": i} for i in range(n_cfg)],
                            CPCV, Thresholds())


def test_judge_passes_a_genuinely_excellent_strategy():
    # реалистичная отличная стратегия: ann Sharpe ~3 (mu/vol=3/sqrt(365)), низкая vol
    vol = 0.008
    rep = _judge(mu=3.0 / np.sqrt(365) * vol, vol=vol)
    assert rep["verdict"] == "PASS", f"судья не пропускает отличную стратегию: {rep['checks']}"
    assert all(rep["checks"].values())


def test_judge_kills_pure_noise():
    rep = _judge(mu=0.0, vol=0.01)
    assert rep["verdict"] == "KILL"
    assert rep["deflated_sharpe_ratio"] < 0.5, "DSR не должен реагировать на шум"


def test_dsr_monotonic_in_signal_strength():
    # DSR обязан расти с силой сигнала — иначе детектор мёртв
    vol = 0.01
    weak = _judge(mu=0.5 / np.sqrt(365) * vol, vol=vol)["deflated_sharpe_ratio"]
    strong = _judge(mu=3.0 / np.sqrt(365) * vol, vol=vol)["deflated_sharpe_ratio"]
    assert strong > weak, f"DSR не растёт с сигналом: weak={weak} strong={strong}"
