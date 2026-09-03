"""CPCV: комбинаторные сплиты с purge/embargo и сборка полных OOS-путей."""
from itertools import combinations
from math import comb

from harness.walk_forward import CPCVConfig, cpcv_path_assignment, cpcv_splits

S, K, N = 6, 2, 600  # 6 групп по 100 баров


def _cfg():
    return CPCVConfig(n_groups=S, k_test=K, purge=3, embargo=4)


def test_cpcv_split_count_and_group_coverage():
    splits = list(cpcv_splits(N, _cfg()))
    assert len(splits) == comb(S, K)
    counts = {}
    for _, _, groups in splits:
        for g in groups:
            counts[g] = counts.get(g, 0) + 1
    assert all(counts[g] == comb(S - 1, K - 1) for g in range(S))


def test_cpcv_no_leakage_purge_embargo():
    group = N // S
    for train, test, groups in cpcv_splits(N, _cfg()):
        train_s, test_s = set(train.tolist()), set(test.tolist())
        assert not (train_s & test_s), "train пересекается с test"
        for g in groups:
            start, end = g * group, (g + 1) * group
            for i in range(max(0, start - 3), start):
                assert i not in train_s, "purge нарушен (бар перед test-блоком в train)"
            for i in range(end, min(N, end + 4)):
                assert i not in train_s, "embargo нарушен (бар после test-блока в train)"


def test_run_cpcv_zero_edge_kill_with_path_distribution():
    import numpy as np

    from harness.backtest import Costs
    from harness.data import make_synthetic
    from harness.runner import Thresholds, run_cpcv

    bars = make_synthetic(n=3000, seed=2, edge=0.0)
    grid = [{"family": "momentum", "lookback": lb, "threshold": 0.02, "allow_short": True}
            for lb in (6, 24, 96)]
    cfg = CPCVConfig(n_groups=6, k_test=2, purge=1, embargo=96)
    rep = run_cpcv(bars.df, grid, cfg, Costs(apply_funding=False), Thresholds())
    assert rep["verdict"] == "KILL"
    assert rep["n_paths"] == comb(5, 1)
    assert len(rep["path_sharpes_annualized"]) == rep["n_paths"]
    assert np.isfinite(rep["deflated_sharpe_ratio"])
    assert rep["oos_bars"] == 3000, "каждый путь должен покрывать всю серию"


def test_run_cpcv_returns_matrix_zero_edge_kill():
    """Судья должен принимать готовую матрицу (configs x T) доходностей —
    для портфельных (XS) конфигов, где нет одного сигнала на символ."""
    import numpy as np
    import pandas as pd

    from harness.runner import Thresholds, run_cpcv_returns

    rng = np.random.default_rng(3)
    idx = pd.date_range("2025-01-01", periods=2000, freq="4h", tz="UTC")
    R = rng.normal(0.0, 0.001, (5, 2000))
    trades = (rng.random((5, 2000)) < 0.05).astype(int)
    grid = [{"family": "xs_momentum", "cfg": i} for i in range(5)]
    rep = run_cpcv_returns(R, trades, idx, grid,
                           CPCVConfig(n_groups=6, k_test=2, purge=1, embargo=42),
                           Thresholds())
    assert rep["verdict"] == "KILL"
    assert rep["n_paths"] == comb(5, 1)
    assert rep["approx_oos_trades"] > 0
    assert len(rep["path_sharpes_annualized"]) == rep["n_paths"]


def test_cpcv_paths_cover_each_group_exactly_once():
    n_paths, assignment = cpcv_path_assignment(S, K)
    assert n_paths == comb(S - 1, K - 1)
    seen = [set() for _ in range(n_paths)]
    for s, groups in enumerate(combinations(range(S), K)):
        for j, g in enumerate(groups):
            p = assignment[s][j]
            assert g not in seen[p], "группа попала в путь дважды"
            seen[p].add(g)
    assert all(len(x) == S for x in seen), "путь покрывает не все группы"
