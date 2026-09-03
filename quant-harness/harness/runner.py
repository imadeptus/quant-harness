"""Walk-forward sweep + Deflated Sharpe + honest report.

This ties the guardrails together:
1. For each walk-forward fold, pick the IS-best config, evaluate it OOS.
2. Concatenate OOS returns across folds -> the only equity curve we believe.
3. Deflate the winning Sharpe by the number of configs tried (N).
4. Emit a numeric report + PASS/KILL verdict against pre-registered thresholds.

The selection-inside-each-fold + OOS-concatenation is what makes this honest:
parameters are always chosen on data the OOS segment never saw.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .backtest import Costs, annualization_factor, max_drawdown, run_backtest
from .deflated_sharpe import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from .families import build_signal, momentum_grid
from .walk_forward import (CPCVConfig, WalkForwardConfig, _group_bounds,
                           assemble_oos, cpcv_path_assignment, cpcv_splits,
                           walk_forward_folds)


@dataclass
class Thresholds:
    min_trades: int = 200
    min_oos_sharpe: float = 0.7        # annualized
    max_drawdown: float = 0.20         # fraction (positive number)
    min_dsr: float = 0.95


def _annualized(sharpe_periodic: float, ann: float) -> float:
    return sharpe_periodic * math.sqrt(ann)


def run(df: pd.DataFrame, grid: List[Dict], wf: WalkForwardConfig,
        costs: Costs, thr: Thresholds) -> Dict:
    n = len(df)
    ann = annualization_factor(df.index)
    N = len(grid)

    R, per_config_signal = _config_matrix(df, grid, costs)

    fold_oos_returns: List[np.ndarray] = []
    picks: List[Dict] = []
    fold_trial_vars: List[float] = []
    for train_idx, test_idx in walk_forward_folds(n, wf):
        # In-sample: pick config with best periodic Sharpe on train slice.
        train = R[:, train_idx]
        mu = train.mean(axis=1)
        sd = train.std(axis=1, ddof=1)
        sd[sd == 0] = np.inf
        is_sharpe = mu / sd
        if N > 1:
            fold_trial_vars.append(float(np.var(is_sharpe, ddof=1)))
        best = int(np.argmax(is_sharpe))
        # Out-of-sample: that config's returns on the untouched test slice.
        oos = R[best, test_idx]
        fold_oos_returns.append(oos)
        picks.append({"params": grid[best], "is_sharpe_periodic": float(is_sharpe[best]),
                      "oos_bars": int(test_idx.size)})

    oos_returns = assemble_oos(fold_oos_returns)
    oos = pd.Series(oos_returns)

    # Honest OOS trade count: turnover of each fold's chosen signal, on that
    # fold's OOS slice only (position held = signal shifted by 1).
    approx_trades = 0
    for (train_idx, test_idx), pick in zip(walk_forward_folds(n, wf), picks):
        sig = per_config_signal[_grid_index(grid, pick["params"])].values
        approx_trades += _oos_turnover_count(sig, test_idx)  # cost-consistent count

    if oos.size < 3 or oos.std(ddof=1) == 0:
        return {"error": "insufficient OOS data or zero variance", "n_folds": len(picks)}

    oos_sharpe_periodic = float(oos.mean() / oos.std(ddof=1))
    oos_sharpe_ann = _annualized(oos_sharpe_periodic, ann)
    dd = abs(max_drawdown(oos))
    # Trial-Sharpe variance for DSR: дисперсия IS-Sharpe по всем N конфигурациям
    # внутри фолда, усреднённая по фолдам — «вектор Sharpe всех trials» из
    # metrics.md. При N=1 множественного тестирования нет — консервативный 1/n.
    if fold_trial_vars:
        trial_var = float(np.mean(fold_trial_vars))
    else:
        trial_var = 1.0 / oos.size
    sr_var = max(trial_var, 1e-12)

    psr = probabilistic_sharpe_ratio(oos.values)
    dsr = deflated_sharpe_ratio(oos.values, N, sr_var)

    checks = {
        "trades_ok": approx_trades >= thr.min_trades,
        "oos_sharpe_ok": oos_sharpe_ann > thr.min_oos_sharpe,
        "drawdown_ok": dd < thr.max_drawdown,
        "dsr_ok": dsr >= thr.min_dsr,
    }
    verdict = "PASS" if all(checks.values()) else "KILL"

    return {
        "n_configs_tried": N,
        "n_folds": len(picks),
        "ann_factor": round(ann, 1),
        "oos_bars": int(oos.size),
        "approx_oos_trades": approx_trades,
        "oos_sharpe_annualized": round(oos_sharpe_ann, 3),
        "oos_max_drawdown": round(dd, 3),
        "psr_vs_zero": round(psr, 4),
        "deflated_sharpe_ratio": round(dsr, 4),
        "trial_sharpe_variance": trial_var,
        "thresholds": asdict(thr),
        "checks": checks,
        "verdict": verdict,
        "most_common_pick": _mode_params(picks),
    }


def _config_matrix(df: pd.DataFrame, grid: List[Dict], costs: Costs):
    """Per-bar net returns of every config over the full series, precomputed
    once; fold/split slicing then just indexes into these (identical accounting)."""
    per_config_returns = []
    per_config_signal = []
    for params in grid:
        sig = build_signal(df, params)
        res = run_backtest(df, sig, costs)
        per_config_returns.append(res.returns.values)
        per_config_signal.append(sig)
    return np.vstack(per_config_returns), per_config_signal


def _oos_turnover_count(signal_values: np.ndarray, idx) -> int:
    """Число оборотов внутри `idx`, посчитанное РОВНО так, как run_backtest
    списывает издержки: |Δ held| по ГЛОБАЛЬНОЙ позиции с задержкой в 1 бар
    (held[t] против held[t-1]), а НЕ пересев с нуля на границе блока. Раньше
    ведущий 0.0 на старте блока считал устойчивую ненулевую позицию, переходящую
    границу, за новый вход — фантомный оборот, которому в R не соответствует
    никакая издержка. `idx` — булев/целочисленный индекс или slice."""
    held = np.r_[0.0, signal_values[:-1]]                 # одно-баровая задержка (глобально)
    turnover = np.abs(np.diff(held, prepend=0.0))         # как в run_backtest: |held.diff()|
    return int((turnover[idx] > 1e-9).sum())


def _block_trades(signal_values: np.ndarray, start: int, end: int) -> int:
    """OOS-обороты выбранного конфига внутри блока [start, end) — согласованно с
    издержечной моделью (см. _oos_turnover_count)."""
    return _oos_turnover_count(signal_values, slice(start, end))


def run_cpcv(df: pd.DataFrame, grid: List[Dict], cpcv: CPCVConfig,
             costs: Costs, thr: Thresholds) -> Dict:
    """CPCV-вариант run(): вместо одной OOS-кривой — C(S-1, k-1) полных путей.

    Каждый сплит выбирает IS-лучший конфиг на train и отдаёт его returns на
    каждый из своих test-блоков; path-сборка складывает блоки в полные серии.
    Вердикт — по медианному пути против тех же пред-зарегистрированных порогов."""
    R, per_config_signal = _config_matrix(df, grid, costs)

    def block_trades(best: int, s: int, e: int) -> int:
        return _block_trades(per_config_signal[best].values, s, e)

    return _cpcv_core(R, annualization_factor(df.index), grid, cpcv, thr, block_trades)


def run_cpcv_returns(R, trades_per_bar, index: pd.DatetimeIndex, grid: List[Dict],
                     cpcv: CPCVConfig, thr: Thresholds,
                     n_trials: Optional[int] = None) -> Dict:
    """Судья по готовой матрице доходностей (configs × T) — для портфельных
    (cross-sectional) конфигов, где нет одного сигнала на один символ.
    trades_per_bar — матрица той же формы: число изменений позиций на бар.
    n_trials — общее число испытаний для дефляции DSR (по умолчанию len(grid)):
    конфиги, чьи доходности не переданы, учитываются числом, без материализации."""
    R = np.asarray(R, dtype=float)
    trades_per_bar = np.asarray(trades_per_bar)

    def block_trades(best: int, s: int, e: int) -> int:
        return int(trades_per_bar[best, s:e].sum())

    return _cpcv_core(R, annualization_factor(index), grid, cpcv, thr, block_trades,
                      n_trials=n_trials)


def _cpcv_core(R, ann: float, grid: List[Dict], cpcv: CPCVConfig,
               thr: Thresholds, block_trades, n_trials: Optional[int] = None) -> Dict:
    n = R.shape[1]
    N = int(n_trials) if n_trials is not None else len(grid)
    bounds = _group_bounds(n, cpcv.n_groups)
    n_paths, assignment = cpcv_path_assignment(cpcv.n_groups, cpcv.k_test)

    path_segments: List[Dict[int, tuple]] = [dict() for _ in range(n_paths)]
    fold_trial_vars: List[float] = []
    picks: List[Dict] = []
    for (train_idx, test_idx, groups), row in zip(cpcv_splits(n, cpcv), assignment):
        train = R[:, train_idx]
        mu = train.mean(axis=1)
        sd = train.std(axis=1, ddof=1)
        sd[sd == 0] = np.inf
        is_sharpe = mu / sd
        if R.shape[0] > 1:          # trial dispersion needs >= 2 supplied configs
            fold_trial_vars.append(float(np.var(is_sharpe, ddof=1)))
        best = int(np.argmax(is_sharpe))
        picks.append({"params": grid[best]})
        for g, p in zip(groups, row):
            path_segments[p][g] = (best, bounds[g])

    trial_var = float(np.mean(fold_trial_vars)) if fold_trial_vars else 1.0 / max(n, 2)
    sr_var = max(trial_var, 1e-12)

    path_sharpes, path_dds, path_trades, path_psrs, path_dsrs = [], [], [], [], []
    for p in range(n_paths):
        rets = np.concatenate([R[best, s:e] for g, (best, (s, e))
                               in sorted(path_segments[p].items())])
        sd = float(np.std(rets, ddof=1))
        if rets.size < 3 or sd == 0:
            return {"error": f"path {p}: insufficient data or zero variance",
                    "n_paths": n_paths}
        path_sharpes.append(_annualized(float(np.mean(rets)) / sd, ann))
        path_dds.append(abs(max_drawdown(pd.Series(rets))))
        path_trades.append(sum(block_trades(best, s, e)
                               for g, (best, (s, e)) in sorted(path_segments[p].items())))
        path_psrs.append(probabilistic_sharpe_ratio(rets))
        path_dsrs.append(deflated_sharpe_ratio(rets, N, sr_var))

    med = lambda x: float(np.median(x))  # noqa: E731
    oos_sharpe_ann = med(path_sharpes)
    dd = med(path_dds)
    trades = int(med(path_trades))
    psr = med(path_psrs)
    dsr = med(path_dsrs)

    checks = {
        "trades_ok": trades >= thr.min_trades,
        "oos_sharpe_ok": oos_sharpe_ann > thr.min_oos_sharpe,
        "drawdown_ok": dd < thr.max_drawdown,
        "dsr_ok": dsr >= thr.min_dsr,
    }
    verdict = "PASS" if all(checks.values()) else "KILL"

    return {
        "cv": "cpcv",
        "n_configs_tried": N,
        "n_splits": len(picks),
        "n_paths": n_paths,
        "ann_factor": round(ann, 1),
        "oos_bars": int(sum(e - s for _, (_, (s, e)) in sorted(path_segments[0].items()))),
        "approx_oos_trades": trades,
        "oos_sharpe_annualized": round(oos_sharpe_ann, 3),
        "path_sharpes_annualized": [round(s, 3) for s in path_sharpes],
        "worst_path_sharpe_annualized": round(min(path_sharpes), 3),
        "oos_max_drawdown": round(dd, 3),
        "psr_vs_zero": round(psr, 4),
        "deflated_sharpe_ratio": round(dsr, 4),
        "trial_sharpe_variance": trial_var,
        "thresholds": asdict(thr),
        "checks": checks,
        "verdict": verdict,
        "most_common_pick": _mode_params(picks),
    }


def _grid_index(grid: List[Dict], params: Dict) -> int:
    for i, p in enumerate(grid):
        if p == params:
            return i
    raise ValueError("pick not found in grid")


def _mode_params(picks: List[Dict]) -> Dict:
    from collections import Counter
    key = Counter(json.dumps(p["params"], sort_keys=True) for p in picks)
    top = key.most_common(1)[0][0]
    return json.loads(top)
