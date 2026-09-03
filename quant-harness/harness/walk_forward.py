"""Leakage-safe walk-forward splitting.

The single most common way a backtest lies is by measuring performance on data
the parameters were fit on. Walk-forward fixes this: fit on a past window,
measure on the *next, untouched* window, roll forward. The concatenation of the
untouched test windows is the only equity curve you are allowed to believe.

Two extras that prevent subtle leakage:
- `embargo`: drop a few bars between train and test so a feature computed with a
  lookback near the boundary cannot peek across it.
- anchored vs rolling: anchored keeps train start fixed (growing window);
  rolling uses a fixed-length train window (adapts to regime, forgets old data).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Iterator, List, Tuple

import numpy as np


@dataclass
class WalkForwardConfig:
    train_size: int          # bars in each training window
    test_size: int           # bars in each out-of-sample test window
    embargo: int = 0         # bars skipped between train end and test start
    anchored: bool = False   # True = growing train window from a fixed start
    step: int | None = None  # roll step; defaults to test_size (non-overlapping OOS)


def walk_forward_folds(n: int, cfg: WalkForwardConfig) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) integer-index arrays over a series of length n.

    Test windows are non-overlapping by default, so concatenating them gives a
    clean, gap-free OOS series with no bar reused.
    """
    step = cfg.step or cfg.test_size
    if cfg.train_size <= 0 or cfg.test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    test_start = cfg.train_size + cfg.embargo
    while test_start + cfg.test_size <= n:
        train_start = 0 if cfg.anchored else max(0, test_start - cfg.embargo - cfg.train_size)
        train_end = test_start - cfg.embargo          # exclusive
        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(test_start, test_start + cfg.test_size)
        yield train_idx, test_idx
        test_start += step


def assemble_oos(fold_test_returns: List[np.ndarray]) -> np.ndarray:
    """Concatenate per-fold OOS return arrays into one honest OOS return series."""
    if not fold_test_returns:
        return np.array([])
    return np.concatenate(fold_test_returns)


def n_folds(n: int, cfg: WalkForwardConfig) -> int:
    return sum(1 for _ in walk_forward_folds(n, cfg))


@dataclass
class CPCVConfig:
    """Combinatorial Purged CV (López de Prado, AFML ch.7).

    Серия режется на n_groups смежных блоков; каждый тест-сет — комбинация
    k_test блоков, train — остальное минус purge-бары перед каждым тест-блоком
    (метка train-бара не должна заезжать в test) и embargo-бары после (фича с
    lookback'ом не должна заглядывать в test)."""
    n_groups: int = 10
    k_test: int = 2
    purge: int = 1
    embargo: int = 0


def _group_bounds(n: int, n_groups: int) -> List[Tuple[int, int]]:
    edges = np.linspace(0, n, n_groups + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_groups)]


def cpcv_splits(n: int, cfg: CPCVConfig) -> Iterator[Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]]:
    """Yield (train_idx, test_idx, test_groups) для каждой комбинации C(S, k)."""
    if cfg.k_test <= 0 or cfg.n_groups <= cfg.k_test:
        raise ValueError("need 0 < k_test < n_groups")
    bounds = _group_bounds(n, cfg.n_groups)
    for groups in combinations(range(cfg.n_groups), cfg.k_test):
        test_mask = np.zeros(n, dtype=bool)
        banned = np.zeros(n, dtype=bool)
        for g in groups:
            s, e = bounds[g]
            test_mask[s:e] = True
            banned[max(0, s - cfg.purge):s] = True
            banned[e:min(n, e + cfg.embargo)] = True
        train_mask = ~test_mask & ~banned
        yield np.where(train_mask)[0], np.where(test_mask)[0], groups


def cpcv_path_assignment(n_groups: int, k_test: int) -> Tuple[int, List[List[int]]]:
    """Раскладка (split, test-группа) -> путь: C(S-1, k-1) полных OOS-путей,
    каждый путь содержит каждую группу ровно один раз.

    Жадное правило «первый путь, где этой группы ещё нет» корректно: каждая
    группа встречается в тест-сетах ровно C(S-1, k-1) раз — по разу на путь."""
    n_paths = comb(n_groups - 1, k_test - 1)
    used: List[set] = [set() for _ in range(n_paths)]
    assignment: List[List[int]] = []
    for groups in combinations(range(n_groups), k_test):
        row = []
        for g in groups:
            for p in range(n_paths):
                if g not in used[p]:
                    used[p].add(g)
                    row.append(p)
                    break
            else:
                raise RuntimeError(f"path assignment failed for group {g}")
        assignment.append(row)
    return n_paths, assignment


if __name__ == "__main__":
    # Self-check: folds must be ordered, non-overlapping in test, and train must
    # always end strictly before test begins (no look-ahead across the boundary).
    n = 1000
    cfg = WalkForwardConfig(train_size=300, test_size=100, embargo=5, anchored=False)
    prev_test_end = -1
    for tr, te in walk_forward_folds(n, cfg):
        assert tr[-1] < te[0], "train leaks into test"
        assert te[0] - tr[-1] - 1 >= cfg.embargo, "embargo violated"
        assert te[0] > prev_test_end, "test windows overlap or go backwards"
        prev_test_end = te[-1]
    print(f"walk_forward self-check passed: {n_folds(n, cfg)} folds, no leakage.")
