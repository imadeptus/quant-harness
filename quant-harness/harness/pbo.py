"""Probability of Backtest Overfitting (PBO) via CSCV.

Bailey, Borwein, Lopez de Prado, Zhu (2015). Complements the Deflated Sharpe:
DSR asks "is THIS winner significant?"; PBO asks "does my SELECTION PROCESS
overfit?" — i.e. does the in-sample-best config tend to underperform OOS?

Method (combinatorially symmetric CV): split the (config x time) return matrix
into S blocks; for every way to choose S/2 blocks as in-sample, find the IS-best
config, then look at its performance RANK on the complementary OOS blocks. PBO =
fraction of splits where the IS-best lands below the OOS median. High PBO (>0.5)
means the search is fooling itself regardless of any single strategy's DSR.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def pbo_cscv(returns_matrix: np.ndarray, n_blocks: int = 10) -> float:
    """returns_matrix: shape (n_configs, n_time). Returns PBO in [0,1] or nan."""
    R = np.asarray(returns_matrix, dtype=float)
    n_cfg, n_t = R.shape
    if n_cfg < 2 or n_t < n_blocks or n_blocks < 4 or n_blocks % 2 != 0:
        return float("nan")
    # Trim to a multiple of n_blocks and split into contiguous blocks.
    usable = (n_t // n_blocks) * n_blocks
    R = R[:, :usable]
    blocks = np.array_split(np.arange(usable), n_blocks)

    def sharpe(x):
        s = x.std(axis=1, ddof=1)
        s[s == 0] = np.inf
        return x.mean(axis=1) / s

    logits = []
    half = n_blocks // 2
    for is_sel in combinations(range(n_blocks), half):
        is_idx = np.concatenate([blocks[b] for b in is_sel])
        oos_sel = [b for b in range(n_blocks) if b not in is_sel]
        oos_idx = np.concatenate([blocks[b] for b in oos_sel])
        is_sr = sharpe(R[:, is_idx])
        oos_sr = sharpe(R[:, oos_idx])
        best = int(np.argmax(is_sr))
        # rank of the IS-best among OOS Sharpes (1 = worst .. n = best)
        rank = float((oos_sr <= oos_sr[best]).sum()) / n_cfg
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.array(logits)
    return float((logits <= 0).mean())  # fraction of splits where best <= median OOS
