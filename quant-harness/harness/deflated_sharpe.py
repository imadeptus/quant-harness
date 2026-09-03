"""Deflated Sharpe Ratio (DSR) and Probabilistic Sharpe Ratio (PSR).

Reference: Bailey & Lopez de Prado, "The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting and Non-Normality" (2014).

Why this matters
----------------
If you try N strategy configurations and keep the best Sharpe, the best is
inflated purely by selection. The DSR asks: given that I ran N trials, and given
the variance of Sharpe estimates across those trials, is the winning Sharpe
still statistically distinguishable from zero? DSR is a probability in [0, 1];
values above ~0.95 mean the observed Sharpe is unlikely to be a fluke of the
search. A raw Sharpe reported without N is not interpretable.

The PSR is the same idea for a single strategy (N implicitly 1): probability the
true Sharpe exceeds a benchmark (default 0) given sample length, skew, kurtosis.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats


def _sharpe_and_moments(returns: np.ndarray):
    """Non-annualized Sharpe of a per-period returns series, plus skew/kurtosis.

    Sharpe here is per-period (mean/std). Annualize separately if you report it;
    PSR/DSR operate on the per-period Sharpe and the sample length n.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    n = returns.size
    if n < 3:
        raise ValueError("need at least 3 return observations")
    mean = returns.mean()
    sd = returns.std(ddof=1)
    if sd == 0:
        raise ValueError("zero variance in returns")
    sr = mean / sd
    skew = float(stats.skew(returns, bias=False))
    # Fisher=False -> raw (non-excess) kurtosis, which the PSR formula expects.
    kurt = float(stats.kurtosis(returns, fisher=False, bias=False))
    return sr, skew, kurt, n


def probabilistic_sharpe_ratio(returns: Sequence[float], sr_benchmark: float = 0.0) -> float:
    """PSR: P(true per-period Sharpe > sr_benchmark) given sample & non-normality.

    Returns a probability in [0, 1]. sr_benchmark is in the SAME per-period units
    as the estimated Sharpe (default 0 = "is the strategy better than nothing").
    """
    sr, skew, kurt, n = _sharpe_and_moments(np.asarray(returns, dtype=float))
    # Standard error of the Sharpe estimator under non-normality.
    denom = math.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2)
    if denom <= 0:
        return float("nan")
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_variance_across_trials: float) -> float:
    """Expected maximum per-period Sharpe from N independent trials of noise.

    This is the benchmark the winner must beat: E[max Sharpe] under the null that
    all trials are zero-edge, given how much Sharpe estimates vary across trials.
    Uses the standard extreme-value approximation (Euler-Mascheroni gamma).
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    gamma = 0.5772156649015329  # Euler-Mascheroni
    e = math.e
    sigma = math.sqrt(sr_variance_across_trials)
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * e))
    return sigma * ((1.0 - gamma) * z1 + gamma * z2)


def deflated_sharpe_ratio(
    returns: Sequence[float],
    n_trials: int,
    sr_variance_across_trials: float,
) -> float:
    """DSR: PSR with the benchmark set to the expected max Sharpe of N trials.

    Parameters
    ----------
    returns : the OOS per-period returns of the WINNING strategy.
    n_trials : how many configurations you tried before selecting this one.
        Be honest — include every parameter set you swept, not just the finalists.
    sr_variance_across_trials : variance of the (per-period) Sharpe estimates
        across all trials. If you have the vector of trial Sharpes, pass its
        np.var(ddof=1). If you cannot reconstruct it, a conservative proxy is
        1/n (n = sample length), i.e. the sampling variance of a single Sharpe.

    Returns
    -------
    Probability in [0, 1]. Interpret like a p-value complement: > 0.95 means the
    winning OOS Sharpe survives the multiple-testing correction.
    """
    sr_benchmark = expected_max_sharpe(n_trials, sr_variance_across_trials)
    return probabilistic_sharpe_ratio(returns, sr_benchmark=sr_benchmark)


def _cli():
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Deflated / Probabilistic Sharpe Ratio")
    p.add_argument("csv", help="CSV/one-column file of per-period returns (no header)")
    p.add_argument("--trials", type=int, default=1, help="number of configs tried")
    p.add_argument("--sr-var", type=float, default=None,
                   help="variance of Sharpe across trials; default 1/n proxy")
    args = p.parse_args()
    r = np.loadtxt(args.csv)
    n = r.size
    sr_var = args.sr_var if args.sr_var is not None else 1.0 / n
    psr = probabilistic_sharpe_ratio(r)
    if args.trials > 1:
        dsr = deflated_sharpe_ratio(r, args.trials, sr_var)
    else:
        dsr = psr
    sr, skew, kurt, _ = _sharpe_and_moments(r)
    print(f"n={n}  per-period Sharpe={sr:.4f}  skew={skew:.3f}  kurtosis={kurt:.3f}")
    print(f"PSR (vs 0)            = {psr:.4f}")
    print(f"DSR (trials={args.trials}, sr_var={sr_var:.2e}) = {dsr:.4f}")
    if dsr < 0.95:
        print("VERDICT: does NOT survive multiple-testing correction (DSR < 0.95).")
        sys.exit(1)
    print("VERDICT: survives (DSR >= 0.95).")


if __name__ == "__main__":
    _cli()
