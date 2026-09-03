"""max_drawdown must be a well-defined drawdown: bounded in [-1, 0] for any input.

Regression guard for the synthetic-overflow bug (edge 0.6 -> MDD ~-6.4e10): a
compounded-equity drawdown can never be worse than total ruin (-100%). On sane,
realistic returns the value is unchanged; on ruinous / heavy synthetic returns
it must saturate at -1.0 instead of overflowing via (1+r).cumprod() sign-flips.
"""
import numpy as np
import pandas as pd

from harness.backtest import max_drawdown


def _naive(returns: pd.Series) -> float:
    """The old, non-robust definition — correct only while equity stays > 0."""
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def test_bounded_on_realistic_returns_and_matches_naive():
    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(0.0003, 0.01, 912))  # realistic per-bar returns
    mdd = max_drawdown(r)
    assert -1.0 <= mdd <= 0.0
    # On sane data the robust and naive formulas must agree to float precision.
    assert abs(mdd - _naive(r)) < 1e-9


def test_saturates_at_full_loss_on_ruin_bar():
    # A bar with net return <= -100% wipes the account -> drawdown is exactly
    # -100%, never worse; the old cumprod sign-flip returned -1.416 here.
    r = pd.Series([0.02, -1.4, 0.03, 0.01])
    assert max_drawdown(r) == -1.0


def test_bounded_on_strong_edge_synthetic():
    # The exact regime that produced MDD ~ -6.4e10 via float overflow.
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.6, 1.0, 912))
    mdd = max_drawdown(r)
    assert -1.0 <= mdd <= 0.0


def test_monotone_up_has_zero_drawdown():
    r = pd.Series([0.01] * 100)
    assert max_drawdown(r) == 0.0


def test_known_drawdown_value():
    # equity: 1.5 then 0.75 -> peak 1.5, trough 0.75 => -50% drawdown.
    r = pd.Series([0.5, -0.5])
    assert abs(max_drawdown(r) - (-0.5)) < 1e-12


def test_empty_and_all_nan_are_safe():
    assert max_drawdown(pd.Series([], dtype=float)) == 0.0
    assert max_drawdown(pd.Series([np.nan, np.nan])) == 0.0
