"""OOS trade counting must agree with how run_backtest books turnover costs.

Regression guard for a boundary-convention bug (found in adversarial review): the
per-fold / per-block OOS trade counter re-seeded a flat (0) position at each block
start, so a position held STEADY and nonzero across a block boundary was counted
as a fresh entry — even though the returns matrix charges NO turnover cost there.
The count must instead reflect the same |Δ held| on the global one-bar-delayed
position that the cost model charges. Only inflates the `trades_ok` floor gate, so
low severity, but it is a real inconsistency in the judge.
"""
import numpy as np

from harness.runner import _block_trades, _oos_turnover_count


def test_steady_position_across_boundary_is_not_a_phantom_trade():
    # +1 from the start, never changes -> exactly ONE turnover event (the entry at
    # the global first bar), and ZERO inside any later window.
    sig = np.ones(100)
    assert _block_trades(sig, 40, 60) == 0          # steady across the left boundary
    assert _block_trades(sig, 0, 20) == 1           # window contains the entry
    # the integer-index form run() uses on test_idx must agree
    assert _oos_turnover_count(sig, np.arange(40, 60)) == 0


def test_trade_count_equals_actual_position_changes():
    # held = shift-by-1 of sig; turnover fires on each real change of held.
    sig = np.array([0, 0, 1, 1, 1, -1, -1, 0, 0, 1.0])
    # held = [0,0,0,1,1,1,-1,-1,0,0] -> changes at held-index 3, 6, 8 => 3 trades
    assert _oos_turnover_count(sig, np.arange(len(sig))) == 3
