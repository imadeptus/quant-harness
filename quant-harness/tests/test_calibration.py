"""Regression guards for the judge's calibration (see harness/calibration.py).

These lock the detector's essential properties so a future refactor of the judge
cannot silently break them: noise is (almost) never passed, power rises with the
true edge, realistic cost kills a gross edge, multiple-testing deflates the DSR,
heavy tails do not manufacture a PASS, and a dead edge is not extrapolated.

Everything here is deterministic — every draw uses np.random.default_rng(fixed
seed) — so the thresholds are stable, not flaky. Studies are computed once at
module load at a low seed count for speed; the full-resolution numbers live in
reports/CALIBRATION.md.
"""
import numpy as np

from harness.calibration import (StudyGrid, SynthConfig, gen_ar1, gen_normal,
                                 gen_regime_shift, gen_student_t, mu_for_sharpe,
                                 study_cost, study_fat_tails, study_multiplicity,
                                 study_null, study_power, study_regime_shift,
                                 study_sample_size)

FAST = StudyGrid(cfg=SynthConfig(), n_seeds=30, with_pbo=False)

# Compute each study once (deterministic, fast without PBO).
NULL = study_null(FAST)
POWER = study_power(FAST)
COST = study_cost(FAST)
MULT = study_multiplicity(FAST)
FAT = study_fat_tails(FAST)
REGIME = study_regime_shift(FAST)
SAMPLE = study_sample_size(FAST)


def _by(rows, key, val):
    return next(r for r in rows if r.params[key] == val)


# ---- data-generating processes are well-formed --------------------------------

def test_dgp_generators_hit_target_vol():
    rng = np.random.default_rng(7)
    for gen, kw in [(gen_normal, {}), (gen_student_t, {"df": 3.0}),
                    (gen_ar1, {"phi": 0.4}), (gen_regime_shift, {"active_frac": 0.5})]:
        x = gen(40_000, np.random.default_rng(7), 0.01, mu=0.0, **kw)
        assert 0.009 < x.std() < 0.011, f"{gen.__name__} vol off: {x.std()}"


def test_mu_for_sharpe_yields_target_sharpe():
    vol = 0.01
    mu = mu_for_sharpe(3.0, vol)
    x = np.random.default_rng(0).normal(mu, vol, 200_000)
    ann_sharpe = x.mean() / x.std(ddof=1) * np.sqrt(365.0)
    assert abs(ann_sharpe - 3.0) < 0.2


# ---- the six calibration invariants ------------------------------------------

def test_null_false_positive_rate_is_low():
    # At N=1 the DSR reduces to the PSR (no deflation): a one-sided 0.95 test, so
    # its false-positive rate on noise sits at the nominal ~5%, and median DSR ~0.5
    # by symmetry. Both are correct calibration, not a leak.
    for r in NULL:
        assert r.pass_rate <= 0.06, f"FPR too high at N={r.params['n_configs']}: {r.pass_rate}"
    # Once there IS multiple testing (N>1) — which every real grid in this project
    # had — the deflation pulls noise well below 0.5 and the pass rate collapses.
    for r in [x for x in NULL if x.params["n_configs"] >= 6]:
        assert r.median_dsr < 0.5, f"DSR should deflate on noise at N={r.params['n_configs']}"
        assert r.pass_rate <= 0.02, f"real multiple-testing FPR ~0, got {r.pass_rate}"


def test_power_rises_with_edge_and_detects_strong_edge():
    assert _by(POWER, "true_ann_sharpe", 0.0).pass_rate <= 0.10   # no edge -> ~no pass
    assert _by(POWER, "true_ann_sharpe", 4.0).pass_rate >= 0.80   # strong edge -> detected
    # monotone in the large: stronger edge is passed at least as often.
    assert (_by(POWER, "true_ann_sharpe", 3.0).pass_rate
            >= _by(POWER, "true_ann_sharpe", 1.0).pass_rate)


def test_realistic_cost_kills_a_gross_edge():
    free = _by(COST, "cost_per_trade", 0.0).pass_rate
    dear = _by(COST, "cost_per_trade", 0.008).pass_rate
    assert free >= 0.5, "a gross ann-Sharpe-3 edge should pass when costless"
    assert dear <= 0.10, "high cost must kill it"
    assert dear < free


def test_multiple_testing_deflates_the_dsr():
    # Same true edge, wider search -> the deflated Sharpe must not increase, and
    # the pass rate must not increase, as N grows.
    dsr_by_n = [(_by(MULT, "n_configs", n).median_dsr,
                 _by(MULT, "n_configs", n).pass_rate) for n in (1, 6, 20, 50, 100)]
    dsrs = [d for d, _ in dsr_by_n]
    passes = [p for _, p in dsr_by_n]
    assert dsrs[0] >= dsrs[-1], f"DSR should deflate with N: {dsrs}"
    assert passes[0] >= passes[-1], f"pass rate should not rise with N: {passes}"


def test_fat_tails_do_not_manufacture_a_pass():
    for r in [x for x in FAT if x.label == "fat_tails_null"]:
        assert r.pass_rate <= 0.10, f"heavy tails faked a PASS at df={r.params['df']}: {r.pass_rate}"


def test_regime_shift_is_not_extrapolated():
    full = _by(REGIME, "active_frac", 1.0).pass_rate
    quarter = _by(REGIME, "active_frac", 0.25).pass_rate
    assert full >= quarter, "shrinking the alive era must not raise the pass rate"
    assert quarter < full, "a mostly-dead edge should pass less than an always-on one"


def test_more_data_gives_more_power():
    # At a fixed true edge, a longer series must be detected at least as often —
    # statistical power rises with sample size.
    at2 = [r for r in SAMPLE if r.params["true_ann_sharpe"] == 2.0]
    short = min(at2, key=lambda r: r.params["n_bars"]).pass_rate
    long = max(at2, key=lambda r: r.params["n_bars"]).pass_rate
    assert long >= short, f"more bars should not lower power: {short} -> {long}"
    assert long > short, "the sample-size effect should be visible at ann Sharpe 2"
