English | [Русский](FINDINGS-CALIBRATION.ru.md)

# FINDINGS — Calibration of the judge (detection-power study)

**Status:** realised · 2026-07-24 · engine `harness/calibration.py`, report
`reports/CALIBRATION.md` (regenerate: `python run_calibration.py --seeds 200`).

## Why this exists

Every verdict this project ever produced was a KILL (10 specs × 3 venues,
KILL-1..8). That is only trustworthy if the judge is a *calibrated detector* and
not a stuck null-machine. `test_detection_power.py` already proved the judge
*can* PASS a great strategy and *can* KILL noise on two hand-picked points. This
study maps the **whole response surface** on synthetic data with known ground
truth, so each KILL is now readable against a *measured* false-positive rate and
a *measured* detection threshold — not a hope that the judge "probably works".

Method: feed the CPCV judge (`run_cpcv_returns`) synthetic (N configs × T bars)
matrices from known data-generating processes; each config is an **independent**
draw from the same DGP (the adversarial choice — independence maximises in-sample
selection luck, the hardest case for the Deflated Sharpe). 200 Monte-Carlo seeds
per cell, 912 daily bars, per-bar vol 0.01, CPCV(10, k=2, purge=1, embargo=5),
pre-registered `Thresholds()`. The trade-count gate is deliberately satisfied so
the studies isolate the *statistical* detector (Sharpe / DSR / drawdown).

## Headline results (200 seeds)

| property | result | reading |
|---|---|---|
| **False positive rate** (noise, N=6) | **0.0%** PASS, median DSR 0.07 | judge does not pass noise |
| **Detection threshold** | PASS≥50% at true ann Sharpe **~2.2**, ≥90% at **~3.0** | the honest bar a real edge must clear |
| **Cost** | gross ann-Sharpe-3 edge holds PASS≥50% to ~0.001/trade-bar; dead by 0.004 | net-kills-gross, exactly the project pattern |
| **Multiple testing** | true ann Sharpe 2: PASS 82%→16% as N goes 1→100; median DSR 0.999→0.510 | deflation actively protecting |
| **Fat tails** (df=3, zero edge) | 0.0% FPR; power at real edge survives (86–91%) | PSR/DSR non-normality correction holds |
| **Autocorrelation** (φ up to 0.6, zero edge) | FPR stays ≤1% | feared failure mode did **not** materialise |
| **Regime shift** (edge alive part of sample) | PASS 91% (100% alive) → 6% (25% alive) | judge does not extrapolate a dead edge |
| **Sample size** (true ann Sharpe 2) | PASS 10% at 250 bars → 77% at 1825 bars | power scales with data length; even Sharpe-2 needs years of daily data |

Full power curve (PASS rate vs true annualised Sharpe):

```
ann Sharpe  0.0  0.5  1.0  1.5  2.0  2.5  3.0  3.5  4.0
PASS rate    0%  1.5% 8.5% 30%  46%  81%  97%  99% 100%
```

## The one inference that matters

The best out-of-sample result this project ever produced was **net Sharpe ~0.88**
(spec 0002 `xs_momentum`, itself a KILL). The measured 50%-detection threshold is
**~2.2**. At an ann Sharpe of ~0.9 the judge PASSes roughly **5–8%** of the time.

So the entire KILL record sits far below the detector's floor: the observed edges
are consistent with **"no edge of detectable magnitude"**, not with a judge that
refuses to say PASS. The detection threshold (~2.2) matches the pre-registered
intuition recorded in `NEXT_SESSION.md` ("Планка PASS: реальный OOS Sharpe ~2.2+")
— which was written before this study existed. Independent confirmation.

## Two findings worth keeping

1. **Autocorrelation is handled, and here is *why*.** Positive serial correlation
   inflates a naive Sharpe by understating the variance of the mean — the classic
   backtest lie. It did not fool this judge (FPR ≤1% at φ=0.6). The mechanism is
   visible in the table: median drawdown *rises* with φ (0.31 → 0.49), so the
   drawdown gate becomes a second line of defence exactly when the Sharpe gate is
   under the most pressure. The CPCV purge/embargo and the DSR both contribute.

2. **At N=1 the judge is a nominal ~5%-level test.** With a single config there is
   no multiple testing, so DSR collapses to the PSR — a one-sided 0.95 test whose
   false-positive rate on noise is the nominal ~5% and whose median DSR sits at
   ~0.5 by symmetry. This is correct, not a leak. Every real grid in this project
   had N≫1, which drove the effective FPR to ~0 (see the null table: 0% at N≥6).
   Takeaway: a **single** pre-registered config is the weakest configuration of
   this judge; prefer a small honest grid so the deflation can work.

## Limitations (honest)

- **Synthetic, i.i.d.-config model.** Configs are independent draws from one DGP.
  Real strategy grids have *correlated* configs (neighbouring parameters share
  returns), which reduces the effective number of trials — so the real deflation
  is, if anything, milder than modelled here. This study bounds the adversarial
  case, not the typical one.
- **PBO is muted by design.** All configs share one edge, so there is no "wrong
  config to overfit onto"; PBO hovers at ~0.5 throughout and is reported only as
  a secondary diagnostic. PBO discriminates when configs differ in true quality —
  which is exactly the real-grid case the point above describes.
- **Thresholds are the pre-registered ones.** This study characterises the judge
  *as configured*; it does not re-tune the thresholds (doing so post-hoc would be
  the overfitting this whole harness exists to prevent).

## Bottom line

The judge is a calibrated detector: ~0% false positives under multiple testing,
power rising smoothly to a ~2.2 detection threshold, robust to fat tails, serial
correlation, and regime instability, and correctly cost-sensitive. The project's
uniform KILL record is therefore **evidence about the markets tested, not an
artefact of a broken judge.** The crown-jewel claim — "the method is valid" — now
rests on a measured response surface, reproducible with one command.
