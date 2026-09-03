English | [Русский](METHODOLOGY.ru.md)

# How to run an honest backtest (and why almost none are)

Methodology behind `quant-harness`. Not *what we found* (that lives in
`../../RESEARCH-CONCLUSION.md`) — this is **how not to fool yourself**, and,
more importantly, *proof* the method has teeth rather than a list of good
intentions.

Thesis: a backtest lies by default. Not because the author is cheating, but
because the most natural way to write one bakes the lie in structurally. Below
are seven ways it lies, each with its defence, then the measured evidence that
the defences hold.

---

## Seven ways a backtest lies

| # | Lie | What it looks like | Defence in the harness |
|---|---|---|---|
| 1 | **Look-ahead** | the signal is computed on a bar's close and "executed" on the same bar | bar `t` position = signal `t−1`: `held = target_pos.shift(1)` in `backtest.py` — structural, no signal family can peek forward |
| 2 | **Evaluated on training data** | parameters are fit and scored on the same sample | walk-forward / CPCV: fit on train, score on untouched test; you only trust the concatenation of test windows (`walk_forward.py`) |
| 3 | **Boundary leakage** | a lookback feature peeks into the test set right at the seam | purge (train labels never reach into test) + embargo (a feature never peeks just after) in `CPCVConfig` |
| 4 | **Costs "added later"** | fees/slippage/funding are ignored or lowballed | taker fee + slippage on every turnover, funding as a periodic cash flow — on EVERY run, not optional (`Costs`) |
| 5 | **Silent search** | 200 configurations were tried, only the best is shown | Deflated Sharpe deflates by the number of trials N; PBO catches overfitting of the *search itself* (`deflated_sharpe.py`, `pbo.py`) |
| 6 | **Non-normality and autocorrelation** | fat tails and serial correlation inflate the Sharpe | PSR/DSR correct for skew/kurtosis and sample length; CPCV purge/embargo damps autocorrelation |
| 7 | **Survivorship** | delisted symbols silently vanish from the universe | point-in-time universe from the full symbol listing (`data.py`/specs — 33% of symbols are delisted) |

Every defence closes a specific lie. But a stack of defences is worth nothing
if the **judge itself is not calibrated**. That is the point that separates an
honest harness from a nice-sounding methodology.

---

## Why the method can be trusted: the judge is measured

Any `PASS/KILL` verdict is only worth as much as the calibration behind the
judge that issues it. A null-machine stuck on `KILL` would produce the same
stream of refusals as a real detector — and be useless. So the judge was run
across the **whole decision surface** on synthetic data with known ground
truth (`harness/calibration.py`, report `reports/CALIBRATION.md`, 200
Monte-Carlo seeds per cell, 4 data-generating processes).

What was measured:

- **False positives on pure noise (N≥6 configs): 0.0%.** The judge does not
  pass noise. At N=1 it degenerates to the PSR — a one-sided ~5%-level test;
  every real grid in this project had N≫1, which drives the effective FPR to
  zero.
- **Power curve** (PASS rate vs. true annualised Sharpe):

  ```
  Sharpe   0.0  0.5  1.0  1.5  2.0  2.5  3.0  3.5  4.0
  PASS      0%  1.5% 8.5% 30%  46%  81%  97%  99% 100%
  ```

  **Detection threshold ~2.2** (PASS≥50%), confidently ≥90% at Sharpe ~3.0.
  That is the honest bar a real strategy must clear to be noticed.
- **Costs** cut the passing gross edge exactly like real runs do: gross
  Sharpe 3 holds PASS≥50% to ~0.001/trade-bar and is dead by 0.004
  (net-kills-gross).
- **DSR deflation works**: the same true edge (Sharpe 2), as N grows from 1
  to 100 → PASS 82%→16%, median DSR 0.999→0.510.
- **Fat tails** (Student-t, df=3) at zero edge → 0% false PASSes; power on a
  real edge is preserved. The non-normality correction holds.
- **Autocorrelation** (φ up to 0.6) at zero edge → FPR ≤1%. The feared
  failure mode of a Sharpe-based judge did **not** materialise: median
  drawdown *rises* with φ (0.31→0.49), so the drawdown gate becomes a second
  line of defence exactly where the Sharpe gate is under pressure — the
  mechanism is visible in the table, not asserted.
- **Regime shift**: an edge alive for the full sample → PASS 91%; alive for
  25% of it → PASS 6%. The judge does not extrapolate a dead edge.
- **Sample size**: at true Sharpe 2, detection rises 10%→77% as the series
  grows from 250 to 1825 bars; at Sharpe 3, 32%→95%. Power scales with
  sample length — even a real edge is invisible in a short backtest.

Bottom line: the judge is a **calibrated detector** — near-zero FPR under
multiple testing, a smooth power curve up to a ~2.2 threshold, robust to fat
tails, autocorrelation, and regime instability. Full write-up:
`reports/FINDINGS-CALIBRATION.md`; regression guards:
`tests/test_calibration.py`, `tests/test_detection_power.py`.

---

## Discipline in practice: 10 pre-registrations, all KILL

The method was not proven in a vacuum. **10 pre-registered hypotheses**
(specs written in `docs/specs` *before* touching data) went through the
harness across three venues (Binance, Bybit, Hyperliquid), on time-series,
cross-sectional, and event-driven signals. **All — KILL.** None was
massaged into a PASS by running it again differently.

The closest to tradeable — post-shock continuation on Binance: net Sharpe
0.75, gross 0.94, the strongest pulse of the entire phase. Still KILL: after
correcting for multiple testing, DSR 0.57 — indistinguishable from the best
of pure noise. That is exactly how it should work — **the best observed net
(0.88) sits at less than half the measured detection threshold (~2.2).** The
uniform KILL stream is evidence *about the markets*, not a stuck detector.
Full registry: `../../RESEARCH-CONCLUSION.md`.

---

## Practical lessons

1. **Pre-register thresholds before touching data.** `Thresholds` are fixed.
   Nudging a threshold after the fact to manufacture a PASS is exactly the
   overfitting this whole harness exists to prevent. If a result makes you
   want to "just move the bar a little" — that is the bar working.

2. **A small honest grid beats a single config.** Counter-intuitively, N=1
   is the judge's *weakest* configuration (DSR degenerates to the PSR, FPR
   ~5%). A modest grid lets the deflation work and drives FPR toward zero —
   but the grid must be honest: every config tried counts toward N, not just
   the finalists.

3. **~2.2 is a sanity anchor, and it needs a lot of data.** A backtest
   showing an annualised Sharpe of 1.5 "after costs" sits below the
   50%-detection threshold — statistically indistinguishable from luck on
   this data. A reasonable bar for a real edge is OOS Sharpe ~3+, or it is
   simply invisible through the noise. And even a genuine Sharpe-2 edge is
   caught only 10% of the time at 250 bars — a short backtest doesn't "prove
   no edge", it is just blind (see sample-size, above).

4. **Costs kill most "edge".** In calibration, a gross Sharpe-3 edge holds
   PASS≥50% only up to ~10 bps per turnover (0.001), drops below 50% by 20 bps
   and is dead by ~40 bps (0.004). In real runs the same pattern showed up again and again:
   gross positive, net killed by realistic costs. Compute net from day one.

5. **Meta-multiplicity is real; stopping is itself a result.** Ten attempts
   is multiple testing at the *project* level. DSR corrects for configs
   within a spec, not across specs. More specs raise the odds of a lucky
   false PASS somewhere. Knowing when to stop searching is part of the
   method, not a concession.

---

## Reproduce

```bash
pip install -e ".[dev]"
python -m pytest                        # full suite, including calibration guards
python run_calibration.py --seeds 200   # rebuild reports/CALIBRATION.md
python examples/quickstart.py           # KILL on noise / PASS on edge, no data needed
```

Everything is deterministic (fixed seeds) — the numbers above reproduce to
the digit.

---

*An honest backtest does not promise profit. It promises that if there is no
profit, you will find out — instead of talking yourself into believing
otherwise. The only conclusion it protects from tampering is the true one.*
