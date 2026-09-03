# I built a backtester designed to prove me wrong. It did — 10 times.

*An engineering case study: how I tested trading hypotheses on crypto with a method built not to fool me, found no edge at all — and why that is a result, not a failure.*

---

**TL;DR** (for skimmers). I built a research harness designed so that you **cannot fool yourself** with a pretty backtest: leakage-safe execution, walk-forward / CPCV, Deflated Sharpe corrected for the number of trials, realistic costs, hypotheses pre-registered before touching data. I ran 10 pre-registered hypotheses across three exchanges — **every one was rejected**. Then I did the part almost nobody does: I **measured the detector itself** — proving my "rejection machine" wasn't just stuck on reject, but actually distinguishes signal from noise (0% false positives on noise, detection threshold ~2.2 annualized Sharpe). Stack: Python, numpy/scipy/pandas, 76 tests, CI, pip package. Repo: https://github.com/imadeptus/quant-harness.

---

## The problem: a backtest lies by default

Almost anyone can show a backtest with a pretty Sharpe. The trouble is that the most natural way to write one bakes the lie in structurally — training and scoring on the same data, look-ahead at the bar close, "costs later," and, above all, silently trying hundreds of configurations and showing the best.

I didn't want another "profitable bot." I wanted a system that **couldn't lie to me** — and an honest answer to "is there any edge here at all?"

## What I built

A research harness where every common lie is closed by construction:

- **No look-ahead.** The position held during bar `t` is the signal from bar `t−1`. A strategy physically cannot peek at the future.
- **Honest out-of-sample.** Walk-forward and Combinatorial Purged CV with purge/embargo. The only equity curve you may believe is the concatenation of untouched test windows.
- **Realistic costs.** Taker + slippage on every turnover, funding as a cash flow — on every run, never optional.
- **Multiple-testing correction.** The Deflated Sharpe Ratio deflates the winner by the number of configs tried; PBO catches whether the *search itself* overfits.
- **A mechanical verdict.** Four pre-registered thresholds → PASS/KILL. A result can't be talked up after the fact.

This is an engineering problem about data discipline and correctness, not about trading.

## The hard part — trusting the tool itself

Here's the part that separates this from a pet project. My harness kept returning nothing but KILLs. That raises an honest question: **is it stuck? Maybe it never passes anything.** A reject-everything machine would produce the exact same stream of rejections — and be useless.

So I **measured the detector** on synthetic data with known ground truth — mapping its entire decision surface (200 Monte-Carlo runs per cell):

- **False positives on pure noise: 0%.** The judge does not pass noise.
- **Power curve** (PASS rate vs. true annualized Sharpe):

  ```
  Sharpe   0.0  0.5  1.0  1.5  2.0  2.5  3.0  3.5  4.0
  PASS      0%  1.5% 8.5% 30%  46%  81%  97%  99% 100%
  ```

  **Detection threshold ~2.2** — the honest bar a real strategy must clear to be noticed.
- Robust to what inflates a naive Sharpe: fat tails, autocorrelation, an edge that only lived for part of the sample.
- I also measured how much data you need: even a true Sharpe-2 edge is caught only 10% of the time on 250 bars — a short backtest doesn't "prove there's no edge," it's simply blind.

Separately, I ran an adversarial correctness review of the statistical core: the Probabilistic Sharpe Ratio implementation matched the Bailey & López de Prado formula to machine precision; I found and fixed one bug in the turnover count.

## The result

**10 pre-registered hypotheses, 3 exchanges (Binance, Bybit, Hyperliquid), all KILL.** Time-series, cross-sectional, and event-driven signals. No tradeable predictive edge on public data. The closest to passing was, after the multiple-testing correction, indistinguishable from the best of noise.

And that is the **correct answer, not a dead end.** The best observed result (Sharpe 0.88) sits at half the measured detection threshold. So the stream of KILLs is evidence about the markets, not my broken tool. I know this because I measured the tool.

## What this project demonstrates

Strip out the trading, and what remains is engineering I can show:

- **Building with AI — and building the check on AI** — pointing AI tools at a non-trivial system while owning the validation that decides which output to trust and which to reject.
- **Systems thinking** — designing a pipeline so an entire class of error is impossible by construction, not caught in review.
- **Statistical rigor** — CPCV, Deflated Sharpe, detector calibration, handling multiple testing and non-normality.
- **Intellectual honesty and discipline** — 10 pre-registrations, none bent toward the desired answer; willingness to kill my own work.
- **Knowing when to stop** — recognizing that further search was statistically against me, and stopping. Rarer than it sounds.

For a hiring manager, the last two matter more than a "profitable bot": this is someone who builds systems you can trust, and won't fool themselves for a nice number.

## Tech stack

Python 3.11+, numpy / scipy / pandas. Combinatorial Purged CV, Deflated Sharpe Ratio (Bailey & López de Prado), PBO. Leakage-safe engine, data cache, loaders for three exchanges. 76 tests (including calibration and a positive control for the detector), CI on 3.11–3.13, pip-installable package.

Full methodology — `docs/METHODOLOGY.md`. Calibration report — `reports/CALIBRATION.md`. Everything reproduces with one command.

## Honest about the process

This is an **AI-assisted project**. I designed the research question, the methodology, the pre-registration discipline, and — above all — the **validation that lets me trust a conclusion or reject it**; I directed the execution and made every verdict myself.

Today anyone can generate code or a strategy with AI in minutes. The scarce skill is **knowing which output to trust**. I built the system that measures exactly that — and used it to reject 10 of my own AI-assisted hypotheses. Orchestrating AI rigorously and verifying its output rather than taking it on faith is precisely what's shown here. That's why the heart of the project isn't a "strategy" — it's measuring my own detector: you can only trust what you've checked.

---

**Open to AI builder roles** — designing, orchestrating, and validating AI pipelines. Repository: https://github.com/imadeptus/quant-harness. Contact: [LinkedIn](https://www.linkedin.com/in/%D1%8F%D1%80%D0%BE%D1%81%D0%BB%D0%B0%D0%B2-%D1%82%D0%BE%D0%BA%D0%B0%D1%80%D0%B5%D0%B2-a76629244/).

*An honest backtest doesn't promise profit. It promises that if there's no profit, you'll know — instead of convincing yourself otherwise. I prefer systems that tell me the truth.*
