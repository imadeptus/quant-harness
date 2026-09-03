English | [Русский](README.ru.md)

# quant-ai-lab

A research lab built around one question: **is there a tradeable edge in
crypto perpetuals that survives an honest check?** Not "build a bot" — get a
conclusion that cannot be faked by fooling yourself.

**The answer is in.** 10 pre-registered hypotheses × 3 venues (Binance, Bybit,
Hyperliquid), time-series / cross-sectional / event-driven signals — **all
KILL**. No predictive alpha is visible on public data. The one positive cash
flow found is harvesting funding on an inefficient venue (~7%/yr realistic in
backtest), and that is tail-risk management, not alpha. This is a real
conclusion, not a dead end.

And — more important for trusting that conclusion — **the method is
measured**: the judge issuing every verdict was run across the whole decision
surface on synthetic data with known ground truth. False-positive rate on
noise: 0%. Detection threshold: ~2.2 annualised Sharpe. The best net Sharpe
this project ever observed (0.88) sits at less than half that threshold — so
the KILL stream is evidence *about the markets*, not a stuck detector.

That detector — and the pre-registration discipline behind it — is what this
repo sells: a mechanical, calibrated PASS/KILL verdict, as a CLI, a Python
library, or a hosted API, for anyone who needs an independent check on a
strategy's returns before committing capital. See **What's for sale**, below.

## Navigation

| Document | What's there |
|---|---|
| [`RESEARCH-CONCLUSION.md`](RESEARCH-CONCLUSION.md) | Research-phase conclusion: the registry of all 10 hypotheses, what it means, the decision |
| [`quant-harness/`](quant-harness/) | **The tool** — an honest backtest framework (pip-installable, 198 tests). Start with its [README](quant-harness/README.md) |
| [`quant-harness/docs/METHODOLOGY.md`](quant-harness/docs/METHODOLOGY.md) | Methodology: the seven ways a backtest lies, and the defence against each |
| [`quant-harness/reports/CALIBRATION.md`](quant-harness/reports/CALIBRATION.md) | Judge calibration — the measured detection power (FPR, power curve, robustness) |
| [`quant-harness/reports/FINDINGS-CALIBRATION.md`](quant-harness/reports/FINDINGS-CALIBRATION.md) | Calibration write-up and why the KILL record can be trusted |
| [`quant-harness/api/README.md`](quant-harness/api/README.md) | Hosted verdict API — endpoints, auth, payment gates, environment variables |
| [`docs/specs/`](docs/specs/) | Hypothesis pre-registrations (specs written before touching data) |
| [`docs/HARVEST-PATH-A.md`](docs/HARVEST-PATH-A.md) | Funding-harvest: real basis risk, costs, venue tail — a research position, not a live product |
| [`CASE-STUDY-EN.md`](CASE-STUDY-EN.md) | Public write-up of the project: what was built, what was found, why it's trustworthy |
| [`site/index.html`](site/index.html) | Landing page for the tool and the case study |
| `NEXT_SESSION.md` | Current status and open decisions for the owner (internal working file — not part of the public snapshot) |

## Quick start

```bash
cd quant-harness
pip install -e ".[dev,api]"
python -m pytest                  # 198 tests (dev+api extras), including calibration guards
python examples/quickstart.py     # KILL on noise / PASS on a real edge, no data needed
qh-audit --returns examples/audit_sample_returns.csv --trials 20   # PASS/KILL verdict on a CSV
```

## What's for sale

The output of this lab is not a trading signal — it's the **verdict tool**
built to check one honestly. Two ways to use it:

- **`qh-audit`** — a CLI and Python library that turns a CSV of per-period
  returns into a mechanical PASS/KILL verdict (Combinatorial Purged CV,
  Deflated Sharpe, PBO, four pre-registered gates), plus a Markdown/JSON
  report. Runs locally, no network call.
- **Hosted verdict API** (`quant-harness/api/`) — the same judge behind a
  FastAPI service (`POST /v1/verdict`): upload a returns matrix, get the
  verdict, metrics, cost sensitivity, and a short report back over HTTP. Built
  so an AI agent can request an independent check as easily as a human can.

**Who it's for:** allocators, prop desks, bot marketplaces, and AI-agent
builders who need an independent, calibrated verdict on a strategy's returns
— their own, a vendor's, or a counterparty's — before capital is committed.

**What it is not:** not a signal generator, not trading advice, not a
promise of returns. A KILL means "not visible on this data, at these costs,
after correcting for how many configurations were tried" — not "no profit
exists anywhere." See `quant-harness/README.md` for the full picture,
including the boundaries of what a PASS does and doesn't mean.

## Why this is worth trusting

1. **Discipline carried through to a conclusion.** 10 honest pre-registrations,
   none "pushed" to a PASS. Most retail backtests break on the first of seven
   common ways to fool yourself; all seven are closed here — structurally.
2. **A mature tool.** `quant-harness` is a reusable honest-backtest framework:
   leakage-safe WFA/CPCV, realistic costs, Deflated Sharpe, PBO, a mechanical
   verdict — with proven, *measured* detection power (see Navigation, above).
3. **An honest negative result is still a result.** "No edge visible on this
   data at these costs" is knowledge that saves capital — and it's stated as
   a KILL, not spun as a near-miss.
4. **The funding-harvest research result is disclosed honestly.** A live
   paper track (started 2026-07-22, cron/launchd-driven, no capital at risk)
   currently sits negative — about −1.5% over six weeks, roughly −13%/yr
   annualised — against a ~7%/yr realistic backtest estimate. Both numbers
   are shown; neither is advertised as a return you should expect.

---

*Stack: Python 3.11+, numpy/scipy/pandas, FastAPI for the hosted API. Data:
public Binance/Bybit dumps, the Hyperliquid API. Tool license: MIT. Repository:
https://github.com/imadeptus/quant-harness.*
