English | [Русский](README.ru.md)

# quant-harness

**An honest backtest framework.** Its job is not to "find a profitable
strategy" — it is to make sure **you cannot fool yourself** into believing you
found one. It issues a mechanical `PASS | KILL` verdict that cannot be spun as
a success after the fact.

Most backtests lie by default: train and evaluate on the same data,
look-ahead on the bar's close, costs "added later", and — the big one — a
silent search across hundreds of configurations, with only the best one
shown. `quant-harness` closes every one of those holes structurally.

```python
import numpy as np, pandas as pd
from harness import run_cpcv_returns, Thresholds, CPCVConfig

# 6 configs x 900 bars of pure noise -> the judge MUST return KILL
R = np.random.default_rng(0).normal(0.0, 0.01, (6, 900))
T = np.zeros_like(R); T[:, ::3] = 1                       # turnover every 3 bars
idx = pd.date_range("2024-01-01", periods=900, freq="1D", tz="UTC")
rep = run_cpcv_returns(R, T, idx, [{"c": i} for i in range(6)],
                       CPCVConfig(n_groups=10, k_test=2), Thresholds())
print(rep["verdict"])   # -> "KILL"
```

## Who this is for

Allocators, prop desks, bot marketplaces, and AI-agent builders who need an
**independent, mechanical verdict before capital is committed** to a
strategy — whether the strategy comes from a human researcher or is generated
by an LLM agent. Point it at a returns series (yours, a vendor's, or a
counterparty's) and get a pre-registered PASS/KILL you did not tune after
seeing the answer, either as a CLI/library call or as a hosted API an agent
can call directly.

## What this is not

Not a signal generator, not a source of alpha, not investment advice, and not
a promise of returns. It answers one narrow question: *is this edge visible on
this data, after realistic costs and a correction for how many configurations
were tried, at the pre-registered thresholds?* A KILL means "not visible here"
— not "no profit exists anywhere." A PASS is not a recommendation to deploy
capital.

## What it guarantees (the guardrails)

| Guardrail | How it's implemented | File |
|---|---|---|
| **No look-ahead** | bar `t` position = signal `t−1` (`held = target_pos.shift(1)`) | `harness/backtest.py` |
| **Honest OOS** | rolling/anchored walk-forward and Combinatorial Purged CV with purge+embargo; you only trust the concatenation of untouched test windows | `harness/walk_forward.py` |
| **Realistic costs** | taker fee + slippage on every turnover, funding as a periodic cash flow | `harness/backtest.py` |
| **Multiplicity correction** | Deflated Sharpe deflates the best Sharpe by the number of configurations tried N; PBO catches overfitting of the *search* itself | `harness/deflated_sharpe.py`, `harness/pbo.py` |
| **Mechanical verdict** | four pre-registered `Thresholds` -> PASS/KILL (`run.py` exits 2 on KILL; `qh-audit` exits 1 on KILL and 2 on invalid input) | `harness/runner.py`, `run.py` |

## The judge is measured, not "seems to work"

A verdict is only worth as much as the calibration behind the judge issuing
it. So the judge was run across the whole PASS/KILL response surface on
synthetic data with known ground truth (`harness/calibration.py`, report
`reports/CALIBRATION.md`, 200 seeds/cell):

- **False positives on noise (N≥6): 0.0%** — the judge does not pass noise.
- **Detection threshold: PASS≥50% at true annualised Sharpe ~2.2, ≥90% at
  ~3.0** — the honest bar a real strategy has to clear.
- **Costs** cut the passing gross edge exactly like real runs (net-kills-gross).
- **DSR deflation**: the same true edge, as N grows 1→100 → PASS 82%→16%.
- **Fat tails** (Student-t, df=3) and **autocorrelation** (φ up to 0.6) at
  zero edge **do not** break the FPR — the non-normality correction holds.
- **Regime shift**: an edge alive for only part of the sample is not
  extrapolated.

Full write-up: `reports/FINDINGS-CALIBRATION.md`. Regression guards:
`tests/test_calibration.py`, `tests/test_detection_power.py`. Complete
methodology (the seven ways a backtest lies, and the defence against each):
`docs/METHODOLOGY.md`.

## Install

```bash
pip install -e .                 # core: numpy, scipy, pandas, requests
pip install -e ".[fast]"         # + pyarrow (parquet cache for downloaded klines)
pip install -e ".[exchange]"     # + ccxt (extra venue loaders)
pip install -e ".[api]"          # + fastapi, uvicorn, pydantic (hosted verdict API)
pip install -e ".[dev]"          # + pytest, httpx
pip install -e ".[dev,api]"      # everything the full test suite needs (see Quick start)
```

Python >= 3.11. Current package version: **0.3.0**.

## Quick start

```bash
python examples/quickstart.py    # two demos, no data download
python -m pytest                 # 198 passed with the dev+api extras (154 passed, 44 API tests skipped with dev only)
qh-dsr returns.csv --trials 20   # DSR/PSR from a returns CSV (console utility)
qh-audit --returns returns.csv --trials 20   # PASS/KILL verdict + report (see below)
```

`examples/quickstart.py` shows both entry points: (A) the judge's verdict —
KILL on noise / PASS on a real edge; (B) the leakage-safe engine on a single
instrument (gross→net, turnover, drawdown).

Real data (free public Binance dumps):

```bash
python run.py --symbol BTCUSDT --interval 1h \
  --months 2024-01,2024-02,2024-03,2024-04,2024-05,2024-06
python run.py --synthetic --n-synth 5000            # offline sanity check -> KILL
python run_calibration.py --seeds 200                # rebuild the calibration report
```

## Strategy audit (`qh-audit`)

A CLI (and Python API) that turns a CSV of per-period returns into a
mechanical PASS/KILL verdict — the same judge as the hosted API, run
locally, no network call:

```bash
qh-audit --returns examples/audit_sample_returns.csv --trials 20 \
  --out audit_report.md --json audit_report.json
```

```
qh-audit: VERDICT KILL  (audit_sample_returns)
  data       : 730 periods x 4 configs, 2.00 years, 365 periods/year
  trials     : 20 (effective)
  OOS Sharpe : +1.066 median path, +0.461 worst path
  max DD     : 0.114   trades: 730
  PSR 0.9335   DSR 0.5885 (N=20)   PBO 0.6865
  checks     : trades_ok=yes oos_sharpe_ok=yes drawdown_ok=yes dsr_ok=NO
  ASSUMPTION: trades not provided; turnover assumed at 1 trade(s) per bar for every config
  report     : audit_report.md      json       : audit_report.json
  statistical report, not investment advice
```

Four pre-registered gates (same calibrated thresholds as the judge above):
minimum trades (200), minimum OOS Sharpe (0.7), maximum drawdown (0.20),
minimum Deflated Sharpe (0.95). PBO is reported as an informational
diagnostic and never changes the verdict. Costs (`--costs-bps`), a
cost-sensitivity table at 0x/0.5x/1x/2x, custom CPCV geometry, and threshold
overrides are all supported — run `qh-audit --help` for the full flag list.
Exit codes: `0` PASS, `1` KILL, `2` invalid input or unwritable `--out`/`--json`,
`3` internal error — a crash never exits `1`, so it can never be read as a KILL.

```bash
python examples/audit_quickstart.py   # generates examples/audit_sample_returns.csv (~1s, no network)
```

Python API: `from harness import audit_returns, render_markdown, AuditInputError`.

## Hosted verdict API

A thin FastAPI service around the exact same judge (`harness.audit.audit_returns`)
— upload a returns matrix over HTTP, get a calibrated verdict back. Built so
an AI agent (or a human) can request an independent PASS/KILL without
installing anything.

```bash
pip install -e ".[api]"
uvicorn api.app:app --host 0.0.0.0 --port 8000 --no-access-log   # or: make api
# interactive docs at http://localhost:8000/docs
```

```bash
docker build -t qh-api:dev .          # or: make docker
docker run --rm -p 8000:8000 --env-file api/.env.example qh-api:dev
```

`GET /healthz` is always open. `POST /v1/verdict` takes a returns matrix (and
optional trades, frequency, thresholds, CPCV geometry, and costs) and returns
verdict, per-gate checks, metrics (OOS Sharpe, DSR, PBO, drawdown), a
cost-sensitivity table, assumptions/warnings, a short Markdown report, and a
disclaimer. Optional API-key auth (`X-API-Key`) and a payment gate
(`QH_PAYMENT_GATE`: `noop` free / `x402` / `nowpayments` — **the paid gates are
stubs that issue the payment challenge but do not verify payment**; read
`api/README.md#payments-what-is-real-and-what-is-a-stub` before relying on
either in production). Full endpoint reference, error codes, and environment
variables: `api/README.md`.

## Public API

```python
from harness import (
    run, run_cpcv, run_cpcv_returns, Thresholds,   # judge / verdict
    run_backtest, Costs, max_drawdown,             # engine
    WalkForwardConfig, CPCVConfig,                 # splitters
    deflated_sharpe_ratio, probabilistic_sharpe_ratio, pbo_cscv,  # statistics
    build_signal, grid_for, all_families,          # signal families
    audit_returns, render_markdown, AuditInputError,  # qh-audit
)
```

- `run(df, grid, wf, costs, thr)` — walk-forward judge on a single instrument.
- `run_cpcv(df, grid, cpcv, costs, thr)` — CPCV variant, verdict on the median path.
- `run_cpcv_returns(R, trades, index, grid, cpcv, thr)` — judge on a
  ready-made returns matrix (for portfolio/cross-sectional strategies).
- `audit_returns(R, ...)` — the same judge behind `qh-audit` and the hosted API.
- `build_signal(df, params)` — momentum / mean_reversion / breakout / carry families.

## Layout

```
quant-harness/
├── pyproject.toml            # package + dependencies + console scripts
├── api/                      # hosted verdict API (FastAPI; not part of the wheel)
├── run.py, run_*.py          # research runners (per-spec runs)
├── run_calibration.py        # judge calibration -> reports/CALIBRATION.{json,md}
├── examples/                 # quickstart.py, audit_quickstart.py — data-free demos
├── harness/
│   ├── __init__.py           # public API
│   ├── backtest.py           # engine: anti-look-ahead, costs, funding, max_drawdown
│   ├── walk_forward.py       # leakage-safe WFA + CPCV (purge/embargo)
│   ├── deflated_sharpe.py    # DSR / PSR (Bailey & López de Prado) + qh-dsr CLI
│   ├── pbo.py                # Probability of Backtest Overfitting (CSCV)
│   ├── runner.py              # assembly: WFA/CPCV -> OOS -> DSR -> verdict
│   ├── audit.py               # qh-audit: CSV in, PASS/KILL report out
│   ├── families.py           # parametric signal families (grid = N)
│   ├── calibration.py        # detection-power study
│   ├── data.py                # Binance loader + synthetic fallback (offline)
│   └── {bybit,hyperliquid,basis,listing,cascade,harvest,paper}.py  # venue/strategy modules
├── reports/                  # JSON+MD reports and FINDINGS-*
└── tests/                    # pytest (198 passed)
```

## What has already been tested with this framework

Full registry: `../RESEARCH-CONCLUSION.md`. In short: 10 pre-registered specs
across 3 venues (Binance/Bybit/Hyperliquid) — **all KILL** on public data; no
tested mechanism shows a tradeable edge net of realistic costs. The one
positive cash flow found was funding-harvest on an inefficient venue — a
**research result**, not a live product: a small live paper track (27 ticks
over 42 calendar days) currently sits at −1.5% (about −13%/yr annualised),
consistent with the strategy being a tail-risk carry trade with real basis
risk, not free money. See `../docs/HARVEST-PATH-A.md` for the position,
`../CASE-STUDY-EN.md` for the write-up of the project as a whole, and
`../site/index.html` for the public landing page. Calibration
(above) shows this is a conclusion **about the markets tested**, not a judge
artefact: the best observed net (0.88) sits well below the measured detection
threshold (~2.2).

## Boundaries of honesty (what this is and is NOT)

- **It is** an edge detector and an anti-self-deception device. It says "not
  visible on this data with these costs" — not "no profit exists in the
  universe."
- **It is not** an execution engine, a live trader, or a source of alpha.
- `Thresholds` are **pre-registered**. Turning the dials after the fact to
  manufacture a PASS is exactly the overfitting the whole framework exists to
  prevent — don't do it.
- The calibration study uses synthetic, independent configs (the adversarial,
  worst case for DSR). Real grids have correlated configs, so real deflation
  is milder.

## Relationship to the skill

`deflated_sharpe.py` and `walk_forward.py` are the canonical implementations
from the `quant-backtest-guardrails` skill, vendored here for standalone use.
The skill holds the methodology; the harness is the working substrate under it.

---

MIT License.
