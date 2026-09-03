# Changelog

All notable changes to `quant-harness` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-09-02

### Added
- `qh-audit` CLI (`harness/audit.py`, `python -m harness.audit` / `qh-audit`): a
  mechanical PASS/KILL verdict on a CSV of per-period returns via Combinatorial
  Purged CV, trial-corrected Deflated Sharpe, PBO (informational), a drawdown gate
  and a trade-count gate — the same calibrated judge as `reports/CALIBRATION.md`.
  Markdown and JSON report output, optional cost sensitivity table, `--trials`,
  `--freq`, and threshold overrides. Python API: `harness.audit_returns`,
  `harness.render_markdown`, `harness.AuditInputError`.
- Hosted verdict API skeleton (`api/`, FastAPI): `GET /healthz`, `POST /v1/verdict`
  running the same `audit_returns` judge as `qh-audit` (parity enforced by tests).
  Optional API-key auth (`X-API-Key`), and payment-gate stubs (`x402`,
  `nowpayments`) that issue the 402 challenge but do not verify payment yet — see
  `api/README.md#payments-what-is-real-and-what-is-a-stub`. JSON request logging on
  stdout. `Dockerfile` (multi-stage, non-root, `HEALTHCHECK`) and `.dockerignore`
  for a container build. New `api` optional dependency (`fastapi`,
  `uvicorn[standard]`, `pydantic`).
- `examples/audit_quickstart.py`: data-free example that audits three synthetic
  submissions (noise, weak edge, strong edge) with `qh-audit`'s Python API and
  writes the bundled `examples/audit_sample_returns.csv` sample.
- English documentation for the new public surface: `api/README.md`,
  `docs/OPS-PAPER-TRACKER.md`.
- Release tooling: `CHANGELOG.md` (this file), `.github/workflows/publish.yml`
  (build + PyPI trusted publishing on `v*` tags).

### Changed
- Paper tracker (`run_paper.py`, `harness/paper.py`) hardened for unattended daily
  runs: retries with fallback hosts for Binance spot price lookups
  (`harness.netutil.get_json`/`post_json`, `NetworkError`), an idempotency guard
  (`MIN_TICK_GAP_HOURS = 20`, safe under overlapping cron/launchd triggers),
  atomic state writes (`.tmp` + `os.replace`), and annualization computed from
  calendar days between the first and last tick rather than tick count.
  `harness.hyperliquid` retries through the same `netutil` primitives.
- Scheduling moved from `crontab` to a macOS `launchd` agent
  (`ops/launchd/com.quant-ai-lab.paper.plist`, installer
  `ops/install-paper-launchd.sh`); documented in
  `quant-harness/docs/OPS-PAPER-TRACKER.md`.
- `harness/__init__.py` resolves `audit_returns`, `render_markdown`,
  `AuditInputError`, `get_json`, `post_json`, `NetworkError` lazily
  (`module.__getattr__`, PEP 562) so `python -m harness.audit` runs without a
  `RuntimeWarning`.
- CI installs `.[dev,api]` so the API test suite runs in CI instead of being
  skipped, and runs `examples/audit_quickstart.py` as a smoke check.

### Fixed
- `qh-audit`: a headerless CSV whose first column is a date no longer loses its
  first row to header detection; `--freq` units are case-sensitive (`1M` is
  rejected as a calendar month instead of being read as one minute); `purge`/
  `embargo` that empty a CPCV train set are rejected (exit 2 / HTTP 422) instead
  of producing a `nan` DSR; `n_trials` is capped at 1 000 000 and passed to the
  judge as a number (no per-trial placeholders in memory), so
  `judge.n_configs_tried` always equals `n_trials_effective`; an unwritable
  `--out`/`--json` exits 2 and an unexpected crash exits 3 — never 1 (KILL).
- API: `X-API-Key` and the payment gate are checked in ASGI middleware from the
  headers alone, before the body is read or parsed; `freq` above one year is a
  422 instead of a 500; the `x402`/`nowpayments` stubs require
  `QH_ALLOW_STUB_PAYMENT_GATE=1` to start and mark every unverified response
  with `X-QH-Payment: unverified-stub` plus a `PAYMENT:` assumption line;
  `harness.__version__` is the single version string for reports and responses
  (pinned to `pyproject.toml` by `tests/test_version.py`).
- Paper tracker: a held coin whose funding history fails to load is frozen
  instead of closed and re-opened at full cost; a file lock serialises two
  schedulers starting in the same second.
- `.env` files are git- and docker-ignored and stripped from the publish
  snapshot; `api/.env.example` has no inline comments (`docker --env-file`
  takes values literally).

### Notes
- The paper tracker remains a research instrument, not a product: the live paper
  track is negative over its observed history (see `RESEARCH-CONCLUSION.md` and
  `docs/OPS-PAPER-TRACKER.md`) and nothing in this package is investment advice.
- Payment gates (`x402`, `nowpayments`) are stubs: they issue the correct 402
  challenge shape but do not verify payment, settle funds, or protect against
  replay. Treat `/v1/verdict` as unauthenticated-for-payment until a real
  facilitator/IPN integration lands.

## [0.2.0] — 2026-07-27

### Added
- Packaged `harness` as an installable pip package: `pyproject.toml`
  (`setuptools`), public API surface re-exported from `harness/__init__.py`, and
  the `qh-dsr` console script (`harness.deflated_sharpe:_cli`) for Deflated /
  Probabilistic Sharpe from a returns CSV.
- `examples/quickstart.py`: a data-free, no-network hello-world example plus a
  smoke test, so the framework can be evaluated without a data cache.
- Top-level `README.md` as the lab's front door (navigation across the research
  artifacts and the packaged harness), and `LICENSE` (MIT).
- `docs/METHODOLOGY.md`: a publishable write-up of the honest-backtest method
  (walk-forward / CPCV, cost modeling, Deflated Sharpe, PBO, the PASS/KILL gate
  logic).
- GitHub Actions CI (`.github/workflows/ci.yml`): pytest across Python 3.11–3.13
  plus the quickstart smoke run; a `dev` Makefile target.
- Calibration study of the judge's detection power: `run_calibration.py`, 200-seed
  sweep, `reports/CALIBRATION.md` / `FINDINGS-CALIBRATION.md` — measured FPR 0% and
  a detection threshold around Sharpe 2.2 — plus a companion sample-size study
  (how much data is needed to see an edge), folded into `docs/METHODOLOGY.md`.

### Fixed
- `max_drawdown` made numerically robust: saturates at −1.0 on ruin instead of
  overflowing, computed in log-space.

## [0.1.0] — 2026-07-18 to 2026-07-24

Initial research harness and the pre-registered hypothesis campaign that
established the calibrated PASS/KILL judge used by every later release.

### Added
- Core harness: leakage-safe walk-forward and Combinatorial Purged
  Cross-Validation (CPCV) with purge/embargo, path assembly, and a returns-matrix
  judge (`run_cpcv_returns`), trial-corrected Deflated / Probabilistic Sharpe with
  exact trial-variance estimation across IS folds, and a positive control verifying
  the judge can both PASS a genuine edge and KILL noise.
- Data loaders and caching: parquet klines cache, Binance and Bybit funding/kline
  loaders, Hyperliquid daily close/funding loader (with corrected funding-history
  pagination), point-in-time universe construction for cross-sectional strategies.
- Strategy engines for the 10 pre-registered hypotheses (the baseline time-series
  families plus specs 0001–0009, 0008 being the funding-harvest capstone), each
  with its own findings report under
  `reports/FINDINGS-*.md`:
  - 0001/0002 — cross-sectional momentum/carry (core basket, then liquidity-tail
    basket) — **KILL**
  - 0003 — Binance–Bybit inter-exchange funding basis — **KILL** (edge thinner
    than 4-leg costs)
  - 0004/0005 — post-listing drift/reversal, hedged and clean — **KILL**
    (cost pathology in the hedge)
  - 0006 — Hyperliquid cross-sectional carry/momentum — **KILL**
  - 0007 — cascade-reversal proxy (Binance + Hyperliquid) — **KILL**
  - 0008 / 0008b — cash-and-carry funding harvest, including a real-basis P&L
    variant (not the spot≈perp assumption) and a cost-stress sweep — the one
    surviving research result, carried forward as a research-only signal, never
    as investment advice
  - 0009 — post-shock continuation — **KILL**
- `harness/paper.py` + `run_paper.py`: a daily simulated paper tracker for the
  funding-harvest result, plus `RESEARCH-CONCLUSION.md` and `HARVEST-PATH-A.md`
  documenting the basis-risk analysis behind it.

[0.3.0]: https://github.com/imadeptus/quant-harness/releases/tag/v0.3.0
