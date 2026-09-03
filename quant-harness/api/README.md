# quant-harness verdict API

**Upload returns, get a calibrated verdict.** A thin HTTP layer over
`harness.audit.audit_returns` — the exact function behind the `qh-audit` CLI:
Combinatorial Purged CV, Deflated Sharpe Ratio, PBO, four pre-registered gates,
one mechanical `PASS | KILL`. The HTTP verdict cannot drift from the CLI verdict
(`tests/test_api.py::test_verdict_matches_audit_returns`). Built for AI agents
and humans alike.

The judge behind this endpoint is measured, not assumed: on synthetic data with
known ground truth it PASSes **0.0 % of pure noise at N >= 6 configs** and crosses
**PASS >= 50 % between a true annualized Sharpe of 2.0 (46 %) and 2.5 (81 %) —
about 2.2 by interpolation** (>= 90 % at about 3.0). See `../reports/CALIBRATION.md`.

> Not investment advice. The API screens the numbers you send against
> pre-registered thresholds. It cannot see look-ahead, survivorship, data errors
> or cost mis-specification upstream of the request, and it says nothing about
> future returns.

## Run locally

```bash
cd quant-harness
python -m venv .venv && . .venv/bin/activate
pip install -e ".[api]"          # fastapi, uvicorn, pydantic  (add `dev` for the tests)
uvicorn api.app:app --host 0.0.0.0 --port 8000 --no-access-log    # or: make api
```

Interactive docs: `http://localhost:8000/docs`. Health: `GET /healthz`.

## Docker

```bash
docker build -t qh-api:dev .                 # or: make docker
docker run --rm -p 8000:8000 --env-file api/.env.example qh-api:dev
```

Multi-stage image, non-root user (`app`, uid 10001), `HEALTHCHECK` on
`/healthz`, `PORT` env (default 8000). uvicorn handles `SIGTERM` for a graceful
shutdown; logs are JSON lines on stdout (nothing is written to disk).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | `{"status": "ok", "version": "<package version>"}` — always open |
| `POST` | `/v1/verdict` | returns matrix in, verdict out (auth + payment gate apply) |

### `POST /v1/verdict` — request

```jsonc
{
  "returns": [[0.001, -0.002, ...], ...],   // (configs x T) matrix, or a single series [..]
  "trades":  [[1, 0, ...], ...],            // optional, same shape: position changes per bar
  "freq": "1d",                             // bar spacing: 1m, 15m, 1h, 4h, 1d, 1w ('m' = minutes)
  "start": "2024-01-01T00:00:00Z",          // optional first-bar timestamp (UTC if naive)
  "n_trials": 120,                          // optional: total configs tried, if > rows uploaded
  "thresholds": {"min_trades": 200, "min_oos_sharpe": 0.7, "max_drawdown": 0.20, "min_dsr": 0.95},
  "cpcv": {"n_groups": 10, "k_test": 2, "purge": 1, "embargo": 5},   // defaults = calibrated geometry
  "costs_bps": 5.0,                         // optional: bps charged per unit of `trades` per bar
  "assume_trades_per_bar": 1.0              // optional: turnover assumed when `trades` is omitted
}
```

Every field except `returns` is optional; the values shown for `thresholds` and
`cpcv` are the defaults (the pre-registered gates and the CPCV geometry the judge
was calibrated with — the same defaults as `qh-audit`).

Returns are per-bar fractions (`0.01` = 1 %). At least 100 periods are required.
If `trades` is omitted, `assume_trades_per_bar` trades per bar (default 1.0, an
upper bound that makes the `min_trades` gate lenient) are assumed and the
response says so under `assumptions`. If `costs_bps` is omitted, returns are
treated as already net of costs; if it is given, returns are treated as gross and
`costs_bps / 1e4 x trades` is subtracted on every bar. `n_trials` must be at
least the number of rows uploaded; the DSR is deflated by
`max(n_trials, rows)`.

### Response

Actual output for 6 configs x 600 daily bars of pure noise (`rng(0).normal(0, 0.01)`),
one trade every 3 bars, `costs_bps: 5.0`:

```jsonc
{
  "verdict": "KILL",
  "checks": {"trades_ok": true, "oos_sharpe_ok": false, "drawdown_ok": false, "dsr_ok": false},
  "metrics": {
    "oos_sharpe_annualized": -0.169, "worst_path_sharpe_annualized": -0.983,
    "oos_max_drawdown": 0.215, "psr_vs_zero": 0.4144, "deflated_sharpe_ratio": 0.0551,
    "pbo": 0.4286, "n_paths": 9, "n_configs_tried": 6, "oos_bars": 600,
    "approx_oos_trades": 200, "ann_factor": 365.0
  },
  "thresholds": {"min_trades": 200.0, "min_oos_sharpe": 0.7, "max_drawdown": 0.2, "min_dsr": 0.95},
  "cpcv": {"n_groups": 10, "k_test": 2, "purge": 1, "embargo": 5},
  "assumptions": [
    "costs_bps=5.0: returns treated as gross; 5.0 bps x trades subtracted on every bar before judging.",
    "freq=1d: annualization factor 365.0 periods/year inferred from bar spacing; CPCV verdict uses the median across paths."
  ],
  "cost_sensitivity": [                      // only when both `trades` and `costs_bps` are given, else null
    {"multiplier": 0.0, "costs_bps": 0.0,  "oos_sharpe_annualized": 0.145,  "worst_path_sharpe_annualized": -0.684, "oos_max_drawdown": 0.200, "deflated_sharpe_ratio": 0.1176, "verdict": "KILL"},
    {"multiplier": 0.5, "costs_bps": 2.5,  "oos_sharpe_annualized": -0.012, "worst_path_sharpe_annualized": -0.834, "oos_max_drawdown": 0.207, "deflated_sharpe_ratio": 0.0819, "verdict": "KILL"},
    {"multiplier": 1.0, "costs_bps": 5.0,  "oos_sharpe_annualized": -0.169, "worst_path_sharpe_annualized": -0.983, "oos_max_drawdown": 0.215, "deflated_sharpe_ratio": 0.0551, "verdict": "KILL"},
    {"multiplier": 2.0, "costs_bps": 10.0, "oos_sharpe_annualized": -0.483, "worst_path_sharpe_annualized": -1.102, "oos_max_drawdown": 0.229, "deflated_sharpe_ratio": 0.0225, "verdict": "KILL"}
  ],
  "report_md": "# quant-harness verdict: KILL\n...",
  "disclaimer": "Not investment advice. ...",
  "version": "0.3.0"
}
```

- `verdict` is `PASS` only when all four `checks` are true; `metrics.n_configs_tried`
  is the number the DSR was deflated by (`max(n_trials, rows)`).
- `assumptions` lists everything the judge had to assume (an `ASSUMPTION:` line
  when `trades` was omitted, `WARNING:` lines from the audit, the cost and
  annualization conventions).
- `cost_sensitivity` re-runs the judge at 0x / 0.5x / 1x / 2x of `costs_bps`
  (the 1x row equals the headline `metrics`); it is `null` unless both `trades`
  and `costs_bps` were supplied — with assumed turnover the table would only
  restate the assumption, and the response says so.
- `report_md` is a short English markdown summary suitable for pasting into a
  thread or a PR. Non-finite metrics are returned as `null`. The full
  FINDINGS-style report is what `qh-audit --out` writes.

### Errors

| Status | When |
|---|---|
| `401` | `QH_API_KEYS` is set and `X-API-Key` is missing or wrong |
| `402` | a payment gate is configured and its header is absent (see below) |
| `413` | more than `QH_MAX_CONFIGS` rows, more than `QH_MAX_PERIODS` bars, or a body over `QH_MAX_BODY_BYTES` |
| `422` | ragged matrix, NaN/inf, fewer than 100 periods, `trades` shape mismatch or negative, unsupported `freq` (calendar months/quarters/years, or a spacing above one year), `n_trials` below the rows uploaded or above 1 000 000, negative `assume_trades_per_bar`, a zero-variance series (also after costs), CPCV `n_groups` leaving fewer than 3 periods per group, or `purge`/`embargo` leaving a CPCV train set under 3 periods |

Every error body has `detail` (structured) and `error` (one readable string).
The request body is never echoed back. Order of checks: `413` from
`Content-Length`, then `401` and `402` from the headers alone — the body is not
read, let alone parsed, before auth and payment are settled — then `422`/`413`
on the parsed body, then the judge.

### curl

```bash
python - <<'PY' > body.json
import json, numpy as np
R = np.random.default_rng(0).normal(0.0, 0.01, (6, 600))      # 6 configs x 600 daily bars of noise
T = np.zeros_like(R); T[:, ::3] = 1
print(json.dumps({"returns": R.tolist(), "trades": T.tolist(), "freq": "1d"}))
PY
curl -s -X POST http://localhost:8000/v1/verdict \
     -H 'Content-Type: application/json' -H 'X-API-Key: changeme' \
     --data-binary @body.json | python -m json.tool | head -20
# -> "verdict": "KILL"
```

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `QH_MAX_CONFIGS` | `200` | max rows per request (413 above) |
| `QH_MAX_PERIODS` | `50000` | max bars per row (413 above) |
| `QH_MAX_BODY_BYTES` | `67108864` (64 MiB) | raw body cap, checked before parsing (413 above) |
| `QH_API_KEYS` | *(empty = open)* | comma-separated accepted `X-API-Key` values. Empty means anyone can call the judge: the app has no rate limit and no concurrency cap, so never expose an open instance without a reverse proxy that rate-limits |
| `QH_PAYMENT_GATE` | `noop` | `noop` \| `x402` \| `nowpayments` |
| `QH_ALLOW_STUB_PAYMENT_GATE` | *(unset)* | must be `1` to start `x402`/`nowpayments` at all — both are stubs that never verify payment |
| `QH_X402_PAY_TO` | — | x402 receiving address (required for `x402`) |
| `QH_X402_PRICE_USDC` | — | price per call in USDC (required for `x402`) |
| `QH_X402_NETWORK` | `base-sepolia` | network quoted in the 402 challenge |
| `QH_X402_ASSET` | `USDC` | asset quoted in the challenge (set the token contract for mainnet) |
| `NOWPAYMENTS_API_KEY` | — | required for `nowpayments`; never logged |
| `QH_NOWPAYMENTS_PRICE_USDC` | — | price quoted in the NOWPayments 402 instructions |
| `QH_PUBLIC_URL` | `http://localhost:8000` | used as the x402 `resource` URL |
| `QH_LOG_LEVEL` | `INFO` | log level of the JSON request log |
| `PORT` | `8000` | Docker only |

A half-configured gate (e.g. `QH_PAYMENT_GATE=x402` without `QH_X402_PAY_TO`)
makes the process fail at start-up rather than silently serving for free.

Limits are deliberately conservative: a 200 x 20 000 matrix is about 90 MB of
JSON, so in practice `QH_MAX_BODY_BYTES` binds before `QH_MAX_PERIODS`.
CPCV layouts are capped at C(n_groups, k_test) <= 120 splits per call.

## Logging

One JSON object per request on stdout, no files:

```json
{"ts": "2026-09-02T10:00:00.000+00:00", "level": "INFO", "logger": "qh.api", "msg": "request",
 "path": "/v1/verdict", "method": "POST", "status": 200, "ms": 187.3,
 "n_configs": 6, "n_periods": 600, "verdict": "KILL"}
```

## Payments: what is real and what is a stub

**Real today:** `noop` (free), API-key auth, all limits, the verdict itself.

**Stubs:** both paid gates issue the *challenge* only. They never verify that
money moved, and they let any request through as soon as the payment header is
present (with a `WARNING` log line saying so). The 402 challenge carries
`"stub": true`; a verdict served through a stub (header present, unverified)
carries the response header `X-QH-Payment: unverified-stub` and a
`PAYMENT: ... NOT verified` line under `assumptions` and in `report_md`. Both
stubs refuse to start unless `QH_ALLOW_STUB_PAYMENT_GATE=1` is set explicitly.
Do not put a stub gate in front of anything you charge for.

| Gate | What it does now | What is missing to make it real |
|---|---|---|
| `x402` | Without `X-PAYMENT`: `402` with an x402-style body — `x402Version`, `accepts[{scheme: "exact", network, maxAmountRequired (USDC atomic units), resource, payTo, asset, description, ...}]`. With the header: pass-through, unverified. | Forward the `X-PAYMENT` payload to an **x402 facilitator** (`POST /verify`, then `POST /settle` after serving), reject on failure, set `QH_X402_NETWORK` to a mainnet and `QH_X402_ASSET` to the USDC contract on it, return the `X-PAYMENT-RESPONSE` header, and handle replay (nonce/idempotency) and timeouts. |
| `nowpayments` | Without `X-Payment-Id`: `402` with instructions to create a NOWPayments payment and retry with the id. With the header: pass-through, unverified. | Either look the id up with `GET /v1/payment/{id}` (API key) and require `payment_status == "finished"` for the *expected amount*, or receive the signed **IPN callback**, verify its HMAC with the IPN secret, and store paid ids so each one buys exactly one call. |

Code pointers: `api/payments.py` (each stub is marked `TODO(stub)`),
`api/settings.py` (env parsing), `api/app.py` (`AuthGateMiddleware`).

## Layout

```
api/
  app.py        FastAPI factory, middleware (auth/payment gate, body limit, JSON request log), endpoints
  models.py     pydantic schemas + validation (shapes, NaN/inf, freq, limits on CPCV)
  payments.py   PaymentGate protocol, NoopGate, X402Gate (stub), NowPaymentsGate (stub)
  report.py     report_md renderer + disclaimer
  settings.py   Settings dataclass, Settings.from_env()
  logs.py       JSON stdout logging (stdlib only)
tests/test_api.py
Dockerfile, .dockerignore
```
