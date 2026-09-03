# Paper tracker — operations

Status: realised (2026-09-02). Applies to `run_paper.py`, `harness/netutil.py`,
`ops/launchd/`, `ops/install-paper-launchd.sh`.

The paper tracker is a **research instrument, not a product**. It records, once a
day, what a simulated delta-neutral funding-harvest book (short Hyperliquid perp
+ long Binance spot, `$250` per coin, `$10,000` book) would have done at the real
quotes of both legs. No money is involved, nothing is traded, and nothing in this
document is investment advice. The point of the track is to measure the things a
backtest cannot — actual basis drift between the two legs and the cost of
rebalancing — before anyone considers real capital.

## 1. What one tick does

`run_paper.py` is meant to run once a day at 09:00 local time. One run:

1. Loads `reports/paper_state.json` (creates a fresh `$10k` book if absent).
2. **Idempotency guard.** If the previous tick is younger than 20 hours it prints
   `tick already taken today` and exits `0` without touching anything. This makes
   it safe to have cron and launchd both scheduled during the migration, or to
   re-run by hand. Two runs that start in the same second (cron and launchd both
   at 09:00:00) are serialised by a file lock on `reports/paper_state.json.lock`:
   the second one prints `another tick in progress` and exits `0`, so the log
   never shows two `=== PAPER TICK` blocks for one day.
3. Pulls Binance spot prices (`/api/v3/ticker/price`) through the retrying,
   host-rotating client in `harness/netutil.py`, and the Hyperliquid perp snapshot
   (`metaAndAssetCtxs`) through `hl._post`, which uses the same client.
4. If either feed is unavailable after all retries — or comes back empty — it
   prints one `NO DATA: ...` line plus the list of failed attempts and exits `3`.
   The state file is not modified. (An empty price map would otherwise be read as
   "close every position" and charge exit costs, so an empty feed is treated as
   no feed.)
5. Selects the target universe (trailing 7-day mean daily funding above
   `θ = 2·(5+2)bp/7 ≈ 0.02%/day`, Binance-hedgeable, ≥ `$5M` 24h notional on HL,
   at most 40 names), marks existing positions at both legs' real prices, accrues
   funding since the last tick, charges taker+slippage (`7+3 bp` per leg) on every
   entry and exit, and appends a tick record.
6. Writes the state atomically (`.tmp` + rename) and prints the summary block.

Exit codes: `0` tick written or already taken; `3` no market data; anything else
is an unexpected error and will carry a traceback.

## 2. State file fields (`reports/paper_state.json`)

| Field | Meaning |
|---|---|
| `capital` | Starting book, `10000.0`. Never changes. |
| `notional` | Size of each leg per coin, `250.0`. |
| `positions` | `{coin: {spot, perp, entered}}` — last mark prices of both legs and the ISO timestamp of entry. |
| `cum_funding` | Sum of `notional × funding` accrued while positions were held. |
| `cum_basis_pnl` | Sum of `notional × (spot_return − perp_return)` per tick — the P&L of the hedge itself. |
| `cum_costs` | Sum of `notional × 2 legs × (taker + slip)` on every entry and exit. |
| `equity` | `capital + cum_funding + cum_basis_pnl − cum_costs`. |
| `ticks` | One record per tick: `ts`, `n_positions`, `funding`, `basis_pnl`, `cost`, `equity`, `warnings` (basis divergence > 3% on a coin, or a coin with no quote that tick — its position is frozen, not closed). |
| `last_tick_ms` | Unix ms of the last tick; drives the idempotency guard and the funding window. |

The structure is unchanged by the 2026-09-02 hardening; only the console output
was extended.

## 3. Why ticks were missed (27 ticks in 42 calendar days)

Two independent causes, visible in `reports/paper.log`:

1. **Cron does not fire while the Mac is asleep.** A missed 09:00 leaves no trace
   at all — the majority of the 15 missing days look like this (no header, no
   traceback, nothing).
2. **Single hard-coded host, no retries.** Three runs did start and died on the
   first request to `api.binance.com`: twice with `NameResolutionError` (DNS not
   yet up, late July) and once with
   `SSLError: UNEXPECTED_EOF_WHILE_READING` (TLS connection cut mid-handshake,
   2026-09-02). Each left a ~40-line traceback and no tick.

What changed:

- `harness/netutil.get_json` / `post_json`: up to `max(tries, len(hosts))`
  attempts, exponential backoff `1.5·2^i` s plus jitter (capped at 30 s), round-robin
  over hosts, one `WARNING` line per failed attempt naming the host and the
  exception class, and a final `NetworkError` listing every attempt.
- Binance spot is fetched from six interchangeable hosts
  (`api`, `api1`–`api4.binance.com`, `data-api.binance.vision`), overridable via
  `BINANCE_SPOT_HOSTS`.
- `hl._post` retries 3× with 1 s / 2 s waits and can rotate over `HL_INFO_URLS`.
- No data ⇒ exit `3` with a one-line reason instead of a traceback; state untouched.
- The idempotency guard (section 1, step 2).
- Annualisation is now by calendar days, not by tick count (section 5).
- launchd replaces cron (section 4) so a tick missed during sleep runs on wake.

Note what did *not* change: a tick that runs after a gap still accrues funding
for the whole gap (`funding_since` queries HL funding history from the previous
`last_tick_ms`), and basis is marked from the last stored prices. Gaps therefore
do not lose P&L, they only lose observation points and rebalancing opportunities.

## 4. Migrating from cron to launchd

launchd's `StartCalendarInterval` runs a job that was missed while the machine
slept as soon as it wakes; cron just skips it. The agent definition lives in
`ops/launchd/com.quant-ai-lab.paper.plist`:

- program: `/Users/art/quant-ai-lab/quant-harness/.venv/bin/python run_paper.py`
- working directory: `/Users/art/quant-ai-lab/quant-harness`
- schedule: 09:00 local, daily; `RunAtLoad=false`
- stdout and stderr appended to `reports/paper.log` (same file cron used)
- `PYTHONUNBUFFERED=1` so log lines stay in order

Steps:

```sh
cd /Users/art/quant-ai-lab/quant-harness
ops/install-paper-launchd.sh          # copies the plist, bootstraps, prints status
# optional smoke test — harmless after today's cron tick (guard exits 0):
ops/install-paper-launchd.sh --now
tail -5 reports/paper.log
```

Once the first launchd tick shows up in the log, delete the cron entry with
`crontab -e` (the installer prints the exact line; it never edits the crontab
itself). Until then both schedulers may fire at 09:00 — the second one exits with
`tick already taken today`.

Useful commands:

```sh
launchctl print gui/$(id -u)/com.quant-ai-lab.paper | grep -E 'state|last exit'
launchctl kickstart gui/$(id -u)/com.quant-ai-lab.paper     # run now
ops/install-paper-launchd.sh --uninstall                     # remove the agent
```

If the plist is edited, re-run the installer; it replaces the loaded copy.

## 5. Reading the log

Each run appends one block to `reports/paper.log`. A healthy tick looks like:

```
2026-09-03 09:00:02,113 INFO run_paper: paper tick start now=2026-09-03T04:00:02+00:00 state=reports/paper_state.json
=== PAPER TICK 2026-09-03 (tick #28) ===
positions: 18 | tick: funding +1.10 basis -3.20 costs 0.00
equity: $9846.09 (start $10000) | cum: funding +67.4 basis -124.6 costs 96.7
annualized by calendar days: -13.1%/yr | ticks: 28 of 43 calendar days
RISK FLAGS:
  ! XMR: ...
holding: AAVE, BTC, ...
state -> reports/paper_state.json
```

- `annualized by calendar days` divides the cumulative return by the number of
  calendar days between the first and the last tick (minimum 1) and scales to 365.
  The previous output divided by the *number of ticks*, which overstates the rate
  whenever days are missed.
- `ticks: N of D calendar days` is the coverage ratio; if `N` stops growing while
  `D` does, the scheduler is not firing.
- `WARNING harness.netutil: attempt 2/6 failed: GET api1.binance.com: SSLError: ...`
  lines are retries, not failures. A run that ends in `NO DATA:` after them is a
  failure (exit `3`, no tick).
- `tick already taken today` is the idempotency guard — informational.
- `RISK FLAGS` come from the accounting core (`harness/paper.py`): a basis move
  above 3% on one coin in one tick, or a coin with no quote (position frozen).

Quick checks:

```sh
grep -c '=== PAPER TICK' reports/paper.log            # ticks recorded
grep -E 'NO DATA|Traceback' reports/paper.log | tail  # failed runs
grep 'WARNING' reports/paper.log | tail -20           # retry noise
```

## 6. Current result (as of 2026-09-02, honest version)

27 ticks over 42 calendar days (2026-07-22 → 2026-09-02):

- equity **9,848.19** of 10,000 (**−1.52 %**)
- funding **+66.3**, basis **−121.4**, costs **−96.7**

Basis drift between the perp and the spot leg has so far cost almost twice what
funding earned, and rebalancing costs exceeded the funding income on their own.
Annualised by calendar days that is about **−13.2 %/yr** (the earlier tick-based
figure of −20.5 %/yr divided by 27 ticks instead of 42 days). Six weeks of a
single book is far too short to conclude anything in either direction; the number
is reported because the track exists to report it, not because it supports a
strategy. See `../RESEARCH-CONCLUSION.md` (repository root) for the research
verdict on the harvest idea.

## 7. Configuration

| Variable | Default | Effect |
|---|---|---|
| `BINANCE_SPOT_HOSTS` | `api.binance.com,api1.binance.com,api2.binance.com,api3.binance.com,api4.binance.com,data-api.binance.vision` | Comma-separated spot hosts, tried round-robin. |
| `HL_INFO_URLS` | `https://api.hyperliquid.xyz/info` | Comma-separated `POST /info` endpoints. |

Retry counts and backoff are constants (`run_paper.NET_TRIES=4`,
`hl.HL_TRIES=3`, `hl.HL_BACKOFF=1.0`). Worst case for a fully dead Binance is six
attempts: under a minute of backoff (1.5+3+6+12+24 s plus jitter) plus up to 20 s
of connect/read timeout per attempt, then exit `3`.

Tests: `tests/test_paper_net.py` (retries, host rotation, `NetworkError`,
parsing, idempotency, calendar-day annualisation, `main()` exit paths — all
without network) and `tests/test_paper.py` (accounting core).
