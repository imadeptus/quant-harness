#!/usr/bin/env python3
"""Paper tracker for the funding-harvest research track (HARVEST-PATH-A §7).

Run ONCE A DAY. Each run pulls live Hyperliquid perp + Binance spot quotes,
selects the universe by the harvest rule (trailing daily funding > θ, hedgeable
with Binance spot, liquid), marks the simulated delta-neutral positions at the
REAL prices of both legs, accrues funding / basis / costs and writes the state.

No money is involved. The value is the out-of-sample record itself.

Exit codes: 0 = tick written (or already taken today, or another tick holds the
lock), 3 = no market data after all retries (state untouched). Operational notes:
docs/OPS-PAPER-TRACKER.md.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import IO, Any, Dict, Iterable, List, Mapping, Optional, Sequence

import requests

from harness import hyperliquid as hl
from harness.netutil import NetworkError, get_json
from harness.paper import new_state, tick

logger = logging.getLogger("run_paper")

STATE_FILE = "reports/paper_state.json"
THETA = 2 * (0.0005 + 0.0002) / 7      # same θ as spec 0008, per day
# Realistic costs (not the optimistic 5+2): blended per-leg including Binance
# spot standard taker ~10bps -> the ~6.7%/yr scenario of HARVEST-PATH-A §3b.
COSTS = {"taker": 0.0007, "slip": 0.0003}
MIN_DAY_VOL = 5_000_000.0              # HL 24h notional liquidity floor
MAX_POSITIONS = 40

# Binance spot: the same REST API is served from several hosts; rotate on failure.
DEFAULT_BINANCE_HOSTS: Sequence[str] = (
    "api.binance.com", "api1.binance.com", "api2.binance.com",
    "api3.binance.com", "api4.binance.com", "data-api.binance.vision",
)
BINANCE_TICKER_PATH = "/api/v3/ticker/price"
NET_TRIES = 4
NET_TIMEOUT = 20.0

# Idempotency guard: a tick younger than this is "today's" -> do nothing.
MIN_TICK_GAP_HOURS = 20.0
EXIT_NO_DATA = 3

# Per-coin funding history is best-effort: transport failure (wrapped by hl._post),
# malformed rows, or cache I/O problems skip the coin instead of killing the tick.
_LOADER_ERRORS = (hl.HyperliquidError, KeyError, ValueError, TypeError, OSError)


# --------------------------------------------------------------------------- market data
def binance_hosts(env: Mapping[str, str] | None = None) -> List[str]:
    """Spot hosts from env ``BINANCE_SPOT_HOSTS`` (comma-separated) or the defaults."""
    env = os.environ if env is None else env
    raw = env.get("BINANCE_SPOT_HOSTS", "")
    hosts = [h.strip() for h in raw.split(",") if h.strip()]
    return hosts or list(DEFAULT_BINANCE_HOSTS)


def binance_spot_urls(hosts: Iterable[str]) -> List[str]:
    return [f"https://{h}{BINANCE_TICKER_PATH}" for h in hosts]


def parse_binance_spot(payload: Any) -> Dict[str, float]:
    """{base: price} for every USDT spot pair (the hedge leg). Bad rows are skipped."""
    if not isinstance(payload, list):
        raise ValueError(f"binance ticker/price: expected a list, got {str(payload)[:80]}")
    out: Dict[str, float] = {}
    for row in payload:
        try:
            symbol = str(row["symbol"])
            if symbol.endswith("USDT"):
                out[symbol[:-4]] = float(row["price"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def binance_spot_prices(hosts: Sequence[str] | None = None, *,
                        session: requests.Session | None = None) -> Dict[str, float]:
    """Live Binance spot prices via the retrying/rotating client."""
    urls = binance_spot_urls(hosts if hosts is not None else binance_hosts())
    payload = get_json(urls, tries=NET_TRIES, timeout=NET_TIMEOUT, session=session)
    return parse_binance_spot(payload)


def parse_hl_snapshot(meta: Any) -> Dict[str, Dict[str, float]]:
    """{coin: {perp, funding_hourly, day_vol}} from metaAndAssetCtxs. Bad rows skipped."""
    try:
        universe = meta[0]["universe"]
        ctxs = meta[1]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"metaAndAssetCtxs: unexpected shape ({type(e).__name__})") from e
    out: Dict[str, Dict[str, float]] = {}
    for u, c in zip(universe, ctxs):
        try:
            out[u["name"]] = {"perp": float(c["markPx"]),
                              "funding_hourly": float(c["funding"]),
                              "day_vol": float(c.get("dayNtlVlm", 0.0))}
        except (KeyError, TypeError, ValueError):
            continue
    return out


def hl_snapshot() -> Dict[str, Dict[str, float]]:
    """Live HL perp snapshot; hl._post retries and rotates hosts."""
    return parse_hl_snapshot(hl._post({"type": "metaAndAssetCtxs"}))


def _hl_base_to_binance(coin: str) -> str:
    # HL kPEPE/kBONK -> PEPE/BONK on Binance spot
    return coin[1:].upper() if coin.startswith("k") and coin[1:].upper() else coin.upper()


def select_universe(hl_snap: Dict[str, Dict[str, float]], spot: Dict[str, float],
                    now_ms: int, held: Iterable[str] = ()) -> List[str]:
    """Target universe: trailing daily funding > θ, Binance spot exists, liquid.

    ``held`` — coins currently in the book. A held coin whose funding history
    cannot be loaded (transient HL failure) is kept in the target — frozen, like
    a coin without a quote — instead of being closed and re-opened at full
    round-trip cost on the next tick. "No data" is not "funding below θ".
    """
    held_set = set(held)
    cand = []
    for coin, d in hl_snap.items():
        base = _hl_base_to_binance(coin)
        if base not in spot:
            continue
        if d["day_vol"] < MIN_DAY_VOL:
            continue
        if d["funding_hourly"] <= 0:
            continue
        cand.append(coin)
    start = now_ms - 8 * 86_400_000
    scored = []
    frozen: List[str] = []
    for coin in cand:
        try:
            f = hl.load_hl_funding_daily(coin, start, now_ms)
        except _LOADER_ERRORS as e:
            if coin in held_set:
                logger.warning("funding history %s unavailable (%s: %s) - held position "
                               "kept (frozen), not closed", coin, type(e).__name__, e)
                frozen.append(coin)
            else:
                logger.warning("funding history %s skipped: %s: %s", coin, type(e).__name__, e)
            continue
        if len(f) >= 3 and f.tail(7).mean() > THETA:
            scored.append((float(f.tail(7).mean()), coin))
    scored.sort(reverse=True)
    target = [c for _, c in scored[:MAX_POSITIONS]]
    return target + [c for c in frozen if c not in target]


def funding_since(coins: Iterable[str], last_ms: int, now_ms: int) -> Dict[str, float]:
    """Funding accrued on held coins since the previous tick (0.0 when unavailable)."""
    out: Dict[str, float] = {}
    for coin in coins:
        try:
            f = hl.load_hl_funding_daily(coin, last_ms, now_ms)
            out[coin] = float(f.sum()) if len(f) else 0.0
        except _LOADER_ERRORS as e:
            logger.warning("funding since last tick for %s unavailable, using 0: %s: %s",
                           coin, type(e).__name__, e)
            out[coin] = 0.0
    return out


# --------------------------------------------------------------------------- pure helpers
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def tick_already_taken(last_tick_ms: int | None, now_ms: int,
                       min_gap_hours: float = MIN_TICK_GAP_HOURS) -> bool:
    """True when the previous tick is younger than ``min_gap_hours`` (cron+launchd guard)."""
    if last_tick_ms is None:
        return False
    return (now_ms - int(last_tick_ms)) < min_gap_hours * 3_600_000


def _tick_date(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def calendar_days(ticks: Sequence[Mapping[str, Any]]) -> int:
    """Calendar days spanned by the first..last tick (date difference), at least 1."""
    if len(ticks) < 2:
        return 1
    try:
        first = _tick_date(str(ticks[0]["ts"])).date()
        last = _tick_date(str(ticks[-1]["ts"])).date()
    except (KeyError, ValueError, TypeError):
        return max(len(ticks), 1)
    return max((last - first).days, 1)


def annualized_pct(equity: float, capital: float, days: int) -> float:
    """Simple (non-compounded) annualisation of the cumulative return over ``days``."""
    if days < 1:
        return 0.0
    return (equity / capital - 1.0) / days * 365 * 100


def summary_lines(state: Mapping[str, Any], now: datetime) -> List[str]:
    """Human-readable tick report (what ends up in reports/paper.log)."""
    ticks = state["ticks"]
    last = ticks[-1]
    n_ticks = len(ticks)
    days = calendar_days(ticks)
    lines = [
        f"=== PAPER TICK {now.date()} (tick #{n_ticks}) ===",
        f"positions: {last['n_positions']} | tick: funding {last['funding']:+.2f} "
        f"basis {last['basis_pnl']:+.2f} costs {last['cost']:.2f}",
        f"equity: ${state['equity']:.2f} (start ${state['capital']:.0f}) | "
        f"cum: funding {state['cum_funding']:+.1f} basis {state['cum_basis_pnl']:+.1f} "
        f"costs {state['cum_costs']:.1f}",
    ]
    if n_ticks > 1:
        ann = annualized_pct(state["equity"], state["capital"], days)
        lines.append(f"annualized by calendar days: {ann:+.1f}%/yr | "
                     f"ticks: {n_ticks} of {days} calendar days")
    if last["warnings"]:
        lines.append("RISK FLAGS:")
        lines.extend("  ! " + w for w in last["warnings"])
    lines.append(f"holding: {', '.join(sorted(state['positions'].keys())) or '-'}")
    lines.append(f"state -> {STATE_FILE}")
    return lines


# --------------------------------------------------------------------------- state I/O
def load_state(path: str, now_ms: int) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        state.setdefault("last_tick_ms", now_ms - 86_400_000)
        return state
    state = new_state(capital=10_000.0, notional=250.0)   # $250 per coin, $10k book
    state["last_tick_ms"] = now_ms - 86_400_000
    return state


def save_state(path: str, state: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def lock_path(state_path: str) -> str:
    return state_path + ".lock"


def _try_lock(path: str) -> Optional[IO[str]]:
    """Non-blocking exclusive ``flock`` on ``path``; None when another tick holds it.

    The 20 h guard cannot stop two schedulers (cron + launchd) that start in the
    same second — both read yesterday's state and both go to the network. The lock
    serialises them; the loser exits 0 without touching state or the log summary."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fh = open(path, "w", encoding="utf-8")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    return fh


# --------------------------------------------------------------------------- entry point
def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> int:
    _setup_logging()
    lock = _try_lock(lock_path(STATE_FILE))
    if lock is None:
        print("another tick in progress (lock held) - nothing to do")
        return 0
    with lock:
        return _run_tick()


def _run_tick() -> int:
    now = _utcnow()
    now_ms = int(now.timestamp() * 1000)
    logger.info("paper tick start now=%s state=%s", now.isoformat(timespec="seconds"),
                STATE_FILE)

    state = load_state(STATE_FILE, now_ms)
    last_ms = int(state["last_tick_ms"])

    if tick_already_taken(last_ms, now_ms):
        age_h = (now_ms - last_ms) / 3_600_000
        print(f"tick already taken today (last tick {age_h:.1f}h ago < "
              f"{MIN_TICK_GAP_HOURS:.0f}h) - nothing to do")
        return 0

    try:
        spot = binance_spot_prices()
        if not spot:
            raise ValueError("binance spot returned no USDT pairs")
        hl_snap = hl_snapshot()
        if not hl_snap:
            raise ValueError("hyperliquid snapshot returned no perps")
    except (NetworkError, hl.HyperliquidError, ValueError) as e:
        print(f"NO DATA: {e} - state unchanged, exiting {EXIT_NO_DATA}")
        for line in getattr(e, "attempts", []):
            print("  attempt: " + line)
        return EXIT_NO_DATA

    target = select_universe(hl_snap, spot, now_ms, held=state["positions"].keys())

    # both legs' prices for every coin in play (held + target)
    coins = set(target) | set(state["positions"].keys())
    prices: Dict[str, Dict[str, float]] = {}
    for coin in coins:
        base = _hl_base_to_binance(coin)
        if coin in hl_snap and base in spot:
            prices[coin] = {"perp": hl_snap[coin]["perp"], "spot": spot[base]}

    fsince = funding_since(list(state["positions"].keys()), last_ms, now_ms)
    state = tick(state, prices, fsince, [c for c in target if c in prices],
                 COSTS, ts=now.isoformat(), basis_alert=0.03)
    state["last_tick_ms"] = now_ms
    save_state(STATE_FILE, state)

    for line in summary_lines(state, now):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
