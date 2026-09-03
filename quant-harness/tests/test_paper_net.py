"""Paper tracker plumbing: retrying HTTP helpers and the pure parts of run_paper.

No network. ``requests.get``/``requests.post`` are monkeypatched, sleep and the
jitter RNG are injected, so the backoff schedule is deterministic.
"""
from __future__ import annotations

import json
import logging
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests

from harness import hyperliquid as hl
from harness import netutil
from harness.netutil import NetworkError, get_json, post_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_paper  # noqa: E402  (top-level script, not part of the package)


# --------------------------------------------------------------------------- helpers
class _Resp:
    def __init__(self, payload=None, status=200, raw=None):
        self._p = payload
        self.status_code = status
        self._raw = raw

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        if self._raw is not None:
            return json.loads(self._raw)   # raises ValueError on garbage
        return self._p


class _Rng:
    """Deterministic stand-in for random.Random: uniform() always returns ``value``."""

    def __init__(self, value: float = 0.25):
        self.value = value

    def uniform(self, a: float, b: float) -> float:
        return self.value


def _script(outcomes):
    """Build a fake requests.get/post that plays ``outcomes`` in order.

    An outcome is either an exception instance (raised) or a _Resp (returned).
    Records every URL it was called with."""
    calls: list[str] = []
    it = iter(outcomes)

    def fake(url, *args, **kwargs):
        calls.append(url)
        out = next(it)
        if isinstance(out, BaseException):
            raise out
        return out

    fake.calls = calls  # type: ignore[attr-defined]
    return fake


# --------------------------------------------------------------------------- netutil
def test_get_json_returns_payload_on_first_success(monkeypatch):
    fake = _script([_Resp([{"symbol": "BTCUSDT", "price": "1"}])])
    monkeypatch.setattr(netutil.requests, "get", fake)
    slept: list[float] = []
    out = get_json(["https://a.example/p"], sleep=slept.append, rng=_Rng())
    assert out == [{"symbol": "BTCUSDT", "price": "1"}]
    assert fake.calls == ["https://a.example/p"]
    assert slept == [], "no backoff after a success"


def test_get_json_retries_with_exponential_backoff_and_jitter(monkeypatch):
    fake = _script([requests.exceptions.SSLError("eof"),
                    requests.exceptions.ConnectionError("dns"),
                    _Resp({"ok": 1})])
    monkeypatch.setattr(netutil.requests, "get", fake)
    slept: list[float] = []
    out = get_json(["https://a.example/p"], tries=4, backoff=1.5,
                   sleep=slept.append, rng=_Rng(0.25))
    assert out == {"ok": 1}
    assert len(fake.calls) == 3
    # i-th wait = backoff * 2**i + jitter, jitter fixed at 0.25 by the injected rng
    assert slept == pytest.approx([1.5 + 0.25, 3.0 + 0.25])


def test_get_json_rotates_hosts_round_robin(monkeypatch):
    urls = ["https://a.example/p", "https://b.example/p", "https://c.example/p"]
    fake = _script([requests.exceptions.SSLError("eof"),
                    requests.exceptions.SSLError("eof"),
                    requests.exceptions.SSLError("eof"),
                    _Resp({"ok": 1})])
    monkeypatch.setattr(netutil.requests, "get", fake)
    out = get_json(urls, tries=4, sleep=lambda s: None, rng=_Rng())
    assert out == {"ok": 1}
    # attempt k goes to urls[k % len(urls)]: a, b, c, then back to a
    assert fake.calls == [urls[0], urls[1], urls[2], urls[0]]


def test_get_json_tries_every_host_at_least_once(monkeypatch):
    """With more hosts than ``tries`` every host still gets one shot."""
    urls = [f"https://h{i}.example/p" for i in range(6)]
    fake = _script([requests.exceptions.SSLError("eof")] * 5 + [_Resp({"ok": 1})])
    monkeypatch.setattr(netutil.requests, "get", fake)
    out = get_json(urls, tries=4, sleep=lambda s: None, rng=_Rng())
    assert out == {"ok": 1}
    assert fake.calls == urls


def test_get_json_raises_network_error_with_attempt_log(monkeypatch, caplog):
    urls = ["https://a.example/p", "https://b.example/p"]
    fake = _script([requests.exceptions.SSLError("eof"),
                    requests.exceptions.ConnectionError("dns"),
                    requests.exceptions.Timeout("slow"),
                    _Resp(status=502)])
    monkeypatch.setattr(netutil.requests, "get", fake)
    with caplog.at_level(logging.WARNING, logger="harness.netutil"):
        with pytest.raises(NetworkError) as ei:
            get_json(urls, tries=4, sleep=lambda s: None, rng=_Rng())
    err = ei.value
    assert len(err.attempts) == 4
    assert "SSLError" in err.attempts[0] and "a.example" in err.attempts[0]
    assert "ConnectionError" in err.attempts[1] and "b.example" in err.attempts[1]
    assert "Timeout" in err.attempts[2]
    assert "HTTPError" in err.attempts[3]
    assert isinstance(err.__cause__, requests.HTTPError)
    assert "4 attempts" in str(err)
    # one WARNING line per failed attempt, naming host and error class
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 4
    assert "a.example" in warns[0].getMessage() and "SSLError" in warns[0].getMessage()


def test_get_json_treats_bad_json_as_failed_attempt(monkeypatch):
    fake = _script([_Resp(raw="<html>not json</html>"), _Resp({"ok": 1})])
    monkeypatch.setattr(netutil.requests, "get", fake)
    out = get_json(["https://a.example/p"], tries=2, sleep=lambda s: None, rng=_Rng())
    assert out == {"ok": 1}
    assert len(fake.calls) == 2


def test_get_json_passes_params_and_timeout(monkeypatch):
    seen = {}

    def fake(url, params=None, timeout=None):
        seen.update(url=url, params=params, timeout=timeout)
        return _Resp({"ok": 1})

    monkeypatch.setattr(netutil.requests, "get", fake)
    get_json(["https://a.example/p"], params={"symbol": "BTCUSDT"}, timeout=7)
    assert seen == {"url": "https://a.example/p", "params": {"symbol": "BTCUSDT"}, "timeout": 7}


def test_get_json_uses_injected_session(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("module-level requests.get must not be used with a session")

    monkeypatch.setattr(netutil.requests, "get", boom)

    class _Session:
        def __init__(self):
            self.calls = []

        def get(self, url, params=None, timeout=None):
            self.calls.append(url)
            return _Resp({"ok": 1})

    s = _Session()
    assert get_json(["https://a.example/p"], session=s) == {"ok": 1}
    assert s.calls == ["https://a.example/p"]


def test_get_json_rejects_empty_url_list():
    with pytest.raises(ValueError):
        get_json([])


def test_post_json_sends_body_and_retries(monkeypatch):
    seen: list[dict] = []
    outcomes = iter([requests.exceptions.SSLError("eof"), _Resp([{"universe": []}, []])])

    def fake(url, json=None, timeout=None):
        seen.append({"url": url, "json": json, "timeout": timeout})
        out = next(outcomes)
        if isinstance(out, BaseException):
            raise out
        return out

    monkeypatch.setattr(netutil.requests, "post", fake)
    slept: list[float] = []
    out = post_json("https://hl.example/info", {"type": "meta"}, tries=3, backoff=1.0,
                    timeout=30, sleep=slept.append, rng=_Rng(0.0))
    assert out == [{"universe": []}, []]
    assert [s["json"] for s in seen] == [{"type": "meta"}] * 2
    assert seen[0]["timeout"] == 30
    assert slept == pytest.approx([1.0])


def test_post_json_raises_network_error_after_exhaustion(monkeypatch):
    fake = _script([requests.exceptions.ConnectionError("down")] * 3)
    monkeypatch.setattr(netutil.requests, "post", fake)
    with pytest.raises(NetworkError) as ei:
        post_json(["https://hl.example/info"], {"type": "meta"}, tries=3,
                  sleep=lambda s: None, rng=_Rng())
    assert len(ei.value.attempts) == 3
    assert len(fake.calls) == 3


def test_backoff_delay_is_capped():
    rng = _Rng(0.0)
    assert netutil.backoff_delay(0, 1.5, rng) == pytest.approx(1.5)
    assert netutil.backoff_delay(3, 1.5, rng) == pytest.approx(12.0)
    assert netutil.backoff_delay(10, 1.5, rng, max_delay=30.0) == pytest.approx(30.0)


def test_backoff_jitter_uses_real_random_by_default():
    d = netutil.backoff_delay(0, 1.5, random.Random(0))
    assert 1.5 <= d < 3.0


# --------------------------------------------------------------------------- hyperliquid._post
def test_hl_post_retries_then_succeeds(monkeypatch):
    fake = _script([requests.exceptions.SSLError("eof"),
                    _Resp([{"universe": [{"name": "BTC"}]}, [{"funding": "0.0001"}]])])
    monkeypatch.setattr(hl.requests, "post", fake)
    monkeypatch.setattr(hl, "_sleep", lambda s: None)
    assert hl.list_hl_perps() == ["BTC"]
    assert len(fake.calls) == 2


def test_hl_post_wraps_network_error_as_hyperliquid_error(monkeypatch):
    fake = _script([requests.exceptions.ConnectionError("down")] * 10)
    monkeypatch.setattr(hl.requests, "post", fake)
    monkeypatch.setattr(hl, "_sleep", lambda s: None)
    with pytest.raises(hl.HyperliquidError) as ei:
        hl._post({"type": "metaAndAssetCtxs"})
    assert isinstance(ei.value.__cause__, NetworkError)
    assert len(fake.calls) == hl.HL_TRIES


def test_hl_info_urls_from_env(monkeypatch):
    monkeypatch.delenv("HL_INFO_URLS", raising=False)
    assert hl._info_urls() == [hl.INFO]
    monkeypatch.setenv("HL_INFO_URLS", "https://a.example/info, https://b.example/info")
    assert hl._info_urls() == ["https://a.example/info", "https://b.example/info"]


# --------------------------------------------------------------------------- run_paper: pure parts
def test_binance_hosts_default_and_env_override(monkeypatch):
    monkeypatch.delenv("BINANCE_SPOT_HOSTS", raising=False)
    hosts = run_paper.binance_hosts()
    assert hosts[0] == "api.binance.com"
    assert "data-api.binance.vision" in hosts
    assert len(hosts) == 6
    monkeypatch.setenv("BINANCE_SPOT_HOSTS", "x.example, y.example")
    assert run_paper.binance_hosts() == ["x.example", "y.example"]


def test_binance_spot_urls_use_ticker_path():
    urls = run_paper.binance_spot_urls(["a.example", "b.example"])
    assert urls == ["https://a.example/api/v3/ticker/price",
                    "https://b.example/api/v3/ticker/price"]


def test_parse_binance_spot_keeps_usdt_pairs_only():
    payload = [{"symbol": "BTCUSDT", "price": "50000.5"},
               {"symbol": "ETHBTC", "price": "0.05"},
               {"symbol": "PEPEUSDT", "price": "0.00001"},
               {"symbol": "BROKEN"},                      # missing price -> skipped
               {"symbol": "BADUSDT", "price": "n/a"}]     # unparsable -> skipped
    out = run_paper.parse_binance_spot(payload)
    assert out == {"BTC": 50000.5, "PEPE": 0.00001}


def test_parse_binance_spot_rejects_non_list():
    with pytest.raises(ValueError):
        run_paper.parse_binance_spot({"code": -1003, "msg": "banned"})


def test_binance_spot_prices_goes_through_get_json(monkeypatch):
    fake = _script([requests.exceptions.SSLError("eof"),
                    _Resp([{"symbol": "BTCUSDT", "price": "1.5"}])])
    monkeypatch.setattr(netutil.requests, "get", fake)
    monkeypatch.setattr(netutil, "_sleep", lambda s: None)
    out = run_paper.binance_spot_prices(["a.example", "b.example"])
    assert out == {"BTC": 1.5}
    assert fake.calls == ["https://a.example/api/v3/ticker/price",
                          "https://b.example/api/v3/ticker/price"]


def test_parse_hl_snapshot():
    meta = [{"universe": [{"name": "BTC"}, {"name": "kPEPE"}, {"name": "BAD"}]},
            [{"markPx": "100", "funding": "0.0001", "dayNtlVlm": "1e7"},
             {"markPx": "0.01", "funding": "0.0002"},
             {"markPx": None, "funding": "x"}]]
    out = run_paper.parse_hl_snapshot(meta)
    assert out == {"BTC": {"perp": 100.0, "funding_hourly": 0.0001, "day_vol": 1e7},
                   "kPEPE": {"perp": 0.01, "funding_hourly": 0.0002, "day_vol": 0.0}}


def test_parse_hl_snapshot_rejects_malformed():
    with pytest.raises(ValueError):
        run_paper.parse_hl_snapshot({"error": "nope"})


def test_tick_already_taken_within_20h():
    now_ms = 1_800_000_000_000
    h = 3_600_000
    assert run_paper.tick_already_taken(now_ms - 5 * h, now_ms) is True
    assert run_paper.tick_already_taken(now_ms - 19 * h, now_ms) is True
    assert run_paper.tick_already_taken(now_ms - 21 * h, now_ms) is False
    assert run_paper.tick_already_taken(now_ms - 3 * 86_400_000, now_ms) is False
    assert run_paper.tick_already_taken(None, now_ms) is False


def _synthetic_state(n_ticks: int, first: datetime, last: datetime, equity: float) -> dict:
    step = (last - first) / max(n_ticks - 1, 1)
    ticks = [{"ts": (first + i * step).isoformat(), "n_positions": 1, "funding": 0.0,
              "basis_pnl": 0.0, "cost": 0.0, "equity": equity, "warnings": []}
             for i in range(n_ticks)]
    return {"capital": 10_000.0, "notional": 250.0, "positions": {"BTC": {}},
            "cum_funding": 66.26, "cum_basis_pnl": -121.37, "cum_costs": 96.7,
            "equity": equity, "ticks": ticks, "last_tick_ms": int(last.timestamp() * 1000)}


def test_calendar_days_between_first_and_last_tick():
    first = datetime(2026, 7, 22, 16, 56, tzinfo=timezone.utc)
    last = datetime(2026, 9, 2, 4, 0, tzinfo=timezone.utc)
    st = _synthetic_state(27, first, last, 9848.19)
    assert run_paper.calendar_days(st["ticks"]) == 42
    # a single tick (or same-day ticks) still counts as one day
    assert run_paper.calendar_days(st["ticks"][:1]) == 1
    assert run_paper.calendar_days([]) == 1


def test_annualized_pct_by_calendar_days():
    assert run_paper.annualized_pct(9848.19, 10_000.0, 42) == pytest.approx(-13.19, abs=0.01)
    assert run_paper.annualized_pct(9848.19, 10_000.0, 27) == pytest.approx(-20.52, abs=0.01)
    assert run_paper.annualized_pct(10_000.0, 10_000.0, 0) == 0.0


def test_summary_lines_report_ticks_vs_calendar_days():
    first = datetime(2026, 7, 22, 16, 56, tzinfo=timezone.utc)
    last = datetime(2026, 9, 2, 4, 0, tzinfo=timezone.utc)
    st = _synthetic_state(27, first, last, 9848.19)
    text = "\n".join(run_paper.summary_lines(st, last))
    assert "ticks: 27 of 42 calendar days" in text
    assert "annualized by calendar days" in text
    assert "-13.2%" in text
    assert "(tick #27)" in text


# --------------------------------------------------------------------------- run_paper.main
def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state))


def test_main_skips_when_tick_already_taken(monkeypatch, tmp_path, capsys):
    now = datetime(2026, 9, 2, 9, 0, tzinfo=timezone.utc)
    st = _synthetic_state(3, now - timedelta(days=2), now - timedelta(hours=5), 9990.0)
    state_file = tmp_path / "paper_state.json"
    _write_state(state_file, st)
    before = state_file.read_text()
    monkeypatch.setattr(run_paper, "STATE_FILE", str(state_file))
    monkeypatch.setattr(run_paper, "_utcnow", lambda: now)

    def no_net(*a, **k):
        raise AssertionError("network must not be touched when the tick is already taken")

    monkeypatch.setattr(run_paper, "binance_spot_prices", no_net)
    monkeypatch.setattr(run_paper, "hl_snapshot", no_net)
    assert run_paper.main() == 0
    assert "tick already taken today" in capsys.readouterr().out
    assert state_file.read_text() == before


def test_main_exits_3_without_data_and_leaves_state_untouched(monkeypatch, tmp_path, capsys):
    now = datetime(2026, 9, 2, 9, 0, tzinfo=timezone.utc)
    st = _synthetic_state(3, now - timedelta(days=3), now - timedelta(days=1), 9990.0)
    state_file = tmp_path / "paper_state.json"
    _write_state(state_file, st)
    before = state_file.read_text()
    monkeypatch.setattr(run_paper, "STATE_FILE", str(state_file))
    monkeypatch.setattr(run_paper, "_utcnow", lambda: now)

    def dead(*a, **k):
        raise NetworkError("GET failed after 6 attempts", ["a: SSLError: eof"])

    monkeypatch.setattr(run_paper, "binance_spot_prices", dead)
    rc = run_paper.main()
    assert rc == run_paper.EXIT_NO_DATA == 3
    out = capsys.readouterr().out
    assert "NO DATA" in out and "state unchanged" in out
    assert "Traceback" not in out
    assert state_file.read_text() == before


def test_main_exits_3_when_hl_snapshot_fails(monkeypatch, tmp_path):
    now = datetime(2026, 9, 2, 9, 0, tzinfo=timezone.utc)
    st = _synthetic_state(3, now - timedelta(days=3), now - timedelta(days=1), 9990.0)
    state_file = tmp_path / "paper_state.json"
    _write_state(state_file, st)
    before = state_file.read_text()
    monkeypatch.setattr(run_paper, "STATE_FILE", str(state_file))
    monkeypatch.setattr(run_paper, "_utcnow", lambda: now)
    monkeypatch.setattr(run_paper, "binance_spot_prices", lambda *a, **k: {"BTC": 1.0})

    def dead(*a, **k):
        raise hl.HyperliquidError("HL /info failed")

    monkeypatch.setattr(run_paper, "hl_snapshot", dead)
    assert run_paper.main() == 3
    assert state_file.read_text() == before


def test_main_exits_3_on_empty_spot_book(monkeypatch, tmp_path):
    """An empty price map would make the tick close every position — refuse it."""
    now = datetime(2026, 9, 2, 9, 0, tzinfo=timezone.utc)
    st = _synthetic_state(3, now - timedelta(days=3), now - timedelta(days=1), 9990.0)
    state_file = tmp_path / "paper_state.json"
    _write_state(state_file, st)
    before = state_file.read_text()
    monkeypatch.setattr(run_paper, "STATE_FILE", str(state_file))
    monkeypatch.setattr(run_paper, "_utcnow", lambda: now)
    monkeypatch.setattr(run_paper, "binance_spot_prices", lambda *a, **k: {})
    monkeypatch.setattr(run_paper, "hl_snapshot", lambda *a, **k: {"BTC": {"perp": 1.0}})
    assert run_paper.main() == 3
    assert state_file.read_text() == before


def test_main_happy_path_writes_state_and_prints_summary(monkeypatch, tmp_path, capsys):
    now = datetime(2026, 9, 2, 9, 0, tzinfo=timezone.utc)
    first = now - timedelta(days=42)          # Jul 22 .. Sep 2 = 42 calendar days
    st = _synthetic_state(26, first, now - timedelta(days=1), 9990.0)
    st["positions"] = {"BTC": {"spot": 100.0, "perp": 100.0, "entered": "t0"}}
    state_file = tmp_path / "paper_state.json"
    _write_state(state_file, st)
    monkeypatch.setattr(run_paper, "STATE_FILE", str(state_file))
    monkeypatch.setattr(run_paper, "_utcnow", lambda: now)
    monkeypatch.setattr(run_paper, "binance_spot_prices", lambda *a, **k: {"BTC": 101.0})
    monkeypatch.setattr(run_paper, "hl_snapshot",
                        lambda *a, **k: {"BTC": {"perp": 100.5, "funding_hourly": 0.0001,
                                                 "day_vol": 1e8}})
    monkeypatch.setattr(run_paper, "select_universe", lambda *a, **k: ["BTC"])
    monkeypatch.setattr(run_paper, "funding_since", lambda *a, **k: {"BTC": 0.001})
    assert run_paper.main() == 0
    saved = json.loads(state_file.read_text())
    assert len(saved["ticks"]) == 27
    assert saved["last_tick_ms"] == int(now.timestamp() * 1000)
    out = capsys.readouterr().out
    assert "=== PAPER TICK 2026-09-02 (tick #27) ===" in out
    assert "ticks: 27 of 42 calendar days" in out


# --------------------------------------------------------------------------- review fixes
def _hl_down(coin, start, end):
    raise hl.HyperliquidError("HL /info 503 after 3 attempts")


def test_select_universe_keeps_held_coin_when_funding_history_fails(monkeypatch):
    snap = {"BTC": {"perp": 1.0, "funding_hourly": 0.0001, "day_vol": 1e8},
            "ETH": {"perp": 1.0, "funding_hourly": 0.0001, "day_vol": 1e8}}
    spot = {"BTC": 1.0, "ETH": 1.0}
    monkeypatch.setattr(hl, "load_hl_funding_daily", _hl_down)
    # a held coin with no funding data is frozen (kept), an unheld one is simply not opened
    assert run_paper.select_universe(snap, spot, 0, held=["BTC"]) == ["BTC"]
    assert run_paper.select_universe(snap, spot, 0) == []


def test_main_freezes_held_positions_on_transient_funding_failure(monkeypatch, tmp_path):
    now = datetime(2026, 9, 2, 9, 0, tzinfo=timezone.utc)
    st = _synthetic_state(3, now - timedelta(days=3), now - timedelta(days=1), 9990.0)
    st["positions"] = {"BTC": {"spot": 100.0, "perp": 100.0, "entered": "t0"}}
    state_file = tmp_path / "paper_state.json"
    _write_state(state_file, st)
    monkeypatch.setattr(run_paper, "STATE_FILE", str(state_file))
    monkeypatch.setattr(run_paper, "_utcnow", lambda: now)
    monkeypatch.setattr(run_paper, "binance_spot_prices", lambda *a, **k: {"BTC": 100.0})
    monkeypatch.setattr(run_paper, "hl_snapshot",
                        lambda *a, **k: {"BTC": {"perp": 100.0, "funding_hourly": 0.0001,
                                                 "day_vol": 1e8}})
    monkeypatch.setattr(hl, "load_hl_funding_daily", _hl_down)
    assert run_paper.main() == 0
    saved = json.loads(state_file.read_text())
    assert "BTC" in saved["positions"]                       # not closed
    assert saved["cum_costs"] == pytest.approx(st["cum_costs"])   # no churn cost
    assert len(saved["ticks"]) == 4


def test_main_exits_0_when_another_tick_holds_the_lock(monkeypatch, tmp_path, capsys):
    import fcntl
    now = datetime(2026, 9, 2, 9, 0, tzinfo=timezone.utc)
    st = _synthetic_state(3, now - timedelta(days=3), now - timedelta(days=1), 9990.0)
    state_file = tmp_path / "paper_state.json"
    _write_state(state_file, st)
    before = state_file.read_text()
    monkeypatch.setattr(run_paper, "STATE_FILE", str(state_file))
    monkeypatch.setattr(run_paper, "_utcnow", lambda: now)

    def no_net(*a, **k):
        raise AssertionError("network must not be touched while another tick holds the lock")

    monkeypatch.setattr(run_paper, "binance_spot_prices", no_net)
    monkeypatch.setattr(run_paper, "hl_snapshot", no_net)
    with open(run_paper.lock_path(str(state_file)), "w") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)      # "another process" holds it
        assert run_paper.main() == 0
    assert "another tick in progress" in capsys.readouterr().out
    assert state_file.read_text() == before
