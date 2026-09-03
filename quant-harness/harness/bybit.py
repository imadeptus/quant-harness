"""Bybit v5 public data: funding-история и дневные закрытия (спека 0003).

Дисциплина: помесячный parquet-кэш (история immutable), никаких silent-фолбэков
— ошибка API поднимает BybitError, решение о пропуске месяца за вызывающим.
"""
from __future__ import annotations

import calendar
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.bybit.com"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data_cache"


class BybitError(RuntimeError):
    pass


def _month_bounds_ms(year: int, month: int):
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    end = start + pd.Timedelta(days=calendar.monthrange(year, month)[1])
    return int(start.value // 1_000_000), int(end.value // 1_000_000)


def _get(path: str, params: dict):
    resp = requests.get(BASE + path, params=params, timeout=30)
    resp.raise_for_status()
    j = resp.json()
    if j.get("retCode") != 0:
        raise BybitError(f"{path}: retCode={j.get('retCode')} {j.get('retMsg')}")
    return j["result"]["list"]


def _cached_series(cache_file: Path, name: str):
    if cache_file.exists():
        s = pd.read_parquet(cache_file)[name]
        s.index = pd.DatetimeIndex(s.index, tz="UTC") if s.index.tz is None else s.index
        return s
    return None


def _write_cache(cache_file: Path, s: pd.Series):
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        s.to_frame().to_parquet(cache_file)
    except (ImportError, OSError) as e:
        print(f"[warn] bybit cache write failed ({e}); continuing uncached")


def load_funding_month(symbol: str, year: int, month: int) -> pd.Series:
    """Funding-платежи символа за месяц (ts -> rate), отсортировано по времени."""
    cache_file = Path(CACHE_DIR) / f"bybit-funding-{symbol}-{year:04d}-{month:02d}.parquet"
    cached = _cached_series(cache_file, "rate")
    if cached is not None:
        return cached
    start, end = _month_bounds_ms(year, month)
    out = {}
    cur_end = end
    for _ in range(10):  # ~90 платежей/мес => обычно 1 страница; потолок для надёжности
        rows = _get("/v5/market/funding/history",
                    {"category": "linear", "symbol": symbol,
                     "startTime": start, "endTime": cur_end, "limit": 200})
        if not rows:
            break
        for r in rows:
            ts = pd.to_datetime(int(r["fundingRateTimestamp"]), unit="ms", utc=True)
            out[ts] = float(r["fundingRate"])
        oldest = min(int(r["fundingRateTimestamp"]) for r in rows)
        if len(rows) < 200 or oldest <= start:
            break
        cur_end = oldest - 1
    s = pd.Series(out, dtype=float).sort_index()
    s.name = "rate"
    _write_cache(cache_file, s)
    return s


def load_daily_closes_month(symbol: str, year: int, month: int) -> pd.Series:
    """Дневные close символа за месяц (ts -> close), по возрастанию времени."""
    cache_file = Path(CACHE_DIR) / f"bybit-close1d-{symbol}-{year:04d}-{month:02d}.parquet"
    cached = _cached_series(cache_file, "close")
    if cached is not None:
        return cached
    start, end = _month_bounds_ms(year, month)
    rows = _get("/v5/market/kline",
                {"category": "linear", "symbol": symbol, "interval": "D",
                 "start": start, "end": end - 1, "limit": 1000})
    out = {pd.to_datetime(int(r[0]), unit="ms", utc=True): float(r[4]) for r in rows}
    s = pd.Series(out, dtype=float).sort_index()
    s.name = "close"
    _write_cache(cache_file, s)
    return s


def linear_symbols() -> set:
    """Текущие линейные USDT-перпы Bybit (для пересечения листингов)."""
    out = set()
    cursor = None
    for _ in range(20):
        params = {"category": "linear", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(BASE + "/v5/market/instruments-info", params=params, timeout=30)
        resp.raise_for_status()
        j = resp.json()
        if j.get("retCode") != 0:
            raise BybitError(f"instruments-info: {j.get('retMsg')}")
        res = j["result"]
        for it in res.get("list", []):
            sym = it.get("symbol", "")
            if sym.endswith("USDT") and it.get("contractType") == "LinearPerpetual":
                out.add(sym)
        cursor = res.get("nextPageCursor")
        if not cursor:
            break
    return out
