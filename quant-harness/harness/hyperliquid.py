"""Hyperliquid public API: дневные close+volume и funding (спека 0006).

HL — on-chain перп-DEX, API открыт без ключа (POST /info). Часовой funding
агрегируется в дневную сумму. Помесячный/по-диапазону parquet-кэш; ошибки API
поднимают HyperliquidError (без silent-фолбэков) — решение о пропуске за вызывающим.

ВНИМАНИЕ (разведка 2026-07-21): бары до 2024 — oracle-backfill с нулевым объёмом,
их использование = look-ahead. Вызывающий обязан стартовать с реального объёма.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests

from harness.netutil import NetworkError, post_json

INFO = "https://api.hyperliquid.xyz/info"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data_cache"

# Retry policy for POST /info. Loaders call _post per coin in a loop, so keep the
# per-call budget small: 3 attempts, waits of ~1s and ~2s (+ jitter).
HL_TRIES = 3
HL_BACKOFF = 1.0
HL_TIMEOUT = 30.0

# Injectable for tests (netutil falls back to time.sleep when None).
_sleep: Callable[[float], None] | None = None


class HyperliquidError(RuntimeError):
    pass


def _info_urls() -> list[str]:
    """POST /info endpoints: env ``HL_INFO_URLS`` (comma-separated) or the default."""
    raw = os.environ.get("HL_INFO_URLS", "")
    urls = [u.strip() for u in raw.split(",") if u.strip()]
    return urls or [INFO]


def _post(body: dict) -> Any:
    """POST /info with retries and host rotation; transport failure -> HyperliquidError."""
    try:
        return post_json(_info_urls(), body, tries=HL_TRIES, backoff=HL_BACKOFF,
                         timeout=HL_TIMEOUT, sleep=_sleep)
    except NetworkError as e:
        kind = body.get("type", "?") if isinstance(body, dict) else "?"
        raise HyperliquidError(f"HL /info {kind}: {e}") from e


def list_hl_perps() -> list:
    """Имена перпов из metaAndAssetCtxs (в порядке вселенной HL)."""
    meta = _post({"type": "metaAndAssetCtxs"})
    return [u["name"] for u in meta[0]["universe"]]


def _cached(cache_file: Path, cols):
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        if df.index.tz is None:
            df.index = pd.DatetimeIndex(df.index, tz="UTC")
        return df
    return None


def _write(cache_file: Path, df: pd.DataFrame):
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
    except (ImportError, OSError) as e:
        print(f"[warn] hl cache write failed ({e}); continuing uncached")


def load_hl_daily(coin: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Дневные бары: close + quote_volume (= base-volume × close). По возрастанию."""
    cache_file = Path(CACHE_DIR) / f"hl-1d-{coin}-{start_ms}-{end_ms}.parquet"
    hit = _cached(cache_file, ["close", "quote_volume"])
    if hit is not None:
        return hit
    rows = _post({"type": "candleSnapshot",
                  "req": {"coin": coin, "interval": "1d",
                          "startTime": start_ms, "endTime": end_ms}})
    if not isinstance(rows, list):
        raise HyperliquidError(f"candleSnapshot({coin}) unexpected: {str(rows)[:80]}")
    idx, close, qv = [], [], []
    for r in rows:
        ts = pd.to_datetime(int(r["t"]), unit="ms", utc=True)
        c = float(r["c"])
        idx.append(ts)
        close.append(c)
        qv.append(float(r["v"]) * c)
    df = pd.DataFrame({"close": close, "quote_volume": qv},
                      index=pd.DatetimeIndex(idx)).sort_index()
    _write(cache_file, df)
    return df


def load_hl_funding_daily(coin: str, start_ms: int, end_ms: int) -> pd.Series:
    """Часовой funding, агрегированный в ДНЕВНУЮ сумму (ts=день -> Σ ставок дня)."""
    cache_file = Path(CACHE_DIR) / f"hl-funding-{coin}-{start_ms}-{end_ms}.parquet"
    hit = _cached(cache_file, ["rate"])
    if hit is not None:
        return hit["rate"]
    # HL капит fundingHistory на 500 строк/вызов и отдаёт oldest-first —
    # пагинируем по времени, иначе теряется вся история дальше ~20 дней.
    rows, cur = [], start_ms
    seen = set()
    for _ in range(500):  # потолок страниц (500×500 = 250k часов истории)
        batch = _post({"type": "fundingHistory", "coin": coin,
                       "startTime": cur, "endTime": end_ms})
        if not isinstance(batch, list):
            raise HyperliquidError(f"fundingHistory({coin}) unexpected: {str(batch)[:80]}")
        fresh = [r for r in batch if int(r["time"]) not in seen]
        if not fresh:
            break
        rows.extend(fresh)
        seen.update(int(r["time"]) for r in fresh)
        last = max(int(r["time"]) for r in fresh)
        if len(batch) < 500 or last >= end_ms:
            break
        cur = last + 1
    if not rows:
        s = pd.Series(dtype=float, name="rate")
        _write(cache_file, s.to_frame())
        return s
    t = pd.to_datetime([int(r["time"]) for r in rows], unit="ms", utc=True)
    rate = np.array([float(r["fundingRate"]) for r in rows])
    hourly = pd.Series(rate, index=t).sort_index()
    daily = hourly.groupby(hourly.index.floor("D")).sum()
    daily.name = "rate"
    _write(cache_file, daily.to_frame())
    return daily
