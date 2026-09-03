"""Data layer: Binance public perp data with a synthetic fallback.

Real path: download klines + funding from data.binance.vision (free bulk dumps).
Fallback: generate a synthetic OHLCV + funding series so the whole pipeline runs
end-to-end offline (for tests and for wiring things up before data is present).

The synthetic generator is deliberately EFFICIENT-MARKET by default: returns are
(near) zero-edge noise with realistic vol and fat tails. That is the honest
prior — if your harness "finds" a great strategy on this data, the harness is
broken (leakage), not brilliant. Use --synthetic-edge to inject a faint,
known momentum edge for validating that the pipeline can detect a real signal.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BINANCE_VISION = "https://data.binance.vision"
# Скачанные месяцы immutable — кэшируем, чтобы не тянуть 24 мес каждый прогон.
CACHE_DIR = Path(__file__).resolve().parents[1] / "data_cache"


@dataclass
class Bars:
    """OHLCV + funding on a single clock. index = UTC timestamps."""
    df: pd.DataFrame  # columns: open, high, low, close, volume, funding

    def __len__(self) -> int:
        return len(self.df)


def load_binance_monthly_klines(symbol: str, interval: str, year: int, month: int) -> pd.DataFrame:
    """Download one month of USD-M futures klines. Returns OHLCV DataFrame.

    Zero-cost source. Raises on network/availability failure so the caller can
    decide to fall back to synthetic.
    """
    cache_file = Path(CACHE_DIR) / f"{symbol}-{interval}-{year:04d}-{month:02d}.parquet"
    if cache_file.exists():
        cached = pd.read_parquet(cache_file)
        # Старый формат кэша (без quote_volume) не годится для universe-ранжирования.
        if "quote_volume" in cached.columns:
            return cached
    url = (f"{BINANCE_VISION}/data/futures/um/monthly/klines/"
           f"{symbol}/{interval}/{symbol}-{interval}-{year:04d}-{month:02d}.zip")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        name = z.namelist()[0]
        raw = pd.read_csv(z.open(name), header=None)
    # Header row presence varies across archive vintages: current um/futures
    # dumps ship "open_time,..." as line 1, older ones may not.
    if str(raw.iloc[0, 0]) == "open_time":
        raw = raw.iloc[1:].reset_index(drop=True)
    # Binance kline schema: open_time, open, high, low, close, volume, close_time,
    # quote_volume, ... — quote_volume нужен для point-in-time universe.
    raw = raw.iloc[:, [0, 1, 2, 3, 4, 5, 7]]
    raw.columns = ["open_time", "open", "high", "low", "close", "volume", "quote_volume"]
    raw["ts"] = pd.to_datetime(raw["open_time"].astype("int64"), unit="ms", utc=True)
    out = raw.set_index("ts")[["open", "high", "low", "close", "volume",
                               "quote_volume"]].astype(float)
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_file)
    except (ImportError, OSError) as e:  # нет pyarrow / нет прав — работаем без кэша
        print(f"[warn] kline cache write failed ({e}); continuing uncached")
    return out


def make_synthetic(
    n: int = 4000,
    interval_hours: int = 1,
    seed: int = 0,
    edge: float = 0.0,
    funding_bps_per_8h: float = 1.0,
) -> Bars:
    """Synthetic OHLCV + funding. edge=0 -> zero-edge noise (honest null).

    edge>0 injects a small AR(1) momentum component so a correct pipeline can
    demonstrate it *can* detect a real signal (sanity check on detection power).
    Returns are Student-t (fat tails) scaled to a realistic ~1%/bar vol.
    """
    rng = np.random.default_rng(seed)
    t_noise = rng.standard_t(df=4, size=n) * 0.007
    if edge > 0:
        mom = np.zeros(n)
        for i in range(1, n):
            mom[i] = 0.85 * mom[i - 1] + rng.normal(0, 1)
        rets = t_noise + edge * (mom / np.std(mom))
    else:
        rets = t_noise
    price = 30000.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2021-01-01", periods=n, freq=f"{interval_hours}h", tz="UTC")
    close = pd.Series(price, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.001, n)))
    volume = pd.Series(np.abs(rng.normal(1000, 200, n)), index=idx)
    # Funding realized every 8h; small mean, noisy sign.
    funding = pd.Series(0.0, index=idx)
    every = max(1, 8 // interval_hours)
    fmask = (np.arange(n) % every) == 0
    funding.iloc[np.where(fmask)[0]] = (
        rng.normal(funding_bps_per_8h, 3.0, fmask.sum()) / 1e4
    )
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": volume, "funding": funding}
    )
    return Bars(df)


def load_funding_rest(symbol: str, start_ms: int, end_ms: int) -> pd.Series:
    """Historical funding via Binance fapi (paginated). Reachable on your machine.

    Returns a Series of funding rates indexed by UTC timestamp. Defensive: on any
    failure returns an EMPTY series so the caller can fall back to zero funding
    with an explicit warning rather than crashing an unattended run.
    """
    cache_file = Path(CACHE_DIR) / f"funding-{symbol}-{start_ms}-{end_ms}.parquet"
    if cache_file.exists():
        s = pd.read_parquet(cache_file)["rate"]
        return s if not s.empty else pd.Series(dtype=float)
    out = {}
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    cur = start_ms
    try:
        for _ in range(50):  # hard cap on pages
            resp = requests.get(url, params={"symbol": symbol, "startTime": cur,
                                             "endTime": end_ms, "limit": 1000}, timeout=30)
            resp.raise_for_status()
            rows = resp.json()
            if not rows:
                break
            for r in rows:
                out[pd.to_datetime(int(r["fundingTime"]), unit="ms", utc=True)] = float(r["fundingRate"])
            last = int(rows[-1]["fundingTime"])
            if len(rows) < 1000 or last >= end_ms:
                break
            cur = last + 1
    except Exception as e:  # noqa: BLE001
        print(f"[warn] funding fetch failed for {symbol}: {e}; using funding=0")
        return pd.Series(dtype=float)
    s = pd.Series(out, dtype=float).sort_index()
    s.name = "rate"
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        s.to_frame().to_parquet(cache_file)
    except (ImportError, OSError) as e:
        print(f"[warn] funding cache write failed ({e}); continuing uncached")
    return s if not s.empty else pd.Series(dtype=float)


def _attach_funding(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    start_ms = int(df.index[0].value // 1_000_000)
    end_ms = int(df.index[-1].value // 1_000_000)
    fund = load_funding_rest(symbol, start_ms, end_ms)
    df["funding"] = 0.0
    if not fund.empty:
        # Binance fundingTime carries ms jitter (e.g. ...00005), so an exact
        # reindex silently drops those payments. Floor each payment to the bar
        # containing it and sum (several payments may land on one bar).
        pos = df.index.searchsorted(fund.index, side="right") - 1
        vals = np.zeros(len(df))
        for p, v in zip(pos, fund.values):
            if 0 <= p < len(df):
                vals[p] += v
        df["funding"] = vals
        print(f"[ok] funding attached for {symbol}: {int((df['funding']!=0).sum())} payments")
    else:
        print(f"[warn] no funding for {symbol} — carry results NOT trustworthy (funding=0)")
    return df


def load(symbol="BTCUSDT", interval="1h", months=None, synthetic=False,
         synthetic_edge=0.0, n_synth=4000, seed=0, with_funding=False) -> Bars:
    """Top-level loader. Tries Binance; falls back to synthetic on any failure
    (or when synthetic=True). Always returns a Bars with a `funding` column.
    """
    if synthetic:
        return make_synthetic(n=n_synth, seed=seed, edge=synthetic_edge)
    months = months or [(2024, 1), (2024, 2), (2024, 3)]
    frames = []
    for (y, m) in months:
        frames.append(load_binance_monthly_klines(symbol, interval, y, m))
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if with_funding:
        df = _attach_funding(df, symbol)
    else:
        # Placeholder zero makes the omission explicit, not silent.
        df["funding"] = 0.0
    return Bars(df)


S3_LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"


def list_all_um_symbols() -> list:
    """Все USD-M перпы, когда-либо имевшие дампы — из S3-листинга бакета,
    ВКЛЮЧАЯ делистнутые (спека 0002: убирает survivorship-bias кандидатского
    уровня). Оставляем только *USDT, без квартальных контрактов (суффикс _YYMMDD).
    """
    import xml.etree.ElementTree as ET
    prefix = "data/futures/um/monthly/klines/"
    params = {"list-type": "2", "prefix": prefix, "delimiter": "/"}
    out = set()
    for _ in range(100):  # жёсткий потолок страниц
        resp = requests.get(S3_LIST_URL, params=params, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        token, truncated = None, False
        for el in root.iter():
            tag = el.tag.rsplit("}", 1)[-1]
            if tag == "Prefix" and el.text and el.text != prefix:
                sym = el.text[len(prefix):].strip("/")
                if sym.endswith("USDT") and "_" not in sym:
                    out.add(sym)
            elif tag == "NextContinuationToken":
                token = el.text
            elif tag == "IsTruncated":
                truncated = (el.text or "").strip().lower() == "true"
        if not (truncated and token):
            break
        params = {**params, "continuation-token": token}
    return sorted(out)


def load_panel(symbols, interval, months, with_funding=True):
    """Панель для cross-sectional семейства: (closes, funding, quote_volume),
    каждая — DataFrame (union-индекс × символы).

    Отличия от load(): НИКАКОГО synthetic-фолбэка (тихая синтетика в панели —
    яд); пропавший месяц символа (404 до листинга / делистинг) — предупреждение
    и NaN, символ остаётся в панели с той историей, что есть.
    """
    closes, qvs, funds = {}, {}, {}
    for sym in symbols:
        frames = []
        for (y, m) in months:
            try:
                frames.append(load_binance_monthly_klines(sym, interval, y, m))
            except (requests.RequestException, zipfile.BadZipFile) as e:
                print(f"[warn] {sym} {y:04d}-{m:02d}: {e}; месяц пропущен")
        if not frames:
            print(f"[warn] {sym}: данных нет вообще — исключён из панели")
            continue
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="first")]
        if with_funding:
            df = _attach_funding(df, sym)
        else:
            df["funding"] = 0.0
        closes[sym] = df["close"]
        qvs[sym] = df["quote_volume"]
        funds[sym] = df["funding"]
    C = pd.DataFrame(closes)
    F = pd.DataFrame(funds).reindex(C.index).fillna(0.0)
    Q = pd.DataFrame(qvs).reindex(C.index)
    return C, F, Q
