"""Klines, скачанные один раз, должны читаться из parquet-кэша без сети."""
import io
import zipfile

import pandas as pd

from harness import data


def _fake_zip_bytes() -> bytes:
    # Сутки часовых баров в формате текущих дампов Binance (с header row).
    rows = ["open_time,open,high,low,close,volume,close_time,quote_volume,count,tb,tq,ig"]
    t0 = 1704067200000  # 2024-01-01 00:00 UTC
    for i in range(24):
        ts = t0 + i * 3_600_000
        rows.append(f"{ts},100,101,99,100.5,10,{ts + 3599999},1000,50,5,500,0")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("BTCUSDT-1h-2024-01.csv", "\n".join(rows))
    return buf.getvalue()


class _Resp:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:
        pass


def test_monthly_klines_cached_to_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path, raising=False)
    calls = {"n": 0}

    def fake_get(url, timeout=0):
        calls["n"] += 1
        return _Resp(_fake_zip_bytes())

    monkeypatch.setattr(data.requests, "get", fake_get)
    df1 = data.load_binance_monthly_klines("BTCUSDT", "1h", 2024, 1)
    assert calls["n"] == 1
    assert len(df1) == 24
    assert len(list(tmp_path.glob("*.parquet"))) == 1, "месяц не закэширован"

    def dead_get(url, timeout=0):
        raise AssertionError("сеть тронута, хотя месяц в кэше")

    monkeypatch.setattr(data.requests, "get", dead_get)
    df2 = data.load_binance_monthly_klines("BTCUSDT", "1h", 2024, 1)
    pd.testing.assert_frame_equal(df1, df2)


def test_loader_keeps_quote_volume_and_refreshes_stale_cache(tmp_path, monkeypatch):
    """Universe-ранжирование требует quote_volume; старый кэш без него — miss."""
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path, raising=False)
    monkeypatch.setattr(data.requests, "get",
                        lambda url, timeout=0: _Resp(_fake_zip_bytes()))
    df = data.load_binance_monthly_klines("BTCUSDT", "1h", 2024, 1)
    assert "quote_volume" in df.columns
    assert (df["quote_volume"] == 1000).all()

    stale = df.drop(columns=["quote_volume"])
    stale.to_parquet(tmp_path / "BTCUSDT-1h-2024-01.parquet")
    df2 = data.load_binance_monthly_klines("BTCUSDT", "1h", 2024, 1)
    assert "quote_volume" in df2.columns, "stale-кэш без quote_volume отдан как есть"


def test_funding_rest_cached_per_range(tmp_path, monkeypatch):
    """funding по (symbol,start,end) кэшируется: повторный вызов не трогает сеть."""
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path, raising=False)
    calls = {"n": 0}
    pay = [{"fundingTime": 1704096000000, "fundingRate": "0.0001"},
           {"fundingTime": 1704124800000, "fundingRate": "0.0002"}]

    class _FR:
        def raise_for_status(self):
            pass

        def json(self):
            return pay if calls_first() else []

    def calls_first():
        return calls["n"] == 1

    def fake_get(url, params=None, timeout=0):
        calls["n"] += 1
        return _FR()

    monkeypatch.setattr(data.requests, "get", fake_get)
    s1 = data.load_funding_rest("BTCUSDT", 1704067200000, 1704153600000)
    assert len(s1) == 2 and calls["n"] >= 1

    monkeypatch.setattr(data.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("сеть")))
    s2 = data.load_funding_rest("BTCUSDT", 1704067200000, 1704153600000)
    assert list(s2.values) == list(s1.values), "range funding не взят из кэша"


def _month_frame(y, m, days, base):
    idx = pd.date_range(f"{y:04d}-{m:02d}-01", periods=days * 24, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": base, "high": base, "low": base, "close": base,
         "volume": 1.0, "quote_volume": base * 10}, index=idx)


def test_load_panel_aligns_skips_missing_and_attaches_funding(monkeypatch):
    """Панель: union-индекс, пропавший месяц -> NaN (не synthetic!), funding по ногам."""
    def fake_month(symbol, interval, year, month):
        if symbol == "Y" and month == 2:
            raise data.requests.RequestException("404")
        return _month_frame(year, month, 28, 100.0 if symbol == "X" else 50.0)

    monkeypatch.setattr(data, "load_binance_monthly_klines", fake_month)
    pay_ts = pd.Timestamp("2024-01-05 08:00:00.005", tz="UTC")
    monkeypatch.setattr(data, "load_funding_rest",
                        lambda sym, s, e: pd.Series({pay_ts: 0.0001}))

    closes, funding, qv = data.load_panel(["X", "Y"], "1h",
                                          [(2024, 1), (2024, 2)], with_funding=True)
    assert list(closes.columns) == ["X", "Y"]
    assert closes["X"].notna().all()
    assert closes.loc["2024-02", "Y"].isna().all(), "пропавший месяц должен быть NaN"
    assert closes.loc["2024-01", "Y"].notna().all()
    # funding с ms-джиттером лёг на содержащий бар 08:00
    assert funding.loc[pd.Timestamp("2024-01-05 08:00", tz="UTC"), "X"] == 0.0001
    assert (qv.loc["2024-01", "X"] == 1000.0).all()
