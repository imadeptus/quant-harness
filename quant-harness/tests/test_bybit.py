"""Bybit-лоадеры (спека 0003): funding и 1d klines, parquet-кэш, без фолбэков."""
import pandas as pd
import pytest

from harness import bybit


class _R:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def _funding_payload(rows):
    return {"retCode": 0, "result": {"list": [
        {"fundingRateTimestamp": str(ts), "fundingRate": str(r)} for ts, r in rows]}}


def test_bybit_funding_month_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(bybit, "CACHE_DIR", tmp_path, raising=False)
    calls = {"n": 0}
    t1 = 1704096000000  # 2024-01-01 08:00 UTC
    t2 = 1704124800000  # 2024-01-01 16:00 UTC

    def fake_get(url, params=None, timeout=0):
        calls["n"] += 1
        return _R(_funding_payload([(t2, 0.0002), (t1, 0.0001)]))  # newest first

    monkeypatch.setattr(bybit.requests, "get", fake_get)
    s1 = bybit.load_funding_month("BTCUSDT", 2024, 1)
    assert calls["n"] == 1
    assert list(s1.values) == [0.0001, 0.0002], "должно быть отсортировано по времени"

    monkeypatch.setattr(bybit.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("сеть")))
    s2 = bybit.load_funding_month("BTCUSDT", 2024, 1)
    pd.testing.assert_series_equal(s1, s2)


def test_bybit_funding_raises_on_api_error(monkeypatch, tmp_path):
    monkeypatch.setattr(bybit, "CACHE_DIR", tmp_path, raising=False)
    monkeypatch.setattr(bybit.requests, "get",
                        lambda *a, **k: _R({"retCode": 10001, "retMsg": "bad"}))
    with pytest.raises(bybit.BybitError):
        bybit.load_funding_month("BTCUSDT", 2024, 1)


def test_bybit_daily_closes_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(bybit, "CACHE_DIR", tmp_path, raising=False)
    t0 = 1704067200000  # 2024-01-01
    rows = [[str(t0 + i * 86_400_000), "1", "2", "0.5", str(100 + i), "9", "9"]
            for i in range(31)][::-1]  # newest first, как отдаёт Bybit

    def fake_get(url, params=None, timeout=0):
        return _R({"retCode": 0, "result": {"list": rows}})

    monkeypatch.setattr(bybit.requests, "get", fake_get)
    s1 = bybit.load_daily_closes_month("BTCUSDT", 2024, 1)
    assert len(s1) == 31 and s1.iloc[0] == 100.0 and s1.iloc[-1] == 130.0
    assert s1.index.is_monotonic_increasing

    monkeypatch.setattr(bybit.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("сеть")))
    s2 = bybit.load_daily_closes_month("BTCUSDT", 2024, 1)
    pd.testing.assert_series_equal(s1, s2)
