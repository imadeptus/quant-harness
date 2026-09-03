"""Hyperliquid-лоадеры (спека 0006): дневные close+volume и funding, кэш, без фолбэков."""
import pandas as pd
import pytest

from harness import hyperliquid as hl


class _R:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def _candles(coin, n, t0=1704067200000):
    return [{"t": t0 + i * 86_400_000, "T": t0 + (i + 1) * 86_400_000 - 1, "s": coin,
             "i": "1d", "o": "100", "c": str(100 + i), "h": "110", "l": "90",
             "v": str(10 + i), "n": 5} for i in range(n)]


def test_list_hl_perps(monkeypatch):
    meta = [{"universe": [{"name": "BTC"}, {"name": "ETH"}, {"name": "VINE"}]},
            [{"funding": "0.00001"}, {"funding": "0.00002"}, {"funding": "0.0003"}]]
    monkeypatch.setattr(hl.requests, "post", lambda url, json=None, timeout=0: _R(meta))
    assert hl.list_hl_perps() == ["BTC", "ETH", "VINE"]


def test_load_hl_daily_close_and_qv_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(hl, "CACHE_DIR", tmp_path, raising=False)
    calls = {"n": 0}

    def fake_post(url, json=None, timeout=0):
        calls["n"] += 1
        return _R(_candles("BTC", 5))

    monkeypatch.setattr(hl.requests, "post", fake_post)
    df = hl.load_hl_daily("BTC", 1704067200000, 1704067200000 + 5 * 86_400_000)
    assert calls["n"] == 1
    assert list(df["close"].values) == [100, 101, 102, 103, 104]
    # quote-объём = base-volume * close (для пола ликвидности)
    assert df["quote_volume"].iloc[0] == pytest.approx(10 * 100)
    assert df.index.is_monotonic_increasing

    monkeypatch.setattr(hl.requests, "post",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("сеть")))
    df2 = hl.load_hl_daily("BTC", 1704067200000, 1704067200000 + 5 * 86_400_000)
    pd.testing.assert_frame_equal(df, df2)


def test_load_hl_funding_daily_aggregates_hourly(tmp_path, monkeypatch):
    monkeypatch.setattr(hl, "CACHE_DIR", tmp_path, raising=False)
    day0 = 1704067200000
    # три часовых платежа в дне 0, два в дне 1
    rows = [{"coin": "BTC", "fundingRate": "0.0001", "time": day0 + 0},
            {"coin": "BTC", "fundingRate": "0.0001", "time": day0 + 3_600_000},
            {"coin": "BTC", "fundingRate": "0.0002", "time": day0 + 7_200_000},
            {"coin": "BTC", "fundingRate": "0.0005", "time": day0 + 86_400_000},
            {"coin": "BTC", "fundingRate": "0.0005", "time": day0 + 86_400_000 + 3_600_000}]
    monkeypatch.setattr(hl.requests, "post",
                        lambda url, json=None, timeout=0: _R(rows))
    s = hl.load_hl_funding_daily("BTC", day0, day0 + 2 * 86_400_000)
    # день 0: 0.0001+0.0001+0.0002 = 0.0004 ; день 1: 0.0005+0.0005 = 0.0010
    assert s.iloc[0] == pytest.approx(0.0004)
    assert s.iloc[1] == pytest.approx(0.0010)
    assert (s.index == s.index.floor("D")).all(), "индекс должен быть дневным"


def test_load_hl_funding_paginates_past_500_cap(tmp_path, monkeypatch):
    """HL капит fundingHistory на 500 строк/вызов — лоадер обязан пагинировать
    по времени, иначе далёкая история теряется (реальный баг spec 0006 smoke)."""
    monkeypatch.setattr(hl, "CACHE_DIR", tmp_path, raising=False)
    day0 = 1704067200000
    hour = 3_600_000
    total_hours = 1300  # > 2 страниц по 500
    all_rows = [{"coin": "BTC", "fundingRate": "0.0001", "time": day0 + i * hour}
                for i in range(total_hours)]
    calls = {"n": 0}

    def paged_post(url, json=None, timeout=0):
        calls["n"] += 1
        st = json["startTime"]
        batch = [r for r in all_rows if r["time"] >= st][:500]  # cap 500, oldest-first
        return _R(batch)

    monkeypatch.setattr(hl.requests, "post", paged_post)
    s = hl.load_hl_funding_daily("BTC", day0, day0 + total_hours * hour)
    assert calls["n"] >= 3, "лоадер не пагинирует (одна страница = потеря истории)"
    # все 1300 часов по 0.0001 -> сумма == 0.13; проверим полноту через сумму
    assert s.sum() == pytest.approx(1300 * 0.0001), "потеряны платежи за пределами 1-й страницы"
    # покрыто ~54 дня, а не 20
    assert len(s) >= 50


def test_load_hl_funding_raises_on_non_list(monkeypatch, tmp_path):
    monkeypatch.setattr(hl, "CACHE_DIR", tmp_path, raising=False)
    monkeypatch.setattr(hl.requests, "post",
                        lambda url, json=None, timeout=0: _R({"error": "bad"}))
    with pytest.raises(hl.HyperliquidError):
        hl.load_hl_funding_daily("BTC", 1704067200000, 1704153600000)
