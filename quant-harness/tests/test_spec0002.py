"""Спека 0002: S3-перечень символов (с делистнутыми) и universe-band."""
import numpy as np
import pandas as pd

from harness import data
from harness.xs import monthly_universe

_PAGE1 = """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Prefix>data/futures/um/monthly/klines/</Prefix>
  <IsTruncated>true</IsTruncated>
  <NextContinuationToken>tok1</NextContinuationToken>
  <CommonPrefixes><Prefix>data/futures/um/monthly/klines/BTCUSDT/</Prefix></CommonPrefixes>
  <CommonPrefixes><Prefix>data/futures/um/monthly/klines/BTCUSDT_210625/</Prefix></CommonPrefixes>
  <CommonPrefixes><Prefix>data/futures/um/monthly/klines/ETHBUSD/</Prefix></CommonPrefixes>
</ListBucketResult>"""

_PAGE2 = """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Prefix>data/futures/um/monthly/klines/</Prefix>
  <IsTruncated>false</IsTruncated>
  <CommonPrefixes><Prefix>data/futures/um/monthly/klines/DELISTEDUSDT/</Prefix></CommonPrefixes>
</ListBucketResult>"""


def test_list_all_um_symbols_paginated_and_filtered(monkeypatch):
    calls = []

    class _R:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            pass

    def fake_get(url, params=None, timeout=0):
        calls.append(dict(params or {}))
        return _R(_PAGE2 if params.get("continuation-token") else _PAGE1)

    monkeypatch.setattr(data.requests, "get", fake_get)
    syms = data.list_all_um_symbols()
    assert syms == ["BTCUSDT", "DELISTEDUSDT"], \
        "только *USDT, без квартальных (_YYMMDD) и BUSD; делистнутые включены"
    assert len(calls) == 2 and calls[1]["continuation-token"] == "tok1"


def test_universe_band_skip_top_and_liquidity_floor():
    idx = pd.date_range("2025-01-01", periods=120, freq="1D", tz="UTC")
    vols = {"A": 1000.0, "B": 900.0, "C": 500.0, "D": 100.0, "E": 4.0, "F": 3.0}
    qv = pd.DataFrame({s: np.full(len(idx), v) for s, v in vols.items()}, index=idx)
    m = monthly_universe(qv, top_k=3, vol_window=30, min_history=30,
                        skip_top=2, min_median_qv=5.0)
    row = m.loc[pd.Timestamp("2025-03-10", tz="UTC")]
    # топ-2 (A, B) пропущены; floor=5 отсекает E, F; остаются C и D (2 < top_k=3)
    assert not row["A"] and not row["B"]
    assert row["C"] and row["D"]
    assert not row["E"] and not row["F"]
