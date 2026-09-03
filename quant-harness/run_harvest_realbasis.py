#!/usr/bin/env python3
"""Real-basis harvest (ночной): honest P&L с ФАКТИЧЕСКИМ базисом HL-перп/Binance-спот.

Спека 0008 полагала спот=перп (базис-P&L=0). Здесь: достаём Binance-спот для всех
хеджируемых HL-монет, считаем реальный базис-return, пересчитываем harvest.
Плюс: сравнение с допущением, per-coin атрибуция после базиса, стресс-археология.
Универс — ТОЛЬКО хеджируемые (исполнимые) монеты. Всё из API+кэша.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from harness import hyperliquid as hl
from harness.harvest import harvest_returns
from harness.xs import monthly_universe

CACHE = Path(__file__).resolve().parent / "data_cache"
START = "2024-01-01"
END = "2026-07-01"
THETA = 2 * (0.0005 + 0.0002) / 7
FLOOR = 1_000_000.0


def binance_spot_daily(base, start_ms, end_ms):
    cf = CACHE / f"binance-spot-{base}-{start_ms}-{end_ms}.parquet"
    if cf.exists():
        return pd.read_parquet(cf)["close"]
    j = requests.get("https://api.binance.com/api/v3/klines",
                     params={"symbol": base + "USDT", "interval": "1d",
                             "startTime": start_ms, "endTime": end_ms, "limit": 1000},
                     timeout=20).json()
    if not isinstance(j, list) or not j:
        return None
    idx = pd.to_datetime([k[0] for k in j], unit="ms", utc=True)
    s = pd.Series([float(k[4]) for k in j], index=idx, name="close").sort_index()
    try:
        cf.parent.mkdir(parents=True, exist_ok=True)
        s.to_frame().to_parquet(cf)
    except (ImportError, OSError):
        pass
    return s


def _hl_to_base(coin):
    return coin[1:].upper() if coin.startswith("k") and coin[1:] else coin.upper()


def main():
    t0 = time.time()
    start_ms = int(pd.Timestamp(START, tz="UTC").value // 1_000_000)
    end_ms = int(pd.Timestamp(END, tz="UTC").value // 1_000_000)
    index = pd.date_range(START, END, freq="1D", tz="UTC")[:-1]

    # Binance-спот-базы (хеджируемость)
    info = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=30).json()
    spot_bases = {s["baseAsset"] for s in info["symbols"]
                  if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"}

    perp_c, funds, qvs, spot_c = {}, {}, {}, {}
    n_hedge = 0
    for coin in hl.list_hl_perps():
        base = _hl_to_base(coin)
        if base not in spot_bases:
            continue  # только хеджируемые
        try:
            d = hl.load_hl_daily(coin, start_ms, end_ms)
            f = hl.load_hl_funding_daily(coin, start_ms, end_ms)
            sp = binance_spot_daily(base, start_ms, end_ms)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {coin}: {type(e).__name__}")
            continue
        if d.empty or sp is None:
            continue
        perp_c[coin] = d["close"]; funds[coin] = f; qvs[coin] = d["quote_volume"]
        spot_c[coin] = sp
        n_hedge += 1
    print(f"[data] хеджируемых монет с полными данными: {n_hedge}")

    PERP = pd.DataFrame(perp_c).reindex(index)
    F = pd.DataFrame(funds).reindex(index).fillna(0.0)
    Q = pd.DataFrame(qvs).reindex(index)
    SPOT = pd.DataFrame(spot_c).reindex(index)
    # реальный базис-return = spot_ret − perp_ret
    BASIS = (SPOT.pct_change() - PERP.pct_change())

    member = monthly_universe(Q, 60, 30, 90, skip_top=0, min_median_qv=FLOOR)

    assumed = harvest_returns(F, member, THETA, 7, 0.0005, 0.0002)          # спот=перп
    real = harvest_returns(F, member, THETA, 7, 0.0005, 0.0002, basis=BASIS)  # реальный базис

    # per-coin атрибуция после базиса: собранный funding + базис по монете
    trailing = F.where(member).rolling(7, min_periods=3).mean()
    hold = (trailing.shift(1) > THETA).fillna(False)
    coin_funding = (hold * F).sum(axis=0)
    coin_basis = (hold * BASIS.fillna(0.0)).sum(axis=0)
    coin_net = (coin_funding + coin_basis).sort_values()

    # стресс-археология: worst дни падения перпа по универсу — funding+базис
    held_prev = hold.shift(1).fillna(False)
    port_perp_ret = ((held_prev * PERP.pct_change()).sum(axis=1)
                     / held_prev.sum(axis=1).replace(0, np.nan)).astype(float)
    worst_days = port_perp_ret.dropna().nsmallest(5)
    stress = []
    for dt in worst_days.index:
        stress.append({"date": str(dt.date()),
                       "port_move": round(float(port_perp_ret.loc[dt]) * 100, 1),
                       "harvest_ret": round(float(real["returns"].loc[dt]) * 100, 3)})

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "hedgeable_coins": n_hedge,
        "assumed_yield_pct": round(assumed["annualized_yield"] * 100, 2),
        "real_basis_yield_pct": round(real["annualized_yield"] * 100, 2),
        "basis_drag_pct": round((assumed["annualized_yield"] - real["annualized_yield"]) * 100, 2),
        "real_sharpe": round(real["sharpe_illusory"], 2),
        "real_max_drawdown_pct": round(real["max_drawdown_from_data"] * 100, 2),
        "avg_positions": round(real["avg_positions"], 1),
        "worst_5_net_coins": {c: round(float(coin_net[c]), 4) for c in coin_net.head(5).index},
        "best_5_net_coins": {c: round(float(coin_net[c]), 4) for c in coin_net.tail(5).index},
        "n_coins_net_negative": int((coin_net < 0).sum()),
        "n_coins_net_positive": int((coin_net > 0).sum()),
        "stress_archaeology": stress,
    }
    with open("reports/harvest_realbasis.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[итог] хеджируемый универс {n_hedge} монет:")
    print(f"  доходность ДОПУЩЕНИЕ (спот=перп): {out['assumed_yield_pct']}%/год")
    print(f"  доходность РЕАЛЬНЫЙ БАЗИС:        {out['real_basis_yield_pct']}%/год")
    print(f"  базис-драг:                      {out['basis_drag_pct']} п.п.")
    print(f"  real Sharpe {out['real_sharpe']} | real MDD {out['real_max_drawdown_pct']}% | поз {out['avg_positions']}")
    print(f"  монет net+ : {out['n_coins_net_positive']} | net− : {out['n_coins_net_negative']}")
    print(f"  стресс-дни (падение портфеля -> harvest-ret):")
    for s in stress:
        print(f"    {s['date']}: перп {s['port_move']}% -> harvest {s['harvest_ret']}%")
    print(f"=== REAL-BASIS DONE in {out['elapsed_sec']}s. reports/harvest_realbasis.json ===")


if __name__ == "__main__":
    main()
