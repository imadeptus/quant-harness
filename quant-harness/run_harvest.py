#!/usr/bin/env python3
"""Прогон спеки 0008 — funding harvest доходность+риск, из кэша (автономно).

--venue binance | hyperliquid. Считает доходную сторону по данным и хвост через
стресс. Итог — профиль «доходность против стресса», НЕ PASS/KILL.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import pandas as pd

from harness.harvest import harvest_returns, basis_stress, funding_flip_stress
from harness.xs import monthly_universe

TAKER, SLIP = 0.0005, 0.0002
THETA = 2.0 * (TAKER + SLIP) / 7.0     # недельный сбор должен покрыть round-trip
TRAIL = 7


def load_binance():
    from harness.data import list_all_um_symbols, load_panel
    months = []
    y, m = 2024, 7
    for _ in range(24):
        months.append((y, m)); m += 1
        if m > 12: y, m = y + 1, 1
    _, funding, qv = load_panel(list_all_um_symbols(), "1d", months)
    return funding, qv, 2_000_000.0, 365


def load_hl():
    from harness import hyperliquid as hl
    start = int(pd.Timestamp("2024-01-01", tz="UTC").value // 1_000_000)
    end = int(pd.Timestamp("2026-07-01", tz="UTC").value // 1_000_000)
    index = pd.date_range("2024-01-01", "2026-07-01", freq="1D", tz="UTC")[:-1]
    funds, qvs = {}, {}
    for c in hl.list_hl_perps():
        try:
            d = hl.load_hl_daily(c, start, end)
            f = hl.load_hl_funding_daily(c, start, end)
        except Exception:  # noqa: BLE001
            continue
        if d.empty:
            continue
        qvs[c] = d["quote_volume"]; funds[c] = f
    F = pd.DataFrame(funds).reindex(index).fillna(0.0)
    Q = pd.DataFrame(qvs).reindex(index)
    return F, Q, 1_000_000.0, 365


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", choices=("binance", "hyperliquid"), required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or f"reports/harvest0008_{args.venue}"

    t0 = time.time()
    if args.venue == "binance":
        funding, qv, floor, ppy = load_binance()
    else:
        funding, qv, floor, ppy = load_hl()
    member = monthly_universe(qv, top_k=60, vol_window=30, min_history=90,
                              skip_top=0, min_median_qv=floor)
    print(f"[data] {args.venue}: {funding.shape[1]} монет, {funding.shape[0]} баров")

    res = harvest_returns(funding, member, theta=THETA, trail=TRAIL,
                          taker=TAKER, slip=SLIP, periods_per_year=ppy)
    stresses = {f"basis_{int(g*100)}pct": basis_stress(res, gap=g)
                for g in (0.01, 0.03, 0.05)}
    flip = funding_flip_stress(res, flip_rate=-THETA, days=30)

    print(f"[yield] {args.venue}: ann {res['annualized_yield']*100:.1f}% | "
          f"Sharpe(иллюз) {res['sharpe_illusory']:.1f} | "
          f"data-MDD {res['max_drawdown_from_data']*100:.1f}% | "
          f"avg позиций {res['avg_positions']:.0f}")
    for k, s in stresses.items():
        print(f"  стресс {k}: разовый убыток {s['stress_loss']*100:.0f}% = "
              f"{s['months_of_yield_lost']:.1f} мес доходности")
    print(f"  funding-flip 30д: {flip['flip_loss']*100:.2f}%")

    out = {
        "spec": "0008", "venue": args.venue,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "theta": THETA, "costs": {"taker": TAKER, "slip": SLIP},
        "annualized_yield": round(res["annualized_yield"], 4),
        "sharpe_illusory": round(res["sharpe_illusory"], 2),
        "max_drawdown_from_data": round(res["max_drawdown_from_data"], 4),
        "avg_positions": round(res["avg_positions"], 1),
        "avg_turnover": round(float(res["turnover"].mean()), 4),
        "stress": {k: {kk: (round(vv, 2) if isinstance(vv, float) else vv)
                       for kk, vv in s.items()} for k, s in stresses.items()},
        "funding_flip_30d": flip,
    }
    with open(out_path + ".json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"=== HARVEST {args.venue} DONE in {out['elapsed_sec']}s. {out_path}.json ===")


if __name__ == "__main__":
    main()
