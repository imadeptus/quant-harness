#!/usr/bin/env python3
"""Прогон спеки 0006 — XS carry/momentum на Hyperliquid, 1d.

Вселенная HL (on-chain перп-DEX) → band рангов 4–40 по объёму (исключить топ-3,
пол $1M) → XS carry/momentum → судья CPCV+DSR+PBO. Старт строго с реального
объёма (пре-2024 = backfill, исключён). Переиспользует xs.py и runner.py.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from harness import hyperliquid as hl
from harness.backtest import Costs
from harness.pbo import pbo_cscv
from harness.runner import Thresholds, run_cpcv_returns
from harness.walk_forward import CPCVConfig
from harness.xs import monthly_universe, run_xs_backtest, xs_carry_weights, xs_momentum_weights

CAR_GRID = [{"family": "xs_carry", "window": F, "quantile": q, "rebalance": R}
            for F in (3, 7, 14) for q in (0.20, 1 / 3) for R in (1, 7)]      # 12
MOM_GRID = [{"family": "xs_momentum", "lookback": L, "skip": 1, "quantile": q, "rebalance": R}
            for L in (7, 14, 30) for q in (0.20, 1 / 3) for R in (1, 7)]     # 12

TAKER, SLIP = 0.0005, 0.0010          # 5 bps + 10 bps (тонкий хвост HL)
MIN_MEDIAN_QV = 1_000_000.0           # пол ликвидности $1M
SKIP_TOP, TOP_K = 3, 37               # band рангов 4–40


def build_matrices(closes, funding, member, grid, costs):
    rows, trades = [], []
    for p in grid:
        if p["family"] == "xs_carry":
            W = xs_carry_weights(funding, member, p["window"], p["quantile"], p["rebalance"])
        else:
            W = xs_momentum_weights(closes, member, p["lookback"], p["skip"],
                                    p["quantile"], p["rebalance"])
        res = run_xs_backtest(closes, funding, W, costs)
        rows.append(res.returns.values)
        trades.append(res.trade_events.values)
    return np.vstack(rows), np.vstack(trades)


def load_hl_panel(coins, start_ms, end_ms, index):
    closes, funds, qvs = {}, {}, {}
    for c in coins:
        try:
            d = hl.load_hl_daily(c, start_ms, end_ms)
            f = hl.load_hl_funding_daily(c, start_ms, end_ms)
        except Exception as e:  # noqa: BLE001 — пропускаем монету, но вслух
            print(f"[warn] HL {c}: {type(e).__name__}: {e}")
            continue
        if d.empty:
            continue
        closes[c] = d["close"]
        qvs[c] = d["quote_volume"]
        funds[c] = f
    C = pd.DataFrame(closes).reindex(index)
    Q = pd.DataFrame(qvs).reindex(index)
    F = pd.DataFrame(funds).reindex(index).fillna(0.0)
    return C, F, Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01")
    ap.add_argument("--end", default="2026-07")
    ap.add_argument("--cap", type=int, default=None, help="обрезка вселенной (smoke)")
    ap.add_argument("--out", default="reports/hl0006_report")
    args = ap.parse_args()

    t0 = time.time()
    start_ms = int(pd.Timestamp(args.start + "-01", tz="UTC").value // 1_000_000)
    end_ms = int(pd.Timestamp(args.end + "-01", tz="UTC").value // 1_000_000)
    index = pd.date_range(args.start + "-01", args.end + "-01", freq="1D", tz="UTC")[:-1]

    coins = hl.list_hl_perps()
    if args.cap:
        coins = coins[:args.cap]
    print(f"[pool] {len(coins)} перпов Hyperliquid")
    closes, funding, qv = load_hl_panel(coins, start_ms, end_ms, index)
    print(f"[data] {closes.shape[1]} монет с данными, {closes.shape[0]} баров")

    member = monthly_universe(qv, TOP_K, vol_window=30, min_history=90,
                              skip_top=SKIP_TOP, min_median_qv=MIN_MEDIAN_QV)
    sizes = member.sum(axis=1)
    member_cells = int(member.to_numpy().sum())
    fz = float(((funding == 0) & member).to_numpy().sum() / max(member_cells, 1))
    print(f"[universe] mean={sizes.mean():.1f} max={int(sizes.max())} | "
          f"member-баров funding=0: {fz:.1%}")

    costs = Costs(taker_fee=TAKER, slippage=SLIP, apply_funding=True)
    thr = Thresholds()
    cpcv = CPCVConfig(n_groups=10, k_test=2, purge=1, embargo=31)
    results = []
    for name, grid in (("xs_carry", CAR_GRID), ("xs_momentum", MOM_GRID)):
        R, T = build_matrices(closes, funding, member, grid, costs)
        rep = run_cpcv_returns(R, T, closes.index, grid, cpcv, thr)
        rep["family"] = name
        rep["pbo"] = round(pbo_cscv(R, n_blocks=10), 4)
        if name == "xs_carry" and fz > 0.10:
            rep["warning"] = f"funding=0 на {fz:.0%} member-баров — carry с оговоркой"
        results.append(rep)
        print(f"[done] {name}: verdict={rep.get('verdict')} "
              f"OOS_Sharpe(med)={rep.get('oos_sharpe_annualized')} "
              f"worst={rep.get('worst_path_sharpe_annualized')} "
              f"DSR={rep.get('deflated_sharpe_ratio')} PBO={rep['pbo']} "
              f"trades={rep.get('approx_oos_trades')}")

    out = {
        "spec": "0006", "venue": "hyperliquid",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "pool_size": len(coins), "universe_size_mean": round(float(sizes.mean()), 2),
        "start": args.start, "end": args.end,
        "member_bars_funding_zero_share": round(fz, 4),
        "band": f"ranks {SKIP_TOP + 1}-{SKIP_TOP + TOP_K}", "min_median_qv": MIN_MEDIAN_QV,
        "slippage": SLIP, "families": results,
    }
    with open(args.out + ".json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    lines = [f"# Hyperliquid XS (спека 0006) — {out['generated_at_utc']}", "",
             f"Вселенная {len(coins)} | band {out['band']} пол ${MIN_MEDIAN_QV:.0f} | "
             f"{args.start}..{args.end} 1d | slip {SLIP} | universe mean {out['universe_size_mean']}",
             "", "| Семейство | Вердикт | OOS Sharpe (med) | worst | DSR | PBO | сделки |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['family']} | {r.get('verdict')} | "
                     f"{r.get('oos_sharpe_annualized')} | {r.get('worst_path_sharpe_annualized')} | "
                     f"{r.get('deflated_sharpe_ratio')} | {r['pbo']} | {r.get('approx_oos_trades')} |")
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"=== HL DONE in {out['elapsed_sec']}s. Report: {args.out}.md ===")


if __name__ == "__main__":
    main()
