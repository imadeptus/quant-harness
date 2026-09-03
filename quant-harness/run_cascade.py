#!/usr/bin/env python3
"""Прогон спеки 0007 — каскад-реверсал прокси, 1d, из КЭША (автономно).

--venue binance | hyperliquid. Обе панели из кэша (0 загрузок). Громкие гарды
валидности: любая вырожденность (мало shock-событий, нет данных) → явный warning
в JSON, не тихий KILL (важно для ночного автономного прогона).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from harness.backtest import Costs
from harness.cascade import cascade_reversal_weights
from harness.pbo import pbo_cscv
from harness.runner import Thresholds, run_cpcv_returns
from harness.walk_forward import CPCVConfig
from harness.xs import monthly_universe, run_xs_backtest

GRID = [{"k": k, "m": m, "hold": h} for k in (2.0, 3.0) for m in (2.0, 3.0)
        for h in (1, 3, 7)]  # 12
TAKER, SLIP = 0.0005, 0.0010


def load_binance_panel(months):
    from harness.data import list_all_um_symbols, load_panel
    pool = list_all_um_symbols()
    closes, funding, qv = load_panel(pool, "1d", months)
    return closes, funding, qv, 2_000_000.0


def load_hl_panel(start, end):
    from harness import hyperliquid as hl
    start_ms = int(pd.Timestamp(start + "-01", tz="UTC").value // 1_000_000)
    end_ms = int(pd.Timestamp(end + "-01", tz="UTC").value // 1_000_000)
    index = pd.date_range(start + "-01", end + "-01", freq="1D", tz="UTC")[:-1]
    coins = hl.list_hl_perps()
    closes, funds, qvs = {}, {}, {}
    for c in coins:
        try:
            d = hl.load_hl_daily(c, start_ms, end_ms)
            f = hl.load_hl_funding_daily(c, start_ms, end_ms)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] HL {c}: {type(e).__name__}")
            continue
        if d.empty:
            continue
        closes[c] = d["close"]; qvs[c] = d["quote_volume"]; funds[c] = f
    C = pd.DataFrame(closes).reindex(index)
    F = pd.DataFrame(funds).reindex(index).fillna(0.0)
    Q = pd.DataFrame(qvs).reindex(index)
    return C, F, Q, 1_000_000.0


def month_range(start_ym, n):
    y, m = map(int, start_ym.split("-")); out = []
    for _ in range(n):
        out.append((y, m)); m += 1
        if m > 12: y, m = y + 1, 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", choices=("binance", "hyperliquid"), required=True)
    ap.add_argument("--direction", type=int, choices=(1, -1), default=1,
                    help="+1 fade (0007), -1 follow/продолжение (0009)")
    ap.add_argument("--gross", action="store_true",
                    help="доп. gross-прогон (costs=0) — есть ли сигнал до издержек")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    spec = "0009" if args.direction == -1 else "0007"
    out_path = args.out or f"reports/cascade{spec}_{args.venue}"

    t0 = time.time()
    if args.venue == "binance":
        closes, funding, qv, floor = load_binance_panel(month_range("2024-07", 24))
    else:
        closes, funding, qv, floor = load_hl_panel("2024-01", "2026-07")
    print(f"[data] {args.venue}: {closes.shape[1]} монет, {closes.shape[0]} баров")

    member = monthly_universe(qv, top_k=60, vol_window=30, min_history=90,
                              skip_top=0, min_median_qv=floor)
    sizes = member.sum(axis=1)
    costs = Costs(taker_fee=TAKER, slippage=SLIP, apply_funding=True)
    thr = Thresholds()
    cpcv = CPCVConfig(n_groups=10, k_test=2, purge=1, embargo=30)

    zero_costs = Costs(taker_fee=0.0, slippage=0.0, apply_funding=True)
    R, T, Rg, warns = [], [], [], []
    for p in GRID:
        W = cascade_reversal_weights(closes, qv, member, p["k"], p["m"], p["hold"],
                                     direction=args.direction)
        res = run_xs_backtest(closes, funding, W, costs)
        R.append(res.returns.values); T.append(res.trade_events.values)
        if args.gross:
            Rg.append(run_xs_backtest(closes, funding, W, zero_costs).returns.values)
    R, T = np.vstack(R), np.vstack(T)

    # ГРОМКИЙ гард: сколько shock-событий сгенерировал сигнал (в среднем по гриду)
    total_events = int(T.sum(axis=1).mean())
    if total_events < 200:
        warns.append(f"МАЛО shock-событий (~{total_events} avg) — сигнал вырожден, "
                     f"вердикт НЕДОСТОВЕРЕН")
    if sizes.mean() < 5:
        warns.append(f"universe вырожден (mean {sizes.mean():.1f})")

    rep = run_cpcv_returns(R, T, closes.index, GRID, cpcv, thr)
    rep["pbo"] = round(pbo_cscv(R, n_blocks=10), 4)
    rep["avg_shock_events"] = total_events
    rep["universe_size_mean"] = round(float(sizes.mean()), 2)
    rep["direction"] = args.direction
    rep["warnings"] = warns
    if args.gross:
        gross = run_cpcv_returns(np.vstack(Rg), T, closes.index, GRID, cpcv, thr)
        rep["gross_oos_sharpe"] = gross.get("oos_sharpe_annualized")
    print(f"[done] {args.venue}: verdict={rep.get('verdict')} "
          f"OOS_Sharpe(med)={rep.get('oos_sharpe_annualized')} "
          f"worst={rep.get('worst_path_sharpe_annualized')} "
          f"DSR={rep.get('deflated_sharpe_ratio')} PBO={rep['pbo']} "
          f"events~{total_events} | warnings={warns}")

    out = {
        "spec": "0007", "venue": args.venue,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "n_coins": int(closes.shape[1]), "result": rep,
    }
    with open(out_path + ".json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    with open(out_path + ".md", "w") as f:
        f.write(f"# Cascade-proxy 0007 — {args.venue} — {out['generated_at_utc']}\n\n"
                f"монет {closes.shape[1]} | universe mean {rep['universe_size_mean']} "
                f"| shock-событий ~{total_events} | warnings: {warns or 'нет'}\n\n"
                f"| Вердикт | OOS Sharpe (med) | worst | DSR | PBO |\n|---|---|---|---|---|\n"
                f"| {rep.get('verdict')} | {rep.get('oos_sharpe_annualized')} | "
                f"{rep.get('worst_path_sharpe_annualized')} | "
                f"{rep.get('deflated_sharpe_ratio')} | {rep['pbo']} |\n")
    print(f"=== CASCADE {args.venue} DONE in {out['elapsed_sec']}s. {out_path}.md ===")


if __name__ == "__main__":
    main()
