#!/usr/bin/env python3
"""Прогон спеки 0004 — пост-листинговый дрейф/разворот перпов, 1d.

Панель 790 перпов из кэша (0002); возраст = дни с первого бара; две семьи
(drift/reversal) × грид §6. Судья общий: CPCV+DSR+PBO. Per-leg издержки.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np

from harness.backtest import Costs
from harness.data import list_all_um_symbols, load_panel
from harness.listing import (run_listing_backtest, young_basket_weights,
                             young_vs_btc_weights, age_in_days)
from harness.pbo import pbo_cscv
from harness.runner import Thresholds, run_cpcv_returns
from harness.walk_forward import CPCVConfig
from run_xs import month_range

GRID = [{"e_start": E, "h": H} for E in (1, 3) for H in (7, 14, 30)]  # 6 на направление
TAKER, SLIP_YOUNG, SLIP_MATURE = 0.0005, 0.010, 0.002
MIN_MEDIAN_QV = 2_000_000.0


def build_matrices(closes, funding, qv, direction, clean=False, rebalance=7,
                   zero_cost=False):
    """Матрицы (configs × T) net-доходностей и сделок. clean=True — хедж одним
    BTC (спека 0005); zero_cost=True — gross-диагностика (издержки обнулены)."""
    age = age_in_days(closes)
    ty = 0.0 if zero_cost else TAKER
    sy = 0.0 if zero_cost else SLIP_YOUNG
    sm = 0.0 if zero_cost else SLIP_MATURE
    rows, trades = [], []
    for p in GRID:
        if clean:
            W = young_vs_btc_weights(closes, qv, p["e_start"], p["h"], direction,
                                     hedge="BTCUSDT", rebalance=rebalance,
                                     min_median_qv=MIN_MEDIAN_QV)
        else:
            W = young_basket_weights(closes, qv, p["e_start"], p["h"], direction,
                                     mature_age=180, min_median_qv=MIN_MEDIAN_QV)
        young = (age >= p["e_start"]) & (age <= p["h"])
        res = run_listing_backtest(closes, funding, W, young, ty, sy, sm)
        rows.append(res.returns.values)
        trades.append(res.trade_events.values)
    return np.vstack(rows), np.vstack(trades)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01")
    ap.add_argument("--n-months", type=int, default=30)
    ap.add_argument("--cap", type=int, default=None, help="обрезка пула (smoke)")
    ap.add_argument("--clean", action="store_true",
                    help="спека 0005: хедж одним BTC, недельный ребаланс, gross-диагностика")
    ap.add_argument("--rebalance", type=int, default=7)
    ap.add_argument("--out", default="reports/listing0004_report")
    args = ap.parse_args()

    t0 = time.time()
    pool = list_all_um_symbols()
    if args.cap:
        pool = pool[:args.cap]
    if args.clean and "BTCUSDT" not in pool:
        pool = ["BTCUSDT"] + pool  # хедж-инструмент обязан присутствовать
    print(f"[pool] {len(pool)} перпов из S3-листинга (кэш)")
    closes, funding, qv = load_panel(pool, "1d", month_range(args.start, args.n_months))

    age = age_in_days(closes)
    # событий листинга в окне = число символов с первым баром после начала окна
    first_bars = closes.apply(lambda s: s.first_valid_index())
    events = int((first_bars > closes.index[5]).sum())
    fund_cover = float((funding != 0).to_numpy().sum()
                       / max((closes.notna()).to_numpy().sum(), 1))
    print(f"[data] {closes.shape[1]} символов, {closes.shape[0]} баров | "
          f"листингов-событий в окне: {events} | funding-покрытие: {fund_cover:.1%}")

    thr = Thresholds()
    cpcv = CPCVConfig(n_groups=10, k_test=2, purge=1, embargo=30)
    results = []
    for name, direction in (("listing_drift", +1), ("listing_reversal", -1)):
        R, T = build_matrices(closes, funding, qv, direction,
                              clean=args.clean, rebalance=args.rebalance)
        rep = run_cpcv_returns(R, T, closes.index, GRID, cpcv, thr)
        rep["family"] = name
        rep["pbo"] = round(pbo_cscv(R, n_blocks=10), 4)
        if args.clean:
            # gross-диагностика: те же веса, издержки обнулены
            Rg, Tg = build_matrices(closes, funding, qv, direction,
                                    clean=True, rebalance=args.rebalance, zero_cost=True)
            gross = run_cpcv_returns(Rg, Tg, closes.index, GRID, cpcv, thr)
            rep["gross_oos_sharpe_annualized"] = gross.get("oos_sharpe_annualized")
        results.append(rep)
        print(f"[done] {name}: verdict={rep.get('verdict')} "
              f"OOS_Sharpe(med)={rep.get('oos_sharpe_annualized')} "
              f"gross={rep.get('gross_oos_sharpe_annualized')} "
              f"worst={rep.get('worst_path_sharpe_annualized')} "
              f"DSR={rep.get('deflated_sharpe_ratio')} PBO={rep['pbo']} "
              f"trades={rep.get('approx_oos_trades')}")

    out = {
        "spec": "0004", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "pool_size": len(pool), "listing_events": events,
        "funding_coverage": round(fund_cover, 4),
        "start": args.start, "n_months": args.n_months,
        "costs": {"taker": TAKER, "slip_young": SLIP_YOUNG, "slip_mature": SLIP_MATURE},
        "families": results,
    }
    with open(args.out + ".json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    lines = [f"# Post-listing (спека 0004) — {out['generated_at_utc']}", "",
             f"Пул {len(pool)} | листингов-событий {events} | funding {fund_cover:.1%} "
             f"| slip young/mature {SLIP_YOUNG}/{SLIP_MATURE}", "",
             "| Семейство | Вердикт | OOS Sharpe (net) | gross | worst | DSR | PBO | сделки |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['family']} | {r.get('verdict')} | "
                     f"{r.get('oos_sharpe_annualized')} | "
                     f"{r.get('gross_oos_sharpe_annualized', '—')} | "
                     f"{r.get('worst_path_sharpe_annualized')} | "
                     f"{r.get('deflated_sharpe_ratio')} | {r['pbo']} | "
                     f"{r.get('approx_oos_trades')} |")
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"=== LISTING DONE in {out['elapsed_sec']}s. Report: {args.out}.md ===")


if __name__ == "__main__":
    main()
