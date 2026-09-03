#!/usr/bin/env python3
"""Прогон спеки 0003 — межбиржевой funding-базис Binance↔Bybit, 1d.

Кандидаты: текущий топ по объёму Binance ∩ листинг Bybit (bias текущего листинга
зафиксирован в спеке §2 — ликвидный сегмент). Universe: point-in-time топ-20 по
30д объёму Binance. Грид §4 заморожен (18). Судья общий: CPCV+DSR+PBO.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from harness import bybit
from harness.backtest import Costs
from harness.basis import basis_weights, run_basis_backtest
from harness.data import load_panel
from harness.pbo import pbo_cscv
from harness.runner import Thresholds, run_cpcv_returns
from harness.walk_forward import CPCVConfig
from harness.xs import monthly_universe
from run_xs import month_range, top_pool

GRID = [{"family": "funding_basis", "window": W, "theta": th, "rebalance": R}
        for W in (3, 7, 14) for th in (0.0, 0.00005, 0.0001) for R in (1, 7)]  # 18


def bybit_daily(symbols, months, index):
    """(closes, funding) Bybit на дневной сетке index; funding = сумма платежей дня."""
    closes, funds = {}, {}
    for sym in symbols:
        c_parts, f_parts = [], []
        for (y, m) in months:
            try:
                c_parts.append(bybit.load_daily_closes_month(sym, y, m))
                f_parts.append(bybit.load_funding_month(sym, y, m))
            except (bybit.BybitError, Exception) as e:  # noqa: BLE001 — месяц пропускаем, но вслух
                print(f"[warn] bybit {sym} {y:04d}-{m:02d}: {type(e).__name__}: {e}")
        closes[sym] = pd.concat(c_parts).sort_index() if c_parts else pd.Series(dtype=float)
        if f_parts:
            f = pd.concat(f_parts).sort_index()
            daily = f.groupby(f.index.floor("D")).sum()
        else:
            daily = pd.Series(dtype=float)
        funds[sym] = daily
        n_pay = sum(len(p) for p in f_parts)
        print(f"[bybit] {sym}: {sum(len(p) for p in c_parts)} closes, {n_pay} funding payments")
    C = pd.DataFrame(closes).reindex(index)
    F = pd.DataFrame(funds).reindex(index).fillna(0.0)
    return C, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01")
    ap.add_argument("--n-months", type=int, default=30)
    ap.add_argument("--pool-size", type=int, default=60)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--taker-fee", type=float, default=0.0005)
    ap.add_argument("--slippage", type=float, default=0.0002)
    ap.add_argument("--cap", type=int, default=None, help="обрезка кандидатов (smoke)")
    ap.add_argument("--out", default="reports/basis0003_report")
    args = ap.parse_args()

    t0 = time.time()
    months = month_range(args.start, args.n_months)
    byb_listed = bybit.linear_symbols()
    candidates = [s for s in top_pool(args.pool_size) if s in byb_listed]
    if args.cap:
        candidates = candidates[:args.cap]
    print(f"[pool] binance top-{args.pool_size} ∩ bybit = {len(candidates)} кандидатов")

    closes_bin, fund_bin, qv = load_panel(candidates, "1d", months)
    closes_byb, fund_byb = bybit_daily(list(closes_bin.columns), months, closes_bin.index)

    member = monthly_universe(qv, args.top_k, vol_window=30, min_history=90)
    sizes = member.sum(axis=1)
    member_cells = int(member.to_numpy().sum())
    cover_bin = float(((fund_bin != 0) & member).to_numpy().sum() / max(member_cells, 1))
    cover_byb = float(((fund_byb != 0) & member).to_numpy().sum() / max(member_cells, 1))
    print(f"[universe] mean={sizes.mean():.1f} | funding coverage: "
          f"binance {cover_bin:.1%}, bybit {cover_byb:.1%} (порог 90%)")
    valid_coverage = cover_bin >= 0.9 and cover_byb >= 0.9

    ret_bin = closes_bin.pct_change()
    ret_byb = closes_byb.pct_change()
    spread = fund_bin - fund_byb

    costs = Costs(taker_fee=args.taker_fee, slippage=args.slippage, apply_funding=True)
    rows, trades = [], []
    for p in GRID:
        W = basis_weights(spread, member, p["window"], p["theta"], p["rebalance"])
        res = run_basis_backtest(ret_bin, ret_byb, fund_bin, fund_byb, W, costs)
        rows.append(res.returns.values)
        trades.append(res.trade_events.values)
    R, T = np.vstack(rows), np.vstack(trades)

    rep = run_cpcv_returns(R, T, closes_bin.index, GRID,
                           CPCVConfig(n_groups=10, k_test=2, purge=1, embargo=14),
                           Thresholds())
    rep["family"] = "funding_basis"
    rep["pbo"] = round(pbo_cscv(R, n_blocks=10), 4)
    if not valid_coverage:
        rep["warning"] = (f"funding coverage ниже 90% (bin {cover_bin:.0%} / "
                          f"byb {cover_byb:.0%}) — ПРОГОН НЕВАЛИДЕН по спеке §5")
    print(f"[done] funding_basis: verdict={rep.get('verdict')} "
          f"OOS_Sharpe(med)={rep.get('oos_sharpe_annualized')} "
          f"worst={rep.get('worst_path_sharpe_annualized')} "
          f"DSR={rep.get('deflated_sharpe_ratio')} PBO={rep['pbo']} "
          f"trades={rep.get('approx_oos_trades')}")

    out = {
        "spec": "0003", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "candidates": candidates, "top_k": args.top_k,
        "start": args.start, "n_months": args.n_months,
        "universe_size_mean": round(float(sizes.mean()), 2),
        "funding_coverage": {"binance": round(cover_bin, 4), "bybit": round(cover_byb, 4)},
        "coverage_valid": valid_coverage,
        "known_biases": ["кандидаты = текущие листинги обеих бирж (ликвидный сегмент)"],
        "result": rep,
    }
    with open(args.out + ".json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    with open(args.out + ".md", "w") as f:
        f.write(f"# Funding-basis (спека 0003) — {out['generated_at_utc']}\n\n"
                f"Кандидатов {len(candidates)} | топ-{args.top_k} | coverage "
                f"bin {cover_bin:.1%} / byb {cover_byb:.1%} | valid={valid_coverage}\n\n"
                f"| Вердикт | OOS Sharpe (med) | worst | DSR | PBO | сделки |\n"
                f"|---|---|---|---|---|---|\n"
                f"| {rep.get('verdict')} | {rep.get('oos_sharpe_annualized')} | "
                f"{rep.get('worst_path_sharpe_annualized')} | "
                f"{rep.get('deflated_sharpe_ratio')} | {rep['pbo']} | "
                f"{rep.get('approx_oos_trades')} |\n")
    print(f"=== BASIS DONE in {out['elapsed_sec']}s. Report: {args.out}.md ===")


if __name__ == "__main__":
    main()
