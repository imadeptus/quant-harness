#!/usr/bin/env python3
"""Прогон XS-спек (0001: топ-ликвидные 4h; 0002: хвост ликвидности 1d).

Гриды и параметры universe заморожены соответствующей спекой в docs/specs;
CLI-переопределения — только для отладки, боевой прогон = чистый пресет.
Судья общий: CPCV (медианный путь) + DSR (точная V) + PBO.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from harness.backtest import Costs
from harness.data import list_all_um_symbols, load_panel
from harness.pbo import pbo_cscv
from harness.runner import Thresholds, run_cpcv_returns
from harness.walk_forward import CPCVConfig
from harness.xs import (monthly_universe, run_xs_backtest, xs_carry_weights,
                        xs_momentum_weights)

MOM_GRID_4H = [{"family": "xs_momentum", "lookback": L, "skip": s,
                "quantile": q, "rebalance": R}
               for L in (42, 84, 180) for s in (0, 6)
               for q in (0.25, 1 / 3) for R in (6, 42)]
CAR_GRID_4H = [{"family": "xs_carry", "window": F, "quantile": q, "rebalance": R}
               for F in (18, 42, 84) for q in (0.25, 1 / 3) for R in (6, 42)]

MOM_GRID_1D = [{"family": "xs_momentum", "lookback": L, "skip": s,
                "quantile": q, "rebalance": R}
               for L in (7, 14, 30) for s in (0, 1)
               for q in (0.20, 0.25) for R in (1, 7)]
CAR_GRID_1D = [{"family": "xs_carry", "window": F, "quantile": q, "rebalance": R}
               for F in (3, 7, 14) for q in (0.20, 0.25) for R in (1, 7)]

PRESETS = {
    "0001": dict(interval="4h", start="2024-07", n_months=24, slippage=0.0002,
                 pool="top", top_k=16, vol_window=180, min_history=540,
                 skip_top=0, min_median_qv=None,
                 grids=(("xs_momentum", MOM_GRID_4H, 186),
                        ("xs_carry", CAR_GRID_4H, 84))),
    "0002": dict(interval="1d", start="2024-01", n_months=30, slippage=0.0005,
                 pool="s3", top_k=60, vol_window=30, min_history=90,
                 skip_top=20, min_median_qv=5_000_000.0,
                 grids=(("xs_momentum", MOM_GRID_1D, 31),
                        ("xs_carry", CAR_GRID_1D, 14))),
}


def month_range(start_ym: str, n: int):
    y, m = map(int, start_ym.split("-"))
    out = []
    for _ in range(n):
        out.append((y, m))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def current_trading_set():
    info = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo",
                        timeout=30).json()
    return {s["symbol"] for s in info["symbols"]
            if s.get("contractType") == "PERPETUAL"
            and s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT"}


def top_pool(size: int):
    """Спека 0001: топ-N по СЕГОДНЯШНЕМУ объёму (bias зафиксирован v1.1)."""
    perps = current_trading_set()
    tick = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr",
                        timeout=30).json()
    ranked = sorted((float(x["quoteVolume"]), x["symbol"])
                    for x in tick if x["symbol"] in perps)
    return [sym for _, sym in ranked[-size:]][::-1]


def synthetic_panel(n_syms=8, n=4000, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    cols = [f"S{i}" for i in range(n_syms)]
    closes = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n, n_syms)), axis=0),
                          index=idx, columns=cols)
    funding = pd.DataFrame(rng.normal(0.0001, 0.0002, (n, n_syms)), index=idx, columns=cols)
    funding.iloc[1::2] = 0.0
    qv = pd.DataFrame(np.tile(1000.0 * (1 + rng.random(n_syms)), (n, 1)),
                      index=idx, columns=cols)
    return closes, funding, qv


def build_matrices(closes, funding, member, grid, costs):
    rows, trades = [], []
    for p in grid:
        if p["family"] == "xs_momentum":
            W = xs_momentum_weights(closes, member, p["lookback"], p["skip"],
                                    p["quantile"], p["rebalance"])
        else:
            W = xs_carry_weights(funding, member, p["window"],
                                 p["quantile"], p["rebalance"])
        res = run_xs_backtest(closes, funding, W, costs)
        rows.append(res.returns.values)
        trades.append(res.trade_events.values)
    return np.vstack(rows), np.vstack(trades)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", choices=sorted(PRESETS), default="0001")
    # None = взять из пресета спеки; явное значение — отладочное переопределение
    ap.add_argument("--interval", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--n-months", type=int, default=None)
    ap.add_argument("--slippage", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--taker-fee", type=float, default=0.0005)
    ap.add_argument("--pool-size", type=int, default=40, help="для pool=top (спека 0001)")
    ap.add_argument("--pool-cap", type=int, default=None, help="обрезка пула для отладки")
    ap.add_argument("--cpcv-groups", type=int, default=10)
    ap.add_argument("--cpcv-k", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    ps = PRESETS[args.spec]
    interval = args.interval or ps["interval"]
    start = args.start or ps["start"]
    n_months = args.n_months or ps["n_months"]
    slippage = args.slippage if args.slippage is not None else ps["slippage"]
    top_k = args.top_k or ps["top_k"]
    out_path = args.out or f"reports/xs{args.spec}_report"

    t0 = time.time()
    delisted_share = None
    if args.synthetic:
        closes, funding, qv = synthetic_panel()
        pool = list(closes.columns)
    else:
        if ps["pool"] == "s3":
            pool = list_all_um_symbols()
            trading = current_trading_set()
            delisted = [s for s in pool if s not in trading]
            delisted_share = round(len(delisted) / max(len(pool), 1), 4)
            print(f"[pool] S3: {len(pool)} символов, из них делистнутых "
                  f"{len(delisted)} ({delisted_share:.0%}) — survivorship-фикс работает")
        else:
            pool = top_pool(args.pool_size)
            print(f"[pool] top-{len(pool)} по текущему объёму")
        if args.pool_cap:
            pool = pool[:args.pool_cap]
            print(f"[pool] отладочная обрезка до {len(pool)}")
        closes, funding, qv = load_panel(pool, interval, month_range(start, n_months))

    funding_payments = int((funding != 0).to_numpy().sum())
    member = monthly_universe(qv, top_k, ps["vol_window"], ps["min_history"],
                              skip_top=ps["skip_top"],
                              min_median_qv=ps["min_median_qv"])
    sizes = member.sum(axis=1)
    member_cells = int(member.to_numpy().sum())
    fz_share = round(float(((funding == 0) & member).to_numpy().sum()
                           / max(member_cells, 1)), 4)
    print(f"[universe] размер: mean={sizes.mean():.1f} min={int(sizes.min())} "
          f"max={int(sizes.max())} | funding payments: {funding_payments} "
          f"| member-баров с funding=0: {fz_share:.1%}")

    costs = Costs(taker_fee=args.taker_fee, slippage=slippage, apply_funding=True)
    thr = Thresholds()
    results = []
    for fam, grid, embargo in ps["grids"]:
        R, T = build_matrices(closes, funding, member, grid, costs)
        cpcv = CPCVConfig(n_groups=args.cpcv_groups, k_test=args.cpcv_k,
                          purge=1, embargo=embargo)
        rep = run_cpcv_returns(R, T, closes.index, grid, cpcv, thr)
        rep["family"] = fam
        rep["pbo"] = round(pbo_cscv(R, n_blocks=10), 4)
        if fam == "xs_carry" and (funding_payments == 0 or fz_share > 0.10):
            rep["warning"] = (f"funding отсутствует на {fz_share:.0%} member-баров "
                              f"— carry-вердикту доверять с оговоркой")
        results.append(rep)
        print(f"[done] {fam}: verdict={rep.get('verdict')} "
              f"OOS_Sharpe(med)={rep.get('oos_sharpe_annualized')} "
              f"worst={rep.get('worst_path_sharpe_annualized')} "
              f"DSR={rep.get('deflated_sharpe_ratio')} PBO={rep['pbo']} "
              f"trades={rep.get('approx_oos_trades')}")

    out = {
        "spec": args.spec,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "interval": interval, "start": start, "n_months": n_months,
        "pool_size": len(pool), "delisted_share": delisted_share,
        "top_k": top_k, "skip_top": ps["skip_top"],
        "min_median_qv": ps["min_median_qv"],
        "slippage": slippage,
        "universe_size_mean": round(float(sizes.mean()), 2),
        "member_bars_funding_zero_share": fz_share,
        "families": results,
    }
    with open(out_path + ".json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    lines = [f"# XS sweep spec {args.spec} — {out['generated_at_utc']}", "",
             f"Пул {len(pool)} (делистнутых: {delisted_share}) | band skip_top="
             f"{ps['skip_top']} top_k={top_k} | {start} +{n_months}м | {interval} "
             f"| slippage {slippage}", "",
             "| Семейство | Вердикт | OOS Sharpe (med) | worst path | DSR | PBO | сделки |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['family']} | {r.get('verdict', 'error')} | "
                     f"{r.get('oos_sharpe_annualized')} | "
                     f"{r.get('worst_path_sharpe_annualized')} | "
                     f"{r.get('deflated_sharpe_ratio')} | {r['pbo']} | "
                     f"{r.get('approx_oos_trades')} |")
    with open(out_path + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"=== XS spec {args.spec} DONE in {out['elapsed_sec']}s. "
          f"Report: {out_path}.md ===")


if __name__ == "__main__":
    main()
