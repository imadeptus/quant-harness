#!/usr/bin/env python3
"""MAXIMUM overnight sweep — run on YOUR machine, zero Claude tokens.

Loops over symbols x families, each with a large parameter grid, over a multi-year
period, applying the full honesty stack (walk-forward + Deflated Sharpe + PBO +
realistic costs + real funding). Ranks every (symbol, family) cell and writes a
Markdown + JSON report of survivors.

Because it downloads real data and runs locally, it is unaffected by any Claude
token budget. Kick it off before sleep:

    bash run_overnight.sh

Design choices for an UNATTENDED run:
- Any single cell that errors is caught and recorded as an error, never crashes
  the whole sweep.
- Funding is fetched for real; if it fails, carry cells are flagged untrustworthy
  rather than silently zeroed.
- Progress is printed per cell so a morning glance at the log shows how far it got.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime

import numpy as np

from harness.backtest import Costs
from harness.data import load
from harness.families import all_families, grid_for
from harness.pbo import pbo_cscv
from harness.runner import Thresholds, run, run_cpcv
from harness.walk_forward import CPCVConfig, WalkForwardConfig


def month_range(start_ym: str, n: int):
    y, m = map(int, start_ym.split("-"))
    out = []
    for _ in range(n):
        out.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--start", default="2023-01", help="first month YYYY-MM")
    ap.add_argument("--n-months", type=int, default=24)
    ap.add_argument("--families", default="all")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--embargo", type=int, default=24)
    ap.add_argument("--cv", choices=("wfa", "cpcv"), default="wfa",
                    help="wfa = rolling walk-forward; cpcv = combinatorial purged CV")
    ap.add_argument("--cpcv-groups", type=int, default=10)
    ap.add_argument("--cpcv-k", type=int, default=2)
    ap.add_argument("--taker-fee", type=float, default=0.0005)
    ap.add_argument("--slippage", type=float, default=0.0002)
    ap.add_argument("--out", default="reports/overnight_report")
    ap.add_argument("--synthetic", action="store_true", help="offline test mode")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    families = all_families() if args.families == "all" else \
        [f.strip() for f in args.families.split(",")]
    months = month_range(args.start, args.n_months)
    wf = WalkForwardConfig(train_size=args.train, test_size=args.test,
                           embargo=args.embargo, anchored=False)
    costs = Costs(taker_fee=args.taker_fee, slippage=args.slippage, apply_funding=True)
    thr = Thresholds()

    cells = []
    t0 = time.time()
    for sym in symbols:
        try:
            bars = load(symbol=sym, interval=args.interval, months=months,
                        synthetic=args.synthetic, n_synth=args.n_months * 720,
                        with_funding=not args.synthetic)
            df = bars.df
            print(f"[data] {sym}: {len(df)} bars {df.index[0]}..{df.index[-1]}")
        except Exception as e:  # noqa: BLE001
            print(f"[error] load {sym}: {e}")
            cells.append({"symbol": sym, "family": None, "error": str(e)})
            continue

        for fam in families:
            label = f"{sym}/{fam}"
            try:
                grid = grid_for(fam)
                if args.cv == "cpcv":
                    cpcv = CPCVConfig(n_groups=args.cpcv_groups, k_test=args.cpcv_k,
                                      purge=1, embargo=args.embargo)
                    rep = run_cpcv(df, grid, cpcv, costs, thr)
                else:
                    rep = run(df, grid, wf, costs, thr)
                if "error" in rep:
                    cells.append({"symbol": sym, "family": fam, "error": rep["error"]})
                    print(f"[skip] {label}: {rep['error']}")
                    continue
                # PBO across this family's config matrix on the full series.
                from harness.families import build_signal
                from harness.backtest import run_backtest
                R = np.vstack([run_backtest(df, build_signal(df, p), costs).returns.values
                               for p in grid])
                rep["pbo"] = round(pbo_cscv(R, n_blocks=10), 4)
                rep["symbol"] = sym
                rep["family"] = fam
                cells.append(rep)
                print(f"[done] {label}: verdict={rep['verdict']} "
                      f"OOS_Sharpe={rep['oos_sharpe_annualized']} "
                      f"DSR={rep['deflated_sharpe_ratio']} PBO={rep['pbo']}")
            except Exception as e:  # noqa: BLE001
                cells.append({"symbol": sym, "family": fam, "error": repr(e)})
                print(f"[error] {label}: {e}")
                traceback.print_exc()

    # Rank: PASS first, then by DSR, then OOS Sharpe.
    scored = [c for c in cells if "deflated_sharpe_ratio" in c]
    scored.sort(key=lambda c: (c["verdict"] == "PASS", c["deflated_sharpe_ratio"],
                               c["oos_sharpe_annualized"]), reverse=True)

    elapsed = time.time() - t0
    result = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "elapsed_sec": round(elapsed, 1),
        "symbols": symbols, "families": families,
        "period_months": args.n_months, "start": args.start,
        "n_cells": len(cells),
        "n_pass": sum(1 for c in scored if c["verdict"] == "PASS"),
        "ranked": scored,
        "errors": [c for c in cells if "error" in c],
    }
    with open(args.out + ".json", "w") as f:
        json.dump(result, f, indent=2)
    _write_markdown(result, args.out + ".md")
    print(f"\n=== SWEEP DONE in {elapsed:.0f}s. "
          f"{result['n_pass']} PASS / {len(scored)} cells. "
          f"Report: {args.out}.md ===")


def _write_markdown(result, path):
    lines = [f"# Overnight sweep — {result['generated_at_utc']}", ""]
    lines.append(f"Symbols: {', '.join(result['symbols'])} | Families: "
                 f"{', '.join(result['families'])} | Period: {result['period_months']} "
                 f"months from {result['start']} | Runtime: {result['elapsed_sec']}s")
    lines.append("")
    lines.append(f"**{result['n_pass']} PASS out of {len(result['ranked'])} cells.** "
                 "PASS = survived walk-forward + Deflated Sharpe(>=0.95) + trades/DD thresholds. "
                 "Watch PBO: high (>0.5) means the selection process overfits regardless of DSR.")
    lines.append("")
    lines.append("| Rank | Symbol | Family | Verdict | OOS Sharpe (ann) | Max DD | DSR | PBO | Trades | N cfg |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for i, c in enumerate(result["ranked"], 1):
        lines.append(f"| {i} | {c['symbol']} | {c['family']} | **{c['verdict']}** | "
                     f"{c['oos_sharpe_annualized']} | {c['oos_max_drawdown']} | "
                     f"{c['deflated_sharpe_ratio']} | {c.get('pbo','-')} | "
                     f"{c['approx_oos_trades']} | {c['n_configs_tried']} |")
    lines.append("")
    if result["errors"]:
        lines.append("## Errors / skipped")
        for e in result["errors"]:
            lines.append(f"- {e.get('symbol')}/{e.get('family')}: {e.get('error')}")
    lines.append("")
    lines.append("## How to read this")
    lines.append("- **No PASS rows → KILL-1**: honest search found no edge surviving costs+DSR. "
                 "That is a real, money-saving result, not a failure of the run.")
    lines.append("- **A PASS row** is a *candidate*, not a green light: confirm funding was "
                 "attached (carry), check PBO is low, then paper-trade before any capital.")
    lines.append("- Carry rows are untrustworthy if the data log warned funding=0.")
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
