#!/usr/bin/env python3
"""Entry point for the AI Quant Research Lab harness (Spike 2).

Runs the honest walk-forward + Deflated Sharpe pipeline on the momentum family
and prints a numeric PASS/KILL report. Designed so a losing result CANNOT be
rationalized into a win — the verdict is mechanical.

Examples
--------
# Offline sanity run on zero-edge synthetic data (SHOULD verdict KILL):
python run.py --synthetic

# Detection-power check: faint injected edge (pipeline SHOULD notice it):
python run.py --synthetic --synthetic-edge 0.15

# Real data (downloads free Binance public klines; wire funding before trusting):
python run.py --symbol BTCUSDT --interval 1h --months 2024-01,2024-02,2024-03,2024-04,2024-05,2024-06
"""
from __future__ import annotations

import argparse
import json

from harness.backtest import Costs
from harness.data import load
from harness.families import momentum_grid
from harness.runner import Thresholds, run, run_cpcv
from harness.walk_forward import CPCVConfig, WalkForwardConfig


def parse_months(s: str):
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        y, m = tok.split("-")
        out.append((int(y), int(m)))
    return out


def main():
    ap = argparse.ArgumentParser(description="Honest walk-forward + Deflated Sharpe harness")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--months", default=None, help="comma list YYYY-MM,YYYY-MM,...")
    ap.add_argument("--synthetic", action="store_true", help="use offline synthetic data")
    ap.add_argument("--synthetic-edge", type=float, default=0.0)
    ap.add_argument("--n-synth", type=int, default=4000)
    ap.add_argument("--train", type=int, default=1000, help="train bars per fold")
    ap.add_argument("--test", type=int, default=300, help="OOS test bars per fold")
    ap.add_argument("--embargo", type=int, default=24)
    ap.add_argument("--cv", choices=("wfa", "cpcv"), default="wfa",
                    help="wfa = rolling walk-forward; cpcv = combinatorial purged CV")
    ap.add_argument("--cpcv-groups", type=int, default=10)
    ap.add_argument("--cpcv-k", type=int, default=2)
    ap.add_argument("--taker-fee", type=float, default=0.0005)
    ap.add_argument("--slippage", type=float, default=0.0002)
    ap.add_argument("--no-funding", action="store_true")
    ap.add_argument("--out", default=None, help="write JSON report to this path")
    args = ap.parse_args()

    bars = load(
        symbol=args.symbol, interval=args.interval,
        months=parse_months(args.months) if args.months else None,
        synthetic=args.synthetic, synthetic_edge=args.synthetic_edge,
        n_synth=args.n_synth,
        with_funding=not (args.synthetic or args.no_funding),
    )
    grid = list(momentum_grid())
    wf = WalkForwardConfig(train_size=args.train, test_size=args.test,
                           embargo=args.embargo, anchored=False)
    costs = Costs(taker_fee=args.taker_fee, slippage=args.slippage,
                  apply_funding=not args.no_funding)
    thr = Thresholds()

    if args.cv == "cpcv":
        cpcv = CPCVConfig(n_groups=args.cpcv_groups, k_test=args.cpcv_k,
                          purge=1, embargo=args.embargo)
        report = run_cpcv(bars.df, grid, cpcv, costs, thr)
    else:
        report = run(bars.df, grid, wf, costs, thr)
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    # Non-zero exit on KILL so this is usable in CI / scripted screening.
    raise SystemExit(0 if report.get("verdict") == "PASS" else 2)


if __name__ == "__main__":
    main()
