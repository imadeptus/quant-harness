#!/usr/bin/env bash
# One-command real-data run. Execute on YOUR machine (Binance is reachable there;
# the cloud sandbox blocks data.binance.vision, so this must run locally).
#
#   bash run_real.sh
#
# It: creates a venv, installs deps, downloads 6 months of free Binance BTCUSDT
# 1h perp klines via the harness loader, runs the honest walk-forward + Deflated
# Sharpe pipeline, and writes reports/btc_real.json with a PASS/KILL verdict.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q numpy scipy pandas requests

echo "Running honest walk-forward + Deflated Sharpe on real Binance BTCUSDT 1h..."
python run.py \
  --symbol BTCUSDT --interval 1h \
  --months 2024-07,2024-08,2024-09,2024-10,2024-11,2024-12 \
  --out reports/btc_real.json || true

echo
echo "=== Verdict written to reports/btc_real.json ==="
echo "Reminder: this run uses funding=0 for downloaded klines (explicit placeholder)."
echo "Add real funding history before trusting a perp result — see README TODO #1."
