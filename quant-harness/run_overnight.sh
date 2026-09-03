#!/usr/bin/env bash
# MAXIMUM overnight sweep — run on YOUR machine before sleep. Zero Claude tokens.
#
#   bash run_overnight.sh
#
# 4 symbols x 4 families x 24 months, full honesty stack (walk-forward + Deflated
# Sharpe + PBO + costs + real funding). Writes a ranked report to
# reports/overnight_report.md (and .json). Runtime is typically tens of minutes
# (downloads + backtests); scale --symbols / --n-months up for a longer grind.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q numpy scipy pandas requests

# Log to a file too, so a morning glance shows how far it got even if you closed the terminal.
mkdir -p reports
echo "Starting overnight sweep at $(date). Logging to reports/overnight.log"
python run_overnight.py \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT \
  --interval 1h \
  --start 2023-01 --n-months 24 \
  --families all \
  --out reports/overnight_report 2>&1 | tee reports/overnight.log

echo
echo "=== DONE. Open reports/overnight_report.md ==="
echo "If a PASS row exists: verify funding attached (carry), PBO low, then paper-trade. No capital yet."
