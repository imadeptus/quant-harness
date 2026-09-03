"""Short English markdown report for a verdict — for humans and for agents that
paste it into a thread. Deliberately terse and free of promises."""
from __future__ import annotations

from typing import Dict, List, Optional

DISCLAIMER = (
    "Not investment advice. This verdict is a statistical screen of the data you supplied "
    "against pre-registered thresholds; it cannot see look-ahead, survivorship, data errors "
    "or cost mis-specification upstream of this API, and it says nothing about future returns.")

CALIBRATION_NOTE = (
    "Calibration (reports/CALIBRATION.md, synthetic ground truth): 0.0% PASS on pure noise "
    "at N>=6 configs; PASS>=50% is crossed between a true annualized Sharpe of 2.0 (46%) and "
    "2.5 (81%) — about 2.2 by interpolation; PASS>=90% at about 3.0.")


def _f(x: Optional[float], nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _ok(flag: bool) -> str:
    return "yes" if flag else "no"


def render_report(verdict: str, checks: Dict[str, bool], metrics: Dict[str, Optional[float]],
                  thresholds: Dict[str, float], assumptions: List[str], version: str) -> str:
    n = metrics.get("n_configs_tried")
    rows = [
        ("OOS Sharpe, annualized (median CPCV path)", _f(metrics.get("oos_sharpe_annualized")),
         f"> {thresholds['min_oos_sharpe']}", _ok(checks.get("oos_sharpe_ok", False))),
        ("Max drawdown (median path)", _f(metrics.get("oos_max_drawdown")),
         f"< {thresholds['max_drawdown']}", _ok(checks.get("drawdown_ok", False))),
        (f"Deflated Sharpe Ratio (N={n})", _f(metrics.get("deflated_sharpe_ratio"), 4),
         f">= {thresholds['min_dsr']}", _ok(checks.get("dsr_ok", False))),
        ("Approx. OOS trades", str(metrics.get("approx_oos_trades")),
         f">= {thresholds['min_trades']}", _ok(checks.get("trades_ok", False))),
    ]
    lines = [
        f"# quant-harness verdict: {verdict}",
        "",
        "Mechanical PASS/KILL against pre-registered thresholds: Combinatorial Purged CV, "
        "Deflated Sharpe (multiple-testing corrected), PBO. All four gates must pass.",
        "",
        "| Gate | Value | Threshold | OK |",
        "|---|---|---|---|",
        *[f"| {g} | {v} | {t} | {ok} |" for g, v, t, ok in rows],
        "",
        f"Worst CPCV path Sharpe: {_f(metrics.get('worst_path_sharpe_annualized'))} · "
        f"PSR vs zero: {_f(metrics.get('psr_vs_zero'), 4)} · "
        f"PBO (CSCV): {_f(metrics.get('pbo'))} · "
        f"paths: {metrics.get('n_paths')} · OOS bars: {metrics.get('oos_bars')} · "
        f"annualization: {_f(metrics.get('ann_factor'), 1)} periods/year.",
        "",
        "## Assumptions",
        *[f"- {a}" for a in assumptions],
        "",
        f"_{CALIBRATION_NOTE}_",
        "",
        f"**Disclaimer.** {DISCLAIMER}",
        "",
        f"quant-harness {version}",
    ]
    return "\n".join(lines)
