"""Strategy audit — a mechanical PASS/KILL verdict on someone else's returns.

The product behind `qh-audit`: a client sends the per-period returns of the
strategy configurations they tried (rows = periods, columns = configs), and gets
back the same judge this project uses on its own research — Combinatorial Purged
CV paths, Deflated Sharpe corrected for the number of trials, PBO, a drawdown and
a liquidity gate — plus a report that states every assumption it had to make.

What the audit adds on top of `harness.runner.run_cpcv_returns`:

- input validation with readable errors (NaN/inf, zero variance, too few
  periods, shape mismatches, size limits);
- `n_trials_effective = max(n_trials, n_configs)` so a client who only sends the
  finalists is still deflated by the full search they ran;
- the single-series case: with one column there is no trial dispersion to
  estimate, so the DSR falls back to the 1/n proxy documented in
  `harness.deflated_sharpe` and is deflated by `n_trials`;
- an explicit turnover ASSUMPTION when no trade counts are supplied;
- a cost-sensitivity table (0x / 0.5x / 1x / 2x of the stated bps per unit of
  turnover) when both turnover and a cost level are supplied;
- PBO (CSCV) as an informational statistic — it is not one of the four gates,
  so the verdict stays identical to the calibrated judge in
  `reports/CALIBRATION.md`.

Nothing here predicts anything: the output is a statement about the evidence in
the supplied returns under the stated assumptions, not investment advice.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .backtest import annualization_factor
from .deflated_sharpe import deflated_sharpe_ratio
from .pbo import pbo_cscv
from .runner import Thresholds, run_cpcv_returns
from .walk_forward import CPCVConfig, cpcv_splits

MIN_PERIODS = 100
MAX_CONFIGS = 500
MAX_PERIODS = 200_000
MAX_TRIALS = 1_000_000
MIN_TRAIN_PERIODS = 3       # smallest CPCV train set the judge will accept
DEFAULT_ASSUMED_TRADES_PER_BAR = 1.0
DEFAULT_START = "2020-01-01"
COST_MULTIPLIERS: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
PBO_BLOCKS = 10
# Same CPCV geometry the judge was calibrated with (reports/CALIBRATION.md).
DEFAULT_CPCV = CPCVConfig(n_groups=10, k_test=2, purge=1, embargo=5)

DISCLAIMER = (
    "This is a statistical report produced mechanically by quant-harness (qh-audit). "
    "It is not investment advice, it does not predict future returns, and a PASS is "
    "not a recommendation to trade. Every number is computed from the supplied "
    "returns under the assumptions listed in the report; garbage in, garbage out."
)

_TIMESTAMP_NAMES = {"timestamp", "time", "date", "datetime", "ts", "open_time",
                    "close_time", "bar_time", "period", "index", "dt"}
# --freq units. Case matters: 'm' is minutes, while 'M' would be a calendar month
# in pandas — months, quarters and years have no fixed bar length and are rejected.
_FREQ_UNITS = {"s": "s", "S": "s", "m": "min", "min": "min", "t": "min", "T": "min",
               "h": "h", "H": "h", "d": "D", "D": "D", "w": "W", "W": "W"}


class AuditInputError(ValueError):
    """The supplied returns/trades/index cannot be audited; message is user-facing."""


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #

def _as_matrix(x: Any, what: str) -> np.ndarray:
    try:
        a = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise AuditInputError(f"{what}: not a numeric array ({exc})") from exc
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2:
        raise AuditInputError(f"{what}: expected a 1-D series or a 2-D (configs x periods) "
                              f"matrix, got {a.ndim} dimensions")
    return a


def _validate(R: Any, trades: Any, index: pd.DatetimeIndex, n_trials: Optional[int],
              cpcv: CPCVConfig, assume_trades_per_bar: float
              ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    R = _as_matrix(R, "returns")
    n_cfg, n_t = R.shape
    if n_t < MIN_PERIODS:
        raise AuditInputError(
            f"returns: only {n_t} periods; at least {MIN_PERIODS} are required for a "
            f"CPCV audit (a verdict on fewer periods would have no statistical power)")
    if n_t > MAX_PERIODS:
        raise AuditInputError(f"returns: {n_t} periods exceeds the limit of {MAX_PERIODS}")
    if n_cfg > MAX_CONFIGS:
        raise AuditInputError(f"returns: {n_cfg} configs exceeds the limit of {MAX_CONFIGS}; "
                              f"send the finalists and declare the full search with n_trials")
    if not np.all(np.isfinite(R)):
        bad = int((~np.isfinite(R)).sum())
        raise AuditInputError(f"returns: {bad} NaN/inf value(s); fill or drop them first")
    sd = R.std(axis=1, ddof=1)
    flat = np.where(sd == 0)[0]
    if flat.size:
        raise AuditInputError(f"returns: zero variance in config(s) {flat.tolist()} — "
                              f"a constant series cannot be audited")
    if not isinstance(index, pd.DatetimeIndex):
        raise AuditInputError("index: expected a pandas DatetimeIndex")
    if len(index) != n_t:
        raise AuditInputError(f"index: length {len(index)} does not match {n_t} return periods")
    if n_trials is not None and (int(n_trials) != n_trials or n_trials < 1):
        raise AuditInputError(f"n_trials: expected a positive integer, got {n_trials!r}")
    if n_trials is not None and n_trials > MAX_TRIALS:
        raise AuditInputError(f"n_trials: {int(n_trials)} exceeds the limit of {MAX_TRIALS}")
    if cpcv.k_test <= 0 or cpcv.n_groups <= cpcv.k_test:
        raise AuditInputError(f"cpcv: need 0 < k_test < n_groups, got k_test={cpcv.k_test}, "
                              f"n_groups={cpcv.n_groups}")
    if n_t // cpcv.n_groups < 3:
        raise AuditInputError(f"cpcv: {n_t} periods split into {cpcv.n_groups} groups leaves "
                              f"fewer than 3 periods per group; lower n_groups")
    if cpcv.purge < 0 or cpcv.embargo < 0:
        raise AuditInputError(f"cpcv: purge and embargo must be >= 0, got purge={cpcv.purge}, "
                              f"embargo={cpcv.embargo}")
    # purge/embargo are absolute bar counts: too many of them empty the train set of
    # some split, and the judge would then score nan instead of data.
    smallest_train = min(train.size for train, _, _ in cpcv_splits(n_t, cpcv))
    if smallest_train < MIN_TRAIN_PERIODS:
        raise AuditInputError(
            f"cpcv: purge={cpcv.purge} / embargo={cpcv.embargo} leave a train set of only "
            f"{smallest_train} period(s) in some CPCV split (need >= {MIN_TRAIN_PERIODS}); "
            f"lower purge/embargo or supply more data")
    if assume_trades_per_bar < 0:
        raise AuditInputError("assume_trades_per_bar must be >= 0")
    if trades is None:
        return R, None
    T = _as_matrix(trades, "trades")
    if T.shape != R.shape:
        raise AuditInputError(f"trades: shape {T.shape} does not match returns shape {R.shape}")
    if not np.all(np.isfinite(T)):
        raise AuditInputError("trades: contains NaN/inf")
    if np.any(T < 0):
        raise AuditInputError("trades: negative trade counts are not allowed")
    return R, T


# --------------------------------------------------------------------------- #
# judge
# --------------------------------------------------------------------------- #

def _judge(R: np.ndarray, trades: np.ndarray, index: pd.DatetimeIndex, names: List[str],
           n_trials_eff: int, thr: Thresholds, cpcv: CPCVConfig) -> Dict[str, Any]:
    """run_cpcv_returns, deflated by the *total* number of trials."""
    n_cfg, n_t = R.shape
    grid: List[Dict[str, Any]] = [{"config": name} for name in names]
    # Trials whose returns were not supplied still count for deflation: the runner
    # takes the total as a number, so nothing is materialised per unreported trial
    # and `n_configs_tried` in the judge dict equals n_trials_effective.
    res = run_cpcv_returns(R, trades, index, grid, cpcv, thr, n_trials=n_trials_eff)
    if "error" in res:
        raise AuditInputError(f"judge could not evaluate the CPCV paths: {res['error']} "
                              f"(try fewer --n-groups or more data)")
    res["trial_variance_source"] = ("variance of in-sample Sharpe across the supplied "
                                    "configs, averaged over CPCV splits")
    if n_cfg == 1:
        # With one series there is no selection: every CPCV path is the full series
        # in order, and no trial dispersion can be estimated. Deflate the whole
        # series by n_trials with the 1/n proxy (harness.deflated_sharpe docstring).
        res["trial_variance_source"] = "1/n proxy (single series supplied)"
        res["trial_sharpe_variance"] = 1.0 / n_t
        if n_trials_eff > 1:
            dsr = deflated_sharpe_ratio(R[0], n_trials_eff, 1.0 / n_t)
            res["deflated_sharpe_ratio"] = round(float(dsr), 4)
            res["checks"]["dsr_ok"] = bool(dsr >= thr.min_dsr)
            res["verdict"] = "PASS" if all(res["checks"].values()) else "KILL"
    res["n_trials_effective"] = n_trials_eff
    res["checks"] = {k: bool(v) for k, v in res["checks"].items()}
    return res


def _check_rows(judge: Dict[str, Any], thr: Thresholds, n_trials_eff: int) -> List[Dict[str, Any]]:
    c = judge["checks"]
    return [
        {"key": "trades_ok", "name": "Enough out-of-sample trades",
         "metric": "OOS trades (median CPCV path)", "threshold": f">= {thr.min_trades}",
         "value": judge["approx_oos_trades"], "ok": c["trades_ok"]},
        {"key": "oos_sharpe_ok", "name": "OOS Sharpe above the floor",
         "metric": "OOS Sharpe, annualized (median path)",
         "threshold": f"> {thr.min_oos_sharpe:g}",
         "value": judge["oos_sharpe_annualized"], "ok": c["oos_sharpe_ok"]},
        {"key": "drawdown_ok", "name": "Max drawdown within the limit",
         "metric": "max drawdown (median path)", "threshold": f"< {thr.max_drawdown:g}",
         "value": judge["oos_max_drawdown"], "ok": c["drawdown_ok"]},
        {"key": "dsr_ok", "name": "Deflated Sharpe survives the trials",
         "metric": f"DSR (N = {n_trials_eff} trials)", "threshold": f">= {thr.min_dsr:g}",
         "value": judge["deflated_sharpe_ratio"], "ok": c["dsr_ok"]},
    ]


def _sensitivity_row(mult: float, costs_bps: float, judge: Dict[str, Any]) -> Dict[str, Any]:
    return {"multiplier": mult, "costs_bps": round(costs_bps * mult, 4),
            "oos_sharpe_annualized": judge["oos_sharpe_annualized"],
            "worst_path_sharpe_annualized": judge["worst_path_sharpe_annualized"],
            "oos_max_drawdown": judge["oos_max_drawdown"],
            "deflated_sharpe_ratio": judge["deflated_sharpe_ratio"],
            "verdict": judge["verdict"]}


def _harness_version() -> str:
    from . import __version__  # lazy: the package __init__ imports this module
    return __version__


def audit_returns(R: Any, trades_per_bar: Any = None, index: Optional[pd.DatetimeIndex] = None,
                  n_trials: Optional[int] = None, thresholds: Optional[Thresholds] = None,
                  cpcv: Optional[CPCVConfig] = None, costs_bps: Optional[float] = None,
                  assume_trades_per_bar: float = DEFAULT_ASSUMED_TRADES_PER_BAR,
                  config_names: Optional[Sequence[str]] = None,
                  title: Optional[str] = None) -> Dict[str, Any]:
    """Audit a (configs x periods) returns matrix and return the full report dict.

    Parameters
    ----------
    R : per-period returns, shape (n_configs, n_periods) or (n_periods,).
        Treated as *net* unless `costs_bps` is given, in which case they are
        treated as gross and `costs_bps / 1e4 * trades_per_bar` is subtracted.
    trades_per_bar : same shape; number of position changes per bar. None ->
        turnover is ASSUMED at `assume_trades_per_bar` on every bar and the
        report says so.
    index : DatetimeIndex of length n_periods (only bar spacing is used).
    n_trials : how many configurations were tried in total. The DSR is deflated
        by max(n_trials, n_configs).
    thresholds, cpcv : pre-registered gates and CPCV geometry; defaults are the
        calibrated ones (Thresholds(), CPCV groups=10, k=2, purge=1, embargo=5).
    costs_bps : cost per unit of turnover in basis points, applied on top of R.
    """
    thr = thresholds if thresholds is not None else Thresholds()
    cpcv = cpcv if cpcv is not None else DEFAULT_CPCV
    if index is None:
        raise AuditInputError("index: a DatetimeIndex is required (bar spacing sets "
                              "the annualization factor)")
    R, T = _validate(R, trades_per_bar, index, n_trials, cpcv, assume_trades_per_bar)
    n_cfg, n_t = R.shape
    names = list(config_names) if config_names else [f"cfg_{i}" for i in range(n_cfg)]
    if len(names) != n_cfg:
        raise AuditInputError(f"config_names: {len(names)} names for {n_cfg} configs")

    assumptions: List[str] = []
    warnings: List[str] = []
    if T is None:
        T = np.full_like(R, float(assume_trades_per_bar))
        turnover_source = "assumed"
        assumptions.append(
            f"ASSUMPTION: trades not provided; turnover assumed at {assume_trades_per_bar:g} "
            f"trade(s) per bar for every config — the trade-count gate and any cost "
            f"deduction rest on this assumption, not on data")
    else:
        turnover_source = "provided"

    n_trials_given = int(n_trials) if n_trials is not None else n_cfg
    n_trials_eff = max(n_trials_given, n_cfg)
    if n_trials is not None and n_trials_given < n_cfg:
        warnings.append(f"n_trials={n_trials_given} is smaller than the {n_cfg} configs "
                        f"supplied; deflating by {n_cfg}.")
    if n_cfg == 1 and n_trials is None:
        warnings.append("A single series was supplied with no n_trials: the DSR is "
                        "deflated by 1 trial only. If more configurations were tried "
                        "before this one, declare them with n_trials / --trials.")

    if costs_bps is not None and costs_bps < 0:
        raise AuditInputError("costs_bps must be >= 0")
    per_unit = 0.0 if costs_bps is None else float(costs_bps) / 1e4
    R_net = R - T * per_unit
    if costs_bps is not None:
        try:
            _validate(R_net, None, index, None, cpcv, assume_trades_per_bar)
        except AuditInputError as exc:
            raise AuditInputError(f"after applying {costs_bps} bps of costs: {exc}") from exc

    judge = _judge(R_net, T, index, names, n_trials_eff, thr, cpcv)
    checks = _check_rows(judge, thr, n_trials_eff)
    failed = [c["key"] for c in checks if not c["ok"]]

    pbo_val = pbo_cscv(R_net, PBO_BLOCKS) if n_cfg >= 2 else float("nan")
    pbo: Optional[float] = None if math.isnan(pbo_val) else round(float(pbo_val), 4)

    ann = annualization_factor(index)
    deltas = index.to_series().diff().dropna()
    data = {
        "n_periods": n_t, "n_configs": n_cfg, "config_names": names,
        "ann_factor": round(float(ann), 1), "years": round(n_t / ann, 3),
        "start": index[0].isoformat(), "end": index[-1].isoformat(),
        "median_bar": str(deltas.median()) if len(deltas) else "n/a",
    }
    turnover = {
        "source": turnover_source,
        "assumed_trades_per_bar": (float(assume_trades_per_bar)
                                   if turnover_source == "assumed" else None),
        "mean_trades_per_bar": round(float(T.mean()), 4),
        "median_total_trades_per_config": float(np.median(T.sum(axis=1))),
    }
    costs = {"applied": costs_bps is not None, "costs_bps": costs_bps,
             "note": ("costs applied on top of the supplied returns (treated as gross)"
                      if costs_bps is not None else
                      "not applied: returns audited as provided (treated as net of costs)")}

    if costs_bps is None:
        cost_sensitivity: Dict[str, Any] = {
            "available": False, "rows": [],
            "reason": "costs_bps not given: returns audited as provided; pass --costs-bps "
                      "to stress-test the verdict against 0.5x / 1x / 2x costs"}
    elif turnover_source == "assumed":
        cost_sensitivity = {
            "available": False, "rows": [],
            "reason": f"trades not provided; turnover assumed at {assume_trades_per_bar:g} "
                      f"trades per bar — a sensitivity table would only restate that "
                      f"assumption (costs of {costs_bps:g} bps were applied to the headline "
                      f"on the assumed turnover)"}
    else:
        rows = []
        for mult in COST_MULTIPLIERS:
            if mult == 1.0:
                rows.append(_sensitivity_row(mult, float(costs_bps), judge))
                continue
            j = _judge(R - T * per_unit * mult, T, index, names, n_trials_eff, thr, cpcv)
            rows.append(_sensitivity_row(mult, float(costs_bps), j))
        cost_sensitivity = {"available": True, "costs_bps": float(costs_bps), "rows": rows,
                            "reason": None}

    return {
        "tool": "quant-harness qh-audit",
        "version": _harness_version(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "title": title or "strategy returns",
        "verdict": judge["verdict"],
        "checks": checks,
        "failed_checks": failed,
        "n_trials": n_trials_given,
        "n_trials_effective": n_trials_eff,
        "pbo": pbo,
        "pbo_blocks": PBO_BLOCKS,
        "data": data,
        "turnover": turnover,
        "costs": costs,
        "cost_sensitivity": cost_sensitivity,
        "judge": judge,
        "thresholds": asdict(thr),
        "cpcv": asdict(cpcv),
        "assumptions": assumptions,
        "warnings": warnings,
        "reproduce": None,
        "disclaimer": DISCLAIMER,
    }


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return "n/a" if math.isnan(float(x)) else f"{float(x):.{nd}f}"
    return str(x)


def _fmt_bar(median_bar: str) -> str:
    """'1 days 00:00:00' -> '1 day'; '0 days 04:00:00' -> '4 hours'; else as is."""
    try:
        td = pd.Timedelta(median_bar)
    except (ValueError, TypeError):
        return median_bar
    secs = td.total_seconds()
    for unit, size in (("day", 86400.0), ("hour", 3600.0), ("minute", 60.0), ("second", 1.0)):
        if secs >= size and secs % size == 0:
            n = int(secs // size)
            return f"{n} {unit}{'s' if n != 1 else ''}"
    return str(td)


def _fmt_stamp(iso: str) -> str:
    try:
        return pd.Timestamp(iso).strftime("%Y-%m-%d %H:%M %Z").strip()
    except (ValueError, TypeError):
        return iso


def _meaning(report: Dict[str, Any]) -> str:
    n = report["n_trials_effective"]
    if report["verdict"] == "PASS":
        return (f"The median out-of-sample CPCV path cleared all four pre-registered gates, "
                f"including the Deflated Sharpe corrected for {n} trials. Under the stated "
                f"assumptions the supplied returns are unlikely to be a selection artefact. "
                f"This is a statement about past evidence, not a forecast: it does not mean "
                f"the strategy will keep performing, and the judge's own detection limits "
                f"are documented in reports/CALIBRATION.md.")
    failed = ", ".join(report["failed_checks"]) or "none"
    return (f"At least one pre-registered gate failed ({failed}). Under the stated "
            f"assumptions the supplied returns cannot be told apart from what a search over "
            f"{n} trials produces by luck, or they fail a risk/liquidity gate. A KILL is a "
            f"statement about the evidence in this sample, not a claim that the idea is "
            f"worthless; more data, fewer trials or honest turnover figures can change it.")


def render_markdown(report: Dict[str, Any]) -> str:
    j, d, thr, cp = report["judge"], report["data"], report["thresholds"], report["cpcv"]
    checks = report["checks"]
    n_ok = sum(1 for c in checks if c["ok"])
    L: List[str] = []
    L.append(f"# Strategy audit — {report['title']}")
    L.append("")
    L.append(f"> ## VERDICT: {report['verdict']}")
    L.append(">")
    L.append(f"> {n_ok} of {len(checks)} pre-registered checks passed"
             + (f"; failed: {', '.join(report['failed_checks'])}." if report["failed_checks"]
                else "."))
    L.append("")
    L.append(f"Generated by quant-harness `qh-audit` v{report['version']} on "
             f"{report['generated_at']}. Statistical report — not investment advice "
             f"(see Disclaimer).")
    L.append("")
    L.append("## Data")
    L.append("")
    L.append("| | |")
    L.append("|---|---|")
    L.append(f"| Periods | {d['n_periods']} |")
    names = ", ".join(d["config_names"][:8]) + (" …" if len(d["config_names"]) > 8 else "")
    L.append(f"| Configs supplied | {d['n_configs']} ({names}) |")
    L.append(f"| Span | {_fmt_stamp(d['start'])} → {_fmt_stamp(d['end'])} "
             f"({d['years']:.2f} years) |")
    L.append(f"| Bar | {_fmt_bar(d['median_bar'])} → {d['ann_factor']:.1f} periods/year |")
    L.append(f"| Trials counted for deflation | {report['n_trials_effective']} "
             f"(declared {report['n_trials']}, supplied {d['n_configs']}) |")
    t = report["turnover"]
    if t["source"] == "assumed":
        L.append(f"| Turnover | ASSUMPTION: trades not provided; turnover assumed at "
                 f"{t['assumed_trades_per_bar']:g} trade(s)/bar |")
    else:
        L.append(f"| Turnover | provided (mean {t['mean_trades_per_bar']:.3f} trades/bar, "
                 f"median total {t['median_total_trades_per_config']:.0f} per config) |")
    c = report["costs"]
    L.append(f"| Costs | {c['costs_bps']:g} bps per unit turnover — {c['note']} |"
             if c["applied"] else f"| Costs | {c['note']} |")
    L.append(f"| Judge | CPCV groups={cp['n_groups']}, k_test={cp['k_test']}, "
             f"purge={cp['purge']}, embargo={cp['embargo']} → {j['n_splits']} splits, "
             f"{j['n_paths']} OOS paths |")
    L.append("")
    L.append("## Checks")
    L.append("")
    L.append("| Check | Metric | Threshold | Value | OK |")
    L.append("|---|---|---|---|---|")
    for ch in checks:
        L.append(f"| {ch['name']} | {ch['metric']} | {ch['threshold']} | "
                 f"{_fmt(ch['value'], 4)} | {'yes' if ch['ok'] else 'NO'} |")
    L.append("")
    L.append("## Out-of-sample statistics")
    L.append("")
    L.append("| Statistic | Value |")
    L.append("|---|---|")
    ps = j["path_sharpes_annualized"]
    L.append(f"| OOS Sharpe, annualized — median path | {_fmt(j['oos_sharpe_annualized'])} |")
    L.append(f"| OOS Sharpe, annualized — worst path | {_fmt(j['worst_path_sharpe_annualized'])} |")
    L.append(f"| OOS Sharpe, annualized — best path | {_fmt(max(ps))} |")
    L.append(f"| Max drawdown — median path | {_fmt(j['oos_max_drawdown'])} |")
    L.append(f"| OOS trades — median path | {j['approx_oos_trades']} |")
    L.append(f"| PSR vs zero — median path | {_fmt(j['psr_vs_zero'], 4)} |")
    L.append(f"| DSR — median path, N = {report['n_trials_effective']} trials | "
             f"{_fmt(j['deflated_sharpe_ratio'], 4)} |")
    L.append(f"| Trial Sharpe variance | {_fmt(j['trial_sharpe_variance'], 6)} "
             f"({j['trial_variance_source']}) |")
    pbo_txt = (_fmt(report["pbo"], 4) if report["pbo"] is not None
               else "n/a (needs >= 2 configs)")
    L.append(f"| PBO (CSCV, {report['pbo_blocks']} blocks) — informational, not a gate | "
             f"{pbo_txt} |")
    L.append(f"| Most often selected config | {j['most_common_pick'].get('config', 'n/a')} |")
    L.append("")
    L.append(f"All {len(ps)} path Sharpes (annualized): "
             + ", ".join(f"{s:+.2f}" for s in ps))
    L.append("")
    L.append("## Cost sensitivity")
    L.append("")
    cs = report["cost_sensitivity"]
    if cs["available"]:
        L.append(f"Cost basis: {cs['costs_bps']:g} bps per unit of turnover, using the "
                 f"supplied trade counts. The 1x row is the headline verdict.")
        L.append("")
        L.append("| Multiplier | bps / unit turnover | OOS Sharpe (median) | worst path | "
                 "max DD | DSR | Verdict |")
        L.append("|---|---|---|---|---|---|---|")
        for r in cs["rows"]:
            L.append(f"| {r['multiplier']:g}x | {r['costs_bps']:g} | "
                     f"{_fmt(r['oos_sharpe_annualized'])} | "
                     f"{_fmt(r['worst_path_sharpe_annualized'])} | "
                     f"{_fmt(r['oos_max_drawdown'])} | {_fmt(r['deflated_sharpe_ratio'], 4)} | "
                     f"{r['verdict']} |")
    else:
        L.append(f"Not computed — {cs['reason']}.")
    L.append("")
    if report["assumptions"] or report["warnings"]:
        L.append("## Assumptions and warnings")
        L.append("")
        for a in report["assumptions"]:
            L.append(f"- {a}")
        for w in report["warnings"]:
            L.append(f"- WARNING: {w}")
        L.append("")
    L.append("## What this verdict means")
    L.append("")
    L.append(_meaning(report))
    L.append("")
    L.append("## Reproduce")
    L.append("")
    if report["reproduce"]:
        L.append("```bash")
        L.append(report["reproduce"])
        L.append("```")
        L.append("")
        L.append("`python -m harness.audit …` is equivalent to `qh-audit …`. Thresholds: "
                 + ", ".join(f"{k}={v}" for k, v in thr.items()) + ".")
    else:
        L.append("```python")
        L.append("from harness.audit import audit_returns")
        L.append(f"audit_returns(R, trades_per_bar, index, n_trials={report['n_trials']}, "
                 f"costs_bps={c['costs_bps']!r})")
        L.append("```")
    L.append("")
    L.append("## Disclaimer")
    L.append("")
    L.append(report["disclaimer"])
    L.append("")
    return "\n".join(L)


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return o


# --------------------------------------------------------------------------- #
# CSV loading
# --------------------------------------------------------------------------- #

@dataclass
class LoadedCSV:
    values: np.ndarray                 # (n_periods, n_columns)
    names: List[str]
    index: Optional[pd.DatetimeIndex]  # from a timestamp column, if any
    had_header: bool


_NA_TOKENS = {"", "nan", "null", "na", "none"}


def _epoch_unit(median_value: float) -> str:
    """Epoch timestamps: seconds up to ~year 5000 are < 1e11; then ms / us / ns."""
    if median_value > 1e17:
        return "ns"
    if median_value > 1e14:
        return "us"
    if median_value > 1e11:
        return "ms"
    return "s"


def _parse_timestamps(col: pd.Series, path: Path) -> pd.DatetimeIndex:
    numeric = pd.to_numeric(col, errors="coerce")
    try:
        if numeric.notna().all():
            unit = _epoch_unit(float(numeric.median()))
            ts = pd.to_datetime(numeric.to_numpy(dtype="int64"), unit=unit, utc=True)
        else:
            try:
                ts = pd.to_datetime(col, utc=True)
            except (ValueError, TypeError):
                ts = pd.to_datetime(col, utc=True, format="mixed")
    except (ValueError, TypeError, OverflowError) as exc:
        raise AuditInputError(f"{path}: first column looks like timestamps but cannot be "
                              f"parsed as dates ({exc})") from exc
    idx = pd.DatetimeIndex(ts)
    if idx.isna().any():
        raise AuditInputError(f"{path}: timestamp column contains missing values")
    if idx.has_duplicates:
        raise AuditInputError(f"{path}: timestamp column contains duplicates")
    if not idx.is_monotonic_increasing:
        raise AuditInputError(f"{path}: timestamps must be sorted in increasing order")
    return idx


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _looks_like_date(token: str) -> bool:
    """A non-numeric token containing digits that pandas parses as a timestamp."""
    if _is_number(token) or not any(ch.isdigit() for ch in token):
        return False
    try:
        return not pd.isna(pd.Timestamp(token))
    except (ValueError, TypeError, OverflowError):
        return False


def _has_header(first: List[str]) -> bool:
    """The first row is a header when it has a non-numeric, non-empty cell — unless
    that cell is a date in column 0 and every other cell is numeric, which is a
    headerless data row with a timestamp column (its first bar must not be eaten)."""
    non_numeric = [tok != "" and not _is_number(tok) for tok in first]
    if not any(non_numeric):
        return False
    if non_numeric[0] and not any(non_numeric[1:]) and _looks_like_date(first[0]):
        return False
    return True


def _peek(path: Path) -> Tuple[str, List[str]]:
    """Delimiter (sniffed from a 64 KB sample; ',' when undecidable) and the
    first non-blank line split into stripped tokens."""
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            sample = fh.read(65536)
    except FileNotFoundError as exc:
        raise AuditInputError(f"{path}: file not found") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise AuditInputError(f"{path}: cannot read file ({exc})") from exc
    lines = [ln for ln in sample.splitlines() if ln.strip()]
    if not lines:
        raise AuditInputError(f"{path}: file is empty")
    try:
        sep = csv.Sniffer().sniff("\n".join(lines[:50]), delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    return sep, [tok.strip() for tok in lines[0].split(sep)]


def load_returns_csv(path: Path) -> LoadedCSV:
    """Rows = periods, columns = configs. Delimiter sniffed (',' ';' tab '|').
    Header optional — auto-detected as a first row with any non-numeric,
    non-empty cell (a lone date in column 0 is data, not a header). A timestamp
    column is recognised as the first column when its name is timestamp-like (timestamp/time/date/datetime/ts/...) or when its
    values are not numeric but parse as dates; epoch seconds/ms/us/ns are
    accepted under a timestamp-like name."""
    sep, first = _peek(path)
    had_header = _has_header(first)
    try:
        df = pd.read_csv(path, sep=sep, header=0 if had_header else None,
                         skip_blank_lines=True, skipinitialspace=True, encoding="utf-8-sig")
    except pd.errors.EmptyDataError as exc:
        raise AuditInputError(f"{path}: file is empty") from exc
    except (pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError) as exc:
        raise AuditInputError(f"{path}: cannot parse as CSV ({exc})") from exc
    if df.empty:
        raise AuditInputError(f"{path}: header only, no data rows")
    names = ([str(c).strip() for c in df.columns] if had_header
             else [f"col_{i}" for i in range(df.shape[1])])

    index: Optional[pd.DatetimeIndex] = None
    col0 = df.iloc[:, 0]
    ts_by_name = had_header and names[0].lower() in _TIMESTAMP_NAMES
    ts_by_value = (not is_numeric_dtype(col0)
                   and pd.to_numeric(col0, errors="coerce").isna().all()
                   and col0.notna().any())
    body = df
    if ts_by_name or ts_by_value:
        index = _parse_timestamps(col0, path)
        names, body = names[1:], df.iloc[:, 1:]
        if not had_header:                       # number the return columns from 0
            names = [f"col_{i}" for i in range(body.shape[1])]
    if body.shape[1] == 0:
        raise AuditInputError(f"{path}: no return columns after the timestamp column")
    values = body.apply(pd.to_numeric, errors="coerce")
    if bool(values.isna().to_numpy().any()):
        # Distinguish genuine NaN tokens (reported downstream as NaN/inf) from text.
        text = body.astype("string").apply(lambda s: s.str.strip().str.lower())
        non_numeric = values.isna() & text.notna() & ~text.isin(_NA_TOKENS)
        if bool(non_numeric.to_numpy().any()):
            r, cidx = np.argwhere(non_numeric.to_numpy())[0]
            raise AuditInputError(f"{path}: non-numeric value {body.iat[r, cidx]!r} in "
                                  f"column '{names[cidx]}', data row {r + 1}")
    return LoadedCSV(values=values.to_numpy(dtype=float), names=names, index=index,
                     had_header=had_header)


def _normalize_freq(freq: str) -> str:
    """'1d' -> '1D', '15m' -> '15min', '4H' -> '4h'. Units are case-sensitive: 'm' is
    minutes and 'M' (a calendar month) is rejected, like every alias without a
    fixed bar length."""
    m = re.fullmatch(r"\s*(\d*)\s*([A-Za-z]+)\s*", freq)
    if not m or m.group(2) not in _FREQ_UNITS:
        raise AuditInputError(
            f"--freq {freq!r} is not supported: use <n><unit> with unit in s, m/min, h, d, w "
            f"('m' = minutes; calendar months, quarters and years have no fixed bar length) "
            f"— or supply a timestamp column so the bar spacing is read from the data")
    n, unit = m.groups()
    return n + _FREQ_UNITS[unit]


def _build_index(n: int, freq: Optional[str], start: str) -> pd.DatetimeIndex:
    if not freq:
        raise AuditInputError("no timestamp column found: --freq is required "
                              "(<n><unit>, e.g. 1h, 4h, 1d, 15m, 1w)")
    alias = _normalize_freq(freq)
    try:
        return pd.date_range(start, periods=n, freq=alias, tz="UTC")
    except (ValueError, TypeError) as exc:
        raise AuditInputError(f"--freq {freq!r} / --start {start!r} rejected by pandas: "
                              f"{exc}") from exc


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qh-audit",
        description="Strategy audit: mechanical PASS/KILL verdict on a returns CSV "
                    "(rows = periods, columns = configs) via CPCV + Deflated Sharpe + PBO.",
        epilog="Exit codes: 0 PASS, 1 KILL, 2 invalid input or unwritable --out/--json, "
               "3 internal error (QH_DEBUG=1 prints the traceback). Statistical report, "
               "not investment advice.")
    p.add_argument("--returns", required=True, metavar="PATH",
                   help="CSV of per-period returns; header optional; an optional timestamp "
                        "column may come first")
    p.add_argument("--trades", metavar="PATH",
                   help="CSV of trades (position changes) per bar, same shape as --returns")
    p.add_argument("--assume-trades-per-bar", type=float, default=DEFAULT_ASSUMED_TRADES_PER_BAR,
                   metavar="X", help="turnover to ASSUME when --trades is absent (default 1.0; "
                                     "the report flags this as an assumption)")
    p.add_argument("--trials", type=int, metavar="N",
                   help="total number of configurations tried (default: number of columns)")
    p.add_argument("--freq", metavar="ALIAS",
                   help="bar frequency as <n><unit> with unit in s, m/min, h, d, w (1h, 4h, "
                        "1d, 15m, 1w; 'm' = minutes, calendar months are not supported); "
                        "required when the CSV has no timestamp column")
    p.add_argument("--start", default=DEFAULT_START, metavar="DATE",
                   help=f"first bar date used to build the index with --freq (default "
                        f"{DEFAULT_START}; only bar spacing matters)")
    p.add_argument("--costs-bps", type=float, metavar="BPS",
                   help="cost per unit of turnover in basis points; if given, returns are "
                        "treated as gross and a cost-sensitivity table is produced")
    g = p.add_argument_group("threshold overrides (defaults = calibrated Thresholds)")
    g.add_argument("--min-sharpe", type=float, metavar="X", help="min OOS annualized Sharpe")
    g.add_argument("--min-dsr", type=float, metavar="X", help="min Deflated Sharpe Ratio")
    g.add_argument("--max-dd", type=float, metavar="X", help="max drawdown (fraction)")
    g.add_argument("--min-trades", type=int, metavar="N", help="min OOS trades")
    c = p.add_argument_group("CPCV geometry")
    c.add_argument("--n-groups", type=int, default=DEFAULT_CPCV.n_groups, metavar="S")
    c.add_argument("--k-test", type=int, default=DEFAULT_CPCV.k_test, metavar="K")
    c.add_argument("--purge", type=int, default=DEFAULT_CPCV.purge, metavar="B")
    c.add_argument("--embargo", type=int, default=DEFAULT_CPCV.embargo, metavar="B")
    p.add_argument("--out", metavar="PATH.md", help="write the Markdown report here")
    p.add_argument("--json", metavar="PATH.json", help="write the full report as JSON here")
    p.add_argument("--title", help="report title (default: returns file name)")
    return p


def _thresholds_from(args: argparse.Namespace) -> Thresholds:
    thr = Thresholds()
    if args.min_sharpe is not None:
        thr.min_oos_sharpe = args.min_sharpe
    if args.min_dsr is not None:
        thr.min_dsr = args.min_dsr
    if args.max_dd is not None:
        thr.max_drawdown = args.max_dd
    if args.min_trades is not None:
        thr.min_trades = args.min_trades
    return thr


def _summary(report: Dict[str, Any], md_path: Optional[str], json_path: Optional[str]) -> str:
    j, d = report["judge"], report["data"]
    checks = " ".join(f"{c['key']}={'yes' if c['ok'] else 'NO'}" for c in report["checks"])
    lines = [
        f"qh-audit: VERDICT {report['verdict']}  ({report['title']})",
        f"  data       : {d['n_periods']} periods x {d['n_configs']} configs, "
        f"{d['years']:.2f} years, {d['ann_factor']:.0f} periods/year",
        f"  trials     : {report['n_trials_effective']} (effective)",
        f"  OOS Sharpe : {j['oos_sharpe_annualized']:+.3f} median path, "
        f"{j['worst_path_sharpe_annualized']:+.3f} worst path",
        f"  max DD     : {j['oos_max_drawdown']:.3f}   trades: {j['approx_oos_trades']}",
        f"  PSR {j['psr_vs_zero']:.4f}   DSR {j['deflated_sharpe_ratio']:.4f} "
        f"(N={report['n_trials_effective']})   PBO {_fmt(report['pbo'], 4)}",
        f"  checks     : {checks}",
    ]
    for a in report["assumptions"]:
        lines.append(f"  {a}")
    for w in report["warnings"]:
        lines.append(f"  WARNING: {w}")
    if md_path:
        lines.append(f"  report     : {md_path}")
    if json_path:
        lines.append(f"  json       : {json_path}")
    lines.append("  statistical report, not investment advice")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the audit; returns the exit code: 0 PASS, 1 KILL, 2 invalid input or
    unwritable output, 3 unexpected internal error (never a traceback with exit 1,
    which a caller would read as KILL)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:            # argparse already printed usage / help
        return int(exc.code or 0)
    try:
        ret = load_returns_csv(Path(args.returns))
        if ret.index is not None:
            index = ret.index
        else:
            index = _build_index(ret.values.shape[0], args.freq, args.start)
        trades: Optional[np.ndarray] = None
        if args.trades:
            tr = load_returns_csv(Path(args.trades))
            if tr.values.shape != ret.values.shape:
                raise AuditInputError(
                    f"trades: shape {tr.values.shape} (periods x configs) does not match "
                    f"returns shape {ret.values.shape}")
            trades = tr.values.T
        cpcv = CPCVConfig(n_groups=args.n_groups, k_test=args.k_test, purge=args.purge,
                          embargo=args.embargo)
        title = args.title or Path(args.returns).stem
        report = audit_returns(ret.values.T, trades, index, n_trials=args.trials,
                               thresholds=_thresholds_from(args), cpcv=cpcv,
                               costs_bps=args.costs_bps,
                               assume_trades_per_bar=args.assume_trades_per_bar,
                               config_names=ret.names, title=title)
    except AuditInputError as exc:
        print(f"qh-audit: input error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # CLI boundary: a crash must not exit 1 and read as KILL
        print(f"qh-audit: internal error: {type(exc).__name__}: {exc} "
              f"(set QH_DEBUG=1 for the traceback)", file=sys.stderr)
        if os.environ.get("QH_DEBUG"):
            traceback.print_exc()
        return 3
    report["reproduce"] = "qh-audit " + " ".join(shlex.quote(a) for a in argv)
    try:
        if args.out:
            Path(args.out).write_text(render_markdown(report), encoding="utf-8")
        if args.json:
            Path(args.json).write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"qh-audit: output error: cannot write {exc.filename or 'the report'}: "
              f"{exc.strerror or exc}", file=sys.stderr)
        return 2
    print(_summary(report, args.out, args.json))
    return 0 if report["verdict"] == "PASS" else 1


def _cli() -> None:
    sys.exit(main())


if __name__ == "__main__":
    _cli()
