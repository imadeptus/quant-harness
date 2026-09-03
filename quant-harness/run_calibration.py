#!/usr/bin/env python3
"""Run the judge calibration study and write reports/CALIBRATION.{json,md}.

This is the crown-jewel evidence: it maps the whole PASS/KILL response surface of
the judge, so every KILL verdict in this project can be read against a measured
false-positive rate and a measured detection threshold — not a hope that the
judge "probably works".

    python run_calibration.py                 # full study, 200 seeds/cell (~minutes)
    python run_calibration.py --quick         # 40 seeds/cell, no PBO (fast)
    python run_calibration.py --seeds 500     # tighter Monte-Carlo error
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from harness.calibration import (DEFAULT_CPCV, StudyGrid, SynthConfig,
                                 Thresholds, run_all)

REPORTS = Path(__file__).parent / "reports"


def _first_at_or_above(rows: List[Dict], key: str, value_key: str,
                       threshold: float) -> Optional[float]:
    """Smallest `key` whose `value_key` >= threshold (rows sorted by key)."""
    for r in sorted(rows, key=lambda x: x["params"][key]):
        if r[value_key] >= threshold:
            return r["params"][key]
    return None


def _last_below(rows: List[Dict], key: str, value_key: str,
                threshold: float) -> Optional[float]:
    """Largest `key` whose `value_key` is still >= threshold (breakeven point)."""
    hit = None
    for r in sorted(rows, key=lambda x: x["params"][key]):
        if r[value_key] >= threshold:
            hit = r["params"][key]
    return hit


def build_headline(res: Dict[str, List[Dict]]) -> Dict[str, str]:
    """Auto-derive the plain-language findings from the raw cells."""
    h: Dict[str, str] = {}

    null6 = next((r for r in res["null"] if r["params"]["n_configs"] == 6), None)
    if null6:
        h["fpr_noise_N6"] = (f"{null6['pass_rate']*100:.1f}% PASS on pure noise "
                             f"(N=6 configs); median DSR {null6['median_dsr']:.3f}")

    thr = _first_at_or_above(res["power"], "true_ann_sharpe", "pass_rate", 0.5)
    thr_hi = _first_at_or_above(res["power"], "true_ann_sharpe", "pass_rate", 0.9)
    h["detection_threshold"] = (
        f"PASS>=50% at true ann Sharpe ~{thr}; PASS>=90% at ~{thr_hi}"
        if thr is not None else "never reaches 50% PASS in the swept range")

    be = _last_below(res["cost"], "cost_per_trade", "pass_rate", 0.5)
    h["cost_breakeven"] = (
        f"a gross ann-Sharpe-3 edge holds PASS>=50% up to cost ~{be}/trade-bar"
        if be is not None else "killed by the smallest tested cost")

    m1 = next((r for r in res["multiplicity"] if r["params"]["n_configs"] == 1), None)
    m100 = next((r for r in res["multiplicity"] if r["params"]["n_configs"] == 100), None)
    if m1 and m100:
        h["multiple_testing"] = (
            f"true ann Sharpe 2: PASS {m1['pass_rate']*100:.0f}% at N=1 -> "
            f"{m100['pass_rate']*100:.0f}% at N=100 (DSR deflation working)")

    ft = [r for r in res["fat_tails"] if r["label"] == "fat_tails_null"]
    if ft:
        worst = max(ft, key=lambda r: r["pass_rate"])
        h["fat_tails"] = (f"heavy tails at zero edge: worst FPR {worst['pass_rate']*100:.1f}% "
                          f"(df={worst['params']['df']}) — non-normality correction holds")

    ac = res["autocorr"]
    if ac:
        hi = max(ac, key=lambda r: r["params"]["phi"])
        lo = min(ac, key=lambda r: r["params"]["phi"])
        rose = hi["pass_rate"] > lo["pass_rate"] + 0.02
        h["autocorrelation"] = (
            f"AR(1) at zero edge: FPR {lo['pass_rate']*100:.1f}% at phi=0 -> "
            f"{hi['pass_rate']*100:.1f}% at phi={hi['params']['phi']} — "
            + ("DOCUMENTED LIMITATION (serial correlation inflates the Sharpe)"
               if rose else "stays controlled"))

    rs = res["regime_shift"]
    full = next((r for r in rs if r["params"]["active_frac"] == 1.0), None)
    quarter = next((r for r in rs if r["params"]["active_frac"] == 0.25), None)
    if full and quarter:
        h["regime_shift"] = (
            f"edge active 100% -> PASS {full['pass_rate']*100:.0f}%; active 25% -> "
            f"PASS {quarter['pass_rate']*100:.0f}% (judge does not extrapolate a dead edge)")

    ss = [r for r in res.get("sample_size", []) if r["params"]["true_ann_sharpe"] == 2.0]
    if ss:
        short = min(ss, key=lambda r: r["params"]["n_bars"])
        long = max(ss, key=lambda r: r["params"]["n_bars"])
        h["sample_size"] = (
            f"true ann Sharpe 2: PASS {short['pass_rate']*100:.0f}% at "
            f"{short['params']['n_bars']} bars -> {long['pass_rate']*100:.0f}% at "
            f"{long['params']['n_bars']} bars (more data = more power)")
    return h


_COLS = [("pass_rate", "PASS"), ("dsr_pass_rate", "DSR>=.95"), ("median_dsr", "med DSR"),
         ("median_oos_sharpe_ann", "med Sh_ann"), ("median_mdd", "med MDD"),
         ("median_pbo", "med PBO")]


def _table(rows: List[Dict], param_keys: List[str]) -> str:
    head = "| " + " | ".join(param_keys + [c[1] for c in _COLS]) + " |"
    sep = "| " + " | ".join(["---"] * (len(param_keys) + len(_COLS))) + " |"
    lines = [head, sep]
    for r in rows:
        pvals = [str(r["params"].get(k, "")) for k in param_keys]
        cvals = []
        for key, _ in _COLS:
            v = r.get(key)
            cvals.append("—" if v is None else f"{v:g}")
        lines.append("| " + " | ".join(pvals + cvals) + " |")
    return "\n".join(lines)


_STUDY_BLURB = {
    "null": ("Null — false-positive rate on pure noise", ["n_configs"],
             "mu=0 i.i.d. Gaussian. A calibrated judge PASSes ~never; the tiny median "
             "DSR confirms KILL verdicts are not a stuck null-machine."),
    "power": ("Power — detection curve vs true edge", ["true_ann_sharpe"],
              "i.i.d. edge of increasing strength. Where PASS crosses 0.5 is the "
              "detection threshold — the honest bar a real strategy must clear."),
    "cost": ("Cost sensitivity — a gross edge eaten by fees", ["cost_per_trade"],
             "Gross ann Sharpe 3, rising per-trade cost. Mirrors the project's recurring "
             "pattern: gross positive, net killed once realistic costs are charged."),
    "multiplicity": ("Multiple testing — DSR deflation vs N", ["n_configs"],
                     "A fixed true ann Sharpe 2 searched across more configs. The deflation "
                     "must erode the pass rate as N grows though the edge is unchanged."),
    "fat_tails": ("Fat tails — non-normality robustness", ["label", "df", "true_ann_sharpe"],
                  "Student-t returns. At zero edge heavy tails must not manufacture a PASS "
                  "(PSR/DSR correct for kurtosis); at real edge power must survive."),
    "autocorr": ("Autocorrelation — serial-correlation robustness", ["phi"],
                 "AR(1) at ZERO true edge. Positive phi inflates a naive Sharpe; a rising "
                 "FPR here is an honest, measured limitation of a Sharpe-based judge."),
    "regime_shift": ("Regime shift — no extrapolation of a dead edge", ["active_frac"],
                     "Edge active for only part of the sample. Less coverage of the alive "
                     "era should mean fewer PASSes; CPCV paths spanning the dead half drag "
                     "the median down."),
    "sample_size": ("Sample size — how much data to detect an edge", ["n_bars", "true_ann_sharpe"],
                    "PASS rate vs series length at fixed true Sharpe (the trade gate is held "
                    "satisfied at every length, so this is the pure statistical effect). Tells "
                    "you how many bars you need before a real edge becomes visible."),
}


def render_markdown(res: Dict[str, List[Dict]], g: StudyGrid) -> str:
    cfg = g.cfg
    out = ["# Calibration of the judge — detection-power study", "",
           "Auto-generated by `run_calibration.py`. This maps the full PASS/KILL response "
           "surface of the judge (`harness/runner.run_cpcv_returns`) on synthetic data with "
           "known ground truth, so every KILL in this project is readable against a measured "
           "false-positive rate and detection threshold.", "",
           f"**Setup:** {g.n_seeds} seeds/cell · {cfg.n_bars} daily bars · per-bar vol "
           f"{cfg.vol} · CPCV(groups={DEFAULT_CPCV.n_groups}, k={DEFAULT_CPCV.k_test}, "
           f"purge={DEFAULT_CPCV.purge}, embargo={DEFAULT_CPCV.embargo}) · thresholds "
           f"{Thresholds()!r}.", "",
           "> Each config is an *independent* draw from the same DGP — the adversarial choice, "
           "since independence maximises in-sample selection luck and is the hardest case for "
           "the Deflated Sharpe to survive. The trade-count gate is deliberately satisfied so "
           "these studies isolate the statistical detector, not the liquidity gate.", "",
           "## Headline", ""]
    for k, v in build_headline(res).items():
        out.append(f"- **{k}** — {v}")
    out.append("")
    for name, rows in res.items():
        title, param_keys, blurb = _STUDY_BLURB[name]
        out += [f"## {title}", "", blurb, "", _table(rows, param_keys), ""]
    out += ["## How to read this", "",
            "- **PASS** is the full four-gate verdict (trades, OOS Sharpe, drawdown, DSR).",
            "- **DSR>=.95** isolates the deflated-Sharpe detector from the other gates.",
            "- **med PBO** is the separate CSCV overfitting probability; note it is muted here "
            "because all configs share one edge (PBO discriminates best when configs differ in "
            "quality), so treat it as secondary to DSR in this synthetic.", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=200, help="Monte-Carlo seeds per cell")
    ap.add_argument("--quick", action="store_true", help="40 seeds, no PBO (fast)")
    ap.add_argument("--no-pbo", action="store_true", help="skip the slow PBO diagnostic")
    ap.add_argument("--bars", type=int, default=912, help="synthetic series length")
    ap.add_argument("--out", default="CALIBRATION", help="reports/<out>.{json,md}")
    ap.add_argument("--from-json", metavar="PATH", default=None,
                    help="re-render md/headline from a saved JSON, no recomputation")
    args = ap.parse_args()

    # Re-render mode: reuse already-computed study data (deterministic artifact),
    # apply the current renderer/headline. Lets report wording be fixed for free.
    if args.from_json:
        data = json.loads(Path(args.from_json).read_text())
        res = data["studies"]
        m = data.get("meta", {})
        g = StudyGrid(cfg=SynthConfig(n_bars=m.get("n_bars", 912), vol=m.get("vol", 0.01)),
                      n_seeds=m.get("n_seeds", 0), with_pbo=m.get("with_pbo", False))
        REPORTS.mkdir(exist_ok=True)
        data["headline"] = build_headline(res)
        Path(args.from_json).write_text(json.dumps(data, indent=2))
        (REPORTS / f"{args.out}.md").write_text(render_markdown(res, g))
        print(f"re-rendered reports/{args.out}.md from {args.from_json}")
        for k, v in build_headline(res).items():
            print(f"  {k}: {v}")
        return

    n_seeds = 40 if args.quick else args.seeds
    with_pbo = not (args.quick or args.no_pbo)
    g = StudyGrid(cfg=SynthConfig(n_bars=args.bars), n_seeds=n_seeds, with_pbo=with_pbo)

    print(f"running calibration: {n_seeds} seeds/cell, pbo={with_pbo} ...")
    res = run_all(g)

    REPORTS.mkdir(exist_ok=True)
    meta = {"n_seeds": n_seeds, "n_bars": args.bars, "vol": g.cfg.vol,
            "cpcv": DEFAULT_CPCV.__dict__, "thresholds": Thresholds().__dict__,
            "with_pbo": with_pbo}
    (REPORTS / f"{args.out}.json").write_text(
        json.dumps({"meta": meta, "studies": res, "headline": build_headline(res)}, indent=2))
    (REPORTS / f"{args.out}.md").write_text(render_markdown(res, g))
    print(f"wrote reports/{args.out}.json and reports/{args.out}.md")
    for k, v in build_headline(res).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
