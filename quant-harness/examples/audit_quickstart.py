#!/usr/bin/env python3
"""qh-audit quickstart — runs end to end with NO data download, in seconds.

    python examples/audit_quickstart.py

Three synthetic "client submissions" with known ground truth are audited with
`harness.audit.audit_returns`, the same judge the `qh-audit` CLI runs:

  A. pure noise, 6 configs                 -> KILL  (nothing to find)
  B. weak edge, true ann Sharpe ~1.0       -> KILL on most seeds: below the judge's
                                              measured detection threshold, i.e. the
                                              evidence in 2 years of daily bars is
                                              not enough to separate it from luck
  C. genuine edge, true ann Sharpe ~3.0    -> PASS

It also writes examples/audit_sample_returns.csv (submission B, with a
timestamp column and a header) so the CLI can be tried immediately:

    qh-audit --returns examples/audit_sample_returns.csv --trials 20 --out audit.md

Synthetic data with a known answer is the only honest demo: on real returns the
verdict is whatever the evidence supports, and this script promises nothing about
any strategy's future.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from harness.audit import audit_returns, render_markdown

BARS = 730                                   # 2 years of daily bars
N_CONFIGS = 4
IDX = pd.date_range("2024-01-01", periods=BARS, freq="1D", tz="UTC")
ANN = 365.0
SAMPLE = Path(__file__).resolve().parent / "audit_sample_returns.csv"


def submission(true_ann_sharpe: float, seed: int, n_cfg: int = N_CONFIGS,
               vol: float = 0.01, bars: int = BARS) -> tuple[np.ndarray, np.ndarray]:
    """n_cfg independent per-bar return rows with the given true annual Sharpe,
    plus a trade every 3 bars (what a client's execution log would contain)."""
    mu = true_ann_sharpe / np.sqrt(ANN) * vol
    rng = np.random.default_rng(seed)
    R = np.vstack([rng.normal(mu, vol, bars) for _ in range(n_cfg)])
    T = np.zeros_like(R)
    T[:, ::3] = 1
    return R, T


def line(name: str, rep: dict) -> str:
    j = rep["judge"]
    return (f"  {name:<28} verdict={rep['verdict']:<4} "
            f"OOS Sharpe(ann)={j['oos_sharpe_annualized']:+.2f} "
            f"worst={j['worst_path_sharpe_annualized']:+.2f} "
            f"DSR={j['deflated_sharpe_ratio']:.3f} "
            f"PBO={rep['pbo'] if rep['pbo'] is not None else 'n/a'}")


def main() -> None:
    print("qh-audit quickstart — synthetic submissions with known ground truth")
    print("=" * 72)

    noise_R, noise_T = submission(0.0, seed=11)
    weak_R, weak_T = submission(1.0, seed=12)
    strong_R, strong_T = submission(3.0, seed=13, vol=0.008, bars=912)

    noise = audit_returns(noise_R, noise_T, IDX, n_trials=20)
    weak = audit_returns(weak_R, weak_T, IDX, n_trials=20)
    strong = audit_returns(strong_R, strong_T,
                           pd.date_range("2024-01-01", periods=912, freq="1D", tz="UTC"),
                           n_trials=20)

    print(line("A. pure noise", noise))
    print(line("B. weak edge (Sharpe ~1)", weak))
    print(line("C. genuine edge (Sharpe ~3)", strong))
    assert noise["verdict"] == "KILL", noise["checks"]
    assert strong["verdict"] == "PASS", strong["checks"]
    print("  -> noise is KILLed, the genuine edge PASSes; the weak edge shows why a")
    print("     positive backtest is not evidence until it clears a calibrated bar.")
    print()

    # The judge is honest about what it does NOT know: no trade counts -> ASSUMPTION.
    blind = audit_returns(weak_R, None, IDX, n_trials=20)
    print("  Without trade counts the report flags its assumption:")
    print("   ", blind["assumptions"][0].split(".")[0])
    print()

    # Sample CSV for the CLI: rows = periods, header + timestamp column.
    df = pd.DataFrame(weak_R.T, columns=[f"cfg_{i}" for i in range(N_CONFIGS)])
    df.insert(0, "timestamp", IDX.strftime("%Y-%m-%d"))
    df.to_csv(SAMPLE, index=False, float_format="%.6f")
    shown = SAMPLE.relative_to(Path.cwd()) if SAMPLE.is_relative_to(Path.cwd()) else SAMPLE
    print(f"  Sample CSV written: {shown}")
    print(f"  Try:  qh-audit --returns {shown} --trials 20 --out audit.md")
    print()
    print("  First lines of the Markdown report for submission B:")
    for row in render_markdown(weak).splitlines()[:6]:
        print("   ", row)
    print()
    print("Done. Statistical demo, not investment advice.")


if __name__ == "__main__":
    main()
