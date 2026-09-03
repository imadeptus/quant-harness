"""Strategy audit (qh-audit): the client sends a returns matrix, gets a
mechanical PASS/KILL verdict plus a report.

Positive/negative controls mirror tests/test_detection_power.py: pure noise
must be KILLed (exit 1), a genuine ann-Sharpe-3 edge must PASS (exit 0), and
any malformed input must exit 2 with a readable message — never a traceback.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from harness.audit import AuditInputError, audit_returns, main
from harness.runner import Thresholds
from harness.walk_forward import CPCVConfig

ROOT = Path(__file__).resolve().parents[1]
BARS = 912
ANN = 365.0
IDX = pd.date_range("2024-01-01", periods=BARS, freq="1D", tz="UTC")


def _matrix(mu: float, seed: int, n_cfg: int = 6, vol: float = 0.01,
            bars: int = BARS) -> np.ndarray:
    """n_cfg independent per-bar return rows with mean `mu` (configs x T)."""
    return np.vstack([np.random.default_rng(seed + s).normal(mu, vol, bars)
                      for s in range(n_cfg)])


def _edge(ann_sharpe: float, vol: float = 0.008) -> float:
    return ann_sharpe / np.sqrt(ANN) * vol


def _trades(R: np.ndarray, every: int = 3) -> np.ndarray:
    T = np.zeros_like(R)
    T[:, ::every] = 1
    return T


def _write_csv(path: Path, R: np.ndarray, header: bool = True,
               timestamp: bool = False, names=None) -> Path:
    """Write a (configs x T) matrix as the client-facing CSV: rows = periods."""
    n_cfg, n_t = R.shape
    names = names or [f"cfg_{i}" for i in range(n_cfg)]
    df = pd.DataFrame(R.T, columns=names)
    if timestamp:
        df.insert(0, "timestamp", IDX[:n_t].strftime("%Y-%m-%d %H:%M:%S"))
    df.to_csv(path, index=False, header=header)
    return path


# --------------------------------------------------------------------------- #
# library API
# --------------------------------------------------------------------------- #

def test_audit_kills_pure_noise():
    R = _matrix(0.0, seed=0)
    rep = audit_returns(R, _trades(R), IDX)
    assert rep["verdict"] == "KILL"
    assert rep["judge"]["deflated_sharpe_ratio"] < 0.5
    assert rep["n_trials_effective"] == 6
    assert 0.0 <= rep["pbo"] <= 1.0
    assert {c["key"] for c in rep["checks"]} == {"trades_ok", "oos_sharpe_ok",
                                                 "drawdown_ok", "dsr_ok"}
    assert all(c["name"] and c["threshold"] for c in rep["checks"])
    assert "not investment advice" in rep["disclaimer"]


def test_audit_passes_a_genuine_edge():
    R = _matrix(_edge(3.0), seed=100, vol=0.008)
    rep = audit_returns(R, _trades(R), IDX)
    assert rep["verdict"] == "PASS", rep["checks"]
    assert rep["data"]["n_periods"] == BARS
    assert rep["data"]["n_configs"] == 6
    assert rep["data"]["ann_factor"] == pytest.approx(365.0, rel=0.01)
    assert rep["data"]["years"] == pytest.approx(BARS / 365.0, rel=0.01)


def test_n_trials_effective_is_max_of_given_and_configs():
    R = _matrix(_edge(3.0), seed=100, vol=0.008)
    few = audit_returns(R, _trades(R), IDX, n_trials=2)
    many = audit_returns(R, _trades(R), IDX, n_trials=200)
    assert few["n_trials_effective"] == 6
    assert many["n_trials_effective"] == 200
    # deflating by more trials can only lower the DSR
    assert many["judge"]["deflated_sharpe_ratio"] <= few["judge"]["deflated_sharpe_ratio"]


def test_single_series_dsr_is_deflated_by_trials():
    R = _matrix(_edge(3.0), seed=7, n_cfg=1, vol=0.008)
    one = audit_returns(R, _trades(R), IDX, n_trials=1)
    fifty = audit_returns(R, _trades(R), IDX, n_trials=50)
    assert one["pbo"] is None                      # PBO needs >= 2 configs
    assert fifty["judge"]["deflated_sharpe_ratio"] < one["judge"]["deflated_sharpe_ratio"]
    assert "1/n" in fifty["judge"]["trial_variance_source"]


def test_missing_trades_is_flagged_as_assumption():
    R = _matrix(0.0, seed=1)
    rep = audit_returns(R, None, IDX)
    assert rep["turnover"]["source"] == "assumed"
    assert any("ASSUMPTION" in a for a in rep["assumptions"])
    assert rep["cost_sensitivity"]["available"] is False


def test_costs_reduce_sharpe_and_sensitivity_table_is_monotone():
    R = _matrix(_edge(3.0), seed=100, vol=0.008)
    T = _trades(R)
    free = audit_returns(R, T, IDX, costs_bps=None)
    paid = audit_returns(R, T, IDX, costs_bps=20.0)
    assert paid["judge"]["oos_sharpe_annualized"] < free["judge"]["oos_sharpe_annualized"]
    cs = paid["cost_sensitivity"]
    assert cs["available"] is True
    mults = [row["multiplier"] for row in cs["rows"]]
    assert mults == [0.0, 0.5, 1.0, 2.0]
    sharpes = [row["oos_sharpe_annualized"] for row in cs["rows"]]
    assert sharpes == sorted(sharpes, reverse=True)
    # the 1x row IS the headline verdict
    assert cs["rows"][2]["oos_sharpe_annualized"] == paid["judge"]["oos_sharpe_annualized"]


def test_thresholds_and_cpcv_overrides_are_honoured():
    R = _matrix(_edge(3.0), seed=100, vol=0.008)
    strict = audit_returns(R, _trades(R), IDX, thresholds=Thresholds(min_oos_sharpe=99.0))
    assert strict["verdict"] == "KILL"
    assert [c for c in strict["checks"] if c["key"] == "oos_sharpe_ok"][0]["ok"] is False
    rep = audit_returns(R, _trades(R), IDX, cpcv=CPCVConfig(n_groups=6, k_test=2))
    assert rep["judge"]["n_paths"] == 5 and rep["judge"]["n_splits"] == 15


@pytest.mark.parametrize("bad, msg", [
    (lambda R: np.where(np.arange(R.shape[1]) == 5, np.nan, R), "NaN"),
    (lambda R: np.where(np.arange(R.shape[1]) == 5, np.inf, R), "NaN"),
    (lambda R: np.vstack([R, np.zeros(R.shape[1])]), "zero variance"),
    (lambda R: R[:, :99], "periods"),
    (lambda R: np.vstack([R] * 100), "configs"),
])
def test_input_validation_raises_readable_errors(bad, msg):
    R = bad(_matrix(0.0, seed=2))
    idx = IDX[:R.shape[1]]
    with pytest.raises(AuditInputError, match=msg):
        audit_returns(R, None, idx)


def test_trades_shape_mismatch_and_index_length_are_rejected():
    R = _matrix(0.0, seed=2)
    with pytest.raises(AuditInputError, match="shape"):
        audit_returns(R, np.ones((6, BARS - 1)), IDX)
    with pytest.raises(AuditInputError, match="index"):
        audit_returns(R, None, IDX[:-1])
    with pytest.raises(AuditInputError, match="negative"):
        audit_returns(R, -np.ones_like(R), IDX)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def test_cli_noise_exits_1_and_writes_reports(tmp_path, capsys):
    R = _matrix(0.0, seed=3)
    csv = _write_csv(tmp_path / "noise.csv", R)
    out, js = tmp_path / "r.md", tmp_path / "r.json"
    code = main(["--returns", str(csv), "--freq", "1d", "--out", str(out),
                 "--json", str(js), "--title", "Noise audit"])
    assert code == 1
    stdout = capsys.readouterr().out
    assert "KILL" in stdout
    md = out.read_text()
    assert "# Strategy audit" in md and "Noise audit" in md
    assert "VERDICT: KILL" in md
    assert "ASSUMPTION" in md                     # no --trades given
    assert "trades not provided; turnover assumed" in md
    assert "not investment advice" in md
    assert "## Reproduce" in md and "qh-audit --returns" in md
    data = json.loads(js.read_text())
    assert data["verdict"] == "KILL"
    assert data["n_trials_effective"] == 6
    assert data["turnover"]["source"] == "assumed"


def test_cli_edge_exits_0(tmp_path, capsys):
    R = _matrix(_edge(3.0), seed=100, vol=0.008)
    csv = _write_csv(tmp_path / "edge.csv", R)
    trades = _write_csv(tmp_path / "trades.csv", _trades(R))
    out = tmp_path / "edge.md"
    code = main(["--returns", str(csv), "--trades", str(trades), "--freq", "1d",
                 "--out", str(out)])
    assert code == 0
    assert "PASS" in capsys.readouterr().out
    md = out.read_text()
    assert "VERDICT: PASS" in md
    assert "ASSUMPTION" not in md                 # turnover was provided


def test_cli_single_column_no_header(tmp_path):
    R = _matrix(_edge(3.0), seed=7, n_cfg=1, vol=0.008)
    csv = _write_csv(tmp_path / "one.csv", R, header=False)
    js = tmp_path / "one.json"
    code = main(["--returns", str(csv), "--freq", "1d", "--trials", "30",
                 "--json", str(js)])
    assert code in (0, 1)
    data = json.loads(js.read_text())
    assert data["data"]["n_configs"] == 1
    assert data["data"]["n_periods"] == BARS
    assert data["n_trials_effective"] == 30
    assert data["pbo"] is None


def test_cli_header_with_timestamp_column_infers_frequency(tmp_path):
    R = _matrix(0.0, seed=4, n_cfg=3)
    csv = _write_csv(tmp_path / "ts.csv", R, header=True, timestamp=True)
    js = tmp_path / "ts.json"
    code = main(["--returns", str(csv), "--json", str(js)])   # no --freq needed
    assert code == 1
    data = json.loads(js.read_text())
    assert data["data"]["n_configs"] == 3
    assert data["data"]["config_names"] == ["cfg_0", "cfg_1", "cfg_2"]
    assert data["data"]["ann_factor"] == pytest.approx(365.0, rel=0.01)
    assert data["data"]["start"].startswith("2024-01-01")


def test_cli_epoch_timestamps_and_semicolon_delimiter(tmp_path):
    R = _matrix(0.0, seed=8, n_cfg=2)
    secs = np.array([int(t.timestamp()) for t in IDX])
    df = pd.DataFrame(R.T, columns=["a", "b"])
    df.insert(0, "timestamp", secs * 1000)                # epoch milliseconds
    df.to_csv(tmp_path / "epoch.csv", index=False, sep=";")
    js = tmp_path / "epoch.json"
    assert main(["--returns", str(tmp_path / "epoch.csv"), "--json", str(js)]) == 1
    data = json.loads(js.read_text())
    assert data["data"]["config_names"] == ["a", "b"]
    assert data["data"]["start"].startswith("2024-01-01")
    assert data["data"]["ann_factor"] == pytest.approx(365.0, rel=0.01)


def test_cli_requires_freq_without_timestamp(tmp_path, capsys):
    R = _matrix(0.0, seed=4, n_cfg=3)
    csv = _write_csv(tmp_path / "nots.csv", R)
    assert main(["--returns", str(csv)]) == 2
    assert "--freq" in capsys.readouterr().err


def test_cli_trades_file_drives_trades_check(tmp_path):
    R = _matrix(_edge(3.0), seed=100, vol=0.008)
    csv = _write_csv(tmp_path / "edge.csv", R)
    zero = _write_csv(tmp_path / "zero.csv", np.zeros_like(R))
    js = tmp_path / "z.json"
    assert main(["--returns", str(csv), "--trades", str(zero), "--freq", "1d",
                 "--json", str(js)]) == 1
    data = json.loads(js.read_text())
    assert data["judge"]["checks"]["trades_ok"] is False
    assert data["turnover"]["source"] == "provided"
    # with turnover the check is satisfied again
    some = _write_csv(tmp_path / "some.csv", _trades(R))
    assert main(["--returns", str(csv), "--trades", str(some), "--freq", "1d",
                 "--json", str(js)]) == 0
    assert json.loads(js.read_text())["judge"]["checks"]["trades_ok"] is True


def test_cli_costs_bps_lowers_sharpe(tmp_path):
    R = _matrix(_edge(3.0), seed=100, vol=0.008)
    csv = _write_csv(tmp_path / "edge.csv", R)
    trades = _write_csv(tmp_path / "trades.csv", _trades(R))
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    main(["--returns", str(csv), "--trades", str(trades), "--freq", "1d", "--json", str(a)])
    main(["--returns", str(csv), "--trades", str(trades), "--freq", "1d", "--json", str(b),
          "--costs-bps", "15"])
    sa = json.loads(a.read_text())["judge"]["oos_sharpe_annualized"]
    sb = json.loads(b.read_text())["judge"]["oos_sharpe_annualized"]
    assert sb < sa
    md_rows = json.loads(b.read_text())["cost_sensitivity"]["rows"]
    assert len(md_rows) == 4


def test_cli_threshold_overrides(tmp_path):
    R = _matrix(_edge(3.0), seed=100, vol=0.008)
    csv = _write_csv(tmp_path / "edge.csv", R)
    trades = _write_csv(tmp_path / "trades.csv", _trades(R))
    js = tmp_path / "t.json"
    code = main(["--returns", str(csv), "--trades", str(trades), "--freq", "1d",
                 "--json", str(js), "--min-sharpe", "99", "--n-groups", "6", "--k-test", "2"])
    assert code == 1
    data = json.loads(js.read_text())
    assert data["thresholds"]["min_oos_sharpe"] == 99.0
    assert data["judge"]["n_paths"] == 5


@pytest.mark.parametrize("content", [
    "this is not a csv at all\nreally, it is not\n",
    "a,b\n1,2\n",                                   # too few periods
    "",                                             # empty file
])
def test_cli_invalid_csv_exits_2(tmp_path, capsys, content):
    bad = tmp_path / "bad.csv"
    bad.write_text(content)
    assert main(["--returns", str(bad), "--freq", "1d"]) == 2
    err = capsys.readouterr().err
    assert "error" in err.lower() and "Traceback" not in err


def test_cli_nan_and_shape_mismatch_exit_2(tmp_path, capsys):
    R = _matrix(0.0, seed=5)
    R[0, 10] = np.nan
    csv = _write_csv(tmp_path / "nan.csv", R)
    assert main(["--returns", str(csv), "--freq", "1d"]) == 2
    assert "NaN" in capsys.readouterr().err
    good = _write_csv(tmp_path / "good.csv", _matrix(0.0, seed=5))
    short = _write_csv(tmp_path / "short.csv", np.ones((6, BARS - 1)))
    assert main(["--returns", str(good), "--trades", str(short), "--freq", "1d"]) == 2
    assert "shape" in capsys.readouterr().err


def test_cli_missing_file_and_bad_freq_exit_2(tmp_path, capsys):
    assert main(["--returns", str(tmp_path / "nope.csv"), "--freq", "1d"]) == 2
    csv = _write_csv(tmp_path / "x.csv", _matrix(0.0, seed=6, n_cfg=2))
    assert main(["--returns", str(csv), "--freq", "banana"]) == 2
    assert "freq" in capsys.readouterr().err.lower()


def test_module_entrypoint_and_help():
    proc = subprocess.run([sys.executable, "-m", "harness.audit", "--help"],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0
    assert "--returns" in proc.stdout and "--trials" in proc.stdout


def test_audit_quickstart_example_runs():
    proc = subprocess.run([sys.executable, str(ROOT / "examples" / "audit_quickstart.py")],
                          capture_output=True, text=True, cwd=ROOT, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert "KILL" in proc.stdout and "PASS" in proc.stdout
    sample = ROOT / "examples" / "audit_sample_returns.csv"
    assert sample.exists() and sample.stat().st_size < 200_000


# --------------------------------------------------------------------------- #
# review fixes: input edge cases, exit codes, trial accounting
# --------------------------------------------------------------------------- #

def test_headerless_csv_with_date_column_keeps_the_first_row(tmp_path):
    """A date in column 0 must not be mistaken for a header row."""
    R = _matrix(0.0, seed=9, n_cfg=3, bars=400)
    idx = IDX[:400]
    lines = [d.strftime("%Y-%m-%d") + "," + ",".join(f"{v:.6f}" for v in R[:, i])
             for i, d in enumerate(idx)]
    csv = tmp_path / "headerless_date.csv"
    csv.write_text("\n".join(lines) + "\n")
    js = tmp_path / "hd.json"
    assert main(["--returns", str(csv), "--json", str(js)]) in (0, 1)
    data = json.loads(js.read_text())
    assert data["data"]["n_periods"] == 400
    assert data["data"]["n_configs"] == 3
    assert data["data"]["config_names"] == ["col_0", "col_1", "col_2"]
    assert data["data"]["start"].startswith("2024-01-01")


@pytest.mark.parametrize("freq", ["1M", "M", "ME", "1Y", "1Q", "1mo"])
def test_calendar_freq_is_rejected_not_read_as_minutes(tmp_path, capsys, freq):
    csv = _write_csv(tmp_path / "m.csv", _matrix(0.0, seed=6, n_cfg=2, bars=120), header=False)
    assert main(["--returns", str(csv), "--freq", freq]) == 2
    err = capsys.readouterr().err
    assert "not supported" in err and "Traceback" not in err


def test_freq_units_accept_upper_case_d_and_h(tmp_path):
    csv = _write_csv(tmp_path / "f.csv", _matrix(0.0, seed=6, n_cfg=2, bars=120), header=False)
    js = tmp_path / "f.json"
    main(["--returns", str(csv), "--freq", "1D", "--json", str(js)])
    assert json.loads(js.read_text())["data"]["ann_factor"] == pytest.approx(365.0, rel=0.01)
    main(["--returns", str(csv), "--freq", "4h", "--json", str(js)])
    assert json.loads(js.read_text())["data"]["ann_factor"] == pytest.approx(365.0 * 6, rel=0.01)
    main(["--returns", str(csv), "--freq", "15m", "--json", str(js)])
    assert json.loads(js.read_text())["data"]["ann_factor"] == pytest.approx(365.0 * 96, rel=0.01)


def test_purge_embargo_that_empty_the_train_set_are_rejected(tmp_path, capsys):
    R = _matrix(0.0, seed=2, bars=400)
    with pytest.raises(AuditInputError, match="train"):
        audit_returns(R, None, IDX[:400], cpcv=CPCVConfig(10, 2, 1, 300))
    with pytest.raises(AuditInputError, match="purge"):
        audit_returns(R, None, IDX[:400], cpcv=CPCVConfig(10, 2, -1, 5))
    csv = _write_csv(tmp_path / "e.csv", R)
    assert main(["--returns", str(csv), "--freq", "1d", "--embargo", "300"]) == 2
    err = capsys.readouterr().err
    assert "train" in err and "Traceback" not in err


def test_n_trials_is_capped_and_never_materialised():
    from harness.audit import MAX_TRIALS
    R = _matrix(0.0, seed=2)
    with pytest.raises(AuditInputError, match="n_trials"):
        audit_returns(R, None, IDX, n_trials=MAX_TRIALS + 1)
    rep = audit_returns(R, None, IDX, n_trials=MAX_TRIALS)   # N is passed, not listed
    assert rep["judge"]["n_configs_tried"] == MAX_TRIALS == rep["n_trials_effective"]
    assert rep["verdict"] == "KILL"


def test_judge_n_configs_tried_equals_effective_trials_for_one_series():
    R = _matrix(_edge(3.0), seed=7, n_cfg=1, vol=0.008)
    rep = audit_returns(R, _trades(R), IDX, n_trials=50)
    assert rep["judge"]["n_configs_tried"] == 50 == rep["n_trials_effective"]


def test_cli_unwritable_output_exits_2_not_kill(tmp_path, capsys):
    csv = _write_csv(tmp_path / "n.csv", _matrix(0.0, seed=3))
    code = main(["--returns", str(csv), "--freq", "1d",
                 "--out", str(tmp_path / "no" / "such" / "dir" / "r.md")])
    assert code == 2
    err = capsys.readouterr().err
    assert "output error" in err and "Traceback" not in err


def test_cli_internal_error_exits_3(tmp_path, capsys, monkeypatch):
    import harness.audit as audit_mod

    def boom(*a, **k):
        raise RuntimeError("scipy exploded")

    monkeypatch.setattr(audit_mod, "audit_returns", boom)
    csv = _write_csv(tmp_path / "n.csv", _matrix(0.0, seed=3))
    assert main(["--returns", str(csv), "--freq", "1d"]) == 3
    err = capsys.readouterr().err
    assert "internal error" in err and "Traceback" not in err
