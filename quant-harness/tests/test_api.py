"""Hosted verdict API — contract tests (no network, small matrices, < 60 s).

Every test builds its own app via ``create_app(Settings(...))`` so the payment /
auth / limit configuration is explicit and independent of the process env.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi", reason="API tests need the `api` extra: pip install -e '.[dev,api]'")
from fastapi.testclient import TestClient  # noqa: E402

from api.app import create_app  # noqa: E402
from api.settings import Settings  # noqa: E402
from harness.audit import DEFAULT_CPCV, audit_returns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N_CFG, T = 6, 600
VOL = 0.008
ANN = 365.0  # daily bars


def _matrix(ann_sharpe: float, seed: int = 0) -> List[List[float]]:
    mu = ann_sharpe / np.sqrt(ANN) * VOL
    rng = np.random.default_rng(seed)
    return rng.normal(mu, VOL, (N_CFG, T)).tolist()


def _trades(every: int = 2) -> List[List[int]]:
    t = np.zeros((N_CFG, T), dtype=int)
    t[:, ::every] = 1
    return t.tolist()


def _body(ann_sharpe: float, **extra: Any) -> Dict[str, Any]:
    body: Dict[str, Any] = {"returns": _matrix(ann_sharpe), "trades": _trades(), "freq": "1d"}
    body.update(extra)
    return body


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(Settings()))


# ---- healthz -------------------------------------------------------------------

def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["version"], str) and body["version"]


# ---- verdict: the detector is wired correctly -----------------------------------

def test_noise_is_killed(client: TestClient) -> None:
    r = client.post("/v1/verdict", json=_body(0.0))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] == "KILL"
    assert body["checks"]["dsr_ok"] is False
    m = body["metrics"]
    assert m["n_configs_tried"] == N_CFG
    assert m["n_paths"] == 9                       # C(9, 1) for CPCV(10, 2)
    assert m["oos_bars"] == T
    assert 0.0 <= m["deflated_sharpe_ratio"] <= 1.0
    assert 0.0 <= m["pbo"] <= 1.0
    assert "not investment advice" in body["disclaimer"].lower()
    assert "not investment advice" in body["report_md"].lower()
    assert body["report_md"].startswith("#")


def test_strong_edge_passes(client: TestClient) -> None:
    r = client.post("/v1/verdict", json=_body(4.0))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] == "PASS", body["checks"]
    assert all(body["checks"].values())
    assert body["metrics"]["deflated_sharpe_ratio"] >= 0.95
    assert body["metrics"]["worst_path_sharpe_annualized"] > 0
    assert body["thresholds"]["min_dsr"] == 0.95


def test_single_series_without_trades(client: TestClient) -> None:
    body = {"returns": _matrix(4.0)[0], "freq": "1d"}
    r = client.post("/v1/verdict", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["metrics"]["n_configs_tried"] == 1
    assert out["metrics"]["pbo"] is None            # PBO needs >= 2 configs
    assert out["metrics"]["approx_oos_trades"] == T  # assumed one trade per bar
    assert any("trades" in a for a in out["assumptions"])


def test_n_trials_deflates_single_series(client: TestClient) -> None:
    series = _matrix(2.0)[0]
    plain = client.post("/v1/verdict", json={"returns": series, "freq": "1d"}).json()
    deflated = client.post("/v1/verdict",
                           json={"returns": series, "freq": "1d", "n_trials": 50}).json()
    assert deflated["metrics"]["n_configs_tried"] == 50
    assert deflated["metrics"]["deflated_sharpe_ratio"] < plain["metrics"]["deflated_sharpe_ratio"]
    assert any("n_trials" in a for a in deflated["assumptions"])


def test_n_trials_deflates_matrix(client: TestClient) -> None:
    plain = client.post("/v1/verdict", json=_body(2.0)).json()
    deflated = client.post("/v1/verdict", json=_body(2.0, n_trials=100)).json()
    assert deflated["metrics"]["n_configs_tried"] == 100
    assert deflated["metrics"]["deflated_sharpe_ratio"] < plain["metrics"]["deflated_sharpe_ratio"]


def test_costs_bps_reduce_sharpe(client: TestClient) -> None:
    gross = client.post("/v1/verdict", json=_body(3.0)).json()
    net = client.post("/v1/verdict", json=_body(3.0, costs_bps=20.0)).json()
    assert net["metrics"]["oos_sharpe_annualized"] < gross["metrics"]["oos_sharpe_annualized"]
    assert any("costs_bps" in a for a in net["assumptions"])


def test_custom_thresholds_and_cpcv_are_echoed(client: TestClient) -> None:
    body = _body(4.0, thresholds={"min_trades": 10, "min_oos_sharpe": 0.5,
                                  "max_drawdown": 0.5, "min_dsr": 0.9},
                 cpcv={"n_groups": 6, "k_test": 2})
    r = client.post("/v1/verdict", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["thresholds"] == {"min_trades": 10, "min_oos_sharpe": 0.5,
                                 "max_drawdown": 0.5, "min_dsr": 0.9}
    assert out["metrics"]["n_paths"] == 5          # C(5, 1)


def test_hourly_freq_changes_annualization(client: TestClient) -> None:
    daily = client.post("/v1/verdict", json=_body(1.0, freq="1d")).json()
    hourly = client.post("/v1/verdict", json=_body(1.0, freq="1h")).json()
    ratio = hourly["metrics"]["oos_sharpe_annualized"] / daily["metrics"]["oos_sharpe_annualized"]
    assert ratio == pytest.approx(np.sqrt(24.0), rel=0.05)


# ---- the API is the same judge as qh-audit ---------------------------------------

def test_default_cpcv_is_the_calibrated_geometry(client: TestClient) -> None:
    out = client.post("/v1/verdict", json=_body(0.0)).json()
    assert out["cpcv"] == {"n_groups": DEFAULT_CPCV.n_groups, "k_test": DEFAULT_CPCV.k_test,
                           "purge": DEFAULT_CPCV.purge, "embargo": DEFAULT_CPCV.embargo}
    assert out["cpcv"]["embargo"] == 5                 # reports/CALIBRATION.md geometry


def test_verdict_matches_audit_returns(client: TestClient) -> None:
    body = _body(2.5, n_trials=40, costs_bps=5.0)
    out = client.post("/v1/verdict", json=body).json()
    idx = pd.date_range("2020-01-01", periods=T, freq="1D", tz="UTC")
    ref = audit_returns(np.asarray(body["returns"]), np.asarray(body["trades"]), idx,
                        n_trials=40, costs_bps=5.0)
    assert out["verdict"] == ref["verdict"]
    assert out["checks"] == {c["key"]: c["ok"] for c in ref["checks"]}
    assert out["metrics"]["n_configs_tried"] == ref["n_trials_effective"] == 40
    assert out["metrics"]["deflated_sharpe_ratio"] == pytest.approx(
        ref["judge"]["deflated_sharpe_ratio"])
    assert out["metrics"]["pbo"] == pytest.approx(ref["pbo"])
    assert out["thresholds"] == ref["thresholds"]


def test_cost_sensitivity_table_when_trades_and_costs(client: TestClient) -> None:
    out = client.post("/v1/verdict", json=_body(3.0, costs_bps=20.0)).json()
    table = out["cost_sensitivity"]
    assert [row["multiplier"] for row in table] == [0.0, 0.5, 1.0, 2.0]
    assert [row["costs_bps"] for row in table] == [0.0, 10.0, 20.0, 40.0]
    sharpes = [row["oos_sharpe_annualized"] for row in table]
    assert sharpes == sorted(sharpes, reverse=True)           # more cost, less Sharpe
    assert table[2]["oos_sharpe_annualized"] == out["metrics"]["oos_sharpe_annualized"]
    assert table[2]["verdict"] == out["verdict"]
    assert all(row["verdict"] in ("PASS", "KILL") for row in table)


def test_cost_sensitivity_is_null_without_trades_or_costs(client: TestClient) -> None:
    assert client.post("/v1/verdict", json=_body(3.0)).json()["cost_sensitivity"] is None
    no_trades = client.post("/v1/verdict", json={"returns": _matrix(3.0), "freq": "1d",
                                                 "costs_bps": 20.0}).json()
    assert no_trades["cost_sensitivity"] is None
    assert any("turnover assumed" in a for a in no_trades["assumptions"])


def test_assume_trades_per_bar_drives_the_trade_count(client: TestClient) -> None:
    body = {"returns": _matrix(4.0)[0], "freq": "1d", "assume_trades_per_bar": 0.5}
    out = client.post("/v1/verdict", json=body).json()
    assert out["metrics"]["approx_oos_trades"] == T // 2
    assert any("0.5" in a for a in out["assumptions"])
    r = client.post("/v1/verdict", json={**body, "assume_trades_per_bar": -1})
    assert r.status_code == 422


# ---- validation -> 422 ----------------------------------------------------------

@pytest.mark.parametrize("mutate, needle", [
    (lambda b: b.__setitem__("returns", [[0.0] * T, [0.0] * (T - 1)]), "rectangular"),
    (lambda b: b["returns"][0].__setitem__(3, float("nan")), "finite"),
    (lambda b: b["returns"][1].__setitem__(5, float("inf")), "finite"),
    (lambda b: b.__setitem__("returns", [[0.001] * 50] * 3), "100"),
    (lambda b: b.__setitem__("trades", _trades()[:3]), "trades"),
    (lambda b: b.__setitem__("freq", "1mo"), "freq"),
    (lambda b: b.__setitem__("n_trials", 2), "n_trials"),
    (lambda b: b.__setitem__("cpcv", {"n_groups": 4, "k_test": 4}), "k_test"),
    (lambda b: b.__setitem__("returns", [[[0.1]]]), ""),
    (lambda b: b.__setitem__("returns", "not a list"), ""),
])
def test_invalid_input_is_422(client: TestClient, mutate, needle: str) -> None:
    body = _body(0.0)
    mutate(body)
    # Raw text on purpose: httpx's json= refuses NaN/inf, but real clients send them.
    r = client.post("/v1/verdict", content=json.dumps(body),
                    headers={"content-type": "application/json"})
    assert r.status_code == 422, r.text
    out = r.json()
    assert "detail" in out and "error" in out
    assert needle in json.dumps(out).lower() if needle else True


def test_zero_variance_series_is_422(client: TestClient) -> None:
    r = client.post("/v1/verdict", json={"returns": [0.0] * T, "freq": "1d"})
    assert r.status_code == 422, r.text
    assert "variance" in r.text.lower()


# ---- limits -> 413 --------------------------------------------------------------

def test_too_many_configs_is_413() -> None:
    c = TestClient(create_app(Settings(max_configs=3)))
    r = c.post("/v1/verdict", json=_body(0.0))
    assert r.status_code == 413, r.text
    assert "QH_MAX_CONFIGS" in r.text


def test_too_many_periods_is_413() -> None:
    c = TestClient(create_app(Settings(max_periods=500)))
    r = c.post("/v1/verdict", json=_body(0.0))
    assert r.status_code == 413, r.text
    assert "QH_MAX_PERIODS" in r.text


def test_body_too_large_is_413() -> None:
    c = TestClient(create_app(Settings(max_body_bytes=1024)))
    r = c.post("/v1/verdict", json=_body(0.0))
    assert r.status_code == 413, r.text
    assert "QH_MAX_BODY_BYTES" in r.text


def test_chunked_body_too_large_is_413() -> None:
    # No Content-Length: the limiter must count the streamed bytes instead.
    c = TestClient(create_app(Settings(max_body_bytes=1024)))
    payload = json.dumps(_body(0.0)).encode()
    chunks = (payload[i:i + 4096] for i in range(0, len(payload), 4096))
    r = c.post("/v1/verdict", content=chunks, headers={"content-type": "application/json"})
    assert r.status_code == 413, r.text
    assert "QH_MAX_BODY_BYTES" in r.text


# ---- auth -----------------------------------------------------------------------

def test_api_key_required_when_configured() -> None:
    c = TestClient(create_app(Settings(api_keys=("s3cret", "other"))))
    assert c.get("/healthz").status_code == 200                   # health stays open
    assert c.post("/v1/verdict", json=_body(0.0)).status_code == 401
    assert c.post("/v1/verdict", json=_body(0.0),
                  headers={"X-API-Key": "wrong"}).status_code == 401
    r = c.post("/v1/verdict", json=_body(0.0), headers={"X-API-Key": "other"})
    assert r.status_code == 200, r.text


# ---- payment gates (stubs) -------------------------------------------------------

def test_x402_gate_requires_payment_header() -> None:
    c = TestClient(create_app(Settings(payment_gate="x402", x402_pay_to="0xabc",
                                       x402_price_usdc=0.05, allow_stub_payment_gate=True)))
    r = c.post("/v1/verdict", json=_body(0.0))
    assert r.status_code == 402, r.text
    body = r.json()
    req = body["accepts"][0]
    assert req["scheme"] == "exact"
    assert req["payTo"] == "0xabc"
    assert req["maxAmountRequired"] == "50000"      # 0.05 USDC in 6-decimal atomic units
    assert req["asset"] and req["network"] and req["description"]
    assert body["stub"] is True

    r = c.post("/v1/verdict", json=_body(0.0), headers={"X-PAYMENT": "anything"})
    assert r.status_code == 200, r.text            # stub: header accepted, not verified


def test_nowpayments_gate_requires_payment_id() -> None:
    c = TestClient(create_app(Settings(payment_gate="nowpayments",
                                       nowpayments_api_key="np-key",
                                       allow_stub_payment_gate=True)))
    r = c.post("/v1/verdict", json=_body(0.0))
    assert r.status_code == 402, r.text
    body = r.json()
    assert body["provider"] == "nowpayments"
    assert "X-Payment-Id" in body["instructions"]
    assert body["stub"] is True

    r = c.post("/v1/verdict", json=_body(0.0), headers={"X-Payment-Id": "12345"})
    assert r.status_code == 200, r.text


def test_misconfigured_gate_fails_fast() -> None:
    with pytest.raises(ValueError):
        create_app(Settings(payment_gate="x402"))          # no pay_to / price
    with pytest.raises(ValueError):
        create_app(Settings(payment_gate="stripe"))        # unknown gate


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QH_MAX_CONFIGS", "7")
    monkeypatch.setenv("QH_API_KEYS", "a, b ,")
    monkeypatch.setenv("QH_PAYMENT_GATE", "x402")
    monkeypatch.setenv("QH_X402_PAY_TO", "0xdead")
    monkeypatch.setenv("QH_X402_PRICE_USDC", "0.10")
    s = Settings.from_env()
    assert s.max_configs == 7
    assert s.api_keys == ("a", "b")
    assert s.payment_gate == "x402" and s.x402_pay_to == "0xdead"
    assert s.x402_price_usdc == pytest.approx(0.10)


# ---- JSON request log ------------------------------------------------------------

def test_request_log_is_json_with_verdict(capsys: pytest.CaptureFixture[str]) -> None:
    c = TestClient(create_app(Settings()))
    r = c.post("/v1/verdict", json=_body(0.0))
    assert r.status_code == 200
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("{")]
    recs = [json.loads(ln) for ln in lines]
    rec = next(x for x in recs if x.get("path") == "/v1/verdict")
    assert rec["status"] == 200
    assert rec["verdict"] == "KILL"
    assert rec["n_configs"] == N_CFG and rec["n_periods"] == T
    assert isinstance(rec["ms"], (int, float)) and rec["ms"] >= 0


# ---- review fixes: limits, freq parsing, processing order, stub opt-in --------------

def test_huge_n_trials_is_422(client: TestClient) -> None:
    r = client.post("/v1/verdict", json=_body(0.0, n_trials=10**9))
    assert r.status_code == 422, r.text
    assert "n_trials" in r.text


def test_calendar_month_freq_is_422_not_minutes(client: TestClient) -> None:
    r = client.post("/v1/verdict", json=_body(0.0, freq="1M"))
    assert r.status_code == 422, r.text
    assert "month" in r.text.lower()
    r = client.post("/v1/verdict", json=_body(0.0, freq="1D"))
    assert r.status_code == 200, r.text
    assert r.json()["metrics"]["ann_factor"] == pytest.approx(365.0, rel=0.01)


def test_absurd_freq_multiplier_is_422_not_500(client: TestClient) -> None:
    r = client.post("/v1/verdict", json=_body(0.0, freq="1000000000d"))
    assert r.status_code == 422, r.text
    assert "freq" in r.text.lower()


def test_degenerate_embargo_is_422_not_a_nan_kill(client: TestClient) -> None:
    r = client.post("/v1/verdict", json=_body(0.0, cpcv={"embargo": 1000}))
    assert r.status_code == 422, r.text
    assert "train" in r.text.lower()


def test_api_key_is_checked_before_the_body_is_parsed() -> None:
    c = TestClient(create_app(Settings(api_keys=("k",))))
    hdr = {"content-type": "application/json"}
    assert c.post("/v1/verdict", content=b"{not json", headers=hdr).status_code == 401
    assert c.post("/v1/verdict", content=b"{not json",
                  headers={**hdr, "X-API-Key": "k"}).status_code == 422


def test_payment_gate_is_checked_before_the_body_is_parsed() -> None:
    c = TestClient(create_app(Settings(payment_gate="x402", x402_pay_to="0xabc",
                                       x402_price_usdc=0.05, allow_stub_payment_gate=True)))
    r = c.post("/v1/verdict", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 402, r.text


def test_stub_gate_needs_explicit_opt_in_and_marks_unverified_responses() -> None:
    with pytest.raises(ValueError, match="stub"):
        create_app(Settings(payment_gate="x402", x402_pay_to="0xabc", x402_price_usdc=0.05))
    with pytest.raises(ValueError, match="stub"):
        create_app(Settings(payment_gate="nowpayments", nowpayments_api_key="np-key"))
    c = TestClient(create_app(Settings(payment_gate="x402", x402_pay_to="0xabc",
                                       x402_price_usdc=0.05, allow_stub_payment_gate=True)))
    r = c.post("/v1/verdict", json=_body(0.0), headers={"X-PAYMENT": "x"})
    assert r.status_code == 200, r.text
    assert r.headers["X-QH-Payment"] == "unverified-stub"
    assert any("not verified" in a.lower() for a in r.json()["assumptions"])
    assert "not verified" in r.json()["report_md"].lower()
    # the free gate adds no marker
    plain = client_free = TestClient(create_app(Settings())).post("/v1/verdict", json=_body(0.0))
    assert "X-QH-Payment" not in plain.headers


def test_env_example_has_no_inline_comments() -> None:
    # `docker run --env-file` takes values literally: `X=1  # note` would set X to "1  # note".
    for line in (ROOT / "api" / ".env.example").read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        assert re.fullmatch(r"[A-Z_][A-Z0-9_]*=[^#]*", s), line


def test_dotenv_files_are_ignored_by_git_and_docker() -> None:
    for f in (ROOT / ".dockerignore", ROOT.parent / ".gitignore"):
        text = f.read_text(encoding="utf-8")
        assert "\n.env\n" in text and "*.env" in text and "!*.env.example" in text, f
