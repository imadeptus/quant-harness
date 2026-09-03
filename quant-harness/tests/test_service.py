"""``harness.service`` — the hosted judge as an importable package (0.3.1).

What 0.3.1 added on top of the contract tests in ``tests/test_api.py``:
internal-secret auth (``QH_INTERNAL_SECRET`` / ``X-Internal-Secret``), mounting
under a path prefix (``QH_ROOT_PATH``; Vercel serves the judge at ``/api/judge``),
the ``api/*`` compatibility shims, and the Vercel function entry
``web/api/judge/index.py`` (skipped when ``web/`` is not part of the checkout,
e.g. in the published sdist).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="service tests need the `api` extra: pip install -e '.[dev,api]'")
from fastapi.testclient import TestClient  # noqa: E402

import harness  # noqa: E402
from harness.service.app import create_app  # noqa: E402
from harness.service.settings import Settings  # noqa: E402
from harness.service.vercel import DEFAULT_ROOT_PATH, create_vercel_app, vercel_settings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INDEX_PY = ROOT.parent / "web" / "api" / "judge" / "index.py"
SECRET = "s3cret-internal"
N_CFG, T = 6, 600
JSON_HDR = {"content-type": "application/json"}


def _body(**extra: Any) -> Dict[str, Any]:
    rng = np.random.default_rng(0)
    body: Dict[str, Any] = {"returns": rng.normal(0.0, 0.008, (N_CFG, T)).tolist(), "freq": "1d"}
    body.update(extra)
    return body


def _log_records(out: str) -> List[Dict[str, Any]]:
    return [json.loads(ln) for ln in out.splitlines() if ln.startswith("{")]


# ---- internal secret (QH_INTERNAL_SECRET / X-Internal-Secret) ----------------------

def test_internal_secret_required_when_configured() -> None:
    c = TestClient(create_app(Settings(internal_secret=SECRET)))
    assert c.get("/healthz").status_code == 200                   # health stays open
    r = c.post("/v1/verdict", json=_body())
    assert r.status_code == 401, r.text
    assert "X-Internal-Secret" in r.text
    assert r.headers["WWW-Authenticate"]
    assert c.post("/v1/verdict", json=_body(),
                  headers={"X-Internal-Secret": "wrong"}).status_code == 401
    assert c.post("/v1/verdict", json=_body(),
                  headers={"X-Internal-Secret": SECRET + "x"}).status_code == 401
    r = c.post("/v1/verdict", json=_body(), headers={"X-Internal-Secret": SECRET})
    assert r.status_code == 200, r.text
    assert r.json()["verdict"] == "KILL"


def test_internal_secret_gates_everything_but_healthz() -> None:
    c = TestClient(create_app(Settings(internal_secret=SECRET)))
    assert c.get("/docs").status_code == 401
    assert c.get("/openapi.json").status_code == 401
    assert c.get("/does-not-exist").status_code == 401          # 401 before 404: no route probing
    assert c.get("/openapi.json", headers={"x-internal-secret": SECRET}).status_code == 200


def test_internal_secret_is_checked_before_the_body_is_parsed() -> None:
    c = TestClient(create_app(Settings(internal_secret=SECRET)))
    assert c.post("/v1/verdict", content=b"{not json", headers=JSON_HDR).status_code == 401
    assert c.post("/v1/verdict", content=b"{not json",
                  headers={**JSON_HDR, "X-Internal-Secret": SECRET}).status_code == 422


def test_internal_secret_combines_with_api_key_gate() -> None:
    c = TestClient(create_app(Settings(internal_secret=SECRET, api_keys=("k",))))
    # secret first (401), then the API key (401), then the judge
    assert c.post("/v1/verdict", json=_body(), headers={"X-API-Key": "k"}).status_code == 401
    r = c.post("/v1/verdict", json=_body(), headers={"X-Internal-Secret": SECRET})
    assert r.status_code == 401 and "X-API-Key" in r.text
    r = c.post("/v1/verdict", json=_body(), headers={"X-Internal-Secret": SECRET, "X-API-Key": "k"})
    assert r.status_code == 200, r.text


def test_no_internal_secret_means_the_gate_is_a_noop() -> None:
    c = TestClient(create_app(Settings()))
    assert c.post("/v1/verdict", json=_body()).status_code == 200
    assert c.get("/openapi.json").status_code == 200


def test_internal_secret_is_never_logged(capsys: pytest.CaptureFixture[str]) -> None:
    c = TestClient(create_app(Settings(internal_secret=SECRET)))
    c.post("/v1/verdict", json=_body(), headers={"X-Internal-Secret": SECRET})
    c.post("/v1/verdict", json=_body(), headers={"X-Internal-Secret": "wrong-" + SECRET})
    assert SECRET not in capsys.readouterr().out


# ---- root path (QH_ROOT_PATH): / in Docker, /api/judge on Vercel -------------------

@pytest.mark.parametrize("path", ["/api/judge/healthz", "/healthz"])
def test_root_path_serves_prefixed_and_bare_paths(path: str) -> None:
    c = TestClient(create_app(Settings(root_path="/api/judge")))
    r = c.get(path)
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "ok", "version": harness.__version__}


def test_without_root_path_the_prefix_is_a_404() -> None:
    c = TestClient(create_app(Settings()))
    assert c.get("/healthz").status_code == 200
    assert c.get("/api/judge/healthz").status_code == 404


def test_verdict_under_root_path() -> None:
    c = TestClient(create_app(Settings(root_path="/api/judge")))
    r = c.post("/api/judge/v1/verdict", json=_body())
    assert r.status_code == 200, r.text
    assert r.json()["verdict"] == "KILL"


def test_auth_gates_apply_under_root_path() -> None:
    # The gates must key on the route path, not on the raw path — otherwise a
    # prefix would silently bypass them.
    c = TestClient(create_app(Settings(root_path="/api/judge", internal_secret=SECRET,
                                       api_keys=("k",))))
    assert c.post("/api/judge/v1/verdict", json=_body()).status_code == 401
    assert c.post("/api/judge/v1/verdict", json=_body(),
                  headers={"X-Internal-Secret": SECRET}).status_code == 401
    assert c.post("/v1/verdict", json=_body()).status_code == 401
    r = c.post("/api/judge/v1/verdict", json=_body(),
               headers={"X-Internal-Secret": SECRET, "X-API-Key": "k"})
    assert r.status_code == 200, r.text
    assert c.get("/api/judge/healthz").status_code == 200


def test_docs_are_served_under_root_path() -> None:
    c = TestClient(create_app(Settings(root_path="/api/judge")))
    assert c.get("/api/judge/docs").status_code == 200
    schema = c.get("/api/judge/openapi.json").json()
    assert schema["servers"][0]["url"] == "/api/judge"


def test_root_path_is_normalised() -> None:
    assert Settings(root_path="api/judge/").root_path == "/api/judge"
    assert Settings(root_path="/").root_path == ""
    assert Settings(root_path="  ").root_path == ""
    assert Settings().root_path == ""


def test_settings_from_env_reads_secret_and_root_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QH_ROOT_PATH", "/api/judge")
    monkeypatch.setenv("QH_INTERNAL_SECRET", "top")
    s = Settings.from_env()
    assert s.root_path == "/api/judge"
    assert s.internal_secret == "top"
    s = Settings.from_env({"QH_ROOT_PATH": "judge", "QH_INTERNAL_SECRET": ""})
    assert s.root_path == "/judge"
    assert s.internal_secret is None


def test_request_log_uses_the_route_path(capsys: pytest.CaptureFixture[str]) -> None:
    c = TestClient(create_app(Settings(root_path="/api/judge")))
    assert c.get("/api/judge/healthz").status_code == 200
    rec = next(x for x in _log_records(capsys.readouterr().out) if x.get("msg") == "request")
    assert rec["path"] == "/healthz"
    assert rec["root_path"] == "/api/judge"
    assert rec["status"] == 200


# ---- api/* compatibility shims -----------------------------------------------------

def test_api_shims_reexport_the_service_package() -> None:
    import api
    import api.app
    import api.logs
    import api.models
    import api.payments
    import api.report
    import api.settings
    from harness.service import app as s_app
    from harness.service import logs as s_logs
    from harness.service import models as s_models
    from harness.service import payments as s_payments
    from harness.service import report as s_report
    from harness.service import settings as s_settings

    assert api.app.app is s_app.app
    assert api.app.create_app is s_app.create_app
    assert api.app.compute_verdict is s_app.compute_verdict
    assert api.settings.Settings is s_settings.Settings
    assert api.models.VerdictRequest is s_models.VerdictRequest
    assert api.models.VerdictResponse is s_models.VerdictResponse
    assert api.payments.build_gate is s_payments.build_gate
    assert api.payments.NoopGate is s_payments.NoopGate
    assert api.report.render_report is s_report.render_report
    assert api.report.DISCLAIMER is s_report.DISCLAIMER
    assert api.logs.configure_logging is s_logs.configure_logging
    assert api.__version__ == harness.__version__


def test_shim_app_answers_healthz() -> None:
    from api.app import app as shim_app
    assert TestClient(shim_app).get("/healthz").json()["status"] == "ok"


def test_service_package_lazy_exports() -> None:
    import harness.service as svc
    assert svc.Settings is Settings
    assert svc.create_app is create_app
    with pytest.raises(AttributeError):
        svc.nope  # type: ignore[attr-defined]


# ---- Vercel function entry -----------------------------------------------------------

def test_vercel_settings_defaults() -> None:
    s = vercel_settings({})
    assert s.root_path == DEFAULT_ROOT_PATH == "/api/judge"
    assert s.internal_secret is None
    assert s.max_configs == Settings.max_configs
    assert s.max_body_bytes == Settings.max_body_bytes


def test_vercel_settings_bridges_the_node_layer_env() -> None:
    s = vercel_settings({"JUDGE_INTERNAL_SECRET": "from-node", "APP_URL": "https://qh.example/",
                         "MAX_CONFIGS": "7", "MAX_PERIODS": "1000", "MAX_BODY_MB": "20"})
    assert s.internal_secret == "from-node"
    assert s.public_url == "https://qh.example"
    assert s.max_configs == 7 and s.max_periods == 1000
    assert s.max_body_bytes == 20 * 1024 * 1024
    # QH_* always wins over the bridged names
    s = vercel_settings({"QH_INTERNAL_SECRET": "own", "JUDGE_INTERNAL_SECRET": "node",
                         "QH_ROOT_PATH": "/judge", "QH_MAX_CONFIGS": "3", "MAX_CONFIGS": "9"})
    assert s.internal_secret == "own" and s.root_path == "/judge" and s.max_configs == 3


def test_vercel_settings_rejects_bad_bridged_values() -> None:
    with pytest.raises(ValueError, match="MAX_BODY_MB"):
        vercel_settings({"MAX_BODY_MB": "lots"})
    with pytest.raises(ValueError, match="MAX_CONFIGS"):
        vercel_settings({"MAX_CONFIGS": "0"})


def test_vercel_production_without_secret_fails_fast() -> None:
    with pytest.raises(RuntimeError, match="JUDGE_INTERNAL_SECRET"):
        vercel_settings({"VERCEL_ENV": "production"})
    assert vercel_settings({"VERCEL_ENV": "production", "JUDGE_INTERNAL_SECRET": "x"}).internal_secret == "x"
    assert vercel_settings({"VERCEL_ENV": "preview"}).internal_secret is None   # warns, serves


def test_create_vercel_app_serves_under_api_judge() -> None:
    app = create_vercel_app({"JUDGE_INTERNAL_SECRET": SECRET})
    c = TestClient(app)
    assert c.get("/api/judge/healthz").status_code == 200
    assert c.get("/healthz").status_code == 200
    assert c.post("/api/judge/v1/verdict", json=_body()).status_code == 401
    r = c.post("/api/judge/v1/verdict", json=_body(), headers={"X-Internal-Secret": SECRET})
    assert r.status_code == 200, r.text
    assert r.json()["verdict"] == "KILL"


def _load_index_py() -> ModuleType:
    if not INDEX_PY.exists():
        pytest.skip("web/api/judge/index.py is not part of this checkout")
    spec = importlib.util.spec_from_file_location("qh_vercel_index_under_test", INDEX_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_web_api_judge_index_exposes_the_app(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VERCEL_ENV", raising=False)
    monkeypatch.delenv("QH_ROOT_PATH", raising=False)
    monkeypatch.delenv("QH_INTERNAL_SECRET", raising=False)
    monkeypatch.setenv("JUDGE_INTERNAL_SECRET", SECRET)     # the Node layer's variable name
    index = _load_index_py()
    c = TestClient(index.app)                                # `app`: the name Vercel looks for
    r = c.get("/api/judge/healthz")
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "ok", "version": harness.__version__}
    assert c.post("/api/judge/v1/verdict", json=_body()).status_code == 401
    r = c.post("/api/judge/v1/verdict", json=_body(), headers={"X-Internal-Secret": SECRET})
    assert r.status_code == 200, r.text


def test_web_requirements_pin_this_package_version() -> None:
    req = ROOT.parent / "web" / "requirements.txt"
    if not req.exists():
        pytest.skip("web/requirements.txt is not part of this checkout")
    lines = [ln.strip() for ln in req.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    assert f"quant-harness=={harness.__version__}" in lines
    assert any(ln.startswith("fastapi") for ln in lines)
    assert any(ln.startswith("pydantic") for ln in lines)


# ---- metrics.path_sharpes_annualized (the product plots the per-path distribution) -----

def test_metrics_carry_the_per_path_sharpes() -> None:
    c = TestClient(create_app(Settings()))
    r = c.post("/v1/verdict", json=_body())
    assert r.status_code == 200, r.text
    m = r.json()["metrics"]
    ps = m["path_sharpes_annualized"]
    assert isinstance(ps, list) and len(ps) == m["n_paths"] == 9        # C(10, 2) -> 9 paths
    assert all(isinstance(s, float) for s in ps)
    assert m["worst_path_sharpe_annualized"] == min(ps)
    assert m["oos_sharpe_annualized"] == pytest.approx(float(np.median(ps)), abs=1e-3)
