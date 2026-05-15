"""Tests for the /api/gpu/topology route.

The route shells out to ``sddc gpu top --json``; we mock the
subprocess so the test runs without sddc-cli installed and without a
GPU. Coverage:
  * admin gate (no cookie → 401)
  * sddc missing on PATH → 200 with {"error": "..."}
  * sddc returns valid json → 200 with the parsed payload
  * second call within TTL hits the cache (subprocess called once)
"""

from __future__ import annotations

import hmac as _hmac
import json
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from meeting_scribe.routes.diagnostics import router as diagnostics_router
from meeting_scribe.runtime import state
from meeting_scribe.terminal.auth import AdminSecretStore, CookieSigner

_TEST_CRED = "TopologyTestPwd"


@pytest.fixture
def app(tmp_path, monkeypatch):
    """FastAPI app wired with the diagnostics router and a cookie signer."""
    secret_store = AdminSecretStore.load_or_create(tmp_path / "admin-secret")
    state_dir = tmp_path / "scribe-state"
    state_dir.mkdir(parents=True, exist_ok=True)
    expected = _hmac.new(secret_store.secret, _TEST_CRED.encode(), "sha256").hexdigest()
    (state_dir / "admin-password-hmac").write_text(expected)
    monkeypatch.setenv("SCRIBE_STATE_DIR", str(state_dir))
    cookie_signer = CookieSigner(secret_store.secret, max_age_seconds=60)
    monkeypatch.setattr(state, "_terminal_cookie_signer", cookie_signer)
    a = FastAPI()
    a.include_router(diagnostics_router)
    # Reset module-level cache so each test starts cold.
    import meeting_scribe.routes.diagnostics as diag

    diag._GPU_CACHE = None
    diag._GPU_CACHE_TS = 0.0
    diag._GPU_REFRESH_IN_FLIGHT = False
    return a, cookie_signer


def _admin_cookie(cookie_signer):
    return cookie_signer.issue()


def test_topology_requires_admin_cookie(app):
    a, _ = app
    with TestClient(a, base_url="https://test") as client:
        resp = client.get("/api/gpu/topology")
    # _require_admin_response returns 403 (not 401) for back-compat with
    # existing JSON clients — see admin_guard._require_admin_response.
    assert resp.status_code == 403


def test_topology_reports_missing_sddc(app):
    a, signer = app
    with (
        mock.patch("meeting_scribe.routes.diagnostics._resolve_sddc_cli", return_value=None),
        TestClient(a, base_url="https://test") as client,
    ):
        client.cookies.set("scribe_admin", _admin_cookie(signer))
        resp = client.get("/api/gpu/topology")
    assert resp.status_code == 200
    body = resp.json()
    assert "error" in body and "PATH" in body["error"]


def test_resolve_sddc_cli_falls_back_to_mise(tmp_path, monkeypatch):
    """When shutil.which fails, walk the mise python install dirs."""
    import meeting_scribe.routes.diagnostics as diag

    fake_home = tmp_path
    mise_dir = fake_home / ".local/share/mise/installs/python/3.14.4/bin"
    mise_dir.mkdir(parents=True)
    sddc_bin = mise_dir / "sddc"
    sddc_bin.write_text("#!/bin/sh\n")
    sddc_bin.chmod(0o755)

    monkeypatch.setattr(diag._gpu_shutil, "which", lambda _name: None)
    monkeypatch.setenv("HOME", str(fake_home))
    # Path.home() reads HOME on POSIX, so the env override takes effect.
    found = diag._resolve_sddc_cli()
    assert found == str(sddc_bin)


def test_topology_returns_parsed_json_and_caches(app):
    a, signer = app
    payload = {
        "timestamp": 1700000000.0,
        "card": {"sm_pct": 96, "vram_used_mb": 99000, "vram_total_mb": 121600},
        "processes": [
            {"pid": 1, "role": "translate", "container": "autosre", "sm_pct": 92, "vram_mb": 85635}
        ],
        "engines": [
            {
                "name": "autosre-translate",
                "reachable": True,
                "model": "Qwen3.6-FP8",
                "running": 1,
                "waiting": 0,
                "kv_pct": 0.0,
                "gen_tok_per_s": None,
                "ttft_p50_ms": None,
                "tpot_p50_ms": None,
                "mfu_pct": None,
            }
        ],
    }
    completed = mock.Mock(returncode=0, stdout=json.dumps(payload), stderr="")
    with (
        mock.patch("shutil.which", return_value="/usr/bin/sddc"),
        mock.patch("subprocess.run", return_value=completed) as run_mock,
        TestClient(a, base_url="https://test") as client,
    ):
        client.cookies.set("scribe_admin", _admin_cookie(signer))

        first = client.get("/api/gpu/topology")
        assert first.status_code == 200
        assert first.json()["card"]["sm_pct"] == 96

        second = client.get("/api/gpu/topology")
        assert second.status_code == 200
        # Cache TTL is 2s; the second call inside the same TestClient
        # context window must not re-shell out.
        assert run_mock.call_count == 1


def test_topology_handles_subprocess_timeout(app):
    """A hung nvidia-smi must not block the diagnostics panel."""
    a, signer = app
    import subprocess as _sp

    with (
        mock.patch("shutil.which", return_value="/usr/bin/sddc"),
        mock.patch("subprocess.run", side_effect=_sp.TimeoutExpired(cmd="sddc", timeout=8)),
        TestClient(a, base_url="https://test") as client,
    ):
        client.cookies.set("scribe_admin", _admin_cookie(signer))
        resp = client.get("/api/gpu/topology")
    assert resp.status_code == 200
    assert "timed out" in resp.json()["error"]
