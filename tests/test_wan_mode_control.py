"""Operator control surface for WAN egress mode + migration regression.

Three surfaces:
  * CLI:  ``wifi wan mode get | set``
  * REST: ``GET/PUT /api/admin/wan/mode``
  * Migration: AP-up flips block→captive only when source=default.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient


@pytest.fixture
def fresh_settings(tmp_path: Path, monkeypatch):
    settings_path = tmp_path / "settings.json"
    import meeting_scribe.server_support.settings_store as store

    monkeypatch.setattr(store, "SETTINGS_OVERRIDE_FILE", settings_path)
    monkeypatch.setattr(store, "_settings_cache", None)
    monkeypatch.setattr(store, "_settings_cache_mtime", 0.0)
    monkeypatch.setattr(store, "_legacy_migration_attempted", False)
    return store


# ─── CLI ──────────────────────────────────────────────────────


def _invoke(*args: str):
    from meeting_scribe.cli import cli

    return CliRunner().invoke(cli, list(args), catch_exceptions=False)


def test_cli_wan_mode_get_prints_mode_and_source(fresh_settings) -> None:
    fresh_settings._set_wan_egress_mode("captive")  # source=operator
    result = _invoke("wifi", "wan", "mode", "get")
    assert result.exit_code == 0, result.output
    assert "mode:" in result.output and "captive" in result.output
    assert "source:" in result.output and "operator" in result.output


def test_cli_wan_mode_set_stamps_operator_source(fresh_settings, monkeypatch) -> None:
    async def _noop_reconcile() -> None:
        return None

    monkeypatch.setattr("meeting_scribe.wifi.reconcile_network_state", _noop_reconcile)
    result = _invoke("wifi", "wan", "mode", "set", "captive")
    assert result.exit_code == 0, result.output
    assert fresh_settings._effective_wan_egress_mode() == "captive"
    assert fresh_settings._wan_egress_mode_source() == "operator"


def test_cli_wan_mode_set_rejects_unknown_value(fresh_settings) -> None:
    result = _invoke("wifi", "wan", "mode", "set", "garbage")
    assert result.exit_code != 0
    # Click's standard rejection message.
    assert "garbage" in result.output


# ─── REST ─────────────────────────────────────────────────────


def _build_app(*, admin_ok: bool = True) -> FastAPI:
    from meeting_scribe.routes import admin_wan as admin_wan_mod

    if admin_ok:
        admin_wan_mod._require_admin_response = lambda req: None
    else:
        admin_wan_mod._require_admin_response = lambda req: JSONResponse(
            {"error": "admin required"}, status_code=403
        )
    app = FastAPI()
    app.include_router(admin_wan_mod.router)
    return app


def test_get_wan_mode_returns_mode_and_source(fresh_settings) -> None:
    fresh_settings._set_wan_egress_mode("captive")
    app = _build_app()
    with TestClient(app, base_url="http://test") as client:
        resp = client.get("/api/admin/wan/mode")
    assert resp.status_code == 200
    assert resp.json() == {"mode": "captive", "source": "operator"}


def test_get_wan_mode_admin_gated(fresh_settings) -> None:
    app = _build_app(admin_ok=False)
    with TestClient(app, base_url="http://test") as client:
        resp = client.get("/api/admin/wan/mode")
    assert resp.status_code == 403


def test_put_wan_mode_sets_and_stamps_operator(fresh_settings, monkeypatch) -> None:
    async def _noop_reconcile() -> None:
        return None

    monkeypatch.setattr("meeting_scribe.wifi.reconcile_network_state", _noop_reconcile)
    app = _build_app()
    with TestClient(app, base_url="http://test") as client:
        resp = client.put("/api/admin/wan/mode", json={"mode": "gateway"})
    assert resp.status_code == 200
    assert resp.json() == {"mode": "gateway", "source": "operator"}
    assert fresh_settings._effective_wan_egress_mode() == "gateway"
    assert fresh_settings._wan_egress_mode_source() == "operator"


def test_put_wan_mode_rejects_invalid(fresh_settings) -> None:
    app = _build_app()
    with TestClient(app, base_url="http://test") as client:
        resp = client.put("/api/admin/wan/mode", json={"mode": "garbage"})
    assert resp.status_code == 400
    assert "mode must be" in resp.json()["error"]


def test_put_wan_mode_admin_gated(fresh_settings) -> None:
    app = _build_app(admin_ok=False)
    with TestClient(app, base_url="http://test") as client:
        resp = client.put("/api/admin/wan/mode", json={"mode": "captive"})
    assert resp.status_code == 403


# ─── Migration ladder ─────────────────────────────────────────


def _run_migration_ladder(store) -> None:
    """Mimic the migration logic inside ``wifi.wifi_up`` so we don't have
    to bring up the full AP for the test."""
    if (
        store._wan_egress_mode_source() == "default"
        and store._effective_wan_egress_mode() == "block"
    ):
        store._set_wan_egress_mode("captive", source="default")


def test_migration_upgrades_block_default_to_captive(fresh_settings) -> None:
    """Legacy settings file: no source field, mode=block. AP bring-up
    flips to captive without operator action."""
    # Mode defaults to block, source defaults to default. Run ladder.
    assert fresh_settings._effective_wan_egress_mode() == "block"
    assert fresh_settings._wan_egress_mode_source() == "default"
    _run_migration_ladder(fresh_settings)
    assert fresh_settings._effective_wan_egress_mode() == "captive"
    assert fresh_settings._wan_egress_mode_source() == "default"


def test_migration_respects_operator_source(fresh_settings) -> None:
    """Hard stop: operator explicitly picked block → migration MUST NOT
    revert them to captive. Regression for Codex P0 in the plan review."""
    fresh_settings._set_wan_egress_mode("block")  # source=operator
    assert fresh_settings._wan_egress_mode_source() == "operator"
    _run_migration_ladder(fresh_settings)
    # Mode must still be block; source must still be operator.
    assert fresh_settings._effective_wan_egress_mode() == "block"
    assert fresh_settings._wan_egress_mode_source() == "operator"


def test_migration_does_not_touch_non_block_modes(fresh_settings) -> None:
    """gateway or captive (even with source=default) are not in scope
    for the block→captive ladder."""
    fresh_settings._set_wan_egress_mode("gateway", source="default")
    _run_migration_ladder(fresh_settings)
    assert fresh_settings._effective_wan_egress_mode() == "gateway"
    assert fresh_settings._wan_egress_mode_source() == "default"


def test_migration_is_idempotent_on_already_captive(fresh_settings) -> None:
    fresh_settings._set_wan_egress_mode("captive", source="default")
    _run_migration_ladder(fresh_settings)
    assert fresh_settings._effective_wan_egress_mode() == "captive"
    assert fresh_settings._wan_egress_mode_source() == "default"
