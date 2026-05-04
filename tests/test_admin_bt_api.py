"""Tests for routes/admin_bt.py — admin gating, request validation,
JSON shape. Subprocess-mocked so the routes can run end-to-end without
real BlueZ / pactl on the test box.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient


def _build_app(*, admin_ok: bool) -> FastAPI:
    """FastAPI app with admin_bt routes registered.

    ``admin_ok`` toggles the gate so we can exercise the 401 / 403
    short-circuit without setting cookies. Patches
    ``_require_admin_response`` to return None (admin) or a JSONResponse
    (denied).
    """
    from meeting_scribe.routes import admin_bt as admin_bt_mod

    if admin_ok:
        admin_bt_mod._require_admin_response = lambda req: None
    else:
        admin_bt_mod._require_admin_response = lambda req: JSONResponse(
            {"error": "admin required"}, status_code=403
        )
    app = FastAPI()
    app.include_router(admin_bt_mod.router)
    return app


def test_status_requires_admin() -> None:
    app = _build_app(admin_ok=False)
    with TestClient(app, base_url="http://test") as client:
        resp = client.get("/api/admin/bt/status")
    assert resp.status_code == 403


def test_status_returns_snapshot() -> None:
    """``GET /api/admin/bt/status`` runs ``bt_status_sync`` and returns
    the JSON snapshot."""
    app = _build_app(admin_ok=True)
    with patch(
        "meeting_scribe.routes.admin_bt.bt_mod.bt_status_sync",
        new=AsyncMock(return_value={"powered": True, "devices": []}),
    ):
        with TestClient(app, base_url="http://test") as client:
            resp = client.get("/api/admin/bt/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["powered"] is True
    assert body["devices"] == []


def test_connect_requires_mac() -> None:
    app = _build_app(admin_ok=True)
    with TestClient(app, base_url="http://test") as client:
        resp = client.post("/api/admin/bt/connect", json={})
    assert resp.status_code == 400
    assert "mac required" in resp.json()["error"]


def test_connect_invokes_bt_connect() -> None:
    app = _build_app(admin_ok=True)
    with patch(
        "meeting_scribe.routes.admin_bt.bt_mod.bt_connect",
        new=AsyncMock(return_value=None),
    ) as mock:
        with TestClient(app, base_url="http://test") as client:
            resp = client.post(
                "/api/admin/bt/connect",
                json={"mac": "AA:BB:CC:DD:EE:FF"},
            )
    assert resp.status_code == 200
    assert mock.await_args is not None
    assert mock.await_args.args == ("AA:BB:CC:DD:EE:FF",)


def test_disconnect_accepts_no_mac() -> None:
    app = _build_app(admin_ok=True)
    with patch(
        "meeting_scribe.routes.admin_bt.bt_mod.bt_disconnect",
        new=AsyncMock(return_value=None),
    ):
        with TestClient(app, base_url="http://test") as client:
            resp = client.post("/api/admin/bt/disconnect", json={})
    assert resp.status_code == 200


def test_pair_surfaces_bluetooth_error() -> None:
    """When bt_pair raises (passkey-required, etc.), the route
    surfaces a 502 with the error string so the UI can show the CLI
    fallback."""
    from meeting_scribe.bt import BluetoothError

    app = _build_app(admin_ok=True)
    with patch(
        "meeting_scribe.routes.admin_bt.bt_mod.bt_pair",
        new=AsyncMock(side_effect=BluetoothError("passkey required")),
    ):
        with TestClient(app, base_url="http://test") as client:
            resp = client.post(
                "/api/admin/bt/pair",
                json={"mac": "AA:BB:CC:DD:EE:FF"},
            )
    assert resp.status_code == 502
    assert "passkey required" in resp.json()["error"]


def test_mic_persists_setting(tmp_path, monkeypatch) -> None:
    """``POST /api/admin/bt/mic`` writes the toggle to the settings
    store. With the bridge inactive the persistence is the only side
    effect; the live state machine consumes it on next start."""
    from meeting_scribe.bt import SETTINGS_BT_INPUT_ACTIVE

    captured: dict[str, Any] = {}

    def fake_save(overrides: dict[str, Any]) -> None:
        captured.update(overrides)

    monkeypatch.setattr(
        "meeting_scribe.server_support.settings_store._save_settings_override",
        fake_save,
    )
    app = _build_app(admin_ok=True)
    with TestClient(app, base_url="http://test") as client:
        resp = client.post("/api/admin/bt/mic", json={"enabled": True})
    assert resp.status_code == 200
    assert captured.get(SETTINGS_BT_INPUT_ACTIVE) is True


def test_mic_rejects_non_bool_enabled() -> None:
    app = _build_app(admin_ok=True)
    with TestClient(app, base_url="http://test") as client:
        resp = client.post("/api/admin/bt/mic", json={"enabled": "yes"})
    assert resp.status_code == 400
    assert "enabled bool required" in resp.json()["error"]
