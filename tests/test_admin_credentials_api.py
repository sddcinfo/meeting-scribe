"""Tests for the split admin-credentials endpoints:

* ``GET /api/admin/admin-password`` — reusable admin login password,
  fetched only on explicit Reveal click. Returns ``Cache-Control:
  no-store, private``.
* ``GET /api/admin/guest-pin`` — 4-digit guest PIN (already public via
  the Admin hotspot SSID suffix). Fetched on Credentials-tab activation.

Both endpoints sit behind the admin-cookie guard. The split exists so
the reusable password never leaks into the UI's JS layer through
routine tab navigation; only the PIN hydrates the pane by default.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient


def _build_app(*, admin_ok: bool) -> FastAPI:
    """FastAPI app with the admin router registered and the guard stubbed.

    Mirrors ``test_admin_audio_api._build_app``: the gate is monkey-patched
    on the ``server_support.admin_guard`` module so the handler's lazy
    ``from meeting_scribe.server_support.admin_guard import …`` resolves
    to the patched callable at call time.
    """
    from meeting_scribe.routes import admin as admin_mod
    from meeting_scribe.server_support import admin_guard as guard_mod

    if admin_ok:
        guard_mod._require_admin_response = lambda req: None
    else:
        guard_mod._require_admin_response = lambda req: JSONResponse(
            {"error": "admin_session_required"}, status_code=401
        )
    app = FastAPI()
    app.include_router(admin_mod.router)
    return app


@pytest.fixture
def fake_pin(monkeypatch):
    """Pin the appliance-pin derivation to a deterministic value so the
    test doesn't depend on the dev box's actual appliance id."""
    monkeypatch.setattr("meeting_scribe.cli._common.appliance_pin", lambda: "4242")
    return "4242"


def test_admin_password_requires_admin(fake_pin) -> None:
    app = _build_app(admin_ok=False)
    with TestClient(app, base_url="http://test") as client:
        resp = client.get("/api/admin/admin-password")
    assert resp.status_code == 401
    assert resp.json() == {"error": "admin_session_required"}


def test_admin_password_returns_minted_value(fake_pin) -> None:
    app = _build_app(admin_ok=True)
    with TestClient(app, base_url="http://test") as client:
        resp = client.get("/api/admin/admin-password")
    assert resp.status_code == 200
    assert resp.json() == {"password": f"DellMeetingAdmin{fake_pin}"}


def test_admin_password_cache_control_no_store(fake_pin) -> None:
    """The reusable password must NOT sit in any intermediate cache —
    proxies, browser disk cache, service workers. ``no-store, private``
    is the belt-and-braces header set."""
    app = _build_app(admin_ok=True)
    with TestClient(app, base_url="http://test") as client:
        resp = client.get("/api/admin/admin-password")
    assert resp.status_code == 200
    assert resp.headers.get("Cache-Control") == "no-store, private"
    assert resp.headers.get("Pragma") == "no-cache"


def test_guest_pin_requires_admin(fake_pin) -> None:
    app = _build_app(admin_ok=False)
    with TestClient(app, base_url="http://test") as client:
        resp = client.get("/api/admin/guest-pin")
    assert resp.status_code == 401


def test_guest_pin_returns_minted_value(fake_pin) -> None:
    app = _build_app(admin_ok=True)
    with TestClient(app, base_url="http://test") as client:
        resp = client.get("/api/admin/guest-pin")
    assert resp.status_code == 200
    assert resp.json() == {"pin": fake_pin}


def test_endpoints_are_separate(fake_pin) -> None:
    """No combined /api/admin/credentials endpoint exists — the lazy-fetch
    discipline depends on the split. A future refactor that reintroduces
    a combined endpoint would silently leak the password on tab activation."""
    app = _build_app(admin_ok=True)
    with TestClient(app, base_url="http://test") as client:
        resp = client.get("/api/admin/credentials")
    assert resp.status_code == 404
