"""Admin-mode AP landing: skip the setup/portal interstitials and land
the operator on ``/auth`` (the sign-in page) directly.

Two redirect points covered:
* ``GET /setup`` when ``setup-complete`` marker exists → 302 ``/auth``.
* ``GET /`` when ``wifi_mode=admin`` and no cookies → 302 ``/auth``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient


@pytest.fixture
def fresh_settings(tmp_path: Path, monkeypatch):
    """Pin settings_store to a per-test JSON file."""
    settings_path = tmp_path / "settings.json"
    import meeting_scribe.server_support.settings_store as store

    monkeypatch.setattr(store, "SETTINGS_OVERRIDE_FILE", settings_path)
    monkeypatch.setattr(store, "_settings_cache", None)
    monkeypatch.setattr(store, "_settings_cache_mtime", 0.0)
    monkeypatch.setattr(store, "_legacy_migration_attempted", False)
    return store


# ─── /setup once setup is complete ─────────────────────────────


def test_setup_route_redirects_to_auth_when_complete(tmp_path, monkeypatch):
    """``GET /setup`` post-completion sends operators straight to /auth."""
    monkeypatch.setenv("SCRIBE_STATE_DIR", str(tmp_path))
    # Drop the marker the route checks.
    (tmp_path / "setup-complete").write_text("1\n")

    from meeting_scribe.routes import setup as setup_routes

    app = FastAPI()
    app.include_router(setup_routes.router)
    with TestClient(app, base_url="http://test", follow_redirects=False) as client:
        resp = client.get("/setup")
    assert resp.status_code == 302
    assert resp.headers["location"] == "/auth"
    # No-store header is set so an aggressive proxy can't cache the
    # redirect and serve it forever even after a factory reset.
    assert "no-store" in resp.headers.get("cache-control", "")


def test_setup_route_renders_wizard_when_not_complete(tmp_path, monkeypatch):
    """Regression: when setup is NOT complete, the wizard still renders."""
    monkeypatch.setenv("SCRIBE_STATE_DIR", str(tmp_path))
    # No marker.

    from meeting_scribe.routes import setup as setup_routes

    app = FastAPI()
    app.include_router(setup_routes.router)
    with TestClient(app, base_url="http://test", follow_redirects=False) as client:
        resp = client.get("/setup")
    # Not a 302 — the wizard HTML is served (or a status-conflict
    # response if another claim is mid-flight). The important thing
    # is we don't bounce to /auth.
    assert resp.status_code in (200, 409)


# ─── / when wifi_mode=admin ────────────────────────────────────


def _build_views_app() -> FastAPI:
    """A minimal app with just the views router so /-routing is testable."""
    from meeting_scribe.routes import views as views_routes

    app = FastAPI()
    app.include_router(views_routes.router)
    return app


def test_index_redirects_to_auth_in_admin_mode_without_cookies(fresh_settings, monkeypatch):
    """Admin AP, no cookies → /auth. The portal interstitial is for the
    guest-pin SSID flow, not the fixed admin SSID."""
    fresh_settings._save_settings_override({"wifi_mode": "admin"})

    # Stub out the captive_ack side effect so we don't need a real
    # hotspot state file for the test.
    monkeypatch.setattr("meeting_scribe.routes.views._captive_ack", lambda req: None)
    monkeypatch.setattr("meeting_scribe.routes.views.has_admin_session", lambda req: False)

    app = _build_views_app()
    with TestClient(app, base_url="http://test", follow_redirects=False) as client:
        resp = client.get("/")
    assert resp.status_code == 302
    assert resp.headers["location"] == "/auth"


def test_index_serves_admin_page_when_admin_session_present(fresh_settings, monkeypatch):
    fresh_settings._save_settings_override({"wifi_mode": "admin"})
    monkeypatch.setattr("meeting_scribe.routes.views.has_admin_session", lambda req: True)
    monkeypatch.setattr("meeting_scribe.routes.views._HTML", {"index": "<html>admin-index</html>"})

    app = _build_views_app()
    with TestClient(app, base_url="http://test", follow_redirects=False) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert "admin-index" in resp.text


def test_index_serves_guest_when_portal_cookie_present(fresh_settings, monkeypatch):
    """A guest who already dismissed the portal lands on the guest view,
    regardless of AP mode."""
    fresh_settings._save_settings_override({"wifi_mode": "admin"})
    monkeypatch.setattr("meeting_scribe.routes.views._captive_ack", lambda req: None)
    monkeypatch.setattr("meeting_scribe.routes.views.has_admin_session", lambda req: False)
    monkeypatch.setattr("meeting_scribe.routes.views._HTML", {"guest": "<html>guest-view</html>"})

    app = _build_views_app()
    with TestClient(app, base_url="http://test", follow_redirects=False) as client:
        client.cookies.set("scribe_portal", "done")
        resp = client.get("/")
    assert resp.status_code == 200
    assert "guest-view" in resp.text


def test_index_serves_portal_when_wifi_mode_is_meeting(fresh_settings, monkeypatch):
    """In guest-flow (meeting) AP mode, the portal interstitial is the
    intended landing — preserve the existing behavior."""
    fresh_settings._save_settings_override({"wifi_mode": "meeting"})
    monkeypatch.setattr("meeting_scribe.routes.views._captive_ack", lambda req: None)
    monkeypatch.setattr("meeting_scribe.routes.views.has_admin_session", lambda req: False)
    monkeypatch.setattr("meeting_scribe.routes.views._HTML", {"portal": "<html>portal</html>"})

    app = _build_views_app()
    with TestClient(app, base_url="http://test", follow_redirects=False) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert "portal" in resp.text


def test_admin_mode_ap_active_swallows_settings_failures(monkeypatch):
    """Defensive: a settings_store hiccup must NOT crash the / route."""
    from meeting_scribe.routes import views as views_routes

    def _explode():
        raise RuntimeError("simulated settings_store failure")

    # Patch the imported loader inside views.py.
    monkeypatch.setattr(
        "meeting_scribe.server_support.settings_store._load_settings_override",
        _explode,
    )
    assert views_routes._admin_mode_ap_active() is False
