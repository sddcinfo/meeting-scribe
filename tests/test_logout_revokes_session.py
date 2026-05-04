"""Tests for /api/admin/logout — revoke session_id + close admin WS +
delete cookie. Plan §Tests test_logout_revokes_session.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.testclient import TestClient

from meeting_scribe.runtime import state
from meeting_scribe.terminal.auth import (
    AdminSecretStore,
    COOKIE_NAME,
    CookieSigner,
    decode_verified_cookie,
)
from meeting_scribe.terminal.bootstrap import (
    BootstrapConfig,
    _close_admin_ws_for_session,
    register_bootstrap_routes,
)


@pytest.fixture(autouse=True)
def _clean_state() -> None:
    """Per-test isolation: state is process-global, the auth fixtures
    here mutate it. Snapshot + restore."""
    saved_revoked = dict(state._revoked_sessions)
    saved_ws = {k: set(v) for k, v in state._admin_ws_by_session.items()}
    state._revoked_sessions.clear()
    state._admin_ws_by_session.clear()
    yield
    state._revoked_sessions.clear()
    state._revoked_sessions.update(saved_revoked)
    state._admin_ws_by_session.clear()
    state._admin_ws_by_session.update(saved_ws)


def _build_app(tmp_path) -> tuple[FastAPI, BootstrapConfig]:
    secret_store = AdminSecretStore.load_or_create(tmp_path / "admin-secret")
    signer = CookieSigner(
        secret=secret_store.secret,
        is_revoked=lambda sid: sid in state._revoked_sessions,
    )
    cfg = BootstrapConfig(
        admin_secret=secret_store,
        cookie_signer=signer,
        is_guest_scope=lambda req: False,
    )
    app = FastAPI()
    register_bootstrap_routes(app, cfg)
    return app, cfg


def test_logout_endpoint_revokes_session(tmp_path) -> None:
    app, cfg = _build_app(tmp_path)
    secret = cfg.admin_secret.secret.decode()
    with TestClient(app, base_url="https://test") as client:
        # Sign in: the helper drops the cookie on the client.
        r = client.post("/api/admin/authorize", json={"secret": secret})
        assert r.status_code == 200
        cookie_value = client.cookies.get(COOKIE_NAME)
        assert cookie_value
        ok, sid, _ = decode_verified_cookie(cfg.cookie_signer, cookie_value)
        assert ok
        assert sid is not None

        # Verify cookie is currently valid.
        assert cfg.cookie_signer.verify(cookie_value)

        # Logout — revoke session + delete cookie.
        r = client.post("/api/admin/logout")
        assert r.status_code == 200
        assert r.headers["cache-control"] == "no-store, private"
        # Session_id is now in the revocation set.
        assert sid in state._revoked_sessions
        # Verify rejects the previously-valid cookie.
        assert cfg.cookie_signer.verify(cookie_value) is False


def test_logout_idempotent_without_cookie(tmp_path) -> None:
    """Logout works when the request has no cookie — returns 200 OK,
    no error path leaks to unauthenticated callers."""
    app, _ = _build_app(tmp_path)
    with TestClient(app, base_url="https://test") as client:
        r = client.post("/api/admin/logout")
    assert r.status_code == 200
    assert state._revoked_sessions == {}


def test_logout_idempotent_with_garbage_cookie(tmp_path) -> None:
    app, _ = _build_app(tmp_path)
    with TestClient(app, base_url="https://test") as client:
        client.cookies.set(COOKIE_NAME, "totally.bogus.cookie")
        r = client.post("/api/admin/logout")
    assert r.status_code == 200


def test_deauthorize_alias_invokes_same_path(tmp_path) -> None:
    """/api/admin/deauthorize is a deprecated alias for /api/admin/logout
    — same revocation behavior."""
    app, cfg = _build_app(tmp_path)
    secret = cfg.admin_secret.secret.decode()
    with TestClient(app, base_url="https://test") as client:
        client.post("/api/admin/authorize", json={"secret": secret})
        cookie_value = client.cookies.get(COOKIE_NAME)
        ok, sid, _ = decode_verified_cookie(cfg.cookie_signer, cookie_value)
        r = client.post("/api/admin/deauthorize")
    assert r.status_code == 200
    assert ok and sid is not None and sid in state._revoked_sessions


def test_reauth_revokes_prior_session(tmp_path) -> None:
    """Plan R40: /api/admin/authorize from a request that already
    carries a valid admin cookie revokes the prior session_id before
    minting the new cookie."""
    app, cfg = _build_app(tmp_path)
    secret = cfg.admin_secret.secret.decode()
    with TestClient(app, base_url="https://test") as client:
        # First auth.
        r1 = client.post("/api/admin/authorize", json={"secret": secret})
        assert r1.status_code == 200
        first_cookie = client.cookies.get(COOKIE_NAME)
        ok, first_sid, _ = decode_verified_cookie(cfg.cookie_signer, first_cookie)
        assert ok and first_sid is not None

        # Re-auth from the same browser → prior session revoked.
        r2 = client.post("/api/admin/authorize", json={"secret": secret})
        assert r2.status_code == 200
        second_cookie = client.cookies.get(COOKIE_NAME)
        ok2, second_sid, _ = decode_verified_cookie(cfg.cookie_signer, second_cookie)

    assert second_sid is not None
    assert first_sid != second_sid
    assert first_sid in state._revoked_sessions
    # New cookie is valid.
    assert second_sid not in state._revoked_sessions


def test_close_admin_ws_for_session_handles_empty_bucket() -> None:
    """No registered WS for the session → no-op, no error."""
    state._admin_ws_by_session.pop("absent", None)
    # Should not raise.
    _close_admin_ws_for_session("absent")


def test_close_admin_ws_for_session_closes_each(tmp_path) -> None:
    """Registered WS objects are popped + told to close."""

    closed: list[tuple[int, str]] = []

    class FakeWS:
        async def close(self, code: int, reason: str) -> None:
            closed.append((code, reason))

    async def scenario() -> None:
        state._admin_ws_by_session["sid-z"] = {FakeWS(), FakeWS()}
        _close_admin_ws_for_session("sid-z")
        # Give the scheduled tasks a chance to run.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert len(closed) == 2
    assert all(code == 1008 and reason == "revoked" for code, reason in closed)
    assert "sid-z" not in state._admin_ws_by_session


def test_authorize_uses_strict_samesite(tmp_path) -> None:
    """Plan §A.4 — cookie attrs upgraded to SameSite=Strict."""
    app, cfg = _build_app(tmp_path)
    secret = cfg.admin_secret.secret.decode()
    with TestClient(app, base_url="https://test") as client:
        r = client.post("/api/admin/authorize", json={"secret": secret})
    set_cookie = r.headers.get("set-cookie", "")
    assert "scribe_admin=" in set_cookie
    # Starlette emits the attribute lowercase or capitalized depending
    # on version; match case-insensitively.
    assert "samesite=strict" in set_cookie.lower()
    assert "secure" in set_cookie.lower()
    assert "httponly" in set_cookie.lower()
    assert "path=/" in set_cookie.lower()
