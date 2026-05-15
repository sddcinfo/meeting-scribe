"""Phase H captive-gateway bridge: ``hotspot_guard`` accepts ipset
membership as a valid auth signal and mints a guest cookie inline.

Why this exists:
  * iOS / Android captive sheets (CNA) run in a sandboxed browser
    context with cookie isolation from the user's real browser. A PIN
    entered through ``POST /captive/guest-pin`` on port 80 drops the
    caller's IP into ``ms-allowed-guests`` but cannot set a cookie that
    the user's Safari/Chrome ever sees.
  * After PIN auth the captive sub-app redirects the user to the
    canonical HTTPS origin (cert-matching mDNS name). hotspot_guard
    runs on every HTTPS request — if it only checked cookies it would
    bounce the post-PIN user back to ``/auth`` even though they just
    authenticated.
  * Fix: hotspot_guard ALSO checks ipset membership. If the IP is in
    ``ms-allowed-guests`` (or ``ms-allowed-admins``) and no cookie is
    present, mint a fresh guest cookie inline so subsequent requests
    on this browser session short-circuit the ipset call.

Regression scope:
  * Cookie auth path unchanged (no ipset lookup if cookie validates).
  * The minted cookie has the same flags as the canonical guest cookie
    (HttpOnly, Secure, SameSite=Strict, path=/).
  * A request that fails BOTH cookie AND ipset still 302s to /auth.
  * Admin ipset membership does NOT mint an admin cookie (that would
    require the live AdminSecretStore signer rotation; we let the
    canonical admin sign-in handle it).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from starlette.testclient import TestClient

from meeting_scribe.middlewares import register_middlewares


def _build_guarded_app() -> FastAPI:
    """Minimal FastAPI app with the production middleware stack +
    one guest-scope route that hotspot_guard protects."""
    app = FastAPI()

    @app.get("/")
    async def root() -> HTMLResponse:
        return HTMLResponse("<html>meeting view</html>")

    @app.get("/auth")
    async def auth_page() -> HTMLResponse:
        return HTMLResponse("<html>auth gate</html>")

    register_middlewares(app)
    return app


def _force_setup_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """hotspot_guard short-circuits to /setup if setup isn't complete.
    Force the post-setup branch for these tests. Also clear the
    autouse ``SCRIBE_HOTSPOT_GUARD_BYPASS=1`` set by conftest.py — these
    tests EXIST to exercise the guard, not bypass it."""
    monkeypatch.setattr(
        "meeting_scribe.setup_state.is_setup_complete",
        lambda: True,
    )
    monkeypatch.delenv("SCRIBE_HOTSPOT_GUARD_BYPASS", raising=False)


def test_ipset_guest_membership_mints_cookie(monkeypatch: pytest.MonkeyPatch) -> None:
    """A client with no cookie but IP in ms-allowed-guests gets:
    * 200 from the guarded route (no /auth redirect)
    * Set-Cookie: ms_guest=... with HttpOnly + Secure + SameSite=Strict
    """
    _force_setup_complete(monkeypatch)
    app = _build_guarded_app()

    with (
        patch(
            "meeting_scribe.server_support.firewall_allowlist.is_admin",
            return_value=False,
        ),
        patch(
            "meeting_scribe.server_support.firewall_allowlist.is_guest",
            return_value=True,
        ),
        patch("meeting_scribe.routes.guest_auth._admin_secret_store") as mock_store,
        patch(
            "meeting_scribe.routes.guest_auth._sign_guest_cookie",
            return_value="signed.guest.cookie.value",
        ),
    ):
        mock_store.return_value.secret = b"x" * 32
        with TestClient(app, base_url="https://test") as client:
            resp = client.get("/", follow_redirects=False)

    assert resp.status_code == 200, (
        f"ipset-known guest must reach the route, got {resp.status_code} "
        f"(headers={dict(resp.headers)})"
    )
    set_cookie = resp.headers.get("set-cookie", "")
    assert "ms_guest=" in set_cookie, f"expected guest cookie minted, headers: {set_cookie}"
    assert "signed.guest.cookie.value" in set_cookie
    # Same flags as the canonical guest-pin handler. starlette
    # normalizes flag casing in test output, so compare lowercased.
    lower = set_cookie.lower()
    assert "httponly" in lower
    assert "secure" in lower
    assert "samesite=strict" in lower


def test_no_cookie_no_ipset_still_redirects_to_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A client with no cookie AND no ipset membership stays gated —
    same 302 → /auth as before Phase H."""
    _force_setup_complete(monkeypatch)
    app = _build_guarded_app()

    with (
        patch(
            "meeting_scribe.server_support.firewall_allowlist.is_admin",
            return_value=False,
        ),
        patch(
            "meeting_scribe.server_support.firewall_allowlist.is_guest",
            return_value=False,
        ),
        TestClient(app, base_url="https://test") as client,
    ):
        resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "/auth"


def test_ipset_admin_membership_does_not_mint_admin_cookie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admin ipset alone is NOT a substitute for the admin cookie —
    minting one outside the canonical /api/admin/authorize flow would
    bypass the AdminSecretStore signer rotation. The user's browser
    must already carry the admin cookie from the HTTPS sign-in path,
    which lands here unchanged. Documented as 302 → /auth.
    """
    _force_setup_complete(monkeypatch)
    app = _build_guarded_app()

    with (
        patch(
            "meeting_scribe.server_support.firewall_allowlist.is_admin",
            return_value=True,
        ),
        patch(
            "meeting_scribe.server_support.firewall_allowlist.is_guest",
            return_value=False,
        ),
        TestClient(app, base_url="https://test") as client,
    ):
        resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "/auth"
    # No guest cookie was minted either.
    assert "ms_guest" not in resp.headers.get("set-cookie", "")


def test_cookie_path_skips_ipset_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the guest cookie validates, hotspot_guard MUST NOT call into
    firewall_allowlist — the cookie is the cheap path."""
    _force_setup_complete(monkeypatch)
    app = _build_guarded_app()

    with (
        patch(
            "meeting_scribe.routes.guest_auth.verify_guest_cookie",
            return_value=True,
        ),
        patch("meeting_scribe.server_support.firewall_allowlist.is_guest") as mock_is_guest,
        patch("meeting_scribe.server_support.firewall_allowlist.is_admin") as mock_is_admin,
        TestClient(app, base_url="https://test") as client,
    ):
        resp = client.get("/", follow_redirects=False, cookies={"ms_guest": "any"})
    assert resp.status_code == 200
    mock_is_guest.assert_not_called()
    mock_is_admin.assert_not_called()
    # No fresh cookie minted on the cookie path.
    assert "ms_guest" not in resp.headers.get("set-cookie", "")
