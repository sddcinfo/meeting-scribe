"""Tests for the HTTP captive sub-app on port 80.

Captive_http_app behavior:
  * captive probe routes return their platform responses on GET
  * GET / serves an inline sign-in page (guest PIN form + admin link)
  * POST /captive/guest-pin validates the PIN inside the CNA sandbox
    and drops the caller IP into the ``ms-allowed-guests`` ipset
  * GET/HEAD catch-all 308 to canonical HTTPS, preserving query verbatim
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from meeting_scribe.hotspot.captive_http_app import build_captive_http_app


@pytest.fixture
def http_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("SCRIBE_CANONICAL_HOST", "10.42.0.1")
    return TestClient(build_captive_http_app(), base_url="http://test")


def test_get_anything_redirects_to_canonical(http_client: TestClient) -> None:
    """Operator-typed paths 308 to canonical HTTPS with the path."""
    resp = http_client.get("/some/path", follow_redirects=False)
    assert resp.status_code == 308
    assert resp.headers["location"] == "https://10.42.0.1/some/path"


def test_get_root_serves_signin_page(http_client: TestClient) -> None:
    """Root serves the inline sign-in page: guest PIN form posts to
    the port-80 bridge; admin sign-in is a vanilla ``<a>`` to the
    canonical HTTPS ``/auth``. No setup wording, no wizard chrome —
    those references were the source of operator confusion in
    Phase H hardware testing.
    """
    resp = http_client.get("/", follow_redirects=False)
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    body = resp.text
    # New flavor: sign-in surface, not the legacy "Open setup" handoff.
    assert "<title>Meeting Scribe — Sign in</title>" in body
    assert "Sign in to continue" in body
    # Guest path lives entirely inside the captive sub-app on port 80.
    assert 'action="/captive/guest-pin"' in body
    assert 'name="pin"' in body
    # Admin path links out to canonical HTTPS sign-in.
    assert 'href="https://10.42.0.1/auth"' in body
    # No "setup" / wizard chrome on this page.
    assert "Open setup" not in body
    assert "First-time setup" not in body
    assert "step-indicator" not in body
    # No /api/setup/* call, no cookie set.
    assert "/api/setup/" not in body
    assert "set-cookie" not in {h.lower() for h in resp.headers}


def test_post_guest_pin_valid_adds_to_ipset(http_client: TestClient) -> None:
    """A correct PIN: 200, success page rendered, caller IP added to
    the guest ipset. No cookie set on this port-80 response (CNA
    sandbox isolates it; the canonical HTTPS guard mints one inline
    on the IP's first HTTPS hit). Success page redirects into the
    live meeting view at the mDNS origin."""
    with (
        patch(
            "meeting_scribe.setup_state.verify_guest_pin",
            return_value=True,
        ),
        patch("meeting_scribe.routes.guest_auth._admin_secret_store") as mock_store,
        patch("meeting_scribe.server_support.firewall_allowlist.add_guest") as mock_add_guest,
    ):
        mock_store.return_value.secret = b"x" * 32
        resp = http_client.post(
            "/captive/guest-pin",
            data={"pin": "1234"},
            follow_redirects=False,
        )
    assert resp.status_code == 200
    body = resp.text
    # iOS CNA pattern-matches ``<title>Success</title>`` to mark the
    # captive flow complete — that's what makes the Done button (blue
    # tick) appear in the top-right. Self-signed cert means we can't
    # auto-redirect into the live meeting view inside the CNA WebView
    # (cert validation is strict + no click-through), so we explicitly
    # leave the user on this page with a ``target="_blank"`` link to
    # the meeting URL (launches Safari, which CAN click through).
    assert "<title>Success</title>" in body
    # Visible "Open meeting in Safari" affordance + the URL spelled out
    # for users who prefer to type it.
    assert "Open meeting in Safari" in body
    assert 'target="_blank"' in body
    # mDNS origin appears in both the link and the typed-fallback hint.
    assert "https://" in body  # mDNS-resolved canonical with valid cert
    # Sanity: the page should NOT auto-navigate (the old behavior tried
    # to redirect into the meeting and dead-ended on cert validation).
    assert "window.location.replace" not in body
    assert mock_add_guest.await_count == 1
    # No Set-Cookie on this port-80 response — CNA sandbox isolation
    # makes cookies useless here. The HTTPS guard mints one inline.
    assert "set-cookie" not in {h.lower() for h in resp.headers}


def test_post_guest_pin_invalid_renders_error(http_client: TestClient) -> None:
    """Wrong PIN: 401 + re-rendered form with an inline error message.
    ipset is not touched."""
    with (
        patch(
            "meeting_scribe.setup_state.verify_guest_pin",
            return_value=False,
        ),
        patch("meeting_scribe.routes.guest_auth._admin_secret_store") as mock_store,
        patch("meeting_scribe.server_support.firewall_allowlist.add_guest") as mock_add_guest,
    ):
        mock_store.return_value.secret = b"x" * 32
        resp = http_client.post(
            "/captive/guest-pin",
            data={"pin": "9999"},
            follow_redirects=False,
        )
    assert resp.status_code == 401
    body = resp.text
    assert "Incorrect PIN" in body
    # Form is still present so user can retry.
    assert 'action="/captive/guest-pin"' in body
    assert mock_add_guest.await_count == 0


def test_post_guest_pin_ipset_failure_swallowed(http_client: TestClient) -> None:
    """If the ipset add raises (binary missing, sudo denied), the
    handler still returns 200 success — the PIN was correct and the
    GC tick reconciles missed entries. Same swallow pattern as the
    HTTPS bridge."""
    with (
        patch(
            "meeting_scribe.setup_state.verify_guest_pin",
            return_value=True,
        ),
        patch("meeting_scribe.routes.guest_auth._admin_secret_store") as mock_store,
        patch(
            "meeting_scribe.server_support.firewall_allowlist.add_guest",
            side_effect=RuntimeError("ipset missing"),
        ),
    ):
        mock_store.return_value.secret = b"x" * 32
        resp = http_client.post(
            "/captive/guest-pin",
            data={"pin": "1234"},
            follow_redirects=False,
        )
    assert resp.status_code == 200
    # Success page renders even when the ipset add failed silently —
    # the swallow keeps the PIN flow working when ipset is unavailable.
    assert "<title>Success</title>" in resp.text
    assert "Open meeting in Safari" in resp.text


def test_get_query_string_preserved_verbatim(http_client: TestClient) -> None:
    """Query string survives the 308 redirect intact (regression for
    multi-pair queries that previously got dropped)."""
    resp = http_client.get(
        "/auth?next=/foo&token=abc%20def",
        follow_redirects=False,
    )
    assert resp.status_code == 308
    assert resp.headers["location"] == ("https://10.42.0.1/auth?next=/foo&token=abc%20def")


def test_post_anywhere_returns_405_or_308(http_client: TestClient) -> None:
    """Non-GET/HEAD methods on the captive sub-app are not accepted —
    framework returns 405 (registered routes) or 308 (catch-all). Either
    is fine: the captive surface is read-only, the wizard runs on the
    canonical TLS app at https://meeting-scribe-${id4}.local."""
    for method in ("POST", "PUT", "DELETE", "PATCH"):
        resp = http_client.request(method, "/api/admin/authorize")
        assert resp.status_code in (308, 405), f"{method} got {resp.status_code}"


def test_head_redirects_like_get(http_client: TestClient) -> None:
    """HEAD also receives the canonical 308 (no body, but the redirect
    target lets compliant browsers/CLI tools follow up over HTTPS)."""
    resp = http_client.head("/some/path", follow_redirects=False)
    assert resp.status_code == 308
    assert resp.headers["location"] == "https://10.42.0.1/some/path"


def test_apple_captive_probe_serves(http_client: TestClient) -> None:
    """The Apple captive probe is reachable on the HTTP sub-app — not
    swallowed by the catch-all 308."""
    resp = http_client.get("/hotspot-detect.html", follow_redirects=False)
    # Unacknowledged → 302 to portal; acknowledged → 200 with Success body.
    # Either is a non-308 result that proves the probe ran.
    assert resp.status_code in (200, 302), resp.status_code


def test_rfc8910_captive_api_serves(http_client: TestClient) -> None:
    resp = http_client.get("/api/captive", follow_redirects=False)
    # Unacknowledged → 200 + {"captive": true}; we just check the route
    # ran and didn't fall through to the 308 catch-all.
    assert resp.status_code == 200
    body = resp.json()
    assert "captive" in body


def test_admin_path_post_not_routed(http_client: TestClient) -> None:
    """The captive sub-app must NOT carry the main app's admin routes —
    a POST to /api/admin/authorize never reaches a real auth handler.
    Framework returns 405 (registered route, wrong method) or 308
    (catch-all redirect)."""
    resp = http_client.post(
        "/api/admin/authorize",
        data={"secret": "hunter2"},
    )
    assert resp.status_code in (308, 405)


def test_unknown_get_path_redirects(http_client: TestClient) -> None:
    """Random GET path nobody registered → 308 to canonical, no 404."""
    resp = http_client.get(
        "/garbage/path/with/many/segments?x=1&y=2",
        follow_redirects=False,
    )
    assert resp.status_code == 308
    assert resp.headers["location"].startswith("https://10.42.0.1/garbage/")
    assert "?x=1&y=2" in resp.headers["location"]


def test_canonical_host_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """The canonical host can be overridden via SCRIBE_CANONICAL_HOST so
    deploys / tests can repoint without rebuilding the binary."""
    monkeypatch.setenv("SCRIBE_CANONICAL_HOST", "demo.local")
    client = TestClient(build_captive_http_app(), base_url="http://test")
    resp = client.get("/whatever", follow_redirects=False)
    assert resp.headers["location"] == "https://demo.local/whatever"
