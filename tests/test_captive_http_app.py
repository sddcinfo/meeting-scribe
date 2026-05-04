"""Tests for the strict HTTP captive sub-app on port 80.

Covers Plan §A.1 captive_http_app behavior:
  * captive probe routes return their platform responses on GET
  * GET/HEAD catch-all 308 to canonical HTTPS, preserving query verbatim
  * any non-GET/HEAD method on any path returns 426 Upgrade Required
    with the Upgrade header — including admin-API paths to prove they
    are NOT registered on the HTTP sub-app
"""

from __future__ import annotations

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


def test_get_root_redirects_to_canonical_root(http_client: TestClient) -> None:
    resp = http_client.get("/", follow_redirects=False)
    assert resp.status_code == 308
    assert resp.headers["location"] == "https://10.42.0.1/"


def test_get_query_string_preserved_verbatim(http_client: TestClient) -> None:
    """Query string survives the 308 redirect intact (regression for
    multi-pair queries that previously got dropped)."""
    resp = http_client.get(
        "/admin/bootstrap?next=/foo&token=abc%20def",
        follow_redirects=False,
    )
    assert resp.status_code == 308
    assert resp.headers["location"] == (
        "https://10.42.0.1/admin/bootstrap?next=/foo&token=abc%20def"
    )


def test_post_anywhere_returns_426(http_client: TestClient) -> None:
    """Plan §A.1: any non-GET/HEAD on any path returns 426 with the
    Upgrade header, regardless of whether the path is captive, admin,
    or unknown.
    """
    for method in ("POST", "PUT", "DELETE", "PATCH"):
        resp = http_client.request(method, "/api/admin/authorize")
        assert resp.status_code == 426, f"{method} should be 426"
        assert resp.headers.get("upgrade") == "TLS/1.2, HTTP/1.1"
        assert resp.headers.get("connection", "").lower() == "upgrade"
        # The body must be empty per the contract.
        assert resp.content == b""


def test_options_returns_426(http_client: TestClient) -> None:
    resp = http_client.options("/api/admin/authorize")
    assert resp.status_code == 426
    assert resp.headers.get("upgrade") == "TLS/1.2, HTTP/1.1"


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


def test_admin_path_post_426_not_routed(http_client: TestClient) -> None:
    """The captive sub-app must NOT carry the main app's admin routes —
    a POST to /api/admin/authorize sees the 426 method guard, not a
    real auth handler."""
    resp = http_client.post(
        "/api/admin/authorize",
        data={"secret": "hunter2"},
    )
    assert resp.status_code == 426


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
