"""Tests for the new security middlewares: host canonicalization, origin
allowlist (CSRF), CSP injector, and cache-headers.

The middlewares are config-driven via env vars so existing tests that hit
``127.0.0.1`` keep working when the env vars are unset. These tests set the
env vars per-case via ``monkeypatch`` and verify the on-the-wire behavior
through Starlette's TestClient.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.testclient import TestClient

from meeting_scribe.middlewares import (
    _CSP_HEADER_VALUE,
    _origin_allowed,
    register_middlewares,
)


def _build_app() -> FastAPI:
    """Minimal FastAPI app with the production middleware stack."""
    app = FastAPI()

    @app.get("/")
    async def root() -> HTMLResponse:
        return HTMLResponse("<html><body>ok</body></html>")

    @app.get("/static/foo.css")
    async def static_css() -> HTMLResponse:
        # Returns text/html to make the cache-headers + CSP exemption check
        # explicit — CSP applies to text/html, the static path is /static/.
        resp = HTMLResponse("body{}")
        resp.headers["content-type"] = "text/css"
        return resp

    @app.get("/admin/bootstrap")
    async def bootstrap() -> HTMLResponse:
        return HTMLResponse("<html>form</html>")

    # Both routes use sign-in-surface paths so the hotspot guard's
    # always-allowed list lets unauthenticated test clients through; we
    # are isolating the origin/cache/CSP middlewares, not the hotspot guard.
    @app.get("/api/admin/authorize")
    async def admin_get() -> JSONResponse:
        return JSONResponse({"ok": True})

    @app.post("/api/admin/authorize")
    async def admin_post() -> JSONResponse:
        return JSONResponse({"ok": True})

    @app.post("/api/meeting/create")
    async def meeting_post() -> JSONResponse:
        return JSONResponse({"ok": True})

    @app.get("/api/status")
    async def status() -> JSONResponse:
        return JSONResponse({"status": "live"})

    register_middlewares(app)
    return app


# ── Host canonicalization ─────────────────────────────────────────────────


def test_host_canon_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """No env var set → canonicalization no-ops; loopback Host succeeds."""
    monkeypatch.delenv("SCRIBE_CANONICAL_HOST", raising=False)
    app = _build_app()
    with TestClient(app, base_url="https://127.0.0.1") as client:
        resp = client.get("/api/status")
    assert resp.status_code == 200


def test_host_canon_redirects_non_canonical(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-canonical Host on HTTPS → 308 to canonical; query preserved."""
    monkeypatch.setenv("SCRIBE_CANONICAL_HOST", "10.42.0.1")
    app = _build_app()
    with TestClient(app, base_url="https://example.invalid") as client:
        resp = client.get("/api/status?foo=bar&baz=qux", follow_redirects=False)
    assert resp.status_code == 308
    assert resp.headers["location"] == "https://10.42.0.1/api/status?foo=bar&baz=qux"


def test_host_canon_canonical_host_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Canonical Host header → handler runs; no redirect."""
    monkeypatch.setenv("SCRIBE_CANONICAL_HOST", "10.42.0.1")
    app = _build_app()
    with TestClient(app, base_url="https://10.42.0.1") as client:
        resp = client.get("/api/status", follow_redirects=False)
    assert resp.status_code == 200


def test_host_canon_strips_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """``Host: 10.42.0.1:443`` → matches canonical, no redirect."""
    monkeypatch.setenv("SCRIBE_CANONICAL_HOST", "10.42.0.1")
    app = _build_app()
    with TestClient(app, base_url="https://10.42.0.1:443") as client:
        resp = client.get("/api/status", follow_redirects=False)
    assert resp.status_code == 200


# ── Origin allowlist (CSRF) ───────────────────────────────────────────────


def test_origin_allowlist_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """No env var set → admin POST without Origin succeeds."""
    monkeypatch.delenv("SCRIBE_CANONICAL_ORIGINS", raising=False)
    app = _build_app()
    with TestClient(app, base_url="http://test") as client:
        resp = client.post("/api/admin/authorize")
    assert resp.status_code == 200


def test_origin_allowlist_blocks_missing_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allowlist set + admin POST with no Origin → 403."""
    monkeypatch.setenv("SCRIBE_CANONICAL_ORIGINS", "https://10.42.0.1")
    app = _build_app()
    with TestClient(app, base_url="http://test") as client:
        resp = client.post("/api/admin/authorize")
    assert resp.status_code == 403
    assert resp.json()["error"] == "csrf_origin_disallowed"


def test_origin_allowlist_blocks_disallowed_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allowlist set + admin POST with wrong Origin → 403."""
    monkeypatch.setenv("SCRIBE_CANONICAL_ORIGINS", "https://10.42.0.1")
    app = _build_app()
    with TestClient(app, base_url="http://test") as client:
        resp = client.post(
            "/api/admin/authorize",
            headers={"Origin": "https://attacker.com"},
        )
    assert resp.status_code == 403


def test_origin_allowlist_allows_canonical(monkeypatch: pytest.MonkeyPatch) -> None:
    """Canonical Origin allows admin POST."""
    monkeypatch.setenv("SCRIBE_CANONICAL_ORIGINS", "https://10.42.0.1")
    app = _build_app()
    with TestClient(app, base_url="http://test") as client:
        resp = client.post(
            "/api/admin/authorize",
            headers={"Origin": "https://10.42.0.1"},
        )
    assert resp.status_code == 200


def test_origin_allowlist_get_exempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """GET requests on guarded paths bypass Origin enforcement."""
    monkeypatch.setenv("SCRIBE_CANONICAL_ORIGINS", "https://10.42.0.1")
    app = _build_app()
    with TestClient(app, base_url="http://test") as client:
        resp = client.get("/api/admin/authorize")
    assert resp.status_code == 200


def test_origin_allowlist_localhost_not_implicitly_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loopback origins are NOT implicitly allowlisted — local apps cannot
    ride the admin cookie via cross-origin fetch.
    """
    monkeypatch.setenv("SCRIBE_CANONICAL_ORIGINS", "https://10.42.0.1")
    app = _build_app()
    for hostile in ("https://localhost", "https://127.0.0.1"):
        with TestClient(app, base_url="http://test") as client:
            resp = client.post(
                "/api/admin/authorize",
                headers={"Origin": hostile},
            )
        assert resp.status_code == 403, hostile


def test_origin_allowed_helper_exact_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_origin_allowed` exact-matches scheme + host + port."""
    monkeypatch.setenv("SCRIBE_CANONICAL_ORIGINS", "https://10.42.0.1,https://10.42.0.1:443")
    assert _origin_allowed("https://10.42.0.1") is True
    assert _origin_allowed("https://10.42.0.1:443") is True
    # Default-port form vs explicit-port form: the spec wants exact match.
    # Both forms are listed separately in the env so both pass; an unlisted
    # port is rejected.
    assert _origin_allowed("https://10.42.0.1:8443") is False
    assert _origin_allowed("http://10.42.0.1") is False
    assert _origin_allowed("https://attacker.com") is False
    assert _origin_allowed(None) is False
    assert _origin_allowed("") is False


def test_origin_allowlist_meeting_post_guarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """`/api/meeting/*` is included in the guarded prefix list."""
    monkeypatch.setenv("SCRIBE_CANONICAL_ORIGINS", "https://10.42.0.1")
    app = _build_app()
    with TestClient(app, base_url="http://test") as client:
        resp = client.post("/api/meeting/create")
    assert resp.status_code == 403


# ── CSP injector ──────────────────────────────────────────────────────────


def test_csp_applied_to_html(monkeypatch: pytest.MonkeyPatch) -> None:
    """text/html responses carry the strict CSP header."""
    monkeypatch.delenv("SCRIBE_CANONICAL_HOST", raising=False)
    app = _build_app()
    with TestClient(app, base_url="http://test") as client:
        resp = client.get("/")
    assert resp.headers.get("content-security-policy") == _CSP_HEADER_VALUE


def test_csp_skipped_for_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-HTML responses (JSON, etc.) do NOT receive the CSP header."""
    monkeypatch.delenv("SCRIBE_CANONICAL_HOST", raising=False)
    app = _build_app()
    with TestClient(app, base_url="http://test") as client:
        resp = client.get("/api/status")
    assert "content-security-policy" not in {k.lower() for k in resp.headers.keys()}


def test_csp_includes_required_directives() -> None:
    """The CSP value lists the directives this v1.0 surface depends on."""
    for required in (
        "default-src 'self'",
        "script-src 'self'",
        "media-src 'self' blob:",
        "worker-src 'self'",
        "connect-src 'self' wss://10.42.0.1",
        "object-src 'none'",
        "frame-ancestors 'none'",
        "require-trusted-types-for 'script'",
    ):
        assert required in _CSP_HEADER_VALUE, required


# ── Cache headers ─────────────────────────────────────────────────────────


def test_cache_no_store_on_root() -> None:
    app = _build_app()
    with TestClient(app, base_url="http://test") as client:
        resp = client.get("/")
    assert resp.headers["cache-control"] == "no-store, private"
    assert resp.headers["vary"] == "Cookie"


def test_cache_no_store_on_admin_bootstrap() -> None:
    app = _build_app()
    with TestClient(app, base_url="http://test") as client:
        resp = client.get("/admin/bootstrap")
    assert resp.headers["cache-control"] == "no-store, private"


def test_cache_no_store_on_admin_api_prefix() -> None:
    app = _build_app()
    with TestClient(app, base_url="http://test") as client:
        resp = client.get("/api/admin/authorize")
    assert resp.headers["cache-control"] == "no-store, private"


def test_cache_static_not_no_store() -> None:
    """Static asset paths do NOT get the auth-sensitive cache headers."""
    app = _build_app()
    with TestClient(app, base_url="http://test") as client:
        resp = client.get("/static/foo.css")
    cc = resp.headers.get("cache-control", "")
    assert "no-store" not in cc, cc


def test_cache_status_endpoint_not_no_store() -> None:
    """Random JSON endpoints don't carry no-store / Vary: Cookie."""
    app = _build_app()
    with TestClient(app, base_url="http://test") as client:
        resp = client.get("/api/status")
    cc = resp.headers.get("cache-control", "")
    assert "no-store" not in cc
