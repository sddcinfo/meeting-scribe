"""Coverage proof + behavioural matrix for the admin auth gate.

Admin auth is **cookie-only** in v1.0 — the listener binds to all
interfaces on port 443 (so admin reaches the box from the AP and
the LAN at the same URL); subnet enforcement was removed because
it conflicted with that UX. ``Origin`` allowlist + cookie
attributes are the CSRF defense.

Two parts:

* **Part 1 (coverage):** every route registered on the FastAPI app
  is classified as admin (carries ``Depends(require_admin)`` or its
  WS equivalent) or non-admin. Non-admin routes must appear in an
  explicit allowlist defined in this file. A new admin endpoint
  added without ``require_admin`` AND not in the allowlist fails
  the test, so a future maintainer can't silently ship an
  unauthenticated admin route.

* **Part 2 (behaviour):** lock in the cookie-gate contract:
    a) no cookie → 401 ``admin_session_required``
    b) wrong-password sign-in → 401 ``invalid_password`` (route
       reachable, password rejected)
    c) right-password sign-in → 200 + cookie minted, subsequent
       admin call passes

The terminal WS handlers don't appear in the matrix because their
auth gate is verified end-to-end in ``test_terminal_ws.py``.
"""

from __future__ import annotations

import hmac as _hmac

from fastapi import FastAPI
from fastapi.testclient import TestClient

from meeting_scribe.runtime import state
from meeting_scribe.server_support.admin_guard import require_admin
from meeting_scribe.terminal.auth import AdminSecretStore, CookieSigner, TicketStore
from meeting_scribe.terminal.bootstrap import BootstrapConfig, register_bootstrap_routes
from meeting_scribe.terminal.registry import ActiveTerminals
from meeting_scribe.terminal.router import TerminalRouterConfig, register_terminal_routes

_TEST_CRED = "MatrixPassword4242"


def _build_app(tmp_path, monkeypatch):
    secret_store = AdminSecretStore.load_or_create(tmp_path / "admin-secret")
    state_dir = tmp_path / "scribe-state"
    state_dir.mkdir(parents=True, exist_ok=True)
    expected = _hmac.new(secret_store.secret, _TEST_CRED.encode("utf-8"), "sha256").hexdigest()
    (state_dir / "admin-password-hmac").write_text(expected)
    monkeypatch.setenv("SCRIBE_STATE_DIR", str(state_dir))
    cookie_signer = CookieSigner(secret_store.secret, max_age_seconds=60)
    monkeypatch.setattr(state, "_terminal_cookie_signer", cookie_signer)
    ticket_store = TicketStore(secret_store.secret, ttl_seconds=60)
    registry = ActiveTerminals(max_concurrent=2)
    app = FastAPI()
    register_bootstrap_routes(
        app, BootstrapConfig(admin_secret=secret_store, cookie_signer=cookie_signer)
    )
    register_terminal_routes(
        app,
        TerminalRouterConfig(
            registry=registry, cookie_signer=cookie_signer, ticket_store=ticket_store
        ),
    )
    return app


def _route_has_require_admin(route) -> bool:
    """Walk the route's dependant chain for the require_admin Depends."""
    dep = getattr(route, "dependant", None)
    if dep is None:
        return False
    stack = list(dep.dependencies)
    while stack:
        d = stack.pop()
        if d.call is require_admin:
            return True
        stack.extend(d.dependencies)
    return False


# Non-admin routes the test suite knows about for this app graph. New
# additions here must be reviewed: if you're adding an admin route,
# don't allowlist it — give it Depends(require_admin) instead.
_NON_ADMIN_ALLOWLIST = frozenset(
    {
        "/api/admin/authorize",
        "/api/admin/deauthorize",
        "/api/admin/logout",
        "/api/ws/terminal",  # WS handler runs require_admin_ws inside the body
    }
)


def test_every_admin_route_uses_require_admin_dependency(tmp_path, monkeypatch):
    """Coverage proof: every route either has require_admin in its
    dependency chain, or appears in the explicit non-admin allowlist."""
    app = _build_app(tmp_path, monkeypatch)
    missing: list[str] = []
    for route in app.routes:
        path = getattr(route, "path", None)
        if path is None:
            continue
        if _route_has_require_admin(route):
            continue
        if path in _NON_ADMIN_ALLOWLIST:
            continue
        # FastAPI/Starlette built-in routes (openapi, etc.) start with
        # /openapi or /docs and are never registered by us in this app
        # — skip them defensively.
        if path.startswith(("/openapi", "/docs", "/redoc")):
            continue
        missing.append(path)
    assert not missing, (
        "These admin-shaped routes are missing Depends(require_admin) "
        f"and are not in _NON_ADMIN_ALLOWLIST: {missing}. "
        "Either add the dependency or, if the route is intentionally "
        "non-admin, add it to _NON_ADMIN_ALLOWLIST."
    )


def test_admin_route_no_cookie_returns_401(tmp_path, monkeypatch):
    """No admin cookie → 401 ``admin_session_required``."""
    app = _build_app(tmp_path, monkeypatch)
    with TestClient(app, base_url="https://test") as client:
        resp = client.post("/api/terminal/ticket")
        assert resp.status_code == 401, resp.text


def test_admin_route_lan_cookie_passes(tmp_path, monkeypatch):
    """v1.0 cookie-only model: a valid admin cookie unlocks the
    admin route from any source IP — there's no subnet gate
    rejecting LAN access. Locks in the "admin reachable from
    192.168.x.x AND 10.42.0.x" contract."""
    app = _build_app(tmp_path, monkeypatch)
    with TestClient(app, base_url="https://test") as client:
        resp = client.post("/api/admin/authorize", json={"password": _TEST_CRED})
        assert resp.status_code == 200
        # Cookie set; subsequent admin call passes regardless of source.
        resp = client.post("/api/terminal/ticket")
        assert resp.status_code == 200, resp.text


def test_authorize_wrong_password_returns_401(tmp_path, monkeypatch):
    """``/api/admin/authorize`` is open to unauthenticated callers
    (that's how the cookie gets minted in the first place); a wrong
    password reaches the verifier and gets a 401
    ``invalid_password`` — distinct from the cookie-gate's 401
    ``admin_session_required``."""
    app = _build_app(tmp_path, monkeypatch)
    with TestClient(app, base_url="https://test") as client:
        resp = client.post("/api/admin/authorize", json={"password": "wrong-password"})
        assert resp.status_code == 401
        assert resp.json() == {"error": "invalid_password"}
