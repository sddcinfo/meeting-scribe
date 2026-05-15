"""End-to-end WS integration: bootstrap → ticket → attach → echo.

Stands up a minimal FastAPI app that wires only the terminal routes, so
these tests run in process without booting all of meeting-scribe's
backends. Uses ``SCRIBE_TERM_SHELL=/bin/sh`` so tmux isn't required.
"""

from __future__ import annotations

import hmac as _hmac
import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from meeting_scribe.terminal import protocol
from meeting_scribe.terminal.auth import AdminSecretStore, CookieSigner, TicketStore
from meeting_scribe.terminal.bootstrap import BootstrapConfig, register_bootstrap_routes
from meeting_scribe.terminal.registry import ActiveTerminals
from meeting_scribe.terminal.router import TerminalRouterConfig, register_terminal_routes

# The wizard password the test suite uses. ``_authorize()`` writes its
# HMAC under ``admin-password-hmac`` so ``setup_state.verify_admin_password``
# accepts it. The string is arbitrary — this is unit-test scaffolding,
# not the production deterministic password.
_TEST_CRED = "TestPassword1234"

# ── Test fixtures ────────────────────────────────────────────────


@pytest.fixture
def admin_secret(tmp_path, monkeypatch) -> AdminSecretStore:
    secret_path = tmp_path / "admin-secret"
    monkeypatch.setenv("SCRIBE_ADMIN_SECRET_FILE", str(secret_path))
    monkeypatch.setenv("SCRIBE_TERM_SHELL", "/bin/sh")
    # v1.0 admin auth is cookie-only; no AP-subnet gate to bypass
    # for unit tests. Persist the wizard ``admin-password-hmac`` the
    # new ``setup_state.verify_admin_password`` reads from. Live
    # production writes this in ``persist_authoritative`` after
    # ``/api/setup/finish``; tests just lay down the file directly.
    state_dir = tmp_path / "scribe-state"
    state_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SCRIBE_STATE_DIR", str(state_dir))
    store = AdminSecretStore.load_or_create(secret_path)
    expected = _hmac.new(store.secret, _TEST_CRED.encode("utf-8"), "sha256").hexdigest()
    (state_dir / "admin-password-hmac").write_text(expected)
    return store


@pytest.fixture
def test_app_factory(admin_secret, monkeypatch):
    """Factory producing a FastAPI app with terminal routes wired.

    ``has_admin_session`` reads ``state._terminal_cookie_signer`` —
    the same global the production server populates at startup.
    Tests install the per-test signer there for the duration of the
    fixture so the cookie minted by ``/api/admin/authorize`` is the
    one ``require_admin`` validates.
    """
    from meeting_scribe.runtime import state as _state

    def make(
        *, max_concurrent: int = 4, ticket_ttl: float = 60.0
    ) -> tuple[FastAPI, ActiveTerminals, TicketStore, CookieSigner]:
        app = FastAPI()
        cookie_signer = CookieSigner(admin_secret.secret, max_age_seconds=60)
        ticket_store = TicketStore(admin_secret.secret, ttl_seconds=ticket_ttl)
        registry = ActiveTerminals(max_concurrent=max_concurrent)
        monkeypatch.setattr(_state, "_terminal_cookie_signer", cookie_signer)
        register_bootstrap_routes(
            app,
            BootstrapConfig(
                admin_secret=admin_secret,
                cookie_signer=cookie_signer,
            ),
        )
        register_terminal_routes(
            app,
            TerminalRouterConfig(
                registry=registry,
                cookie_signer=cookie_signer,
                ticket_store=ticket_store,
            ),
        )
        return app, registry, ticket_store, cookie_signer

    return make


def _make_client(app: FastAPI) -> TestClient:
    # https base_url so cookies marked Secure round-trip correctly.
    return TestClient(app, base_url="https://testserver")


def _authorize(client: TestClient, admin_secret: AdminSecretStore) -> None:
    """POST the wizard password so subsequent requests are cookie-authenticated.

    The route now reads the typed password and HMAC-compares it
    against ``admin-password-hmac`` (the wizard's persistent admin
    credential) instead of the master secret. The fixture wrote the
    HMAC for ``_TEST_CRED`` at admin-secret-store creation
    time so this just matches that.
    """
    del admin_secret  # signature kept for callers; HMAC's already on disk
    resp = client.post(
        "/api/admin/authorize",
        json={"password": _TEST_CRED},
    )
    assert resp.status_code == 200, resp.text


def _mint_ticket(client: TestClient) -> str:
    resp = client.post("/api/terminal/ticket")
    assert resp.status_code == 200, resp.text
    return resp.json()["ticket"]


def _ws_headers(client: TestClient | None = None) -> dict[str, str]:
    """Browser-style headers the router's origin check will accept.

    Starlette's TestClient does not auto-attach session cookies to WS
    upgrade requests — pass them explicitly via the ``Cookie`` header.
    """
    h = {
        "origin": "https://testserver",
        "host": "testserver",
    }
    if client is not None:
        cookies = "; ".join(f"{k}={v}" for k, v in client.cookies.items())
        if cookies:
            h["cookie"] = cookies
    return h


def _attach_payload(ticket: str, *, session: str = "test", cols: int = 80, rows: int = 24) -> str:
    return json.dumps(
        {
            "type": "attach",
            "ticket": ticket,
            "tmux_session": session,
            "cols": cols,
            "rows": rows,
        }
    )


def _read_output_until(ws, needle: bytes, *, max_frames: int = 30) -> bytes:
    """Drain frames until *needle* shows up in an O-prefixed binary body.

    ``WebSocketTestSession.receive()`` has no timeout — each call blocks
    until the server emits a frame. We rely on the server to emit promptly
    (PTY echo is milliseconds) and bound the scan with a frame count to
    guard against runaway blocking if the server deadlocks.
    """
    collected = bytearray()
    seen_frames: list[tuple[str, bytes, str]] = []
    for _ in range(max_frames):
        msg = ws.receive()
        mtype = msg.get("type", "")
        kb = msg.get("bytes") or b""
        kt = msg.get("text") or ""
        seen_frames.append((mtype, kb[:80], kt[:120]))
        if mtype == "websocket.disconnect":
            break
        if mtype in ("websocket.receive", "websocket.send"):
            raw = kb
            if raw and raw[:1] == b"O":
                collected.extend(raw[1:])
                if needle in collected:
                    return bytes(collected)
    raise AssertionError(
        f"did not see {needle!r}; saw {bytes(collected)!r}; frames={seen_frames!r}"
    )


# ── Happy path ────────────────────────────────────────────────────


def test_happy_path_echo(test_app_factory, admin_secret):
    app, _, _, _ = test_app_factory()
    with _make_client(app) as client:
        _authorize(client, admin_secret)
        ticket = _mint_ticket(client)
        with client.websocket_connect("/api/ws/terminal", headers=_ws_headers(client)) as ws:
            ws.send_text(_attach_payload(ticket))
            attached = json.loads(ws.receive_text())
            assert attached["type"] == "attached"
            assert attached["tmux_session"] == "test"
            ws.send_bytes(b"I" + b"echo HELLO-123\n")
            out = _read_output_until(ws, b"HELLO-123")
            assert b"HELLO-123" in out


def test_ping_pong(test_app_factory, admin_secret):
    app, _, _, _ = test_app_factory()
    with _make_client(app) as client:
        _authorize(client, admin_secret)
        ticket = _mint_ticket(client)
        with client.websocket_connect("/api/ws/terminal", headers=_ws_headers(client)) as ws:
            ws.send_text(_attach_payload(ticket))
            assert json.loads(ws.receive_text())["type"] == "attached"
            ws.send_text('{"type":"ping"}')
            # Keep reading until we see the pong (output frames may interleave)
            found_pong = False
            for _ in range(10):
                msg = ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("text"):
                    payload = json.loads(msg["text"])
                    if payload.get("type") == "pong":
                        found_pong = True
                        break
            assert found_pong


# ── Auth failures ────────────────────────────────────────────────


def test_mint_rejected_without_cookie(test_app_factory):
    """No admin cookie → 401 ``admin_session_required`` regardless of
    source. v1.0 admin auth is cookie-only; the subnet gate that
    used to also rejected off-AP callers was removed so admin works
    from the LAN as well as the AP."""
    app, _, _, _ = test_app_factory()
    with _make_client(app) as client:
        resp = client.post("/api/terminal/ticket")
        assert resp.status_code == 401


def test_authorize_wrong_secret(test_app_factory, admin_secret):
    app, _, _, _ = test_app_factory()
    with _make_client(app) as client:
        resp = client.post("/api/admin/authorize", json={"password": "not-the-password"})
        assert resp.status_code == 401


def test_authorize_form_redirects_on_success(test_app_factory, admin_secret):
    """Browser POSTs (``application/x-www-form-urlencoded``) get a 303
    redirect to ``/`` with the cookie set, so a `<form>` submission
    lands on the live UI instead of an inline JSON body."""
    app, _, _, _ = test_app_factory()
    with _make_client(app) as client:
        resp = client.post(
            "/api/admin/authorize",
            data={"password": _TEST_CRED},
            follow_redirects=False,
        )
        assert resp.status_code == 303, resp.text
        assert resp.headers["location"] == "/"
        # Cookie was set on the redirect response.
        assert "scribe_admin=" in resp.headers.get("set-cookie", "")


def test_authorize_form_redirects_on_failure(test_app_factory, admin_secret):
    """Form POSTs with a bad password get a 303 to ``/auth?err=1`` so
    the browser-history-friendly redirect surface is preserved.
    Browsers can't follow a 401-with-Location reliably."""
    app, _, _, _ = test_app_factory()
    with _make_client(app) as client:
        resp = client.post(
            "/api/admin/authorize",
            data={"password": "wrong-password"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers["location"] == "/auth?err=1"
        assert "scribe_admin=" not in resp.headers.get("set-cookie", "")


def test_authorize_xhr_returns_ok(test_app_factory, admin_secret):
    """JSON XHR clients still get the original ``{ok: true}`` body for
    backward compat with any non-browser caller."""
    app, _, _, _ = test_app_factory()
    with _make_client(app) as client:
        resp = client.post(
            "/api/admin/authorize",
            json={"password": _TEST_CRED},
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}


def test_authorize_xhr_returns_401_on_failure(test_app_factory, admin_secret):
    app, _, _, _ = test_app_factory()
    with _make_client(app) as client:
        resp = client.post("/api/admin/authorize", json={"password": "wrong-password"})
        assert resp.status_code == 401
        assert resp.json() == {"error": "invalid_password"}


def test_authorize_accepts_charset_suffix(test_app_factory, admin_secret):
    """Both ``application/json; charset=UTF-8`` and
    ``application/x-www-form-urlencoded; charset=UTF-8`` parse — the
    media-type matcher strips parameters before comparing."""
    app, _, _, _ = test_app_factory()
    with _make_client(app) as client:
        # JSON with charset.
        resp = client.post(
            "/api/admin/authorize",
            content=f'{{"password":"{_TEST_CRED}"}}',
            headers={"Content-Type": "application/json; charset=UTF-8"},
        )
        assert resp.status_code == 200, resp.text
        client.cookies.clear()
        # Form with charset.
        resp = client.post(
            "/api/admin/authorize",
            content=f"password={_TEST_CRED}",
            headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"},
            follow_redirects=False,
        )
        assert resp.status_code == 303, resp.text


def test_authorize_unsupported_media_type(test_app_factory, admin_secret):
    """``text/plain`` (or any other unknown type) → 415."""
    app, _, _, _ = test_app_factory()
    with _make_client(app) as client:
        resp = client.post(
            "/api/admin/authorize",
            content=_TEST_CRED,
            headers={"Content-Type": "text/plain"},
        )
        assert resp.status_code == 415
        assert resp.json() == {"error": "unsupported_media_type"}


def test_authorize_malformed_body(test_app_factory, admin_secret):
    """Malformed JSON body → 400 without leaking parse details."""
    app, _, _, _ = test_app_factory()
    with _make_client(app) as client:
        resp = client.post(
            "/api/admin/authorize",
            content="{not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        assert resp.json() == {"error": "invalid_body"}


def test_ws_rejected_without_cookie(test_app_factory):
    app, _, _, _ = test_app_factory()
    # No authorize(); no cookie. Three with-contexts collapsed per
    # SIM117 — the client, the pytest.raises, and the websocket
    # handshake all live in one scope.
    with (
        _make_client(app) as client,
        pytest.raises(Exception),
        client.websocket_connect("/api/ws/terminal", headers=_ws_headers(client)) as ws,
    ):
        ws.receive()  # connection refused before this


def test_ws_rejected_invalid_ticket(test_app_factory, admin_secret):
    app, _, _, _ = test_app_factory()
    with _make_client(app) as client:
        _authorize(client, admin_secret)
        fake_ticket = "a" * 64 + "." + "b" * 64
        with (
            pytest.raises(Exception),
            client.websocket_connect("/api/ws/terminal", headers=_ws_headers(client)) as ws,
        ):
            ws.send_text(_attach_payload(fake_ticket))
            # Should get error frame and disconnect
            for _ in range(5):
                msg = ws.receive(timeout=1.0)
                if msg.get("type") == "websocket.disconnect":
                    break


def test_ticket_is_single_use(test_app_factory, admin_secret):
    app, _, _, _ = test_app_factory()
    with _make_client(app) as client:
        _authorize(client, admin_secret)
        ticket = _mint_ticket(client)
        # First use — happy path
        with client.websocket_connect("/api/ws/terminal", headers=_ws_headers(client)) as ws:
            ws.send_text(_attach_payload(ticket))
            attached = json.loads(ws.receive_text())
            assert attached["type"] == "attached"
        # Reusing the same ticket must fail
        with (
            pytest.raises(Exception),
            client.websocket_connect("/api/ws/terminal", headers=_ws_headers(client)) as ws,
        ):
            ws.send_text(_attach_payload(ticket))
            for _ in range(5):
                msg = ws.receive(timeout=1.0)
                if msg.get("type") == "websocket.disconnect":
                    break


def test_ws_rejected_bad_origin(test_app_factory, admin_secret):
    app, _, _, _ = test_app_factory()
    with _make_client(app) as client:
        _authorize(client, admin_secret)
        ticket = _mint_ticket(client)
        with (
            pytest.raises(Exception),
            client.websocket_connect(
                "/api/ws/terminal",
                headers={"origin": "https://evil.example.com", "host": "testserver"},
            ) as ws,
        ):
            ws.send_text(_attach_payload(ticket))
            for _ in range(5):
                msg = ws.receive(timeout=1.0)
                if msg.get("type") == "websocket.disconnect":
                    break


def test_ws_accepts_same_origin_http(test_app_factory, admin_secret):
    """HTTP same-origin is allowed — the scope gate handles plain-http guest
    listeners; the origin check only needs to defend against cross-site WS."""
    app, _, _, _ = test_app_factory()
    with _make_client(app) as client:
        _authorize(client, admin_secret)
        ticket = _mint_ticket(client)
        cookies = "; ".join(f"{k}={v}" for k, v in client.cookies.items())
        with client.websocket_connect(
            "/api/ws/terminal",
            headers={"origin": "http://testserver", "host": "testserver", "cookie": cookies},
        ) as ws:
            ws.send_text(_attach_payload(ticket))
            first = json.loads(ws.receive_text())
            assert first["type"] == "attached"


# ── Capacity ─────────────────────────────────────────────────────


def test_capacity_full_error_frame(test_app_factory, admin_secret):
    app, registry, _, _ = test_app_factory(max_concurrent=1)
    with _make_client(app) as client:
        _authorize(client, admin_secret)
        ticket_a = _mint_ticket(client)
        ticket_b = _mint_ticket(client)
        with client.websocket_connect("/api/ws/terminal", headers=_ws_headers(client)) as ws_a:
            ws_a.send_text(_attach_payload(ticket_a))
            assert json.loads(ws_a.receive_text())["type"] == "attached"
            # Second connection — capacity is now full.
            with (
                pytest.raises(Exception),
                client.websocket_connect("/api/ws/terminal", headers=_ws_headers(client)) as ws_b,
            ):
                ws_b.send_text(_attach_payload(ticket_b))
                for _ in range(5):
                    msg = ws_b.receive(timeout=1.0)
                    if msg.get("type") == "websocket.disconnect":
                        break
        # After first ws closes, the registry token must be released.
        # Poll — close is async and proc.wait can take a moment.
        for _ in range(100):
            if registry.summary()["available"] == 1:
                break
            time.sleep(0.05)
        assert registry.summary()["available"] == 1


# ── Frame size ───────────────────────────────────────────────────


def test_oversized_binary_frame_closes_ws(test_app_factory, admin_secret):
    app, _, _, _ = test_app_factory()
    with _make_client(app) as client:
        _authorize(client, admin_secret)
        ticket = _mint_ticket(client)
        with (
            pytest.raises(Exception),
            client.websocket_connect("/api/ws/terminal", headers=_ws_headers(client)) as ws,
        ):
            ws.send_text(_attach_payload(ticket))
            assert json.loads(ws.receive_text())["type"] == "attached"
            ws.send_bytes(b"I" + b"x" * (protocol.INBOUND_FRAME_MAX + 1))
            for _ in range(10):
                msg = ws.receive(timeout=1.0)
                if msg.get("type") == "websocket.disconnect":
                    break


# ── Lifecycle ────────────────────────────────────────────────────


def test_ws_drop_closes_pty(test_app_factory, admin_secret):
    app, registry, _, _ = test_app_factory()
    with _make_client(app) as client:
        _authorize(client, admin_secret)
        ticket = _mint_ticket(client)
        with client.websocket_connect("/api/ws/terminal", headers=_ws_headers(client)) as ws:
            ws.send_text(_attach_payload(ticket))
            assert json.loads(ws.receive_text())["type"] == "attached"
            assert registry.summary()["count"] == 1
        # After context exit, WS is closed. Registry unwinds asynchronously —
        # close() can take up to CLOSE_WAIT_S for proc.wait, so poll generously.
        for _ in range(100):
            if registry.summary()["count"] == 0:
                break
            time.sleep(0.05)
        assert registry.summary()["count"] == 0


def test_resize_roundtrips(test_app_factory, admin_secret):
    app, _registry, _, _ = test_app_factory()
    with _make_client(app) as client:
        _authorize(client, admin_secret)
        ticket = _mint_ticket(client)
        with client.websocket_connect("/api/ws/terminal", headers=_ws_headers(client)) as ws:
            ws.send_text(_attach_payload(ticket, cols=80, rows=24))
            assert json.loads(ws.receive_text())["type"] == "attached"
            ws.send_text(json.dumps({"type": "resize", "cols": 132, "rows": 43}))
            # Can't easily introspect session cols from outside without a hook —
            # settle for "command reads it back".
            ws.send_bytes(b"I" + b"stty size\n")
            _read_output_until(ws, b"43 132")


def test_bootstrap_deauthorize_clears_cookie(test_app_factory, admin_secret):
    app, _, _, _ = test_app_factory()
    with _make_client(app) as client:
        _authorize(client, admin_secret)
        resp = client.post("/api/terminal/ticket")
        assert resp.status_code == 200
        r = client.post("/api/admin/deauthorize")
        assert r.status_code == 200
        # Cookie is now cleared; next mint must be denied (no cookie
        # → 401 from require_admin's session layer).
        client.cookies.clear()
        resp = client.post("/api/terminal/ticket")
        assert resp.status_code == 401


def test_terminal_access_endpoint_was_removed(test_app_factory, admin_secret):
    """``/api/admin/terminal-access`` returned the master HMAC secret to
    any caller in admin scope. The wizard now mints a memorable
    password and persists only its HMAC, so there is nothing for this
    endpoint to surface — it was removed wholesale to close the leak.
    """
    del admin_secret  # fixture used only for setup_state side-effects
    app, _, _, _ = test_app_factory()
    with _make_client(app) as client:
        resp = client.get("/api/admin/terminal-access")
        assert resp.status_code == 404


def test_history_log_captures_pty_output(tmp_path, admin_secret, monkeypatch):
    """Every PTY byte is tee'd to the per-meeting log — the meeting
    folder ends up with a durable record of shell activity just like
    slides/audio.
    """
    from meeting_scribe.runtime import state as _state

    log_path = tmp_path / "terminal.log"
    cookie_signer = CookieSigner(admin_secret.secret, max_age_seconds=60)
    ticket_store = TicketStore(admin_secret.secret, ttl_seconds=60)
    registry = ActiveTerminals(max_concurrent=2)
    monkeypatch.setattr(_state, "_terminal_cookie_signer", cookie_signer)

    app = FastAPI()
    register_bootstrap_routes(
        app,
        BootstrapConfig(
            admin_secret=admin_secret,
            cookie_signer=cookie_signer,
        ),
    )
    register_terminal_routes(
        app,
        TerminalRouterConfig(
            registry=registry,
            cookie_signer=cookie_signer,
            ticket_store=ticket_store,
            history_path_fn=lambda: log_path,
        ),
    )

    with _make_client(app) as client:
        _authorize(client, admin_secret)
        ticket = _mint_ticket(client)
        with client.websocket_connect("/api/ws/terminal", headers=_ws_headers(client)) as ws:
            ws.send_text(_attach_payload(ticket))
            assert json.loads(ws.receive_text())["type"] == "attached"
            ws.send_bytes(b"I" + b"echo HIST-ONE\n")
            _read_output_until(ws, b"HIST-ONE")

        # Let the close-side flush hit the log before we inspect.
        time.sleep(0.3)
        assert log_path.exists(), "history log was not created"
        # Log contains the shell echo — the shape of a real meeting artifact.
        assert b"HIST-ONE" in log_path.read_bytes()


def test_history_path_resolver_none_disables_logging(tmp_path, admin_secret, monkeypatch):
    """When the resolver returns None, history is silently disabled: the
    terminal still works, nothing is written to disk.
    """
    from meeting_scribe.runtime import state as _state

    cookie_signer = CookieSigner(admin_secret.secret, max_age_seconds=60)
    ticket_store = TicketStore(admin_secret.secret, ttl_seconds=60)
    registry = ActiveTerminals(max_concurrent=2)
    monkeypatch.setattr(_state, "_terminal_cookie_signer", cookie_signer)
    app = FastAPI()
    register_bootstrap_routes(
        app,
        BootstrapConfig(
            admin_secret=admin_secret,
            cookie_signer=cookie_signer,
        ),
    )
    register_terminal_routes(
        app,
        TerminalRouterConfig(
            registry=registry,
            cookie_signer=cookie_signer,
            ticket_store=ticket_store,
            history_path_fn=lambda: None,
        ),
    )

    with _make_client(app) as client:
        _authorize(client, admin_secret)
        ticket = _mint_ticket(client)
        with client.websocket_connect("/api/ws/terminal", headers=_ws_headers(client)) as ws:
            ws.send_text(_attach_payload(ticket))
            assert json.loads(ws.receive_text())["type"] == "attached"
            ws.send_bytes(b"I" + b"echo NO-LOG\n")
            _read_output_until(ws, b"NO-LOG")

    # The tmp_path has no log file — nothing was tee'd.
    assert not list(tmp_path.rglob("*.log"))
