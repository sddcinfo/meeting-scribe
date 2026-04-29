"""End-to-end WS integration: bootstrap → ticket → attach → echo.

Stands up a minimal FastAPI app that wires only the terminal routes, so
these tests run in process without booting all of meeting-scribe's
backends. Uses ``SCRIBE_TERM_SHELL=/bin/sh`` so tmux isn't required.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from meeting_scribe.terminal import protocol
from meeting_scribe.terminal.auth import AdminSecretStore, CookieSigner, TicketStore
from meeting_scribe.terminal.bootstrap import BootstrapConfig, register_bootstrap_routes
from meeting_scribe.terminal.registry import ActiveTerminals
from meeting_scribe.terminal.router import TerminalRouterConfig, register_terminal_routes

# ── Test fixtures ────────────────────────────────────────────────


@pytest.fixture
def admin_secret(tmp_path, monkeypatch) -> AdminSecretStore:
    secret_path = tmp_path / "admin-secret"
    monkeypatch.setenv("SCRIBE_ADMIN_SECRET_FILE", str(secret_path))
    monkeypatch.setenv("SCRIBE_TERM_SHELL", "/bin/sh")
    return AdminSecretStore.load_or_create(secret_path)


@pytest.fixture
def test_app_factory(admin_secret):
    """Factory producing a FastAPI app with terminal routes wired.

    Accepts a ``is_guest_scope`` override so we can simulate guest requests
    in tests that specifically exercise that rejection path.
    """

    def make(
        *, is_guest_scope: Any = None, max_concurrent: int = 4, ticket_ttl: float = 60.0
    ) -> tuple[FastAPI, ActiveTerminals, TicketStore, CookieSigner]:
        app = FastAPI()
        cookie_signer = CookieSigner(admin_secret.secret, max_age_seconds=60)
        ticket_store = TicketStore(admin_secret.secret, ttl_seconds=ticket_ttl)
        registry = ActiveTerminals(max_concurrent=max_concurrent)
        # Default: assume admin scope (test client uses http scheme, but our
        # production guard treats http as guest — we bypass that for tests).
        scope_fn = is_guest_scope if is_guest_scope is not None else (lambda _r: False)
        register_bootstrap_routes(
            app,
            BootstrapConfig(
                admin_secret=admin_secret,
                cookie_signer=cookie_signer,
                is_guest_scope=scope_fn,
            ),
        )
        register_terminal_routes(
            app,
            TerminalRouterConfig(
                registry=registry,
                cookie_signer=cookie_signer,
                ticket_store=ticket_store,
                is_guest_scope=scope_fn,
            ),
        )
        return app, registry, ticket_store, cookie_signer

    return make


def _make_client(app: FastAPI) -> TestClient:
    # https base_url so cookies marked Secure round-trip correctly.
    return TestClient(app, base_url="https://testserver")


def _authorize(client: TestClient, admin_secret: AdminSecretStore) -> None:
    """POST the secret so subsequent requests are cookie-authenticated."""
    resp = client.post(
        "/api/admin/authorize",
        json={"secret": admin_secret.secret.decode()},
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
    app, _, _, _ = test_app_factory()
    with _make_client(app) as client:
        resp = client.post("/api/terminal/ticket")
        assert resp.status_code == 403


def test_mint_rejected_for_guest_scope(test_app_factory):
    app, _, _, _ = test_app_factory(is_guest_scope=lambda _r: True)
    with _make_client(app) as client:
        # Even with cookie, guest scope is rejected.
        resp = client.post("/api/terminal/ticket")
        assert resp.status_code == 403


def test_authorize_wrong_secret(test_app_factory, admin_secret):
    app, _, _, _ = test_app_factory()
    with _make_client(app) as client:
        resp = client.post("/api/admin/authorize", json={"secret": "not-the-secret"})
        assert resp.status_code == 401


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


def test_ws_rejected_guest_scope(test_app_factory, admin_secret):
    app, _, _, _ = test_app_factory(is_guest_scope=lambda _r: True)
    with (
        _make_client(app) as client,
        pytest.raises(Exception),
        client.websocket_connect("/api/ws/terminal", headers=_ws_headers(client)) as ws,
    ):
        for _ in range(5):
            msg = ws.receive(timeout=1.0)
            if msg.get("type") == "websocket.disconnect":
                break


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
        # Cookie is now cleared; next mint must be denied.
        client.cookies.clear()
        resp = client.post("/api/terminal/ticket")
        assert resp.status_code == 403


def test_terminal_access_returns_secret_for_admin_scope(test_app_factory, admin_secret):
    app, _, _, _ = test_app_factory()
    with _make_client(app) as client:
        # Pre-auth: no cookie, but admin scope — endpoint still surfaces the
        # secret so the Settings UI can mint a cookie on the user's behalf.
        resp = client.get("/api/admin/terminal-access")
        assert resp.status_code == 200
        data = resp.json()
        assert data["secret"] == admin_secret.secret.decode()
        assert data["cookie_set"] is False
        assert data["secret_path"]
        assert data["cookie_max_age_seconds"] > 0

        _authorize(client, admin_secret)
        resp = client.get("/api/admin/terminal-access")
        assert resp.status_code == 200
        assert resp.json()["cookie_set"] is True


def test_terminal_access_denied_for_guest_scope(test_app_factory, admin_secret):
    app, _, _, _ = test_app_factory(is_guest_scope=lambda _r: True)
    with _make_client(app) as client:
        resp = client.get("/api/admin/terminal-access")
        assert resp.status_code == 403


def test_history_log_captures_pty_output(tmp_path, admin_secret):
    """Every PTY byte is tee'd to the per-meeting log — the meeting
    folder ends up with a durable record of shell activity just like
    slides/audio.
    """
    log_path = tmp_path / "terminal.log"
    cookie_signer = CookieSigner(admin_secret.secret, max_age_seconds=60)
    ticket_store = TicketStore(admin_secret.secret, ttl_seconds=60)
    registry = ActiveTerminals(max_concurrent=2)

    app = FastAPI()
    register_bootstrap_routes(
        app,
        BootstrapConfig(
            admin_secret=admin_secret,
            cookie_signer=cookie_signer,
            is_guest_scope=lambda _r: False,
        ),
    )
    register_terminal_routes(
        app,
        TerminalRouterConfig(
            registry=registry,
            cookie_signer=cookie_signer,
            ticket_store=ticket_store,
            is_guest_scope=lambda _r: False,
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


def test_history_path_resolver_none_disables_logging(tmp_path, admin_secret):
    """When the resolver returns None, history is silently disabled: the
    terminal still works, nothing is written to disk.
    """
    cookie_signer = CookieSigner(admin_secret.secret, max_age_seconds=60)
    ticket_store = TicketStore(admin_secret.secret, ttl_seconds=60)
    registry = ActiveTerminals(max_concurrent=2)
    app = FastAPI()
    register_bootstrap_routes(
        app,
        BootstrapConfig(
            admin_secret=admin_secret,
            cookie_signer=cookie_signer,
            is_guest_scope=lambda _r: False,
        ),
    )
    register_terminal_routes(
        app,
        TerminalRouterConfig(
            registry=registry,
            cookie_signer=cookie_signer,
            ticket_store=ticket_store,
            is_guest_scope=lambda _r: False,
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
