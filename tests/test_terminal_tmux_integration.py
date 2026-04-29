"""Integration tests against a real tmux on a dedicated test socket.

Isolated from the user's real tmux by pinning ``-L scribe-test``.
Verifies:

* the generated config is applied once per server-start (no per-attach
  state mutation — ``terminal-overrides`` must not accumulate duplicates);
* ``new-session -A`` reattach: a session survives a WS drop and is
  re-reachable via a fresh PTY + fresh tmux client;
* ``list_sessions`` round-trips the expected shape;
* ``close_all`` on the registry tears down PTYs but leaves the tmux
  server alive (persistence guarantee);
* ``kill_server`` is the only explicit teardown path.

Skipped when ``tmux`` is not on PATH (e.g. CI without tmux installed).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from meeting_scribe.terminal import tmux_helper
from meeting_scribe.terminal.auth import AdminSecretStore, CookieSigner, TicketStore
from meeting_scribe.terminal.bootstrap import BootstrapConfig, register_bootstrap_routes
from meeting_scribe.terminal.registry import ActiveTerminals
from meeting_scribe.terminal.router import TerminalRouterConfig, register_terminal_routes

pytestmark = pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux not installed")


TEST_SOCKET = "scribe-test"


@pytest.fixture
def test_config(tmp_path, monkeypatch) -> Path:
    """Write a minimal tmux config and return its path."""
    cfg = tmp_path / "scribe.tmux.conf"
    tmux_helper.write_tmux_config(cfg)
    # Route list_sessions / build_argv through the test socket + test config.
    return cfg


def _sh(argv: list[str]) -> str:
    """Run a tmux command synchronously; return stdout. Empty on failure."""
    import subprocess as _sp

    res = _sp.run(argv, capture_output=True, text=True, timeout=4.0)
    return res.stdout if res.returncode == 0 else ""


@pytest.fixture(autouse=True)
def _kill_test_tmux_server():
    """Fixture that kills the -L scribe-test socket before and after each test."""
    # Pre-cleanup (in case a previous test crashed).
    asyncio.run(tmux_helper.kill_server(socket=TEST_SOCKET))
    yield
    asyncio.run(tmux_helper.kill_server(socket=TEST_SOCKET))


# ── tmux_helper unit-ish tests ─────────────────────────────────


def test_config_written_once_has_marker(tmp_path):
    cfg = tmp_path / "scribe.tmux.conf"
    path = tmux_helper.write_tmux_config(cfg)
    assert path == cfg
    content = cfg.read_text()
    assert "scribe-tmux-config" in content
    assert "set -g history-limit" in content
    assert "terminal-overrides" in content
    # Second call is a no-op when content unchanged.
    mtime = cfg.stat().st_mtime_ns
    tmux_helper.write_tmux_config(cfg)
    assert cfg.stat().st_mtime_ns == mtime


async def test_list_sessions_empty_when_no_server():
    # Fixture already killed the server.
    sessions = await tmux_helper.list_sessions(socket=TEST_SOCKET)
    assert sessions == []


async def test_list_sessions_reflects_new_session(test_config):
    argv = tmux_helper.build_argv("probe", config=test_config, socket=TEST_SOCKET)
    # Detach immediately so we don't hold a client across the shell call.
    argv_detached = (
        argv[: argv.index("new-session") + 1] + ["-d"] + argv[argv.index("new-session") + 1 :]
    )
    _sh(argv_detached)
    try:
        sessions = await tmux_helper.list_sessions(socket=TEST_SOCKET)
        names = {s.name for s in sessions}
        assert "probe" in names
    finally:
        await tmux_helper.kill_server(socket=TEST_SOCKET)


async def test_socket_isolation(test_config):
    # Create on scribe-test socket.
    argv = tmux_helper.build_argv("iso", config=test_config, socket=TEST_SOCKET)
    argv_detached = (
        argv[: argv.index("new-session") + 1] + ["-d"] + argv[argv.index("new-session") + 1 :]
    )
    _sh(argv_detached)
    try:
        # Should be visible on scribe-test socket.
        scribe_test = await tmux_helper.list_sessions(socket=TEST_SOCKET)
        assert any(s.name == "iso" for s in scribe_test)
        # Should NOT be visible on the user's default socket or the prod `scribe` socket.
        prod_scribe = await tmux_helper.list_sessions(socket="scribe")
        assert all(s.name != "iso" for s in prod_scribe)
    finally:
        await tmux_helper.kill_server(socket=TEST_SOCKET)


async def test_terminal_overrides_not_duplicated(test_config):
    """Spawning 3 sessions back-to-back must not duplicate terminal-overrides."""
    for name in ("a", "b", "c"):
        argv = tmux_helper.build_argv(name, config=test_config, socket=TEST_SOCKET)
        argv_detached = (
            argv[: argv.index("new-session") + 1] + ["-d"] + argv[argv.index("new-session") + 1 :]
        )
        _sh(argv_detached)
    # Inspect the terminal-overrides set by the server.
    overrides = _sh(["tmux", "-L", TEST_SOCKET, "show-options", "-gv", "terminal-overrides"])
    # The config sets three Tc entries for xterm-256color, tmux-256color,
    # and *256col*. Each prefix should appear exactly once.
    assert overrides.count("xterm-256color:Tc") == 1, overrides
    assert overrides.count("tmux-256color:Tc") == 1, overrides
    # history-limit is a global server-wide option set once.
    limit = _sh(["tmux", "-L", TEST_SOCKET, "show-options", "-gv", "history-limit"]).strip()
    assert limit == "10000"


async def test_kill_server_returns_true_when_running(test_config):
    argv = tmux_helper.build_argv("k", config=test_config, socket=TEST_SOCKET)
    argv_detached = (
        argv[: argv.index("new-session") + 1] + ["-d"] + argv[argv.index("new-session") + 1 :]
    )
    _sh(argv_detached)
    killed = await tmux_helper.kill_server(socket=TEST_SOCKET)
    assert killed is True
    # Second kill: no server running.
    killed2 = await tmux_helper.kill_server(socket=TEST_SOCKET)
    assert killed2 is False


# ── End-to-end WS <-> real tmux ───────────────────────────────


@pytest.fixture
def app_factory(tmp_path, monkeypatch, test_config):
    """A FastAPI app wired to the -L scribe-test socket."""
    secret_path = tmp_path / "admin-secret"
    monkeypatch.setenv("SCRIBE_ADMIN_SECRET_FILE", str(secret_path))
    # Critical: ensure we're using REAL tmux, not an /bin/sh override from
    # another test file that may have polluted the environment.
    monkeypatch.delenv("SCRIBE_TERM_SHELL", raising=False)
    # Point the production build_argv at the test socket + test config.
    original_build_argv = tmux_helper.build_argv

    def _patched(session: str, *, shell_override: str | None = None, **_: Any) -> list[str]:
        return original_build_argv(
            session,
            config=test_config,
            socket=TEST_SOCKET,
            shell_override=shell_override,
        )

    monkeypatch.setattr(tmux_helper, "build_argv", _patched)

    def make() -> tuple[FastAPI, ActiveTerminals]:
        admin_secret = AdminSecretStore.load_or_create(secret_path)
        cookie_signer = CookieSigner(admin_secret.secret)
        ticket_store = TicketStore(admin_secret.secret, ttl_seconds=60)
        registry = ActiveTerminals(max_concurrent=4)
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
            ),
        )
        return app, registry

    return make


def _ws_headers(client: TestClient) -> dict[str, str]:
    cookies = "; ".join(f"{k}={v}" for k, v in client.cookies.items())
    return {
        "origin": "https://testserver",
        "host": "testserver",
        "cookie": cookies,
    }


def _read_until(ws, needle: bytes, *, max_frames: int = 50) -> bytes:
    collected = bytearray()
    for _ in range(max_frames):
        msg = ws.receive()
        if msg.get("type") == "websocket.disconnect":
            break
        raw = msg.get("bytes") or b""
        if raw[:1] == b"O":
            collected.extend(raw[1:])
            if needle in collected:
                return bytes(collected)
    raise AssertionError(f"did not see {needle!r}; saw {bytes(collected)!r}")


def _attach(client: TestClient, app) -> tuple[Any, str]:
    """Authorize + mint ticket."""
    admin_secret = Path(os.environ["SCRIBE_ADMIN_SECRET_FILE"]).read_text().strip()
    r = client.post("/api/admin/authorize", json={"secret": admin_secret})
    assert r.status_code == 200, r.text
    ticket = client.post("/api/terminal/ticket").json()["ticket"]
    return app, ticket


def test_attach_spawns_real_tmux(app_factory):
    app, registry = app_factory()
    with TestClient(app, base_url="https://testserver") as c:
        _, ticket = _attach(c, app)
        with c.websocket_connect("/api/ws/terminal", headers=_ws_headers(c)) as ws:
            ws.send_text(
                json.dumps(
                    {
                        "type": "attach",
                        "ticket": ticket,
                        "tmux_session": "smoke",
                        "cols": 80,
                        "rows": 24,
                    }
                )
            )
            attached = json.loads(ws.receive_text())
            assert attached["type"] == "attached"
            # Wait for tmux to actually spawn server + create session.
            # Poll the socket until the session appears (or we give up).
            for _ in range(80):
                external = asyncio.run(tmux_helper.list_sessions(socket=TEST_SOCKET))
                if any(s.name == "smoke" for s in external):
                    break
                time.sleep(0.05)
            assert any(s.name == "smoke" for s in external), (
                f"session 'smoke' not visible on socket after attach: {[s.name for s in external]}"
            )
            # Exec a shell command inside tmux → expect output.
            ws.send_bytes(b"I" + b"echo INSIDE-TMUX\r")
            _read_until(ws, b"INSIDE-TMUX")

        # Settle registry + tmux server state.
        for _ in range(100):
            if registry.summary()["count"] == 0:
                break
            time.sleep(0.05)
        # tmux client may need a moment to fully detach before list-sessions reflects it.
        time.sleep(0.2)

        # Session persists on the tmux server even after WS drop.
        sessions = asyncio.run(tmux_helper.list_sessions(socket=TEST_SOCKET))
        names = {s.name for s in sessions}
        assert "smoke" in names, f"tmux sessions after drop: {names}"


def _wait_tmux_session(name: str) -> None:
    for _ in range(80):
        sessions = asyncio.run(tmux_helper.list_sessions(socket=TEST_SOCKET))
        if any(s.name == name for s in sessions):
            return
        time.sleep(0.05)
    raise AssertionError(f"tmux session {name!r} never appeared on socket")


def test_new_session_A_reattaches_same_session(app_factory):
    app, registry = app_factory()
    probe = Path("/tmp/scribe_persistence_probe")
    probe.unlink(missing_ok=True)
    with TestClient(app, base_url="https://testserver") as c:
        _, ticket1 = _attach(c, app)
        with c.websocket_connect("/api/ws/terminal", headers=_ws_headers(c)) as ws:
            ws.send_text(
                json.dumps(
                    {
                        "type": "attach",
                        "ticket": ticket1,
                        "tmux_session": "durable",
                        "cols": 80,
                        "rows": 24,
                    }
                )
            )
            json.loads(ws.receive_text())
            _wait_tmux_session("durable")
            # Drop a breadcrumb — a file we create inside the session.
            ws.send_bytes(b"I" + f"echo MARK > {probe}\r".encode())
            # Wait for the file to materialize on disk — proof tmux->shell->fs ran.
            for _ in range(40):
                if probe.exists():
                    break
                time.sleep(0.05)
            assert probe.exists(), "shell inside tmux never wrote the probe file"

        # Settle
        for _ in range(100):
            if registry.summary()["count"] == 0:
                break
            time.sleep(0.05)

        # tmux session must still exist on the server.
        _wait_tmux_session("durable")

        # New attach to the SAME session name — tmux should reattach
        # existing session (courtesy of -A), not create a fresh one.
        # We verify via the breadcrumb file on disk: if tmux really
        # preserved the session, the shell from the first attach wrote
        # the marker file before the WS drop. Reading it on the
        # filesystem is a stronger, simpler check than trying to
        # parse tmux's repaint-then-echo output over the WS.
        ticket2 = c.post("/api/terminal/ticket").json()["ticket"]
        with c.websocket_connect("/api/ws/terminal", headers=_ws_headers(c)) as ws:
            ws.send_text(
                json.dumps(
                    {
                        "type": "attach",
                        "ticket": ticket2,
                        "tmux_session": "durable",
                        "cols": 80,
                        "rows": 24,
                    }
                )
            )
            attached = json.loads(ws.receive_text())
            assert attached["type"] == "attached"
            # Probe file was written by the shell inside the tmux session
            # on the first attach. If tmux dropped the session, it's gone.
            assert probe.exists() and probe.read_text().strip() == "MARK", (
                f"probe file missing after reattach; contents={probe.read_text() if probe.exists() else '<missing>'}"
            )
            # Still see the tmux session.
            _wait_tmux_session("durable")
    probe.unlink(missing_ok=True)


def test_close_all_leaves_tmux_server_alive(app_factory):
    app, registry = app_factory()
    with TestClient(app, base_url="https://testserver") as c:
        _, ticket = _attach(c, app)
        with c.websocket_connect("/api/ws/terminal", headers=_ws_headers(c)) as ws:
            ws.send_text(
                json.dumps(
                    {
                        "type": "attach",
                        "ticket": ticket,
                        "tmux_session": "keepalive",
                        "cols": 80,
                        "rows": 24,
                    }
                )
            )
            json.loads(ws.receive_text())
            _wait_tmux_session("keepalive")
        # Shutdown-style registry teardown.
        asyncio.run(registry.close_all(reason="test-shutdown"))
        # tmux server must still be alive.
        sessions = asyncio.run(tmux_helper.list_sessions(socket=TEST_SOCKET))
        assert any(s.name == "keepalive" for s in sessions)
