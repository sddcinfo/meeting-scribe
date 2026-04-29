"""Terminal REST + WebSocket endpoints.

Wired into the main FastAPI app by
:func:`register_terminal_routes`. The endpoint set is:

* ``POST /api/terminal/ticket``     — mint a single-use WS ticket (admin)
* ``GET  /api/terminal/sessions``   — list live tmux sessions + registry
* ``POST /api/terminal/tmux/kill``  — explicit admin teardown of the socket
* ``WS   /api/ws/terminal``         — attach a PTY

Every endpoint is gated by the admin cookie + the project-wide
``_is_guest_scope`` predicate. The WS additionally consumes a
single-use ticket on the handshake.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from fastapi import Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from meeting_scribe.terminal import protocol, tmux_helper
from meeting_scribe.terminal.auth import COOKIE_NAME, CookieSigner, TicketStore
from meeting_scribe.terminal.history import TerminalHistoryLog
from meeting_scribe.terminal.protocol import ProtocolError
from meeting_scribe.terminal.pty_session import TerminalSession
from meeting_scribe.terminal.registry import ActiveTerminals

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


GuestScopeFn = Callable[[Any], bool]
HistoryPathFn = Callable[[], Any]  # returns pathlib.Path | None

STATUS_PUSH_INTERVAL_S: float = 2.0


@dataclass
class TerminalRouterConfig:
    registry: ActiveTerminals
    cookie_signer: CookieSigner
    ticket_store: TicketStore
    is_guest_scope: GuestScopeFn
    # Resolves the file path for the per-meeting terminal history log.
    # Returns None → history disabled (no log tee, no replay).
    history_path_fn: HistoryPathFn | None = None


# ── Helpers ──────────────────────────────────────────────────────


def _cookie_ok(cookie_signer: CookieSigner, cookies: dict[str, str]) -> bool:
    return cookie_signer.verify(cookies.get(COOKIE_NAME))


def _origin_ok(ws: WebSocket) -> bool:
    """Same-origin check: the WS request's Origin must match its Host header.

    This defends against cross-site WebSocket hijacking: a page served
    from ``evil.example.com`` can open a WS to our host, but its Origin
    header will be ``evil.example.com`` — not our host. We reject.

    We do NOT hard-code https: the scope gate (``_is_guest_scope``)
    already rejects the plain-http guest listener. Treating HTTP-on-LAN
    as forbidden would break local browser testing without adding real
    security on top of scope + cookie + ticket.
    """
    origin = ws.headers.get("origin", "")
    if not origin:
        return False
    try:
        parsed = urlparse(origin)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host_header = (ws.headers.get("host") or "").split(":")[0].lower()
    origin_host = (parsed.hostname or "").lower()
    if not origin_host:
        return False
    if origin_host == host_header:
        return True
    loopback = {"localhost", "127.0.0.1"}
    return origin_host in loopback and host_header in loopback


# ── Registration ─────────────────────────────────────────────────


def register_terminal_routes(app: FastAPI, cfg: TerminalRouterConfig) -> None:
    @app.post("/api/terminal/ticket")
    async def mint_ticket(request: Request) -> JSONResponse:
        if cfg.is_guest_scope(request) or not _cookie_ok(cfg.cookie_signer, request.cookies):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        ticket = await cfg.ticket_store.mint()
        return JSONResponse({"ticket": ticket, "expires_in": cfg.ticket_store.ttl_seconds})

    @app.get("/api/terminal/sessions")
    async def list_sessions(request: Request) -> JSONResponse:
        if cfg.is_guest_scope(request) or not _cookie_ok(cfg.cookie_signer, request.cookies):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        sessions = await tmux_helper.list_sessions()
        return JSONResponse(
            {
                "tmux_sessions": [s.to_dict() for s in sessions],
                "registry": cfg.registry.summary(),
            }
        )

    @app.post("/api/terminal/tmux/kill")
    async def kill_tmux(request: Request) -> JSONResponse:
        if cfg.is_guest_scope(request) or not _cookie_ok(cfg.cookie_signer, request.cookies):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        killed = await tmux_helper.kill_server()
        return JSONResponse({"killed": killed})

    @app.post("/api/terminal/sessions/{name}/reset")
    async def reset_session(name: str, request: Request) -> JSONResponse:
        """Kill a specific tmux session — used by the panel's X button.

        Rejects anything outside the tmux-name allowlist (same regex the
        attach handshake enforces) so a crafted name can't escape into
        a shell argument. Closes any active WS attached to this session
        first so the browser reliably reconnects with a fresh session.
        """
        if cfg.is_guest_scope(request) or not _cookie_ok(cfg.cookie_signer, request.cookies):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        if not protocol.TMUX_NAME_RE.match(name):
            return JSONResponse({"error": "invalid session name"}, status_code=400)
        # Close any WS currently attached to this session; the client's
        # reconnect loop will re-attach and get a fresh tmux session.
        await cfg.registry.close_sessions_by_name(name, reason="reset")
        killed = await tmux_helper.kill_session(name)
        return JSONResponse({"killed": killed, "session": name})

    @app.websocket("/api/ws/terminal")
    async def terminal_ws(websocket: WebSocket) -> None:
        await _handle_ws(websocket, cfg)


async def _handle_ws(websocket: WebSocket, cfg: TerminalRouterConfig) -> None:
    # 1. Scope check (defense in depth)
    if cfg.is_guest_scope(websocket):
        await websocket.close(code=4003, reason="admin only")
        return
    # 2. Cookie check — primary gate
    if not _cookie_ok(cfg.cookie_signer, websocket.cookies):
        await websocket.close(code=4401, reason="cookie required")
        return
    # 3. Origin check — defense in depth against cross-site WS
    if not _origin_ok(websocket):
        await websocket.close(code=4001, reason="bad origin")
        return

    await websocket.accept()

    session: TerminalSession | None = None
    reserved = False
    status_task: asyncio.Task[None] | None = None

    try:
        # 4. First message must be 'attach' with a valid ticket.
        try:
            first = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
        except TimeoutError:
            await websocket.close(code=4008, reason="attach timeout")
            return

        try:
            msg = protocol.parse_client_text(first)
        except ProtocolError as e:
            await websocket.send_text(protocol.encode_error("protocol", str(e)))
            await websocket.close(code=4400, reason="protocol")
            return
        if not isinstance(msg, protocol.AttachMessage):
            await websocket.send_text(protocol.encode_error("protocol", "attach required"))
            await websocket.close(code=4400, reason="no attach")
            return

        # 5. Consume the ticket — single-use.
        if not await cfg.ticket_store.consume(msg.ticket):
            await websocket.send_text(protocol.encode_error("auth"))
            await websocket.close(code=4401, reason="ticket")
            return

        # 6. Reserve capacity.
        if not cfg.registry.reserve(websocket):
            await websocket.send_text(protocol.encode_error("capacity"))
            await websocket.close(code=1013, reason="capacity")
            return
        reserved = True

        # 7. Spawn PTY.
        argv = tmux_helper.build_argv(msg.tmux_session)
        try:
            session = await TerminalSession.spawn(
                tmux_session=msg.tmux_session,
                argv=argv,
                cols=msg.cols,
                rows=msg.rows,
                ws=websocket,
                term=msg.term,
            )
        except Exception as e:
            logger.exception("failed to spawn PTY (tmux=%s): %s", msg.tmux_session, e)
            cfg.registry.rollback(websocket)
            reserved = False
            await websocket.send_text(protocol.encode_error("spawn_failed", str(e)))
            await websocket.close(code=1011, reason="spawn_failed")
            return

        cfg.registry.fulfill(websocket, session)

        # Attach a per-meeting history log (if configured). We tee every
        # PTY byte to the log file — same pattern as slides/audio — but
        # we DO NOT live-replay on attach: raw PTY bytes include terminal
        # device-attribute queries that, when fed back through xterm,
        # generate synthetic keystrokes on the reply path and end up
        # typed into the shell as literal commands ("0c1: command not
        # found"). The log stays as a durable artifact inside the meeting
        # folder, visible with `less data/meetings/<id>/terminal.log`.
        history = _open_history_log(cfg)
        if history is not None:
            session.history = history

        await websocket.send_text(
            protocol.encode_attached(
                cols=session.cols,
                rows=session.rows,
                tmux_session=session.tmux_session,
                pid=session.proc.pid,
            )
        )

        # 8. Kick off periodic status pusher.
        status_task = asyncio.create_task(_status_pusher(session, websocket))

        # 9. Kick off a watchdog that closes the WS as soon as the PTY
        # exits. Without this, a user typing `exit` in their shell leaves
        # the browser tab stuck on the dead session — the read loop is
        # parked on ws.receive() with nothing coming the other way.
        exit_watchdog = asyncio.create_task(_exit_watchdog(session, websocket))

        # 10. Main receive loop.
        try:
            await _receive_loop(session, websocket)
        finally:
            exit_watchdog.cancel()

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("terminal_ws unhandled error")
    finally:
        # Order matters: release the registry slot FIRST (sync, cancellation-safe)
        # so a coroutine cancel mid-finally can't leave a stale entry.
        if reserved or websocket in cfg.registry.items:
            cfg.registry.unregister(websocket)
        if status_task is not None:
            status_task.cancel()
        if session is not None:
            # Shield close so a cancelled WS handler still frees the PTY and child.
            try:
                await asyncio.shield(session.close(reason="ws_closed"))
            except (asyncio.CancelledError, Exception):
                # Fire-and-forget best-effort cleanup if the shield was interrupted.
                with contextlib.suppress(OSError):
                    os.close(session.master_fd)
                with contextlib.suppress(ProcessLookupError):
                    session.proc.kill()


async def _receive_loop(session: TerminalSession, websocket: WebSocket) -> None:
    while True:
        message = await websocket.receive()
        mtype = message.get("type")
        if mtype == "websocket.disconnect":
            return
        if "bytes" in message and message["bytes"] is not None:
            frame = message["bytes"]
            try:
                stdin = protocol.extract_stdin(frame)
            except ProtocolError as e:
                # Oversized / malformed — kill the connection rather than silently dropping.
                logger.info("terminal %s: bad binary frame (%s)", session.tmux_session, e)
                await websocket.close(code=1009, reason="bad frame")
                return
            if stdin:
                try:
                    await session.write_stdin(stdin)
                except ValueError as e:
                    logger.info("terminal %s: stdin rejected (%s)", session.tmux_session, e)
                    await websocket.close(code=1009, reason="bad frame")
                    return
        elif "text" in message and message["text"] is not None:
            text = message["text"]
            try:
                parsed = protocol.parse_client_text(text)
            except ProtocolError as e:
                await websocket.send_text(protocol.encode_error("protocol", str(e)))
                continue
            if isinstance(parsed, protocol.ResizeMessage):
                await session.resize(parsed.cols, parsed.rows)
            elif isinstance(parsed, protocol.AckMessage):
                session.on_client_ack(parsed.bytes_total)
            elif isinstance(parsed, dict) and parsed.get("type") == "ping":
                await websocket.send_text(protocol.encode_pong())
            elif isinstance(parsed, protocol.AttachMessage):
                # Duplicate attach — protocol violation after initial handshake.
                await websocket.send_text(protocol.encode_error("protocol", "duplicate attach"))
                await websocket.close(code=4400, reason="duplicate attach")
                return


def _open_history_log(cfg: TerminalRouterConfig) -> TerminalHistoryLog | None:
    """Resolve the per-meeting log path and open it for appending.

    Returns None if the resolver isn't configured, returns None, or the
    path can't be opened — history is best-effort and must never block
    terminal attach.
    """
    if cfg.history_path_fn is None:
        return None
    try:
        path = cfg.history_path_fn()
    except Exception as e:
        logger.warning("terminal history path resolver raised: %s", e)
        return None
    if path is None:
        return None
    try:
        log = TerminalHistoryLog(path)
        log.open()
        return log
    except Exception as e:
        logger.warning("terminal history open failed (%s): %s", path, e)
        return None


async def _exit_watchdog(session: TerminalSession, websocket: WebSocket) -> None:
    """Close the WebSocket when the PTY exits.

    The receive loop blocks on ``ws.receive()`` forever, so without this
    watchdog a user who types ``exit`` in their shell (or whose tmux
    server crashes) ends up with a silently dead WS — no reconnect
    signal, no auto-respawn, no recovery.
    """
    while not session._closing:
        try:
            await asyncio.sleep(0.25)
        except asyncio.CancelledError:
            return
    # Tell the client WHY before slamming the door.
    reason = getattr(session, "_close_reason", "pty_exited") or "pty_exited"
    try:
        await websocket.send_text(protocol.encode_bye(reason))
    except Exception:
        pass
    try:
        await websocket.close(code=1000, reason=reason[:120])
    except Exception:
        pass


async def _status_pusher(session: TerminalSession, websocket: WebSocket) -> None:
    while not session._closing:
        try:
            await asyncio.sleep(STATUS_PUSH_INTERVAL_S)
            if session._closing:
                return
            payload = protocol.encode_status(
                bytes_in=session.bytes_in,
                bytes_sent_total=session.bytes_sent_total,
                bytes_acked_total=session.bytes_acked_total,
                paused=session.paused,
                cols=session.cols,
                rows=session.rows,
            )
            await websocket.send_text(payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            return  # WS went away — silently exit, main loop will tidy up


# Avoid "unused import" warnings from linters looking at the test surface.
__all__ = [
    "TerminalRouterConfig",
    "_cookie_ok",
    "_handle_ws",
    "_origin_ok",
    "json",  # re-exported convenience for tests that want raw JSON comparisons
    "register_terminal_routes",
]
