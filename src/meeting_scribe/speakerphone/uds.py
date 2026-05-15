"""Unix-domain socket server for the internal speakerphone namespace.

Spawned alongside the public TCP listener by ``server.py``'s lifespan.
A separate FastAPI app holds *only* the internal router — the
``/api/internal/speakerphone/*`` namespace is unreachable from TCP by
construction. Authentication = filesystem permissions on the socket
file (0600, user-owned). A connecting process running as a different
UID gets EACCES.

Default path: ``$XDG_RUNTIME_DIR/meeting-scribe.sock``. Honors the
``MEETING_SCRIBE_UDS_PATH`` env override so tests can redirect to a
tmpdir.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import stat
from pathlib import Path

from fastapi import FastAPI

from meeting_scribe.routes.internal_speakerphone import router as _internal_router

logger = logging.getLogger(__name__)


def default_uds_path() -> Path:
    override = os.environ.get("MEETING_SCRIBE_UDS_PATH")
    if override:
        return Path(override)
    base = os.environ.get("XDG_RUNTIME_DIR")
    if base:
        return Path(base) / "meeting-scribe.sock"
    # Fallback for setups where XDG_RUNTIME_DIR isn't set (e.g. running
    # outside a logged-in session). Pin to a per-user tmpdir so two
    # users on the same box don't fight for the same socket.
    return Path("/tmp") / f"meeting-scribe-{os.getuid()}.sock"


def build_app() -> FastAPI:
    """Construct the FastAPI app for the UDS listener.

    Holds only the internal router. Crucially, this app is **not**
    imported or included by the TCP server, so a misroute can't
    accidentally reach the internal namespace.
    """
    app = FastAPI(
        title="meeting-scribe (internal UDS)",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.include_router(_internal_router)
    return app


def _check_runtime_dir(path: Path) -> None:
    """Refuse to bind if the runtime dir is world-writable.

    A 0777 runtime dir lets a hostile local user replace our socket
    file between bind() and chmod(); that's an unreasonable
    environment. ``XDG_RUNTIME_DIR`` is 0700 by spec.
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    mode = parent.stat().st_mode
    world_writable = bool(mode & stat.S_IWOTH)
    if world_writable:
        raise RuntimeError(
            f"refusing to bind UDS in world-writable directory: {parent} "
            f"(mode={oct(mode & 0o777)})",
        )


def _create_socket(path: Path) -> socket.socket:
    """Bind the UDS, set 0600 perms, return the listening socket.

    Unlinks any prior socket file owned by the current user; refuses to
    proceed if a stale socket exists but is owned by someone else
    (someone could be racing us — abort rather than overwrite).
    """
    _check_runtime_dir(path)
    if path.exists():
        st = path.stat()
        if st.st_uid != os.getuid():
            raise RuntimeError(
                f"stale UDS at {path} owned by uid={st.st_uid}; refusing to unlink",
            )
        if not stat.S_ISSOCK(st.st_mode):
            raise RuntimeError(f"{path} exists but is not a socket")
        path.unlink()

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(path))
    # Lock down permissions BEFORE we listen() so a tiny race window
    # between bind and chmod can't be exploited.
    os.chmod(path, 0o600)
    sock.listen(8)
    sock.setblocking(False)
    return sock


async def serve(
    *,
    path: Path | None = None,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Run the UDS uvicorn server until ``stop_event`` is set.

    Yields control back to the caller on the event so shutdown can
    drain gracefully (matching the pattern used by the main TCP
    listener in ``runtime/net.py``).
    """
    import uvicorn

    target_path = path or default_uds_path()
    sock = _create_socket(target_path)
    logger.info("speakerphone UDS listening on %s (0600)", target_path)
    stop_event = stop_event or asyncio.Event()

    config = uvicorn.Config(
        app=build_app(),
        host=None,
        port=None,
        loop="asyncio",
        log_level="info",
        access_log=False,
        # The socket is already bound; uvicorn picks it up via the fd.
        fd=sock.fileno(),
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # outer orchestrator owns Ctrl-C

    serve_task = asyncio.create_task(server.serve())
    try:
        await stop_event.wait()
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(serve_task, timeout=10)
        except TimeoutError:
            logger.warning("speakerphone UDS uvicorn did not exit within 10s; cancelling")
            serve_task.cancel()
        # Clean up the socket file on shutdown so a stale file doesn't
        # confuse the next startup. Best-effort — a crashed process
        # leaves the file behind and the next ``serve()`` will unlink
        # it (see _create_socket).
        try:
            target_path.unlink()
        except FileNotFoundError:
            pass  # socket file already gone — nothing to clean up
        except OSError as e:
            logger.warning("UDS cleanup failed: %r", e)
