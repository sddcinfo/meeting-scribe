"""Crash detection + sanitised crash metadata for ``/api/status``.

Two layers:

1. **Sanitised crash record**. ``_record_crash`` distills any
   unhandled exception down to a timestamp, a component label, and
   an opaque sha256 fingerprint. The full traceback hits the server
   log; the ``/api/status`` payload only sees the opaque code so
   internal type names + filenames don't leak to a guest device.

2. **Hook installation**. ``_install_crash_hooks`` wires
   ``sys.excepthook`` for sync errors and the asyncio loop's
   exception handler for async ones. Routine peer-side noise (TLS
   handshake rejections, ``ConnectionResetError``, WebSocket
   disconnects) is demoted to debug so it never shows up as a
   "crash" red-dot in the UI.

Pulled out of ``server.py`` so the upcoming ``runtime/health.py``
module can build on top of these without circling back through the
server module.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import sys
import time
import traceback as _tb

logger = logging.getLogger(__name__)


# Sanitised crash metadata. Populated by the excepthook / asyncio
# exception handler on any unhandled exception. Exposed via /api/status
# with opaque error code only — no type names, no messages, no
# tracebacks. Full details live in the server log. [P2-1-i2]
_crash_state: dict | None = None


def _sanitised_crash_state() -> dict | None:
    """Return the sanitised crash metadata dict for /api/status."""
    return _crash_state


def _record_crash(component: str, exc: BaseException) -> None:
    """Record a crash in a way that doesn't leak internals to /api/status.

    The full traceback is logged at CRITICAL level to stdout; only a
    timestamp, component name, and opaque sha256 code land in
    ``_crash_state``. The UI red-dots the server pill based on the
    ``state`` field — nothing more. [P2-1-i2]
    """
    global _crash_state

    try:
        tb_list = _tb.extract_tb(exc.__traceback__)
        if tb_list:
            frame = tb_list[-1]
            fingerprint = f"{type(exc).__name__}|{frame.filename}|{frame.lineno}"
        else:
            fingerprint = f"{type(exc).__name__}|<no traceback>"
        opaque = hashlib.sha256(fingerprint.encode()).hexdigest()
    except Exception:
        opaque = "unknown"

    logger.critical(
        "crash in %s: %s",
        component,
        exc,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    _crash_state = {
        "ts": round(time.monotonic(), 2),
        "component": component,
        "state": "crashed",
        "code": opaque,
    }


def _reset_crash_state() -> None:
    """Reset the crash state (used by tests)."""
    global _crash_state
    _crash_state = None


def _install_crash_hooks() -> None:
    """Install sys.excepthook + asyncio exception handler."""

    def _sys_hook(exc_type, exc, tb):
        _record_crash("sync", exc)

    sys.excepthook = _sys_hook

    def _async_hook(loop, context):
        exc = context.get("exception")
        task = context.get("task")
        task_name = task.get_name() if task else "unknown"

        # WebSocket disconnects (ping timeout, client gone) are normal network
        # events — don't record them as server crashes.
        if exc is not None:
            exc_name = type(exc).__name__
            if exc_name in ("ConnectionClosedError", "ConnectionClosedOK", "WebSocketDisconnect"):
                logger.debug("WS disconnect (not a crash): %s", exc)
                return
            # TLS handshake rejections from clients that don't trust our
            # self-signed admin cert (Safari/iOS, curl without -k, etc.)
            # surface here as asyncio `_accept_connection2` errors. They
            # are a *client* decision, not a server fault — demote to
            # debug so we don't log CRITICAL "crash in other" every time
            # a browser tab re-opens on a device that hasn't trusted
            # the cert yet.
            import ssl as _ssl

            if isinstance(exc, _ssl.SSLError):
                logger.debug("TLS handshake rejected by client (not a crash): %s", exc)
                return
            # Peer-side TCP resets during accept/read (browser tab closed
            # mid-request, mobile connection flipped, hotspot disconnect,
            # load balancer health-probe) bubble through the asyncio
            # exception handler. They never indicate a server fault, so
            # never a "crash in other" banner.
            if isinstance(exc, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
                logger.debug("Peer connection reset (not a crash): %s", exc)
                return
            # SSL handshake timeouts ("SSL handshake is taking longer
            # than N seconds: aborting the connection") are raised as
            # TimeoutError from asyncio's accept path. Client-driven,
            # not a server crash — same reasoning as SSLError above.
            if isinstance(exc, TimeoutError) and "handshake" in str(exc).lower():
                logger.debug("TLS handshake timeout (not a crash): %s", exc)
                return

        component = "other"
        if "tts-worker" in task_name:
            component = "tts_worker"
        elif "loop-lag" in task_name:
            component = "loop_lag_monitor"
        elif "health-evaluator" in task_name:
            component = "health_evaluator"
        elif "catchup" in task_name:
            component = "speaker_catchup"
        if exc is not None:
            _record_crash(component, exc)
        else:
            logger.warning("asyncio exception (no exc): %s", context.get("message"))

    try:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(_async_hook)
        # When a single sync callback blocks the loop for > 500 ms,
        # asyncio's debug mode logs the slow callback's repr + location.
        # This is how we pinpoint which periodic loop is causing the
        # 2.5 s periodic stall seen on the baseline. Cheap to enable.
        loop.slow_callback_duration = 0.5
        # `slow_callback_duration` is only honoured when the loop is in
        # debug mode. Without this line the threshold is silently
        # ignored and we never see the culprit. Also force the
        # asyncio logger to WARNING so the slow-callback message is
        # not buried below the default level.
        loop.set_debug(True)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
    except RuntimeError:
        # Not inside a running loop — the hook will be installed by
        # whichever task calls _install_crash_hooks at startup.
        pass
