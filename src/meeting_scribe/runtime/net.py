"""Network startup helpers — listener sockets, dual-uvicorn, sd_notify.

* sd_notify (`_notify_systemd`) speaks the `sd_notify(3)` datagram protocol
  directly so we don't need `systemd-python`/`sdnotify` as a dep.

* `_make_tcp_socket` creates a TCP listener with optional `IP_FREEBIND`
  so we can pre-bind `10.42.0.1` before nmcli assigns the AP IP.

* `_NoSignalServer` is a `uvicorn.Server` variant that skips its own
  signal handlers so the outer orchestrator owns Ctrl-C draining.

* `_serve_two_apps` runs the canonical HTTPS app + the port-80
  captive HTTP sub-app on a single AP-bound listener.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket

import uvicorn

logger = logging.getLogger(__name__)


def _notify_systemd(message: str) -> None:
    """Send a `sd_notify(3)` message if running under `Type=notify`. No-op
    when `$NOTIFY_SOCKET` is unset.
    """
    sock_path = os.environ.get("NOTIFY_SOCKET")
    if not sock_path:
        return
    try:
        addr: bytes | str
        if sock_path.startswith("@"):
            addr = "\0" + sock_path[1:]
        else:
            addr = sock_path
        s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM | socket.SOCK_CLOEXEC)
        try:
            s.sendto(message.encode("utf-8"), addr)
        finally:
            s.close()
    except OSError as e:
        logger.warning("sd_notify(%r) failed: %r", message, e)


def _make_tcp_socket(host: str, port: int, freebind: bool = False) -> socket.socket:
    """Create and bind a TCP listening socket.

    `freebind=True` sets `IP_FREEBIND` so the bind succeeds even if the
    target IP isn't yet assigned to any local interface — used for the
    AP-bound listener on `10.42.0.1` which only exists after `nmcli`
    brings the AP up.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if freebind:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_FREEBIND, 1)
    sock.bind((host, port))
    sock.listen(128)
    sock.setblocking(False)
    return sock


class _NoSignalServer(uvicorn.Server):
    """uvicorn.Server variant that skips its own SIGINT/SIGTERM handlers.

    When two Servers run in one process, only the last installed signal
    handler wins. We disable per-Server handlers so the outer orchestrator
    can install one handler that drains both instances together.
    """

    def install_signal_handlers(self) -> None:
        return None


async def _serve_two_apps(
    main_server: _NoSignalServer,
    main_sockets: list,
    captive_server: _NoSignalServer,
    captive_sockets: list,
) -> None:
    """Run the canonical HTTPS app + the captive HTTP sub-app.

    Both servers run in one process so in-process globals stay
    shared, but only the main HTTPS app runs the FastAPI lifespan;
    the captive sub-app is stateless.

    Startup order: main runs first (lifespan brings backends up,
    setup-state reconcile, AP up); the captive HTTP sub-app starts
    only after the main app is ready so a port-80 redirect can't
    point at a half-initialized canonical origin.
    """
    import signal

    loop = asyncio.get_running_loop()

    def _request_shutdown() -> None:
        main_server.should_exit = True
        captive_server.should_exit = True

    def _request_config_reload() -> None:
        from meeting_scribe import runtime_config
        from meeting_scribe.server_support.page_cache import reload_cached_html

        try:
            runtime_config.reload_from_disk()
            logger.info(
                "runtime-config reloaded on SIGHUP: %s", runtime_config.instance().as_dict()
            )
        except Exception:
            logger.exception("runtime-config reload failed")
        try:
            # Re-read static/*.html from disk so edits to index.html /
            # portal.html / reader.html / etc. take effect without a
            # full server restart. Without this, the page cache loaded
            # at startup served stale HTML for the lifetime of the
            # process — the 2026-05-07 BT-settings-missing report.
            if reload_cached_html():
                logger.info("page-cache HTML reloaded on SIGHUP")
        except Exception:
            logger.exception("page-cache reload failed")

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _request_shutdown)
    loop.add_signal_handler(signal.SIGHUP, _request_config_reload)

    main_task = asyncio.create_task(
        main_server.serve(sockets=main_sockets),
        name="main-uvicorn",
    )

    while not main_server.started:
        if main_task.done():
            await main_task
            return
        await asyncio.sleep(0.05)

    captive_task = asyncio.create_task(
        captive_server.serve(sockets=captive_sockets),
        name="captive-uvicorn",
    )

    try:
        await asyncio.gather(main_task, captive_task)
    finally:
        main_server.should_exit = True
        captive_server.should_exit = True


async def _serve_three_apps(
    main_server: _NoSignalServer,
    main_sockets: list,
    captive_server: _NoSignalServer | None,
    captive_sockets: list,
    kiosk_server: _NoSignalServer | None,
    kiosk_sockets: list,
) -> None:
    """Run the canonical HTTPS app + the captive HTTP sub-app + the
    kiosk loopback HTTP listener.

    All three share one FastAPI instance, so middlewares, lifespan,
    and in-process state are unified. The captive sub-app uses a
    distinct app instance (no lifespan); the kiosk listener serves
    the same canonical app but is bound to ``127.0.0.1`` and only
    handles routes guarded by ``require_kiosk_listener``.

    Either ``captive_server`` or ``kiosk_server`` may be ``None`` (the
    captive listener is disabled by ``SCRIBE_DISABLE_CAPTIVE_HTTP=1``
    for dev sidecars; the kiosk listener is disabled when the
    appliance hasn't been bootstrapped with the kiosk feature yet).
    """
    import signal

    loop = asyncio.get_running_loop()

    def _request_shutdown() -> None:
        main_server.should_exit = True
        if captive_server is not None:
            captive_server.should_exit = True
        if kiosk_server is not None:
            kiosk_server.should_exit = True

    def _request_config_reload() -> None:
        from meeting_scribe import runtime_config
        from meeting_scribe.server_support.page_cache import reload_cached_html

        try:
            runtime_config.reload_from_disk()
            logger.info(
                "runtime-config reloaded on SIGHUP: %s", runtime_config.instance().as_dict()
            )
        except Exception:
            logger.exception("runtime-config reload failed")
        try:
            if reload_cached_html():
                logger.info("page-cache HTML reloaded on SIGHUP")
        except Exception:
            logger.exception("page-cache reload failed")

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _request_shutdown)
    loop.add_signal_handler(signal.SIGHUP, _request_config_reload)

    main_task = asyncio.create_task(
        main_server.serve(sockets=main_sockets),
        name="main-uvicorn",
    )

    while not main_server.started:
        if main_task.done():
            await main_task
            return
        await asyncio.sleep(0.05)

    tasks: list[asyncio.Task] = [main_task]
    if captive_server is not None:
        tasks.append(
            asyncio.create_task(
                captive_server.serve(sockets=captive_sockets),
                name="captive-uvicorn",
            )
        )
    if kiosk_server is not None:
        tasks.append(
            asyncio.create_task(
                kiosk_server.serve(sockets=kiosk_sockets),
                name="kiosk-uvicorn",
            )
        )

    try:
        await asyncio.gather(*tasks)
    finally:
        main_server.should_exit = True
        if captive_server is not None:
            captive_server.should_exit = True
        if kiosk_server is not None:
            kiosk_server.should_exit = True
