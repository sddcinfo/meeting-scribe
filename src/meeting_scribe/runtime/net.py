"""Network startup helpers — listener sockets, dual-uvicorn, sd_notify, mgmt IP detection.

The split between admin (HTTPS, LAN-only) and guest (HTTP,
hotspot) listeners — both backed by the same FastAPI app — happens
here. Five concerns:

* **sd_notify**. ``_notify_systemd`` speaks the
  ``sd_notify(3)`` datagram protocol directly so we don't need
  ``systemd-python``/``sdnotify`` as a dep.

* **Management IP detection** (``_detect_management_ip``,
  ``_detect_management_ip_via_nm``). Three-tier cascade:
  ``SCRIBE_MANAGEMENT_IP`` env override → ``ip route get 1.1.1.1``
  with retry budget → NetworkManager's sole-active-ethernet IP →
  ``127.0.0.1`` fallback (degraded mode).

* **TCP socket** (``_make_tcp_socket``). Plain ``SO_REUSEADDR`` +
  optional ``IP_FREEBIND`` so we can pre-bind ``10.42.0.1`` before
  nmcli assigns the AP IP.

* **Dual-uvicorn lifecycle** (``_serve_dual``,
  ``_NoSignalServer``). Two ``uvicorn.Server`` instances share one
  app; the admin server runs the FastAPI lifespan and starts first,
  guest server (``lifespan="off"``) starts only once admin is up.
  ``_NoSignalServer`` skips uvicorn's per-Server signal handlers so
  one outer Ctrl-C handler can drain both cleanly.

* **LAN recovery** (``_wait_for_management_ip``). Background task
  that hot-adds an admin LAN socket when the network appears,
  driven by the same detection cascade. Used when boot landed in
  degraded mode.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import subprocess
import time

import uvicorn

logger = logging.getLogger(__name__)


def _notify_systemd(message: str) -> None:
    """Send a ``sd_notify(3)`` message to the service manager if we are
    running under ``Type=notify``. No-op (and no dependency on
    ``systemd-python``/``sdnotify``) when ``$NOTIFY_SOCKET`` is unset,
    e.g. in foreground dev runs.

    Implements the datagram protocol directly via the Unix socket so
    there's nothing to install: the whole contract is "connect to the
    socket at ``$NOTIFY_SOCKET`` and send one UTF-8 message".
    """
    sock_path = os.environ.get("NOTIFY_SOCKET")
    if not sock_path:
        return
    try:
        addr: bytes | str
        if sock_path.startswith("@"):
            # Abstract Linux socket.
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


def _detect_management_ip_via_nm() -> str | None:
    """Query NetworkManager for the IPv4 address of the active wired connection.

    Positive selection: only considers ``802-3-ethernet`` connections, so
    VPN tunnels (``wg0``, ``tun0``), wireless hotspots, docker bridges,
    USB tethering, and mobile broadband are never selected.

    Returns the IP string if exactly one active ethernet connection with an
    IPv4 address is found, otherwise ``None`` (ambiguous or unavailable).
    """
    try:
        cons = subprocess.run(
            ["nmcli", "-t", "-f", "TYPE,DEVICE", "connection", "show", "--active"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None

    # Filter to wired ethernet connections.
    ethernet_devices: list[str] = []
    for line in cons.splitlines():
        parts = line.split(":")
        if len(parts) >= 2 and parts[0] == "802-3-ethernet" and parts[1]:
            ethernet_devices.append(parts[1])

    if len(ethernet_devices) != 1:
        # Zero or multiple wired connections — refuse to guess.
        if ethernet_devices:
            logger.info(
                "NM: %d active ethernet connections — refusing to guess management IP",
                len(ethernet_devices),
            )
        return None

    device = ethernet_devices[0]
    try:
        dev_info = subprocess.run(
            ["nmcli", "-t", "-f", "IP4.ADDRESS", "device", "show", device],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None

    # Parse "IP4.ADDRESS[1]:192.168.8.153/24" → "192.168.8.153"
    for line in dev_info.splitlines():
        if line.startswith("IP4.ADDRESS"):
            _, _, addr = line.partition(":")
            addr = addr.strip()
            if "/" in addr:
                addr = addr.split("/")[0]
            if addr:
                logger.info("NM: management IP from ethernet device %s: %s", device, addr)
                return addr

    return None


def _detect_management_ip() -> str:
    """Detect the management IPv4 address via a three-tier cascade.

    1. ``SCRIBE_MANAGEMENT_IP`` env override (tests, unusual layouts).
    2. ``ip -4 route get 1.1.1.1`` with retry budget — the kernel's
       preferred source address for outbound traffic.
    3. NetworkManager positive selection — the IPv4 of the sole active
       ``802-3-ethernet`` connection.
    4. Fallback to ``127.0.0.1`` — admin listener is localhost-only
       (degraded mode). The guest hotspot portal is unaffected.

    Never raises. Returns ``"127.0.0.1"`` as the final fallback so the
    server always starts, even without a network.
    """
    override = os.environ.get("SCRIBE_MANAGEMENT_IP", "").strip()
    if override:
        return override

    try:
        budget = max(1, int(os.environ.get("SCRIBE_MGMT_IP_WAIT", "30")))
    except ValueError:
        budget = 30

    # Tier 2: ip route get with retry budget.
    for attempt in range(1, budget + 1):
        try:
            out = subprocess.run(
                ["ip", "-4", "route", "get", "1.1.1.1"],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            ).stdout
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            if attempt < budget:
                logger.info(
                    "waiting for default route (attempt %d/%d): %s",
                    attempt,
                    budget,
                    e,
                )
                time.sleep(1)
                continue
            logger.info("ip route get exhausted after %d attempts: %s", budget, e)
            break

        tokens = out.split()
        if "src" in tokens:
            if attempt > 1:
                logger.info("management IP detected after %d attempts", attempt)
            return tokens[tokens.index("src") + 1]
        if attempt < budget:
            logger.info(
                "no 'src' field in ip route get output yet (attempt %d/%d)",
                attempt,
                budget,
            )
            time.sleep(1)
            continue
        logger.info("no 'src' field in ip route get output: %r", out)
        break

    # Tier 3: NetworkManager ethernet lookup.
    nm_ip = _detect_management_ip_via_nm()
    if nm_ip:
        return nm_ip

    # Tier 4: localhost fallback — degraded mode.
    logger.warning(
        "no management IP detected via route or NetworkManager — "
        "admin listener will be localhost-only (degraded mode)"
    )
    return "127.0.0.1"


def _make_tcp_socket(host: str, port: int, freebind: bool = False) -> socket.socket:
    """Create and bind a TCP listening socket.

    When ``freebind=True`` sets ``IP_FREEBIND`` so the bind succeeds even
    if the target IP isn't yet assigned to any local interface. We use
    that for the guest listener on the hotspot AP IP (``10.42.0.1``)
    which only exists after ``nmcli`` brings the AP up.
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
    handler wins. We disable per-Server handlers and install a single
    outer handler in ``main()`` that sets ``should_exit = True`` on both
    instances together, so Ctrl-C cleanly drains both listeners.
    """

    def install_signal_handlers(self) -> None:
        return None


async def _wait_for_management_ip(
    admin_server: _NoSignalServer,
    port: int,
    poll_interval: int = 15,
) -> None:
    """Background task: hot-add an admin LAN socket when the network appears.

    Launched only when ``_detect_management_ip()`` returned ``127.0.0.1``
    at startup (degraded mode). Polls the same detection cascade every
    ``poll_interval`` seconds with a minimal inner budget
    (``SCRIBE_MGMT_IP_WAIT=1``) to avoid blocking the event loop.

    When a non-loopback IP is found, creates a new TCP listener socket,
    wires it into the running admin uvicorn server using the same protocol
    factory pattern as uvicorn's own ``startup()``, and exits (one-shot).
    """

    def _detect_quick() -> str:
        """Run the detection cascade with a 1-attempt budget."""
        saved = os.environ.get("SCRIBE_MGMT_IP_WAIT")
        try:
            os.environ["SCRIBE_MGMT_IP_WAIT"] = "1"
            return _detect_management_ip()
        finally:
            if saved is None:
                os.environ.pop("SCRIBE_MGMT_IP_WAIT", None)
            else:
                os.environ["SCRIBE_MGMT_IP_WAIT"] = saved

    try:
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(poll_interval)

            # Run detection in a thread so subprocess calls don't block
            # the event loop (worst case: ip 3s + 2× nmcli 5s = ~13s).
            ip = await loop.run_in_executor(None, _detect_quick)
            if ip == "127.0.0.1":
                continue

            # Found a real management IP — hot-add the admin socket.
            try:
                sock = _make_tcp_socket(ip, port)
            except OSError as e:
                logger.warning(
                    "LAN admin socket bind failed for %s:%d: %s — will retry",
                    ip,
                    port,
                    e,
                )
                continue

            uv_config = admin_server.config
            # uvicorn's http_protocol_class is typed as `type[Protocol]` on
            # the uvicorn.Config surface but the concrete call site takes
            # config/server_state/app_state kwargs (see uvicorn.protocols.*).
            # mypy can't see through the Protocol aliasing.
            server = await loop.create_server(
                lambda: uv_config.http_protocol_class(  # type: ignore[call-arg]
                    config=uv_config,
                    server_state=admin_server.server_state,
                    app_state=admin_server.lifespan.state,
                ),
                sock=sock,
                ssl=uv_config.ssl,
                backlog=uv_config.backlog,
            )
            admin_server.servers.append(server)
            logger.info("LAN admin listener recovered: https://%s:%d", ip, port)
            return

    except asyncio.CancelledError:
        return


async def _serve_dual(
    admin_server: _NoSignalServer,
    admin_sockets: list,
    guest_server: _NoSignalServer,
    guest_sockets: list,
    *,
    deferred_admin_bind: tuple[int, int] | None = None,
) -> None:
    """Run admin + guest uvicorn servers sharing one FastAPI app.

    Startup order matters: admin runs the FastAPI lifespan (model loads,
    backend wiring, WiFi regdomain checks). Guest has ``lifespan="off"``
    and must only start accepting connections **after** admin's startup
    is complete, otherwise a fast phone hitting the guest portal could
    land in an app with uninitialised globals.

    When ``deferred_admin_bind`` is set (``(port, poll_interval)``), a
    background task polls for the management IP and hot-adds an admin LAN
    socket when the network appears. This handles the degraded-start case
    where no LAN IP was available at boot.
    """
    import signal

    loop = asyncio.get_running_loop()

    def _request_shutdown() -> None:
        admin_server.should_exit = True
        guest_server.should_exit = True

    def _request_config_reload() -> None:
        """Re-read runtime-config from disk.

        Bound to SIGHUP so ``meeting-scribe config reload`` (and the
        Phase 7 rollback procedure) can flip ``translate_url`` /
        ``slide_translate_url`` / ``slide_use_json_schema`` live.  The
        translate backend re-reads on every request, so no restart is
        needed after the reload fires.
        """
        from meeting_scribe import runtime_config

        try:
            runtime_config.reload_from_disk()
            logger.info(
                "runtime-config reloaded on SIGHUP: %s", runtime_config.instance().as_dict()
            )
        except Exception:
            logger.exception("runtime-config reload failed")

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _request_shutdown)
    loop.add_signal_handler(signal.SIGHUP, _request_config_reload)

    admin_task = asyncio.create_task(
        admin_server.serve(sockets=admin_sockets),
        name="admin-uvicorn",
    )

    while not admin_server.started:
        if admin_task.done():
            # Admin died during startup — surface the exception.
            await admin_task
            return
        await asyncio.sleep(0.05)

    # Launch LAN recovery task if we started in degraded mode.
    recovery_task: asyncio.Task | None = None
    if deferred_admin_bind is not None:
        port, interval = deferred_admin_bind
        recovery_task = asyncio.create_task(
            _wait_for_management_ip(admin_server, port, interval),
            name="lan-recovery",
        )

    guest_task = asyncio.create_task(
        guest_server.serve(sockets=guest_sockets),
        name="guest-uvicorn",
    )

    try:
        await asyncio.gather(admin_task, guest_task)
    finally:
        # If one server crashed, make sure the other drains too.
        admin_server.should_exit = True
        guest_server.should_exit = True
        if recovery_task is not None and not recovery_task.done():
            recovery_task.cancel()
            try:
                await recovery_task
            except asyncio.CancelledError:
                pass
