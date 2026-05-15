"""Per-device mDNS hostname publication via ``avahi-publish-address``.

The HTTPS leaf cert already carries DNS SANs for two device-bound
names — ``meeting-scribe-<id4>.local`` and ``meeting-<pin>.local``
(see :func:`meeting_scribe.cli._common._required_leaf_dns_sans`). For
those names to actually resolve on the AP subnet, the avahi daemon
needs to advertise them. The default avahi-daemon only advertises
the system hostname (``promaxgb10-f426.local``), so we spawn an
``avahi-publish-address`` child process per device-bound alias.

Each child is held open for the lifetime of the server process: the
publication disappears the moment the process exits, which is exactly
the behavior we want during factory-reset / appliance-id rotation.

Static ``/etc/avahi/hosts`` entries would be simpler but caused
``Local name collision`` errors when the same IP was already
registered under the system hostname. ``avahi-publish-address``
publishes a *separate* RR, which sidesteps that collision.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from collections.abc import Iterable

logger = logging.getLogger(__name__)

# IP the aliases resolve to. The AP iface (wlP9s9) carries 10.42.0.1;
# avahi will only respond to queries received on the interface that
# carries this IP, so clients on the AP subnet see the right answer.
_DEFAULT_AP_IP = os.environ.get("SCRIBE_AP_IP", "10.42.0.1")


def _publisher_argv(name: str, ip: str) -> list[str]:
    # ``-R`` reuses the resolver socket to avoid burning a port per
    # publication. The publisher runs forever; the parent terminates
    # it on shutdown.
    return ["avahi-publish-address", "-R", name, ip]


async def publish_aliases(
    names: Iterable[str], *, ip: str = _DEFAULT_AP_IP
) -> list[asyncio.subprocess.Process]:
    """Spawn one ``avahi-publish-address`` child per name in ``names``.

    Returns the spawned :class:`asyncio.subprocess.Process` handles so
    the caller (lifespan) can keep them alive + terminate on shutdown.

    Silent / best-effort: any individual publication failure logs a
    warning and skips that name; the rest of the system still works
    via the IP directly (``https://10.42.0.1/``).
    """
    children: list[asyncio.subprocess.Process] = []
    for name in names:
        try:
            child = await asyncio.create_subprocess_exec(
                *_publisher_argv(name, ip),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except (FileNotFoundError, PermissionError) as exc:
            logger.warning("mdns: avahi-publish-address spawn failed for %s: %s", name, exc)
            continue
        # Give the child a brief window to complete its initial
        # registration; if it fails fast (collision, daemon down), we
        # surface a warning so the operator can see why mDNS doesn't
        # resolve.
        try:
            await asyncio.wait_for(asyncio.shield(_probe_publish(child, name)), timeout=2.0)
        except TimeoutError:
            # Successful publication: the child stays alive
            # indefinitely. Timeout here means it didn't exit
            # immediately, which is what we want.
            pass
        children.append(child)
    return children


async def _probe_publish(child: asyncio.subprocess.Process, name: str) -> None:
    """Wait briefly for the publisher; raise on early exit so the
    caller can log a meaningful warning."""
    rc = await child.wait()
    # Reaching here means the publisher exited fast — bad sign.
    err = ""
    if child.stderr is not None:
        try:
            err_bytes = await child.stderr.read()
            err = err_bytes.decode(errors="replace").strip()
        except Exception:
            err = "<unreadable>"
    logger.warning("mdns: avahi-publish-address(%s) exited rc=%d stderr=%s", name, rc, err[:200])


async def stop_aliases(children: Iterable[asyncio.subprocess.Process]) -> None:
    """Terminate every publisher we spawned. Safe to call repeatedly."""
    for child in children:
        if child.returncode is not None:
            continue
        try:
            child.terminate()
        except ProcessLookupError:
            continue
    # Give them a moment to exit cleanly, then SIGKILL stragglers.
    for child in children:
        if child.returncode is not None:
            continue
        try:
            await asyncio.wait_for(child.wait(), timeout=2.0)
        except TimeoutError:
            try:
                child.kill()
            except ProcessLookupError:
                pass  # avahi-publish already exited between the wait timeout and the kill
