"""Per-client captive-gateway allowlist (ipset CRUD).

Two Linux ipsets back the captive flow:

* ``ms-allowed-admins`` (hash:ip) — clients that successfully signed in
  via ``POST /api/admin/authorize``. The captive FORWARD rule lets
  these IPs reach the upstream WAN.
* ``ms-allowed-guests`` (hash:ip) — clients that entered the guest PIN.
  These can reach the box's meeting features but NOT the upstream WAN.
  Membership is enough to satisfy ``_is_captive_acked`` so OS captive
  sheets dismiss.

The module is deliberately small and idempotent: ``ensure_sets()``
creates both sets with ``-exist``, ``add_*``/``remove_*`` swallow
already-in / not-present errors. Every shellout goes through
:data:`IPSET_RUNNER` so tests can inject a fake.

If the ``ipset`` binary is unavailable at runtime (a non-customer image
that skipped the bootstrap apt install), every function logs a single
warning and behaves as a no-op. The existing :data:`wan_egress_mode`
posture stays in force — captive mode degrades to gateway-like
behavior with no per-IP gating, rather than bricking the AP entirely.
The operator guide carries the ``apt install ipset`` remediation.

See ``docs/plans/wifi-wan-gateway.md`` Phase H.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import shutil
import subprocess
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


ADMIN_SET = "ms-allowed-admins"
GUEST_SET = "ms-allowed-guests"

# NM's shared-mode dnsmasq lease file. Parsed by the GC loop to drop
# ipset entries whose lease has expired. Best-effort: a missing file
# means we skip the tick.
LEASES_FILE = "/var/lib/NetworkManager/dnsmasq-wlP9s9.leases"

_IPSET_TIMEOUT_SHORT = 5.0


# ── Subprocess seam (test injection point) ──────────────────────


RunnerFn = Callable[[list[str], float], Awaitable[subprocess.CompletedProcess]]


async def _default_run(argv: list[str], timeout: float) -> subprocess.CompletedProcess:
    """Shell out via ``sudo`` so the systemd-managed service can mutate
    kernel state. ``bradlay`` has ``NOPASSWD: ALL`` on the GB10, mirroring
    the AP-side ``wifi.py`` pattern. Tests bypass this entirely by
    replacing :data:`IPSET_RUNNER` directly."""
    loop = asyncio.get_running_loop()
    if argv and argv[0] == "ipset":
        argv = ["sudo", *argv]

    def _blocking() -> subprocess.CompletedProcess:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )

    return await loop.run_in_executor(None, _blocking)


IPSET_RUNNER: RunnerFn = _default_run


async def _run(
    argv: list[str], *, timeout: float = _IPSET_TIMEOUT_SHORT
) -> subprocess.CompletedProcess:
    return await IPSET_RUNNER(argv, timeout)


# ── Availability check ─────────────────────────────────────────


_AVAILABILITY_WARNED = False


def is_available() -> bool:
    """Return True iff the ``ipset`` binary is on PATH.

    Logs a single warning the first time the binary is missing — the
    captive flow downgrades to gateway-like behavior in that case
    rather than failing AP bring-up.
    """
    global _AVAILABILITY_WARNED
    if shutil.which("ipset"):
        return True
    if not _AVAILABILITY_WARNED:
        logger.warning(
            "ipset binary not on PATH — captive-gateway per-IP allowlist "
            "is disabled. Install with `sudo apt install ipset` and "
            "restart meeting-scribe. AP→WAN forwarding falls back to "
            "the wan_egress_mode posture (block / gateway).",
        )
        _AVAILABILITY_WARNED = True
    return False


def _is_valid_v4(ip: str) -> bool:
    """Reject malformed IPs at the boundary so we never hand garbage
    to ``ipset add``. IPv4 only — captive gating is v4-only in v1
    (per Phase G, v6 is off on AP and STA)."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError, TypeError:
        return False
    return isinstance(addr, ipaddress.IPv4Address)


# ── Set lifecycle ──────────────────────────────────────────────


async def ensure_sets() -> bool:
    """Idempotently create both allowlist ipsets.

    Returns True iff both sets exist after the call. False signals the
    captive firewall branch should not be applied this round (caller
    falls back to non-captive gateway rules).
    """
    if not is_available():
        return False
    for name in (ADMIN_SET, GUEST_SET):
        proc = await _run(["ipset", "create", name, "hash:ip", "-exist"])
        if proc.returncode != 0:
            logger.warning(
                "ipset create %s failed rc=%d stderr=%s",
                name,
                proc.returncode,
                proc.stderr.strip()[:200],
            )
            return False
    return True


async def destroy_sets() -> None:
    """Best-effort: remove both allowlist ipsets.

    Called at AP teardown. Failures (set in use, set missing) are
    swallowed — the next bring-up will ``ensure_sets()`` idempotently.
    """
    if not is_available():
        return
    for name in (ADMIN_SET, GUEST_SET):
        await _run(["ipset", "destroy", name])


# ── Per-set operations ────────────────────────────────────────


async def _add_to(name: str, ip: str) -> bool:
    if not _is_valid_v4(ip):
        logger.warning("ipset add %s: refusing invalid IP %r", name, ip)
        return False
    if not is_available():
        return False
    proc = await _run(["ipset", "add", name, ip, "-exist"])
    if proc.returncode != 0:
        logger.warning(
            "ipset add %s %s failed rc=%d stderr=%s",
            name,
            ip,
            proc.returncode,
            proc.stderr.strip()[:200],
        )
        return False
    return True


async def _remove_from(name: str, ip: str) -> bool:
    if not _is_valid_v4(ip):
        return False
    if not is_available():
        return False
    # ``-exist`` makes ``del`` a no-op when the entry isn't present.
    proc = await _run(["ipset", "del", name, ip, "-exist"])
    if proc.returncode != 0:
        logger.warning(
            "ipset del %s %s failed rc=%d stderr=%s",
            name,
            ip,
            proc.returncode,
            proc.stderr.strip()[:200],
        )
        return False
    return True


async def _list_set(name: str) -> set[str]:
    """Return current members of an ipset. Empty set on any failure."""
    if not is_available():
        return set()
    proc = await _run(["ipset", "list", name, "-o", "save"])
    if proc.returncode != 0:
        return set()
    members: set[str] = set()
    for line in proc.stdout.splitlines():
        # ``-o save`` lines look like: ``add ms-allowed-admins 10.42.0.42``
        m = re.match(rf"^add\s+{re.escape(name)}\s+(\S+)", line)
        if m:
            members.add(m.group(1))
    return members


async def add_admin(ip: str) -> bool:
    return await _add_to(ADMIN_SET, ip)


async def remove_admin(ip: str) -> bool:
    return await _remove_from(ADMIN_SET, ip)


async def add_guest(ip: str) -> bool:
    return await _add_to(GUEST_SET, ip)


async def remove_guest(ip: str) -> bool:
    return await _remove_from(GUEST_SET, ip)


async def list_admins() -> set[str]:
    return await _list_set(ADMIN_SET)


async def list_guests() -> set[str]:
    return await _list_set(GUEST_SET)


# Synchronous variants for the captive sub-app's per-request check.
# We can't await an async function from an arbitrary request handler
# in every framework path; the sync versions read the ipset via a
# subprocess directly. They're cheap (~ms) but still indirect.

_SYNC_TIMEOUT = 1.5


def _sync_list_members(name: str) -> set[str]:
    """Synchronous read of an ipset for the captive sub-app's hot path.

    Used by ``_is_captive_acked`` which runs inside FastAPI route
    handlers. The async ``_list_set`` requires an event loop; this sync
    flavor is fine because the captive probes run a few times per join,
    not per packet.
    """
    if not is_available():
        return set()
    argv = ["sudo", "ipset", "list", name, "-o", "save"]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=_SYNC_TIMEOUT,
        )
    except FileNotFoundError, subprocess.TimeoutExpired, OSError:
        return set()
    if proc.returncode != 0:
        return set()
    members: set[str] = set()
    for line in proc.stdout.splitlines():
        m = re.match(rf"^add\s+{re.escape(name)}\s+(\S+)", line)
        if m:
            members.add(m.group(1))
    return members


def is_admin(ip: str) -> bool:
    """Sync: True iff ``ip`` is currently allow-listed for WAN."""
    if not _is_valid_v4(ip):
        return False
    return ip in _sync_list_members(ADMIN_SET)


def is_guest(ip: str) -> bool:
    """Sync: True iff ``ip`` is currently allow-listed for meeting features."""
    if not _is_valid_v4(ip):
        return False
    return ip in _sync_list_members(GUEST_SET)


# ── Lease-aware GC ────────────────────────────────────────────


def parse_dnsmasq_leases(text: str) -> set[str]:
    """Extract currently-leased IPs from a dnsmasq lease file.

    Tolerates the standard 5-field format
    ``<expiry> <mac> <ip> <name> <client-id>`` plus blank lines.
    Malformed lines are skipped silently — we'd rather under-prune
    than mis-prune on a transient parse error.
    """
    leases: set[str] = set()
    for raw in text.splitlines():
        parts = raw.split()
        if len(parts) < 3:
            continue
        candidate = parts[2]
        if _is_valid_v4(candidate):
            leases.add(candidate)
    return leases


async def gc_once(*, leases_path: str = LEASES_FILE) -> dict[str, int]:
    """One GC tick: prune ipset entries whose IP isn't currently leased.

    Returns ``{"admins_pruned": int, "guests_pruned": int}``.
    Best-effort: a missing/unreadable leases file means we return
    zeros without touching the ipsets — better to leave stale entries
    than to drain trusted clients on a transient read error.
    """
    if not is_available():
        return {"admins_pruned": 0, "guests_pruned": 0}
    try:
        with open(leases_path) as fh:
            leases = parse_dnsmasq_leases(fh.read())
    except (FileNotFoundError, PermissionError, OSError) as exc:
        logger.debug("gc_once: skipping (leases file unreadable: %s)", exc)
        return {"admins_pruned": 0, "guests_pruned": 0}

    pruned: dict[str, int] = {"admins_pruned": 0, "guests_pruned": 0}
    for set_name, key in ((ADMIN_SET, "admins_pruned"), (GUEST_SET, "guests_pruned")):
        members = await _list_set(set_name)
        stale = members - leases
        for ip in stale:
            if await _remove_from(set_name, ip):
                pruned[key] += 1
    if pruned["admins_pruned"] or pruned["guests_pruned"]:
        logger.info(
            "captive gc: pruned %d admin + %d guest ipset entr(ies)",
            pruned["admins_pruned"],
            pruned["guests_pruned"],
        )
    return pruned


async def gc_loop(interval_s: float = 300.0, *, leases_path: str = LEASES_FILE) -> None:
    """Run :func:`gc_once` on a fixed interval forever.

    Started by the server lifespan as an asyncio task; cancelled on
    shutdown. Never raises — any per-tick failure logs + we wait for
    the next interval.
    """
    while True:
        try:
            await gc_once(leases_path=leases_path)
        except Exception:
            logger.exception("captive gc_once tick failed")
        await asyncio.sleep(interval_s)
