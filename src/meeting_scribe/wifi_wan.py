"""WAN orchestration: STA upstream + wired-link claim + per-iface probes.

This module is the **only** caller that mutates ``wlan_sta`` NM
profiles and the only one that claims a metric on the wired
``enP7s7`` profile. All AP-side concerns stay in :mod:`wifi`; this
module is strictly about reaching upstream.

Public surface:

* :func:`wan_up` — bring up an STA association for a stored profile
* :func:`wan_down` — tear down the active STA + delete the NM profile
* :func:`cleanup_orphan_sta_profiles` — boot-time helper (deletes
  ``meeting-scribe-sta-*`` profiles whose id ≠ current active id)
* :func:`wan_status` — read per-interface state (wired + WiFi)
* :func:`claim_wired_profile` — record + enforce route-metric on the
  wired profile so it deterministically wins as the default route
* :func:`probe_wlan_sta_connectivity` — interface-bound captive probe

All subprocess shell-outs go through :func:`_run` so tests can inject
deterministic argv→output mappings via :data:`NMCLI_RUNNER`.

The plaintext PSK only crosses one frame (this module) and one argv
(``nmcli con add/modify wifi-sec.psk <plaintext>``). No log statement
ever sees it; no ``repr`` ever returns it. See
``docs/plans/wifi-wan-gateway.md`` Synthesis for the residual
exposure model.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import re
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from meeting_scribe.server_support import secrets, settings_store
from meeting_scribe.wifi import WIFI_IFACE
from meeting_scribe.wifi_sta import STA_IFACE, ScanEntry

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────


WIRED_IFACE_DEFAULT = "enP7s7"
WIRED_TARGET_METRIC = 100
STA_TARGET_METRIC = 600
STA_PROFILE_PREFIX = "meeting-scribe-sta-"

# Per-iface sysctl key for IPv6 disable. Applied before any NM up so
# the kernel never assigns a link-local v6 address to wlan_sta.
_V6_DISABLE_SYSCTL_KEY = f"net.ipv6.conf.{STA_IFACE}.disable_ipv6"

_NMCLI_TIMEOUT_SHORT = 8.0
_NMCLI_TIMEOUT_UP = 60.0
_CURL_PROBE_TIMEOUT = 6.0

# NM connectivity-check URL used by the per-iface probe.
_CAPTIVE_PROBE_URL = "http://nmcheck.gnome.org/check_network_status.txt"
_CAPTIVE_PROBE_EXPECTED = "NetworkManager is online"


# ── Subprocess seam (test injection point) ──────────────────────


RunnerFn = Callable[[list[str], float], Awaitable[subprocess.CompletedProcess]]


# NM CLI commands that mutate system connections / iface state need
# polkit auth that the systemd-managed service (running as ``bradlay``)
# doesn't carry. The existing AP-side code in ``wifi.py`` prefixes
# ``sudo nmcli`` because ``bradlay`` has ``NOPASSWD: ALL`` on this box.
# Mirror that here so OPEN / WPA / band-bound profiles can all be
# created. Tests bypass this auto-prefix entirely because they replace
# :data:`NMCLI_RUNNER` rather than invoking :func:`_default_run`.
_AUTO_SUDO_FIRST_ARG: frozenset[str] = frozenset({"nmcli"})


async def _default_run(argv: list[str], timeout: float) -> subprocess.CompletedProcess:
    loop = asyncio.get_running_loop()
    if argv and argv[0] in _AUTO_SUDO_FIRST_ARG:
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


# Tests monkeypatch this with a recording stub. Production keeps the
# subprocess shell-out.
NMCLI_RUNNER: RunnerFn = _default_run


async def _run(
    argv: list[str], *, timeout: float = _NMCLI_TIMEOUT_SHORT
) -> subprocess.CompletedProcess:
    """Indirect every subprocess through this seam so tests can stub."""
    return await NMCLI_RUNNER(argv, timeout)


# ── Reconciliation hook ─────────────────────────────────────────


async def _real_reconcile() -> None:
    # Lazy import keeps the reconcile-hook decoupled at import time so
    # tests that don't touch the firewall path don't pull in wifi.py.
    from meeting_scribe.wifi import reconcile_network_state

    await reconcile_network_state()


# Tests override this with a recording stub. Production routes every
# transition through ``wifi.reconcile_network_state``.
RECONCILE_HOOK: Callable[[], Awaitable[None]] = _real_reconcile


# ── Per-iface IPv6 disable ─────────────────────────────────────


async def _ensure_wlan_sta_ipv6_disabled() -> None:
    """Set ``net.ipv6.conf.wlan_sta.disable_ipv6=1``.

    Must run BEFORE ``nmcli con up`` so the kernel never autoconfigures
    a link-local v6 address from the upstream RAs. Idempotent — sysctl
    silently no-ops on a repeat write.
    """
    proc = await _run(
        ["sudo", "sysctl", "-w", f"{_V6_DISABLE_SYSCTL_KEY}=1"],
        timeout=_NMCLI_TIMEOUT_SHORT,
    )
    if proc.returncode != 0:
        # Don't fail the whole wan_up over a sysctl write that may have
        # already been applied. But log loudly so the test/cutover
        # checklist catches misconfiguration.
        logger.warning(
            "sysctl %s=1 failed rc=%d stderr=%s",
            _V6_DISABLE_SYSCTL_KEY,
            proc.returncode,
            proc.stderr.strip()[:200],
        )


# ── NM profile presence helpers ────────────────────────────────


def _profile_name_for(profile_id: str) -> str:
    return f"{STA_PROFILE_PREFIX}{profile_id}"


async def _nm_profile_exists(name: str) -> bool:
    proc = await _run(["nmcli", "-t", "-f", "NAME", "con", "show"])
    if proc.returncode != 0:
        return False
    return any(line.strip() == name for line in proc.stdout.splitlines())


async def _owned_sta_profile_names() -> list[str]:
    """List NM connection names matching ``meeting-scribe-sta-*``."""
    proc = await _run(["nmcli", "-t", "-f", "NAME", "con", "show"])
    if proc.returncode != 0:
        return []
    return [
        line.strip()
        for line in proc.stdout.splitlines()
        if line.strip().startswith(STA_PROFILE_PREFIX)
    ]


# ── wan_up / wan_down ──────────────────────────────────────────


async def wan_up(profile_id: str) -> None:
    """Bring up the upstream STA association for ``profile_id``.

    Reconciliation semantics (see plan):
    * If a different profile is currently active, run :func:`wan_down`
      first so we never leave two owned profiles on disk.
    * Modify the NM profile in place if it already exists (post-reboot
      recovery path — the profile keyfile survives reboot in v1).
    * Otherwise ``nmcli con add`` a fresh profile.

    Always sets per-iface ``disable_ipv6=1`` BEFORE bringing up the
    connection so no v6 LL address is ever assigned to ``wlan_sta``.

    Final invariant: exactly **one** ``meeting-scribe-sta-<id>``
    profile is on disk and it is active.
    """
    profile = settings_store._find_wan_profile_by_id(profile_id)
    if profile is None:
        raise ValueError(f"no wan profile with id {profile_id!r}")

    # If another profile is currently marked active, tear it down so
    # the on-disk set stays at exactly one owned profile.
    current_active = settings_store._effective_wan_active_profile_id()
    if current_active is not None and current_active != profile_id:
        await wan_down()

    # Sysctl v6 disable BEFORE assoc so kernel never autoconfigures v6.
    await _ensure_wlan_sta_ipv6_disabled()

    # Make sure the wlan_sta virtual interface exists. We import lazily
    # to keep the test surface narrow (sta_iface_ensure shells out).
    from meeting_scribe.wifi_sta import sta_iface_ensure

    await sta_iface_ensure()

    # Open networks have no PSK; skip the decrypt step. Otherwise
    # resolve at the boundary and propagate SecretNotFoundError /
    # SecretDecryptError so the CLI/REST layer can report a clean
    # message.
    psk_ref = profile.get("psk_ref")
    psk: str | None = secrets.resolve_psk(psk_ref) if psk_ref else None

    name = _profile_name_for(profile_id)
    ssid = profile["ssid"]
    bssid = profile.get("bssid")
    # Older profiles omit ``band`` entirely; normalize to ``"auto"``.
    band = profile.get("band") or "auto"

    exists = await _nm_profile_exists(name)
    if exists:
        await _nm_modify_profile(name, ssid=ssid, bssid=bssid, band=band, psk=psk)
    else:
        await _nm_add_profile(name, ssid=ssid, bssid=bssid, band=band, psk=psk)

    # Discard the local plaintext PSK reference now that nmcli owns it.
    del psk

    # Bring up. ``con up`` is idempotent — re-applying on an active
    # connection is a no-op.
    up = await _run(["nmcli", "con", "up", name], timeout=_NMCLI_TIMEOUT_UP)
    if up.returncode != 0:
        raise RuntimeError(
            f"nmcli con up {name} failed rc={up.returncode}: {up.stderr.strip()[:200]}"
        )

    settings_store._set_wan_active_profile_id(profile_id)
    await RECONCILE_HOOK()


async def _nm_add_profile(
    name: str, *, ssid: str, bssid: str | None, band: str, psk: str | None
) -> None:
    args = [
        "nmcli",
        "con",
        "add",
        "type",
        "wifi",
        "ifname",
        STA_IFACE,
        "con-name",
        name,
        "ssid",
        ssid,
    ]
    if bssid:
        args += ["bssid", bssid]
    if psk is not None:
        args += [
            "wifi-sec.key-mgmt",
            "wpa-psk",
            "wifi-sec.psk",
            psk,
        ]
    # Open networks: no ``wifi-sec.*`` at all. NM defaults to
    # key-mgmt=none, which is exactly what an OPEN AP expects.
    args += [
        "ipv4.method",
        "auto",
        "ipv4.route-metric",
        str(STA_TARGET_METRIC),
        "ipv6.method",
        "disabled",
        "connection.autoconnect",
        "no",
    ]
    if band in ("a", "bg"):
        # Constrain the supplicant to a single band. Omit the property
        # entirely when "auto" so NM lets the supplicant pick across
        # bands and roam freely.
        args += ["wifi.band", band]
    proc = await _run(args, timeout=_NMCLI_TIMEOUT_UP)
    if proc.returncode != 0:
        raise RuntimeError(
            f"nmcli con add {name} failed rc={proc.returncode}: {proc.stderr.strip()[:200]}"
        )


async def _nm_modify_profile(
    name: str, *, ssid: str, bssid: str | None, band: str, psk: str | None
) -> None:
    """Idempotent in-place re-apply of every owned profile field.

    Called on post-reboot recovery (profile keyfile survived) and when
    the user edits the SSID / band / BSSID / PSK of a saved profile via
    the admin UI. ``psk=None`` is the OPEN-network path — clears any
    prior ``wifi-sec.*`` so a profile flipped from WPA-PSK to OPEN
    actually drops the stale security config on disk.
    """
    args = [
        "nmcli",
        "con",
        "modify",
        name,
        "802-11-wireless.ssid",
        ssid,
        "ipv4.method",
        "auto",
        "ipv4.route-metric",
        str(STA_TARGET_METRIC),
        "ipv6.method",
        "disabled",
        "connection.autoconnect",
        "no",
    ]
    if psk is not None:
        args += [
            "wifi-sec.key-mgmt",
            "wpa-psk",
            "wifi-sec.psk",
            psk,
        ]
    else:
        # Clear any prior security so an OPEN profile cannot inherit a
        # stale WPA-PSK config from a previous round-trip.
        args += ["-wifi-sec.key-mgmt", ""]
        args += ["-wifi-sec.psk", ""]
    # nmcli expects ``-`` prefix to clear a property (e.g. clear a stale
    # BSSID pin when caller passes bssid=None, or clear ``wifi.band``
    # when caller flips back to auto).
    if bssid:
        args += ["802-11-wireless.bssid", bssid]
    else:
        args += ["-802-11-wireless.bssid", ""]
    if band in ("a", "bg"):
        args += ["wifi.band", band]
    else:
        args += ["-wifi.band", ""]
    proc = await _run(args, timeout=_NMCLI_TIMEOUT_UP)
    if proc.returncode != 0:
        raise RuntimeError(
            f"nmcli con modify {name} failed rc={proc.returncode}: {proc.stderr.strip()[:200]}"
        )


async def wan_down() -> None:
    """Tear down the active STA and delete the owned NM profile.

    Deleting the profile removes ``/etc/NetworkManager/system-connections/
    meeting-scribe-sta-<id>.nmconnection`` and the persisted PSK with
    it. Clears ``wan_active_profile_id``. Idempotent — safe to call
    when nothing is active.
    """
    active = settings_store._effective_wan_active_profile_id()
    if active is None:
        # Nothing claimed in settings, but a stray owned profile may
        # still be on disk from an earlier crash. Run the orphan sweep
        # so callers can rely on a clean state after wan_down.
        await cleanup_orphan_sta_profiles()
        await RECONCILE_HOOK()
        return

    name = _profile_name_for(active)
    # con down — ignore rc (may already be inactive).
    await _run(["nmcli", "con", "down", name], timeout=_NMCLI_TIMEOUT_UP)
    # con delete — best-effort; if the profile is already gone, that's
    # a successful end state.
    await _run(["nmcli", "con", "delete", name], timeout=_NMCLI_TIMEOUT_SHORT)
    settings_store._set_wan_active_profile_id(None)
    await RECONCILE_HOOK()


async def cleanup_orphan_sta_profiles() -> int:
    """Delete ``meeting-scribe-sta-*`` profiles whose id ≠ active.

    Returns the number of profiles deleted. By design, the active
    profile is **preserved** across reboot (v1 PSK-at-rest decision);
    only profiles abandoned by prior ``set-active`` changes without
    a clean down are removed.
    """
    active = settings_store._effective_wan_active_profile_id()
    active_name = _profile_name_for(active) if active else None
    names = await _owned_sta_profile_names()
    deleted = 0
    for n in names:
        if n == active_name:
            continue
        proc = await _run(["nmcli", "con", "delete", n], timeout=_NMCLI_TIMEOUT_SHORT)
        if proc.returncode == 0:
            deleted += 1
        else:
            logger.warning(
                "nmcli con delete %s failed rc=%d: %s",
                n,
                proc.returncode,
                proc.stderr.strip()[:200],
            )
    return deleted


# ── Wired-profile claim ────────────────────────────────────────


@dataclass(frozen=True)
class WiredClaim:
    """Result of :func:`claim_wired_profile` for the wired ``enP7s7`` iface.

    ``name`` is the NM connection currently active on ``enP7s7`` (e.g.
    ``"Wired connection 3"``). ``prior_metric`` is ``-1`` when NM had
    the route-metric set to its default (auto); we represent that as
    ``None`` in the returned dataclass for clarity.
    """

    name: str
    prior_metric: int | None


async def claim_wired_profile(iface: str = WIRED_IFACE_DEFAULT) -> WiredClaim | None:
    """Locate the NM connection active on ``iface`` and return its name + metric.

    Returns ``None`` if no NM connection is currently active on
    ``iface`` (the wired port may be down — caller decides whether
    that is an error).

    Pure inspection — no mutations to NM. ``enforce_wired_metric`` is
    a separate idempotent step that actually adjusts the metric.
    """
    proc = await _run(["nmcli", "-t", "-f", "NAME,DEVICE", "con", "show", "--active"])
    if proc.returncode != 0:
        return None
    name: str | None = None
    for line in proc.stdout.splitlines():
        # nmcli -t terse format: NAME:DEVICE with colon separator and
        # backslash-escaped embedded colons in NAME. We split from the
        # right since DEVICE is a fixed-form interface name.
        if not line:
            continue
        if ":" not in line:
            continue
        cand_name, _, cand_dev = line.rpartition(":")
        if cand_dev == iface and cand_name:
            name = cand_name.replace("\\:", ":")
            break
    if name is None:
        return None
    metric = await _read_route_metric(name)
    return WiredClaim(name=name, prior_metric=metric)


async def _read_route_metric(connection_name: str) -> int | None:
    proc = await _run(["nmcli", "-t", "-f", "ipv4.route-metric", "con", "show", connection_name])
    if proc.returncode != 0:
        return None
    # Output form: ``ipv4.route-metric:600`` or ``ipv4.route-metric:-1``
    for line in proc.stdout.splitlines():
        if line.startswith("ipv4.route-metric:"):
            raw = line.split(":", 1)[1].strip()
            try:
                value = int(raw)
            except ValueError:
                return None
            return None if value < 0 else value
    return None


async def enforce_wired_metric(name: str, target: int = WIRED_TARGET_METRIC) -> bool:
    """Set ``ipv4.route-metric`` on ``name`` to ``target`` if not already set.

    Returns True if a change was applied; False if the connection was
    already at the target metric (idempotent no-op).
    """
    current = await _read_route_metric(name)
    if current == target:
        return False
    proc = await _run(
        ["nmcli", "con", "modify", name, "ipv4.route-metric", str(target)],
        timeout=_NMCLI_TIMEOUT_SHORT,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"nmcli con modify {name} ipv4.route-metric {target} failed "
            f"rc={proc.returncode}: {proc.stderr.strip()[:200]}"
        )
    # Re-up to apply the new metric to the active default route.
    await _run(["nmcli", "con", "up", name], timeout=_NMCLI_TIMEOUT_UP)
    return True


# ── wan_status ─────────────────────────────────────────────────


def _parse_iface_state(stdout: str) -> str | None:
    """Return the operstate field from ``ip -br link show <iface>`` output."""
    tokens = stdout.strip().split()
    if len(tokens) < 2:
        return None
    return tokens[1]


async def scan_upstream() -> list[ScanEntry]:
    """Return upstream WiFi APs visible to the radio (one entry per BSS).

    Critical: when the AP is broadcasting on ``wlP9s9``, that
    interface can't do meaningful off-channel scans because the radio
    is busy beaconing. The single-radio dual-mode design routes
    scanning to a separate virtual interface ``wlan_sta`` (created via
    ``iw phy ... interface add``) so the supplicant can timeshare for
    a proper scan without interrupting the AP. We provision the vif
    first, then use NM to rescan + read the cache.

    If ``wlan_sta`` cannot be created (no root, no MT7925 multi-vif
    support), we fall back to ``wlP9s9`` — which still works when the
    AP isn't broadcasting, and degrades gracefully (returns just the
    AP self-beacon) when it is.
    """
    from meeting_scribe.wifi_sta import sta_iface_ensure

    # Best-effort vif provision. If this fails, fall back to scanning
    # on the AP iface — at worst the scan is degraded; we never block.
    sta_ok = await sta_iface_ensure()
    iface = STA_IFACE if sta_ok and Path(f"/sys/class/net/{STA_IFACE}").exists() else WIFI_IFACE
    # Force a fresh rescan. ``--rescan yes`` ensures nmcli kicks the
    # supplicant; we ignore rc because polkit may deny it on a user
    # shell (server context runs as root and succeeds).
    await _run(["nmcli", "dev", "wifi", "rescan", "ifname", iface], timeout=8.0)
    # Wait for the scan to actually complete. NM's rescan is async —
    # the kernel takes ~3-5s to sweep the regdomain's channels. The
    # earlier 0.2s settle returned the stale cache and surfaced as
    # "scan finds only my own SSID".
    await asyncio.sleep(3.5)
    proc = await _run(
        [
            "nmcli",
            "-t",
            "-f",
            "SSID,BSSID,SIGNAL,CHAN,SECURITY",
            "dev",
            "wifi",
            "list",
            "ifname",
            iface,
            "--rescan",
            "no",
        ],
        timeout=10.0,
    )
    if proc.returncode != 0:
        return []
    return _parse_nmcli_wifi_list(proc.stdout)


def _parse_nmcli_wifi_list(text: str) -> list[ScanEntry]:
    """Parse the terse ``nmcli -t dev wifi list`` output.

    The terse format uses ``:`` as a separator with ``\\:`` escaping any
    colons inside fields (notably BSSIDs). Lines look like:

        DellEfficiency_Guest:CC\\:DB\\:93\\:C0\\:4F\\:A1:89:1:

    SSID may legitimately contain colons too, so we walk the string
    splitting on un-escaped colons.
    """
    out: list[ScanEntry] = []
    for raw in text.splitlines():
        fields = _split_nmcli_terse(raw)
        if len(fields) < 5:
            continue
        ssid, bssid, signal, chan, security = fields[:5]
        if not ssid:
            continue  # hidden / probe-only SSID — skip per existing scan semantics
        try:
            channel = int(chan)
        except ValueError:
            continue
        try:
            signal_pct = int(signal)
        except ValueError:
            signal_pct = 0
        # nmcli signal is 0-100 (link quality). Convert to a rough dBm
        # estimate for compatibility with the existing ScanEntry shape.
        # The widely-used mapping is: 100 ~= -50 dBm, 0 ~= -100 dBm.
        signal_dbm = -100.0 + (signal_pct * 0.5)
        out.append(
            ScanEntry(
                bssid=bssid.lower(),
                ssid=ssid,
                channel=channel,
                signal_dbm=signal_dbm,
                rsn_present=bool(security and security.strip()),
            )
        )
    return out


def _split_nmcli_terse(line: str) -> list[str]:
    """Split an ``nmcli -t`` line on un-escaped ``:`` separators."""
    fields: list[str] = []
    buf: list[str] = []
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line) and line[i + 1] == ":":
            buf.append(":")
            i += 2
            continue
        if ch == ":":
            fields.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    fields.append("".join(buf))
    return fields


# ── Band classification + scan consolidation ──────────────────


def channel_to_band(channel: int) -> str:
    """Return the NM band code for ``channel``.

    ``"bg"`` (2.4 GHz, channels 1-14) or ``"a"`` (5 GHz / 6 GHz,
    channel ≥ 36). Anything else falls back to ``"a"`` because the
    MT7925 in the GB10 doesn't support 6 GHz station mode in v1 and
    the >100 channels are still 5 GHz proper.
    """
    return "bg" if 1 <= channel <= 14 else "a"


def band_label(band: str) -> str:
    """Human label for a band code. Used by CLI + UI surfaces."""
    if band == "a":
        return "5 GHz"
    if band == "bg":
        return "2.4 GHz"
    return "Auto"


@dataclass(frozen=True)
class SsidGroup:
    """Consolidated scan result — one row per (SSID, security) network.

    Aggregates all BSSes of a given SSID so the UI can show a single
    selectable row instead of N rows for the same network. The roaming
    decision is then a NM/wpa_supplicant concern (set ``wifi.band`` to
    constrain, omit it for auto roaming across bands).
    """

    ssid: str
    rsn_present: bool
    bands: tuple[str, ...]  # sorted subset of ("bg", "a")
    best_signal_dbm: float
    best_signal_band: str  # the band of the strongest BSS
    ap_count: int
    channels: tuple[int, ...]  # sorted unique channels


def consolidate_scan(entries: list[ScanEntry]) -> list[SsidGroup]:
    """Group raw scan entries by (ssid, rsn_present), drop the BSS-level noise.

    Order: by descending best signal so the strongest network is first.
    Hidden / empty SSIDs are already filtered upstream by the nmcli
    parser; we just need to consolidate.
    """
    buckets: dict[tuple[str, bool], list[ScanEntry]] = {}
    for e in entries:
        if not e.ssid:
            continue
        buckets.setdefault((e.ssid, e.rsn_present), []).append(e)

    groups: list[SsidGroup] = []
    for (ssid, secure), bss_list in buckets.items():
        bss_list.sort(key=lambda b: b.signal_dbm, reverse=True)
        best = bss_list[0]
        bands = sorted({channel_to_band(b.channel) for b in bss_list})
        channels = sorted({b.channel for b in bss_list})
        groups.append(
            SsidGroup(
                ssid=ssid,
                rsn_present=secure,
                bands=tuple(bands),
                best_signal_dbm=best.signal_dbm,
                best_signal_band=channel_to_band(best.channel),
                ap_count=len(bss_list),
                channels=tuple(channels),
            )
        )
    groups.sort(key=lambda g: g.best_signal_dbm, reverse=True)
    return groups


async def scan_upstream_consolidated() -> list[SsidGroup]:
    """Convenience: raw scan + consolidation in one call."""
    return consolidate_scan(await scan_upstream())


async def _iface_link_up(iface: str) -> bool:
    proc = await _run(["ip", "-br", "link", "show", iface])
    if proc.returncode != 0:
        return False
    state = _parse_iface_state(proc.stdout)
    # ``ip -br link`` operstate is UP / DOWN / UNKNOWN / NO-CARRIER /
    # LOWERLAYERDOWN. Only UP and UNKNOWN (used for tunnels / wlan_sta
    # in some kernels) count as carrier-up here.
    return state in ("UP", "UNKNOWN")


async def _iface_v4_lease(iface: str) -> str | None:
    proc = await _run(["ip", "-4", "-br", "addr", "show", iface])
    if proc.returncode != 0:
        return None
    # ``ip -4 -br addr`` prints e.g.: ``enP7s7  UP  192.168.8.153/24``
    parts = proc.stdout.strip().split()
    for part in parts[2:]:
        if "/" in part:
            return part.split("/", 1)[0]
    return None


async def _active_connection_name(iface: str) -> str | None:
    proc = await _run(["nmcli", "-t", "-f", "NAME,DEVICE", "con", "show", "--active"])
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        if ":" not in line:
            continue
        cand_name, _, cand_dev = line.rpartition(":")
        if cand_dev == iface and cand_name:
            return cand_name.replace("\\:", ":")
    return None


async def _default_route_iface() -> str | None:
    """Return the interface carrying the lowest-metric IPv4 default route."""
    proc = await _run(["ip", "-4", "route", "show", "default"])
    if proc.returncode != 0:
        return None
    # Lines look like: ``default via 192.168.8.1 dev enP7s7 proto dhcp metric 100``
    best_iface: str | None = None
    best_metric: int | None = None
    for line in proc.stdout.splitlines():
        parts = line.split()
        if "dev" not in parts:
            continue
        iface = parts[parts.index("dev") + 1]
        metric = None
        if "metric" in parts:
            try:
                metric = int(parts[parts.index("metric") + 1])
            except ValueError, IndexError:
                pass  # malformed/truncated route line — fall back to default metric below
        if metric is None:
            metric = 0  # NM emits no metric for the default
        if best_metric is None or metric < best_metric:
            best_metric = metric
            best_iface = iface
    return best_iface


async def _sta_ssid_and_signal() -> tuple[str | None, str | None, int | None]:
    """Return ``(ssid, bssid, signal_dbm)`` for the current STA association."""
    proc = await _run(
        ["nmcli", "-t", "-f", "GENERAL.CONNECTION,GENERAL.STATE", "dev", "show", STA_IFACE]
    )
    if proc.returncode != 0:
        return None, None, None
    # Find associated SSID/BSSID via the IW link command — more reliable
    # for our purposes than ``nmcli dev wifi list`` which doesn't filter
    # to the STA iface only.
    link = await _run(["iw", "dev", STA_IFACE, "link"])
    if link.returncode != 0 or "Not connected" in link.stdout:
        return None, None, None
    ssid_m = re.search(r"SSID:\s+(.+)$", link.stdout, re.MULTILINE)
    bssid_m = re.search(r"Connected to\s+([0-9a-f:]{17})", link.stdout, re.IGNORECASE)
    signal_m = re.search(r"signal:\s+(-?\d+)\s+dBm", link.stdout)
    ssid = ssid_m.group(1).strip() if ssid_m else None
    bssid = bssid_m.group(1).lower() if bssid_m else None
    signal = int(signal_m.group(1)) if signal_m else None
    return ssid, bssid, signal


async def probe_wlan_sta_connectivity() -> tuple[str, str | None]:
    """Return ``(connectivity, portal_url)`` for the ``wlan_sta`` interface.

    Connectivity values:
      ``"none"``    — link down
      ``"limited"`` — link up, no Internet (DNS/HTTP fail)
      ``"portal"``  — HTTP probe sees a captive portal redirect
      ``"full"``    — HTTP probe matches expected body
      ``"unknown"`` — probe could not run (e.g. curl missing)

    The probe is bound to the STA interface via ``curl --interface``
    so a healthy wired link does not mask STA-side portal redirects.
    NM's per-iface ``IP4.CAPTIVE-PORTAL`` field is consulted first as
    the RFC 8910 CAPPORT URL; the active HTTP probe is the fallback /
    cross-check.
    """
    link = await _run(["ip", "-br", "link", "show", STA_IFACE])
    if link.returncode != 0:
        return "none", None
    state = _parse_iface_state(link.stdout)
    if state not in ("UP", "UNKNOWN"):
        return "none", None

    # CAPPORT (RFC 8910) — NM stores the upstream-advertised portal URL
    # under this per-device field.
    portal_url: str | None = None
    cap = await _run(["nmcli", "-t", "-f", "IP4.CAPTIVE-PORTAL", "dev", "show", STA_IFACE])
    if cap.returncode == 0:
        for line in cap.stdout.splitlines():
            if line.startswith("IP4.CAPTIVE-PORTAL:"):
                value = line.split(":", 1)[1].strip()
                if value and value != "--":
                    portal_url = value
                break

    # Interface-bound active HTTP probe. We deliberately do NOT follow
    # redirects (no ``-L``) so a captive-portal 30x is observable via
    # ``%{http_code}`` and ``%{redirect_url}``. The body is appended
    # before two trailer lines holding http_code and redirect_url.
    probe = await _run(
        [
            "curl",
            "--interface",
            STA_IFACE,
            "-m",
            str(int(_CURL_PROBE_TIMEOUT)),
            "-s",
            "-o",
            "-",
            "-w",
            "\\n%{http_code}\\n%{redirect_url}\\n",
            _CAPTIVE_PROBE_URL,
        ],
        timeout=_CURL_PROBE_TIMEOUT + 2,
    )
    if probe.returncode != 0:
        # curl rc 6 = couldn't resolve; rc 7 = couldn't connect; either
        # way the link is up but the Internet isn't reachable.
        return "limited", portal_url

    lines = probe.stdout.splitlines()
    if len(lines) < 2:
        return "unknown", portal_url
    http_code = lines[-2].strip()
    redirect_url = lines[-1].strip()
    body = "\n".join(lines[:-2])

    if http_code == "200" and _CAPTIVE_PROBE_EXPECTED in body:
        return "full", None  # Internet reachable, no portal.
    # 30x with a non-empty redirect target ⇒ captive portal.
    if http_code and http_code[:1] == "3" and redirect_url:
        return "portal", portal_url or redirect_url
    # 200 to an unexpected body — most portals do a transparent proxy
    # rather than a redirect. Treat as portal if we have a CAPPORT
    # URL to point the admin at, else mark limited.
    if portal_url:
        return "portal", portal_url
    return "limited", None


async def wan_status() -> dict:
    """Return per-interface status. No global ``nmcli connectivity`` mixing."""
    wired_iface = WIRED_IFACE_DEFAULT
    wired_up = await _iface_link_up(wired_iface)
    wired_lease = await _iface_v4_lease(wired_iface) if wired_up else None
    wired_profile = await _active_connection_name(wired_iface)
    wired_metric = await _read_route_metric(wired_profile) if wired_profile else None

    sta_up = (
        await _iface_link_up(STA_IFACE) if Path(f"/sys/class/net/{STA_IFACE}").exists() else False
    )
    sta_lease = await _iface_v4_lease(STA_IFACE) if sta_up else None
    sta_profile_name: str | None = None
    active_id = settings_store._effective_wan_active_profile_id()
    if active_id:
        candidate = _profile_name_for(active_id)
        if await _nm_profile_exists(candidate):
            sta_profile_name = candidate
    sta_metric = await _read_route_metric(sta_profile_name) if sta_profile_name else None
    sta_ssid, sta_bssid, sta_signal = await _sta_ssid_and_signal() if sta_up else (None, None, None)
    if sta_up:
        connectivity, portal_url = await probe_wlan_sta_connectivity()
    else:
        connectivity, portal_url = "none", None

    default_iface = await _default_route_iface()

    return {
        "wired": {
            "iface": wired_iface,
            "up": wired_up,
            "lease": wired_lease,
            "profile_name": wired_profile,
            "route_metric": wired_metric,
            "default_route": default_iface == wired_iface,
        },
        "wifi": {
            "iface": STA_IFACE,
            "up": sta_up,
            "lease": sta_lease,
            "profile_name": sta_profile_name,
            "ssid": sta_ssid,
            "bssid": sta_bssid,
            "signal_dbm": sta_signal,
            "route_metric": sta_metric,
            "default_route": default_iface == STA_IFACE,
            "connectivity": connectivity,
            "portal_url": portal_url,
        },
        "active_default": default_iface,
        "egress_mode": settings_store._effective_wan_egress_mode(),
    }


# ── Captive-portal poller ──────────────────────────────────────


async def _poll_captive_state(interval_s: float = 30.0) -> None:
    """Periodic per-iface captive-portal probe; cache result in settings.

    Started as an asyncio task by the server lifespan when a WAN
    profile is active. Persists each result so the admin UI can render
    without re-probing. Never raises — long-running task is robust to
    transient subprocess failures.
    """
    while True:
        try:
            if settings_store._effective_wan_active_profile_id() is not None:
                conn, url = await probe_wlan_sta_connectivity()
                settings_store._save_wan_captive_state(
                    STA_IFACE,
                    connectivity=conn,
                    portal_url=url,
                    probed_at=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
                )
        except Exception:
            logger.exception("captive-state poll iteration failed")
        await asyncio.sleep(interval_s)
