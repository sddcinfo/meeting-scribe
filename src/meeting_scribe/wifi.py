"""WiFi hotspot state machine for meeting-scribe.

Single owner of all WiFi AP lifecycle: bring-up, teardown, mode switching,
captive portal, firewall, regdomain, and credentials. ``server.py`` imports
from here — never the reverse.

Locking discipline
------------------
``_wifi_lock`` is an ``asyncio.Lock`` acquired ONLY by the three public
async entry points: ``wifi_up``, ``wifi_down``, ``wifi_switch``. Internal
helpers (``_bring_up_ap``, ``_teardown_ap``, …) never acquire the lock
themselves so they can be freely composed without deadlock risk.

State-file invariant
--------------------
``/tmp/meeting-hotspot.json`` is a *derived cache* of what nmcli reports
is actually broadcasting. It is ALWAYS written by reading back from
``nmcli --show-secrets`` — never from in-memory generated credentials.
This makes it impossible for the displayed QR code to encode SSID/psk
that don't match the radio.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import secrets
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Regdomain + settings helpers are imported here for use by the AP
# bring-up path, AND re-exported as ``wifi.X`` for back-compat with
# test_wifi.py's source-grep assertions (e.g. ``wifi._regdomain_modprobe_path``).
from meeting_scribe.server_support.regdomain import (
    _current_regdomain,
    _ensure_regdomain,
    _ensure_regdomain_persistent,
)
from meeting_scribe.server_support.settings_store import (  # noqa: F401
    _DEFAULT_REGDOMAIN,
    SETTINGS_OVERRIDE_FILE,
    _effective_regdomain,
    _load_settings_override,
    _regdomain_modprobe_path,
    _save_settings_override,
)
from meeting_scribe.util.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────

AP_CON_NAME = "DellDemo-AP"
AP_IP = "10.42.0.1"
HOTSPOT_STATE_FILE = Path("/tmp/meeting-hotspot.json")
HOTSPOT_SUBNET = "10.42.0."
HOTSPOT_SUBNET_CIDR = os.environ.get("SCRIBE_HOTSPOT_SUBNET_CIDR", "10.42.0.0/24")
HOTSPOT_IPTABLES_COMMENT = "meeting-scribe-hotspot"
WIFI_IFACE = "wlP9s9"  # MT7925 on GB10

DEFAULT_BAND = "a"  # 5 GHz
DEFAULT_CHANNEL = 36
MEETING_PORT = 8080
# Plan 1 §A unified-hotspot v1.0: a single fixed SSID serves both admin
# and guest traffic — discriminated only by the `scribe_admin` cookie at
# the HTTPS layer. The pre-cutover "rotating per-meeting Demo XXXX" SSID
# is preserved for the legacy `meeting` mode but new deployments default
# to `Dell Meeting`.
SSID_PREFIX = "Dell Demo"
DEFAULT_ADMIN_SSID = "Dell Meeting"
DEFAULT_UNIFIED_SSID = "Dell Meeting"

# Captive portal paths
DNSMASQ_CONF_DIR = Path("/etc/NetworkManager/dnsmasq-shared.d")
DNSMASQ_CAPTIVE_CONF = DNSMASQ_CONF_DIR / "captive-portal.conf"

# PID files — meeting-scribe's own (ports 80 and 443) plus sddc-cli's
PORTAL_PID_FILE = Path("/tmp/meeting-captive-portal.pid")
MEETING_SCRIBE_PORTAL_PID_FILES = (
    Path("/tmp/meeting-captive-80.pid"),
    Path("/tmp/meeting-captive-443.pid"),
)

# Activation polling
_AP_ACTIVATION_WAIT_SECONDS = 45
_AP_ACTIVATION_POLL_INTERVAL = 1.0

# Driver-settle gap between a ``con down`` and the subsequent
# ``con up`` on the same SSID. Pulled out as a module-level
# constant so the unit test suite can monkeypatch it to 0
# (otherwise every rotation/bounce test pays this on the wall
# clock — 4–10 s of dead time across the suite). Production
# default is 1 s; ``ap_control`` rebinds this to its own
# ``_AP_BOUNCE_DRIVER_SETTLE_S`` for the rotation path.
_AP_BOUNCE_DRIVER_SETTLE_S: float = 1.0

# nmcli timeouts
_NMCLI_CON_UP_TIMEOUT = 60
_NMCLI_CON_DOWN_TIMEOUT = 15
_NMCLI_CON_MODIFY_TIMEOUT = 5
_NMCLI_CON_SHOW_TIMEOUT = 5

# ── Dataclasses ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WifiConfig:
    """Immutable WiFi AP configuration.

    ``security``:
      * ``"open"`` — OWE (Opportunistic Wireless Encryption,
        RFC 8110). v1.0 first-touch + ongoing operation. Looks
        like an open AP to phones, link is DH-encrypted, proper
        4-way handshake authorizes the station.
      * ``"sae"`` — WPA3-only (PMF required). Used by the legacy
        meeting/admin modes the operator can flip to from the
        admin UI after first-touch.
    """

    mode: str  # "off", "meeting", "admin", or "setup"
    ssid: str
    password: str
    band: str = DEFAULT_BAND
    channel: int = DEFAULT_CHANNEL
    regdomain: str = _DEFAULT_REGDOMAIN
    ap_ip: str = AP_IP
    security: str = "sae"


@dataclass(frozen=True)
class RollbackSnapshot:
    """Captured state for wifi_switch rollback."""

    config: WifiConfig | None
    settings: dict = field(default_factory=dict)


# ── Lock ────────────────────────────────────────────────────────────────

_wifi_lock = asyncio.Lock()

# ── Settings I/O ────────────────────────────────────────────────────────
#
# Settings load/save lives in ``meeting_scribe.server_support.settings_store``;
# rollback uses ``atomic_write_json`` directly so the cache invariants of
# ``_save_settings_override`` aren't violated when restoring an arbitrary
# snapshot.


def _restore_settings(snapshot: dict) -> None:
    """Overwrite the persisted settings file with *snapshot* contents."""
    atomic_write_json(SETTINGS_OVERRIDE_FILE, snapshot)


def _deep_copy_settings() -> dict:
    """Return an independent deep copy of the current settings."""
    return copy.deepcopy(_load_settings_override())


# ── nmcli helpers ───────────────────────────────────────────────────────


def _run_nmcli_sync(args: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    """Run ``sudo nmcli <args>`` synchronously. Errors are returned, not raised.

    Any call here is potentially state-mutating (``con add``, ``con modify``,
    ``con up``, ``con down``), so we drop the read-side cache on entry —
    the next read after this returns will see fresh state.
    """
    _invalidate_ap_state_cache()
    return subprocess.run(
        ["sudo", "nmcli", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


# ── nmcli read-state cache ──────────────────────────────────────────────
#
# /api/admin/settings polls _nmcli_ap_is_active() and (when active)
# _nmcli_read_live_ap_credentials() on every open of the admin panel.
# Each call shells out to ``sudo nmcli`` — even with passwordless sudo,
# the PAM session + NM D-Bus round-trip adds up. The AP state on a
# customer device only changes when *we* call ``_run_nmcli_sync`` for a
# write (con add/modify/up/down), so a tiny TTL cache saves the bulk of
# the overhead with no risk of stale reads. ``_run_nmcli_sync`` clears
# the cache so the very next read after a write sees the truth.

_AP_STATE_TTL_S = 5.0
# Longer-cached snapshot for high-volume read paths that don't need to
# observe transitions in real time (``/api/status``, admin-panel polls).
# The transition-sensitive rotation activation loop in
# ``ap_control._start_wifi_ap`` keeps using ``_AP_STATE_TTL_S`` so it
# can detect the AP coming up within its 45×1s budget; raising the
# global TTL to 30 s would mask the transition. The cache is still
# write-invalidated on every nmcli mutation, so the slow-TTL surface
# never serves a value from before a write.
_AP_STATE_STATUS_TTL_S = 30.0


@dataclass
class _ApStateCache:
    is_active: bool | None = None
    credentials: tuple[str, str] | None = None
    is_active_at: float = 0.0
    credentials_at: float = 0.0


_ap_state_cache = _ApStateCache()


def _invalidate_ap_state_cache() -> None:
    """Drop cached AP state. Called automatically on every nmcli write."""
    _ap_state_cache.is_active = None
    _ap_state_cache.credentials = None
    _ap_state_cache.is_active_at = 0.0
    _ap_state_cache.credentials_at = 0.0


def _ap_state_cache_stats() -> dict[str, float]:
    """Test-only introspection helper."""
    return {
        "is_active_age_s": (
            (time.monotonic() - _ap_state_cache.is_active_at)
            if _ap_state_cache.is_active is not None
            else float("inf")
        ),
        "credentials_age_s": (
            (time.monotonic() - _ap_state_cache.credentials_at)
            if _ap_state_cache.credentials is not None
            else float("inf")
        ),
    }


def _parse_nmcli_fields(output: str) -> dict[str, str]:
    """Parse ``nmcli -t -f a,b,c con show NAME`` terse output into a dict.

    The ``-t`` (terse) + ``-f`` (fields) combo emits one ``key:value``
    per line. A field may contain additional ``:`` characters (common for
    SSIDs with punctuation), so we only split on the FIRST colon.
    """
    result: dict[str, str] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        result[key.strip()] = value
    return result


def _nmcli_read_live_ap_credentials(
    *, ttl: float | None = None, bypass_cache: bool = False
) -> tuple[str, str] | None:
    """Return ``(ssid, psk)`` for the active AP profile, or ``None``.

    Reads via ``nmcli --show-secrets`` so the psk is returned in plaintext.
    This is the authoritative source for what the radio is actually
    broadcasting — the state file is a derived cache of this.

    ``ttl`` (default ``_AP_STATE_TTL_S``) controls how long the cached
    value stays usable. Pass ``_AP_STATE_STATUS_TTL_S`` from the
    ``/api/status`` payload assembler to keep the journal quiet on
    high-volume polls. Pass ``bypass_cache=True`` for one-shot reads
    that follow a state-mutating call (e.g. the rotation post-condition
    check) where correctness matters more than noise reduction.
    """
    if ttl is None:
        ttl = _AP_STATE_TTL_S
    now = time.monotonic()
    if (
        not bypass_cache
        and _ap_state_cache.credentials is not None
        and now - _ap_state_cache.credentials_at < ttl
    ):
        # Sentinel ``("", "")`` distinguishes "cached: no creds" from
        # "not cached"; collapse it back to None for the caller.
        creds = _ap_state_cache.credentials
        return None if creds == ("", "") else creds

    # ``--show-secrets`` is a NM read; classify it as such so the
    # invalidate side-effect inside _run_nmcli_sync doesn't fire here.
    proc = subprocess.run(
        [
            "sudo",
            "nmcli",
            "--show-secrets",
            "-t",
            "-f",
            "802-11-wireless.ssid,802-11-wireless-security.psk",
            "con",
            "show",
            AP_CON_NAME,
        ],
        capture_output=True,
        text=True,
        timeout=_NMCLI_CON_SHOW_TIMEOUT,
        check=False,
    )
    if proc.returncode != 0:
        result: tuple[str, str] | None = None
    else:
        fields = _parse_nmcli_fields(proc.stdout)
        ssid = fields.get("802-11-wireless.ssid", "")
        psk = fields.get("802-11-wireless-security.psk", "")
        result = (ssid, psk) if ssid else None

    _ap_state_cache.credentials = result if result is not None else ("", "")
    _ap_state_cache.credentials_at = now
    return result


def _nmcli_ap_is_active(*, ttl: float | None = None, bypass_cache: bool = False) -> bool:
    """Return True if the AP profile is currently active on the radio.

    ``ttl`` defaults to ``_AP_STATE_TTL_S`` (5 s) so transition-sensitive
    callers (rotation activation polling) observe state changes in real
    time. Pass ``_AP_STATE_STATUS_TTL_S`` (30 s) from the high-volume
    read paths (``/api/status``, admin-panel polls) to amortize the
    sudo+nmcli cost. ``bypass_cache=True`` forces a fresh subprocess
    read regardless of TTL — useful for one-shot post-condition checks
    where staleness would mask the very failure we're inspecting.

    The cache is invalidated on every nmcli write — see
    ``_run_nmcli_sync``.
    """
    if ttl is None:
        ttl = _AP_STATE_TTL_S
    now = time.monotonic()
    if (
        not bypass_cache
        and _ap_state_cache.is_active is not None
        and now - _ap_state_cache.is_active_at < ttl
    ):
        return _ap_state_cache.is_active

    proc = subprocess.run(
        ["sudo", "nmcli", "-t", "-f", "NAME,DEVICE", "con", "show", "--active"],
        capture_output=True,
        text=True,
        timeout=_NMCLI_CON_SHOW_TIMEOUT,
        check=False,
    )
    if proc.returncode != 0:
        result = False
    else:
        prefix = f"{AP_CON_NAME}:"
        result = any(line.startswith(prefix) for line in proc.stdout.splitlines())

    _ap_state_cache.is_active = result
    _ap_state_cache.is_active_at = now
    return result


def _nmcli_connection_exists() -> bool:
    """Return True if the AP profile exists in NM (active or not)."""
    proc = _run_nmcli_sync(
        ["-t", "-f", "NAME", "con", "show"],
        timeout=_NMCLI_CON_SHOW_TIMEOUT,
    )
    if proc.returncode != 0:
        return False
    return any(line.strip() == AP_CON_NAME for line in proc.stdout.splitlines())


def _wait_for_ap_active(
    timeout: int = _AP_ACTIVATION_WAIT_SECONDS,
) -> bool:
    """Poll nmcli for up to *timeout* seconds waiting for the AP to go live.

    Covers the case where ``nmcli con up`` returned non-zero (supplicant
    timeout) but NetworkManager subsequently auto-retried and succeeded.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _nmcli_ap_is_active():
            return True
        time.sleep(_AP_ACTIVATION_POLL_INTERVAL)
    return False


def _recreate_ap_profile_with_sae(
    ssid: str, psk: str, *, band: str = "bg", channel: int = 6
) -> subprocess.CompletedProcess[str] | None:
    """Recreate the AP NM connection from scratch with WPA3-SAE security.

    Used by the rotation self-heal path (``hotspot.ap_control``) when
    ``nmcli con modify`` rejects with ``key-mgmt: property is missing``
    because the existing profile carries inconsistent wifi-sec state
    from a prior mode swap (the 2026-05-14 demo failure). Same pattern
    the first-run bootstrap uses for SAE profiles (``_bring_up_ap``).

    Returns the ``con add`` CompletedProcess, or None if the delete
    raised. The bring-up path's open-mode delete-recreate is the
    closest analogue (``wifi.py`` around line 1691); this helper makes
    the same primitive reachable from rotation without copying the
    20-line bring-up function wholesale.
    """
    _run_nmcli_sync(
        ["con", "delete", AP_CON_NAME],
        timeout=_NMCLI_CON_DOWN_TIMEOUT,
    )
    # ``_scrub_netplan_ap_yaml`` removes the stale 90-NM-*.yaml mirror
    # NM auto-renders, which can otherwise re-introduce the inconsistent
    # security state on next NM start.
    _scrub_netplan_ap_yaml()
    return _run_nmcli_sync(
        [
            "con",
            "add",
            "type",
            "wifi",
            "ifname",
            WIFI_IFACE,
            "con-name",
            AP_CON_NAME,
            "autoconnect",
            "no",
            "ssid",
            ssid,
            "wifi.mode",
            "ap",
            "wifi.band",
            band,
            "wifi.channel",
            str(channel),
            "ipv4.method",
            "shared",
            "ipv6.method",
            "shared",
            "wifi-sec.key-mgmt",
            "sae",
            "wifi-sec.pmf",
            "required",
            "wifi-sec.proto",
            "rsn",
            "wifi-sec.pairwise",
            "ccmp",
            "wifi-sec.group",
            "ccmp",
            "wifi-sec.psk",
            psk,
        ],
        timeout=_NMCLI_CON_MODIFY_TIMEOUT,
    )


# ── State file I/O ──────────────────────────────────────────────────────


def _write_hotspot_state_sync(mode: str = "meeting") -> bool:
    """Sync ``HOTSPOT_STATE_FILE`` with the live AP credentials.

    Returns True if the state file now matches a currently-broadcasting AP,
    False otherwise. The state file is ONLY written when the AP is both
    readable and actively broadcasting. Never raises.
    """
    try:
        creds = _nmcli_read_live_ap_credentials()
        if creds is None:
            logger.debug("hotspot state sync: AP profile not readable")
            return False
        if not _nmcli_ap_is_active():
            logger.debug("hotspot state sync: AP profile readable but not active")
            return False
        ssid, psk = creds
        state: dict[str, Any] = {
            "ssid": ssid,
            "password": psk,
            "ap_ip": AP_IP,
            "port": 80,
            "mode": mode,
        }
        atomic_write_json(HOTSPOT_STATE_FILE, state)
        return True
    except Exception as exc:
        logger.debug("hotspot state sync failed: %s", exc)
        return False


def _clear_hotspot_state() -> None:
    """Remove the hotspot state file."""
    HOTSPOT_STATE_FILE.unlink(missing_ok=True)


def _load_hotspot_state() -> dict | None:
    """Read hotspot state from the shared state file."""
    if not HOTSPOT_STATE_FILE.exists():
        return None
    try:
        return json.loads(HOTSPOT_STATE_FILE.read_text())
    except json.JSONDecodeError, OSError:
        return None


# ── Regdomain helpers ───────────────────────────────────────────────────
#
# All regdomain logic lives in ``server_support.regdomain`` (apply +
# verify) and ``server_support.settings_store`` (read precedence + paths).
# Imported at the top of this module for use by the AP lifecycle.


# ── WPA supplicant security status ──────────────────────────────────────


def _wpa_supplicant_ap_security() -> dict[str, str] | None:
    """Parse ``wpa_cli -i <iface> status`` for live AP security state.

    Returns a dict with keys ``mode``, ``key_mgmt``, ``pairwise_cipher``,
    ``group_cipher``, ``wpa_state`` — only the fields wpa_supplicant
    actually reports. Returns None on error.
    """
    try:
        proc = subprocess.run(
            ["sudo", "wpa_cli", "-i", WIFI_IFACE, "status"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except FileNotFoundError, subprocess.TimeoutExpired, OSError:
        return None
    if proc.returncode != 0:
        return None
    wanted = {"mode", "key_mgmt", "pairwise_cipher", "group_cipher", "wpa_state"}
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        if key in wanted:
            out[key] = val
    return out or None


# ── Captive portal ──────────────────────────────────────────────────────


def _setup_captive_portal(ap_ip: str) -> None:
    """Configure dnsmasq captive-portal hints + appliance mDNS resolution.

    Three layers:
      1. **DHCP option 114 (RFC 8910)** — advertise the portal URL at
         DHCP ACK. Modern iOS/Android pop the captive sheet from this
         alone, no probe required.
      2. **Appliance mDNS overrides** — ``meeting-<pin>.local`` and
         ``meeting-scribe-<id4>.local`` are mDNS-published by avahi,
         but the iOS CNA WebView (and many other captive clients)
         skip mDNS multicast and only ask the DHCP-supplied resolver
         (us). Without these overrides the post-PIN redirect to the
         cert-matching hostname hangs ~30 s on a stuck lookup
         (GB10 test 2026-05-13). Sourced from the cert SAN list so
         they stay in sync with what the leaf cert actually covers.
      3. **No DNS overrides for captive-probe hostnames** — the
         legacy guest flow used ``address=/#/<ap_ip>`` to pin every
         hostname (including ``captive.apple.com``) to the AP, which
         worked because legacy guests never browsed upstream. In the
         Phase H captive-gateway, pinning ``captive.apple.com`` to
         10.42.0.1 makes iOS try HTTPS to our server with the wrong
         cert (we serve ``meeting-<pin>.local``, not Apple's cert),
         triggering "Wi-Fi does not appear to be connected to the
         internet" (GB10 test 2026-05-13). Instead the captive probe
         is intercepted at the network layer: iptables PREROUTING
         REDIRECTs unauthorized tcp/80 to the local captive sub-app,
         and tcp/443 gets a TCP RST (REJECT --reject-with tcp-reset)
         so iOS sees Connection Refused — which Apple's CNA logic
         treats as "captive detected, follow DHCP-114".

    Port 80 is owned by meeting-scribe's in-process captive sub-app.
    """
    subprocess.run(
        ["sudo", "mkdir", "-p", str(DNSMASQ_CONF_DIR)],
        capture_output=True,
        timeout=5,
        check=False,
    )

    portal_url = f"http://{ap_ip}/"
    try:
        from meeting_scribe.cli._common import _required_leaf_dns_sans

        appliance_mdns = sorted(_required_leaf_dns_sans())
    except Exception:
        appliance_mdns = []
    appliance_lines = "\n".join(f"address=/{h}/{ap_ip}" for h in appliance_mdns)
    conf_content = (
        "# Appliance mDNS hostnames — the iOS CNA WebView and many\n"
        "# clients ask the DHCP-supplied resolver (us) instead of\n"
        "# multicasting, so we serve these directly to avoid stuck\n"
        "# DNS lookups when post-auth redirects target the cert-\n"
        "# matching hostname. Captive-probe hosts (captive.apple.com\n"
        "# etc.) are deliberately NOT in this list — pinning them to\n"
        "# us makes iOS HTTPS-probe-fail on the wrong cert, which\n"
        "# trips the 'not connected to internet' warning. The HTTP\n"
        "# probe is caught at the iptables layer (PREROUTING REDIRECT\n"
        "# on tcp/80); HTTPS probe gets tcp-reset → iOS reads that as\n"
        "# captive + falls back to the DHCP-114 hint below.\n"
        f"{appliance_lines}\n"
        "# RFC 8910 captive-portal URL (DHCP option 114) so the\n"
        "# captive sheet pops at DHCP ACK on iOS 14+/Android 11+.\n"
        f"dhcp-option=114,{portal_url}\n"
    )
    try:
        subprocess.run(
            ["sudo", "tee", str(DNSMASQ_CAPTIVE_CONF)],
            input=conf_content,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("could not write captive portal conf: %s", exc)

    _teardown_iptables()
    _stop_portal_redirector()


def _stop_portal_redirector() -> None:
    """Stop any stale captive-portal HTTP redirector processes.

    Cleans up BOTH meeting-scribe's PID files AND sddc-cli's PID file
    because both writers may bind port 80.
    """
    for pid_file in (PORTAL_PID_FILE, *MEETING_SCRIBE_PORTAL_PID_FILES):
        if pid_file.exists():
            try:
                pid_str = pid_file.read_text().strip()
                if pid_str:
                    pid = int(pid_str)
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError, PermissionError:
                        pass
            except OSError, ValueError:
                pass
            pid_file.unlink(missing_ok=True)

    # Kill orphaned captive portal processes
    for pattern in ("captive-portal-[48]", "meeting-captive-portal.py"):
        try:
            subprocess.run(
                ["pkill", "-f", pattern],
                capture_output=True,
                timeout=5,
                check=False,
            )
        except FileNotFoundError, subprocess.TimeoutExpired, OSError:
            pass


def _teardown_captive_portal() -> None:
    """Remove captive portal: stop redirectors, remove dnsmasq conf, clean iptables."""
    _stop_portal_redirector()
    if DNSMASQ_CAPTIVE_CONF.exists():
        try:
            subprocess.run(
                ["sudo", "rm", "-f", str(DNSMASQ_CAPTIVE_CONF)],
                capture_output=True,
                timeout=5,
                check=False,
            )
        except FileNotFoundError, subprocess.TimeoutExpired, OSError:
            pass
    _teardown_iptables()


def _teardown_iptables() -> None:
    """Remove legacy captive-portal iptables NAT rules (sddc-cli comment tag)."""
    sddc_comment = "meeting-captive-portal"
    for _i in range(10):
        try:
            result = subprocess.run(
                [
                    "sudo",
                    "iptables",
                    "-t",
                    "nat",
                    "-L",
                    "PREROUTING",
                    "-n",
                    "--line-numbers",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except FileNotFoundError, subprocess.TimeoutExpired, OSError:
            break
        if result.returncode != 0:
            break
        line_num = None
        for line in result.stdout.splitlines():
            if sddc_comment in line:
                parts = line.split()
                if parts and parts[0].isdigit():
                    line_num = parts[0]
                    break
        if line_num is None:
            break
        subprocess.run(
            ["sudo", "iptables", "-t", "nat", "-D", "PREROUTING", line_num],
            capture_output=True,
            timeout=5,
            check=False,
        )


def _captive_portal_active() -> bool:
    """Return True if the dnsmasq captive portal conf file exists."""
    return DNSMASQ_CAPTIVE_CONF.exists()


# ── Firewall ────────────────────────────────────────────────────────────


_MS_FW_COMMENT = "ms-fw"


def _scrub_netplan_ap_yaml() -> None:
    """Remove any ``/etc/netplan/90-NM-*.yaml`` files that reference
    our AP SSID or connection name.

    NetworkManager mirrors every connection it manages into a
    netplan YAML so netplan can render them at next boot. After
    a delete-and-readd of ``DellDemo-AP`` (which we do on every
    security-mode rotation — open ↔ sae) the OLD YAML survives.
    NM loads both on the next start and chokes on the now-invalid
    combinations (e.g. ``pmf=required`` from a prior SAE config
    paired with ``key-mgmt=owe`` from the new one), with the
    failure surfacing as ``supplicant-timeout``.

    Idempotent + best-effort; failures here are non-fatal because
    the netplan path is purely a next-boot concern.
    """
    import glob

    candidates = glob.glob("/etc/netplan/90-NM-*.yaml")
    removed: list[str] = []
    for path in candidates:
        try:
            with open(path) as fh:
                body = fh.read()
        except OSError:
            continue
        if AP_CON_NAME in body or "Dell Meeting" in body:
            try:
                subprocess.run(
                    ["sudo", "rm", "-f", path],
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
                removed.append(path)
            except FileNotFoundError, subprocess.TimeoutExpired, OSError:
                pass
    if removed:
        try:
            subprocess.run(
                ["sudo", "netplan", "generate"],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except FileNotFoundError, subprocess.TimeoutExpired, OSError:
            pass
        logger.info("scrubbed %d stale netplan AP yaml files", len(removed))


def _ms_fw_rules(
    ap_iface: str,
    *,
    egress_mode: str = "block",
    wan_ifaces: tuple[str, ...] = (),
    wan_admin_allowed: tuple[str, ...] = (),
) -> tuple[list[list[str]], list[str], list[list[str]], list[list[str]]]:
    """Build the per-iface INPUT + FORWARD allowlist (Phase H + WAN-WAN).

    Same rules for setup-mode and operating-mode — the AP iface
    doesn't change between modes, only the SSID + auth do. Tagged
    with ``--comment ms-fw`` so removal is exact.

    Returns ``(input_accepts, input_drop_catchall, forward_rules,
    nat_post_rules)``. Callers append per-protocol ACCEPTs (icmp,
    DHCP) BETWEEN ``input_accepts`` and ``input_drop_catchall`` so the
    DROP always lands LAST.

    ``egress_mode`` selects the posture:

    * ``"block"`` (default) — preserves today's zero-egress hotspot
      posture for meeting privacy. ``wan_ifaces`` / ``wan_admin_allowed``
      are ignored; the third element matches the legacy 3-tuple
      ``forward`` block byte-for-byte, ``nat_post_rules`` is empty.
    * ``"gateway"`` — permits AP↔WAN forwarding for the listed
      ``wan_ifaces`` and adds MASQUERADE on each. Additionally drops
      inbound new-state connections on WAN interfaces (so the admin
      UI is unreachable from upstream) — except for interfaces in
      ``wan_admin_allowed`` (typically ``("enP7s7",)`` to preserve
      emergency wired admin access).
    * ``"captive"`` — same shape as ``"gateway"`` but the AP→WAN
      ACCEPT requires ``-m set --match-set ms-allowed-admins src``.
      Unauthorized AP clients get DROP'd on FORWARD, and their
      outbound tcp/80 is REDIRECT'd to the local captive sub-app on
      port 80 (via a ``-t nat -A PREROUTING`` rule). The fourth tuple
      element therefore holds both PREROUTING and POSTROUTING rules
      in captive mode; ``-t nat`` is in each rule, so the apply path
      treats them uniformly.
    """
    # Returned tuple shape: (input_accepts, input_drop_catchall, forward_rules).
    # Caller appends per-protocol ACCEPTs (icmp, DHCP) BETWEEN
    # input_accepts and input_drop_catchall so the DROP always
    # lands LAST. Bug from the GB10 walkthrough — previously DROP
    # was inside common_input and the per-protocol appends
    # ended up unreachable.
    common_input = [
        # lo + conntrack accepts MUST be tagged with --comment ms-fw
        # so _ms_fw_remove can clean them on reload — otherwise they
        # accumulate across reapplies. Bug surfaced during the GB10
        # walkthrough (3 duplicate copies after the third apply).
        [
            "-A",
            "INPUT",
            "-i",
            "lo",
            "-m",
            "comment",
            "--comment",
            _MS_FW_COMMENT,
            "-j",
            "ACCEPT",
        ],
        [
            "-A",
            "INPUT",
            "-m",
            "conntrack",
            "--ctstate",
            "ESTABLISHED,RELATED",
            "-m",
            "comment",
            "--comment",
            _MS_FW_COMMENT,
            "-j",
            "ACCEPT",
        ],
        [
            "-A",
            "INPUT",
            "-i",
            ap_iface,
            "-p",
            "tcp",
            "--dport",
            "443",
            "-m",
            "comment",
            "--comment",
            _MS_FW_COMMENT,
            "-j",
            "ACCEPT",
        ],
        [
            "-A",
            "INPUT",
            "-i",
            ap_iface,
            "-p",
            "tcp",
            "--dport",
            "80",
            "-m",
            "comment",
            "--comment",
            _MS_FW_COMMENT,
            "-j",
            "ACCEPT",
        ],
        [
            "-A",
            "INPUT",
            "-i",
            ap_iface,
            "-p",
            "udp",
            "--dport",
            "53",
            "-m",
            "comment",
            "--comment",
            _MS_FW_COMMENT,
            "-j",
            "ACCEPT",
        ],
    ]
    drop_catchall = [
        "-A",
        "INPUT",
        "-i",
        ap_iface,
        "-m",
        "comment",
        "--comment",
        _MS_FW_COMMENT,
        "-j",
        "DROP",
    ]
    # ── FORWARD chain ──
    # In "block" mode (default), every AP→non-lo path is dropped — the
    # legacy zero-egress posture. In "gateway" mode, AP→WAN forwarding
    # is allowed (new + established + related), reverse is restricted
    # to established/related only, and AP↔AP isolation is preserved.
    forward: list[list[str]] = []
    nat_post: list[list[str]] = []

    if egress_mode == "block" or not wan_ifaces:
        forward = [
            [
                "-A",
                "FORWARD",
                "-i",
                ap_iface,
                "-o",
                "lo",
                "-m",
                "comment",
                "--comment",
                _MS_FW_COMMENT,
                "-j",
                "ACCEPT",
            ],
            [
                "-A",
                "FORWARD",
                "-i",
                ap_iface,
                "-o",
                ap_iface,
                "-m",
                "comment",
                "--comment",
                _MS_FW_COMMENT,
                "-j",
                "DROP",
            ],
            [
                "-A",
                "FORWARD",
                "-i",
                ap_iface,
                "!",
                "-o",
                "lo",
                "-m",
                "comment",
                "--comment",
                _MS_FW_COMMENT,
                "-j",
                "DROP",
            ],
        ]
    else:
        # ── gateway / captive FORWARD ──
        # 1) AP↔AP isolation preserved (both modes).
        forward.append(
            [
                "-A",
                "FORWARD",
                "-i",
                ap_iface,
                "-o",
                ap_iface,
                "-m",
                "comment",
                "--comment",
                _MS_FW_COMMENT,
                "-j",
                "DROP",
            ]
        )
        is_captive = egress_mode == "captive"
        for wan in wan_ifaces:
            # AP → WAN: new/established/related allowed.
            # In captive mode, additionally require src IP to be in
            # the ``ms-allowed-admins`` ipset — only signed-in admins
            # reach the WAN. Everyone else falls through to the
            # explicit DROP below.
            ap_to_wan_accept = [
                "-A",
                "FORWARD",
                "-i",
                ap_iface,
                "-o",
                wan,
            ]
            if is_captive:
                ap_to_wan_accept += [
                    "-m",
                    "set",
                    "--match-set",
                    "ms-allowed-admins",
                    "src",
                ]
            ap_to_wan_accept += [
                "-m",
                "conntrack",
                "--ctstate",
                "NEW,ESTABLISHED,RELATED",
                "-m",
                "comment",
                "--comment",
                _MS_FW_COMMENT,
                "-j",
                "ACCEPT",
            ]
            forward.append(ap_to_wan_accept)
            if is_captive:
                # Fail-fast HTTPS: an unauthorized client that types a
                # hostname into a modern browser tries tcp/443 first
                # (HTTPS-by-default upgrade). A silent DROP leaves the
                # browser TCP-timing out for ~30 s with no visible
                # feedback; REJECT --reject-with tcp-reset answers with
                # a RST so the browser surfaces "Connection refused"
                # immediately. The user then notices the OS captive
                # sheet (already open) or retries via HTTP, which the
                # nat PREROUTING REDIRECT below catches. Authorized
                # admins already hit the ACCEPT above and never reach
                # this rule.
                forward.append(
                    [
                        "-A",
                        "FORWARD",
                        "-i",
                        ap_iface,
                        "-o",
                        wan,
                        "-p",
                        "tcp",
                        "--dport",
                        "443",
                        "-m",
                        "set",
                        "!",
                        "--match-set",
                        "ms-allowed-admins",
                        "src",
                        "-m",
                        "comment",
                        "--comment",
                        _MS_FW_COMMENT,
                        "-j",
                        "REJECT",
                        "--reject-with",
                        "tcp-reset",
                    ]
                )
                # Explicit default-deny for AP→WAN traffic from clients
                # who aren't in the admins ipset. Belt-and-suspenders
                # alongside the unconditional default-DROP that follows
                # the WAN→AP rules.
                forward.append(
                    [
                        "-A",
                        "FORWARD",
                        "-i",
                        ap_iface,
                        "-o",
                        wan,
                        "-m",
                        "comment",
                        "--comment",
                        _MS_FW_COMMENT,
                        "-j",
                        "DROP",
                    ]
                )
            # WAN → AP: only return traffic (established/related).
            # Stateless inbound from upstream cannot reach AP clients.
            forward.append(
                [
                    "-A",
                    "FORWARD",
                    "-i",
                    wan,
                    "-o",
                    ap_iface,
                    "-m",
                    "conntrack",
                    "--ctstate",
                    "ESTABLISHED,RELATED",
                    "-m",
                    "comment",
                    "--comment",
                    _MS_FW_COMMENT,
                    "-j",
                    "ACCEPT",
                ]
            )
            forward.append(
                [
                    "-A",
                    "FORWARD",
                    "-i",
                    wan,
                    "-o",
                    ap_iface,
                    "-m",
                    "comment",
                    "--comment",
                    _MS_FW_COMMENT,
                    "-j",
                    "DROP",
                ]
            )
            # MASQUERADE the AP subnet on egress to this WAN.
            nat_post.append(
                [
                    "-t",
                    "nat",
                    "-A",
                    "POSTROUTING",
                    "-s",
                    HOTSPOT_SUBNET_CIDR,
                    "-o",
                    wan,
                    "-m",
                    "comment",
                    "--comment",
                    _MS_FW_COMMENT,
                    "-j",
                    "MASQUERADE",
                ]
            )

        if is_captive:
            # Captive: AP-iface tcp/80 from anyone NOT in the admins
            # ipset gets REDIRECT'd to the local captive sub-app on
            # port 80. ONLY admins skip the redirect — they actually
            # reach upstream via the AP→WAN ACCEPT rule above.
            #
            # Guests deliberately remain inside the REDIRECT so iOS'
            # background captive probe (captive.apple.com:80) keeps
            # landing on our handler. ``_is_captive_acked`` returns
            # True for guest ipset membership, so the handler answers
            # with Apple's Success body — that's what triggers the
            # blue Done tick in the iOS CNA after PIN entry (GB10
            # test 2026-05-13). Excluding guests from the REDIRECT
            # made the probe fall through to the FORWARD chain's
            # default DROP, the probe timed out, and iOS never saw
            # Success → no Done button → CNA stayed stuck.
            #
            # Side effect: guests trying to browse arbitrary external
            # HTTP also land on our sign-in page. That's acceptable —
            # guests aren't granted WAN by design, and the meeting
            # UI lives on HTTPS so it isn't affected.
            nat_post.append(
                [
                    "-t",
                    "nat",
                    "-A",
                    "PREROUTING",
                    "-i",
                    ap_iface,
                    "-p",
                    "tcp",
                    "--dport",
                    "80",
                    "-m",
                    "set",
                    "!",
                    "--match-set",
                    "ms-allowed-admins",
                    "src",
                    "-m",
                    "comment",
                    "--comment",
                    _MS_FW_COMMENT,
                    "-j",
                    "REDIRECT",
                    "--to-ports",
                    "80",
                ]
            )

        # ── gateway mode WAN-side INPUT ──
        # For each WAN iface NOT in wan_admin_allowed, explicitly drop
        # new inbound TCP on 22/80/443 so the admin UI is unreachable
        # from upstream. The existing conntrack ESTABLISHED,RELATED
        # ACCEPT rule near the top of the INPUT chain still lets
        # return traffic for outbound connections through.
        #
        # For interfaces IN wan_admin_allowed (typically enP7s7), do
        # the opposite — explicit ACCEPT so the wired admin path stays
        # open even if some upstream policy tightens the default.
        for wan in wan_ifaces:
            allow_admin = wan in wan_admin_allowed
            verb = "ACCEPT" if allow_admin else "DROP"
            for port in ("22", "80", "443"):
                common_input.append(
                    [
                        "-A",
                        "INPUT",
                        "-i",
                        wan,
                        "-p",
                        "tcp",
                        "--dport",
                        port,
                        "-m",
                        "conntrack",
                        "--ctstate",
                        "NEW",
                        "-m",
                        "comment",
                        "--comment",
                        _MS_FW_COMMENT,
                        "-j",
                        verb,
                    ]
                )

    return common_input, drop_catchall, forward, nat_post


def _ms_fw_remove(binary: str = "iptables") -> None:
    """Idempotent remove: delete every rule that carries ``--comment ms-fw``.

    Loops chains until no tagged rule remains. ``binary`` is
    ``iptables`` or ``ip6tables``; the rule format is identical.
    Also sweeps the ``nat`` table's ``POSTROUTING`` chain so MASQUERADE
    rules tagged ``ms-fw`` are removed cleanly on a posture flip.
    """
    chain_specs: tuple[tuple[str | None, str], ...] = (
        (None, "INPUT"),
        (None, "FORWARD"),
        ("nat", "POSTROUTING"),
        # Phase H added captive REDIRECT in nat PREROUTING. Without
        # this sweep, a reconcile after a posture change duplicates
        # the rule each time it's re-emitted.
        ("nat", "PREROUTING"),
    )
    for table, chain in chain_specs:
        if binary == "ip6tables" and table == "nat":
            # ip6tables nat is gated behind ip6t_nat; we don't emit v6
            # MASQUERADE rules in v1 (no v6 forwarding), so skip.
            continue
        for _ in range(40):
            list_argv: list[str] = ["sudo", binary]
            if table:
                list_argv += ["-t", table]
            list_argv += ["-S", chain]
            try:
                proc = subprocess.run(
                    list_argv,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            except FileNotFoundError, subprocess.TimeoutExpired, OSError:
                return
            found = False
            for line in (proc.stdout or "").splitlines():
                if _MS_FW_COMMENT in line:
                    del_rule = line.replace(f"-A {chain}", f"-D {chain}", 1)
                    del_argv: list[str] = ["sudo", binary]
                    if table:
                        del_argv += ["-t", table]
                    del_argv += del_rule.split()
                    subprocess.run(
                        del_argv,
                        capture_output=True,
                        timeout=5,
                        check=False,
                    )
                    found = True
                    break
            if not found:
                break


_SYSCTL_GATEWAY_CONF = Path("/etc/sysctl.d/99-meeting-scribe-gateway.conf")


def _write_sysctl_conf(egress_mode: str) -> None:
    """Persist the IPv4-forward sysctl based on egress mode.

    ``gateway`` ⇒ ``net.ipv4.ip_forward=1``; ``block`` ⇒ ``=0``.
    v6 forwarding stays off in both modes (v1 scope).
    The file lives under ``/etc/sysctl.d/`` so it survives reboot;
    ``sysctl -p`` runs after the write to apply immediately.
    """
    # Both ``gateway`` and ``captive`` need IPv4 forwarding on; the
    # difference is at the FORWARD/PREROUTING chain level (per-IP
    # gating in captive), not at the kernel sysctl level.
    v4_forward = "1" if egress_mode in ("gateway", "captive") else "0"
    body = (
        "# Managed by meeting-scribe (wifi.py reconcile_network_state)\n"
        "# v1 scope: IPv4 forwarding only; v6 stays off.\n"
        f"net.ipv4.ip_forward = {v4_forward}\n"
        "net.ipv6.conf.all.forwarding = 0\n"
    )
    try:
        tmp = _SYSCTL_GATEWAY_CONF.with_suffix(_SYSCTL_GATEWAY_CONF.suffix + ".tmp")
        subprocess.run(
            ["sudo", "tee", str(tmp)],
            input=body,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        subprocess.run(
            ["sudo", "mv", str(tmp), str(_SYSCTL_GATEWAY_CONF)],
            capture_output=True,
            timeout=5,
            check=False,
        )
        subprocess.run(
            ["sudo", "sysctl", "-p", str(_SYSCTL_GATEWAY_CONF)],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("write sysctl conf failed: %s", exc)


def _enforce_ip6tables_forward_drop() -> None:
    """Set ``ip6tables -P FORWARD DROP`` so no v6 packet is ever forwarded.

    Idempotent — re-applying when already DROP is a no-op.
    """
    try:
        subprocess.run(
            ["sudo", "ip6tables", "-P", "FORWARD", "DROP"],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("ip6tables FORWARD policy DROP failed: %s", exc)


def _apply_simple_firewall(
    ap_iface: str = WIFI_IFACE,
    *,
    egress_mode: str = "block",
    wan_ifaces: tuple[str, ...] = (),
    wan_admin_allowed: tuple[str, ...] = (),
) -> None:
    """Phase H + WAN: dual-stack INPUT + FORWARD + NAT POSTROUTING.

    The default invocation (no kwargs) preserves the legacy zero-egress
    hotspot posture for meeting privacy: AP→WAN forwarding is dropped,
    no MASQUERADE, INPUT only accepts AP-iface 80/443/53/DHCP/ICMP plus
    the global lo + conntrack rules. **Existing callers see byte-identical
    behavior.**

    In ``egress_mode="gateway"`` the function additionally:
      * Allows AP↔WAN forwarding (new + established/related outbound,
        established/related only inbound) for each ``wan_ifaces`` entry.
      * Adds MASQUERADE on each WAN iface (v4 only).
      * Inserts explicit WAN-side INPUT deny rules for 22/80/443 (unless
        the iface is in ``wan_admin_allowed``, in which case ACCEPT).
      * Writes ``/etc/sysctl.d/99-meeting-scribe-gateway.conf`` with
        ``net.ipv4.ip_forward=1``.
      * Sets ``ip6tables -P FORWARD DROP`` (no v6 forwarding in v1).

    Always idempotent — removes ALL ``ms-fw``-tagged rules first.
    """
    _write_sysctl_conf(egress_mode)

    for binary in ("iptables", "ip6tables"):
        _ms_fw_remove(binary)
        v4 = binary == "iptables"
        # v6 never gets WAN-iface rules in v1 because the STA profile
        # has ``ipv6.method=disabled`` and we leave the wired iface's
        # v6 stack untouched. Pass empty wan_ifaces to the v6 builder.
        if v4:
            common_input, drop_catchall, forward, nat_post = _ms_fw_rules(
                ap_iface,
                egress_mode=egress_mode,
                wan_ifaces=wan_ifaces,
                wan_admin_allowed=wan_admin_allowed,
            )
        else:
            common_input, drop_catchall, forward, nat_post = _ms_fw_rules(
                ap_iface,
                egress_mode=egress_mode,
                wan_ifaces=(),
                wan_admin_allowed=(),
            )
        rules: list[list[str]] = []
        rules.extend(common_input)
        # Per-protocol ACCEPTs go BEFORE the catch-all DROP.
        rules.append(
            [
                "-A",
                "INPUT",
                "-i",
                ap_iface,
                "-p",
                "icmp" if v4 else "ipv6-icmp",
                "-m",
                "comment",
                "--comment",
                _MS_FW_COMMENT,
                "-j",
                "ACCEPT",
            ]
        )
        # DHCP server side — v4 listens on 67, v6 on 547. Harmless
        # if the AP iface has no v6 addressing (Phase G keeps
        # ipv6.method=disabled on the AP profile).
        if v4:
            rules.append(
                [
                    "-A",
                    "INPUT",
                    "-i",
                    ap_iface,
                    "-p",
                    "udp",
                    "--dport",
                    "67",
                    "-m",
                    "comment",
                    "--comment",
                    _MS_FW_COMMENT,
                    "-j",
                    "ACCEPT",
                ]
            )
        else:
            rules.append(
                [
                    "-A",
                    "INPUT",
                    "-i",
                    ap_iface,
                    "-p",
                    "udp",
                    "--dport",
                    "547",
                    "-m",
                    "comment",
                    "--comment",
                    _MS_FW_COMMENT,
                    "-j",
                    "ACCEPT",
                ]
            )
        # Catch-all DROP comes LAST in the INPUT block so every
        # ACCEPT above gets a chance to match first.
        rules.append(drop_catchall)
        # Append FORWARD rules.
        rules.extend(forward)
        # INPUT rules apply via -A (append) so they end up in the EXACT
        # order listed: per-port ACCEPTs first, catch-all DROP last
        # (2026-05-05 fix — using -I 1 reversed the block).
        #
        # FORWARD rules need a different strategy: NetworkManager's
        # ``ipv4.method=shared`` adds an ``nm-sh-fw-<iface>`` jump at the
        # top of FORWARD that unconditionally ACCEPTs the AP subnet,
        # which would bypass our captive ipset gate entirely (Phase H
        # GB10 walkthrough 2026-05-13: phone got internet immediately
        # after auth flush because NM's chain matched first). So we
        # convert ``-A FORWARD`` to ``-I FORWARD <pos>`` and track an
        # explicit position counter — pos=1 for the first FORWARD rule,
        # 2 for the second, etc. This puts our entire ms-fw FORWARD
        # block at the head of the chain, above ``nm-sh-fw-*``, while
        # preserving the relative order the rule generator emits.
        forward_pos = 0
        for idx, rule in enumerate(rules, start=1):
            cmd = list(rule)
            if len(cmd) >= 2 and cmd[0] == "-A" and cmd[1] == "FORWARD":
                forward_pos += 1
                cmd = ["-I", "FORWARD", str(forward_pos)] + cmd[2:]
            try:
                subprocess.run(
                    ["sudo", binary] + cmd,
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
                logger.warning("%s rule %d failed: %s", binary, idx, exc)
        # NAT POSTROUTING rules (v4 only — no NAT66 in v1).
        if v4 and nat_post:
            for idx, rule in enumerate(nat_post, start=1):
                try:
                    subprocess.run(
                        ["sudo", binary] + rule,
                        capture_output=True,
                        timeout=5,
                        check=False,
                    )
                except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
                    logger.warning("%s nat rule %d failed: %s", binary, idx, exc)

    # v6 FORWARD policy DROP — enforced even when egress_mode == "block"
    # so this stays a steady-state invariant. No-op when policy is
    # already DROP.
    _enforce_ip6tables_forward_drop()
    logger.info(
        "ms-fw applied on %s (egress_mode=%s wan_ifaces=%s wan_admin_allowed=%s)",
        ap_iface,
        egress_mode,
        wan_ifaces,
        wan_admin_allowed,
    )


# ── Reconciliation entry point ────────────────────────────────


async def reconcile_network_state() -> None:
    """Single authoritative trigger that re-applies the correct firewall
    and sysctl for the current settings + interface state.

    Called from every CLI/REST/boot transition (``wifi_up``, ``wifi_down``,
    ``wan_up``, ``wan_down``, boot lifespan). Pure function of:

      * ``settings_store._effective_wan_egress_mode()``
      * AP iface name (constant ``WIFI_IFACE``)
      * Wired iface name + presence  (we always claim ``enP7s7``)
      * STA iface presence (``/sys/class/net/wlan_sta``)

    Idempotent — five sequential calls produce identical
    ``iptables-save`` output. Removes all ``ms-fw``-tagged rules before
    re-applying so there is no rule drift across calls.
    """
    from meeting_scribe.server_support import settings_store
    from meeting_scribe.wifi_sta import STA_IFACE, sta_iface_present

    egress_mode = settings_store._effective_wan_egress_mode()
    wan_ifaces: list[str] = []
    wan_admin_allowed: list[str] = []

    # The wired iface is always treated as WAN — and as admin-allowed,
    # so the emergency wired admin path is never closed.
    wired_iface = "enP7s7"
    if Path(f"/sys/class/net/{wired_iface}").exists():
        wan_ifaces.append(wired_iface)
        wan_admin_allowed.append(wired_iface)

    if sta_iface_present():
        wan_ifaces.append(STA_IFACE)
        # wlan_sta is NOT admin-allowed — explicit INPUT deny enforced.

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        lambda: _apply_simple_firewall(
            WIFI_IFACE,
            egress_mode=egress_mode,
            wan_ifaces=tuple(wan_ifaces),
            wan_admin_allowed=tuple(wan_admin_allowed),
        ),
    )


def _remove_firewall() -> None:
    """Remove all firewall rules tagged with our iptables comment."""
    for chain in ("INPUT", "FORWARD"):
        for _ in range(20):
            try:
                result = subprocess.run(
                    ["sudo", "iptables", "-S", chain],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            except FileNotFoundError, subprocess.TimeoutExpired, OSError:
                break
            found = False
            for line in (result.stdout or "").splitlines():
                if HOTSPOT_IPTABLES_COMMENT in line:
                    del_rule = line.replace(f"-A {chain}", f"-D {chain}", 1)
                    subprocess.run(
                        ["sudo", "iptables"] + del_rule.split(),
                        capture_output=True,
                        timeout=5,
                        check=False,
                    )
                    found = True
                    break
            if not found:
                break

    # NAT PREROUTING
    for _ in range(20):
        try:
            result = subprocess.run(
                ["sudo", "iptables", "-t", "nat", "-S", "PREROUTING"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except FileNotFoundError, subprocess.TimeoutExpired, OSError:
            break
        found = False
        for line in (result.stdout or "").splitlines():
            if HOTSPOT_IPTABLES_COMMENT in line:
                del_rule = line.replace("-A PREROUTING", "-D PREROUTING", 1)
                subprocess.run(
                    ["sudo", "iptables", "-t", "nat"] + del_rule.split(),
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
                found = True
                break
        if not found:
            break
    logger.info("Hotspot firewall removed")


# ── Internal AP lifecycle (unlocked) ───────────────────────────────────


async def _bring_up_ap(cfg: WifiConfig) -> None:
    """Bring up the WiFi AP with the given config. NOT locked — caller holds lock.

    Steps:
      1. Ensure regdomain (persistent + runtime)
      2. Captive portal: setup for meeting, teardown for admin
      3. Create/modify NM AP profile with WPA3-SAE, PMF required
      4. nmcli con up, poll for activation
      5. Apply mode-appropriate firewall
      6. Write state file with mode
    """
    loop = asyncio.get_event_loop()

    # Step 1: regdomain. Best-effort — the kernel may override our `iw
    # reg set` if STA is associated and the upstream AP broadcasts a
    # country IE. We log the divergence but proceed; the kernel
    # enforces correct regulatory rules at TX time regardless of which
    # block (`global` vs `phy#0`) reports the value.
    await loop.run_in_executor(None, _ensure_regdomain_persistent)
    if not await loop.run_in_executor(None, _ensure_regdomain):
        live = _current_regdomain()
        logger.warning(
            "regdomain divergence: target=%s, live=%r — kernel will enforce "
            "the live regdomain's rules for AP TX power. Common cause: STA "
            "is associated and the upstream AP's country IE override is "
            "applying. Proceeding.",
            cfg.regdomain,
            live,
        )
    else:
        logger.info("regdomain verified: %s", _current_regdomain())

    # Step 2: captive portal. Two layers — dnsmasq wildcard DNS
    # (``address=/#/<ap_ip>``) so every hostname the client looks up
    # resolves to the AP, plus DHCP option 114 (RFC 8910) so the OS
    # captive sheet auto-pops the portal URL at lease time.
    #
    # Required whenever the AP wants unauthorized clients trapped on
    # the local sign-in page:
    #   * "setup"   — first-touch wizard (legacy guest portal flow).
    #   * "meeting" — live-transcription guest portal.
    #   * "admin"   — Phase H captive-gateway. Without wildcard DNS,
    #                 the phone resolves real upstream IPs, tries
    #                 tcp/443, hits our FORWARD REJECT, and sees
    #                 nothing load (no captive sheet ever pops
    #                 because the probe to ``captive.apple.com``
    #                 also gets reset). Bug surfaced 2026-05-13.
    #
    # Cleared when admin mode is paired with the open egress modes
    # ("block" / "gateway") — there is no per-client captive surface
    # to trap clients onto.
    from meeting_scribe.server_support import settings_store as _settings_store

    _current_egress_mode = _settings_store._effective_wan_egress_mode()
    needs_captive_dns = cfg.mode in ("setup", "meeting") or (
        cfg.mode == "admin" and _current_egress_mode == "captive"
    )
    if needs_captive_dns:
        await loop.run_in_executor(None, _setup_captive_portal, cfg.ap_ip)
        logger.info(
            "captive portal configured (probe DNS + DHCP 114) for mode=%s egress=%s",
            cfg.mode,
            _current_egress_mode,
        )
    else:
        await loop.run_in_executor(None, _teardown_captive_portal)
        logger.info(
            "captive portal cleared (mode=%s egress=%s)",
            cfg.mode,
            _current_egress_mode,
        )

    # Step 3: create or modify NM profile. ``security`` controls
    # whether wifi-sec.* arguments are emitted at all — open APs use
    # an empty key-mgmt and skip the PSK entirely. The transition mode
    # (``sae-transition``) drops PMF to ``optional`` so WPA2-PSK
    # clients can still associate.
    sec_args: list[str] = []
    if cfg.security == "open":
        # "Open" first-touch on modern phones uses OWE
        # (Opportunistic Wireless Encryption, RFC 8110): phones
        # connect with no password prompt (looks like an open AP),
        # but the link is DH-encrypted and the 4-way handshake
        # authorizes the station so traffic actually flows. iOS
        # 13+, Android 10+, macOS Big Sur+, Win 11+ all support it.
        #
        # Why not truly-open (key-mgmt=none): NetworkManager treats
        # key-mgmt=none as legacy-WEP and demands a wep-key. There
        # is no clean way through NM to produce a no-security AP.
        # Even when NM accepts the profile, wpa_supplicant's
        # AP-mode authenticator runs the EAPOL framework against
        # the station, deauths the client after 1s, and starts an
        # endless EAP-Identity retransmit. Confirmed on the
        # MT7925 / wpa_supplicant 2.10 / NM 1.46 stack of the
        # customer GB10 (2026-05-05).
        #
        # OWE goes through the same association+handshake path the
        # WPA3-SAE admin/meeting modes already exercise, so we
        # inherit the known-working radio behavior. Without "owe-tm"
        # transition mode, phones older than ~2019 will not see
        # this AP — acceptable for v1.0 (operator-provisioned
        # appliance ships with a recent phone for setup).
        sec_args = [
            "wifi-sec.key-mgmt",
            "owe",
            "wifi-sec.pmf",
            "required",
            "wifi-sec.proto",
            "",
            "wifi-sec.pairwise",
            "",
            "wifi-sec.group",
            "",
            "wifi-sec.psk",
            "",
            "wifi-sec.auth-alg",
            "",
            "wifi-sec.wep-key0",
            "",
            "wifi-sec.wep-key1",
            "",
            "wifi-sec.wep-key2",
            "",
            "wifi-sec.wep-key3",
            "",
        ]
    else:
        sec_args = [
            "wifi-sec.key-mgmt",
            "sae",
            "wifi-sec.pmf",
            "required",
            "wifi-sec.proto",
            "rsn",
            "wifi-sec.pairwise",
            "ccmp",
            "wifi-sec.group",
            "ccmp",
            "wifi-sec.psk",
            cfg.password,
        ]

    exists = await loop.run_in_executor(None, _nmcli_connection_exists)
    # Open AP transitions: an existing WPA3-SAE profile carries
    # ``802-11-wireless-security.*`` properties that nmcli refuses
    # to "modify away" cleanly (we tried setting them to empty;
    # nmcli responds with "key-mgmt: property is missing"). The
    # bulletproof path is to DELETE the profile and re-add it
    # fresh without any wifi-sec args at all. Bug surfaced on the
    # 2026-05-05 GB10 walkthrough.
    if exists and cfg.security == "open":
        await loop.run_in_executor(
            None,
            lambda: _run_nmcli_sync(
                ["con", "delete", AP_CON_NAME],
                timeout=_NMCLI_CON_DOWN_TIMEOUT,
            ),
        )
        # Also nuke any netplan-mirrored YAML entries — NM dumps
        # every connection it manages into /etc/netplan/90-NM-*.yaml
        # for next-boot rendering, and the stale AP YAML from a
        # previous incompatible mode (SAE pmf=required + open
        # key-mgmt=none) makes NM choke at next start with
        # "supplicant-timeout". Self-heal here so an in-place
        # security-mode rotation always lands on a clean state.
        await loop.run_in_executor(None, _scrub_netplan_ap_yaml)
        exists = False
    if exists:
        proc = await loop.run_in_executor(
            None,
            lambda: _run_nmcli_sync(
                [
                    "con",
                    "modify",
                    AP_CON_NAME,
                    "wifi.ssid",
                    cfg.ssid,
                    "wifi.band",
                    cfg.band,
                    "wifi.channel",
                    str(cfg.channel),
                    *sec_args,
                ],
                timeout=_NMCLI_CON_MODIFY_TIMEOUT,
            ),
        )
        if proc.returncode != 0:
            # Phase H+ contract: a failed modify is fatal. Previously
            # we warned + continued, which let the wifi up CLI claim
            # success even though the NM profile still carried the
            # OLD security args (the DellDemo-AP setup-mode rotation
            # hit this on 2026-05-05 — pmf=required from prior SAE
            # config blocked the open-mode key-mgmt change). Raise
            # so the caller surfaces the error to the operator.
            raise RuntimeError(
                f"nmcli con modify failed (rc={proc.returncode}): {proc.stderr.strip()[:300]}"
            )
        logger.info("NM profile updated: ssid=%s security=%s", cfg.ssid, cfg.security)
    else:
        # Fresh add. For open mode we omit wifi-sec args entirely
        # so nmcli doesn't auto-populate a stub security section.
        add_sec_args: list[str] = [] if cfg.security == "open" else sec_args
        proc = await loop.run_in_executor(
            None,
            lambda: _run_nmcli_sync(
                [
                    "con",
                    "add",
                    "type",
                    "wifi",
                    "ifname",
                    WIFI_IFACE,
                    "con-name",
                    AP_CON_NAME,
                    "autoconnect",
                    "no",
                    "ssid",
                    cfg.ssid,
                    "wifi.mode",
                    "ap",
                    "wifi.band",
                    cfg.band,
                    "wifi.channel",
                    str(cfg.channel),
                    "ipv4.method",
                    "shared",
                    "ipv6.method",
                    "shared",
                    *add_sec_args,
                ],
                timeout=_NMCLI_CON_MODIFY_TIMEOUT,
            ),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"nmcli con add failed (rc={proc.returncode}): {proc.stderr.strip()[:200]}"
            )
        logger.info("NM profile created: ssid=%s security=%s", cfg.ssid, cfg.security)

    # Step 4: bring up the AP
    if await loop.run_in_executor(None, _nmcli_ap_is_active):
        await loop.run_in_executor(
            None,
            lambda: _run_nmcli_sync(["con", "down", AP_CON_NAME], timeout=_NMCLI_CON_DOWN_TIMEOUT),
        )
        # Driver beat between an existing-profile down and the
        # subsequent up; tests monkeypatch this to 0 via the
        # constant.
        await asyncio.sleep(_AP_BOUNCE_DRIVER_SETTLE_S)

    up_proc = await loop.run_in_executor(
        None,
        lambda: _run_nmcli_sync(["con", "up", AP_CON_NAME], timeout=_NMCLI_CON_UP_TIMEOUT),
    )
    if up_proc.returncode != 0:
        logger.warning(
            "nmcli con up returned rc=%d (stderr=%s) — waiting for NM auto-retry",
            up_proc.returncode,
            up_proc.stderr.strip()[:120] or "<empty>",
        )

    # Poll for activation
    active = False
    deadline = asyncio.get_event_loop().time() + _AP_ACTIVATION_WAIT_SECONDS
    while asyncio.get_event_loop().time() < deadline:
        if await loop.run_in_executor(None, _nmcli_ap_is_active):
            active = True
            break
        await asyncio.sleep(_AP_ACTIVATION_POLL_INTERVAL)

    if not active:
        raise RuntimeError(f"AP did not become active within {_AP_ACTIVATION_WAIT_SECONDS}s")
    logger.info("AP active (WPA3-SAE, PMF required)")

    # Step 5: apply firewall. Route through reconcile_network_state so
    # the persisted ``wan_egress_mode`` (block / gateway / captive) is
    # honored — calling _apply_simple_firewall directly would force
    # the default ``block`` posture and silently lose the captive
    # gating that the migration ladder just installed.
    await reconcile_network_state()

    # Step 6: write state file
    await loop.run_in_executor(None, lambda: _write_hotspot_state_sync(mode=cfg.mode))
    logger.info("hotspot state written: mode=%s ssid=%s", cfg.mode, cfg.ssid)


async def _teardown_ap() -> None:
    """Tear down the WiFi AP. NOT locked — caller holds lock.

    Steps:
      1. Remove firewall rules
      2. Teardown captive portal
      3. nmcli con down + delete
      4. Clear state file
    """
    loop = asyncio.get_event_loop()

    # Step 1: firewall
    await loop.run_in_executor(None, _remove_firewall)

    # Step 2: captive portal
    await loop.run_in_executor(None, _teardown_captive_portal)

    # Step 3: nmcli
    if await loop.run_in_executor(None, _nmcli_ap_is_active):
        await loop.run_in_executor(
            None,
            lambda: _run_nmcli_sync(["con", "down", AP_CON_NAME], timeout=_NMCLI_CON_DOWN_TIMEOUT),
        )
    if await loop.run_in_executor(None, _nmcli_connection_exists):
        await loop.run_in_executor(
            None,
            lambda: _run_nmcli_sync(
                ["con", "delete", AP_CON_NAME], timeout=_NMCLI_CON_DOWN_TIMEOUT
            ),
        )

    # Step 4: state file
    _clear_hotspot_state()

    # Step 5: captive-gateway ipsets — drop both sets so a future AP
    # bring-up re-creates them clean. Best-effort: failure here is
    # logged but never fails the teardown.
    try:
        from meeting_scribe.server_support import firewall_allowlist

        await firewall_allowlist.destroy_sets()
    except Exception:
        logger.exception("captive: destroy_sets failed (non-fatal)")
    logger.info("AP torn down and state cleared")


def _build_live_config() -> WifiConfig | None:
    """Build a WifiConfig from the live AP state (nmcli). Returns None if no AP."""
    if not _nmcli_ap_is_active():
        return None
    creds = _nmcli_read_live_ap_credentials()
    if creds is None:
        return None
    ssid, psk = creds

    # Read band/channel from the live profile
    proc = _run_nmcli_sync(
        [
            "-t",
            "-f",
            "802-11-wireless.band,802-11-wireless.channel",
            "con",
            "show",
            AP_CON_NAME,
        ],
        timeout=_NMCLI_CON_SHOW_TIMEOUT,
    )
    fields = _parse_nmcli_fields(proc.stdout) if proc.returncode == 0 else {}
    band = fields.get("802-11-wireless.band", DEFAULT_BAND) or DEFAULT_BAND
    channel_str = fields.get("802-11-wireless.channel", str(DEFAULT_CHANNEL))
    try:
        channel = int(channel_str)
    except ValueError, TypeError:
        channel = DEFAULT_CHANNEL

    # Determine mode from state file
    state = _load_hotspot_state()
    mode = (state or {}).get("mode", "meeting")

    return WifiConfig(
        mode=mode,
        ssid=ssid,
        password=psk,
        band=band,
        channel=channel,
        regdomain=_effective_regdomain(),
        ap_ip=AP_IP,
    )


# ── Public API ──────────────────────────────────────────────────────────


def build_config(
    mode: str,
    ssid: str | None = None,
    password: str | None = None,
    band: str = DEFAULT_BAND,
    channel: int = DEFAULT_CHANNEL,
) -> WifiConfig:
    """Build a WifiConfig for the requested mode.

    For admin mode: reads admin_ssid/admin_password from settings.
    For meeting mode: generates a session_id + random password.
    Validates fields and returns a frozen WifiConfig.
    """
    if mode not in ("off", "meeting", "admin"):
        raise ValueError(f"Invalid wifi mode: {mode!r}")

    regdomain = _effective_regdomain()

    if mode == "admin":
        settings = _load_settings_override()
        if not ssid:
            ssid = settings.get("admin_ssid", "") or DEFAULT_ADMIN_SSID
        # Admin mode runs OWE (Opportunistic Wireless Encryption) — no
        # WPA password prompt on the join screen. The actual auth gate
        # is the captive-gateway Phase H ipset, populated only after
        # the user signs in via /auth with ``DellMeetingAdmin<pin>``.
        # The legacy random ``admin_password`` in settings.json was the
        # AP-layer WPA-PSK; with captive gateway it's redundant and is
        # ignored from now on (kept in the file for downgrade safety).
        password = ""
    elif mode == "meeting":
        session_id = secrets.token_hex(2).upper()
        if not ssid:
            ssid = f"{SSID_PREFIX} {session_id}"
        if not password:
            password = secrets.token_hex(4).upper()
    else:
        # mode == "off" — placeholder config
        ssid = ssid or ""
        password = password or ""

    return WifiConfig(
        mode=mode,
        ssid=ssid,
        password=password,
        band=band,
        channel=channel,
        regdomain=regdomain,
        ap_ip=AP_IP,
        # Admin mode joins setup mode in OWE-land — no WPA prompt; the
        # Phase H captive gateway is the application-layer gate.
        security="open" if mode == "admin" else "sae",
    )


async def wifi_up(cfg: WifiConfig) -> None:
    """Bring up the WiFi AP in the requested mode. Thread-safe."""
    async with _wifi_lock:
        updates: dict[str, Any] = {"wifi_mode": cfg.mode}
        if cfg.mode == "admin":
            updates["admin_ssid"] = cfg.ssid
            # Admin mode is OWE post-Phase-H — no WPA-PSK. Clear any
            # stale ``admin_password`` left over from pre-Phase-H boots
            # so the admin UI's "password is set" indicator stops
            # falsely reporting a PSK the AP no longer accepts.
            updates["admin_password"] = ""
        _save_settings_override(updates)
        # One-shot migration ladder: when the AP boots in admin mode
        # and the operator has never explicitly chosen an egress mode
        # (``source == "default"``) AND the persisted value is still
        # the legacy ``"block"``, upgrade to ``"captive"`` so the
        # captive-gateway flow kicks in by default. We keep source
        # ``"default"`` so future migrations can still fire if needed
        # — but ``_set_wan_egress_mode`` from any operator-facing
        # surface (CLI / REST / UI) stamps source ``"operator"`` and
        # makes the choice sticky against future migrations.
        if cfg.mode == "admin":
            from meeting_scribe.server_support.settings_store import (
                _effective_wan_egress_mode,
                _set_wan_egress_mode,
                _wan_egress_mode_source,
            )

            if _wan_egress_mode_source() == "default" and _effective_wan_egress_mode() == "block":
                _set_wan_egress_mode("captive", source="default")
                logger.info(
                    "captive-gateway: migrated wan_egress_mode block→captive "
                    "(source=default — operator may override anytime)",
                )
        # Captive-gateway: ensure the ipsets exist before the firewall
        # apply so the captive FORWARD rules have something to match.
        # Best-effort — a missing ipset binary leaves us in
        # gateway-like behavior with no per-IP gating.
        try:
            from meeting_scribe.server_support import firewall_allowlist

            await firewall_allowlist.ensure_sets()
        except Exception:
            logger.exception("captive: ensure_sets failed (non-fatal)")
        await _bring_up_ap(cfg)


# ── Setup-wizard wrapper ────────────────────────────────────────────


_SETUP_SSID_BASE = "Dell Meeting"


def setup_ssid() -> str:
    """Return the per-device unique setup-mode SSID.

    ``Dell Meeting <PIN>`` where PIN is the 4-digit decimal code
    derived from the appliance ID (see ``cli._common.appliance_pin``).
    The same number is the guest-auth PIN, so attendees only have
    to look at the SSID to know what to type.

    Example: ``Dell Meeting 4239`` → guest PIN is ``4239``.
    """
    from meeting_scribe.cli._common import appliance_pin

    return f"{_SETUP_SSID_BASE} {appliance_pin()}"


def build_setup_config() -> WifiConfig:
    """Build the open-AP config — the ONLY AP mode in v1.0.

    OWE (Opportunistic Wireless Encryption) — looks like an open
    network to phones, but link is DH-encrypted and the proper
    4-way handshake authorizes the station. The captive sub-app
    on ``10.42.0.1:80`` plus the wizard on ``10.42.0.1:443`` are
    the only services exposed on this AP.

    Per-device unique SSID via ``setup_ssid()``. There is no
    rotation to a "locked" mode in v1.0 — auth happens at the
    application layer via the 4-digit guest PIN or admin
    password (see ``routes/guest_auth.py``).
    """
    return WifiConfig(
        mode="setup",
        ssid=setup_ssid(),
        password="",
        band=DEFAULT_BAND,
        channel=DEFAULT_CHANNEL,
        regdomain=_effective_regdomain(),
        ap_ip=AP_IP,
        security="open",
    )


async def wifi_up_setup() -> None:
    """Activate the OWE setup AP. Lifespan callsite + the
    ``meeting-scribe wifi up --mode setup`` CLI."""
    await wifi_up(build_setup_config())


async def wifi_down() -> None:
    """Tear down the WiFi AP. Thread-safe."""
    async with _wifi_lock:
        _save_settings_override({"wifi_mode": "off"})
        await _teardown_ap()


async def wifi_switch(new_cfg: WifiConfig) -> None:
    """Switch WiFi mode with rollback on failure. Thread-safe.

    Called by the settings API background task when the admin changes
    wifi_mode via the UI. On failure, attempts to restore the previous
    AP config; if that also fails, falls back to wifi_mode=off.
    """
    async with _wifi_lock:
        old_settings = _deep_copy_settings()
        old_config = _build_live_config()
        rollback = RollbackSnapshot(config=old_config, settings=old_settings)

        updates: dict[str, Any] = {"wifi_mode": new_cfg.mode}
        if new_cfg.mode == "admin":
            updates["admin_ssid"] = new_cfg.ssid
            updates["admin_password"] = new_cfg.password
        _save_settings_override(updates)

        await _teardown_ap()
        try:
            await _bring_up_ap(new_cfg)
            return
        except Exception as exc:
            logger.error("wifi_switch bring-up failed: %s", exc)
            _restore_settings(rollback.settings)
            if rollback.config is not None:
                try:
                    await _bring_up_ap(rollback.config)
                    return
                except Exception as rollback_exc:
                    logger.error("wifi_switch rollback also failed: %s", rollback_exc)
            _save_settings_override({"wifi_mode": "off"})


async def wifi_status() -> dict:
    """Read live WiFi state from nmcli/wpa_cli. Returns a status dict.

    Does NOT just read the state file — queries the actual radio and
    supplicant for authoritative state.
    """
    loop = asyncio.get_event_loop()

    # Desired mode from settings
    settings = _load_settings_override()
    desired_mode = settings.get("wifi_mode", "admin")

    # Live AP state. This function is the admin-panel + status-bar
    # backing store; it polls every few seconds while the panel is
    # open. Pass the long TTL (30 s) so we don't burn a sudo nmcli
    # round-trip per poll — the cache is still write-invalidated, so
    # any operator-triggered change shows up immediately on the next
    # tick.
    ap_active = await loop.run_in_executor(
        None,
        lambda: _nmcli_ap_is_active(ttl=_AP_STATE_STATUS_TTL_S),
    )

    result: dict[str, Any] = {
        "desired_mode": desired_mode,
        "live_mode": "off",
        "ssid": None,
        "password": None,
        "security": None,
        "regdomain": _effective_regdomain(),
        "regdomain_live": await loop.run_in_executor(None, _current_regdomain),
        "regdomain_drift": False,
        "captive_active": _captive_portal_active(),
        "client_count": 0,
        "ap_ip": AP_IP,
    }

    target_reg = _effective_regdomain()
    live_reg = result["regdomain_live"]
    result["regdomain_drift"] = live_reg is not None and live_reg != target_reg

    if not ap_active:
        return result

    # AP is active — read credentials (same TTL rationale as above).
    creds = await loop.run_in_executor(
        None,
        lambda: _nmcli_read_live_ap_credentials(ttl=_AP_STATE_STATUS_TTL_S),
    )
    if creds:
        result["ssid"] = creds[0]
        result["password"] = creds[1]

    # Determine live mode from state file
    state = _load_hotspot_state()
    result["live_mode"] = (state or {}).get("mode", "meeting")

    # Security from wpa_supplicant
    sec = await loop.run_in_executor(None, _wpa_supplicant_ap_security)
    if sec:
        km = sec.get("key_mgmt", "?")
        pw = sec.get("pairwise_cipher", "?")
        gw = sec.get("group_cipher", "?")
        result["security"] = {
            "key_mgmt": km,
            "pairwise_cipher": pw,
            "group_cipher": gw,
            "wpa_state": sec.get("wpa_state", "?"),
        }

    # Client count from ip neigh
    try:
        neigh_proc = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["ip", "neigh", "show", "dev", WIFI_IFACE],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            ),
        )
        if neigh_proc.returncode == 0:
            count = sum(
                1
                for line in neigh_proc.stdout.splitlines()
                if line.strip() and "FAILED" not in line
            )
            result["client_count"] = count
    except FileNotFoundError, subprocess.TimeoutExpired, OSError:
        pass

    return result


def wifi_status_sync() -> dict:
    """Synchronous version of wifi_status for non-async callers."""
    # Desired mode from settings
    settings = _load_settings_override()
    desired_mode = settings.get("wifi_mode", "admin")

    ap_active = _nmcli_ap_is_active()

    result: dict[str, Any] = {
        "desired_mode": desired_mode,
        "live_mode": "off",
        "ssid": None,
        "password": None,
        "security": None,
        "regdomain": _effective_regdomain(),
        "regdomain_live": _current_regdomain(),
        "regdomain_drift": False,
        "captive_active": _captive_portal_active(),
        "client_count": 0,
        "ap_ip": AP_IP,
    }

    target_reg = _effective_regdomain()
    live_reg = result["regdomain_live"]
    result["regdomain_drift"] = live_reg is not None and live_reg != target_reg

    if not ap_active:
        return result

    creds = _nmcli_read_live_ap_credentials()
    if creds:
        result["ssid"] = creds[0]
        result["password"] = creds[1]

    state = _load_hotspot_state()
    result["live_mode"] = (state or {}).get("mode", "meeting")

    sec = _wpa_supplicant_ap_security()
    if sec:
        km = sec.get("key_mgmt", "?")
        pw = sec.get("pairwise_cipher", "?")
        gw = sec.get("group_cipher", "?")
        result["security"] = {
            "key_mgmt": km,
            "pairwise_cipher": pw,
            "group_cipher": gw,
            "wpa_state": sec.get("wpa_state", "?"),
        }

    try:
        neigh_proc = subprocess.run(
            ["ip", "neigh", "show", "dev", WIFI_IFACE],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if neigh_proc.returncode == 0:
            count = sum(
                1
                for line in neigh_proc.stdout.splitlines()
                if line.strip() and "FAILED" not in line
            )
            result["client_count"] = count
    except FileNotFoundError, subprocess.TimeoutExpired, OSError:
        pass

    return result
