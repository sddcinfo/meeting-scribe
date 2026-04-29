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
SSID_PREFIX = "Dell Demo"
DEFAULT_ADMIN_SSID = "Dell Admin"

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

# nmcli timeouts
_NMCLI_CON_UP_TIMEOUT = 60
_NMCLI_CON_DOWN_TIMEOUT = 15
_NMCLI_CON_MODIFY_TIMEOUT = 5
_NMCLI_CON_SHOW_TIMEOUT = 5

# ── Dataclasses ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WifiConfig:
    """Immutable WiFi AP configuration."""

    mode: str  # "off", "meeting", or "admin"
    ssid: str
    password: str
    band: str = DEFAULT_BAND
    channel: int = DEFAULT_CHANNEL
    regdomain: str = _DEFAULT_REGDOMAIN
    ap_ip: str = AP_IP


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
    """Run ``sudo nmcli <args>`` synchronously. Errors are returned, not raised."""
    return subprocess.run(
        ["sudo", "nmcli", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


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


def _nmcli_read_live_ap_credentials() -> tuple[str, str] | None:
    """Return ``(ssid, psk)`` for the active AP profile, or ``None``.

    Reads via ``nmcli --show-secrets`` so the psk is returned in plaintext.
    This is the authoritative source for what the radio is actually
    broadcasting — the state file is a derived cache of this.
    """
    proc = _run_nmcli_sync(
        [
            "--show-secrets",
            "-t",
            "-f",
            "802-11-wireless.ssid,802-11-wireless-security.psk",
            "con",
            "show",
            AP_CON_NAME,
        ],
        timeout=_NMCLI_CON_SHOW_TIMEOUT,
    )
    if proc.returncode != 0:
        return None
    fields = _parse_nmcli_fields(proc.stdout)
    ssid = fields.get("802-11-wireless.ssid", "")
    psk = fields.get("802-11-wireless-security.psk", "")
    if not ssid:
        return None
    return ssid, psk


def _nmcli_ap_is_active() -> bool:
    """Return True if the AP profile is currently active on the radio."""
    proc = _run_nmcli_sync(
        ["-t", "-f", "NAME,DEVICE", "con", "show", "--active"],
        timeout=_NMCLI_CON_SHOW_TIMEOUT,
    )
    if proc.returncode != 0:
        return False
    prefix = f"{AP_CON_NAME}:"
    return any(line.startswith(prefix) for line in proc.stdout.splitlines())


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
    """Configure dnsmasq wildcard DNS + DHCP option 114 for captive portal.

    Two layers:
      1. Wildcard DNS (``address=/#/<ap_ip>``): every hostname resolves to AP.
      2. DHCP option 114 (RFC 8910): advertise captive portal URL at DHCP ACK.

    Port 80 is owned by meeting-scribe's in-process guest listener.
    """
    subprocess.run(
        ["sudo", "mkdir", "-p", str(DNSMASQ_CONF_DIR)],
        capture_output=True,
        timeout=5,
        check=False,
    )

    portal_url = f"http://{ap_ip}/"
    conf_content = (
        f"# Captive portal — resolve all domains to AP IP\n"
        f"address=/#/{ap_ip}\n"
        f"# RFC 8910 captive-portal URL (DHCP option 114)\n"
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


def _apply_meeting_firewall(admin_port: int = MEETING_PORT) -> None:
    """Apply meeting-mode firewall: allow port 80, reject ``admin_port``/443, default-deny.

    Guests can reach the HTTP portal on port 80. Admin HTTPS (``admin_port``,
    default 8080) and plain HTTPS (443) are rejected with TCP RST for
    instant failure. DNS, DHCP, and ICMP are allowed. Everything else is
    rejected with ICMP port-unreachable. All forwarding is blocked (no
    internet). NAT redirects all port 80 traffic to the local listener.
    """
    _remove_firewall()

    rules = [
        f"-I INPUT 1 -s {HOTSPOT_SUBNET_CIDR} -p tcp --dport 80 -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j ACCEPT",
        f"-I INPUT 2 -s {HOTSPOT_SUBNET_CIDR} -p tcp --dport 443 -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j REJECT --reject-with tcp-reset",
        f"-I INPUT 3 -s {HOTSPOT_SUBNET_CIDR} -p tcp --dport {admin_port} -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j REJECT --reject-with tcp-reset",
        f"-I INPUT 4 -s {HOTSPOT_SUBNET_CIDR} -p udp --dport 53 -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j ACCEPT",
        f"-I INPUT 5 -s {HOTSPOT_SUBNET_CIDR} -p udp --dport 67 -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j ACCEPT",
        f"-I INPUT 6 -s {HOTSPOT_SUBNET_CIDR} -p icmp -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j ACCEPT",
        f"-I INPUT 7 -s {HOTSPOT_SUBNET_CIDR} -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j REJECT --reject-with icmp-port-unreachable",
        f"-I FORWARD 1 -s {HOTSPOT_SUBNET_CIDR} -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j REJECT --reject-with icmp-port-unreachable",
        f"-I FORWARD 2 -d {HOTSPOT_SUBNET_CIDR} -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j REJECT --reject-with icmp-port-unreachable",
        f"-t nat -I PREROUTING 1 -s {HOTSPOT_SUBNET_CIDR} -p tcp --dport 80 -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j REDIRECT --to-port 80",
    ]
    for rule in rules:
        try:
            subprocess.run(
                ["sudo", "iptables"] + rule.split(),
                capture_output=True,
                timeout=5,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("iptables rule failed: %s — %s", rule, exc)
    logger.info(
        "Meeting firewall applied (allow: 80/DNS/DHCP; reject: 443/%d)",
        admin_port,
    )


def _apply_admin_firewall() -> None:
    """Apply admin-mode firewall: allow ports 80 AND 8080, reject 443, default-deny.

    Admin mode allows WiFi clients to reach the admin UI on port 8080
    in addition to the HTTP portal on port 80. HTTPS (443) is still
    rejected. All forwarding is blocked.
    """
    _remove_firewall()

    rules = [
        f"-I INPUT 1 -s {HOTSPOT_SUBNET_CIDR} -p tcp --dport 80 -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j ACCEPT",
        f"-I INPUT 2 -s {HOTSPOT_SUBNET_CIDR} -p tcp --dport {MEETING_PORT} -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j ACCEPT",
        f"-I INPUT 3 -s {HOTSPOT_SUBNET_CIDR} -p tcp --dport 443 -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j REJECT --reject-with tcp-reset",
        f"-I INPUT 4 -s {HOTSPOT_SUBNET_CIDR} -p udp --dport 53 -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j ACCEPT",
        f"-I INPUT 5 -s {HOTSPOT_SUBNET_CIDR} -p udp --dport 67 -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j ACCEPT",
        f"-I INPUT 6 -s {HOTSPOT_SUBNET_CIDR} -p icmp -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j ACCEPT",
        f"-I INPUT 7 -s {HOTSPOT_SUBNET_CIDR} -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j REJECT --reject-with icmp-port-unreachable",
        f"-I FORWARD 1 -s {HOTSPOT_SUBNET_CIDR} -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j REJECT --reject-with icmp-port-unreachable",
        f"-I FORWARD 2 -d {HOTSPOT_SUBNET_CIDR} -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j REJECT --reject-with icmp-port-unreachable",
        f"-t nat -I PREROUTING 1 -s {HOTSPOT_SUBNET_CIDR} -p tcp --dport 80 -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j REDIRECT --to-port 80",
    ]
    for rule in rules:
        try:
            subprocess.run(
                ["sudo", "iptables"] + rule.split(),
                capture_output=True,
                timeout=5,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("iptables rule failed: %s — %s", rule, exc)
    logger.info(
        "Admin firewall applied (allow: 80/%d/DNS/DHCP; reject: 443)",
        MEETING_PORT,
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

    # Step 1: regdomain
    await loop.run_in_executor(None, _ensure_regdomain_persistent)
    if not await loop.run_in_executor(None, _ensure_regdomain):
        raise RuntimeError(
            f"Failed to set regulatory domain to {cfg.regdomain}. "
            f"iw reg get reports: {_current_regdomain()!r}. "
            "5 GHz AP transmit power would be capped."
        )
    logger.info("regdomain verified: %s", _current_regdomain())

    # Step 2: captive portal
    if cfg.mode == "meeting":
        await loop.run_in_executor(None, _setup_captive_portal, cfg.ap_ip)
        logger.info("captive portal configured (DNS wildcard + DHCP 114)")
    elif cfg.mode == "admin":
        await loop.run_in_executor(None, _teardown_captive_portal)
        logger.info("captive portal cleared (admin mode)")

    # Step 3: create or modify NM profile
    exists = await loop.run_in_executor(None, _nmcli_connection_exists)
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
                ],
                timeout=_NMCLI_CON_MODIFY_TIMEOUT,
            ),
        )
        if proc.returncode != 0:
            logger.warning(
                "nmcli con modify failed (rc=%d): %s",
                proc.returncode,
                proc.stderr.strip()[:200],
            )
        else:
            logger.info("NM profile updated: ssid=%s", cfg.ssid)
    else:
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
                ],
                timeout=_NMCLI_CON_MODIFY_TIMEOUT,
            ),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"nmcli con add failed (rc={proc.returncode}): {proc.stderr.strip()[:200]}"
            )
        logger.info("NM profile created: ssid=%s", cfg.ssid)

    # Step 4: bring up the AP
    if await loop.run_in_executor(None, _nmcli_ap_is_active):
        await loop.run_in_executor(
            None,
            lambda: _run_nmcli_sync(["con", "down", AP_CON_NAME], timeout=_NMCLI_CON_DOWN_TIMEOUT),
        )
        await asyncio.sleep(1)

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

    # Step 5: apply firewall
    if cfg.mode == "meeting":
        await loop.run_in_executor(None, _apply_meeting_firewall)
    elif cfg.mode == "admin":
        await loop.run_in_executor(None, _apply_admin_firewall)

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
        if not password or len(password) < 8:
            password = settings.get("admin_password", "")
            if not password or len(password) < 8:
                password = secrets.token_hex(8).upper()
                # Persist so the admin password survives restarts
                _save_settings_override({"admin_ssid": ssid, "admin_password": password})
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
    )


async def wifi_up(cfg: WifiConfig) -> None:
    """Bring up the WiFi AP in the requested mode. Thread-safe."""
    async with _wifi_lock:
        updates: dict[str, Any] = {"wifi_mode": cfg.mode}
        if cfg.mode == "admin":
            updates["admin_ssid"] = cfg.ssid
            updates["admin_password"] = cfg.password
        _save_settings_override(updates)
        await _bring_up_ap(cfg)


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
    desired_mode = settings.get("wifi_mode", "off")

    # Live AP state
    ap_active = await loop.run_in_executor(None, _nmcli_ap_is_active)

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

    # AP is active — read credentials
    creds = await loop.run_in_executor(None, _nmcli_read_live_ap_credentials)
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
    desired_mode = settings.get("wifi_mode", "off")

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
