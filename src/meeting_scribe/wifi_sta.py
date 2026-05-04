"""Concurrent STA + AP — virtual-interface lifecycle, NM profile staging,
keyfile validation, channel preflight, layered egress check.

Plan 2 v36 mainline: the GB10's MT7925 radio runs AP-only via wifi.py
→ NetworkManager. This module owns the STA virtual interface
(``wlan_sta``) so the appliance can connect to an upstream WiFi while
continuing to host the meeting hotspot.

The functions here are **pure code** wrapping nmcli/iw shellouts via
list-form argv — no shell strings, no string interpolation. The
firewall side of the story (MS_FWD egress isolation when
``sta_iface_present=True``) lives in :mod:`meeting_scribe.firewall`.

End-to-end on-hardware verification needs the MT7925 radio + a second
SSID; the unit tests here drive the parsers + state predicates with
synthetic ``iw``/``nmcli`` text.
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

logger = logging.getLogger(__name__)


STA_IFACE = "wlan_sta"
PHY_DEFAULT = "phy0"
AP_IFACE_DEFAULT = "wlP9s9"


# ── Markers (Plan 2 §B.4 / R3) ──────────────────────────────────


STA_DEGRADED_MARKER = Path("/var/lib/meeting-scribe/STA-DEGRADED")
REPAIR_NEEDED_MARKER = Path("/var/lib/meeting-scribe/REPAIR-NEEDED")


def write_sta_degraded(reason: str) -> None:
    """Plan 2 §A: write the STA-DEGRADED marker so subsequent boots
    know to skip auto-recovery and surface the failure to the operator.
    """
    STA_DEGRADED_MARKER.parent.mkdir(parents=True, exist_ok=True)
    STA_DEGRADED_MARKER.write_text(
        f"sta degraded at {time.strftime('%Y-%m-%dT%H:%M:%S')}: {reason}\n",
        encoding="utf-8",
    )


def clear_sta_degraded() -> None:
    """Operator clears the marker to re-enable STA recovery on next boot."""
    STA_DEGRADED_MARKER.unlink(missing_ok=True)


def write_repair_needed(reason: str) -> None:
    REPAIR_NEEDED_MARKER.parent.mkdir(parents=True, exist_ok=True)
    REPAIR_NEEDED_MARKER.write_text(
        f"repair needed at {time.strftime('%Y-%m-%dT%H:%M:%S')}: {reason}\n",
        encoding="utf-8",
    )


# ── Subprocess helper ───────────────────────────────────────────


async def _run(argv: list[str], *, timeout: float = 10.0) -> subprocess.CompletedProcess:
    loop = asyncio.get_running_loop()

    def _blocking() -> subprocess.CompletedProcess:
        return subprocess.run(  # noqa: S603 — list-form argv only
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )

    return await loop.run_in_executor(None, _blocking)


# ── iw scan parsers ─────────────────────────────────────────────


@dataclass(frozen=True)
class ScanEntry:
    """One row from ``iw dev wlan_sta scan``."""

    bssid: str
    ssid: str
    channel: int
    signal_dbm: float
    rsn_present: bool


_BSSID_RE = re.compile(r"^BSS\s+([0-9a-f:]{17})", re.IGNORECASE | re.MULTILINE)
_SSID_RE = re.compile(r"^\s+SSID:\s+(.+)$", re.MULTILINE)
_CHAN_RE = re.compile(r"^\s+(?:freq|primary channel):\s+(\d+)", re.MULTILINE)
_SIGNAL_RE = re.compile(r"^\s+signal:\s+(-?\d+(?:\.\d+)?)\s+dBm", re.MULTILINE)


def parse_iw_scan(text: str) -> list[ScanEntry]:
    """Parse the indent-blocked output of ``iw dev wlan_sta scan``.

    Returns one :class:`ScanEntry` per BSS. Entries lacking an SSID
    or channel are skipped (hidden networks aren't scannable here).
    The parser is line-oriented and tolerant of variations in iw
    output across kernel versions.
    """
    out: list[ScanEntry] = []
    blocks = re.split(r"^BSS\s+", text, flags=re.IGNORECASE | re.MULTILINE)[1:]
    for block in blocks:
        head = block.split("\n", 1)[0]
        bssid_match = re.match(r"([0-9a-f:]{17})", head, re.IGNORECASE)
        if not bssid_match:
            continue
        bssid = bssid_match.group(1).lower()
        ssid_m = re.search(r"^\s*SSID:\s+(.+)$", block, re.MULTILINE)
        if not ssid_m:
            continue
        ssid = ssid_m.group(1).strip()
        chan_m = re.search(
            r"^\s*(?:DS Parameter set:\s+channel\s+|primary channel:\s+)(\d+)",
            block,
            re.MULTILINE,
        )
        if not chan_m:
            # Best-effort: parse 'freq:' and convert.
            freq_m = re.search(r"^\s*freq:\s+(\d+)", block, re.MULTILINE)
            if freq_m:
                chan = _freq_to_channel(int(freq_m.group(1)))
                if chan is None:
                    continue
            else:
                continue
        else:
            chan = int(chan_m.group(1))
        signal_m = re.search(r"^\s*signal:\s+(-?\d+(?:\.\d+)?)", block, re.MULTILINE)
        signal = float(signal_m.group(1)) if signal_m else -100.0
        rsn = "RSN:" in block
        out.append(
            ScanEntry(
                bssid=bssid,
                ssid=ssid,
                channel=chan,
                signal_dbm=signal,
                rsn_present=rsn,
            )
        )
    return out


def _freq_to_channel(freq_mhz: int) -> int | None:
    if 2412 <= freq_mhz <= 2484:
        if freq_mhz == 2484:
            return 14
        return 1 + (freq_mhz - 2412) // 5
    if 5180 <= freq_mhz <= 5825:
        return 36 + (freq_mhz - 5180) // 5
    if 5955 <= freq_mhz <= 7115:
        return (freq_mhz - 5955) // 5 + 1
    return None


# ── Channel preflight ───────────────────────────────────────────


def channel_preflight(
    *,
    target_ssid: str,
    target_bssid: str | None,
    ap_channel: int,
    scan: list[ScanEntry],
) -> tuple[bool, str | None]:
    """Plan 2 §34: refuse to associate with an upstream BSS that would
    require the radio to retune the AP — single-radio dual-mode is
    pinned to the AP's channel.

    Returns ``(ok, reason)``. Refusal reasons:

    * ``"no_match"`` — target SSID/BSSID not visible in the scan.
    * ``"cross_channel"`` — target visible but on a different channel.
    """
    candidates = [s for s in scan if s.ssid == target_ssid]
    if target_bssid:
        candidates = [s for s in candidates if s.bssid == target_bssid.lower()]
    if not candidates:
        return False, "no_match"
    if not any(s.channel == ap_channel for s in candidates):
        return False, "cross_channel"
    return True, None


# ── Layered egress check ────────────────────────────────────────


@dataclass(frozen=True)
class EgressProbe:
    """Result of the layered egress check after a successful associate.

    Plan 2 §STA bringup contract: a successful associate doesn't mean
    we have upstream egress. We probe in order: link → DHCP → DNS →
    HTTP. The first failed layer wins (no point reporting that DNS
    failed when DHCP never returned a lease).
    """

    link_up: bool
    dhcp_lease: bool
    dns_resolves: bool
    http_reachable: bool

    @property
    def ok(self) -> bool:
        return self.link_up and self.dhcp_lease and self.dns_resolves and self.http_reachable

    @property
    def first_failure(self) -> str | None:
        for label, value in (
            ("link", self.link_up),
            ("dhcp", self.dhcp_lease),
            ("dns", self.dns_resolves),
            ("http", self.http_reachable),
        ):
            if not value:
                return label
        return None


async def layered_egress_check(
    *,
    link_probe: Callable[[], Awaitable[bool]],
    dhcp_probe: Callable[[], Awaitable[bool]],
    dns_probe: Callable[[], Awaitable[bool]],
    http_probe: Callable[[], Awaitable[bool]],
) -> EgressProbe:
    """Run the four probes in order, short-circuiting on the first fail.

    Probes are injected so unit tests can drive them deterministically;
    the production wiring binds each to the appropriate
    ``ip``/``udhcpc``/``dig``/``curl`` shellout.
    """
    if not await link_probe():
        return EgressProbe(False, False, False, False)
    if not await dhcp_probe():
        return EgressProbe(True, False, False, False)
    if not await dns_probe():
        return EgressProbe(True, True, False, False)
    if not await http_probe():
        return EgressProbe(True, True, True, False)
    return EgressProbe(True, True, True, True)


# ── Owned-keyfile scope ────────────────────────────────────────


_OWNED_KEYFILE_PREFIX = "meeting-scribe-sta-"


def is_owned_keyfile(path: Path) -> bool:
    """Plan 2 §15: scan helper pre-parse owned-scope check is by
    filename/path only. Never read keyfile contents to decide ownership.
    """
    return path.name.startswith(_OWNED_KEYFILE_PREFIX)


def scan_owned_keyfiles(directory: Path) -> list[Path]:
    """Walk ``directory`` (typically /etc/NetworkManager/system-connections/)
    and return every file whose name starts with ``meeting-scribe-sta-``.
    Pure path-based filter; never reads file content.
    """
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and is_owned_keyfile(p))


def find_duplicate_uuids(keyfiles: Iterable[Path]) -> list[str]:
    """Plan 2 §16: duplicate DellDemo-STA UUIDs are a HARD failure on swap.

    Returns the list of UUIDs that appear in more than one keyfile.
    The keyfile parser here is the trivial INI-line scan (look for
    ``uuid=…``); this is enough for duplicate detection without
    pulling in configparser.
    """
    uuids: dict[str, int] = {}
    for path in keyfiles:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("uuid="):
                uuid = line.split("=", 1)[1].strip()
                uuids[uuid] = uuids.get(uuid, 0) + 1
                break
    return sorted([uuid for uuid, count in uuids.items() if count > 1])


# ── Posture predicate (used by firewall snapshot) ──────────────


def sta_iface_present() -> bool:
    """Plan 2 §A: ``sta_iface_present`` toggles MS_FWD/MS_POST egress
    isolation. Implemented as a /sys check so the firewall layer can
    consult the current state without coupling to NM's view.
    """
    return Path(f"/sys/class/net/{STA_IFACE}").exists()


# ── Async wrappers ─────────────────────────────────────────────


async def iw_scan(*, iface: str = STA_IFACE) -> list[ScanEntry]:
    """Run ``iw dev <iface> scan`` and return parsed entries.

    Empty list on error; the caller decides whether to retry.
    """
    proc = await _run(["iw", "dev", iface, "scan"])
    if proc.returncode != 0:
        return []
    return parse_iw_scan(proc.stdout)


async def sta_iface_ensure(*, phy: str = PHY_DEFAULT) -> bool:
    """Best-effort: create the ``wlan_sta`` virtual interface on
    ``phy`` if it doesn't already exist.

    Returns True iff the interface is present after the call.
    """
    if sta_iface_present():
        return True
    proc = await _run(
        ["iw", "phy", phy, "interface", "add", STA_IFACE, "type", "managed"]
    )
    if proc.returncode != 0:
        return False
    return sta_iface_present()


# ── Boot-time recovery decisions ──────────────────────────────


def boot_should_skip_sta() -> tuple[bool, str | None]:
    """Plan 2 §33: boot recovery fails closed on competing profiles
    or on the STA-DEGRADED marker.

    Returns ``(should_skip, reason)``. The lifespan task consults
    this before scheduling sta_connect.
    """
    if STA_DEGRADED_MARKER.exists():
        return True, "STA-DEGRADED marker present"
    if REPAIR_NEEDED_MARKER.exists():
        return True, "REPAIR-NEEDED marker present"
    return False, None
