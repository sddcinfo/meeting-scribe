"""Bluetooth control plane — mirrors the wifi.py pattern.

Sync subprocess wrappers + ``run_in_executor`` + ``asyncio.Lock``. Owns
device pairing, profile selection, node resolution, settings persistence.
The data plane (PipeWire bridge subprocesses, state machine, listener)
lives in :mod:`meeting_scribe.audio.bt_bridge` and consumes these
primitives.

Per Plan §B.2 + §B.6:

* :func:`bt_pair`, :func:`bt_connect`, :func:`bt_disconnect`,
  :func:`bt_forget` — bluetoothctl wrappers.
* :func:`bt_status_sync` — adapter + paired devices + active profile
  (queried by MAC, never by stale node name).
* :func:`bt_list_profiles`, :func:`bt_choose_profile`,
  :func:`bt_set_profile` — pactl ``--format=json list cards`` parsers
  + capability classification (mSBC > CVSD; LDAC > AAC > SBC).
* :func:`bt_resolve_nodes` — pactl JSON filter on
  ``properties.api.bluez5.address``.
* Settings: ``bt_device_mac``, ``bt_device_name``, ``bt_profile_mode``,
  ``bt_input_active``, ``bt_auto_connect``.

This module is **pure subprocess driving** — every interaction is
testable with ``unittest.mock.patch("subprocess.run")``. The data plane
turns these primitives into the live state machine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)


# ── Settings keys ────────────────────────────────────────────────


SETTINGS_BT_DEVICE_MAC = "bt_device_mac"
SETTINGS_BT_DEVICE_NAME = "bt_device_name"
SETTINGS_BT_PROFILE_MODE = "bt_profile_mode"  # "auto"
SETTINGS_BT_INPUT_ACTIVE = "bt_input_active"  # bool — operator's last toggle
SETTINGS_BT_AUTO_CONNECT = "bt_auto_connect"  # "on"|"off"


# ── Errors ───────────────────────────────────────────────────────


class BluetoothError(RuntimeError):
    """Generic BT operation failure (bluetoothctl returned non-zero)."""


class BluetoothCapabilityError(BluetoothError):
    """The device cannot satisfy the requested profile capability.

    Raised by :func:`bt_choose_profile` when the device has no profile
    matching the desired ``a2dp`` / ``hfp`` request. The bridge state
    machine catches this and surfaces "mic unavailable" to the admin
    UI without flipping the bridge into Disconnected.
    """


# ── Profile description ──────────────────────────────────────────


@dataclass(frozen=True)
class ProfileInfo:
    """One available profile on a BlueZ card.

    Profile names vary per BlueZ stack and device — the canonical
    examples are ``a2dp-sink``, ``headset-head-unit-msbc``,
    ``headset-head-unit``, ``hsp-hs``. We store enough info to rank
    candidates without hardcoding the names.
    """

    name: str
    description: str
    has_input: bool
    has_output: bool
    codec_rank: int
    is_available: bool


# ── Module-level state ──────────────────────────────────────────


_lock = asyncio.Lock()


# ── Helpers ──────────────────────────────────────────────────────


_MAC_RE = re.compile(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")


def _validate_mac(mac: str) -> str:
    """Reject malformed MACs. ``bluetoothctl`` accepts garbage and
    silently no-ops; better to fail loudly here."""
    if not _MAC_RE.match(mac):
        raise BluetoothError(f"invalid MAC: {mac!r}")
    return mac.upper()


def _mac_to_underscored(mac: str) -> str:
    """``AA:BB:CC:DD:EE:FF`` → ``AA_BB_CC_DD_EE_FF`` (pactl card naming)."""
    return _validate_mac(mac).replace(":", "_")


async def _run(argv: list[str], *, timeout: float = 10.0) -> subprocess.CompletedProcess:
    """Run ``argv`` (list-form) in an executor; never blocks the loop."""
    loop = asyncio.get_running_loop()

    def _blocking() -> subprocess.CompletedProcess:
        return subprocess.run(  # noqa: S603 — list-form, no shell
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )

    return await loop.run_in_executor(None, _blocking)


# ── pactl JSON parsing ──────────────────────────────────────────


_CODEC_PRIORITY = (
    "ldac",
    "aac",
    "aptx_hd",
    "aptx",
    "msbc",
    "sbc_xq",
    "sbc",
    "cvsd",
)


def _codec_rank(profile_name: str) -> int:
    """Return a higher rank for higher-quality codecs.

    Ranking is best-effort; if no known codec is in the name, returns 0
    and the caller falls back to ``is_available`` ordering.
    """
    name = profile_name.lower()
    for idx, codec in enumerate(_CODEC_PRIORITY):
        if codec in name:
            return len(_CODEC_PRIORITY) - idx
    return 0


def parse_card_profiles(pactl_json: str, *, mac: str) -> list[ProfileInfo]:
    """Parse ``pactl --format=json list cards`` output and return the
    declared profiles for the card whose ``api.bluez5.address`` matches
    ``mac``.

    Plain function (no I/O) so the test suite can drive it via the
    JSON fixtures from real Ray-Ban Meta + Mijia hardware.
    """
    target = _validate_mac(mac)
    try:
        data = json.loads(pactl_json)
    except json.JSONDecodeError as exc:
        raise BluetoothError(f"pactl JSON parse failed: {exc}") from exc
    if not isinstance(data, list):
        return []
    for card in data:
        if not isinstance(card, dict):
            continue
        props = card.get("properties") or {}
        addr = (props.get("api.bluez5.address") or "").upper()
        if addr != target:
            continue
        profiles_obj = card.get("profiles") or {}
        if isinstance(profiles_obj, list):
            iter_profiles = (
                (p.get("name", ""), p.get("description", ""), bool(p.get("available", True)))
                for p in profiles_obj
                if isinstance(p, dict)
            )
        elif isinstance(profiles_obj, dict):
            iter_profiles = (
                (
                    name,
                    info.get("description", "") if isinstance(info, dict) else "",
                    bool(info.get("available", True)) if isinstance(info, dict) else True,
                )
                for name, info in profiles_obj.items()
            )
        else:
            iter_profiles = iter(())
        out: list[ProfileInfo] = []
        for name, desc, avail in iter_profiles:
            if not name or name == "off":
                continue
            lower = name.lower()
            has_output = "sink" in lower or "headset" in lower or "hsp" in lower or "a2dp" in lower
            has_input = "headset" in lower or "hsp" in lower or "source" in lower
            out.append(
                ProfileInfo(
                    name=name,
                    description=desc or "",
                    has_input=has_input,
                    has_output=has_output,
                    codec_rank=_codec_rank(name),
                    is_available=avail,
                )
            )
        return out
    return []


def choose_profile(
    profiles: list[ProfileInfo],
    *,
    want: Literal["a2dp", "hfp"],
) -> str:
    """Rank ``profiles`` and return the best match for ``want``.

    a2dp → first available sink-only profile, ranked by codec.
    hfp  → first available bidirectional profile, ranked by codec
           (mSBC > CVSD; falls back to ``hsp-*`` if no headset-* profile).

    Raises :class:`BluetoothCapabilityError` when nothing matches.
    """
    if want == "a2dp":
        candidates = [
            p
            for p in profiles
            if p.has_output
            and not p.has_input
            and p.is_available
            and ("a2dp" in p.name.lower() or "sink" in p.name.lower())
        ]
    elif want == "hfp":
        candidates = [
            p
            for p in profiles
            if p.has_input and p.has_output and p.is_available
        ]
    else:
        raise ValueError(f"unknown want: {want!r}")

    if not candidates:
        raise BluetoothCapabilityError(f"no available {want} profile on this device")
    candidates.sort(key=lambda p: (-p.codec_rank, p.name))
    return candidates[0].name


def parse_node_for_mac(
    sources_json: str,
    sinks_json: str,
    *,
    mac: str,
) -> tuple[str | None, str | None]:
    """Find the active source + sink node names for the given MAC.

    Returns ``(source_name, sink_name)`` — either may be ``None`` when
    the device has no active node of that kind (e.g. A2DP-only speakers
    have no source; mid-profile-switch the prior nodes are gone and
    new ones haven't appeared yet).
    """
    target = _validate_mac(mac)

    def _scan(blob: str) -> str | None:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, list):
            return None
        for entry in data:
            if not isinstance(entry, dict):
                continue
            props = entry.get("properties") or {}
            if (props.get("device.bus") or "").lower() != "bluetooth":
                continue
            addr = (props.get("api.bluez5.address") or "").upper()
            if addr != target:
                continue
            name = entry.get("name")
            if isinstance(name, str):
                return name
        return None

    return _scan(sources_json), _scan(sinks_json)


# ── Async wrappers — actual command invocations ──────────────────


async def bt_status_sync() -> dict[str, Any]:
    """Read-only adapter + devices + active-profile snapshot.

    Always queries by MAC, never by node name (Plan §B.2). Returns a
    dict suitable for ``/api/admin/bt/status``.
    """
    powered = False
    devices: list[dict[str, Any]] = []
    show = await _run(["bluetoothctl", "show"])
    if show.returncode == 0:
        for line in show.stdout.splitlines():
            line = line.strip()
            if line.startswith("Powered:"):
                powered = line.endswith("yes")
    paired = await _run(["bluetoothctl", "devices", "Paired"])
    if paired.returncode == 0:
        for line in paired.stdout.splitlines():
            parts = line.strip().split(" ", 2)
            if len(parts) >= 3 and parts[0] == "Device":
                devices.append({"mac": parts[1].upper(), "name": parts[2]})
    return {"powered": powered, "devices": devices}


async def bt_list_profiles(mac: str) -> list[ProfileInfo]:
    """Return the available profiles for the card whose BD_ADDR matches ``mac``."""
    proc = await _run(["pactl", "--format=json", "list", "cards"])
    if proc.returncode != 0:
        raise BluetoothError(f"pactl list cards failed: {proc.stderr.strip()[:200]}")
    return parse_card_profiles(proc.stdout, mac=mac)


async def bt_choose_profile(mac: str, *, want: Literal["a2dp", "hfp"]) -> str:
    return choose_profile(await bt_list_profiles(mac), want=want)


async def bt_set_profile(mac: str, profile_name: str, *, timeout: float = 3.0) -> None:
    """Issue `pactl set-card-profile` and verify via JSON.

    Polls the card's ``active_profile`` after the issue; if it doesn't
    flip within ``timeout`` we raise :class:`BluetoothError` (the bridge
    falls into Disconnected and surfaces the failure).
    """
    underscored = _mac_to_underscored(mac)
    card = f"bluez_card.{underscored}"
    issue = await _run(["pactl", "set-card-profile", card, profile_name])
    if issue.returncode != 0:
        raise BluetoothError(f"set-card-profile failed: {issue.stderr.strip()[:200]}")

    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        proc = await _run(["pactl", "--format=json", "list", "cards"])
        if proc.returncode == 0:
            try:
                data = json.loads(proc.stdout)
            except json.JSONDecodeError:
                data = []
            for entry in data if isinstance(data, list) else []:
                if not isinstance(entry, dict):
                    continue
                if entry.get("name") == card:
                    if entry.get("active_profile") == profile_name:
                        return
        await asyncio.sleep(0.1)
    raise BluetoothError(f"timed out waiting for active_profile={profile_name}")


async def bt_resolve_nodes(mac: str) -> tuple[str | None, str | None]:
    sources = await _run(["pactl", "--format=json", "list", "sources"])
    sinks = await _run(["pactl", "--format=json", "list", "sinks"])
    return parse_node_for_mac(sources.stdout, sinks.stdout, mac=mac)


async def bt_pair(mac: str | None = None, *, timeout: float = 30.0) -> dict[str, Any]:
    """Pair → trust → connect.

    Operator picks the MAC from a scan list. The CLI fronts a
    discoverable-list display + index/MAC selection (Plan §B.7); this
    function takes the resolved MAC and runs the bluetoothctl steps
    list-form so no shell injection risk on a malformed scan name.
    """
    if mac is None:
        raise BluetoothError("bt_pair requires an explicit MAC")
    target = _validate_mac(mac)
    async with _lock:
        for cmd in (
            ["bluetoothctl", "power", "on"],
            ["bluetoothctl", "agent", "on"],
            ["bluetoothctl", "default-agent"],
        ):
            await _run(cmd)
        for cmd in (
            ["bluetoothctl", "pair", target],
            ["bluetoothctl", "trust", target],
            ["bluetoothctl", "connect", target],
        ):
            proc = await _run(cmd, timeout=timeout)
            if proc.returncode != 0:
                raise BluetoothError(f"{' '.join(cmd)} failed: {proc.stderr.strip()[:200]}")
    return {"mac": target, "paired": True}


async def bt_connect(mac: str) -> None:
    target = _validate_mac(mac)
    proc = await _run(["bluetoothctl", "connect", target])
    if proc.returncode != 0:
        raise BluetoothError(f"connect failed: {proc.stderr.strip()[:200]}")


async def bt_disconnect(mac: str | None = None, *, user_initiated: bool = True) -> None:
    """Disconnect the named device (or whatever's connected if None).

    ``user_initiated`` is consumed by the bridge state machine to
    suppress the auto-retry; it doesn't change the bluetoothctl call.
    """
    if mac is None:
        proc = await _run(["bluetoothctl", "disconnect"])
    else:
        proc = await _run(["bluetoothctl", "disconnect", _validate_mac(mac)])
    if proc.returncode != 0:
        raise BluetoothError(f"disconnect failed: {proc.stderr.strip()[:200]}")


async def bt_forget(mac: str) -> None:
    target = _validate_mac(mac)
    proc = await _run(["bluetoothctl", "remove", target])
    if proc.returncode != 0:
        raise BluetoothError(f"remove failed: {proc.stderr.strip()[:200]}")
