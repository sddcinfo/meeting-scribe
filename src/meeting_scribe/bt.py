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
  :func:`bt_set_profile` — pw-dump JSON parsers + ``wpctl set-profile``
  + capability classification (mSBC > CVSD; LDAC > AAC > SBC).
* :func:`bt_resolve_nodes` — pw-dump JSON filter on
  ``info.props["api.bluez5.address"]``.
* Settings: ``bt_device_mac``, ``bt_device_name``, ``bt_profile_mode``,
  ``bt_input_active``, ``bt_auto_connect``.

This module talks to PipeWire through the native ``pw-dump`` (read) and
``wpctl set-profile`` (write) primitives — the ``pactl`` binary from
``pulseaudio-utils`` is not required and is not installed on the GB10
production image. The pure parser functions still consume a
pactl-shaped intermediate so the existing fixtures keep driving them;
the IO layer synthesizes that shape from pw-dump output.
"""

from __future__ import annotations

import asyncio
import contextlib
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
_last_audio_reconcile_by_mac: dict[str, float] = {}
_AUDIO_RECONCILE_COOLDOWN_S = 10.0


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
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )

    return await loop.run_in_executor(None, _blocking)


def _parse_bluetoothctl_devices(text: str) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = line.strip().split(" ", 2)
        if len(parts) >= 3 and parts[0] == "Device":
            devices.append({"mac": parts[1].upper(), "name": parts[2]})
    return devices


def _parse_bluetoothctl_info(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key in {"paired", "bonded", "trusted", "blocked", "connected"}:
            out[key] = value.lower() == "yes"
        elif key in {"name", "alias", "icon"}:
            out[key] = value
        elif key == "uuid":
            out.setdefault("uuids", []).append(value)
    return out


def _is_audio_device(info: dict[str, Any]) -> bool:
    text = " ".join(
        str(v)
        for key, value in info.items()
        for v in (value if key == "uuids" and isinstance(value, list) else [value])
    ).lower()
    return any(token in text for token in ("audio", "headset", "handsfree", "a/v remote"))


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
    """Parse a pactl-shaped cards JSON list and return the declared
    profiles for the card whose ``api.bluez5.address`` matches ``mac``.

    Plain function (no I/O) so the test suite can drive it via the
    JSON fixtures from real Ray-Ban Meta + Mijia hardware. Production
    callers feed it the output of :func:`_pw_dump_to_pactl_cards` —
    the same shape pactl ``--format=json list cards`` would produce.
    """
    target = _validate_mac(mac)
    try:
        data = json.loads(pactl_json)
    except json.JSONDecodeError as exc:
        raise BluetoothError(f"cards JSON parse failed: {exc}") from exc
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
        candidates = [p for p in profiles if p.has_input and p.has_output and p.is_available]
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


async def bt_scan(timeout: float = 10.0) -> list[dict[str, Any]]:
    """Scan nearby Bluetooth devices and return non-paired matches.

    Runs ``bluetoothctl --timeout N scan on`` (returns when the timeout
    elapses) and then enumerates ``bluetoothctl devices`` minus
    ``bluetoothctl devices Paired`` so the caller sees only candidates
    available for first-time pairing — not devices already in the
    paired list (those are surfaced via ``bt_status_sync``).

    Each entry: ``{"mac": "AA:BB:CC:DD:EE:FF", "name": "<friendly>"}``.
    Devices whose ``name`` is just a MAC-with-dashes (e.g.
    ``66-0B-25-96-94-C0``) — the bluetoothctl fallback for adapters
    that haven't broadcast a friendly name yet — are filtered out so
    the UI doesn't show a sea of MAC strings the operator can't
    identify. Re-running the scan after a few seconds usually surfaces
    the friendly name once the device responds.
    """
    timeout = max(2.0, min(float(timeout), 30.0))
    # bluetoothctl uses its --timeout for the scan duration; we add a
    # generous _run timeout to cover startup + cleanup overhead.
    await _run(
        ["bluetoothctl", "--timeout", str(int(timeout)), "scan", "on"],
        timeout=timeout + 5.0,
    )

    known_macs: set[str] = set()
    known_proc = await _run(["bluetoothctl", "devices"])
    if known_proc.returncode == 0:
        for device in _parse_bluetoothctl_devices(known_proc.stdout):
            known_macs.add(device["mac"])

    discovered: list[dict[str, Any]] = []
    all_proc = await _run(["bluetoothctl", "devices"])
    if all_proc.returncode == 0:
        for line in all_proc.stdout.splitlines():
            parts = line.strip().split(" ", 2)
            if len(parts) < 3 or parts[0] != "Device":
                continue
            mac = parts[1].upper()
            name = parts[2].strip()
            if mac in known_macs:
                continue
            # Filter: bluetoothctl falls back to ``MAC-WITH-DASHES`` as
            # the device "name" when no friendly name has been seen.
            # That gives the operator nothing identifiable; skip.
            if name.replace("-", ":").upper() == mac:
                continue
            discovered.append({"mac": mac, "name": name})

    discovered.sort(key=lambda d: d["name"].lower())
    return discovered


async def bt_status_sync() -> dict[str, Any]:
    """Read-only adapter + devices + active-profile snapshot.

    Always queries by MAC, never by node name (Plan §B.2). Returns a
    dict suitable for ``/api/admin/bt/status``.
    """
    powered = False
    devices_by_mac: dict[str, dict[str, Any]] = {}
    show = await _run(["bluetoothctl", "show"])
    if show.returncode == 0:
        for line in show.stdout.splitlines():
            line = line.strip()
            if line.startswith("Powered:"):
                powered = line.endswith("yes")

    all_devices = await _run(["bluetoothctl", "devices"])
    if all_devices.returncode == 0:
        for device in _parse_bluetoothctl_devices(all_devices.stdout):
            devices_by_mac[device["mac"]] = {
                **device,
                "paired": False,
                "connected": False,
                "trusted": False,
                "audio": False,
            }

    paired = await _run(["bluetoothctl", "devices", "Paired"])
    if paired.returncode == 0:
        for device in _parse_bluetoothctl_devices(paired.stdout):
            devices_by_mac.setdefault(
                device["mac"],
                {
                    **device,
                    "paired": False,
                    "connected": False,
                    "trusted": False,
                    "audio": False,
                },
            )["paired"] = True

    connected = await _run(["bluetoothctl", "devices", "Connected"])
    if connected.returncode == 0:
        for device in _parse_bluetoothctl_devices(connected.stdout):
            devices_by_mac.setdefault(
                device["mac"],
                {
                    **device,
                    "paired": False,
                    "connected": False,
                    "trusted": False,
                    "audio": False,
                },
            )["connected"] = True

    for mac, device in list(devices_by_mac.items()):
        info_proc = await _run(["bluetoothctl", "info", mac], timeout=4.0)
        if info_proc.returncode != 0:
            continue
        info = _parse_bluetoothctl_info(info_proc.stdout)
        device["name"] = info.get("name") or info.get("alias") or device["name"]
        for key in ("paired", "connected", "trusted", "blocked", "bonded"):
            if key in info:
                device[key] = info[key]
        device["audio"] = _is_audio_device(info)

    devices = sorted(
        devices_by_mac.values(),
        key=lambda d: (
            not bool(d.get("connected")),
            not bool(d.get("paired")),
            not bool(d.get("trusted")),
            str(d.get("name") or "").lower(),
        ),
    )
    return {
        "powered": powered,
        "devices": devices,
        "paired_count": sum(1 for d in devices if d.get("paired")),
        "connected_count": sum(1 for d in devices if d.get("connected")),
        "known_count": len(devices),
    }


async def broadcast_bt_status() -> dict[str, Any]:
    """Broadcast the current Bluetooth status over the shared UI WS."""
    from meeting_scribe.server_support.broadcast import _broadcast_json

    snap = await bt_status_sync()
    await _broadcast_json({"type": "bt_status", "status": snap})
    return snap


def _bluez_sink_node_for_mac(mac: str) -> str:
    return f"bluez_output.{_mac_to_underscored(mac)}.1"


def _configured_sink_matches_mac(configured_sink: str, mac: str) -> bool:
    return bool(configured_sink) and _mac_to_underscored(mac) in configured_sink


async def reconcile_connected_audio_outputs(*, reason: str = "event") -> dict[str, Any]:
    """Low-impact event-driven repair for connected BT audio sinks.

    This does not run discovery scanning. It reacts to BlueZ state that is
    already known, checks whether PipeWire has an output node for connected
    audio devices, and only nudges the specific MAC whose sink is missing.
    """
    from meeting_scribe.audio.audio_routing import SETTINGS_AUDIO_ADMIN_TTS_SINK_NODE
    from meeting_scribe.server_support.settings_store import (
        _load_settings_override,
        _save_settings_override,
    )

    snap = await bt_status_sync()
    settings = _load_settings_override()
    configured_sink = str(settings.get(SETTINGS_AUDIO_ADMIN_TTS_SINK_NODE) or "")
    bt_input_active = bool(settings.get(SETTINGS_BT_INPUT_ACTIVE, False))
    out: dict[str, Any] = {"reason": reason, "updated": [], "skipped": []}
    now = asyncio.get_running_loop().time()

    for device in snap.get("devices", []):
        if not device.get("connected") or not device.get("audio") or device.get("blocked"):
            continue
        mac = str(device.get("mac") or "").upper()
        if not mac:
            continue
        try:
            _, sink = await bt_resolve_nodes(mac)
            if sink and (not configured_sink or _configured_sink_matches_mac(configured_sink, mac)):
                if configured_sink != sink:
                    _save_settings_override({SETTINGS_AUDIO_ADMIN_TTS_SINK_NODE: sink})
                    configured_sink = sink
                    out["updated"].append({"mac": mac, "sink_node": sink})
                continue

            last = _last_audio_reconcile_by_mac.get(mac, 0.0)
            if now - last < _AUDIO_RECONCILE_COOLDOWN_S:
                out["skipped"].append({"mac": mac, "reason": "cooldown"})
                continue
            _last_audio_reconcile_by_mac[mac] = now

            await bt_connect(mac)
            want: Literal["a2dp", "hfp"] = "hfp" if bt_input_active else "a2dp"
            try:
                profile = await bt_choose_profile(mac, want=want)
                await bt_set_profile(mac, profile, timeout=6.0)
            except BluetoothCapabilityError as exc:
                out["skipped"].append({"mac": mac, "reason": str(exc)})
            _, sink = await bt_resolve_nodes(mac)
            if sink:
                if not configured_sink or _configured_sink_matches_mac(configured_sink, mac):
                    _save_settings_override({SETTINGS_AUDIO_ADMIN_TTS_SINK_NODE: sink})
                    configured_sink = sink
                out["updated"].append({"mac": mac, "sink_node": sink})
            else:
                out["skipped"].append({"mac": mac, "reason": "no_sink_after_reconnect"})
        except BluetoothError as exc:
            out["skipped"].append({"mac": mac, "reason": str(exc)})
        except Exception as exc:
            logger.exception("bt audio reconcile failed for %s", mac)
            out["skipped"].append({"mac": mac, "reason": type(exc).__name__})
    if out["updated"] or out["skipped"]:
        logger.info("bt audio reconcile (%s): %s", reason, out)
    return out


async def bt_status_monitor_loop() -> None:
    """Watch BlueZ events and push status snapshots to connected clients.

    ``bluetoothctl monitor`` is intentionally used instead of a second
    polling loop: it follows BlueZ's D-Bus property changes and lets the
    settings card update immediately when a headset powers on/off, connects,
    disconnects, pairs, or is removed.
    """
    last_payload = ""
    reconcile_task: asyncio.Task | None = None

    async def publish_if_changed(*, force: bool = False) -> None:
        nonlocal last_payload
        try:
            snap = await bt_status_sync()
            payload = json.dumps(snap, sort_keys=True)
            if force or payload != last_payload:
                last_payload = payload
                from meeting_scribe.server_support.broadcast import _broadcast_json

                await _broadcast_json({"type": "bt_status", "status": snap})
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("bt status monitor publish failed")

    def schedule_audio_reconcile(reason: str) -> None:
        nonlocal reconcile_task
        if reconcile_task is not None and not reconcile_task.done():
            return

        async def _run_reconcile() -> None:
            try:
                await asyncio.sleep(1.5)
                result = await reconcile_connected_audio_outputs(reason=reason)
                if result.get("updated"):
                    await publish_if_changed(force=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("bt audio reconcile task failed")

        reconcile_task = asyncio.create_task(_run_reconcile(), name="bt-audio-reconcile")

    await publish_if_changed(force=True)
    schedule_audio_reconcile("startup")
    while True:
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                "dbus-monitor",
                "--system",
                "sender='org.bluez'",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            logger.info("bt status monitor started")
            last_emit = 0.0
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode(errors="replace").strip()
                if not text:
                    continue
                lowered = text.lower()
                should_reconcile = any(
                    token in lowered
                    for token in (
                        "connected",
                        "servicesresolved",
                        "profile",
                        "transport",
                    )
                )
                if not any(
                    token in lowered
                    for token in (
                        "connected",
                        "paired",
                        "trusted",
                        "blocked",
                        "servicesresolved",
                        "device",
                        "controller",
                        "powered",
                    )
                ):
                    continue
                now = asyncio.get_running_loop().time()
                if now - last_emit < 0.75:
                    continue
                last_emit = now
                await publish_if_changed()
                if should_reconcile:
                    schedule_audio_reconcile("bluez-event")
        except asyncio.CancelledError:
            if reconcile_task is not None:
                reconcile_task.cancel()
            if proc and proc.returncode is None:
                proc.terminate()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
            raise
        except FileNotFoundError:
            logger.warning("bt status monitor disabled: bluetoothctl not found")
            return
        except Exception:
            logger.exception("bt status monitor crashed; restarting")
        finally:
            if proc and proc.returncode is None:
                proc.terminate()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
        await asyncio.sleep(2.0)


def _pw_dump_to_pactl_cards(pw_dump_text: str, *, mac: str) -> str:
    """Synthesize a pactl-shaped single-card JSON list from pw-dump output.

    Walks the pw-dump objects, finds the ``Audio/Device`` whose
    ``info.props["api.bluez5.address"]`` matches ``mac``, and returns
    a JSON list of length ≤ 1 with the same shape pactl
    ``--format=json list cards`` would emit. Each profile dict carries
    an extra ``_pw_index`` field so the IO layer can map a profile name
    back to the integer index ``wpctl set-profile`` expects.

    Returns ``"[]"`` if no matching device is present.
    """
    try:
        objects = json.loads(pw_dump_text)
    except json.JSONDecodeError as exc:
        raise BluetoothError(f"pw-dump JSON parse failed: {exc}") from exc
    if not isinstance(objects, list):
        return "[]"
    target = _validate_mac(mac)
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if obj.get("type") != "PipeWire:Interface:Device":
            continue
        info = obj.get("info") or {}
        props = info.get("props") or {}
        if (props.get("api.bluez5.address") or "").upper() != target:
            continue
        params = info.get("params") or {}
        enum_profile = params.get("EnumProfile") or []
        active_profile_list = params.get("Profile") or []
        active_name = ""
        if isinstance(active_profile_list, list) and active_profile_list:
            first = active_profile_list[0]
            if isinstance(first, dict):
                active_name = first.get("name") or ""
        profiles_out: list[dict[str, Any]] = []
        for p in enum_profile if isinstance(enum_profile, list) else []:
            if not isinstance(p, dict):
                continue
            avail_raw = p.get("available", "yes")
            avail_bool = avail_raw is True or avail_raw == "yes"
            profiles_out.append(
                {
                    "name": p.get("name") or "",
                    "description": p.get("description") or "",
                    "available": avail_bool,
                    "_pw_index": p.get("index"),
                }
            )
        card = {
            "name": props.get("device.name") or "",
            "properties": {"api.bluez5.address": target},
            "active_profile": active_name,
            "profiles": profiles_out,
            "_pw_id": obj.get("id"),
        }
        return json.dumps([card])
    return "[]"


def _pw_dump_to_pactl_nodes(pw_dump_text: str, *, kind: str) -> str:
    """Synthesize the pactl ``list sources`` / ``list sinks`` shape from pw-dump.

    ``kind`` is ``"Audio/Source"`` or ``"Audio/Sink"``. The returned list
    contains one entry per matching node, with ``properties.device.bus``
    forced to ``"bluetooth"`` whenever the node carries
    ``api.bluez5.address`` — pw-dump nodes don't propagate ``device.bus``
    from the parent device, so we set it explicitly here to satisfy
    :func:`parse_node_for_mac`.
    """
    try:
        objects = json.loads(pw_dump_text)
    except json.JSONDecodeError:
        return "[]"
    out: list[dict[str, Any]] = []
    for obj in objects if isinstance(objects, list) else []:
        if not isinstance(obj, dict):
            continue
        if obj.get("type") != "PipeWire:Interface:Node":
            continue
        info = obj.get("info") or {}
        props = info.get("props") or {}
        if props.get("media.class") != kind:
            continue
        addr = props.get("api.bluez5.address") or ""
        out.append(
            {
                "name": props.get("node.name") or "",
                "properties": {
                    "api.bluez5.address": addr,
                    "device.bus": "bluetooth" if addr else (props.get("device.bus") or ""),
                },
            }
        )
    return json.dumps(out)


def _resolve_card_id_and_profile_index(
    pw_dump_text: str,
    *,
    mac: str,
    profile_name: str,
) -> tuple[int | None, int | None]:
    """Return ``(device_id, profile_index)`` for ``wpctl set-profile``.

    ``device_id`` is the pipewire numeric id of the bluez5 ``Audio/Device``
    matching ``mac``; ``profile_index`` is the entry in EnumProfile whose
    ``name`` matches ``profile_name``. Either may be ``None`` if absent.
    """
    cards_json = _pw_dump_to_pactl_cards(pw_dump_text, mac=mac)
    cards = json.loads(cards_json)
    if not cards:
        return (None, None)
    card = cards[0]
    device_id = card.get("_pw_id")
    profile_index: int | None = None
    for p in card.get("profiles") or []:
        if p.get("name") == profile_name:
            profile_index = p.get("_pw_index")
            break
    return (device_id if isinstance(device_id, int) else None, profile_index)


async def bt_list_profiles(mac: str) -> list[ProfileInfo]:
    """Return the available profiles for the card whose BD_ADDR matches ``mac``."""
    proc = await _run(["pw-dump"])
    if proc.returncode != 0:
        raise BluetoothError(f"pw-dump failed: {proc.stderr.strip()[:200]}")
    return parse_card_profiles(_pw_dump_to_pactl_cards(proc.stdout, mac=mac), mac=mac)


async def bt_choose_profile(mac: str, *, want: Literal["a2dp", "hfp"]) -> str:
    return choose_profile(await bt_list_profiles(mac), want=want)


async def bt_set_profile(mac: str, profile_name: str, *, timeout: float = 6.0) -> None:
    """Switch the bluez5 device's active profile via ``wpctl set-profile``.

    Resolves the pipewire device id by MAC, looks up the EnumProfile
    index whose name matches ``profile_name``, then issues
    ``wpctl set-profile <device-id> <profile-index>`` and polls
    pw-dump until the active profile name flips. Raises
    :class:`BluetoothError` on timeout or backend failure (the bridge
    catches and falls into Disconnected).

    A profile flip that requires SCO link establishment (HFP/HSP) can
    take several seconds on the GB10's MT7925 — the BlueZ stack
    re-negotiates roles before pw-dump's ``Profile`` param refreshes,
    so the polling window is generous to avoid false-negative timeouts.
    """
    dump = await _run(["pw-dump"])
    if dump.returncode != 0:
        raise BluetoothError(f"pw-dump failed: {dump.stderr.strip()[:200]}")
    device_id, profile_index = _resolve_card_id_and_profile_index(
        dump.stdout, mac=mac, profile_name=profile_name
    )
    if device_id is None:
        raise BluetoothError(f"no bluez5 device for mac {mac}")
    if profile_index is None:
        raise BluetoothError(f"no profile {profile_name!r} on {mac}")
    issue = await _run(["wpctl", "set-profile", str(device_id), str(profile_index)])
    if issue.returncode != 0:
        raise BluetoothError(f"wpctl set-profile failed: {issue.stderr.strip()[:200]}")

    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        check = await _run(["pw-dump"])
        if check.returncode == 0:
            cards = json.loads(_pw_dump_to_pactl_cards(check.stdout, mac=mac))
            if cards and cards[0].get("active_profile") == profile_name:
                return
        await asyncio.sleep(0.1)
    raise BluetoothError(f"timed out waiting for active_profile={profile_name}")


async def bt_resolve_nodes(mac: str) -> tuple[str | None, str | None]:
    dump = await _run(["pw-dump"])
    if dump.returncode != 0:
        return (None, None)
    sources = _pw_dump_to_pactl_nodes(dump.stdout, kind="Audio/Source")
    sinks = _pw_dump_to_pactl_nodes(dump.stdout, kind="Audio/Sink")
    return parse_node_for_mac(sources, sinks, mac=mac)


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


async def _connected_audio_macs() -> list[str]:
    """Return MAC addresses of currently-connected BT devices.

    Used to drive :func:`apply_bt_input_state` without requiring the
    operator to have explicitly stored ``bt_device_mac`` — if you've
    paired + connected a single headset that's the obvious target.
    """
    proc = await _run(["bluetoothctl", "devices", "Connected"])
    out: list[str] = []
    if proc.returncode != 0:
        return out
    for line in proc.stdout.splitlines():
        parts = line.strip().split(" ", 2)
        if len(parts) >= 2 and parts[0] == "Device":
            out.append(parts[1].upper())
    return out


async def apply_bt_input_state(active: bool) -> dict[str, Any]:
    """Reconcile the BT card profile with the operator's HFP toggle.

    Walks every connected BT device and either flips it to the
    highest-codec HFP profile (``active=True`` — mic available) or to
    the highest-codec A2DP profile (``active=False`` — mic dropped,
    high-fidelity playback restored).

    Devices without an HFP profile (e.g. a pure A2DP speaker) are
    skipped without raising; the result dict's ``skipped`` list lets
    the admin UI surface the "device cannot do HFP" case. Called from
    server lifespan (boot reconciliation) and from
    ``/api/admin/bt/mic`` so the operator's toggle has immediate
    physical effect instead of just persisting to settings.
    """
    out: dict[str, Any] = {"updated": [], "skipped": []}
    want: Literal["a2dp", "hfp"] = "hfp" if active else "a2dp"
    for mac in await _connected_audio_macs():
        try:
            target_profile = await bt_choose_profile(mac, want=want)
            await bt_set_profile(mac, target_profile)
            out["updated"].append({"mac": mac, "profile": target_profile})
        except BluetoothCapabilityError as exc:
            out["skipped"].append({"mac": mac, "reason": str(exc)})
        except BluetoothError as exc:
            out["skipped"].append({"mac": mac, "reason": str(exc)})
    return out
