"""Audio routing settings + PipeWire device enumeration.

Single source of truth for "which physical audio device feeds ASR"
(``audio_meeting_mic_node``) and "which physical audio device plays
admin TTS" (``audio_admin_tts_sink_node``). The admin UI in
``admin-audio-card.js`` reads + writes these via
``GET / POST /api/admin/audio/*``; the lifespan starts the
:mod:`server_mic` capture and the :mod:`local_sink` listener with
their target nodes pulled from these settings.

Why a routing layer at all
--------------------------
v1 wired the audio fan-out to a single hardcoded path: browser-mic →
``ws/audio_input`` → ASR, plus a guest WebSocket that distributed TTS.
A meeting room with a USB speakerphone (Poly Sync 20-M, the canonical
case) wants the server itself to capture from the device's mic and
play TTS through the device's speaker — no operator browser tab in
the loop. Splitting source / sink configuration out of bt.py and into
this module lets BT, USB, and ALSA devices be selected uniformly from
one UI control.

Pure-function ``parse_pw_dump_devices`` is unit-tested with a fixture;
the IO functions wrap pw-dump + wpctl in async ``run_in_executor``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import time
from typing import Any, Literal

logger = logging.getLogger(__name__)


# Settings keys (persisted in ``~/.config/meeting-scribe/settings.json``)


SETTINGS_AUDIO_MEETING_MIC_NODE = "audio_meeting_mic_node"
SETTINGS_AUDIO_ADMIN_TTS_SINK_NODE = "audio_admin_tts_sink_node"
SETTINGS_AUDIO_ROOM_TTS_SINK_NODE = "audio_room_tts_sink_node"
SETTINGS_AUDIO_MEETING_MIC_ACTIVE = "audio_meeting_mic_active"
# Durable identity for the configured mic, used by ``reconcile_audio_routing``
# to auto-rebind when the literal node name drifts (wireplumber's per-instance
# ``.N`` suffix changes on USB reconnect). ``stable_id`` is the
# ``_physical_audio_device_id`` output (``usb:<vid:pid+serial>`` /
# ``bluez:<MAC>`` / ``pci:<addr>``). The ``discriminator`` (port / nick /
# channels / direction) narrows when a single physical device exposes
# multiple sources — see ``resolve_node_for_stable_id``.
SETTINGS_AUDIO_MEETING_MIC_STABLE_ID = "audio_meeting_mic_stable_id"
SETTINGS_AUDIO_MEETING_MIC_DISCRIMINATOR = "audio_meeting_mic_discriminator"
SETTINGS_AUDIO_MEETING_MIC_REBOUND_FROM = "audio_meeting_mic_rebound_from"
SETTINGS_AUDIO_MEETING_MIC_REBOUND_TO = "audio_meeting_mic_rebound_to"
SETTINGS_AUDIO_MEETING_MIC_REBOUND_AT = "audio_meeting_mic_rebound_at"
SETTINGS_AUDIO_MIC_AMBIGUOUS_CANDIDATES = "audio_meeting_mic_ambiguous_candidates"

# Legacy storage key (pre-routing-card era). Read only by the one-shot
# migration in :mod:`meeting_scribe.server_support.settings_store`. The
# canonical value lives at :data:`SETTINGS_AUDIO_ADMIN_TTS_SINK_NODE`;
# the legacy key is rewritten as ``..._legacy_backup`` — never read
# by code, kept as a recovery breadcrumb for manual rollback.
LEGACY_SETTINGS_AUDIO_MEETING_SINK_NODE = "audio_meeting_sink_node"
LEGACY_SETTINGS_AUDIO_MEETING_SINK_NODE_BACKUP = "audio_admin_tts_sink_node_legacy_backup"


# Capture format defaults — Poly Sync 20-M and most USB speakerphones
# expose a 48 kHz mono fallback that ``_handle_audio``'s torchaudio
# resampler downsamples to 16 kHz with anti-aliasing. Higher rate +
# better DSP than HFP-mSBC's 16 kHz transparent SCO.
DEFAULT_CAPTURE_RATE = 48000
DEFAULT_CAPTURE_CHANNELS = 1


_DeviceKind = Literal["source", "sink"]


def _classify_device(node_name: str, device_api: str | None) -> str:
    """Return a coarse label the UI groups by: ``"usb"`` / ``"bluetooth"``
    / ``"hdmi"`` / ``"alsa"``.

    Pure heuristic on the node name + ``device.api`` PipeWire prop —
    sufficient for grouping devices in a dropdown, never used for
    routing decisions.
    """
    if (device_api or "").lower() == "bluez5":
        return "bluetooth"
    name = (node_name or "").lower()
    if "usb-" in name:
        return "usb"
    if "hdmi" in name or "displayport" in name:
        return "hdmi"
    return "alsa"


def _parse_pw_dump_arrays(pw_dump_text: str) -> list[dict[str, Any]]:
    """Parse pw-dump output, tolerant of multiple concatenated JSON arrays.

    pw-dump (even without ``-m``) sometimes emits two snapshots in a
    single stdout when PipeWire activity races the dump call — e.g. a
    parallel ``wpctl status`` triggers a Client added/removed event
    that pw-dump rolls into a second array before exiting. The
    standard ``json.loads`` rejects this with "Extra data". We use
    :class:`json.JSONDecoder.raw_decode` in a loop so every snapshot
    is parsed; later snapshots win on node id collisions, so the
    returned list reflects the freshest state.
    """
    decoder = json.JSONDecoder()
    text = pw_dump_text.lstrip()
    merged: dict[Any, dict[str, Any]] = {}
    while text:
        try:
            chunk, idx = decoder.raw_decode(text)
        except json.JSONDecodeError as exc:
            logger.warning("pw-dump JSON parse failed: %s", exc)
            break
        if isinstance(chunk, list):
            for item in chunk:
                if isinstance(item, dict):
                    merged[item.get("id")] = item
        elif isinstance(chunk, dict):
            merged[chunk.get("id")] = chunk
        text = text[idx:].lstrip()
    return list(merged.values())


def parse_pw_dump_devices(pw_dump_text: str) -> dict[str, list[dict[str, Any]]]:
    """Walk pw-dump output and return ``{"sources": [...], "sinks": [...]}``.

    Each entry: ``{"node_id", "node_name", "description", "kind",
    "device_class"}``. ``description`` falls back to ``node_name`` when
    the device didn't supply one. Pure function so the test suite can
    drive it from a fixture without spawning pw-dump.

    Tolerant of multiple concatenated JSON arrays — see
    :func:`_parse_pw_dump_arrays`.
    """
    data = _parse_pw_dump_arrays(pw_dump_text)
    sources: list[dict[str, Any]] = []
    sinks: list[dict[str, Any]] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        if obj.get("type") != "PipeWire:Interface:Node":
            continue
        info = obj.get("info") or {}
        props = info.get("props") or {}
        media_class = props.get("media.class")
        if media_class not in ("Audio/Source", "Audio/Sink"):
            continue
        node_name = props.get("node.name") or ""
        if not node_name:
            continue
        kind: _DeviceKind = "source" if media_class == "Audio/Source" else "sink"
        # Discriminator narrows a stable_id when one physical device exposes
        # multiple sources or profiles (composite USB cards, hubs with two
        # identical headsets, etc.). Pulled straight from pw-dump props so
        # the test suite can drive it from fixtures without device hardware.
        channels = props.get("audio.channels") or props.get("audio.channelmap")
        try:
            channels_int = int(channels) if channels is not None else None
        except TypeError, ValueError:
            channels_int = None
        discriminator = {
            "port": (
                props.get("device.profile.name")
                or props.get("api.alsa.pcm.stream")
                or props.get("port.name")
                or ""
            ),
            "nick": props.get("node.nick") or "",
            "channels": channels_int,
            "direction": kind,
        }
        entry = {
            "node_id": obj.get("id"),
            "node_name": node_name,
            "description": props.get("node.description") or node_name,
            "kind": kind,
            "device_class": _classify_device(node_name, props.get("device.api")),
            "stable_id": _physical_audio_device_id(node_name),
            "discriminator": discriminator,
        }
        if kind == "source":
            sources.append(entry)
        else:
            sinks.append(entry)
    sources.sort(key=lambda e: (e["device_class"], e["description"].lower()))
    sinks.sort(key=lambda e: (e["device_class"], e["description"].lower()))
    return {"sources": sources, "sinks": sinks}


# wpctl status decorates each row with box-drawing chars (``│``,
# ``├─``) that aren't ``\s``, so we can't anchor the regex at the
# start. Match the asterisk + id + name anywhere on the line.
_DEFAULT_RE = re.compile(r"\*\s+(\d+)\.\s+(.+?)\s+\[")


def parse_wpctl_defaults(wpctl_status_text: str) -> dict[str, int | None]:
    """Extract default sink + source ids from ``wpctl status`` output.

    The default markers are the ``*`` rows under the ``Sinks:`` and
    ``Sources:`` sections. Returns ``{"source": id_or_None,
    "sink": id_or_None}``.
    """
    section: str | None = None
    out: dict[str, int | None] = {"source": None, "sink": None}
    for raw in wpctl_status_text.splitlines():
        line = raw.rstrip()
        # Section headers look like ``├─ Sinks:`` / ``├─ Sources:``.
        if line.endswith("Sinks:"):
            section = "sink"
            continue
        if line.endswith("Sources:"):
            section = "source"
            continue
        if line.endswith("Sink endpoints:") or line.endswith("Source endpoints:"):
            section = None
            continue
        if section is None:
            continue
        m = _DEFAULT_RE.search(line)
        if m and out.get(section) is None:
            try:
                out[section] = int(m.group(1))
            except ValueError:
                pass  # pw-metadata default-node id non-numeric — leave section unset for caller
    return out


def _physical_audio_device_id(node_name: str | None) -> str:
    """Return a stable physical-device key for a PipeWire audio node.

    PipeWire exposes input and output nodes with different prefixes and
    profile suffixes. For room TTS we need to know whether a selected speaker
    and mic are the same piece of hardware, e.g. the Poly Sync USB sink/source.
    Empty string means "unknown", and callers should treat unknown as unsafe.
    """
    name = (node_name or "").strip()
    if not name:
        return ""
    bluez = re.match(r"^bluez_(?:input|output)\.([A-Fa-f0-9_:-]+)", name)
    if bluez:
        return f"bluez:{bluez.group(1).replace('_', ':').lower()}"
    usb = re.match(r"^alsa_(?:input|output)\.usb-(.+?)-\d{2}\.", name)
    if usb:
        return f"usb:{usb.group(1).lower()}"
    pci = re.match(r"^alsa_(?:input|output)\.pci-(.+?)\.", name)
    if pci:
        return f"pci:{pci.group(1).lower()}"
    # Last-resort normalization for synthetic test nodes or unusual ALSA
    # names. Keep this conservative so unrelated devices do not match.
    generic = re.sub(r"^alsa_(?:input|output)\.", "", name)
    generic = re.sub(r"\.(?:analog|iec958|hdmi|mono|stereo|surround).*$", "", generic)
    return generic.lower() if generic and generic != name else ""


def audio_nodes_share_physical_device(source_node: str | None, sink_node: str | None) -> bool:
    """True when source + sink appear to be two endpoints of one device."""
    source_id = _physical_audio_device_id(source_node)
    sink_id = _physical_audio_device_id(sink_node)
    return bool(source_id and sink_id and source_id == sink_id)


# ── stable_id resolution (tagged union) ────────────────────────
#
# Persisted mic bindings drift on USB reconnect because wireplumber appends
# a per-instance ``.N`` counter to the node name. The resolver below maps
# a persisted ``(stable_id, discriminator)`` pair back to whatever node
# name the device is currently exposing. It fails closed on ambiguity:
# if the stable_id matches multiple present sources and the discriminator
# can't narrow to exactly one, we return ``Ambiguous(candidates=…)`` and
# the operator gets a banner to pick one.


class ResolveResult:
    """Tagged union returned by :func:`resolve_node_for_stable_id`."""

    kind: str = "abstract"


class Resolved(ResolveResult):
    kind = "resolved"

    def __init__(self, node_name: str) -> None:
        self.node_name = node_name


class Ambiguous(ResolveResult):
    kind = "ambiguous"

    def __init__(self, candidates: list[dict[str, Any]]) -> None:
        self.candidates = candidates


class NotFound(ResolveResult):
    kind = "not_found"


def _discriminator_matches(want: dict[str, Any] | None, have: dict[str, Any] | None) -> bool:
    """Compare a persisted discriminator against a live one.

    Empty / ``None`` ``want`` means the persisted binding was legacy
    (no discriminator), so any live entry matches. ``have`` should be the
    discriminator dict the enumeration step put on the device entry.
    Field-by-field equality on non-empty ``want`` fields only: a field
    that's missing on ``want`` doesn't constrain.
    """
    if not want:
        return True
    have = have or {}
    for key, expected in want.items():
        if expected in (None, ""):
            continue
        actual = have.get(key)
        if actual != expected:
            return False
    return True


def resolve_node_for_stable_id(
    stable_id: str,
    discriminator: dict[str, Any] | None,
    devices: list[dict[str, Any]],
) -> ResolveResult:
    """Find the current node name for a persisted ``(stable_id, discriminator)``.

    ``devices`` is a flat list of device entries (typically
    ``enumerate_audio_devices()["sources"]``). Returns a tagged union so
    callers can distinguish the three outcomes:

    * :class:`Resolved` — exactly one entry matched.
    * :class:`Ambiguous` — multiple entries shared the stable_id and the
      discriminator could not narrow to one. Carries the candidate list so
      the operator can disambiguate from the admin UI.
    * :class:`NotFound` — no entry matched (likely transient disconnect).

    An empty / falsy ``stable_id`` always returns :class:`NotFound` — the
    caller should not have invoked this with an empty identity.
    """
    if not stable_id:
        return NotFound()
    matches_by_stable: list[dict[str, Any]] = [
        d for d in devices if d.get("stable_id") == stable_id
    ]
    if not matches_by_stable:
        return NotFound()
    if len(matches_by_stable) == 1:
        return Resolved(node_name=matches_by_stable[0]["node_name"])
    # Multiple stable_id matches — try to disambiguate by discriminator.
    narrowed = [
        d
        for d in matches_by_stable
        if _discriminator_matches(discriminator, d.get("discriminator"))
    ]
    if len(narrowed) == 1:
        return Resolved(node_name=narrowed[0]["node_name"])
    # Fail closed — surface candidates so the operator can pick one.
    return Ambiguous(
        candidates=[
            {"node_name": d["node_name"], "discriminator": d.get("discriminator") or {}}
            for d in matches_by_stable
        ]
    )


def admin_room_sinks_collide(
    *,
    admin_sink: str | None,
    room_sink: str | None,
    admin_lang: str | None,
    room_lang: str | None,
) -> bool:
    """True when admin TTS and room TTS would render the same language to
    the same physical speaker, producing the dual-render echo we hit on
    Poly Sync 20-M when both routes were pointed at the same device.

    The collision rule: admin and room sinks share a physical PipeWire
    device AND room TTS would also emit the admin's target language —
    either room is "all" (every translated language) or room equals
    admin. Different room/admin language picks on a shared speaker are
    intentional bilingual output, not a bug.
    """
    if not admin_sink or not room_sink:
        return False
    if not audio_nodes_share_physical_device(admin_sink, room_sink):
        return False
    admin_lang = (admin_lang or "").strip().lower() or "en"
    room_lang = (room_lang or "").strip().lower() or "all"
    return room_lang == "all" or room_lang == admin_lang


async def _run(argv: list[str], *, timeout: float = 5.0) -> subprocess.CompletedProcess:
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


_VOLUME_RE = re.compile(r"Volume:\s*([0-9]+(?:\.[0-9]+)?)")


def parse_wpctl_volume(text: str) -> dict[str, Any] | None:
    """Parse ``wpctl get-volume`` output into ``{"volume": float, "muted": bool}``.

    Sample inputs::

        "Volume: 0.85\\n"
        "Volume: 1.00 [MUTED]\\n"

    Returns ``None`` when the line doesn't match (wpctl prints an error
    to stderr and a non-Volume line to stdout when the node id is
    unknown — better to drop the entry than guess a level).
    """
    if not text:
        return None
    m = _VOLUME_RE.search(text)
    if not m:
        return None
    try:
        volume = float(m.group(1))
    except ValueError:
        return None
    muted = "[MUTED]" in text
    return {"volume": volume, "muted": muted}


async def fetch_device_volume(node_ref: int | str) -> dict[str, Any] | None:
    """``wpctl get-volume <node_ref>`` → parsed dict, or ``None`` on failure.

    ``node_ref`` may be the numeric node id (preferred — stable for the
    lifetime of the PipeWire session) or a wpctl alias like
    ``@DEFAULT_AUDIO_SINK@``.
    """
    proc = await _run(["wpctl", "get-volume", str(node_ref)], timeout=2.0)
    if proc.returncode != 0:
        return None
    return parse_wpctl_volume(proc.stdout)


async def set_device_volume(
    node_ref: int | str,
    *,
    volume: float | None = None,
    muted: bool | None = None,
) -> bool:
    """Apply a volume and/or mute change. Returns False if any wpctl call failed.

    ``volume`` is in linear 0.0-1.5 range (1.0 = 100%, anything above 1.0
    is amplification). The wpctl default ceiling is 1.5 — caller is
    expected to validate the range.
    """
    ok = True
    if volume is not None:
        proc = await _run(["wpctl", "set-volume", str(node_ref), f"{volume:.2f}"], timeout=2.0)
        if proc.returncode != 0:
            logger.warning(
                "wpctl set-volume %s %.2f failed: %s",
                node_ref,
                volume,
                (proc.stderr or proc.stdout).strip(),
            )
            ok = False
    if muted is not None:
        proc = await _run(
            ["wpctl", "set-mute", str(node_ref), "1" if muted else "0"],
            timeout=2.0,
        )
        if proc.returncode != 0:
            logger.warning(
                "wpctl set-mute %s %s failed: %s",
                node_ref,
                muted,
                (proc.stderr or proc.stdout).strip(),
            )
            ok = False
    return ok


async def enumerate_audio_devices() -> dict[str, list[dict[str, Any]]]:
    """Live ``pw-dump``-driven enumeration of audio sinks + sources.

    Augments each entry with ``is_default: bool`` from
    ``wpctl status`` and ``volume: float`` + ``muted: bool`` from
    ``wpctl get-volume``. Returns ``{"sources": [...], "sinks": [...]}``.
    """
    dump_proc, status_proc = await asyncio.gather(
        _run(["pw-dump"]),
        _run(["wpctl", "status"]),
    )
    devices = parse_pw_dump_devices(dump_proc.stdout if dump_proc.returncode == 0 else "")
    defaults = parse_wpctl_defaults(status_proc.stdout) if status_proc.returncode == 0 else {}
    for src in devices["sources"]:
        src["is_default"] = src.get("node_id") == defaults.get("source")
    for sink in devices["sinks"]:
        sink["is_default"] = sink.get("node_id") == defaults.get("sink")

    all_entries = devices["sources"] + devices["sinks"]
    if all_entries:
        volumes = await asyncio.gather(
            *(fetch_device_volume(e["node_id"]) for e in all_entries),
            return_exceptions=True,
        )
        for entry, vol in zip(all_entries, volumes):
            if isinstance(vol, dict):
                entry["volume"] = vol["volume"]
                entry["muted"] = vol["muted"]
            else:
                entry["volume"] = None
                entry["muted"] = None
    return devices


def get_routing_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Read the persisted routing config, normalized + with defaults filled.

    Pure function that takes a settings dict (as returned by
    :func:`_load_settings_override`) so callers don't have to thread
    the file IO through their tests.
    """
    discriminator = settings.get(SETTINGS_AUDIO_MEETING_MIC_DISCRIMINATOR)
    if not isinstance(discriminator, dict):
        discriminator = None
    return {
        "mic_node": settings.get(SETTINGS_AUDIO_MEETING_MIC_NODE) or "",
        "admin_sink_node": settings.get(SETTINGS_AUDIO_ADMIN_TTS_SINK_NODE) or "",
        "room_sink_node": settings.get(SETTINGS_AUDIO_ROOM_TTS_SINK_NODE) or "",
        "mic_active": bool(settings.get(SETTINGS_AUDIO_MEETING_MIC_ACTIVE, False)),
        "mic_stable_id": settings.get(SETTINGS_AUDIO_MEETING_MIC_STABLE_ID) or "",
        "mic_discriminator": discriminator,
    }


def validate_routing_payload(
    body: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    """Validate a routing-config payload coming from the setup wizard or
    the admin POST route.

    Returns ``(updates, error_message)`` — exactly one of which is
    non-empty. ``updates`` is the settings.json patch keyed by the
    canonical SETTINGS_AUDIO_* keys; ``error_message`` is a short
    human-readable string ready to embed in a 400 response.

    Accepts any subset of ``{mic_node, admin_sink_node, room_sink_node,
    mic_active}``. Strings are normalized: empty string clears, ``None``
    is treated as empty. The companion ``validate_room_tts_route``
    check happens in the route handler since it needs the merged
    "current settings + this patch" view.
    """
    updates: dict[str, Any] = {}
    if "mic_node" in body:
        v = body.get("mic_node")
        if v is not None and not isinstance(v, str):
            return {}, "mic_node must be string or null"
        updates[SETTINGS_AUDIO_MEETING_MIC_NODE] = (v or "").strip()
    if "admin_sink_node" in body:
        v = body.get("admin_sink_node")
        if v is not None and not isinstance(v, str):
            return {}, "admin_sink_node must be string or null"
        updates[SETTINGS_AUDIO_ADMIN_TTS_SINK_NODE] = (v or "").strip()
    if "room_sink_node" in body:
        v = body.get("room_sink_node")
        if v is not None and not isinstance(v, str):
            return {}, "room_sink_node must be string or null"
        updates[SETTINGS_AUDIO_ROOM_TTS_SINK_NODE] = (v or "").strip()
    if "mic_active" in body:
        v = body.get("mic_active")
        if not isinstance(v, bool):
            return {}, "mic_active must be boolean"
        updates[SETTINGS_AUDIO_MEETING_MIC_ACTIVE] = v
    if "mic_stable_id" in body:
        v = body.get("mic_stable_id")
        if v is not None and not isinstance(v, str):
            return {}, "mic_stable_id must be string or null"
        updates[SETTINGS_AUDIO_MEETING_MIC_STABLE_ID] = (v or "").strip()
    if "mic_discriminator" in body:
        v = body.get("mic_discriminator")
        if v is not None and not isinstance(v, dict):
            return {}, "mic_discriminator must be object or null"
        updates[SETTINGS_AUDIO_MEETING_MIC_DISCRIMINATOR] = v
    return updates, None


async def derive_stable_identity_for_node(node_name: str) -> tuple[str, dict[str, Any] | None]:
    """Look up the live ``(stable_id, discriminator)`` pair for a node name.

    Used by the admin route handler when the operator picks a mic without
    explicitly supplying stable_id / discriminator — we resolve them from
    the live enumeration so persistence is self-consistent without making
    the UI do the work. Returns ``("", None)`` when the node isn't present
    in the current enumeration (the route handler logs a WARN and persists
    only the node name in that case; auto-rebind won't be possible until
    the operator re-saves while the device is connected).
    """
    if not node_name:
        return "", None
    devices = await enumerate_audio_devices()
    for entry in devices.get("sources", []) + devices.get("sinks", []):
        if entry.get("node_name") == node_name:
            return entry.get("stable_id") or "", entry.get("discriminator")
    return "", None


def _clear_audio_route_failure_state() -> None:
    """Flip the route status back to ``"ok"`` and dismiss stale failure notices.

    Called by both success paths in :func:`_reconcile_mic_with_stable_id`.
    Codex review caught that without this, a device that recovers leaves
    ``audio_route_status`` latched on ``"ambiguous"`` / ``"unresolved"`` /
    ``"capture_failed"`` and Phase 5 meeting-start preflight blocks the
    meeting forever. Calling this on every successful reconcile is cheap:
    if nothing was latched, the dismiss helpers no-op.
    """
    from meeting_scribe.runtime import state as _state
    from meeting_scribe.server_support import admin_notifications
    from meeting_scribe.server_support.settings_store import (
        _load_settings_override,
        _save_settings_override,
    )

    _state.audio_route_status = "ok"
    # Drop the ambiguous-candidates row if present.
    if SETTINGS_AUDIO_MIC_AMBIGUOUS_CANDIDATES in _load_settings_override():
        _save_settings_override({SETTINGS_AUDIO_MIC_AMBIGUOUS_CANDIDATES: None})
    admin_notifications.dismiss_if_present("mic_ambiguous")
    admin_notifications.dismiss_if_present("mic_unresolved")
    admin_notifications.dismiss_if_present("mic_capture_failed")


async def _verify_server_mic_started(*, timeout_s: float = 2.0) -> bool:
    """Poll ``state.server_mic_active`` for up to ``timeout_s`` seconds.

    ``ServerMicCapture.start()`` flips the flag synchronously today, but
    the reader task can die immediately after (bad node name, kernel mic
    busy, etc.). A brief poll catches the case where reconcile_server_mic
    "succeeded" by API but the capture is already dead.
    """
    from meeting_scribe.runtime import state as _state

    deadline = asyncio.get_event_loop().time() + timeout_s
    while True:
        if _state.server_mic_active:
            return True
        if asyncio.get_event_loop().time() >= deadline:
            return False
        await asyncio.sleep(0.1)


async def _reconcile_mic_with_stable_id(
    *,
    mic_node: str,
    mic_active: bool,
    mic_stable_id: str,
    mic_discriminator: dict[str, Any] | None,
    reconcile_server_mic_fn: Any,
) -> dict[str, Any]:
    """Resolve the persisted mic binding against the live PipeWire enumeration
    and apply the appropriate routing path. Returns the ``result["mic"]``
    sub-dict described in :func:`reconcile_audio_routing`.

    Split out into its own function so the route handler and meeting-start
    preflight can call it without duplicating the path logic, and so the
    unit-test fixtures can drive it without spawning pw-dump.
    """
    from meeting_scribe.runtime import state as _state
    from meeting_scribe.server_support import admin_notifications
    from meeting_scribe.server_support.settings_store import _save_settings_override

    # mic_active=False shortcuts everything — the operator deliberately
    # disabled the server-side mic. Push the disabled state through and
    # clear any stale failure status.
    if not mic_active or not mic_node:
        try:
            await reconcile_server_mic_fn(mic_node=mic_node, mic_active=mic_active)
        except Exception:
            logger.exception("audio_routing: server-mic reconcile (disabled path) failed")
        _clear_audio_route_failure_state()
        return {"ok": True, "error_message": None, "status": "ok"}

    devices = await enumerate_audio_devices()
    sources = devices.get("sources", [])
    present_node_names = {s.get("node_name") for s in sources}

    if mic_node in present_node_names:
        # ── Path A: node still present. Verify identity hasn't drifted.
        live_entry = next((s for s in sources if s.get("node_name") == mic_node), None)
        live_stable = (live_entry or {}).get("stable_id") or ""
        live_disc = (live_entry or {}).get("discriminator")
        identity_ok = True
        if mic_stable_id and live_stable and mic_stable_id != live_stable:
            identity_ok = False
        if mic_discriminator and not _discriminator_matches(mic_discriminator, live_disc):
            identity_ok = False
        if identity_ok:
            try:
                await reconcile_server_mic_fn(mic_node=mic_node, mic_active=True)
            except Exception as exc:
                logger.exception(
                    "audio_routing: path A — server-mic reconcile raised for node=%s",
                    mic_node,
                )
                return _capture_failed_path(
                    detail=f"{exc!s}", admin_notifications=admin_notifications
                )
            if not await _verify_server_mic_started():
                logger.error(
                    "audio_routing: path A — capture process did not start for node=%s",
                    mic_node,
                )
                return _capture_failed_path(
                    detail="server_mic_active did not flip to True within 2s",
                    admin_notifications=admin_notifications,
                )
            _clear_audio_route_failure_state()
            return {"ok": True, "error_message": None, "status": "ok"}
        # Identity drift — fall through to resolver as if node was missing.
        logger.warning(
            "audio_routing: path A — node %s present but identity drifted "
            "(persisted stable_id=%s discriminator=%s, live stable_id=%s discriminator=%s); "
            "falling through to resolver",
            mic_node,
            mic_stable_id,
            mic_discriminator,
            live_stable,
            live_disc,
        )

    # ── Resolver: stable_id-driven rebind / ambiguous / not_found
    if not mic_stable_id:
        # Legacy persisted binding with no stable_id — the node is gone and
        # we can't auto-rebind. Same as NotFound, just with a different
        # detail message.
        _state.audio_route_status = "unresolved"
        admin_notifications.put_notification(
            "mic_unresolved",
            mic_node=mic_node,
            detail=(
                "Configured mic node is not currently present and no stable identity "
                "is stored — re-save the mic selection while the device is connected."
            ),
        )
        logger.warning(
            "audio_routing: mic node %s not present; no stable_id stored — "
            "no auto-rebind possible until operator re-saves",
            mic_node,
        )
        return {
            "ok": False,
            "error_message": "mic node missing (legacy binding)",
            "status": "unresolved",
        }

    resolved = resolve_node_for_stable_id(mic_stable_id, mic_discriminator, sources)

    if isinstance(resolved, Resolved):
        # ── Path B: rebound. Update persisted mic_node, keep identity.
        rebound_to = resolved.node_name
        rebound_from = mic_node
        # No-op detection: the stable-id resolver returned the same
        # node we already had pinned. Nothing actually changed for the
        # operator — wireplumber refreshed, our `.N` suffix happened to
        # match, capture stayed up the whole time. Surfacing this as a
        # banner notification is a UX wart: the operator sees
        # "Microphone auto-rebound — X → X" and has no idea why they
        # were just told their mic moved. Skip the notification + the
        # persisted "rebind happened" trio for the no-op case; still
        # run the reconcile to be sure the server-mic process is alive.
        # ALSO actively dismiss any stale mic_rebound row whose
        # from == to so existing bad notifications fade on the next
        # /api/status tick after this fix ships.
        if rebound_from == rebound_to:
            admin_notifications.dismiss_if_present("mic_rebound")
            try:
                await reconcile_server_mic_fn(mic_node=rebound_to, mic_active=True)
            except Exception as exc:
                logger.exception(
                    "audio_routing: path B (no-op) — server-mic reconcile raised at %s",
                    rebound_to,
                )
                return _capture_failed_path(
                    detail=f"{exc!s}",
                    admin_notifications=admin_notifications,
                )
            _state.audio_route_status = "ok"
            return {
                "ok": True,
                "error_message": None,
                "mic_node": rebound_to,
                "status": "ok",
            }
        _save_settings_override(
            {
                SETTINGS_AUDIO_MEETING_MIC_NODE: rebound_to,
                SETTINGS_AUDIO_MEETING_MIC_REBOUND_FROM: rebound_from,
                SETTINGS_AUDIO_MEETING_MIC_REBOUND_TO: rebound_to,
                SETTINGS_AUDIO_MEETING_MIC_REBOUND_AT: time.time(),
            }
        )
        admin_notifications.put_notification(
            "mic_rebound",
            mic_from=rebound_from,
            mic_to=rebound_to,
            stable_id=mic_stable_id,
        )
        logger.info(
            "audio_routing: auto-rebound mic %s → %s (stable_id=%s)",
            rebound_from,
            rebound_to,
            mic_stable_id,
        )
        try:
            await reconcile_server_mic_fn(mic_node=rebound_to, mic_active=True)
        except Exception as exc:
            logger.exception(
                "audio_routing: path B — server-mic reconcile raised after rebind to %s",
                rebound_to,
            )
            return _capture_failed_path(
                detail=f"{exc!s}",
                admin_notifications=admin_notifications,
            )
        if not await _verify_server_mic_started():
            logger.error(
                "audio_routing: path B — capture process did not start after rebind to %s",
                rebound_to,
            )
            return _capture_failed_path(
                detail="server_mic_active did not flip to True within 2s after rebind",
                admin_notifications=admin_notifications,
            )
        # Clear stale failure notifications (the new rebound notice is the
        # only one that should remain visible) — call AFTER put_notification
        # so we don't accidentally dismiss the row we just persisted.
        _state.audio_route_status = "ok"
        admin_notifications.dismiss_if_present("mic_ambiguous")
        admin_notifications.dismiss_if_present("mic_unresolved")
        admin_notifications.dismiss_if_present("mic_capture_failed")
        return {
            "ok": True,
            "error_message": None,
            "status": "rebound",
            "rebound_from": rebound_from,
            "rebound_to": rebound_to,
        }

    if isinstance(resolved, Ambiguous):
        # ── Path C: keep persisted, surface candidates.
        _state.audio_route_status = "ambiguous"
        _save_settings_override({SETTINGS_AUDIO_MIC_AMBIGUOUS_CANDIDATES: resolved.candidates})
        admin_notifications.put_notification(
            "mic_ambiguous",
            stable_id=mic_stable_id,
            candidates=resolved.candidates,
            detail=(
                "Multiple sources match the configured device identity — open the "
                "audio route admin page to pick the intended source."
            ),
        )
        logger.error(
            "audio_routing: mic stable_id=%s matches %d sources; need operator pick",
            mic_stable_id,
            len(resolved.candidates),
        )
        return {
            "ok": False,
            "error_message": "ambiguous mic identity (multiple matching sources)",
            "status": "ambiguous",
            "candidates": resolved.candidates,
        }

    # ── Path D: NotFound (transient disconnect or removed device).
    _state.audio_route_status = "unresolved"
    admin_notifications.put_notification(
        "mic_unresolved",
        mic_node=mic_node,
        stable_id=mic_stable_id,
        detail=(
            "Configured mic is not currently connected. The binding will be "
            "restored automatically when the device reappears."
        ),
    )
    logger.warning(
        "audio_routing: mic stable_id=%s not present in current enumeration (last node=%s)",
        mic_stable_id,
        mic_node,
    )
    return {
        "ok": False,
        "error_message": "mic device not currently connected",
        "status": "unresolved",
    }


def _capture_failed_path(*, detail: str, admin_notifications: Any = None) -> dict[str, Any]:
    """Build the path-E result + notification. Shared by the two callers
    in :func:`_reconcile_mic_with_stable_id` so the failure-state shape
    cannot drift. ``admin_notifications`` is passed in to keep the
    caller-side imports tight; falls back to a local import for unit
    tests that exercise this helper directly."""
    from meeting_scribe.runtime import state as _state

    if admin_notifications is None:
        from meeting_scribe.server_support import admin_notifications as _an

        admin_notifications = _an

    _state.audio_route_status = "capture_failed"
    admin_notifications.put_notification(
        "mic_capture_failed",
        detail=detail,
    )
    return {
        "ok": False,
        "error_message": f"mic capture failed to start: {detail}",
        "status": "capture_failed",
        "detail": detail,
    }


async def reconcile_audio_routing() -> dict[str, Any]:
    """Apply the persisted routing config to the live audio subsystem.

    Combines :func:`reconcile_server_mic` + the local-sink listener
    reconcile + interpretation buffer wiring into one helper. Returns
    a structured result so callers can decide whether to surface the
    failure as a 500/503 (route handler) or a soft warning (meeting
    start when no explicit ``audio_config`` was provided).

    Result shape::

        {
            "ok": bool,
            "mic": {"ok": bool, "error_message": str | None},
            "sink": {"ok": bool, "error_message": str | None},
        }
    """
    from meeting_scribe.audio.interpretation_buffer import InterpretationBuffer
    from meeting_scribe.audio.local_sink import (
        ensure_local_sink_listener_registered,
    )
    from meeting_scribe.audio.server_mic import reconcile_server_mic
    from meeting_scribe.runtime import state as _state
    from meeting_scribe.server_support.settings_store import (
        _effective_interpretation_enabled,
        _effective_interpretation_idle_drain_ms,
        _effective_interpretation_pause_flush_ms,
        _load_settings_override,
        _save_settings_override,
    )

    settings = _load_settings_override()

    # Self-heal a colliding admin/room sink config left over from a prior
    # session. The route validator now blocks new collisions, but a
    # settings.json that pre-dates the validator (or was hand-edited)
    # would still echo on first meeting after a restart. Persist the
    # auto-clear so the UI shows the corrected state on first paint.
    admin_sink_now = (settings.get(SETTINGS_AUDIO_ADMIN_TTS_SINK_NODE) or "").strip()
    room_sink_now = (settings.get(SETTINGS_AUDIO_ROOM_TTS_SINK_NODE) or "").strip()
    admin_lang_now = (
        settings.get("admin_tts_language") or settings.get("local_sink_language") or "en"
    )
    room_lang_now = settings.get("room_tts_language") or "all"
    if admin_room_sinks_collide(
        admin_sink=admin_sink_now,
        room_sink=room_sink_now,
        admin_lang=admin_lang_now,
        room_lang=room_lang_now,
    ):
        logger.warning(
            "audio_routing: self-healing colliding admin sink %s vs room sink %s "
            "(admin_lang=%s, room_lang=%s) — clearing admin sink to prevent dual-render echo",
            admin_sink_now,
            room_sink_now,
            admin_lang_now,
            room_lang_now,
        )
        _save_settings_override({SETTINGS_AUDIO_ADMIN_TTS_SINK_NODE: ""})
        settings = _load_settings_override()

    selection = get_routing_settings(settings)

    result: dict[str, Any] = {
        "ok": True,
        "mic": {"ok": True, "error_message": None, "status": "ok"},
        "sink": {"ok": True, "error_message": None},
    }

    # ── Stable-identity mic auto-rebind ─────────────────────────
    # Five outcomes, all set ``_state.audio_route_status`` and ``result["mic"]``:
    #   path A — node still present, identity matches → reconcile + clear failure state
    #   path B — node missing, resolver returns Resolved → rebind + emit notice
    #   path C — resolver returns Ambiguous → fail closed, surface candidates
    #   path D — resolver returns NotFound → keep persisted, treat as transient
    #   path E — routing resolved but capture process refused to start
    mic_result = await _reconcile_mic_with_stable_id(
        mic_node=selection["mic_node"],
        mic_active=selection["mic_active"],
        mic_stable_id=selection["mic_stable_id"],
        mic_discriminator=selection["mic_discriminator"],
        reconcile_server_mic_fn=reconcile_server_mic,
    )
    result["mic"] = mic_result
    if not mic_result["ok"]:
        result["ok"] = False

    try:
        interpretation_enabled = _effective_interpretation_enabled()
        if interpretation_enabled and _state.interpretation_buffer is None:
            _state.interpretation_buffer = InterpretationBuffer(
                pause_flush_ms=_effective_interpretation_pause_flush_ms(),
                idle_drain_ms=_effective_interpretation_idle_drain_ms(),
            )
        elif not interpretation_enabled and _state.interpretation_buffer is not None:
            await _state.interpretation_buffer.set_enabled(False)
            _state.interpretation_buffer = None

        ensure_local_sink_listener_registered()
    except Exception:
        logger.exception("audio_routing: local-sink reconcile failed")
        result["ok"] = False
        result["sink"] = {
            "ok": False,
            "error_message": (
                "Could not switch the speaker routing — check device availability and retry."
            ),
        }

    return result
