"""Admin Audio routing API — pairs with admin-audio-card.js.

Endpoints:

* ``GET  /api/admin/audio/devices`` — live ``pw-dump`` enumeration of
  all audio sinks + sources, plus the currently-persisted routing
  selection (``mic_node`` / ``sink_node`` / ``mic_active``).
* ``POST /api/admin/audio/route``   — accepts any subset of
  ``{mic_node, sink_node, mic_active}``; persists to the settings
  override file, then reconciles live state:

    * ``ServerMicCapture`` is started, retargeted, or stopped to match
      ``(mic_active, mic_node)``.
    * The :class:`LocalSinkListener` is retargeted to the new
      ``sink_node`` without a full re-registration so an operator
      switching from BT to USB doesn't drop any in-flight TTS.
* ``GET/POST /api/admin/audio/interpretation`` — room-speaker
  interpretation enablement, pause tuning, targeted mute controls, and
  per-transport listener counts.

Cookie-gated by ``_require_admin_response``; CSRF-protected by the
Origin allowlist middleware. Cache-Control: no-store applies via the
existing ``/api/admin/`` middleware.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from meeting_scribe.audio.audio_routing import (
    SETTINGS_AUDIO_ADMIN_TTS_SINK_NODE,
    SETTINGS_AUDIO_MEETING_MIC_ACTIVE,
    SETTINGS_AUDIO_MEETING_MIC_NODE,
    SETTINGS_AUDIO_ROOM_TTS_SINK_NODE,
    admin_room_sinks_collide,
    audio_nodes_share_physical_device,
    enumerate_audio_devices,
    get_routing_settings,
    set_device_volume,
)
from meeting_scribe.runtime import state
from meeting_scribe.server_support.admin_guard import _require_admin_response
from meeting_scribe.server_support.settings_store import (
    _effective_interpretation_enabled,
    _effective_interpretation_idle_drain_ms,
    _effective_interpretation_pause_flush_ms,
    _load_settings_override,
    _save_settings_override,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _bad_request(detail: str, code: int = 400) -> JSONResponse:
    return JSONResponse({"error": detail}, status_code=code)


# Per-operation operator-safe failure text. Full exception (including
# traceback) goes to the structured log; the HTTP response carries the
# stable operation code + a short message so the UI can render a useful
# error without leaking implementation detail.
_ROUTE_ERROR_MESSAGES: dict[str, str] = {
    "reconcile_server_mic": (
        "Could not switch the microphone routing — check device availability and retry."
    ),
    "ensure_local_sink_listener": (
        "Could not switch the speaker routing — check device availability and retry."
    ),
    "reconcile_audio_routing": (
        "Could not apply the audio routing change — check device availability and retry."
    ),
    "interpretation_status": "Could not refresh interpretation status.",
    "apply_interpretation_mute": "Could not change the mute state.",
    "apply_mic_mute": "Could not change the mic mute state.",
    "apply_device_volume": (
        "Could not adjust the device volume — wpctl rejected the change. "
        "Confirm the device is still attached."
    ),
}


def _route_failure(operation: str) -> JSONResponse:
    """Log + return a 500 JSON envelope without leaking exception text."""
    logger.exception("admin_audio %s failed", operation)
    return JSONResponse(
        {
            "error": operation,
            "message": _ROUTE_ERROR_MESSAGES.get(
                operation, "Audio operation failed — check server logs."
            ),
        },
        status_code=500,
    )


def _audio_transport_counts() -> dict[str, dict[str, int]]:
    counts = {
        "room_sink": {"total": 0, "active": 0, "muted": 0},
        "web_browser": {"total": 0, "active": 0, "muted": 0},
        "admin_monitor": {"total": 0, "active": 0, "muted": 0},
        "bt_headset": {"total": 0, "active": 0, "muted": 0},
    }
    for pref in state._audio_out_prefs.values():
        transport = getattr(pref, "transport", "web_browser")
        if transport not in counts:
            counts[transport] = {"total": 0, "active": 0, "muted": 0}
        counts[transport]["total"] += 1
        if getattr(pref, "delivery_mode", "simultaneous") == "drop":
            counts[transport]["muted"] += 1
        else:
            counts[transport]["active"] += 1
    return counts


def _effective_settings_after_updates(
    current: dict[str, Any],
    updates: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(current)
    merged.update(updates)
    return merged


def _validate_room_tts_route(
    *,
    body: dict[str, Any],
    current: dict[str, Any],
    updates: dict[str, Any],
) -> JSONResponse | None:
    """Keep room speaker TTS physically paired with server-side capture.

    The Poly can cancel its own speaker output from its microphone path, but a
    laptop/browser mic cannot. Requiring the same physical source and sink keeps
    room TTS from feeding back into an unrelated capture device.
    """
    effective = _effective_settings_after_updates(current, updates)
    room_sink = str(effective.get(SETTINGS_AUDIO_ROOM_TTS_SINK_NODE) or "").strip()
    mic_node = str(effective.get(SETTINGS_AUDIO_MEETING_MIC_NODE) or "").strip()
    mic_active = bool(effective.get(SETTINGS_AUDIO_MEETING_MIC_ACTIVE, False))

    if not mic_active or not mic_node:
        if body.get("room_sink_node"):
            return _bad_request("Room TTS requires an active matching server mic.")
        if room_sink:
            updates[SETTINGS_AUDIO_ROOM_TTS_SINK_NODE] = ""
        return None

    if room_sink and not audio_nodes_share_physical_device(mic_node, room_sink):
        if "room_sink_node" in body:
            return _bad_request(
                "Room TTS output must use the same physical device as the server mic."
            )
        updates[SETTINGS_AUDIO_ROOM_TTS_SINK_NODE] = ""
    return None


def _validate_admin_room_collision(
    *,
    body: dict[str, Any],
    current: dict[str, Any],
    updates: dict[str, Any],
) -> JSONResponse | None:
    """Reject (or auto-clear) configurations that would render the same
    language to one physical speaker via two parallel TTS pipelines.

    Background: when ``audio_admin_tts_sink_node`` and
    ``audio_room_tts_sink_node`` resolve to the same Poly speaker AND
    the room TTS language covers the admin's target language ("all" or
    equal), both render paths emit the same audio at slight offsets,
    producing the dual-render echo Brad hit on J→E during a 2026-05-11
    live meeting. Admin output via the same shared speaker is always
    redundant in that overlap — room sink already covers it.

    Resolution rule:

    * If the operator explicitly set ``admin_sink_node`` in this request
      (and it would create the collision), reject with a 400 so the
      conflict surfaces in the UI rather than getting silently dropped.
    * Otherwise the collision was caused by changing the room sink (or
      room language) underneath an existing admin sink — auto-clear
      ``audio_admin_tts_sink_node`` in the same patch so the user
      doesn't have to chase it.
    """
    effective = _effective_settings_after_updates(current, updates)
    admin_sink = str(effective.get(SETTINGS_AUDIO_ADMIN_TTS_SINK_NODE) or "").strip()
    room_sink = str(effective.get(SETTINGS_AUDIO_ROOM_TTS_SINK_NODE) or "").strip()
    admin_lang = effective.get("admin_tts_language") or effective.get("local_sink_language") or "en"
    room_lang = effective.get("room_tts_language") or "all"

    if not admin_room_sinks_collide(
        admin_sink=admin_sink,
        room_sink=room_sink,
        admin_lang=admin_lang,
        room_lang=room_lang,
    ):
        return None

    if "admin_sink_node" in body:
        return _bad_request(
            "Admin TTS output cannot share the room speaker — they would render the "
            "same language twice and echo. Pick a different physical device "
            "(e.g. a headset), or leave Default sink."
        )
    # Caller changed the room sink / language out from under the admin
    # sink; silently clear the redundant admin route.
    logger.info(
        "admin_audio: auto-clearing admin TTS sink (%s) — collides with room sink (%s) "
        "after caller changed %s",
        admin_sink,
        room_sink,
        sorted(body.keys()),
    )
    updates[SETTINGS_AUDIO_ADMIN_TTS_SINK_NODE] = ""
    return None


def _reconcile_local_sink_or_log() -> bool:
    """Reconcile the local sink listeners; return False on failure.

    The companion to :func:`_interpretation_payload` for callers that
    need a hard signal (route handlers that should return 500 on
    failure). Broadcast paths that call ``_interpretation_payload``
    directly keep swallowing the failure — there's nowhere to surface
    an error on a websocket fan-out.
    """
    try:
        from meeting_scribe.audio.local_sink import ensure_local_sink_listener_registered

        ensure_local_sink_listener_registered()
        return True
    except Exception:
        logger.exception("local-sink reconcile failed")
        return False


def _interpretation_payload() -> dict[str, Any]:
    """Build the interpretation status snapshot.

    Pure payload assembly — no reconcile side-effect. Route handlers
    that mutate routing call :func:`_reconcile_local_sink_or_log` and
    then this helper; broadcast paths emit the snapshot as-is.
    """
    from meeting_scribe.languages import LANGUAGE_REGISTRY
    from meeting_scribe.server_support.settings_store import (
        _effective_tts_voice_mode,
    )

    settings = _load_settings_override()
    language_options = [
        {"code": code, "name": lang.name}
        for code, lang in LANGUAGE_REGISTRY.items()
        if lang.tts_native
    ]
    return {
        "enabled": _effective_interpretation_enabled(),
        "pause_flush_ms": _effective_interpretation_pause_flush_ms(),
        "idle_drain_ms": _effective_interpretation_idle_drain_ms(),
        "admin_tts_language": settings.get(
            "admin_tts_language", settings.get("local_sink_language", "en")
        ),
        "room_tts_language": settings.get("room_tts_language", "all"),
        "local_sink_language": settings.get(
            "admin_tts_language", settings.get("local_sink_language", "en")
        ),
        "local_sink_language_options": language_options,
        # Voice mode is the same value /api/admin/settings exposes; included
        # here so the in-meeting + setup-screen audio cards can render and
        # update it without a second fetch. Listener WS handshakes still
        # read this value at connect time via _effective_tts_voice_mode().
        "tts_voice_mode": _effective_tts_voice_mode(),
        "tts_voice_mode_options": [
            {"code": "studio", "name": "Studio (built-in voices)"},
            {"code": "cloned", "name": "Participant (cloned voices)"},
        ],
        "listener_counts": _audio_transport_counts(),
        # Soft input mute is meeting-scoped runtime state — never read
        # from settings.json. Surfaced here so the UI's "Mute mic"
        # button has a single status payload to subscribe to.
        "mic_muted": bool(state.mic_input_muted),
        "room_sink_mode": next(
            (
                getattr(pref, "delivery_mode", "simultaneous")
                for pref in state._audio_out_prefs.values()
                if getattr(pref, "transport", "") == "room_sink"
            ),
            "unregistered",
        ),
    }


async def _apply_interpretation_live(enabled: bool) -> None:
    from meeting_scribe.audio.interpretation_buffer import InterpretationBuffer
    from meeting_scribe.audio.local_sink import (
        ensure_local_sink_listener_registered,
    )

    if enabled and state.interpretation_buffer is None:
        state.interpretation_buffer = InterpretationBuffer(
            pause_flush_ms=_effective_interpretation_pause_flush_ms(),
            idle_drain_ms=_effective_interpretation_idle_drain_ms(),
        )
    elif not enabled and state.interpretation_buffer is not None:
        await state.interpretation_buffer.set_enabled(False)
        state.interpretation_buffer = None

    ensure_local_sink_listener_registered()


async def _apply_interpretation_mute(mute: str) -> None:
    target_transports: set[str]
    unmute = mute.startswith("unmute_")
    if mute.endswith("room_speaker"):
        target_transports = {"room_sink"}
    elif mute.endswith("web"):
        target_transports = {"web_browser", "admin_monitor"}
    else:
        target_transports = {"bt_headset"}

    for pref in state._audio_out_prefs.values():
        if getattr(pref, "transport", "web_browser") not in target_transports:
            continue
        if unmute:
            if getattr(pref, "transport", "") == "room_sink":
                pref.delivery_mode = (
                    "consecutive" if _effective_interpretation_enabled() else "simultaneous"
                )
            else:
                pref.delivery_mode = "simultaneous"
        else:
            pref.delivery_mode = "drop"
    if "room_sink" in target_transports and state.interpretation_buffer is not None:
        await state.interpretation_buffer.cancel_all(clear=True)


@router.get("/api/admin/audio/devices")
async def audio_devices_endpoint(request: Request) -> JSONResponse:
    """Enumerate audio sinks + sources for the routing UI.

    Returns the live device list plus the persisted selection so the
    UI can render the dropdowns + select the operator's last choice
    in one round-trip.
    """
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    devices = await enumerate_audio_devices()
    selection = get_routing_settings(_load_settings_override())
    selection["server_mic_active_live"] = bool(state.server_mic_active)
    return JSONResponse({"devices": devices, "selection": selection})


@router.get("/api/admin/audio/route")
async def audio_route_get(request: Request) -> JSONResponse:
    """Return just the persisted selection (no device enumeration)."""
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    selection = get_routing_settings(_load_settings_override())
    selection["server_mic_active_live"] = bool(state.server_mic_active)
    return JSONResponse(selection)


@router.get("/api/admin/audio/interpretation")
async def audio_interpretation_get(request: Request) -> JSONResponse:
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    if not _reconcile_local_sink_or_log():
        return _route_failure("ensure_local_sink_listener")
    return JSONResponse(_interpretation_payload())


@router.post("/api/admin/audio/interpretation")
async def audio_interpretation_post(request: Request) -> JSONResponse:
    deny = _require_admin_response(request)
    if deny is not None:
        return deny

    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        return _bad_request("JSON object expected")

    updates: dict[str, Any] = {}
    if "enabled" in body:
        enabled = body.get("enabled")
        if not isinstance(enabled, bool):
            return _bad_request("enabled must be a boolean")
        updates["interpretation_enabled"] = enabled
        # When the caller re-enables interpretation WITHOUT specifying a
        # direction, fall back to the persisted "last active direction"
        # so the second long-press of the speakerphone Phone button
        # restores the user's most recent choice. The speakerphone daemon
        # relies on this contract for the Phone-long-press toggle.
        if enabled and "room_tts_language" not in body:
            from meeting_scribe.server_support.settings_store import (
                _effective_interpretation_last_room_tts_language,
            )

            last = _effective_interpretation_last_room_tts_language()
            updates["room_tts_language"] = last
    if "pause_flush_ms" in body:
        pause_ms = body.get("pause_flush_ms")
        if not isinstance(pause_ms, int) or pause_ms < 100 or pause_ms > 60000:
            return _bad_request("pause_flush_ms must be an integer between 100 and 60000")
        updates["interpretation_pause_flush_ms"] = pause_ms
    if "idle_drain_ms" in body:
        idle_ms = body.get("idle_drain_ms")
        if not isinstance(idle_ms, int) or idle_ms < 100 or idle_ms > 60000:
            return _bad_request("idle_drain_ms must be an integer between 100 and 60000")
        updates["interpretation_idle_drain_ms"] = idle_ms
    if "tts_voice_mode" in body:
        raw_mode = body.get("tts_voice_mode")
        if raw_mode not in ("studio", "cloned"):
            return _bad_request("tts_voice_mode must be 'studio' or 'cloned'")
        updates["tts_voice_mode"] = raw_mode

    for lang_key in ("local_sink_language", "admin_tts_language", "room_tts_language"):
        if lang_key not in body:
            continue
        from meeting_scribe.languages import LANGUAGE_REGISTRY

        raw_lang = body.get(lang_key)
        allowed = {code for code, lang in LANGUAGE_REGISTRY.items() if lang.tts_native}
        if lang_key == "room_tts_language":
            allowed = allowed | {"all"}
        if not isinstance(raw_lang, str) or raw_lang not in allowed:
            return _bad_request(f"{lang_key} must be a TTS-supported language code")
        updates["admin_tts_language" if lang_key == "local_sink_language" else lang_key] = raw_lang
        # Persist room direction changes as the "last active direction"
        # so a subsequent re-enable (with no explicit direction) restores
        # the user's choice. Speakerphone Phone-long-press relies on this.
        # GUI direction changes and daemon direction changes both flow
        # through here, so the persisted value is always in lock-step
        # with the currently-applied state — single writer guarantee.
        if lang_key == "room_tts_language":
            updates["interpretation_last_room_tts_language"] = raw_lang

    mute = body.get("mute")
    if mute is not None and mute not in (
        "mute_room_speaker",
        "mute_web",
        "mute_bt_headsets",
        "unmute_room_speaker",
        "unmute_web",
        "unmute_bt_headsets",
    ):
        return _bad_request("mute must be a known mute/unmute command")

    if not updates and mute is None:
        return _bad_request("no recognized interpretation controls in body")

    if updates:
        # A change to ``room_tts_language`` (or ``admin_tts_language``)
        # can transition the routing into the dual-render echo state
        # (admin sink == room sink, room lang covers admin lang). Run
        # the same collision validator the route POST uses so the bad
        # config never gets persisted via this endpoint either.
        current_settings = _load_settings_override()
        collision_error = _validate_admin_room_collision(
            body=body, current=current_settings, updates=updates
        )
        if collision_error is not None:
            return collision_error
        _save_settings_override(updates)
        enabled = _effective_interpretation_enabled()
        await _apply_interpretation_live(enabled)
        if state.interpretation_buffer is not None:
            state.interpretation_buffer.pause_flush_ms = _effective_interpretation_pause_flush_ms()
            state.interpretation_buffer.idle_drain_ms = _effective_interpretation_idle_drain_ms()

    if mute is not None:
        await _apply_interpretation_mute(mute)

    return JSONResponse(_interpretation_payload())


@router.post("/api/admin/audio/mic")
async def audio_mic_mute_post(request: Request) -> JSONResponse:
    """Toggle the soft input-mute.

    Body: ``{"muted": bool}``. Mutes the captured microphone at the
    input boundary — true privacy pause, no audio reaches the PCM file
    and no transcript is generated for the muted span. The state is
    runtime-only (not persisted, reset on meeting start) so a stale
    operator toggle cannot quietly suppress a future meeting.
    """
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    try:
        body = await request.json()
    except Exception:
        return _bad_request("invalid JSON")
    if not isinstance(body, dict) or "muted" not in body:
        return _bad_request("body must include muted (bool)")
    muted = body.get("muted")
    if not isinstance(muted, bool):
        return _bad_request("muted must be a boolean")

    try:
        state.mic_input_muted = bool(muted)
    except Exception:
        return _route_failure("apply_mic_mute")

    logger.info("admin_audio: mic_input_muted=%s", state.mic_input_muted)
    return JSONResponse(_interpretation_payload())


@router.post("/api/admin/audio/route")
async def audio_route_post(request: Request) -> JSONResponse:
    """Update the routing selection.

    Body accepts any subset of
    ``{mic_node, admin_sink_node, room_sink_node, mic_active}``
    so the UI can patch one field at a time. Unspecified fields are
    left untouched. Strings are normalized: empty string clears.
    """
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    body: Any
    try:
        body = await request.json()
    except Exception:
        return _bad_request("invalid JSON")
    if not isinstance(body, dict):
        return _bad_request("body must be an object")

    updates: dict[str, Any] = {}
    if "mic_node" in body:
        v = body.get("mic_node")
        if v is not None and not isinstance(v, str):
            return _bad_request("mic_node must be string or null")
        updates[SETTINGS_AUDIO_MEETING_MIC_NODE] = (v or "").strip()
    if "admin_sink_node" in body:
        v = body.get("admin_sink_node")
        if v is not None and not isinstance(v, str):
            return _bad_request("admin_sink_node must be string or null")
        updates[SETTINGS_AUDIO_ADMIN_TTS_SINK_NODE] = (v or "").strip()
    if "room_sink_node" in body:
        v = body.get("room_sink_node")
        if v is not None and not isinstance(v, str):
            return _bad_request("room_sink_node must be string or null")
        updates[SETTINGS_AUDIO_ROOM_TTS_SINK_NODE] = (v or "").strip()
    if "mic_active" in body:
        v = body.get("mic_active")
        if not isinstance(v, bool):
            return _bad_request("mic_active must be boolean")
        updates[SETTINGS_AUDIO_MEETING_MIC_ACTIVE] = v

    if not updates:
        return _bad_request("no fields provided")

    current_settings = _load_settings_override()
    route_error = _validate_room_tts_route(body=body, current=current_settings, updates=updates)
    if route_error is not None:
        return route_error
    collision_error = _validate_admin_room_collision(
        body=body, current=current_settings, updates=updates
    )
    if collision_error is not None:
        return collision_error

    # Derive (stable_id, discriminator) for the new mic_node when the
    # operator supplied one. Phase-1 auto-rebind needs this pair so a
    # subsequent USB reconnect can be resolved without operator help.
    # Done BEFORE the save so they land in the same settings transaction.
    if "mic_node" in body and updates.get(SETTINGS_AUDIO_MEETING_MIC_NODE):
        from meeting_scribe.audio.audio_routing import (
            SETTINGS_AUDIO_MEETING_MIC_DISCRIMINATOR,
            SETTINGS_AUDIO_MEETING_MIC_STABLE_ID,
            derive_stable_identity_for_node,
        )

        new_node = updates[SETTINGS_AUDIO_MEETING_MIC_NODE]
        stable_id, discriminator = await derive_stable_identity_for_node(new_node)
        updates[SETTINGS_AUDIO_MEETING_MIC_STABLE_ID] = stable_id
        updates[SETTINGS_AUDIO_MEETING_MIC_DISCRIMINATOR] = discriminator
        if not stable_id:
            logger.warning(
                "audio_route_post: no stable_id extractable for mic_node=%s — "
                "auto-rebind disabled until device re-saved while connected",
                new_node,
            )
    elif "mic_node" in body and updates.get(SETTINGS_AUDIO_MEETING_MIC_NODE) == "":
        # Mic explicitly cleared — drop the identity too so a future
        # bind starts from a clean slate.
        from meeting_scribe.audio.audio_routing import (
            SETTINGS_AUDIO_MEETING_MIC_DISCRIMINATOR,
            SETTINGS_AUDIO_MEETING_MIC_STABLE_ID,
        )

        updates[SETTINGS_AUDIO_MEETING_MIC_STABLE_ID] = ""
        updates[SETTINGS_AUDIO_MEETING_MIC_DISCRIMINATOR] = None

    _save_settings_override(updates)
    settings = _load_settings_override()
    selection = get_routing_settings(settings)

    # Apply mic + sink side via the shared self-heal helper. This wires
    # in the stable-id auto-rebind, surfaces failure-path notifications,
    # and returns a ReconcileOutcome the SPA can render directly.
    from meeting_scribe.audio.audio_routing import reconcile_audio_routing

    try:
        outcome = await reconcile_audio_routing()
    except Exception:
        return _route_failure("reconcile_audio_routing")

    selection["server_mic_active_live"] = bool(state.server_mic_active)
    selection["reconcile_outcome"] = outcome
    return JSONResponse(selection)


# Volume / mute is a different beast from routing — it's per-PipeWire-node
# state owned by WirePlumber, not a meeting-scribe setting. We don't
# persist it: the operator's intent is "make the Poly louder right
# now", and the value survives reboots because WirePlumber writes its
# own state file. Persisting in settings.json too would just create a
# stale shadow.
_VOLUME_MAX = 1.5  # matches wpctl's default ceiling (150% / +3.5 dB)


@router.post("/api/admin/audio/volume")
async def audio_volume_post(request: Request) -> JSONResponse:
    """Adjust per-device volume + mute.

    Body: ``{"node_name": str, "volume"?: float in [0, 1.5], "muted"?: bool}``.
    The operator's UI knows the persisted ``node.name`` strings (stable
    across PipeWire restarts), so we resolve them here against the
    live pw-dump enumeration and call wpctl with the numeric node id.
    """
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    try:
        body = await request.json()
    except Exception:
        return _bad_request("invalid JSON")
    if not isinstance(body, dict):
        return _bad_request("body must be an object")

    node_name = body.get("node_name")
    if not isinstance(node_name, str) or not node_name.strip():
        return _bad_request("node_name (string) required")
    node_name = node_name.strip()

    volume = body.get("volume")
    if volume is not None:
        if isinstance(volume, bool) or not isinstance(volume, (int, float)):
            return _bad_request("volume must be a number")
        volume = float(volume)
        if volume < 0.0 or volume > _VOLUME_MAX:
            return _bad_request(f"volume must be between 0 and {_VOLUME_MAX}")

    muted = body.get("muted")
    if muted is not None and not isinstance(muted, bool):
        return _bad_request("muted must be boolean")

    if volume is None and muted is None:
        return _bad_request("provide volume and/or muted")

    devices = await enumerate_audio_devices()
    match = next(
        (
            d
            for d in (devices.get("sources", []) + devices.get("sinks", []))
            if d.get("node_name") == node_name
        ),
        None,
    )
    if match is None or match.get("node_id") is None:
        return _bad_request(f"unknown node_name: {node_name}", code=404)

    try:
        ok = await set_device_volume(match["node_id"], volume=volume, muted=muted)
    except Exception:
        return _route_failure("apply_device_volume")
    if not ok:
        return _route_failure("apply_device_volume")

    refreshed = await enumerate_audio_devices()
    refreshed_match = next(
        (
            d
            for d in (refreshed.get("sources", []) + refreshed.get("sinks", []))
            if d.get("node_name") == node_name
        ),
        match,
    )
    return JSONResponse(
        {
            "node_name": node_name,
            "volume": refreshed_match.get("volume"),
            "muted": refreshed_match.get("muted"),
        }
    )


# ── Admin notifications ────────────────────────────────────────
#
# Notifications surfaced via the ``.meeting-banner`` SPA component.
# Producers (mostly ``audio_routing.reconcile_audio_routing``) write rows
# into ``state.pending_admin_notifications`` via
# ``server_support.admin_notifications.put_notification`` and the active
# rows are pulled into ``/api/status``. The dismiss endpoint below flips
# the operator-clicked row's ``dismissed_at`` so subsequent polls hide it
# (the row stays in storage so reloading the SPA doesn't resurrect it).


_KNOWN_NOTIFICATION_KINDS: set[str] = {
    "mic_rebound",
    "mic_ambiguous",
    "mic_unresolved",
    "mic_capture_failed",
}


@router.post("/api/admin/notifications/{kind}/dismiss")
async def admin_notification_dismiss(kind: str, request: Request) -> JSONResponse:
    """Mark an admin notification dismissed.

    The kind is the same key the producer used (``mic_rebound`` etc.).
    Unknown kinds return 404 to keep the surface tight — a typo in the
    SPA shouldn't be able to write arbitrary keys.
    """
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    if kind not in _KNOWN_NOTIFICATION_KINDS:
        return _bad_request(f"unknown notification kind: {kind}", code=404)

    from meeting_scribe.server_support import admin_notifications

    dismissed = admin_notifications.dismiss_if_present(kind)
    return JSONResponse({"kind": kind, "dismissed": dismissed})
