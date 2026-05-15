"""Internal speakerphone endpoints — daemon-facing, UDS-only.

These routes are mounted **only** on the UDS ASGI app
(``speakerphone/uds.py``). They are not reachable from the public TCP
listener. Authentication = filesystem permissions on the UDS socket
(0600, user-owned).

The daemon calls these endpoints to advance interpretation, toggle the
mic, toggle the meeting record state, and report button presses. The
GUI never touches this namespace.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from meeting_scribe.speakerphone import api as sp_api

logger = logging.getLogger(__name__)

router = APIRouter()


def _bad_request(msg: str) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=400)


@router.get("/api/internal/speakerphone/state")
async def state_get() -> JSONResponse:
    return JSONResponse(sp_api.build_state_payload())


@router.post("/api/internal/speakerphone/interpretation")
async def interpretation_post(request: Request) -> JSONResponse:
    """Apply an interpretation update.

    Accepts ``{enabled?, room_tts_language?}``. The server side updates
    ``interpretation_last_room_tts_language`` as a side effect on
    direction changes. Re-enable without a direction picks up the
    persisted last direction automatically.
    """
    try:
        body = await request.json()
    except Exception:
        return _bad_request("JSON object expected")
    if not isinstance(body, dict):
        return _bad_request("JSON object expected")

    enabled = body.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        return _bad_request("enabled must be a boolean")
    room = body.get("room_tts_language")
    if room is not None and not isinstance(room, str):
        return _bad_request("room_tts_language must be a string")

    payload = await sp_api.apply_interpretation(
        enabled=enabled,
        room_tts_language=room,
    )
    return JSONResponse(payload)


@router.post("/api/internal/speakerphone/mic-mute")
async def mic_mute_post() -> JSONResponse:
    """Toggle the soft mic mute. Body is ignored."""
    payload = await sp_api.apply_mic_mute_toggle()
    return JSONResponse(payload)


@router.post("/api/internal/speakerphone/meeting-toggle")
async def meeting_toggle_post() -> JSONResponse:
    """Start a meeting (default profile) or stop the active one atomically.

    Server decides start vs. stop based on ``state.current_meeting``.
    On start, the default profile in the sidecar mapping is applied
    *before* the meeting create. If start fails, the prior settings
    snapshot is restored so a half-applied profile is never visible
    to the next press.
    """
    payload = await sp_api.apply_meeting_record_toggle()
    return JSONResponse(payload)


@router.post("/api/internal/speakerphone/press")
async def press_post(request: Request) -> JSONResponse:
    """Record a button-press event for the GUI to surface."""
    try:
        body = await request.json()
    except Exception:
        return _bad_request("JSON object expected")
    if not isinstance(body, dict):
        return _bad_request("JSON object expected")
    device_key = body.get("device_key")
    button = body.get("button")
    press_kind = body.get("press_kind")
    if not all(isinstance(x, str) for x in (device_key, button, press_kind)):
        return _bad_request("device_key, button, press_kind must be strings")
    sp_api.record_press(device_key, button, press_kind)
    return JSONResponse({"ok": True})


@router.post("/api/internal/speakerphone/speak")
async def speak_post(request: Request) -> JSONResponse:
    """Synthesize + play a button-feedback label through the room sink.

    Body: ``{label_id: str, language?: str, overrides?: dict}``.
    Server re-reads the mapping fresh on every call so language /
    ``enabled`` changes apply on the very next press — there is no
    daemon-side cache of feedback config. ``respect_enabled=True`` so
    a disabled mapping skips synthesis.

    Returns ``{ok: True, ...}`` on success or skip; 400 on unknown
    label_id.
    """
    try:
        body = await request.json()
    except Exception:
        return _bad_request("JSON object expected")
    if not isinstance(body, dict):
        return _bad_request("JSON object expected")

    label_id = body.get("label_id")
    if not isinstance(label_id, str) or not label_id:
        return _bad_request("label_id (string) is required")
    language = body.get("language")
    if language is not None and not isinstance(language, str):
        return _bad_request("language must be a string")
    overrides = body.get("overrides")
    if overrides is not None and not isinstance(overrides, dict):
        return _bad_request("overrides must be an object")

    try:
        payload = await sp_api.apply_speak(
            label_id=label_id,
            language=language,
            overrides_inline=overrides,
            respect_enabled=True,
        )
    except sp_api.FeedbackError as exc:
        return JSONResponse(
            {"error": str(exc), "label_id": label_id},
            status_code=400,
        )
    return JSONResponse(payload)
