"""``/api/ws/audio-out`` — guest-scope listener WebSocket.

Sends synthesized TTS audio (and optional pass-through original
audio) to phones / iPads on the hotspot. Two wire formats:

* **wav-pcm** — one complete RIFF WAV per audio segment, no
  prefix byte. Backward-compatible with legacy clients that don't
  negotiate.
* **mse-fmp4-aac** — one fMP4 init frame prefixed with ``b'\\x49'``
  ('I'), then media fragments prefixed with ``b'\\x46'`` ('F').

Format negotiation runs during a 1-second grace window after WS
accept. Audio that arrives during the grace window is buffered
per-listener (capped at 1 second of audio) and flushed once the
format is known. Legacy clients that never send ``set_format`` get
wav-pcm via the grace-window default — no URL versioning required
across deploys.

Hard cap of ``_MAX_AUDIO_OUT_CLIENTS=32`` prevents accidental
tab-spam or a misbehaving client reopening the WS in a loop from
overwhelming the server.
"""

from __future__ import annotations

import json as _json
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from meeting_scribe.runtime import state
from meeting_scribe.server_support.peer import _peer_str
from meeting_scribe.server_support.request_scope import _is_guest_scope, _is_hotspot_client
from meeting_scribe.server_support.sessions import ClientSession
from meeting_scribe.server_support.settings_store import _effective_tts_voice_mode
from meeting_scribe.server_support.translation_demand import _norm_lang

logger = logging.getLogger(__name__)

router = APIRouter()


# Hotspot listener hard cap. Hotspot guard already restricts this
# endpoint to the meeting subnet, but this is belt-and-braces against
# accidental tab-spam or a misbehaving client reopening the WS in a
# loop.
_MAX_AUDIO_OUT_CLIENTS = 32
# Grace window after WS accept during which a client may negotiate
# `audio_format` via `{"type": "set_format", ...}`. If the grace
# expires with no negotiation, we default to "wav-pcm" (backward
# compat with legacy clients). Audio deliveries that arrive during
# the grace window are held in `pref.pending_audio` (capped) and
# flushed once the format is known.
_AUDIO_FORMAT_GRACE_S = 1.0
# Max buffered audio seconds while waiting for `set_format`. Per-
# listener bound so a stuck handshake can't balloon memory.
_AUDIO_FORMAT_PENDING_CAP_S = 1.0
_VALID_AUDIO_FORMATS = frozenset({"wav-pcm", "mse-fmp4-aac"})


def _create_audio_out_session(ws: WebSocket) -> ClientSession:
    """Create a ClientSession for an audio-out listener and register it.

    Pure helper — always succeeds.  Does NOT check capacity or scope
    (those guards live in the handler).  Does NOT touch
    ``state._audio_out_clients`` — slot reservation is the handler's job.
    """
    pref = ClientSession(
        send_audio=True,
        voice_mode=_effective_tts_voice_mode(),
        grace_deadline=time.monotonic() + _AUDIO_FORMAT_GRACE_S,
    )
    state._audio_out_prefs[ws] = pref
    return pref


def _unregister_audio_out_client(ws: WebSocket) -> None:
    """Remove an audio-out listener from globals and close its encoder.

    Idempotent — safe to call multiple times for the same ``ws``.
    """
    pref = state._audio_out_prefs.pop(ws, None)
    state._audio_out_clients.discard(ws)
    if pref is not None and pref.mse_encoder is not None:
        try:
            pref.mse_encoder.close()
        except Exception as e:
            logger.debug("mse_encoder close on disconnect: %r", e)
        pref.mse_encoder = None


async def _flush_pending_audio(websocket: WebSocket, pref: ClientSession) -> None:
    """Drain the format-negotiation grace-window buffer for one listener."""
    from meeting_scribe.audio.output_pipeline import _deliver_audio_to_listener

    pending = pref.pending_audio
    pref.pending_audio = []
    if not pending:
        return
    for audio, source_rate in pending:
        ok = await _deliver_audio_to_listener(websocket, pref, audio, source_rate, None)
        if not ok:
            break


async def _handle_audio_out_message(ws: WebSocket, pref: ClientSession, text: str) -> str | None:
    """Parse and apply one JSON control message from an audio-out client.

    Returns the ``format_ack`` JSON string to send back when the message
    is a valid ``set_format``, or ``None`` otherwise.
    """
    try:
        msg = _json.loads(text)
    except Exception:
        return None
    if not isinstance(msg, dict):
        return None

    msg_type = msg.get("type")
    if msg_type == "set_format":
        fmt = msg.get("format", "")
        if fmt not in _VALID_AUDIO_FORMATS:
            logger.warning(
                "Audio-out set_format: rejected %r from %s",
                fmt,
                _peer_str(ws),
            )
            return None
        old_format = pref.audio_format
        pref.audio_format = fmt
        logger.info("Audio-out format: %s for %s", fmt, _peer_str(ws))
        # If the client changed its mind about an existing encoder,
        # tear the old one down.
        if old_format is not None and old_format != fmt and pref.mse_encoder is not None:
            try:
                pref.mse_encoder.close()
            except Exception as e:
                logger.debug("old mse_encoder close: %r", e)
            pref.mse_encoder = None
        # Flush any audio held during the negotiation grace window.
        await _flush_pending_audio(ws, pref)
        return _json.dumps({"type": "format_ack", "format": fmt})

    if msg_type == "set_language":
        lang = _norm_lang(msg.get("language", ""))
        if lang:
            pref.preferred_language = lang
            logger.info("Audio-out WS set language preference: %s", lang)
    elif msg_type == "set_mode":
        mode = msg.get("mode", "translation")
        if mode in ("translation", "full"):
            pref.interpretation_mode = mode
            logger.info("Audio-out WS set interpretation mode: %s", mode)
    elif msg_type == "set_voice":
        voice = msg.get("voice", "studio")
        if voice in ("studio", "cloned"):
            pref.voice_mode = voice
            logger.info("Audio-out WS set voice mode: %s", voice)
    return None


@router.websocket("/api/ws/audio-out")
async def websocket_audio_out(websocket: WebSocket) -> None:
    """Audio-out WebSocket — sends synthesized TTS audio to guest-scope clients.

    Clients send JSON text messages for preference updates:
        {"type": "set_format",   "format":   "wav-pcm" | "mse-fmp4-aac"}
        {"type": "set_language", "language": "en"}
        {"type": "set_mode",     "mode":     "translation" | "full"}
        {"type": "set_voice",    "voice":    "studio" | "cloned"}

    On `set_format`, server responds with a text message:
        {"type": "format_ack", "format": "<accepted-format>"}

    Server sends binary audio frames:
        - wav-pcm listeners:   one complete RIFF WAV per audio segment,
                               no prefix byte (backward-compatible with
                               legacy clients)
        - mse-fmp4-aac listeners: one fMP4 init frame prefixed with
                                  b'\\x49' ('I'), then media fragments
                                  prefixed with b'\\x46' ('F').

    Legacy clients that never send `set_format` receive wav-pcm via the
    grace-window default — no URL versioning or protocol upgrade prompt
    required across a deploy.
    """
    # Audio-out is guest-scope ONLY. The admin interface must never play
    # interpretation audio — the operator is typically in the room with
    # the speaker, and bleeding TTS out of the admin console creates
    # instant feedback loops and confuses the meeting.
    if not _is_guest_scope(websocket):
        await websocket.close(code=1008, reason="audio-out is guest-scope only")
        return
    # Atomic cap check + slot reservation (no await between them).
    if len(state._audio_out_clients) >= _MAX_AUDIO_OUT_CLIENTS:
        logger.warning(
            "Audio-out WS refused: at capacity (%d/%d)",
            len(state._audio_out_clients),
            _MAX_AUDIO_OUT_CLIENTS,
        )
        await websocket.close(code=1013, reason="audio-out at capacity")
        return
    state._audio_out_clients.add(websocket)  # reserve slot synchronously
    try:
        await websocket.accept()
    except Exception:
        state._audio_out_clients.discard(websocket)  # release slot on accept failure
        return
    pref = _create_audio_out_session(websocket)
    client = getattr(websocket, "client", None)
    logger.info(
        "Audio-out WS connected from %s (hotspot=%s, total=%d)",
        client.host if client else "?",
        _is_hotspot_client(websocket),
        len(state._audio_out_clients),
    )

    try:
        while True:
            text = await websocket.receive_text()
            ack = await _handle_audio_out_message(websocket, pref, text)
            if ack:
                await websocket.send_text(ack)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _unregister_audio_out_client(websocket)
        logger.info("Audio-out WS disconnected (remaining=%d)", len(state._audio_out_clients))
