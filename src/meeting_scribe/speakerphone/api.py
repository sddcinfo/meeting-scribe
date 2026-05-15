"""Shared business logic for the public + internal speakerphone routes.

Both ``routes/admin_speakerphone.py`` (cookie-auth, GUI-facing) and
``routes/internal_speakerphone.py`` (UDS-only, daemon-facing) call into
these functions. Keeping the logic here means the two routers stay
mechanical and the behavior is provably the same.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from meeting_scribe.runtime import state
from meeting_scribe.server_support.settings_store import (
    _effective_interpretation_enabled,
    _effective_interpretation_last_room_tts_language,
    _load_settings_override,
    _save_settings_override,
)
from meeting_scribe.speakerphone import mapping

logger = logging.getLogger(__name__)


# In-memory snapshot of the most recent button press across all
# connected devices. Updated by the daemon via the internal "press"
# endpoint and surfaced verbatim in the public state payload so the
# GUI can flash the matching row.
_LAST_PRESS: dict[str, Any] | None = None


# ── State payload ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class _InterpretationView:
    enabled: bool
    room_tts_language: str
    admin_tts_language: str
    last_active_room_tts_language: str
    mic_muted: bool


def _read_interpretation_view() -> _InterpretationView:
    settings = _load_settings_override()
    return _InterpretationView(
        enabled=_effective_interpretation_enabled(),
        room_tts_language=str(settings.get("room_tts_language", "all")),
        admin_tts_language=str(
            settings.get("admin_tts_language", settings.get("local_sink_language", "en")),
        ),
        last_active_room_tts_language=_effective_interpretation_last_room_tts_language(),
        mic_muted=bool(getattr(state, "mic_muted", False)),
    )


def _meeting_languages() -> list[str]:
    """Resolve the active meeting's language pair.

    Reads from the persisted settings overrides. The exact key has
    drifted over time — try a few known names and fall back to the
    default profile in the mapping document so a fresh install still
    has something to cycle through.
    """
    settings = _load_settings_override()
    for key in ("meeting_languages", "participant_languages"):
        val = settings.get(key)
        if isinstance(val, list) and val:
            return [str(code) for code in val if isinstance(code, str)]
    # Fall back to the default-profile languages in the mapping doc.
    doc = mapping.load()
    profile = doc.get("default_meeting_profile", {})
    langs = profile.get("languages")
    if isinstance(langs, list):
        return [str(c) for c in langs if isinstance(c, str)]
    return []


def _recording_state() -> dict[str, Any]:
    meeting = getattr(state, "current_meeting", None)
    if meeting is None:
        return {"recording": False, "meeting_id": None}
    return {
        "recording": True,
        "meeting_id": getattr(meeting, "meeting_id", None),
    }


def build_state_payload() -> dict[str, Any]:
    """Build the response body for ``GET /…/speakerphone/state``.

    Same shape from both the public and internal endpoints (the
    daemon-only field ``interpretation_last_room_tts_language`` is
    included in the interpretation block so both surfaces share the
    schema; the GUI just doesn't render it directly).
    """
    doc = mapping.load()
    etag = mapping.compute_etag(doc)
    interp = _read_interpretation_view()
    return {
        "etag": etag,
        "mapping": doc,
        "interpretation": {
            "enabled": interp.enabled,
            "room_tts_language": interp.room_tts_language,
            "admin_tts_language": interp.admin_tts_language,
            "last_active_room_tts_language": interp.last_active_room_tts_language,
            "mic_muted": interp.mic_muted,
        },
        "meeting": _recording_state(),
        "meeting_languages": _meeting_languages(),
        "last_press": dict(_LAST_PRESS) if _LAST_PRESS else None,
        "as_of": time.time(),
    }


def record_press(device_key: str, button: str, press_kind: str) -> None:
    """Record a button-press timestamp so the GUI can flash that row."""
    global _LAST_PRESS
    _LAST_PRESS = {
        "device_key": device_key,
        "button": button,
        "press_kind": press_kind,
        "at": time.time(),
    }


# ── Interpretation pass-throughs ────────────────────────────────────────


async def apply_interpretation(
    *,
    enabled: bool | None = None,
    room_tts_language: str | None = None,
) -> dict[str, Any]:
    """Apply an interpretation update, mirroring admin_audio's POST path.

    Re-uses the same validation + persistence the GUI route uses, so
    the daemon and GUI cannot disagree on what's valid. Returns the
    refreshed state payload after applying.

    On re-enable without a direction, the underlying admin_audio
    handler reads ``interpretation_last_room_tts_language`` and applies
    that — daemon never has to know about it.
    """
    from meeting_scribe.routes.admin_audio import audio_interpretation_post

    body: dict[str, Any] = {}
    if enabled is not None:
        body["enabled"] = enabled
    if room_tts_language is not None:
        body["room_tts_language"] = room_tts_language

    request = _make_synthetic_request(body)
    # The admin handler dispatches based on the cookie-session admin
    # guard; we bypass that by setting a marker on the request scope
    # that the handler honors (see _require_admin_response wrapper
    # below). For now we route through directly since the internal
    # caller has already been authorized by transport-layer means
    # (UDS file permissions or admin cookie).
    response = await audio_interpretation_post(request)
    _ = response  # ignored; we re-read state below for a clean payload
    return build_state_payload()


def _make_synthetic_request(body: dict[str, Any]) -> Any:
    """Build a starlette Request that the admin handler can read JSON from."""
    import json as _json

    from starlette.requests import Request

    raw = _json.dumps(body).encode()

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": raw, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "headers": [(b"content-type", b"application/json")],
        "path": "/api/internal/speakerphone/synthetic",
        "raw_path": b"/api/internal/speakerphone/synthetic",
        "query_string": b"",
        # Mark the request as internal so the cookie-session guard can
        # skip CSRF + admin-password checks. The guard wrapper in
        # routes/admin_speakerphone.py honours this.
        "state": {"speakerphone_internal": True},
    }
    return Request(scope, receive)


# ── Mic mute ────────────────────────────────────────────────────────────


async def apply_mic_mute_toggle() -> dict[str, Any]:
    """Toggle the soft mic mute, mirroring the GUI's POST flow."""
    from meeting_scribe.routes.admin_audio import audio_mic_mute_post

    current = bool(getattr(state, "mic_muted", False))
    body = {"muted": not current}
    request = _make_synthetic_request(body)
    await audio_mic_mute_post(request)
    return build_state_payload()


# ── Meeting record toggle (the atomic Teams-from-idle path) ─────────────


async def apply_meeting_record_toggle() -> dict[str, Any]:
    """Start a meeting (with the default profile) or stop the active one.

    Atomicity: settings updates are persisted *before* attempting to
    start the meeting. If start fails, the prior settings snapshot is
    restored so a half-applied profile is never visible to the next
    button press. Stop has no rollback path (it's already idempotent
    end-state for the daemon).
    """
    meeting = getattr(state, "current_meeting", None)
    if meeting is not None:
        # Currently recording — stop. No settings touched.
        return await _stop_meeting()

    return await _start_meeting_with_default_profile()


async def _stop_meeting() -> dict[str, Any]:
    # Re-use the same lifecycle lock the GUI route uses.
    from meeting_scribe.routes.meeting_lifecycle import (
        _get_meeting_lifecycle_lock,
        _stop_meeting_locked,
    )

    async with _get_meeting_lifecycle_lock():
        await _stop_meeting_locked()
    return build_state_payload()


async def _start_meeting_with_default_profile() -> dict[str, Any]:
    from meeting_scribe.routes.meeting_lifecycle import (
        _get_meeting_lifecycle_lock,
        _start_meeting_locked,
    )

    doc = mapping.load()
    profile = doc.get("default_meeting_profile", {})
    languages = profile.get("languages") or []
    settings_before = dict(_load_settings_override())

    updates: dict[str, Any] = {}
    if isinstance(languages, list) and languages:
        updates["meeting_languages"] = list(languages)
    if isinstance(profile.get("interpretation_enabled"), bool):
        updates["interpretation_enabled"] = profile["interpretation_enabled"]
    if isinstance(profile.get("room_tts_language"), str):
        updates["room_tts_language"] = profile["room_tts_language"]
        updates["interpretation_last_room_tts_language"] = profile["room_tts_language"]
    if isinstance(profile.get("admin_tts_language"), str):
        updates["admin_tts_language"] = profile["admin_tts_language"]

    if updates:
        _save_settings_override(updates)

    request = _make_synthetic_request({})
    try:
        async with _get_meeting_lifecycle_lock():
            await _start_meeting_locked(request)
    except Exception:
        # Restore prior settings so a failed start doesn't leave a
        # half-applied profile visible to the next press. Best-effort —
        # if rollback itself fails we still want to surface the original
        # exception.
        with contextlib.suppress(Exception):
            _save_settings_override(settings_before)
        logger.exception("speakerphone: meeting start failed; rolled back settings")
        raise
    return build_state_payload()


# ── Button-feedback speak path ─────────────────────────────────────────


class FeedbackError(Exception):
    """Generic 4xx-equivalent for the speak handler."""


async def apply_speak(
    *,
    label_id: str,
    language: str | None,
    overrides_inline: Mapping[str, Mapping[str, str]] | None,
    respect_enabled: bool,
) -> dict[str, Any]:
    """Resolve + synthesize + play a button-feedback label.

    Server-side single source of truth for feedback config:

    * Always re-reads ``mapping.load()`` fresh so language/enabled
      changes apply on the very next call (no daemon-poll lag).
    * ``respect_enabled`` controls whether ``button_feedback.enabled``
      gates synthesis. The internal physical-press route passes
      ``True``; the admin preview route passes ``False`` so operators
      can audition labels even while feedback is globally off.
    * Reservations against the shared TTS backend go through
      ``state.tts_dispatch_gate`` (a token queue). Translation
      workers acquire blocking; feedback uses ``get_nowait()`` for an
      atomic non-blocking try-acquire and drops cleanly on
      ``QueueEmpty``. The token is held across the full synthesize
      stream so a translation cannot start mid-feedback.

    Returns one of:
    * ``{ok: True, text, language, source: "feedback"}`` — synthesized + queued
    * ``{ok: True, skipped: True, reason: ...}`` — disabled or backend busy
    * raises ``FeedbackError`` for unknown label_id (router maps to 400).
    """
    from meeting_scribe.speakerphone import labels as sp_labels

    doc = mapping.load()
    feedback_cfg = doc.get("button_feedback", {})

    if respect_enabled and not feedback_cfg.get("enabled", True):
        return {
            "ok": True,
            "skipped": True,
            "reason": "feedback disabled",
        }

    effective_language = (
        language
        if isinstance(language, str) and language.strip()
        else feedback_cfg.get("language", "en")
    )

    # Merge: inline (preview Test button) > saved override > catalog.
    saved_overrides = feedback_cfg.get("overrides", {})
    effective_overrides: dict[str, dict[str, str]] = {
        k: dict(v) for k, v in saved_overrides.items() if isinstance(v, dict)
    }
    if overrides_inline:
        for label, per_label in overrides_inline.items():
            if not isinstance(per_label, Mapping):
                continue
            merged = dict(effective_overrides.get(label, {}))
            for lang, text in per_label.items():
                if isinstance(text, str) and text.strip():
                    merged[lang] = text
            if merged:
                effective_overrides[label] = merged

    try:
        text = sp_labels.resolve(label_id, effective_language, effective_overrides)
    except sp_labels.LabelNotFoundError as exc:
        raise FeedbackError(str(exc)) from exc

    # Non-blocking reservation against the shared TTS dispatch gate.
    # ``get_nowait()`` is the asyncio idiom for atomic try-acquire —
    # it either returns a token immediately or raises ``QueueEmpty``.
    # The ``wait_for(sem.acquire(), timeout=0)`` idiom is broken
    # because the freshly-created acquire awaitable is never complete
    # at t=0; we explicitly use a Queue here to avoid that trap.
    gate = getattr(state, "tts_dispatch_gate", None)
    if gate is None:
        logger.info("apply_speak: tts_dispatch_gate not initialized; dropping")
        return {
            "ok": True,
            "skipped": True,
            "reason": "backend not initialised",
        }
    try:
        token = gate.get_nowait()
    except asyncio.QueueEmpty:
        return {
            "ok": True,
            "skipped": True,
            "reason": "backend busy",
        }

    tts_backend = getattr(state, "tts_backend", None)
    if tts_backend is None:
        gate.put_nowait(token)
        return {
            "ok": False,
            "reason": "tts backend unavailable",
        }

    try:
        diag = await _stream_feedback_to_local_sink(
            tts_backend=tts_backend,
            text=text,
            language=effective_language,
        )
    except Exception:
        logger.exception("apply_speak: synthesis failed for %r", label_id)
        return {"ok": False, "reason": "synthesis failed"}
    finally:
        # ALWAYS release. An exception in synthesize_stream must not
        # leak the gate token — that would starve every future
        # feedback call (and every translation call too, since they
        # share the same gate).
        gate.put_nowait(token)

    return {
        "ok": True,
        "text": text,
        "language": effective_language,
        "source": "feedback",
        **(diag or {}),
    }


async def _stream_feedback_to_local_sink(
    *,
    tts_backend: Any,
    text: str,
    language: str,
) -> dict[str, Any]:
    """Pull synthesize_stream chunks and pipe them to the room sink.

    The room sink is the local-sink listener PipeWire created for the
    SP325 (or whatever device is currently configured as the meeting
    output). Feedback audio plays through the same path translated
    speech uses, so the operator hears it on the SP325 speaker.

    Wire format: synthesize_stream yields float32 mono at the backend's
    sample rate (typically 24 kHz). LocalSinkListener.send_bytes
    expects WAV-headered int16 PCM. We accumulate float32 chunks,
    convert to int16, wrap in a WAV header, and send once at the end
    (matches what the translation worker does at line 615+ — one
    coherent WAV per utterance, no mid-stream click boundaries).

    Returns a small diagnostic dict so ``apply_speak`` can distinguish
    "synthesized + played" from "backend returned zero chunks" and
    "no listeners to fan out to" — both of which used to silently
    masquerade as success.
    """
    import numpy as np

    from meeting_scribe.audio.local_sink import LOCAL_SINK_PCM_RATE
    from meeting_scribe.backends.tts_voices import studio_voice_for

    # The Qwen3 TTS backend rejects requests with neither
    # ``voice_reference`` nor ``studio_voice`` (logs
    # "TTS: no voice specified and no ref_audio; skipping" and returns
    # an empty stream). Feedback has no per-listener voice clone, so
    # we pick the language's catalog studio voice — same helper the
    # translation worker uses.
    studio_voice = studio_voice_for(language)
    logger.info(
        "feedback synth: lang=%s voice=%s chars=%d",
        language,
        studio_voice,
        len(text),
    )

    pcm_chunks: list[np.ndarray] = []
    async for chunk in tts_backend.synthesize_stream(
        text=text,
        language=language,
        voice_reference=None,
        studio_voice=studio_voice,
    ):
        if chunk is None or chunk.size == 0:
            continue
        pcm_chunks.append(chunk)
    if not pcm_chunks:
        return {"played": False, "reason": "tts backend returned no audio"}
    audio = np.concatenate(pcm_chunks)
    # float32 [-1, 1] → int16 LE PCM
    int16 = np.clip(audio * 32767.0, -32768, 32767).astype("<i2").tobytes()

    # WAV header — pyroomac... no, just hand-roll it. 44-byte canonical RIFF.
    sample_rate = getattr(tts_backend, "sample_rate", LOCAL_SINK_PCM_RATE)
    wav_bytes = _wrap_pcm_in_wav(int16, sample_rate=sample_rate, channels=1)

    # Fan out to every registered local-sink listener (typically just
    # one, the room sink). Best-effort: a single listener failure
    # shouldn't block others.
    listeners = list(getattr(state, "_audio_out_clients", []) or [])
    if not listeners:
        return {
            "played": False,
            "reason": "no local-sink listeners registered",
            "bytes": len(wav_bytes),
        }
    fanned_out = 0
    for listener in listeners:
        send = getattr(listener, "send_bytes", None)
        if not callable(send):
            continue
        try:
            await send(wav_bytes)
            fanned_out += 1
        except Exception:
            logger.exception("feedback fan-out failed for listener %r", listener)
    return {
        "played": fanned_out > 0,
        "listeners": fanned_out,
        "bytes": len(wav_bytes),
    }


def _wrap_pcm_in_wav(pcm: bytes, *, sample_rate: int, channels: int) -> bytes:
    """Build a 44-byte RIFF header in front of int16 LE PCM data."""
    import struct as _struct

    data_size = len(pcm)
    byte_rate = sample_rate * channels * 2
    block_align = channels * 2
    header = (
        b"RIFF"
        + _struct.pack("<I", 36 + data_size)
        + b"WAVE"
        + b"fmt "
        + _struct.pack("<I", 16)
        + _struct.pack("<H", 1)  # PCM
        + _struct.pack("<H", channels)
        + _struct.pack("<I", sample_rate)
        + _struct.pack("<I", byte_rate)
        + _struct.pack("<H", block_align)
        + _struct.pack("<H", 16)  # bits per sample
        + b"data"
        + _struct.pack("<I", data_size)
    )
    return header + pcm


# ── LED test (drives each pattern in sequence so the user can preview) ──


async def play_led_test(write_cb) -> None:
    """Play each LED pattern for ~1 s in sequence.

    ``write_cb`` is a callback taking ``bool`` and writing the Mute LED.
    Used by both the GUI (which records the names of patterns it played
    for display) and the daemon (which actually drives the hardware).
    """
    from meeting_scribe.speakerphone.constants import LED_PATTERNS

    for name in ("solid", "blink", "slow_pulse", "fast_blink", "off"):
        schedule = LED_PATTERNS[name]
        # Play roughly one cycle of the pattern, ~1 s window.
        end_at = time.monotonic() + 1.0
        on = True
        idx = 0
        while time.monotonic() < end_at:
            pair = schedule[idx % len(schedule)]
            for slot in pair:
                if slot <= 0:
                    on = not on
                    continue
                write_cb(on)
                await asyncio.sleep(slot / 1000.0)
                on = not on
            idx += 1
    write_cb(False)
