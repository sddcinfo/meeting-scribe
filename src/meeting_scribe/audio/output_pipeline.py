"""Audio fan-out from the TTS / passthrough sources to listeners.

Five functions own the encoding + send path:

* ``_build_riff_wav`` — pure WAV header construction (RIFF/16-bit
  PCM/mono).
* ``_buffer_pending_audio`` — bounded queue while a listener is
  still mid-format-negotiation (grace window).
* ``_deliver_audio_to_listener`` — the dispatcher: routes audio to
  one listener in their negotiated wire format (wav-pcm /
  mse-fmp4-aac), handling the grace-window default and lazy MSE
  encoder construction.
* ``_send_passthrough_audio`` — fans original ASR audio out to
  ``full``-mode listeners whose preferred language matches the
  source.
* ``_send_audio_to_listeners`` — fans synthesized TTS audio to
  listeners in the target language + voice mode.

Pulled out of ``server.py`` once the listener metrics
(``state.metrics.listener_*``) and listener registry
(``state._audio_out_clients`` / ``state._audio_out_prefs``) became
canonical-state. ``ws.audio_output`` consumes
``_deliver_audio_to_listener`` from here for the grace-window flush.
"""

from __future__ import annotations

import logging
import struct
import time

import numpy as np
from fastapi import WebSocket

from meeting_scribe.runtime import state
from meeting_scribe.server_support.peer import _peer_str
from meeting_scribe.server_support.sessions import ClientSession
from meeting_scribe.server_support.translation_demand import _norm_lang
from meeting_scribe.ws.audio_output import _AUDIO_FORMAT_PENDING_CAP_S

logger = logging.getLogger(__name__)


def _build_riff_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Build a complete RIFF WAV (16-bit PCM, mono) for float32 audio in [-1, 1].

    Centralises the struct.pack header construction that used to be
    duplicated between ``_send_passthrough_audio`` and
    ``_send_audio_to_listeners``. Zero behavior change for legacy
    clients.
    """
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    data_bytes = pcm.tobytes()
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(data_bytes),
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        1,  # mono
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        len(data_bytes),
    )
    return header + data_bytes


def _buffer_pending_audio(pref: ClientSession, audio: np.ndarray, source_rate: int) -> None:
    """Queue audio for a listener still mid-format-negotiation, bounded.

    Capped at ``_AUDIO_FORMAT_PENDING_CAP_S`` seconds per listener so
    a stuck handshake cannot balloon memory. When the cap is exceeded,
    the new item is silently dropped — by the time this fires the
    grace period has been running for well under 1 s and the
    worst-case loss is tiny.
    """
    total = sum(len(a) / max(sr, 1) for a, sr in pref.pending_audio) + len(audio) / max(
        source_rate, 1
    )
    if total > _AUDIO_FORMAT_PENDING_CAP_S:
        return
    pref.pending_audio.append((audio, source_rate))


async def _deliver_audio_to_listener(
    ws: WebSocket,
    pref: ClientSession,
    audio: np.ndarray,
    source_rate: int,
    wav_cache: dict | None,
) -> bool:
    """Encode + send one segment of audio to one listener in their format.

    Dispatches on ``pref.audio_format``:
      - ``None`` during grace → buffer into ``pref.pending_audio``
      - ``None`` after grace  → default to ``"wav-pcm"`` and continue
      - ``"wav-pcm"``         → send one RIFF WAV binary frame, no prefix
      - ``"mse-fmp4-aac"``    → lazy-create the encoder (sending the
                                 init frame prefixed with ``b'\\x49'``),
                                 then feed ``audio`` and send any
                                 resulting fragment prefixed with
                                 ``b'\\x46'``.

    Returns True on success OR when the listener is in a valid held
    state (grace buffer, empty below-threshold MSE return). Returns
    False only when the WS send raised — caller discards the listener.
    """
    fmt = pref.audio_format
    if fmt is None:
        now = time.monotonic()
        if now > pref.grace_deadline:
            pref.audio_format = "wav-pcm"
            fmt = "wav-pcm"
            logger.info(
                "Audio-out format: defaulted to wav-pcm for %s (no set_format received)",
                _peer_str(ws),
            )
        else:
            # Still in grace; buffer and flush when set_format arrives.
            _buffer_pending_audio(pref, audio, source_rate)
            return True

    if fmt == "wav-pcm":
        # Reuse one cached WAV across every wav-pcm listener in this
        # fan-out call. Cache key covers the source rate because
        # passthrough (16 kHz) and TTS (24 kHz) coexist.
        cache_key = ("wav", source_rate, id(audio))
        wav_bytes: bytes | None = None
        if wav_cache is not None:
            wav_bytes = wav_cache.get(cache_key)
        if wav_bytes is None:
            wav_bytes = _build_riff_wav(audio, source_rate)
            if wav_cache is not None:
                wav_cache[cache_key] = wav_bytes
        try:
            await ws.send_bytes(wav_bytes)
            state.metrics.listener_deliveries += 1
            return True
        except Exception as e:
            logger.warning("audio_out wav send failed peer=%s: %s", _peer_str(ws), e)
            state.metrics.listener_send_failed += 1
            return False

    if fmt == "mse-fmp4-aac":
        from meeting_scribe.backends.mse_encoder import Fmp4AacEncoder

        # Lazy encoder creation on first audio delivery. The init
        # frame is sent once, immediately before the first media
        # fragment for this listener. After this point pref.mse_encoder
        # stays live until WS disconnect.
        if pref.mse_encoder is None:
            try:
                pref.mse_encoder = Fmp4AacEncoder()
            except Exception as e:
                logger.warning(
                    "mse encoder construction failed peer=%s: %s",
                    _peer_str(ws),
                    e,
                )
                return False
            logger.info("mse encoder lazy-created for %s", _peer_str(ws))
            try:
                await ws.send_bytes(b"\x49" + pref.mse_encoder.init_segment())
            except Exception as e:
                logger.warning("mse init send failed peer=%s: %s", _peer_str(ws), e)
                return False

        # Encode. Empty return is normal below-threshold accumulation —
        # NOT a fault. Do not log, do not recreate the encoder.
        try:
            fragment = pref.mse_encoder.encode(audio, source_rate)
        except Exception as e:
            logger.warning("mse encode failed peer=%s: %s", _peer_str(ws), e)
            return False
        pref.bytes_in_since_last_emit += len(audio)
        if not fragment:
            # Accumulating — still a successful "delivery" from the
            # listener's perspective. Stuck-detection is a separate
            # concern handled by the health check below.
            return True
        try:
            await ws.send_bytes(b"\x46" + fragment)
            state.metrics.listener_deliveries += 1
            pref.last_fragment_at = time.monotonic()
            pref.bytes_in_since_last_emit = 0
            return True
        except Exception as e:
            logger.warning("mse fragment send failed peer=%s: %s", _peer_str(ws), e)
            state.metrics.listener_send_failed += 1
            return False

    # Unknown format — impossible in practice (validated in set_format
    # handler) but fail closed.
    logger.warning("audio_out deliver: unknown format %r for %s", fmt, _peer_str(ws))
    return False


async def _send_passthrough_audio(audio: np.ndarray, source_language: str) -> None:
    """Send original audio to ``full`` interpretation clients whose
    preferred language matches.

    Only sent to clients in ``full`` mode where ``preferred_language ==
    source_language``. Translation-only clients never receive
    pass-through audio.
    """
    source_norm = _norm_lang(source_language)
    recipients = [
        ws
        for ws in state._audio_out_clients
        if state._audio_out_prefs.get(ws)
        and state._audio_out_prefs[ws].interpretation_mode == "full"
        and _norm_lang(state._audio_out_prefs[ws].preferred_language) == source_norm
    ]
    if not recipients:
        return

    wav_cache: dict = {}
    sent = 0
    dead: list[WebSocket] = []
    for ws in recipients:
        pref = state._audio_out_prefs.get(ws)
        if pref is None:
            continue
        ok = await _deliver_audio_to_listener(ws, pref, audio, 16000, wav_cache)
        if ok:
            sent += 1
        else:
            dead.append(ws)
    for ws in dead:
        state._audio_out_clients.discard(ws)
        state._audio_out_prefs.pop(ws, None)
        state.metrics.listener_removed_on_send_error += 1

    # One INFO line per delivery so we can validate "full" mode
    # end-to-end from the journal. Rate is bounded by the final-ASR-
    # segment cadence (~1/sec at most), so this won't flood.
    if sent:
        logger.info(
            "passthrough sent: source_lang=%s listeners=%d",
            source_norm,
            sent,
        )


async def _send_audio_to_listeners(
    audio: np.ndarray,
    target_language: str,
    voice_mode: str = "cloned",
) -> int:
    """Send synthesized audio to audio-out clients in their negotiated format.

    Filters listeners by:
      - preferred_language matching ``target_language`` (normalized —
        both sides pass through ``_norm_lang`` so ``en-US`` matches
        ``en``)
      - voice_mode matching the synthesized voice mode

    Returns the number of listeners the audio was successfully sent
    to. Each matching listener receives the audio in the format it
    negotiated via ``set_format`` — wav-pcm (one RIFF WAV frame) or
    mse-fmp4-aac (init frame on first delivery, fragments thereafter).
    See ``_deliver_audio_to_listener`` for the dispatch logic.
    """
    target_norm = _norm_lang(target_language)

    wav_cache: dict = {}
    sent = 0
    dead: list[WebSocket] = []
    for ws in list(state._audio_out_clients):
        pref = state._audio_out_prefs.get(ws)
        if pref is None:
            continue
        pref_lang = _norm_lang(pref.preferred_language)
        if pref_lang and pref_lang != target_norm:
            continue
        client_voice = getattr(pref, "voice_mode", "studio")
        if client_voice != voice_mode:
            continue
        send_start = time.monotonic()
        ok = await _deliver_audio_to_listener(ws, pref, audio, 24000, wav_cache)
        if ok:
            sent += 1
            state.metrics.listener_send_ms.append((time.monotonic() - send_start) * 1000.0)
        else:
            dead.append(ws)
    for ws in dead:
        state._audio_out_clients.discard(ws)
        state._audio_out_prefs.pop(ws, None)
        state.metrics.listener_removed_on_send_error += 1
    return sent
