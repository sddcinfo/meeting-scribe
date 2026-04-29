"""``/api/ws`` — admin-only audio-input WebSocket.

Hot path: receives binary chunks of s16le PCM from the browser
AudioWorklet, optionally resamples them to 16 kHz, writes them to
the recording file, and forwards to ASR + diarization. Also
accepts JSON control messages (currently just
``{"type": "set_language", ...}``).

The handler delegates everything past the receive loop to
``_handle_audio``, which is the single audio-bytes pipeline shared
with future audio-source backends.

Wire format for binary frames (legacy-compatible):
    NEW:    [4B LE uint32 sample_rate][N*2 bytes s16le PCM]
    LEGACY: [N*2 bytes s16le PCM at 16 kHz]

Why server-side resampling: browser JS linear interpolation
introduces audible aliasing. torchaudio's Kaiser-windowed sinc
runs on GPU and is effectively transparent.
"""

from __future__ import annotations

import json as _json
import logging
import time
from contextlib import suppress

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from meeting_scribe.runtime import state
from meeting_scribe.server_support.backend_health import (
    _record_backend_failure,
    _record_backend_success,
)
from meeting_scribe.server_support.broadcast import _broadcast_json
from meeting_scribe.server_support.request_scope import _is_guest_scope
from meeting_scribe.server_support.sessions import ClientSession

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/api/ws")
async def websocket_audio(websocket: WebSocket) -> None:
    """WebSocket for audio streaming and transcript events.

    Sends binary audio, receives JSON transcript events.
    Also accepts JSON text messages for language preference:
        {"type": "set_language", "language": "en"}

    Admin-only: rejects hotspot clients AND any client connecting over the
    plain-HTTP guest listener (``ws://`` scheme). Admin must use ``wss://``
    via the HTTPS listener on the LAN.
    """
    if _is_guest_scope(websocket):
        await websocket.close(code=4003, reason="Admin endpoint — use wss://<gb10>:8080")
        return
    await websocket.accept()
    state.ws_connections.add(websocket)
    state._client_prefs[websocket] = ClientSession()
    logger.info("WS connected (total=%d)", len(state.ws_connections))

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("bytes"):
                await _handle_audio(msg["bytes"])
            elif msg.get("text"):
                try:
                    parsed = _json.loads(msg["text"])
                    if isinstance(parsed, dict) and parsed.get("type") == "set_language":
                        lang = parsed.get("language", "")
                        if lang:
                            state._client_prefs[websocket].preferred_language = lang
                            logger.info("Audio WS set language preference: %s", lang)
                except Exception:
                    pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        # "Cannot call receive once a disconnect message has been received"
        # is the expected race when the client tears the WS down between
        # receive() calls — fires on every clean disconnect. Demote to
        # DEBUG so the log isn't spammed; everything else stays WARN.
        if "disconnect message" in str(e):
            logger.debug("WS receive after disconnect (expected): %s", e)
        else:
            logger.warning("WS error: %s", e)
    finally:
        state.ws_connections.discard(websocket)
        state._client_prefs.pop(websocket, None)
        logger.info("WS disconnected (remaining=%d)", len(state.ws_connections))


async def _handle_audio(data: bytes) -> None:
    """Handle audio from browser AudioWorklet.

    Wire format (per chunk, legacy-compatible):
        - NEW: [4B LE uint32 sample_rate][N*2 bytes s16le PCM at native rate]
          Server resamples to 16kHz using torchaudio Kaiser-windowed sinc.
        - LEGACY: [N*2 bytes s16le PCM, assumed 16kHz]
          Detected by the first 4 bytes NOT being a plausible sample rate.

    Why server-side resampling: browser JS linear interpolation causes aliasing.
    torchaudio's Kaiser sinc is effectively transparent and runs on GPU.
    Bandwidth cost (~3× at 48kHz) is trivial on local WiFi.
    """
    # ``state.last_audio_chunk_ts`` is the silence watchdog's heartbeat
    # signal — mutate it on every inbound audio frame.
    state.last_audio_chunk_ts = time.monotonic()

    if len(data) < 6:
        return

    # Parse optional sample-rate header. A valid header has a uint32 in the
    # expected range (8000..192000). Anything else is legacy raw 16kHz PCM.
    header_rate = int.from_bytes(data[:4], "little")
    if 8000 <= header_rate <= 192000:
        source_rate = header_rate
        pcm_bytes = data[4:]
    else:
        source_rate = 16000
        pcm_bytes = data

    if len(pcm_bytes) < 2:
        return

    state.metrics.audio_chunks += 1
    num_source_samples = len(pcm_bytes) // 2
    state.metrics.audio_seconds += num_source_samples / source_rate

    # Decode s16le → float32 [-1, 1]
    source = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Resample to 16kHz using torchaudio Kaiser sinc (anti-aliased).
    if source_rate != 16000:
        try:
            resampled = state.resampler.resample(source, source_rate=source_rate, target_rate=16000)
        except Exception as e:
            logger.warning("Resample %d→16000 failed: %s — falling back to raw", source_rate, e)
            resampled = source
    else:
        resampled = source

    # Back to s16le for downstream consumers (audio writer + ASR).
    # Clip to [-1, 1] before quantizing to avoid wraparound on overshoot.
    resampled_clipped = np.clip(resampled, -1.0, 1.0)
    pcm16_bytes = (resampled_clipped * 32767).astype(np.int16).tobytes()

    # Log audio level at chunk #1 (proves audio is flowing) and every
    # 1500 after (~1 minute @ 16ms chunks). Was every 40 chunks at WARN
    # level which spammed the log with ~50 lines/min during recording —
    # the once-per-minute INFO cadence keeps the diagnostic without the
    # noise. First chunk stays WARN so it surfaces clearly.
    if state.metrics.audio_chunks == 1 or state.metrics.audio_chunks % 1500 == 0:
        peak = float(np.max(np.abs(resampled_clipped))) if len(resampled_clipped) > 0 else 0.0
        rms = float(np.sqrt(np.mean(resampled_clipped**2))) if len(resampled_clipped) > 0 else 0.0
        emit = logger.warning if state.metrics.audio_chunks == 1 else logger.info
        emit(
            "Audio chunk #%d: %dHz→16kHz, %d→%d samples, peak=%.4f rms=%.4f",
            state.metrics.audio_chunks,
            source_rate,
            num_source_samples,
            len(resampled_clipped),
            peak,
            rms,
        )

    # Write 16kHz audio to recording file and capture the absolute sample
    # offset this chunk lands at — single source of truth for ASR and
    # timeline alignment.
    chunk_sample_offset: int | None = None
    if state.audio_writer and state.meeting_start_time > 0:
        elapsed_ms = int((time.monotonic() - state.meeting_start_time) * 1000)
        # Snapshot the audio writer's byte position BEFORE this write.
        pre_bytes = state.audio_writer.total_bytes
        state.audio_writer.write_at(pcm16_bytes, elapsed_ms)
        chunk_sample_offset = pre_bytes // 2  # s16le → 1 sample = 2 bytes

        # One-shot wall-clock anchor for the recording file. Set on the
        # very first sample so `recording.pcm` becomes an absolute-time
        # record: byte offset → unix epoch is deterministic forever.
        # On resume (append mode) pre_bytes>0 and the anchor was already
        # written at the original start.
        if (
            state.current_meeting is not None
            and pre_bytes == 0
            and getattr(state.current_meeting, "recording_started_epoch_ms", 0) == 0
        ):
            state.current_meeting.recording_started_epoch_ms = int(time.time() * 1000)
            with suppress(Exception):
                state.storage._write_meta(state.current_meeting)
            logger.info(
                "Recording anchor: meeting=%s started_epoch_ms=%d",
                state.current_meeting.meeting_id,
                state.current_meeting.recording_started_epoch_ms,
            )

        # Audio drift detection: wall clock vs cumulative audio bytes
        audio_elapsed_ms = int(state.metrics.audio_seconds * 1000)
        drift_ms = elapsed_ms - audio_elapsed_ms
        if abs(drift_ms) > 500 and state.metrics.audio_chunks % 20 == 0:
            logger.warning(
                "Audio drift detected: %dms (wall=%dms, audio=%dms)",
                drift_ms,
                elapsed_ms,
                audio_elapsed_ms,
            )
            await _broadcast_json({"type": "audio_drift", "drift_ms": drift_ms})

    if not state.asr_backend:
        return

    # Feed 16kHz s16le bytes to ASR with the absolute audio-file sample
    # offset so every emitted event is stamped against the ACTUAL audio
    # playback position, not an internal counter that resets on restart.
    await state.asr_backend.process_audio_bytes(pcm16_bytes, sample_offset=chunk_sample_offset)

    # Also feed to diarization backend (runs in parallel, buffers internally)
    if state.diarize_backend:
        try:
            await state.diarize_backend.process_audio(
                resampled_clipped.astype(np.float32),
                state.metrics.audio_chunks * len(resampled_clipped),
            )
            _record_backend_success("diarize")
        except Exception as e:
            _record_backend_failure("diarize", str(e))
            # Diarization failures shouldn't block ASR
