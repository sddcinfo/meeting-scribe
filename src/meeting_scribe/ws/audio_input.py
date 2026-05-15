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

import asyncio
import json as _json
import logging
import os
import time
from contextlib import suppress
from dataclasses import dataclass

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from meeting_scribe.runtime import state
from meeting_scribe.server_support.admin_guard import require_admin_ws
from meeting_scribe.server_support.backend_health import (
    _record_backend_failure,
    _record_backend_success,
)
from meeting_scribe.server_support.broadcast import _broadcast_json
from meeting_scribe.server_support.sessions import ClientSession

logger = logging.getLogger(__name__)

# Keep below the ASR buffer-level VAD threshold because this is per-chunk
# speech activity, not a transcript decision. It only delays room-speaker
# TTS release while the room is audibly active.
_SPEECH_ACTIVITY_RMS_THRESHOLD = float(
    os.environ.get("SCRIBE_SPEECH_ACTIVITY_RMS_THRESHOLD", "0.0035")
)
# Capture-liveness floor — much lower than the speech threshold so any
# non-zero ADC sample bumps ``state.last_nonzero_audio_ts``. A healthy mic
# in a quiet room sits around ~1e-3 from background noise alone, well
# above this. We compare against ``peak`` (max abs sample), not RMS, so a
# single non-zero sample in a noisy buffer is enough to prove the ADC
# is producing data.
_AUDIO_LIVENESS_FLOOR = float(os.environ.get("SCRIBE_AUDIO_LIVENESS_FLOOR", "0.0001"))
# Broadcast `mic_level` to clients every Nth chunk. Frame cadence is
# ~50 chunks/sec on the typical 16ms hop, so N=3 lands the live meter
# at ~16 Hz — smooth animation without flooding the WS fan-out.
_MIC_LEVEL_BROADCAST_EVERY_N = int(os.environ.get("SCRIBE_MIC_LEVEL_BROADCAST_EVERY_N", "3"))
_DIARIZE_QUEUE_MAX = int(os.environ.get("SCRIBE_LIVE_DIARIZE_QUEUE_MAX", "256"))
_diarize_queue: asyncio.Queue[_DiarizeItem] | None = None
_diarize_worker: asyncio.Task | None = None
_diarize_dropped = 0


@dataclass(slots=True)
class _DiarizeItem:
    meeting_id: str
    audio: np.ndarray
    sample_offset: int


def _ensure_diarize_worker() -> asyncio.Queue[_DiarizeItem]:
    global _diarize_queue, _diarize_worker
    if _diarize_queue is None:
        _diarize_queue = asyncio.Queue(maxsize=_DIARIZE_QUEUE_MAX)
    if _diarize_worker is None or _diarize_worker.done():
        _diarize_worker = asyncio.create_task(_diarize_worker_loop(), name="live-diarize-worker")
    return _diarize_queue


async def _diarize_worker_loop() -> None:
    assert _diarize_queue is not None
    while True:
        item = await _diarize_queue.get()
        try:
            backend = state.diarize_backend
            current = state.current_meeting
            if backend is None or current is None or current.meeting_id != item.meeting_id:
                continue
            await backend.process_audio(item.audio, item.sample_offset)
            _record_backend_success("diarize")
        except Exception as e:
            _record_backend_failure("diarize", str(e))
            logger.warning("Live diarization worker error: %s", e)
        finally:
            _diarize_queue.task_done()


def _queue_diarization(audio: np.ndarray, sample_offset: int | None) -> None:
    """Queue live diarization without blocking the audio receive hot path.

    Recording and ASR must keep up with mic wall-clock. Live speaker attribution
    is useful, but final diarization reprocesses the full recording after stop,
    so it is allowed to lag or drop oldest live chunks under pressure.
    """
    global _diarize_dropped
    meeting = state.current_meeting
    if state.diarize_backend is None or meeting is None:
        return
    queue = _ensure_diarize_worker()
    offset = sample_offset if sample_offset is not None else state.metrics.audio_chunks * len(audio)
    item = _DiarizeItem(meeting.meeting_id, audio.copy(), offset)
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        _diarize_dropped += 1
        with suppress(Exception):
            queue.get_nowait()
            queue.task_done()
        with suppress(asyncio.QueueFull):
            queue.put_nowait(item)
        if _diarize_dropped == 1 or _diarize_dropped % 100 == 0:
            logger.warning(
                "Live diarization queue full; dropped %d chunk(s) to protect ASR ingest",
                _diarize_dropped,
            )


router = APIRouter()


@router.websocket("/api/ws")
async def websocket_audio(websocket: WebSocket) -> None:
    """WebSocket for audio streaming and transcript events.

    Sends binary audio, receives JSON transcript events.
    Also accepts JSON text messages for language preference:
        {"type": "set_language", "language": "en"}

    Admin-only: gated by ``require_admin_ws`` (AP subnet + admin
    cookie). Off-AP or unauthenticated callers get the WS closed with
    the appropriate code; the body never executes.
    """
    if not await require_admin_ws(websocket):
        return
    await websocket.accept()
    state.ws_connections.add(websocket)
    # The audio WS IS the recorder side of the pipeline; track it in
    # the dedicated set so ``/api/status.connections`` reflects only
    # recorder sockets (the meeting reconciler's "upgrade to recorder"
    # branch needs this count to reach 0 after a stale recorder drops -
    # if it counts view-only / kiosk WSs too, the admin tab wedges
    # forever in view-only mode).
    state.recorder_ws_connections.add(websocket)
    state._client_prefs[websocket] = ClientSession()
    logger.info("WS connected (total=%d)", len(state.ws_connections))

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("bytes"):
                # Server-side mic capture (audio_routing) owns ASR audio
                # when active — drop browser-mic frames so the two
                # streams don't interleave into ``_handle_audio``. Keep
                # the WS open so JSON control messages (set_language)
                # still flow; the JS side discovers it's idle from the
                # absence of resampler peak/RMS feedback.
                if state.server_mic_active:
                    continue
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
        state.recorder_ws_connections.discard(websocket)
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
    # signal — mutate it on every inbound audio frame. We bump it BEFORE
    # the mute gate so a privacy pause doesn't trigger a "connection
    # stalled, please reconnect" prompt while the operator deliberately
    # silences the room.
    state.last_audio_chunk_ts = time.monotonic()

    # Soft input mute (audio_routing): a true privacy pause. Drop the
    # frame before anything reaches disk OR the ASR backend so the
    # recording and the transcript both honor the muted span.
    if state.mic_input_muted:
        state.metrics.mic_muted_chunks_dropped += 1
        return

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
    rms = float(np.sqrt(np.mean(resampled_clipped**2))) if len(resampled_clipped) > 0 else 0.0
    peak = float(np.max(np.abs(resampled_clipped))) if len(resampled_clipped) > 0 else 0.0
    # ``last_nonzero_audio_ts`` is the silence-watchdog liveness signal:
    # bumped on any frame with a non-zero sample (peak above the liveness
    # floor). Distinct from ``last_speech_audio_ts`` so a quiet healthy
    # mic stays alive without tripping the silence alarm, but a frame-of-
    # zeros stream (dead-ADC failure) does trip it.
    if peak >= _AUDIO_LIVENESS_FLOOR:
        state.last_nonzero_audio_ts = time.monotonic()
    if rms >= _SPEECH_ACTIVITY_RMS_THRESHOLD:
        state.last_speech_audio_ts = time.monotonic()
    # Live mic-level broadcast. Throttled to ~16 Hz so the meter
    # animates smoothly without saturating the WS fan-out. Same handler
    # services both browser-mic frames and SP325/server-mic frames
    # (audio/server_mic.py funnels into this function), so subscribers
    # see a continuous level regardless of the active mic source.
    if state.metrics.audio_chunks % _MIC_LEVEL_BROADCAST_EVERY_N == 0:
        peak_pct = min(100, int(peak * 100))
        await _broadcast_json({"type": "mic_level", "peak_pct": peak_pct})
    pcm16_bytes = (resampled_clipped * 32767).astype(np.int16).tobytes()

    # Log audio level at chunk #1 (proves audio is flowing) and every
    # 1500 after (~1 minute @ 16ms chunks). Was every 40 chunks at WARN
    # level which spammed the log with ~50 lines/min during recording —
    # the once-per-minute INFO cadence keeps the diagnostic without the
    # noise. First chunk stays WARN so it surfaces clearly.
    if state.metrics.audio_chunks == 1 or state.metrics.audio_chunks % 1500 == 0:
        peak = float(np.max(np.abs(resampled_clipped))) if len(resampled_clipped) > 0 else 0.0
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

        # Audio drift detection: wall clock vs cumulative audio bytes.
        # Skip when server-mic owns ASR — in that mode browser frames are
        # dropped at the top of receive_audio (state.server_mic_active gate),
        # so state.metrics.audio_seconds is advanced by the server-mic path
        # while elapsed_ms is anchored to the websocket OPEN time. The two
        # clocks have no shared origin once the browser stops contributing
        # bytes, and the metric reports a steadily-growing fictitious drift
        # that pollutes logs and fires `audio_drift` broadcasts to every
        # connected client. The drift signal is only meaningful while the
        # browser is the active ASR audio source.
        if not state.server_mic_active:
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

    _queue_diarization(resampled_clipped.astype(np.float32), chunk_sample_offset)
