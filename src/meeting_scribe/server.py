"""FastAPI server — POC simplified for localhost single-meeting use.

Single meeting at a time. No auth (localhost only). No TLS needed.
Wires together: storage, audio resample, ASR (WhisperLiveKit), translation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING

import fastapi
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from meeting_scribe.audio.resample import (  # kept for legacy/test clients
    Resampler,
)
from meeting_scribe.config import ServerConfig
from meeting_scribe.models import (
    DetectedSpeaker,
    MeetingMeta,
    MeetingState,
    RoomLayout,
    SpeakerAttribution,
    TranscriptEvent,
)
from meeting_scribe.speaker.enrollment import EnrolledSpeaker, SpeakerEnrollmentStore
from meeting_scribe.storage import AudioWriter, MeetingStorage
from meeting_scribe.translation.queue import TranslationQueue

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    import numpy as np

    from meeting_scribe.backends.base import ASRBackend, TranslateBackend

logger = logging.getLogger(__name__)

# Global state
config: ServerConfig
storage: MeetingStorage
resampler: Resampler
translation_queue: TranslationQueue | None = None

# Backends
asr_backend: ASRBackend | None = None
translate_backend: TranslateBackend | None = None
tts_backend = None  # Qwen3TTSBackend (optional)

# Speaker enrollment + room layout draft
enrollment_store: SpeakerEnrollmentStore = SpeakerEnrollmentStore()
draft_layout: RoomLayout = RoomLayout()

# Single active meeting + connections
current_meeting: MeetingMeta | None = None
ws_connections: set[WebSocket] = set()
audio_writer: AudioWriter | None = None
meeting_start_time: float = 0.0  # monotonic time for audio alignment
detected_speakers: list[DetectedSpeaker] = []  # per-meeting speaker state
speaker_verifier = None  # SpeakerVerifier instance
_background_tasks: set[asyncio.Task] = set()  # prevent GC of fire-and-forget tasks


# ── Real-time metrics ──────────────────────────────────────────


class Metrics:
    """Server-wide performance metrics, reset per meeting."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.meeting_start: float = 0.0
        self.audio_chunks: int = 0
        self.audio_seconds: float = 0.0
        self.asr_events: int = 0
        self.asr_partials: int = 0
        self.asr_finals: int = 0
        self.translations_submitted: int = 0
        self.translations_completed: int = 0
        self.translations_failed: int = 0
        self.translation_total_ms: float = 0.0
        self.last_asr_event_time: float = 0.0
        self.ollama_warmup_ms: float = 0.0
        self.asr_load_ms: float = 0.0

    @property
    def elapsed_seconds(self) -> float:
        if self.meeting_start == 0:
            return 0.0
        return time.monotonic() - self.meeting_start

    @property
    def asr_events_per_second(self) -> float:
        elapsed = self.elapsed_seconds
        return self.asr_events / elapsed if elapsed > 0 else 0.0

    @property
    def avg_translation_ms(self) -> float:
        if self.translations_completed == 0:
            return 0.0
        return self.translation_total_ms / self.translations_completed

    def to_dict(self) -> dict:
        return {
            "elapsed_s": round(self.elapsed_seconds, 1),
            "audio_chunks": self.audio_chunks,
            "audio_s": round(self.audio_seconds, 1),
            "asr_events": self.asr_events,
            "asr_partials": self.asr_partials,
            "asr_finals": self.asr_finals,
            "asr_eps": round(self.asr_events_per_second, 1),
            "translations_submitted": self.translations_submitted,
            "translations_completed": self.translations_completed,
            "translations_failed": self.translations_failed,
            "avg_translation_ms": round(self.avg_translation_ms),
            "ollama_warmup_ms": round(self.ollama_warmup_ms),
            "asr_load_ms": round(self.asr_load_ms),
        }


metrics = Metrics()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    global config, storage, resampler, translation_queue
    global asr_backend, translate_backend

    config = ServerConfig.from_env()
    storage = MeetingStorage(config)
    resampler = Resampler()

    storage.recover_interrupted()
    storage.cleanup_retention()

    # Init backends
    await _init_backends()

    # Translation queue
    if translate_backend and config.translate_enabled:
        translation_queue = TranslationQueue(
            maxsize=config.translate_queue_maxsize,
            concurrency=config.translate_queue_concurrency,
            timeout=config.translate_timeout_seconds,
            on_result=_broadcast_translation,
        )
        await translation_queue.start(translate_backend)

    # TTS — voice cloning (optional, requires HF token for Qwen3-TTS)
    global tts_backend
    try:
        from meeting_scribe.backends.tts_qwen3 import Qwen3TTSBackend

        tts = Qwen3TTSBackend(
            vllm_url=config.translate_vllm_url if config.translate_backend == "vllm" else None
        )
        await tts.start()
        if tts.available:
            tts_backend = tts
    except Exception as e:
        logger.info("TTS disabled: %s", e)

    logger.info("Meeting Scribe ready on port %d", config.port)
    yield

    # Shutdown
    if translation_queue:
        await translation_queue.stop()
    if tts_backend:
        await tts_backend.stop()
    if asr_backend:
        await asr_backend.stop()
    if translate_backend:
        await translate_backend.stop()


async def _init_backends() -> None:
    """Initialize ASR and translation backends based on config."""
    global asr_backend, translate_backend

    # ASR — Qwen3-ASR (best quality) or Whisper (fallback)
    try:
        from meeting_scribe.backends.asr_qwen3 import Qwen3ASRBackend

        be = Qwen3ASRBackend()
        be.set_event_callback(_process_event)
        t0 = time.monotonic()
        await be.start()
        metrics.asr_load_ms = (time.monotonic() - t0) * 1000
        asr_backend = be
        logger.info("ASR: Qwen3-ASR-0.6B (MLX) loaded in %.0fms", metrics.asr_load_ms)
    except Exception as e:
        logger.warning("Qwen3-ASR failed: %s — falling back to mlx-whisper", e)
        try:
            from meeting_scribe.backends.asr_mlx import MlxWhisperBackend

            model_repo = f"mlx-community/whisper-{config.asr_model}"
            be = MlxWhisperBackend(model=model_repo)
            be.set_event_callback(_process_event)
            t0 = time.monotonic()
            await be.start()
            metrics.asr_load_ms = (time.monotonic() - t0) * 1000
            asr_backend = be
            logger.info(
                "ASR: mlx-whisper (%s) loaded in %.0fms", config.asr_model, metrics.asr_load_ms
            )
        except Exception as e2:
            logger.warning("ASR disabled: %s", e2)

    # Translation — select backend based on config
    if config.translate_enabled:
        t0 = time.monotonic()
        try:
            if config.translate_backend == "vllm":
                from meeting_scribe.backends.translate_vllm import VllmTranslateBackend

                be = VllmTranslateBackend(
                    base_url=config.translate_vllm_url,
                    model=config.translate_vllm_model or None,
                )
                await be.start()
                warmup = await be.translate("Hello", "en", "ja")
                metrics.ollama_warmup_ms = (time.monotonic() - t0) * 1000
                translate_backend = be
                logger.info(
                    "Translation: vLLM (%.0fms, test: 'Hello' → '%s')",
                    metrics.ollama_warmup_ms,
                    warmup[:30],
                )
            elif config.translate_backend == "ollama":
                from meeting_scribe.backends.translate_ollama import OllamaTranslateBackend

                be = OllamaTranslateBackend(
                    model=config.translate_model, base_url=config.translate_ollama_url
                )
                await be.start()
                translate_backend = be
                metrics.ollama_warmup_ms = (time.monotonic() - t0) * 1000
                logger.info(
                    "Translation: Ollama (%s, %.0fms)",
                    config.translate_model,
                    metrics.ollama_warmup_ms,
                )
            else:  # ct2 (default)
                from meeting_scribe.backends.translate_ct2 import CTranslate2TranslateBackend

                be = CTranslate2TranslateBackend()
                await be.start()
                warmup = await be.translate("Hello", "en", "ja")
                metrics.ollama_warmup_ms = (time.monotonic() - t0) * 1000
                translate_backend = be
                logger.info(
                    "Translation: NLLB CT2 (%.0fms, test: 'Hello' → '%s')",
                    metrics.ollama_warmup_ms,
                    warmup[:30],
                )
        except Exception as e:
            logger.warning("Translation backend '%s' failed: %s", config.translate_backend, e)
            # Fallback chain: vllm → ct2 → ollama → disabled
            if config.translate_backend != "ct2":
                try:
                    from meeting_scribe.backends.translate_ct2 import CTranslate2TranslateBackend

                    be = CTranslate2TranslateBackend()
                    await be.start()
                    translate_backend = be
                    logger.info("Translation: CT2 fallback")
                except Exception:
                    logger.warning("Translation disabled: all backends failed")


# --- App ---

app = FastAPI(title="Meeting Scribe", lifespan=lifespan)
STATIC_DIR = Path(__file__).parent.parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


@app.get("/how-it-works")
async def how_it_works() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "how-it-works.html").read_text())


@app.get("/api/status")
async def get_status() -> JSONResponse:
    """Current server and meeting status."""
    return JSONResponse(
        {
            "meeting": {
                "id": current_meeting.meeting_id if current_meeting else None,
                "state": current_meeting.state.value if current_meeting else None,
            },
            "backends": {
                "asr": asr_backend is not None,
                "diarize": config.diarize_enabled,
                "translate": translate_backend is not None,
                "tts": tts_backend is not None and tts_backend.available,
            },
            "connections": len(ws_connections),
            "metrics": metrics.to_dict(),
        }
    )


# ── Meetings History ──────────────────────────────────────────


@app.get("/api/meetings")
async def list_meetings() -> JSONResponse:
    """List all meetings with metadata."""
    import json as _json

    meetings_dir = storage._meetings_dir
    results = []
    if meetings_dir.exists():
        for d in meetings_dir.iterdir():
            meta_path = d / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = _json.loads(meta_path.read_text())
                journal_path = d / "journal.jsonl"
                event_count = 0
                if journal_path.exists():
                    event_count = sum(1 for _ in journal_path.open())

                results.append(
                    {
                        "meeting_id": meta["meeting_id"],
                        "state": meta["state"],
                        "created_at": meta.get("created_at", ""),
                        "event_count": event_count,
                        "has_room": (d / "room.json").exists(),
                        "has_speakers": (d / "speakers.json").exists(),
                    }
                )
            except Exception:
                continue

    # Most recent first
    results.sort(key=lambda m: m["created_at"], reverse=True)
    return JSONResponse({"meetings": results})


@app.put("/api/meetings/{meeting_id}/events/{segment_id}/speaker")
async def update_segment_speaker(
    meeting_id: str, segment_id: str, request: fastapi.Request
) -> JSONResponse:
    """Manually assign a speaker to a segment. Appends correction to journal."""
    import json as _json

    body = await request.json()
    speaker_name = body.get("display_name", "").strip()
    if not speaker_name:
        return JSONResponse({"error": "display_name required"}, status_code=400)

    journal_path = storage._meetings_dir / meeting_id / "journal.jsonl"
    if not journal_path.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    # Append correction event to journal (append-only, no rewrite)
    correction = {
        "type": "speaker_correction",
        "segment_id": segment_id,
        "speaker_name": speaker_name,
    }
    with journal_path.open("a") as f:
        f.write(_json.dumps(correction) + "\n")

    return JSONResponse(
        {"status": "updated", "segment_id": segment_id, "speaker_name": speaker_name}
    )


@app.get("/api/meetings/{meeting_id}/speakers")
async def get_meeting_speakers(meeting_id: str) -> JSONResponse:
    """Get detected speakers for a meeting."""
    speakers = storage.load_detected_speakers(meeting_id)
    return JSONResponse({"speakers": speakers})


@app.get("/api/meetings/{meeting_id}/tts/{segment_id}")
async def get_tts_audio(meeting_id: str, segment_id: str) -> fastapi.responses.Response:
    """Get synthesized TTS audio for a specific segment."""
    from fastapi.responses import FileResponse

    tts_path = storage._meetings_dir / meeting_id / "tts" / f"{segment_id}.wav"
    if not tts_path.exists():
        return JSONResponse({"error": "No TTS audio for this segment"}, status_code=404)
    return FileResponse(str(tts_path), media_type="audio/wav")


@app.get("/api/meetings/{meeting_id}/audio")
async def get_meeting_audio(
    meeting_id: str, request: fastapi.Request
) -> fastapi.responses.Response:
    """Stream meeting audio as WAV for a time range.

    Query params: start_ms (default 0), end_ms (default full duration).
    Returns audio/wav with proper headers for browser <audio> element.
    """
    import struct as _struct

    from fastapi.responses import StreamingResponse

    meeting_dir = storage._meetings_dir / meeting_id / "audio" / "recording.pcm"
    if not meeting_dir.exists():
        return JSONResponse({"error": "No audio recording"}, status_code=404)

    start_ms = int(request.query_params.get("start_ms", 0))
    end_ms_param = request.query_params.get("end_ms")
    total_duration = storage.audio_duration_ms(meeting_id)
    end_ms = int(end_ms_param) if end_ms_param else total_duration

    # Clamp
    start_ms = max(0, min(start_ms, total_duration))
    end_ms = max(start_ms, min(end_ms, total_duration))

    pcm_data = storage.read_audio_segment(meeting_id, start_ms, end_ms)
    if not pcm_data:
        return JSONResponse({"error": "No audio in range"}, status_code=404)

    # Build WAV header
    sample_rate = 16000
    channels = 1
    bits_per_sample = 16
    data_size = len(pcm_data)
    header = _struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        sample_rate * channels * bits_per_sample // 8,
        channels * bits_per_sample // 8,
        bits_per_sample,
        b"data",
        data_size,
    )

    def stream():
        yield header
        # Stream PCM in 8KB chunks
        for i in range(0, len(pcm_data), 8192):
            yield pcm_data[i : i + 8192]

    return StreamingResponse(
        stream(),
        media_type="audio/wav",
        headers={
            "Content-Length": str(44 + data_size),
            "Accept-Ranges": "bytes",
        },
    )


@app.get("/api/meetings/{meeting_id}/timeline")
async def get_timeline(meeting_id: str) -> JSONResponse:
    """Get the timeline manifest for podcast player."""
    import json as _json

    path = storage._meetings_dir / meeting_id / "timeline.json"
    if not path.exists():
        return JSONResponse({"error": "No timeline"}, status_code=404)
    return JSONResponse(_json.loads(path.read_text()))


@app.delete("/api/meetings/{meeting_id}")
async def delete_meeting(meeting_id: str) -> JSONResponse:
    """Delete a meeting and all its artifacts."""
    import shutil

    meeting_dir = storage._meetings_dir / meeting_id
    if not meeting_dir.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    # Don't delete the currently active meeting
    if current_meeting and current_meeting.meeting_id == meeting_id:
        return JSONResponse({"error": "Cannot delete active meeting"}, status_code=400)

    shutil.rmtree(meeting_dir)
    logger.info("Deleted meeting: %s", meeting_id)
    return JSONResponse({"status": "deleted", "meeting_id": meeting_id})


@app.get("/api/meetings/{meeting_id}")
async def get_meeting(meeting_id: str) -> JSONResponse:
    """Get a specific meeting's full data (meta + transcript + room layout)."""
    import json as _json

    meeting_dir = storage._meetings_dir / meeting_id
    if not meeting_dir.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    result = {}

    # Meta
    meta_path = meeting_dir / "meta.json"
    if meta_path.exists():
        result["meta"] = _json.loads(meta_path.read_text())

    # Room layout
    room_path = meeting_dir / "room.json"
    if room_path.exists():
        result["room"] = _json.loads(room_path.read_text())

    # Transcript events (last 500)
    journal_path = meeting_dir / "journal.jsonl"
    events = []
    if journal_path.exists():
        for line in journal_path.open():
            line = line.strip()
            if line:
                with suppress(Exception):
                    events.append(_json.loads(line))
    result["events"] = events[-500:]
    result["total_events"] = len(events)

    return JSONResponse(result)


# ── Room Setup & Speaker Enrollment ────────────────────────────


@app.get("/api/room/layout")
async def get_room_layout() -> JSONResponse:
    """Get the current draft room layout."""
    return JSONResponse(draft_layout.model_dump())


@app.put("/api/room/layout")
async def put_room_layout(request: fastapi.Request) -> JSONResponse:
    """Update the draft room layout."""
    global draft_layout
    try:
        body = await request.json()
        draft_layout = RoomLayout.model_validate(body)
        return JSONResponse({"status": "ok"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=422)


@app.get("/api/room/presets")
async def get_presets() -> JSONResponse:
    """List available room layout presets."""
    return JSONResponse(
        {
            "presets": [
                {
                    "id": "boardroom",
                    "label": "Boardroom",
                    "description": "Oval table, seats around",
                },
                {
                    "id": "round",
                    "label": "Round Table",
                    "description": "Circular table, seats around",
                },
                {
                    "id": "square",
                    "label": "Square Table",
                    "description": "Square table, seats around",
                },
                {
                    "id": "rectangle",
                    "label": "Rectangle",
                    "description": "Rectangular table, seats around",
                },
                {
                    "id": "classroom",
                    "label": "Classroom",
                    "description": "Front desk, seats in rows",
                },
                {"id": "u_shape", "label": "U-Shape", "description": "U-shaped table arrangement"},
                {"id": "pods", "label": "Pods", "description": "Small group tables"},
                {
                    "id": "freeform",
                    "label": "Free-form",
                    "description": "No table, place seats freely",
                },
            ],
        }
    )


@app.get("/api/room/speakers")
async def get_speakers() -> JSONResponse:
    """Get all enrolled speakers."""
    speakers = enrollment_store.speakers
    return JSONResponse(
        {
            "speakers": [
                {
                    "enrollment_id": s.enrollment_id,
                    "name": s.name,
                    "audio_duration_seconds": s.audio_duration_seconds,
                }
                for s in speakers.values()
            ],
        }
    )


@app.post("/api/room/enroll")
async def enroll_speaker(request: fastapi.Request) -> JSONResponse:
    """Enroll a speaker from audio.

    Accepts raw s16le 16kHz PCM audio in body.
    Transcribes the audio to extract the speaker's name,
    then extracts a voice embedding for later identification.
    Optional ?name= query param overrides ASR-detected name.
    """
    import uuid

    import numpy as np

    body = await request.body()
    if len(body) < 16000:
        return JSONResponse({"error": "Need at least 0.5s of audio"}, status_code=400)

    audio = np.frombuffer(body, dtype=np.int16).astype(np.float32) / 32768.0
    duration = len(audio) / 16000

    # Transcribe audio to extract name
    name = request.query_params.get("name", "").strip()
    if not name:
        try:
            name = await _transcribe_name(audio)
        except Exception as e:
            logger.warning("Name transcription failed: %s", e)
            name = f"Speaker {len(enrollment_store.speakers) + 1}"

    if not name or len(name) < 1:
        name = f"Speaker {len(enrollment_store.speakers) + 1}"

    # Extract voice embedding
    try:
        embedding = await _extract_embedding(audio)
    except Exception as e:
        logger.warning("Embedding extraction failed: %s", e)
        embedding = _simple_embedding(audio)

    enrollment_id = str(uuid.uuid4())[:8]
    speaker = EnrolledSpeaker(
        name=name,
        enrollment_id=enrollment_id,
        embedding=embedding,
        audio_duration_seconds=duration,
    )
    enrollment_store.add(speaker)

    logger.info("Enrolled '%s' (id=%s, %.1fs audio)", name, enrollment_id, duration)
    return JSONResponse(
        {
            "enrollment_id": enrollment_id,
            "name": name,
            "audio_duration_seconds": round(duration, 1),
        }
    )


@app.post("/api/room/enroll/rename")
async def rename_speaker(request: fastapi.Request) -> JSONResponse:
    """Rename an enrolled speaker."""
    eid = request.query_params.get("id", "")
    name = request.query_params.get("name", "").strip()
    if not eid or not name:
        return JSONResponse({"error": "id and name required"}, status_code=400)

    speakers = enrollment_store.speakers
    if eid in speakers:
        speaker = speakers[eid]
        speaker.name = name
        enrollment_store.add(speaker)  # re-add to persist
        return JSONResponse({"status": "renamed", "name": name})
    return JSONResponse({"error": "not found"}, status_code=404)


@app.delete("/api/room/speakers/{enrollment_id}")
async def remove_speaker(enrollment_id: str) -> JSONResponse:
    """Remove an enrolled speaker."""
    if enrollment_store.remove(enrollment_id):
        return JSONResponse({"status": "removed"})
    return JSONResponse({"error": "not found"}, status_code=404)


async def _transcribe_name(audio: np.ndarray) -> str:
    """Transcribe enrollment audio to extract the speaker's name.

    Uses mlx-whisper directly for a quick one-shot transcription.
    The speaker is expected to say something like "Hi, my name is Brad"
    or just "Brad" — we extract the name from the transcription.
    """
    import mlx_whisper

    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo="mlx-community/whisper-medium",
        language="en",  # enrollment is typically in the dominant language
        no_speech_threshold=0.5,
        condition_on_previous_text=False,
    )
    text = result.get("text", "").strip()
    logger.info("Enrollment transcription: '%s'", text)

    if not text:
        return ""

    # Extract name from common patterns:
    # "Hi, my name is Brad" → "Brad"
    # "I'm Brad Lay" → "Brad Lay"
    # "Brad" → "Brad"
    # "こんにちは、ブラッドです" → "ブラッド"
    lower = text.lower()
    for prefix in [
        "my name is ",
        "i'm ",
        "i am ",
        "this is ",
        "hey, i'm ",
        "hi, i'm ",
        "hi, my name is ",
    ]:
        if prefix in lower:
            idx = lower.index(prefix) + len(prefix)
            name = text[idx:].strip().rstrip(".")
            return name

    # Japanese patterns: ~desu
    if "です" in text:
        # "ブラッドです" → "ブラッド"
        name = text.split("です")[0].strip()
        # Remove common prefixes
        for jp_prefix in ["私は", "僕は", "わたしは"]:
            if name.startswith(jp_prefix):
                name = name[len(jp_prefix) :]
        return name.strip()

    # Fallback: use the whole transcription as the name (first 20 chars)
    # Strip common filler
    for filler in ["hello", "hi", "hey", "um", "uh", "so", "okay", "well"]:
        if lower.startswith(filler):
            text = text[len(filler) :].strip().lstrip(",").lstrip(".").strip()

    return text[:20].strip().rstrip(".")


async def _extract_embedding(audio: np.ndarray) -> np.ndarray:
    """Extract speaker embedding using SpeechBrain ECAPA-TDNN."""

    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/speechbrain-ecapa",
        )
        waveform = torch.from_numpy(audio).unsqueeze(0)
        embedding = classifier.encode_batch(waveform).squeeze().numpy()
        return embedding
    except ImportError:
        logger.warning("SpeechBrain not installed — using simple embedding")
        return _simple_embedding(audio)


def _simple_embedding(audio: np.ndarray) -> np.ndarray:
    """Fallback: extract simple audio features as a pseudo-embedding.

    Uses MFCCs as a basic voice fingerprint. Not as good as ECAPA-TDNN
    but works without SpeechBrain.
    """
    import numpy as np

    # Compute basic spectral features
    n_fft = 512
    # Simple energy + zero-crossing based features
    frame_size = 16000  # 1s frames
    n_frames = max(1, len(audio) // frame_size)
    features = []
    for i in range(n_frames):
        frame = audio[i * frame_size : (i + 1) * frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        rms = float(np.sqrt(np.mean(frame**2)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(frame)))))
        # Simple FFT features (16 bins)
        fft = np.abs(np.fft.rfft(frame, n=n_fft))[: n_fft // 2]
        spectral = np.array([np.mean(fft[j::16]) for j in range(16)])
        features.append(np.concatenate([[rms, zcr], spectral]))

    embedding = np.mean(features, axis=0).astype(np.float32)
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


# ── Meeting Lifecycle ──────────────────────────────────────────


@app.post("/api/meeting/start")
async def start_meeting() -> JSONResponse:
    """Start a new meeting, or return the current one if already recording."""
    global current_meeting

    if current_meeting and current_meeting.state == MeetingState.RECORDING:
        return JSONResponse(
            {
                "meeting_id": current_meeting.meeting_id,
                "state": "recording",
                "resumed": True,
            }
        )

    # Stop any previous meeting
    if current_meeting and current_meeting.state in (MeetingState.CREATED, MeetingState.RECORDING):
        with suppress(Exception):
            storage.transition_state(current_meeting.meeting_id, MeetingState.INTERRUPTED)

    # Close any previous audio writer
    global audio_writer, meeting_start_time
    if audio_writer:
        audio_writer.close()
        audio_writer = None

    # Create new meeting and persist room setup
    meta = storage.create_meeting()

    # Persist draft room layout + enrolled speakers into meeting directory
    storage.save_room_layout(meta.meeting_id, draft_layout)
    meeting_dir = storage._meeting_dir(meta.meeting_id)
    speakers_path = meeting_dir / "speakers.json"
    enrollment_store._storage_path = speakers_path
    enrollment_store._persist()
    logger.info(
        "Persisted room layout (%d seats) + speakers to %s", len(draft_layout.seats), meeting_dir
    )

    storage.transition_state(meta.meeting_id, MeetingState.RECORDING)
    current_meeting = meta
    current_meeting.state = MeetingState.RECORDING
    metrics.reset()
    metrics.meeting_start = time.monotonic()

    # Open audio writer for recording
    audio_writer = storage.open_audio_writer(meta.meeting_id)
    meeting_start_time = time.monotonic()

    # Initialize speaker tracking
    global detected_speakers, speaker_verifier
    detected_speakers = []
    from meeting_scribe.speaker.verification import SpeakerVerifier

    speaker_verifier = SpeakerVerifier(enrollment_store)

    logger.info("Audio recording started for %s", meta.meeting_id)

    logger.info("Meeting started: %s", meta.meeting_id)
    return JSONResponse(
        {
            "meeting_id": meta.meeting_id,
            "state": "recording",
            "resumed": False,
        }
    )


@app.post("/api/meeting/stop")
async def stop_meeting() -> JSONResponse:
    """Stop the current meeting."""
    global current_meeting

    if not current_meeting:
        return JSONResponse({"error": "No active meeting"}, status_code=400)

    mid = current_meeting.meeting_id

    # Flush ASR — finalizes all pending lines
    if asr_backend:
        async for event in asr_backend.flush():
            await _process_event(event)

    # Flush translation merge gate and wait for pending translations
    if translation_queue:
        await translation_queue.flush_merge_gate()
        logger.info("Waiting for pending translations...")
        for _i in range(100):  # up to 10s
            if translation_queue.is_idle():
                break
            await asyncio.sleep(0.1)
        else:
            logger.warning("Translation queue still busy after 10s — proceeding with stop")

    storage.flush_journal(mid)

    # Close audio writer
    global audio_writer, meeting_start_time
    if audio_writer:
        duration_ms = audio_writer.duration_ms
        audio_writer.close()
        audio_writer = None
        meeting_start_time = 0.0
        logger.info("Audio recording: %dms (%.1fs)", duration_ms, duration_ms / 1000)

    # Save detected speakers
    if detected_speakers:
        storage.save_detected_speakers(mid, detected_speakers)
        logger.info("Saved %d detected speakers", len(detected_speakers))

    # Generate timeline.json for podcast player
    _generate_timeline(mid)

    try:
        storage.transition_state(mid, MeetingState.FINALIZING)
        storage.transition_state(mid, MeetingState.COMPLETE)
    except Exception as e:
        logger.warning("Stop transition error: %s", e)

    # Close all websockets
    for ws in list(ws_connections):
        with suppress(Exception):
            await ws.close(1000, "Meeting ended")
    ws_connections.clear()

    current_meeting = None
    logger.info("Meeting stopped: %s", mid)
    return JSONResponse({"status": "complete", "meeting_id": mid})


def _generate_timeline(meeting_id: str) -> None:
    """Generate timeline.json from journal for podcast player."""
    import json as _json

    journal_path = storage._meeting_dir(meeting_id) / "journal.jsonl"
    timeline_path = storage._meeting_dir(meeting_id) / "timeline.json"

    segments = []
    if journal_path.exists():
        for line in journal_path.open():
            line = line.strip()
            if not line:
                continue
            try:
                e = _json.loads(line)
                if e.get("text") and e.get("is_final"):
                    segments.append(
                        {
                            "start_ms": e.get("start_ms", 0),
                            "end_ms": e.get("end_ms", 0),
                            "language": e.get("language", "unknown"),
                            "speaker_id": (e.get("speakers") or [{}])[0].get("identity")
                            if e.get("speakers")
                            else None,
                            "text": e.get("text", "")[:100],
                        }
                    )
            except Exception:
                continue

    duration_ms = storage.audio_duration_ms(meeting_id)
    timeline = {"duration_ms": duration_ms, "segments": segments}
    timeline_path.write_text(_json.dumps(timeline, indent=2))
    logger.info("Generated timeline.json: %d segments, %dms", len(segments), duration_ms)


# --- WebSocket (no auth for POC localhost) ---


@app.websocket("/api/ws")
async def websocket_audio(websocket: WebSocket) -> None:
    """WebSocket for audio streaming and transcript events.

    No auth required for POC (localhost only).
    Sends binary audio, receives JSON transcript events.
    """
    await websocket.accept()
    ws_connections.add(websocket)
    logger.info("WS connected (total=%d)", len(ws_connections))

    try:
        while True:
            data = await websocket.receive_bytes()
            await _handle_audio(data)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("WS error: %s", e)
    finally:
        ws_connections.discard(websocket)
        logger.info("WS disconnected (remaining=%d)", len(ws_connections))


async def _handle_audio(data: bytes) -> None:
    """Feed raw s16le 16kHz PCM from AudioWorklet to ASR backend."""
    if len(data) < 2:
        return

    metrics.audio_chunks += 1
    # s16le: 2 bytes per sample, 16kHz
    num_samples = len(data) // 2
    metrics.audio_seconds += num_samples / 16000

    # Log audio level periodically
    if metrics.audio_chunks % 40 == 1:
        import numpy as np

        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
        logger.info(
            "Audio chunk #%d: %d samples, peak=%.4f",
            metrics.audio_chunks,
            num_samples,
            peak,
        )

    # Write audio to recording file (time-aligned)
    if audio_writer and meeting_start_time > 0:
        elapsed_ms = int((time.monotonic() - meeting_start_time) * 1000)
        audio_writer.write_at(data, elapsed_ms)

    if not asr_backend:
        return

    # Send raw s16le bytes directly to backend
    await asr_backend.process_audio_bytes(data)


def _extract_name_from_text(text: str) -> str | None:
    """Detect self-introductions in transcript text and extract the name.

    Patterns: "my name is Brad", "I'm Brad", "this is Brad speaking",
    "ブラッドです", "私はブラッドです", etc.
    """
    import re

    text.lower().strip()

    # English patterns — capture only the name (one or two capitalized words)
    # Common words that get false-positive matched as names
    _NOT_NAMES = {
        "the",
        "a",
        "an",
        "it",
        "so",
        "just",
        "very",
        "really",
        "here",
        "there",
        "not",
        "going",
        "ready",
        "happy",
        "sorry",
        "sure",
        "glad",
        "excited",
        "able",
        "trying",
        "looking",
        "working",
        "coming",
        "leaving",
        "done",
        "fine",
        "good",
        "great",
        "well",
        "okay",
        "back",
        "still",
        "also",
        "about",
        "from",
        "with",
        "your",
        "this",
        "that",
        "what",
        "when",
        "all",
        "new",
        "old",
        "big",
        "now",
        "like",
        "only",
    }

    # Only match explicit "my name is X" — NOT "I'm X" (too many false positives)
    for pattern in [
        r"my name is ([A-Z][a-z]+)",
        r"(?:hi|hello),?\s*my name is ([A-Z][a-z]+)",
        r"call me ([A-Z][a-z]+)",
        r"this is ([A-Z][a-z]+) speaking",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if name.lower() not in _NOT_NAMES and len(name) >= 2:
                return name

    # Japanese patterns — ONLY match explicit self-introductions
    # "私は田中です" or "田中です" (short name + です at end of string)
    import re as _re

    ja_patterns = [
        r"私は(.{1,8})です[。]?$",
        r"僕は(.{1,8})です[。]?$",
        r"わたしは(.{1,8})です[。]?$",
        r"^(.{1,6})です[。]?$",  # Only match if the ENTIRE text is "Xです"
    ]
    for pat in ja_patterns:
        m = _re.search(pat, text.strip())
        if m:
            name = m.group(1).strip()
            # Filter: name should be short (real names are 1-6 chars in JA)
            # and should NOT contain particles/common words
            if 1 <= len(name) <= 6 and not any(
                w in name
                for w in ["の", "に", "を", "は", "が", "で", "と", "も", "から", "まで", "より"]
            ):
                return name

    return None


async def _process_event(event: TranscriptEvent) -> None:
    """Store event, broadcast to WS clients, queue translation."""
    metrics.asr_events += 1
    metrics.last_asr_event_time = time.monotonic()
    if event.is_final:
        metrics.asr_finals += 1
    else:
        metrics.asr_partials += 1

    # Speaker matching: compare audio chunk against enrolled speaker embeddings
    if event.is_final and event.text and enrollment_store.speakers:
        try:
            audio_chunk = getattr(asr_backend, "last_audio_chunk", None)
            if audio_chunk is not None and len(audio_chunk) > 0:
                # Run embedding extraction off the event loop
                import asyncio
                from concurrent.futures import ThreadPoolExecutor

                _speaker_executor = getattr(_process_event, "_executor", None)
                if _speaker_executor is None:
                    _speaker_executor = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="speaker"
                    )
                    _process_event._executor = _speaker_executor

                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(
                    _speaker_executor, _simple_embedding, audio_chunk
                )

                best_name = None
                best_score = 0.0
                best_eid = None
                from meeting_scribe.speaker.verification import cosine_similarity

                for eid, name, enrolled_emb in enrollment_store.get_all_embeddings():
                    score = cosine_similarity(embedding, enrolled_emb)
                    if score > best_score:
                        best_score = score
                        best_name = name
                        best_eid = eid

                if best_score > 0.3 and best_name:
                    event = event.model_copy(
                        update={
                            "speakers": [
                                SpeakerAttribution(
                                    cluster_id=0,
                                    identity=best_name,
                                    identity_confidence=best_score,
                                    source="enrolled",
                                )
                            ],
                        }
                    )
                    existing = [s for s in detected_speakers if s.speaker_id == best_eid]
                    if existing:
                        existing[0].segment_count += 1
                        existing[0].last_seen_ms = event.end_ms
                    else:
                        detected_speakers.append(
                            DetectedSpeaker(
                                speaker_id=best_eid or "",
                                display_name=best_name,
                                matched_enrollment_id=best_eid,
                                match_confidence=best_score,
                                segment_count=1,
                                first_seen_ms=event.start_ms,
                                last_seen_ms=event.end_ms,
                            )
                        )
        except Exception as e:
            logger.debug("Speaker matching error: %s", e)

    # Auto-detect self-introductions + propagate known speaker
    if event.is_final and event.text:
        detected_name = _extract_name_from_text(event.text)
        if detected_name:
            # Only register NEW names — don't overwrite existing assignments
            already_known = any(
                s.display_name.lower() == detected_name.lower() for s in detected_speakers
            )
            if not already_known:
                detected_speakers.append(
                    DetectedSpeaker(
                        display_name=detected_name,
                        segment_count=1,
                        first_seen_ms=event.start_ms,
                        last_seen_ms=event.end_ms,
                    )
                )
                logger.info("New speaker detected: '%s'", detected_name)

                # Assign to first UNNAMED seat only (don't overwrite named seats)
                for seat in draft_layout.seats:
                    if not seat.speaker_name:
                        seat.speaker_name = detected_name
                        logger.info("Auto-assigned '%s' to seat %s", detected_name, seat.seat_id)
                        import json as _json

                        seat_update = _json.dumps(
                            {
                                "type": "seat_update",
                                "seat_id": seat.seat_id,
                                "speaker_name": detected_name,
                            }
                        )
                        for ws in ws_connections:
                            with suppress(Exception):
                                task = asyncio.create_task(ws.send_text(seat_update))
                                _background_tasks.add(task)
                                task.add_done_callback(_background_tasks.discard)
                        break

            # Tag this segment with the speaker
            event = event.model_copy(
                update={
                    "speakers": [
                        SpeakerAttribution(
                            cluster_id=0,
                            identity=detected_name,
                            identity_confidence=0.9,
                            source="self_introduction",
                        )
                    ]
                }
            )

        # Without diarization, we can't propagate names — different speakers
        # would all get the same name. GB10 Sortformer will fix this.

    if current_meeting:
        storage.append_event(current_meeting.meeting_id, event)

    await _broadcast(event)

    if event.is_final and translation_queue:
        metrics.translations_submitted += 1
        logger.info(
            "Submitting for translation: seg=%s lang=%s text='%s'",
            event.segment_id[:8],
            event.language,
            event.text[:60],
        )
        await translation_queue.submit(event)
        # Flush merge gate immediately — don't hold segments waiting for merge
        await translation_queue.flush_merge_gate()


async def _broadcast(event: TranscriptEvent) -> None:
    """Send event to all connected WebSocket clients."""
    data = event.model_dump_json()
    dead: list[WebSocket] = []
    for ws in ws_connections:
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_connections.discard(ws)


async def _broadcast_translation(event: TranscriptEvent) -> None:
    """Callback from translation queue — track metrics and broadcast."""
    if event.translation:
        status = event.translation.status.value
        if status == "done":
            metrics.translations_completed += 1
        elif status == "failed":
            metrics.translations_failed += 1

    # Broadcast to all clients (includes in_progress, done, failed, skipped)
    await _broadcast(event)

    # TTS: synthesize translated text in speaker's voice
    if (
        tts_backend
        and tts_backend.available
        and event.translation
        and event.translation.status.value == "done"
        and event.translation.text
    ):
        try:
            # Get voice reference from speaker
            speaker = event.speakers[0] if event.speakers else None
            speaker_id = speaker.identity if speaker else "default"
            voice_ref = None

            # Cache voice from the audio chunk if available
            audio_chunk = getattr(asr_backend, "last_audio_chunk", None)
            if audio_chunk is not None and speaker_id:
                tts_backend.cache_voice(speaker_id, audio_chunk)
                if tts_backend.has_voice(speaker_id):
                    voice_ref = tts_backend._voice_cache.get(speaker_id)

            # Synthesize
            audio = await tts_backend.synthesize(
                event.translation.text,
                event.translation.target_language,
                voice_reference=voice_ref,
            )

            if len(audio) > 0:
                # Save TTS audio for playback
                if current_meeting:
                    tts_dir = storage._meeting_dir(current_meeting.meeting_id) / "tts"
                    tts_dir.mkdir(exist_ok=True)
                    sf_path = tts_dir / f"{event.segment_id}.wav"
                    import soundfile as _sf

                    _sf.write(str(sf_path), audio, 24000)

                # Notify clients that TTS audio is available
                import json as _json

                tts_msg = _json.dumps(
                    {
                        "type": "tts_audio",
                        "segment_id": event.segment_id,
                        "audio_url": f"/api/meetings/{current_meeting.meeting_id}/tts/{event.segment_id}"
                        if current_meeting
                        else "",
                    }
                )
                for ws in ws_connections:
                    with suppress(Exception):
                        await ws.send_text(tts_msg)
        except Exception as e:
            logger.debug("TTS synthesis failed: %s", e)


def main() -> None:
    """Entry point."""
    server_config = ServerConfig.from_env()
    uvicorn.run(
        "meeting_scribe.server:app",
        host="127.0.0.1",
        port=server_config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
