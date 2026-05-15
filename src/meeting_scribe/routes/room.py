"""Room layout + speaker enrollment routes.

Two route groups under one module because both back the room-setup
UI: speakers are added during enrollment with audio fingerprints,
and the room layout describes where the speakers sit so live
attribution can use proximity as a tiebreaker.

* ``/api/meetings/{id}/room`` and ``/api/meetings/{id}/room/layout``
  — per-meeting persisted layout. PUT broadcasts to live WS
  clients via ``server_support.broadcast`` when the meeting is
  active.

* ``/api/room/layout`` (GET/PUT) — session-scoped DRAFT layout used
  before a meeting starts. Backed by an in-memory dict keyed on a
  rotating session cookie so two browser tabs editing the same
  draft don't trample each other.

* ``/api/room/presets`` — static list of preset shapes for the
  room-setup UI dropdown.

* ``/api/room/speakers`` + ``/api/room/enroll`` family — speaker
  enrollment surface. The streaming-detection probe at
  ``enroll/detect-name`` is single-flight via a module-scope
  semaphore — concurrent probes return ``{"reason": "busy"}``
  immediately rather than queuing, since the polling clients retry
  in ~1.5 s and a queue depth blowup once killed the ASR backend
  during a reprocess storm.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path

import fastapi
import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from meeting_scribe.models import RoomLayout
from meeting_scribe.runtime import state
from meeting_scribe.server_support.broadcast import _broadcast_json
from meeting_scribe.server_support.safe_paths import _safe_meeting_dir
from meeting_scribe.server_support.sessions import (
    _get_draft_layout,
    _get_session_id,
    _set_draft_layout,
)
from meeting_scribe.server_support.speaker_audio import (
    _asr_transcribe,
    _extract_embedding,
    _simple_embedding,
    _transcribe_name,
)
from meeting_scribe.speaker.enrollment import EnrolledSpeaker
from meeting_scribe.speaker.name_extraction import (
    extract_name as _extract_name_from_text,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/meetings/{meeting_id}/room")
async def get_meeting_room(meeting_id: str) -> JSONResponse:
    """Get the persisted room layout for a meeting."""
    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir:
        return JSONResponse({"error": "Invalid meeting ID"}, status_code=400)
    layout = state.storage.load_room_layout(meeting_id)
    if layout is None:
        return JSONResponse({"error": "No room layout for this meeting"}, status_code=404)
    return JSONResponse(layout.model_dump())


@router.put("/api/meetings/{meeting_id}/room/layout")
async def put_meeting_room_layout(meeting_id: str, request: fastapi.Request) -> JSONResponse:
    """Persist an updated room layout for a meeting.

    Works for both active meetings (live edit) and past meetings (review edit).
    If the meeting is currently recording, broadcasts a room_layout_update
    event over WebSocket so all connected clients see the change.
    """
    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir:
        return JSONResponse({"error": "Invalid meeting ID"}, status_code=400)
    try:
        body = await request.json()
        layout = RoomLayout.model_validate(body)
    except Exception as e:
        return JSONResponse({"error": f"Invalid layout: {e}"}, status_code=422)

    state.storage.save_room_layout(meeting_id, layout)

    # Broadcast to all WS clients if this is the active meeting
    if state.current_meeting and state.current_meeting.meeting_id == meeting_id:
        await _broadcast_json(
            {
                "type": "room_layout_update",
                "layout": layout.model_dump(),
            }
        )

    return JSONResponse({"status": "ok", "layout": layout.model_dump()})


@router.get("/api/room/layout")
async def get_room_layout(request: fastapi.Request) -> JSONResponse:
    """Get the current draft room layout for this session."""
    sid = _get_session_id(request)
    layout = _get_draft_layout(sid)
    resp = JSONResponse(layout.model_dump())
    if "scribe_session" not in request.cookies:
        resp.set_cookie("scribe_session", sid, path="/", samesite="lax", httponly=False)
    return resp


@router.put("/api/room/layout")
async def put_room_layout(request: fastapi.Request) -> JSONResponse:
    """Update the draft room layout for this session."""
    sid = _get_session_id(request)
    try:
        body = await request.json()
        _set_draft_layout(sid, RoomLayout.model_validate(body))
        resp = JSONResponse({"status": "ok"})
        if "scribe_session" not in request.cookies:
            resp.set_cookie("scribe_session", sid, path="/", samesite="lax", httponly=False)
        return resp
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=422)


@router.get("/api/room/presets")
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


@router.get("/api/room/speakers")
async def get_speakers() -> JSONResponse:
    """Get all enrolled speakers."""
    speakers = state.enrollment_store.speakers
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


@router.post("/api/room/enroll")
async def enroll_speaker(request: fastapi.Request) -> JSONResponse:
    """Enroll a speaker from audio.

    Accepts raw s16le 16kHz PCM audio in body.
    Transcribes the audio to extract the speaker's name,
    then extracts a voice embedding for later identification.
    Optional ?name= query param overrides ASR-detected name.
    """
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
            name = f"Speaker {len(state.enrollment_store.speakers) + 1}"

    if not name or len(name) < 1:
        name = f"Speaker {len(state.enrollment_store.speakers) + 1}"

    # Extract voice embedding
    try:
        embedding = await _extract_embedding(audio)
    except Exception as e:
        logger.warning("Embedding extraction failed: %s", e)
        embedding = _simple_embedding(audio)

    enrollment_id = str(uuid.uuid4())

    # Stash the raw enrollment WAV — reused as the TTS voice-clone seed
    # at meeting start so participant-voice mode is ready from segment 0.
    ref_path = ""
    try:
        import wave

        enroll_dir = Path.home() / ".config" / "meeting-scribe" / "enrollments"
        enroll_dir.mkdir(parents=True, exist_ok=True)
        wav_path = enroll_dir / f"{enrollment_id}.wav"
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm.tobytes())
        ref_path = str(wav_path)
    except Exception as e:
        logger.warning("Could not persist enrollment WAV: %s", e)

    speaker = EnrolledSpeaker(
        name=name,
        enrollment_id=enrollment_id,
        embedding=embedding,
        audio_duration_seconds=duration,
        reference_wav_path=ref_path,
    )
    state.enrollment_store.add(speaker)

    logger.info("Enrolled '%s' (id=%s, %.1fs audio)", name, enrollment_id, duration)
    return JSONResponse(
        {
            "enrollment_id": enrollment_id,
            "name": name,
            "audio_duration_seconds": round(duration, 1),
        }
    )


# Single-flight guard for the name-detect probe. The streaming
# enrollment flow polls every ~1.5s, but multiple clients/tabs (and the
# occasional front-end retry storm) will pile concurrent ASR calls onto
# the same vLLM backend. We saw the ASR worker die after 14+ probes in
# 7 seconds — the reprocess pipeline + the name-detect storm together
# exceeded the backend's working-set budget.
#
# A semaphore caps in-flight probes: by default 2 concurrent ASR calls
# (matches the TTS replica count and gives the UX a 2x speedup on
# multi-speaker enrollment). Raise via SCRIBE_NAME_DETECT_CONCURRENCY env
# if the backend has more headroom; lower to 1 if you see storm-induced
# backend failures recur. Excess probes return immediately with "busy"
# rather than queue, so a client polling storm can't blow up queue depth.
_NAME_DETECT_CONCURRENCY = max(1, int(os.environ.get("SCRIBE_NAME_DETECT_CONCURRENCY", "2")))
_NAME_DETECT_SEMAPHORE = asyncio.Semaphore(_NAME_DETECT_CONCURRENCY)


@router.post("/api/room/enroll/detect-name")
async def enroll_detect_name(request: fastapi.Request) -> JSONResponse:
    """Probe enrollment audio for a self-stated name.

    Used by the streaming enrollment flow: the client records continuously
    and POSTs the accumulated buffer every ~1.5s. The server runs ASR on
    the buffer and only reports a name when a self-intro pattern actually
    fires. The client stops recording as soon as it receives a name, then
    calls /api/room/enroll?name=<name> with the final audio to extract the
    voice embedding.

    Single-flight: at most one ASR probe runs at a time. Concurrent probes
    return ``{"reason": "busy"}`` immediately so a polling storm can't
    overwhelm the vLLM ASR worker (which previously died mid-inference
    when the storm coincided with reprocess-driven load).
    """
    if _NAME_DETECT_SEMAPHORE.locked():
        return JSONResponse({"name": "", "text": "", "reason": "busy"})

    body = await request.body()
    # Need roughly ~0.5s of audio before bothering with ASR — shorter clips
    # can work for short names but rarely produce usable transcriptions.
    # The name extraction patterns are conservative enough to reject garbage.
    if len(body) < 16000:
        return JSONResponse({"name": "", "text": "", "reason": "too_short"})

    audio = np.frombuffer(body, dtype=np.int16).astype(np.float32) / 32768.0
    async with _NAME_DETECT_SEMAPHORE:
        try:
            text = await _asr_transcribe(audio)
        except Exception as e:
            logger.warning("Name detection probe failed: %s", e)
            return JSONResponse({"name": "", "text": "", "reason": "asr_error"})

    # _extract_name_from_text returns None unless a high-confidence
    # self-intro pattern matches, which is exactly what we want for
    # probe-based streaming detection.
    name = _extract_name_from_text(text) if text else None
    if not name:
        return JSONResponse({"name": "", "text": text or "", "reason": "no_pattern_match"})

    return JSONResponse(
        {
            "name": name,
            "text": text,
            "duration_seconds": round(len(audio) / 16000, 2),
        }
    )


@router.post("/api/room/enroll/rename")
async def rename_speaker(request: fastapi.Request) -> JSONResponse:
    """Rename an enrolled speaker."""
    eid = request.query_params.get("id", "")
    name = request.query_params.get("name", "").strip()
    if not eid or not name:
        return JSONResponse({"error": "id and name required"}, status_code=400)

    speakers = state.enrollment_store.speakers
    if eid in speakers:
        speaker = speakers[eid]
        speaker.name = name
        state.enrollment_store.add(speaker)  # re-add to persist
        return JSONResponse({"status": "renamed", "name": name})
    return JSONResponse({"error": "not found"}, status_code=404)


@router.delete("/api/room/speakers/{enrollment_id}")
async def remove_speaker(enrollment_id: str) -> JSONResponse:
    """Remove an enrolled speaker."""
    if state.enrollment_store.remove(enrollment_id):
        return JSONResponse({"status": "removed"})
    return JSONResponse({"error": "not found"}, status_code=404)
