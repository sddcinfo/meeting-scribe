"""Background TTS voice-seeding from enrollment WAVs.

Called via ``asyncio.create_task`` from ``/api/meeting/start`` so the
HTTP response returns in ≈1 s instead of waiting for disk I/O and
per-speaker ``seed_voice`` calls. WAV reads and backend calls run on
the thread-pool executor so the event loop stays responsive for
incoming audio WebSocket frames.

A bounded ``Semaphore(2)`` keeps a room with many enrolled speakers
from stampeding the executor / TTS backend. Per-speaker errors are
logged but don't abort the whole seed — the participant simply falls
back to live-audio cloning on their first utterance (same path as a
never-enrolled speaker).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np

from meeting_scribe.runtime import state

logger = logging.getLogger(__name__)


def _read_enrollment_wav_f32(ref_path: str) -> np.ndarray:
    """Read a 16-bit PCM WAV file into a float32 [-1, 1] numpy array.

    Pure file I/O; runs on an executor thread via ``asyncio.to_thread``
    so the event loop stays free for audio WS frames during meeting
    start-up.
    """
    import wave

    with wave.open(ref_path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(frames, dtype=np.int16)
    return pcm.astype(np.float32) / 32767.0


async def _seed_tts_from_enrollments_async(speakers: list[tuple[str, Any]]) -> None:
    """Seed TTS voice cache from enrollment WAVs in the background."""
    if state.tts_backend is None or not hasattr(state.tts_backend, "seed_voice"):
        return
    sem = asyncio.Semaphore(2)

    async def _seed_one(eid: str, speaker: Any) -> None:
        ref_path = getattr(speaker, "reference_wav_path", "")
        if not ref_path:
            return
        path = Path(ref_path)
        if not path.exists():
            logger.info("TTS seed: enrollment '%s' ref wav missing at %s", eid, ref_path)
            return
        async with sem:
            try:
                audio = await asyncio.to_thread(_read_enrollment_wav_f32, str(path))
                await asyncio.to_thread(
                    state.tts_backend.seed_voice, eid, audio, source="enrollment"
                )
            except Exception as e:
                logger.warning("TTS seed failed for '%s': %s", eid, e)

    await asyncio.gather(*(_seed_one(eid, spk) for eid, spk in speakers))
