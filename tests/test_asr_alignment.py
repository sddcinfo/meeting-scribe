"""Regression tests for ASR transcript ↔ audio-file alignment.

Meeting bd1652db-2e11 (2026-04-13) surfaced a 6-minute alignment gap
caused by the ASR backend using an internal `_base_offset` counter
that reset to 0 on every meeting resume, stamping new events on top
of existing ones at start_ms=0, 1504, 3008…

The fix: callers pass an explicit `sample_offset` per chunk that
reflects the TRUE position of this audio in the meeting's PCM file,
and the backend uses it verbatim as the start of the current buffer.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from meeting_scribe.backends.asr_vllm import SAMPLE_RATE, VllmASRBackend
from meeting_scribe.models import TranscriptEvent


@pytest.fixture
def backend():
    b = VllmASRBackend(base_url="http://localhost:8003", buffer_seconds=1.5)
    b._client = MagicMock()
    b._model = "test-model"
    b.audio_wall_at_start = 0.0

    # Mock the vLLM chat completions response so emit path fires.
    async def fake_post(url, json=None):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(
            return_value={"choices": [{"message": {"content": "language English<asr_text>hello"}}]}
        )
        return resp

    b._client.post = AsyncMock(side_effect=fake_post)

    captured: list[TranscriptEvent] = []

    async def on_event(ev):
        captured.append(ev)

    b.set_event_callback(on_event)
    return b, captured


def _audio_chunk(ms: int) -> bytes:
    """Generate `ms` of non-silent s16le audio at 16 kHz."""
    n_samples = int(ms * SAMPLE_RATE / 1000)
    # Above VAD threshold (0.005 RMS) so ASR path actually emits
    samples = np.ones(n_samples, dtype=np.int16) * 2000
    return samples.tobytes()


class TestAbsoluteSampleOffset:
    async def test_caller_supplied_offset_drives_start_ms(self, backend):
        be, captured = backend
        # Send 2s of audio starting at sample offset 48000 (= 3000ms into file)
        await be.process_audio_bytes(_audio_chunk(2000), sample_offset=48000)
        assert captured, "expected an emitted event"
        assert captured[0].start_ms == 3000
        assert captured[0].end_ms == 5000  # 3000 + 2000

    async def test_second_chunk_advances_from_first(self, backend):
        be, captured = backend
        await be.process_audio_bytes(_audio_chunk(2000), sample_offset=0)
        await be.process_audio_bytes(_audio_chunk(2000), sample_offset=32000)
        assert len(captured) == 2
        assert captured[0].start_ms == 0
        assert captured[0].end_ms == 2000
        assert captured[1].start_ms == 2000
        assert captured[1].end_ms == 4000

    async def test_offset_jump_on_resume(self, backend):
        """Simulates a meeting resume: the audio file is at ~10 minutes
        when a new chunk arrives, and the ASR backend was just created
        fresh (so its internal counters are zero). The caller-supplied
        sample_offset must override the internal counter — otherwise
        the new transcript overlaps the original at t=0."""
        be, captured = backend
        ten_min_samples = SAMPLE_RATE * 600  # 10 min
        await be.process_audio_bytes(_audio_chunk(2000), sample_offset=ten_min_samples)
        assert captured
        assert captured[0].start_ms == 600_000
        assert captured[0].end_ms == 602_000


class TestFallbackToInternalCounter:
    async def test_no_offset_uses_internal_counter(self, backend):
        """Back-compat: callers that don't pass `sample_offset` still
        get monotonically-advancing timestamps (they just can't survive
        a restart)."""
        be, captured = backend
        await be.process_audio_bytes(_audio_chunk(2000))
        await be.process_audio_bytes(_audio_chunk(2000))
        assert len(captured) == 2
        assert captured[0].start_ms == 0
        assert captured[1].start_ms == 2000


class TestAudioWriterResumeTotalBytes:
    """The audio-file position is the source of truth for ASR alignment.
    After a server restart, the new AudioWriterProcess (opened in append
    mode) must report the EXISTING file size so sample_offset is right."""

    def test_append_mode_seeds_total_bytes(self, tmp_path):
        from meeting_scribe.storage import AudioWriterProcess

        path = tmp_path / "recording.pcm"
        # Pretend a previous run wrote 20 min of audio.
        twenty_min_bytes = 16000 * 2 * 60 * 20
        path.write_bytes(b"\x00" * twenty_min_bytes)

        w = AudioWriterProcess(path, append=True)
        # BEFORE start(), so we're testing __init__ seeding
        assert w.total_bytes == twenty_min_bytes

    def test_fresh_mode_zeros_total_bytes(self, tmp_path):
        from meeting_scribe.storage import AudioWriterProcess

        path = tmp_path / "fresh.pcm"
        path.write_bytes(b"\x00" * 1000)  # pre-existing junk

        w = AudioWriterProcess(path, append=False)
        # append=False means we're going to truncate, so counter starts at 0
        assert w.total_bytes == 0
