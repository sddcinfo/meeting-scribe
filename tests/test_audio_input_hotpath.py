from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import numpy as np
import pytest

from meeting_scribe.runtime import state
from meeting_scribe.ws import audio_input


class _IdentityResampler:
    def resample(self, source: np.ndarray, *, source_rate: int, target_rate: int) -> np.ndarray:
        return source


class _FakeWriter:
    def __init__(self) -> None:
        self.total_bytes = 0

    @property
    def current_offset(self) -> int:
        return self.total_bytes

    def write_at(self, pcm: bytes, elapsed_ms: int) -> None:
        self.total_bytes += len(pcm)


class _FastASR:
    def __init__(self) -> None:
        self.calls = 0

    async def process_audio_bytes(self, pcm: bytes, sample_offset: int | None = None) -> None:
        self.calls += 1


class _SlowDiarize:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.calls = 0

    async def process_audio(
        self,
        audio: np.ndarray,
        sample_offset: int,
        sample_rate: int = 16000,
    ) -> list:
        self.calls += 1
        self.started.set()
        await asyncio.sleep(0.2)
        return []


@pytest.mark.asyncio
async def test_audio_input_does_not_await_live_diarization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recording + ASR stay on the hot path; live diarization is backgrounded."""

    previous = {
        name: getattr(state, name)
        for name in (
            "asr_backend",
            "diarize_backend",
            "audio_writer",
            "current_meeting",
            "meeting_start_time",
            "resampler",
        )
    }
    previous_queue = audio_input._diarize_queue
    previous_worker = audio_input._diarize_worker
    previous_dropped = audio_input._diarize_dropped

    fake_asr = _FastASR()
    fake_diarize = _SlowDiarize()
    monkeypatch.setattr(state, "asr_backend", fake_asr)
    monkeypatch.setattr(state, "diarize_backend", fake_diarize)
    monkeypatch.setattr(state, "audio_writer", _FakeWriter())
    monkeypatch.setattr(
        state,
        "current_meeting",
        SimpleNamespace(meeting_id="hotpath-test", recording_started_epoch_ms=1),
    )
    monkeypatch.setattr(state, "meeting_start_time", time.monotonic())
    monkeypatch.setattr(state, "resampler", _IdentityResampler())
    monkeypatch.setattr(
        state,
        "storage",
        SimpleNamespace(_write_meta=lambda meeting: None),
        raising=False,
    )
    state.metrics.audio_chunks = 0
    state.metrics.audio_seconds = 0.0
    audio_input._diarize_queue = None
    audio_input._diarize_worker = None
    audio_input._diarize_dropped = 0

    try:
        pcm = (np.ones(4096, dtype=np.int16) * 1000).tobytes()
        payload = (16000).to_bytes(4, "little") + pcm

        started = time.monotonic()
        await asyncio.wait_for(audio_input._handle_audio(payload), timeout=0.05)
        elapsed = time.monotonic() - started

        assert elapsed < 0.05
        assert fake_asr.calls == 1
        await asyncio.wait_for(fake_diarize.started.wait(), timeout=0.1)
    finally:
        if audio_input._diarize_worker is not None:
            audio_input._diarize_worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await audio_input._diarize_worker
        audio_input._diarize_queue = previous_queue
        audio_input._diarize_worker = previous_worker
        audio_input._diarize_dropped = previous_dropped
        for name, value in previous.items():
            monkeypatch.setattr(state, name, value)
