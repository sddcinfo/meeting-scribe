"""End-to-end test for ``VllmASRBackend.replay_until_caught_up``.

Drives the **real** replay loop against a fixture ``recording.pcm`` in
a tmp meeting directory. The W6a/W6b recovery test suite mocks this
method (`tests/test_asr_recovery_replay.py`, `tests/test_recovery_supervisor.py`),
which let `state.storage.meeting_dir(...)` ship without anyone
catching that the public method is `get_meeting_dir(...)`. A whole
recovery cycle would raise AttributeError and silently lose
transcript audio.

This test exercises:
- The ``state.storage.get_meeting_dir(...)`` call site (regression
  guard for the AttributeError that triggered this fix).
- The PCM replay loop, including chunk sizing and the
  ``process_audio_bytes(_is_replay=True)`` submission contract.
- Termination when ``offset >= state.current_recording_pcm_offset()``.

CPU-only — no vLLM, no GPU, no network. ``process_audio_bytes`` is
spied with an AsyncMock so the test never reaches the HTTP submit
path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from meeting_scribe.backends.asr_vllm import VllmASRBackend
from meeting_scribe.runtime import state as runtime_state
from meeting_scribe.runtime.metrics import Metrics

# Test PCM is s16le @ 16 kHz, so byte_offset / 2 = sample_offset.
_BYTES_PER_SEC = 16000 * 2
# 30 s of zero-filled PCM is enough to exercise multiple chunked reads
# (the replay loop reads 3.5 s per step) without bloating fixtures.
_FIXTURE_DURATION_S = 30
_FIXTURE_BYTES = _BYTES_PER_SEC * _FIXTURE_DURATION_S


@dataclass
class _StubMeeting:
    """Minimal ``MeetingMeta``-shaped object — only ``meeting_id`` is
    read by ``replay_until_caught_up``."""

    meeting_id: str


class _StubStorage:
    """Subset of ``MeetingStorage`` that ``replay_until_caught_up``
    needs. Returns the per-test fixture meeting dir when asked."""

    def __init__(self, meeting_dir: Path) -> None:
        self._meeting_dir = meeting_dir

    def get_meeting_dir(self, meeting_id: str) -> Path:
        return self._meeting_dir


class _StubAudioWriter:
    """Mimics the ``AudioWriter.current_offset`` attribute that
    ``runtime_state.current_recording_pcm_offset()`` reads."""

    def __init__(self, current_offset: int) -> None:
        self.current_offset = current_offset


@pytest.fixture
def fixture_recording_pcm(tmp_path: Path) -> Path:
    """Build ``<tmp>/<meeting_id>/audio/recording.pcm`` with 30 s of
    zero-filled s16le @ 16 kHz audio. Returns the meeting directory
    path (parent of ``audio/``)."""
    meeting_id = "test-meeting-replay"
    meeting_dir = tmp_path / meeting_id
    audio_dir = meeting_dir / "audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / "recording.pcm").write_bytes(b"\x00" * _FIXTURE_BYTES)
    return meeting_dir


@pytest.fixture
def backend_with_state(monkeypatch, fixture_recording_pcm: Path) -> VllmASRBackend:
    """A ``VllmASRBackend`` wired against fixture state: a stub meeting
    pointing at the tmp recording, a stub storage returning the meeting
    dir, and an audio writer head past the end of the fixture so the
    replay loop traverses the full 30 s."""
    m = Metrics()
    monkeypatch.setattr(runtime_state, "metrics", m, raising=False)

    storage = _StubStorage(meeting_dir=fixture_recording_pcm)
    monkeypatch.setattr(runtime_state, "storage", storage, raising=False)

    monkeypatch.setattr(
        runtime_state,
        "current_meeting",
        _StubMeeting(meeting_id=fixture_recording_pcm.name),
        raising=False,
    )

    # Audio writer head sits at the end of the fixture so the loop
    # walks all 30 s and then breaks on `offset >= live_offset`.
    monkeypatch.setattr(
        runtime_state,
        "audio_writer",
        _StubAudioWriter(current_offset=_FIXTURE_BYTES),
        raising=False,
    )

    return VllmASRBackend()


@pytest.mark.asyncio
async def test_replay_walks_pcm_to_caught_up(backend_with_state: VllmASRBackend) -> None:
    """The fix for ``asr_vllm.py:352`` turns the AttributeError into a
    successful walk. This test would have failed (AttributeError) on
    every invocation before the fix; now it walks the 30 s of PCM and
    returns the live-head offset."""
    spy = AsyncMock()
    backend_with_state.process_audio_bytes = spy

    end_offset = await backend_with_state.replay_until_caught_up(start_offset=0)

    # Loop should terminate at exactly the live-head offset.
    assert end_offset == _FIXTURE_BYTES

    # 30 s of audio at 3.5 s per chunk = 9 calls (the last clamps to
    # `live_offset - offset`). Verifying the exact count locks the
    # chunk-cadence contract — but allow ±1 for arithmetic edge cases
    # since the cadence calc rounds.
    assert 8 <= spy.await_count <= 10

    # Every call must have ``_is_replay=True`` so the live-suppression
    # check skips the submission. This contract is the whole point of
    # ``replay_until_caught_up`` — dropping it would re-introduce the
    # double-transcribe bug.
    for call in spy.await_args_list:
        assert call.kwargs.get("_is_replay") is True


@pytest.mark.asyncio
async def test_replay_with_no_meeting_returns_offset_unchanged(
    monkeypatch, backend_with_state: VllmASRBackend
) -> None:
    """If ``state.current_meeting`` is None (no live meeting), the
    method short-circuits and returns the start offset. Locks the
    early-return path so a stale recovery call after meeting-end
    doesn't crash."""
    monkeypatch.setattr(runtime_state, "current_meeting", None, raising=False)
    spy = AsyncMock()
    backend_with_state.process_audio_bytes = spy

    end_offset = await backend_with_state.replay_until_caught_up(start_offset=12345)

    assert end_offset == 12345
    assert spy.await_count == 0


@pytest.mark.asyncio
async def test_replay_with_missing_pcm_file_returns_offset_unchanged(
    monkeypatch, fixture_recording_pcm: Path
) -> None:
    """If the meeting directory exists but ``recording.pcm`` is
    missing, the method logs a warning and returns the start offset.
    Locks the defensive path that protects against a meeting whose
    audio file was wiped (e.g. by ``finalize`` cleanup) being asked to
    replay."""
    # Remove the recording.pcm but keep the directory shape.
    (fixture_recording_pcm / "audio" / "recording.pcm").unlink()

    m = Metrics()
    monkeypatch.setattr(runtime_state, "metrics", m, raising=False)
    monkeypatch.setattr(
        runtime_state,
        "storage",
        _StubStorage(meeting_dir=fixture_recording_pcm),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_state,
        "current_meeting",
        _StubMeeting(meeting_id=fixture_recording_pcm.name),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_state,
        "audio_writer",
        _StubAudioWriter(current_offset=999_999),
        raising=False,
    )

    backend = VllmASRBackend()
    spy = AsyncMock()
    backend.process_audio_bytes = spy

    end_offset = await backend.replay_until_caught_up(start_offset=42)

    assert end_offset == 42
    assert spy.await_count == 0
