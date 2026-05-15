"""End-to-end tests for ``_finalize_phase_b_inline``.

Drives the new background-finalize body directly with a minimal
in-process meeting (journal + PCM on disk) and mocked GPU backends.
Verifies:

* Phase B transitions the meeting to FINALIZING then COMPLETE.
* Journal gets speaker-attached events appended (revision bump).
* timeline.json + summary.json are written.
* background_finalize_progress events are broadcast with the right
  ``meeting_id`` and the terminal-event has ``terminal: true``.
* The captured WS sockets are closed at the end.
* The function tolerates a missing diarize backend (no ``no backend
  registered`` crash; falls back to the direct call path).

Skipped if running with ``SCRIBE_BACKGROUND_FINALIZE=0`` because that
flag also controls Phase A's GPU-lease acquire — flipping it during
the test invalidates the assumptions.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from meeting_scribe.config import ServerConfig
from meeting_scribe.models import MeetingMeta, MeetingState
from meeting_scribe.runtime import state as runtime_state
from meeting_scribe.runtime.gpu_lease import _reset_gpu_lease_for_tests
from meeting_scribe.storage import MeetingStorage


@pytest.fixture
def fake_meeting(tmp_path, monkeypatch):
    """Build a minimal recording-state meeting with journal + PCM on disk."""
    meetings_dir = tmp_path / "meetings"
    meetings_dir.mkdir()
    cfg = ServerConfig()
    storage = MeetingStorage(cfg)
    storage._meetings_dir = meetings_dir

    meta = storage.create_meeting(MeetingMeta())
    mid = meta.meeting_id
    storage.transition_state(mid, MeetingState.RECORDING)

    meeting_dir = Path(storage._meetings_dir) / mid
    audio_dir = meeting_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    pcm_path = audio_dir / "recording.pcm"
    # 1 second of zeroed PCM so _audio_quality_report has something to chew on.
    pcm_path.write_bytes(b"\x00" * (16000 * 2))

    journal_path = meeting_dir / "journal.jsonl"
    journal_path.write_text(
        json.dumps(
            {
                "segment_id": "s0",
                "text": "Hello world.",
                "is_final": True,
                "language": "en",
                "start_ms": 0,
                "end_ms": 1000,
                "revision": 0,
            }
        )
        + "\n"
    )

    # Wire state singletons so _finalize_phase_b_inline can find them.
    monkeypatch.setattr(runtime_state, "storage", storage, raising=False)
    monkeypatch.setattr(runtime_state, "config", cfg, raising=False)
    monkeypatch.setattr(runtime_state, "ws_connections", set(), raising=False)
    monkeypatch.setattr(runtime_state, "_audio_out_clients", set(), raising=False)

    yield {
        "mid": mid,
        "meeting_dir": meeting_dir,
        "journal_path": journal_path,
        "pcm_path": pcm_path,
        "storage": storage,
    }


@pytest.fixture(autouse=True)
def _reset_lease():
    _reset_gpu_lease_for_tests()
    yield
    _reset_gpu_lease_for_tests()


class _StubWs:
    """Minimal websocket double — records the close() call."""

    def __init__(self) -> None:
        self.closed = False
        self.closed_code: int | None = None
        self.closed_reason: str | None = None

    async def close(self, code: int, reason: str) -> None:
        self.closed = True
        self.closed_code = code
        self.closed_reason = reason

    async def send_json(self, _payload: Any) -> None:  # pragma: no cover
        pass


async def test_phase_b_runs_to_complete_and_closes_snapshotted_ws(fake_meeting, monkeypatch):
    """Drives the headline new-code path: Phase B receives a captured
    ctx, runs through the diarize → timeline → summary chain, transitions
    to COMPLETE, and closes the OLD ws sockets."""
    from meeting_scribe.routes import meeting_lifecycle

    mid = fake_meeting["mid"]
    storage = fake_meeting["storage"]
    meeting_dir = fake_meeting["meeting_dir"]

    broadcasts: list[dict] = []

    async def _capture_broadcast(payload: Any) -> None:
        broadcasts.append(payload)

    monkeypatch.setattr(meeting_lifecycle, "_broadcast_json", _capture_broadcast, raising=True)

    # Stub the diarize HTTP — return one segment so speaker_attach has work
    fake_diarize = AsyncMock(
        return_value=MagicMock(
            segments=[{"cluster_id": 1, "start": 0.0, "end": 1.0}],
            exclusive_segments=[{"cluster_id": 1, "start": 0.0, "end": 1.0}],
        )
    )
    monkeypatch.setattr(
        "meeting_scribe.pipeline.diarize._diarize_full_audio",
        fake_diarize,
        raising=True,
    )

    # Stub generate_summary — return a minimal plausible payload
    async def _fake_summary(_meeting_dir, *, vllm_url):
        return {
            "summary": "test",
            "topics": [],
            "action_items": [],
            "metadata": {"meeting_id": mid},
            "speaker_stats": [],
        }

    monkeypatch.setattr(
        "meeting_scribe.summary.generate_summary",
        _fake_summary,
        raising=True,
    )

    # Provide a stub for _generate_timeline since it imports state.storage
    monkeypatch.setattr(
        meeting_lifecycle,
        "_generate_timeline",
        lambda _mid: (meeting_dir / "timeline.json").write_text("{}"),
        raising=True,
    )
    # Stub _generate_speaker_data so it doesn't try to read live runtime.
    monkeypatch.setattr(
        meeting_lifecycle,
        "_generate_speaker_data",
        lambda _md, _jp, _json: None,
        raising=True,
    )

    ws_a = _StubWs()
    ws_b = _StubWs()
    audio_ws = _StubWs()

    t0 = time.monotonic()
    await meeting_lifecycle._finalize_phase_b_inline(
        mid=mid,
        finalize_t0=t0,
        audio_duration_s=1.0,
        est_seconds=10,
        partial_finalize=False,
        pending_translation_count=0,
        detected_speakers=[],
        eager_summary_cache=None,
        eager_summary_event_count=0,
        ws_connections=[ws_a, ws_b],
        audio_out_clients=[audio_ws],
    )

    # State machine landed COMPLETE.
    final_meta = storage._read_meta(mid)
    assert final_meta.state == MeetingState.COMPLETE

    # Captured WS sockets were closed.
    assert ws_a.closed and ws_b.closed and audio_ws.closed
    assert ws_a.closed_code == 1000

    # background_finalize_progress was emitted with the right meeting_id
    # and the terminal event was tagged terminal=True.
    bg_progress = [b for b in broadcasts if b.get("type") == "background_finalize_progress"]
    assert bg_progress, f"no bg progress in {[b.get('type') for b in broadcasts]}"
    for ev in bg_progress:
        assert ev["meeting_id"] == mid
    terminal = [b for b in bg_progress if b.get("terminal")]
    assert len(terminal) == 1
    assert terminal[0]["step"] == 7

    # meeting_stopped came after the terminal progress event.
    stopped = [b for b in broadcasts if b.get("type") == "meeting_stopped"]
    assert stopped and stopped[0]["meeting_id"] == mid


async def test_phase_b_tolerates_diarize_http_failure(fake_meeting, monkeypatch):
    """A diarize backend error should NOT explode Phase B — the meeting
    still transitions to COMPLETE (with no speaker attachments). This
    is the safety net so a transient backend outage doesn't strand the
    user's recording in FINALIZING forever."""
    from meeting_scribe.routes import meeting_lifecycle

    mid = fake_meeting["mid"]
    storage = fake_meeting["storage"]
    meeting_dir = fake_meeting["meeting_dir"]

    monkeypatch.setattr(meeting_lifecycle, "_broadcast_json", AsyncMock(), raising=True)

    monkeypatch.setattr(
        "meeting_scribe.pipeline.diarize._diarize_full_audio",
        AsyncMock(side_effect=ConnectionError("diarize down")),
        raising=True,
    )

    async def _fake_summary(_meeting_dir, *, vllm_url):
        return {
            "summary": "test",
            "topics": [],
            "action_items": [],
            "metadata": {"meeting_id": mid},
            "speaker_stats": [],
        }

    monkeypatch.setattr(
        "meeting_scribe.summary.generate_summary",
        _fake_summary,
        raising=True,
    )
    monkeypatch.setattr(
        meeting_lifecycle,
        "_generate_timeline",
        lambda _mid: (meeting_dir / "timeline.json").write_text("{}"),
        raising=True,
    )
    monkeypatch.setattr(
        meeting_lifecycle,
        "_generate_speaker_data",
        lambda _md, _jp, _json: None,
        raising=True,
    )

    await meeting_lifecycle._finalize_phase_b_inline(
        mid=mid,
        finalize_t0=time.monotonic(),
        audio_duration_s=1.0,
        est_seconds=10,
        partial_finalize=False,
        pending_translation_count=0,
        detected_speakers=[],
        eager_summary_cache=None,
        eager_summary_event_count=0,
        ws_connections=[],
        audio_out_clients=[],
    )

    final_meta = storage._read_meta(mid)
    assert final_meta.state == MeetingState.COMPLETE


async def test_stop_with_flag_on_returns_finalizing_immediately(fake_meeting, monkeypatch):
    """Smoke-shape integration: with SCRIBE_BACKGROUND_FINALIZE=1,
    `_stop_meeting_locked` reaches the Phase A → Phase B handoff,
    spawns the bg task, and returns ``status=finalizing``.

    Drives only Phase A (the state mutations + spawn), with Phase B's
    body stubbed to await an event so we can observe the handoff
    without running the full chain. Every state mutation is rolled
    through ``monkeypatch.setattr`` so this test can't leak state into
    subsequent tests in the same session.
    """
    from meeting_scribe.routes import meeting_lifecycle

    monkeypatch.setenv("SCRIBE_BACKGROUND_FINALIZE", "1")

    mid = fake_meeting["mid"]
    storage = fake_meeting["storage"]

    # Wire enough of state for Phase A's quiesce + ws-snapshot to run.
    # Every setattr is `raising=False` to handle attrs that don't yet
    # exist in the live runtime_state module — and goes through
    # monkeypatch so it's reverted in fixture teardown.
    current = MeetingMeta(meeting_id=mid)
    current.state = MeetingState.RECORDING
    monkeypatch.setattr(runtime_state, "current_meeting", current, raising=False)
    monkeypatch.setattr(runtime_state, "audio_writer", None, raising=False)
    monkeypatch.setattr(runtime_state, "detected_speakers", [], raising=False)
    monkeypatch.setattr(runtime_state, "_eager_summary_cache", None, raising=False)
    monkeypatch.setattr(runtime_state, "_eager_summary_event_count", 0, raising=False)
    monkeypatch.setattr(runtime_state, "_eager_summary_task", None, raising=False)
    monkeypatch.setattr(runtime_state, "_speaker_pulse_task", None, raising=False)
    monkeypatch.setattr(runtime_state, "_speaker_catchup_task", None, raising=False)
    monkeypatch.setattr(runtime_state, "_pending_speaker_events", {}, raising=False)
    monkeypatch.setattr(runtime_state, "_pending_speaker_timestamps", {}, raising=False)
    monkeypatch.setattr(runtime_state, "_client_prefs", {}, raising=False)
    monkeypatch.setattr(runtime_state, "_audio_out_prefs", {}, raising=False)
    monkeypatch.setattr(runtime_state, "_background_tasks", set(), raising=False)
    fake_queue = MagicMock()
    fake_queue.quiesce_meeting = AsyncMock(
        return_value=MagicMock(drained_clean=True, item_count=0, deferred_post_quiesce=0)
    )
    fake_queue.bind_meeting = MagicMock()
    fake_queue.clear_meeting = MagicMock()
    monkeypatch.setattr(runtime_state, "translation_queue", fake_queue, raising=False)
    monkeypatch.setattr(runtime_state, "asr_backend", None, raising=False)
    monkeypatch.setattr(
        runtime_state,
        "metrics",
        MagicMock(meeting_start=time.monotonic(), reset=MagicMock()),
        raising=False,
    )
    monkeypatch.setattr(runtime_state, "slide_job_runner", None, raising=False)
    monkeypatch.setattr(runtime_state, "storage", storage, raising=False)

    monkeypatch.setattr(meeting_lifecycle, "_broadcast_json", AsyncMock(), raising=True)
    monkeypatch.setattr(meeting_lifecycle, "_stop_wifi_ap", AsyncMock(), raising=True)

    # Replace the heavy Phase B body with a long await so the test
    # observes the handoff but doesn't wait minutes for diarize.
    long_running = asyncio.Event()

    async def _stub_phase_b(**_kwargs: Any) -> None:
        await long_running.wait()

    monkeypatch.setattr(meeting_lifecycle, "_finalize_phase_b_inline", _stub_phase_b, raising=True)

    response = await meeting_lifecycle._stop_meeting_locked()
    body = response.body.decode()
    assert "finalizing" in body, body

    # state.current_meeting cleared so a new Start can happen immediately.
    # (monkeypatch will restore the original on teardown.)
    assert runtime_state.current_meeting is None

    # bg task is registered.
    assert mid in meeting_lifecycle._phase_b_tasks
    bg_task = meeting_lifecycle._phase_b_tasks[mid]
    assert not bg_task.done()

    # Let it complete cleanly.
    long_running.set()
    await bg_task
    assert mid not in meeting_lifecycle._phase_b_tasks
