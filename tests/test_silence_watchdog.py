"""Phase-1 silence-watchdog tests — `last_nonzero_audio_ts` as the unified
signal for both "frames stopped arriving" AND "frames arriving but all-zero".

The 2026-05-14 demo failure surfaced this gap: ``state.last_audio_chunk_ts``
is bumped on every inbound frame regardless of content, so a microphone
captured 41 s of all-zero samples with no warning. Phase 1.4 pivots the
watchdog onto a peak-gated timestamp that only advances when a frame has
any non-zero sample.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from meeting_scribe.runtime import health_monitors, state


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    """Watchdog reads module-level state; reset it between tests."""
    state.current_meeting = None
    state.last_nonzero_audio_ts = 0.0
    state.last_audio_chunk_ts = 0.0
    state.silence_warn_sent = False
    if hasattr(state, "metrics"):
        state.metrics.meeting_start = 0.0
    yield
    state.current_meeting = None
    state.last_nonzero_audio_ts = 0.0
    state.last_audio_chunk_ts = 0.0
    state.silence_warn_sent = False


async def _step_watchdog(times: int = 1, interval_s: float = 0.001) -> None:
    """Run the watchdog body N times, fast.

    Replaces the module-level poll interval so each loop tick advances
    in milliseconds rather than seconds. Cancels after N iterations.
    """
    health_monitors.SILENCE_WATCHDOG_POLL_INTERVAL_S = interval_s
    task = asyncio.create_task(health_monitors.silence_watchdog_loop())
    # Give the loop enough wall-clock to complete ``times`` iterations.
    await asyncio.sleep(max(interval_s * times * 4, 0.02))
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_no_meeting_clears_warn_state(monkeypatch) -> None:
    """Without ``current_meeting`` the watchdog never fires."""
    broadcasts: list[dict] = []
    monkeypatch.setattr(
        health_monitors,
        "_broadcast_json",
        AsyncMock(side_effect=lambda payload: broadcasts.append(payload)),
    )
    state.current_meeting = None
    state.silence_warn_sent = True  # pretend a prior warn was sent
    await _step_watchdog()
    assert state.silence_warn_sent is False
    assert broadcasts == []


@pytest.mark.asyncio
async def test_fires_when_last_nonzero_is_stale(monkeypatch) -> None:
    """A meeting recording with samples arriving but RMS at the noise
    floor (``last_nonzero_audio_ts`` stays unset) trips the watchdog
    after the threshold."""
    broadcasts: list[dict] = []
    monkeypatch.setattr(
        health_monitors,
        "_broadcast_json",
        AsyncMock(side_effect=lambda payload: broadcasts.append(payload)),
    )
    # Threshold is 10 s in the production module. Compress it to make the
    # test fast.
    monkeypatch.setattr(health_monitors, "_SILENCE_WARN_THRESHOLD_S", 0.05)

    state.current_meeting = SimpleNamespace(meeting_id="m-1")
    state.metrics.meeting_start = time.monotonic() - 1.0  # started 1 s ago
    state.last_nonzero_audio_ts = 0.0  # never saw a non-zero frame

    await _step_watchdog(times=5)

    no_audio = [b for b in broadcasts if b.get("reason") == "no_audio"]
    assert no_audio, f"expected no_audio warning, got broadcasts={broadcasts}"
    assert no_audio[0]["meeting_id"] == "m-1"
    assert state.silence_warn_sent is True


@pytest.mark.asyncio
async def test_does_not_fire_when_nonzero_is_recent(monkeypatch) -> None:
    """Quiet-but-non-zero ambient audio (peak > 1e-4, RMS < speech threshold)
    keeps ``last_nonzero_audio_ts`` advancing and the watchdog stays quiet."""
    broadcasts: list[dict] = []
    monkeypatch.setattr(
        health_monitors,
        "_broadcast_json",
        AsyncMock(side_effect=lambda payload: broadcasts.append(payload)),
    )
    monkeypatch.setattr(health_monitors, "_SILENCE_WARN_THRESHOLD_S", 0.05)

    state.current_meeting = SimpleNamespace(meeting_id="m-2")
    state.metrics.meeting_start = time.monotonic() - 1.0
    state.last_nonzero_audio_ts = time.monotonic()  # fresh non-zero frame

    await _step_watchdog(times=5)

    no_audio = [b for b in broadcasts if b.get("reason") == "no_audio"]
    assert no_audio == [], f"unexpected no_audio warning: {broadcasts}"


@pytest.mark.asyncio
async def test_cross_session_contamination_blocked_by_meeting_start_anchor(
    monkeypatch,
) -> None:
    """If a previous meeting bumped ``last_nonzero_audio_ts`` long ago, a
    fresh meeting that immediately stops receiving non-zero frames should
    still trip the watchdog (anchored against meeting_start_time)."""
    broadcasts: list[dict] = []
    monkeypatch.setattr(
        health_monitors,
        "_broadcast_json",
        AsyncMock(side_effect=lambda payload: broadcasts.append(payload)),
    )
    monkeypatch.setattr(health_monitors, "_SILENCE_WARN_THRESHOLD_S", 0.05)

    # Pretend the prior meeting had activity 10 s ago.
    now = time.monotonic()
    state.last_nonzero_audio_ts = now - 10.0
    # ... but this meeting started 1 s ago, AFTER the stale activity.
    state.current_meeting = SimpleNamespace(meeting_id="m-3")
    state.metrics.meeting_start = now - 1.0

    await _step_watchdog(times=5)

    no_audio = [b for b in broadcasts if b.get("reason") == "no_audio"]
    assert no_audio, (
        "anchor must be max(last_nonzero, meeting_start) — stale prior-session "
        "activity should not mask a dead mic in the new meeting"
    )


@pytest.mark.asyncio
async def test_clears_when_audio_resumes(monkeypatch) -> None:
    """After ``silence_warn_sent`` flips True, a fresh non-zero frame
    must trigger the ``meeting_warning_cleared`` broadcast."""
    broadcasts: list[dict] = []
    monkeypatch.setattr(
        health_monitors,
        "_broadcast_json",
        AsyncMock(side_effect=lambda payload: broadcasts.append(payload)),
    )
    monkeypatch.setattr(health_monitors, "_SILENCE_WARN_THRESHOLD_S", 0.05)
    health_monitors.SILENCE_WATCHDOG_POLL_INTERVAL_S = 0.001

    state.current_meeting = SimpleNamespace(meeting_id="m-4")
    state.metrics.meeting_start = time.monotonic() - 1.0
    state.last_nonzero_audio_ts = 0.0  # silent
    state.silence_warn_sent = False

    # Phase 1: fire the warn.
    task = asyncio.create_task(health_monitors.silence_watchdog_loop())
    await asyncio.sleep(0.02)
    # Phase 2: simulate audio resumption.
    state.last_nonzero_audio_ts = time.monotonic()
    await asyncio.sleep(0.02)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    cleared = [b for b in broadcasts if b.get("type") == "meeting_warning_cleared"]
    assert cleared, f"expected meeting_warning_cleared, got broadcasts={broadcasts}"
    assert state.silence_warn_sent is False
