"""Phase 5 — mic-readiness probe tests.

The probe shipped with Phase 5 must:

1. Reject when no mic is configured.
2. Reject when the audio route is in a known-failure state
   (Phase 1's ``ambiguous`` / ``unresolved`` / ``capture_failed``).
3. Honor the probe-local epoch contract — a timestamp set BEFORE the
   probe started never counts, even if it's brand new.
4. Never accept ``server_mic_active = True`` alone as proof; require
   evidence of non-zero samples observed inside the wait window.
5. Pass when the capture pushes a fresh frame inside the window.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from meeting_scribe.audio.mic_liveness import (
    LivenessResult,
    probe_mic_liveness,
)
from meeting_scribe.runtime import state


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Redirect settings.json to a tmp file so probes don't see real state."""
    from meeting_scribe.server_support import settings_store

    monkeypatch.setattr(settings_store, "SETTINGS_OVERRIDE_FILE", tmp_path / "settings.json")
    settings_store._settings_cache = None
    settings_store._settings_cache_mtime = 0.0
    state.last_nonzero_audio_ts = 0.0
    state.server_mic_active = False
    state.audio_route_status = "ok"
    yield
    settings_store._settings_cache = None
    settings_store._settings_cache_mtime = 0.0
    state.last_nonzero_audio_ts = 0.0
    state.server_mic_active = False
    state.audio_route_status = "ok"


def _write_mic(monkeypatch, node: str) -> None:
    from meeting_scribe.audio.audio_routing import SETTINGS_AUDIO_MEETING_MIC_NODE
    from meeting_scribe.server_support.settings_store import _save_settings_override

    _save_settings_override({SETTINGS_AUDIO_MEETING_MIC_NODE: node})


@pytest.mark.asyncio
async def test_returns_no_mic_configured_when_empty():
    result = await probe_mic_liveness()
    assert isinstance(result, LivenessResult)
    assert result.ok is False
    assert result.reason == "no_mic_configured"


@pytest.mark.asyncio
async def test_returns_unresolved_route_when_status_ambiguous(monkeypatch):
    _write_mic(monkeypatch, "alsa_input.usb-foo-00.input.2")
    state.audio_route_status = "ambiguous"
    result = await probe_mic_liveness()
    assert result.ok is False
    assert result.reason == "unresolved_route"
    assert "ambiguous" in result.detail


@pytest.mark.asyncio
async def test_returns_unresolved_route_for_capture_failed_status(monkeypatch):
    _write_mic(monkeypatch, "alsa_input.usb-foo-00.input.2")
    state.audio_route_status = "capture_failed"
    result = await probe_mic_liveness()
    assert result.ok is False
    assert result.reason == "unresolved_route"


@pytest.mark.asyncio
async def test_does_not_accept_pre_reset_timestamp(monkeypatch):
    """A stale timestamp from before the probe was called — even if
    very recent — must not pass. The probe resets the timestamp to
    0.0 at entry and requires strict > epoch_ts to clear."""
    _write_mic(monkeypatch, "alsa_input.usb-foo-00.input.2")
    state.server_mic_active = True
    state.last_nonzero_audio_ts = time.monotonic()  # very fresh, but pre-reset

    result = await probe_mic_liveness(timeout_s=0.2)

    assert result.ok is False
    assert result.reason == "samples_all_zero"


@pytest.mark.asyncio
async def test_does_not_accept_server_mic_active_alone(monkeypatch):
    """``server_mic_active=True`` without fresh evidence must fail.
    Codex review caught this: the flag is "capture process is
    running" — it says nothing about non-zero samples."""
    _write_mic(monkeypatch, "alsa_input.usb-foo-00.input.2")
    state.server_mic_active = True
    state.last_nonzero_audio_ts = 0.0  # capture process running, but all zero

    result = await probe_mic_liveness(timeout_s=0.2)

    assert result.ok is False
    assert result.reason == "samples_all_zero"


@pytest.mark.asyncio
async def test_fails_when_server_mic_not_running(monkeypatch):
    _write_mic(monkeypatch, "alsa_input.usb-foo-00.input.2")
    state.server_mic_active = False

    result = await probe_mic_liveness()

    assert result.ok is False
    assert result.reason == "samples_all_zero"


@pytest.mark.asyncio
async def test_passes_when_fresh_frame_arrives_during_wait(monkeypatch):
    """The capture pushes a non-zero frame inside the wait window
    (after the probe started). Probe must clear with ok=True and
    record the observed timestamp."""
    _write_mic(monkeypatch, "alsa_input.usb-foo-00.input.2")
    state.server_mic_active = True
    state.last_nonzero_audio_ts = 0.0

    async def _bump_after_delay() -> None:
        await asyncio.sleep(0.05)
        state.last_nonzero_audio_ts = time.monotonic()

    bump_task = asyncio.create_task(_bump_after_delay())
    result = await probe_mic_liveness(timeout_s=1.0)
    await bump_task

    assert result.ok is True
    assert result.reason == "ok"
    assert result.observed_at is not None
    assert result.observed_at > result.probe_epoch_ts
