"""Local room-sink playback guard tests."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

import meeting_scribe.audio.local_sink as local_sink
from meeting_scribe.audio.local_sink import LocalSinkListener
from meeting_scribe.audio.output_pipeline import _build_riff_wav
from meeting_scribe.runtime import state


def test_resolve_local_sink_language_keeps_configured_target() -> None:
    assert local_sink.resolve_local_sink_language({"local_sink_language": "ja"}) == "ja"
    assert local_sink.resolve_local_sink_language({}) == "en"
    assert local_sink.resolve_local_sink_language({"local_sink_language": "all"}) == "en"
    assert local_sink.resolve_room_sink_language({"room_tts_language": "all"}) == ""


def test_should_enable_local_sink_requires_explicit_tts_target(monkeypatch) -> None:
    monkeypatch.setattr(
        local_sink,
        "_load_settings_override",
        lambda: {"bt_input_active": True},
    )

    assert local_sink.should_enable_local_sink() is False


def test_should_enable_local_sink_allows_safe_room_target(monkeypatch) -> None:
    monkeypatch.setattr(
        local_sink,
        "_load_settings_override",
        lambda: {
            "audio_meeting_mic_active": True,
            "audio_meeting_mic_node": (
                "alsa_input.usb-Plantronics_Poly_Sync_20-M_8B33ABCDEF-00.mono-fallback"
            ),
            "audio_room_tts_sink_node": (
                "alsa_output.usb-Plantronics_Poly_Sync_20-M_8B33ABCDEF-00.analog-stereo"
            ),
        },
    )

    assert local_sink.should_enable_local_sink() is True


@pytest.fixture(autouse=True)
def restore_playback_guard():
    previous = getattr(state, "local_tts_playback_until", 0.0)
    state.local_tts_playback_until = 0.0
    yield
    state.local_tts_playback_until = previous


@pytest.mark.asyncio
async def test_local_sink_extends_playback_guard(monkeypatch):
    listener = LocalSinkListener(
        target_node="alsa_output.poly",
        role=local_sink.LOCAL_SINK_ROLE_ROOM,
        echo_guard=True,
    )
    writes: list[int] = []

    def fake_write(pcm: bytes) -> None:
        writes.append(len(pcm))

    monkeypatch.setattr(listener, "_write_pcm_sync", fake_write)
    audio = np.ones(24_000, dtype=np.float32) * 0.05
    before = time.monotonic()

    await listener.send_bytes(_build_riff_wav(audio, 24_000))
    await listener.drain()

    assert writes
    assert state.local_tts_playback_until > before + 1.0
    listener.shutdown()


@pytest.mark.asyncio
async def test_local_sink_queues_tts_burst_without_dropping(monkeypatch):
    monkeypatch.setattr(local_sink, "LOCAL_SINK_QUEUE_MAXSIZE", 8)
    listener = LocalSinkListener(
        target_node="alsa_output.dell",
        role=local_sink.LOCAL_SINK_ROLE_ROOM,
        echo_guard=True,
    )
    monkeypatch.setattr(listener, "_ensure_writer_task", lambda: None)
    audio = np.ones(24_000, dtype=np.float32) * 0.05
    wav = _build_riff_wav(audio, 24_000)
    loop_now = asyncio.get_running_loop().time()

    for _ in range(6):
        await listener.send_bytes(wav)

    assert listener._queue is not None
    assert listener._queue.qsize() == 6
    assert listener.dropped_buffers == 0
    assert state.local_tts_playback_until > loop_now + 6.0
    listener.shutdown()


@pytest.mark.asyncio
async def test_local_sink_drops_only_after_configured_safety_cap(monkeypatch):
    monkeypatch.setattr(local_sink, "LOCAL_SINK_QUEUE_MAXSIZE", 2)
    listener = LocalSinkListener(target_node="alsa_output.dell")
    monkeypatch.setattr(listener, "_ensure_writer_task", lambda: None)
    audio = np.ones(24_000, dtype=np.float32) * 0.05
    wav = _build_riff_wav(audio, 24_000)

    for _ in range(3):
        await listener.send_bytes(wav)

    assert listener._queue is not None
    assert listener._queue.qsize() == 2
    assert listener.dropped_buffers == 1
    listener.shutdown()
