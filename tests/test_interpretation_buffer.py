from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from meeting_scribe.audio.interpretation_buffer import InterpretationBuffer
from meeting_scribe.models import TranscriptEvent, TranslationState, TranslationStatus
from meeting_scribe.runtime import state
from meeting_scribe.server_support.sessions import ClientSession


@pytest.fixture(autouse=True)
def isolated_audio_out_state():
    old_clients = set(state._audio_out_clients)
    old_prefs = dict(state._audio_out_prefs)
    old_buffer = state.interpretation_buffer
    old_asr = state.asr_backend
    state._audio_out_clients.clear()
    state._audio_out_prefs.clear()
    state.interpretation_buffer = None
    state.asr_backend = None
    yield
    state._audio_out_clients = old_clients
    state._audio_out_prefs = old_prefs
    state.interpretation_buffer = old_buffer
    state.asr_backend = old_asr


class FakeRoomSink:
    def __init__(self, *, fail: bool = False, delay_s: float = 0.0) -> None:
        self.frames: list[bytes] = []
        self.fail = fail
        self.delay_s = delay_s
        self.playback_until = 0.0
        self.last_played_target_lang = None
        self.last_played_text = ""

    async def send_bytes(self, data: bytes) -> None:
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail:
            raise OSError("sink failed")
        self.frames.append(data)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _event(source: str = "en", target: str = "ja", text: str = "hello") -> TranscriptEvent:
    return TranscriptEvent(
        is_final=True,
        language=source,
        text=text,
        utterance_end_at=time.monotonic(),
        translation=TranslationState(
            status=TranslationStatus.DONE,
            text=f"{text} translated",
            target_language=target,
        ),
    )


def _audio(duration_s: float = 0.02) -> np.ndarray:
    return np.zeros(int(24000 * duration_s), dtype=np.float32)


def _register(listener, pref: ClientSession) -> None:
    state._audio_out_clients.add(listener)
    state._audio_out_prefs[listener] = pref


def _cleanup(*listeners) -> None:
    for listener in listeners:
        state._audio_out_clients.discard(listener)
        state._audio_out_prefs.pop(listener, None)
    state.interpretation_buffer = None
    state.asr_backend = None


def _room_pref() -> ClientSession:
    return ClientSession(
        send_audio=True,
        audio_format="wav-pcm",
        delivery_mode="consecutive",
        transport="room_sink",
        voice_mode="studio",
    )


def test_language_pair_change_filters_source_and_target_membership():
    async def _drive():
        buf = InterpretationBuffer()
        await buf.append(
            event=_event("ja", "en", "a"),
            audio=_audio(),
            source_lang="ja",
            target_lang="en",
            speaker_id="s",
        )
        await buf.append(
            event=_event("en", "ja", "b"),
            audio=_audio(),
            source_lang="en",
            target_lang="ja",
            speaker_id="s",
        )
        dropped = await buf.filter_for_language_pair(["en", "zh"])
        return dropped, [item.target_lang for item in buf._items]

    dropped, remaining_targets = _run(_drive())

    assert dropped == 2
    assert remaining_targets == []


def test_sink_failure_discards_inflight_but_retains_queued_tail():
    sink = FakeRoomSink(fail=True)
    _register(sink, _room_pref())
    try:

        async def _drive():
            buf = InterpretationBuffer(write_timeout_s=0.1)
            await buf.append(
                event=_event("en", "ja", "first"),
                audio=_audio(),
                source_lang="en",
                target_lang="ja",
                speaker_id="s",
            )
            await buf.append(
                event=_event("en", "ja", "second"),
                audio=_audio(),
                source_lang="en",
                target_lang="ja",
                speaker_id="s",
            )
            await buf.try_flush()
            await asyncio.sleep(0.05)
            return len(buf._items), sink in state._audio_out_clients

        queued, still_registered = _run(_drive())
    finally:
        _cleanup(sink)

    assert queued == 1
    assert still_registered is False


def test_web_listener_unregister_does_not_cancel_room_release():
    room = FakeRoomSink()
    web = FakeRoomSink()
    _register(room, _room_pref())
    _register(
        web,
        ClientSession(
            send_audio=True,
            audio_format="wav-pcm",
            delivery_mode="simultaneous",
            transport="web_browser",
            voice_mode="studio",
        ),
    )
    try:

        async def _drive():
            buf = InterpretationBuffer()
            await buf.append(
                event=_event("en", "ja", "hello"),
                audio=_audio(),
                source_lang="en",
                target_lang="ja",
                speaker_id="s",
            )
            state._audio_out_clients.discard(web)
            state._audio_out_prefs.pop(web, None)
            await buf.try_flush()
            await asyncio.sleep(0.05)
            return len(room.frames), buf.release_generation

        delivered, generation = _run(_drive())
    finally:
        _cleanup(room, web)

    assert delivered == 1
    assert generation == 0
