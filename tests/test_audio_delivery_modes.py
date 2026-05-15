from __future__ import annotations

import asyncio

import numpy as np

from meeting_scribe.audio.output_pipeline import _send_audio_to_listeners
from meeting_scribe.runtime import state
from meeting_scribe.server_support.sessions import ClientSession


class FakeListener:
    def __init__(self) -> None:
        self.frames: list[bytes] = []

    async def send_bytes(self, data: bytes) -> None:
        self.frames.append(data)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _pref(*, transport: str, delivery_mode: str = "simultaneous") -> ClientSession:
    return ClientSession(
        preferred_language="ja",
        send_audio=True,
        audio_format="wav-pcm",
        voice_mode="studio",
        delivery_mode=delivery_mode,
        transport=transport,
    )


def _register(listener, pref: ClientSession) -> None:
    state._audio_out_clients.add(listener)
    state._audio_out_prefs[listener] = pref


def _cleanup(*listeners) -> None:
    for listener in listeners:
        state._audio_out_clients.discard(listener)
        state._audio_out_prefs.pop(listener, None)
    state.interpretation_buffer = None


def test_mute_web_does_not_mute_bt_headset_delivery():
    web = FakeListener()
    bt = FakeListener()
    _register(web, _pref(transport="web_browser", delivery_mode="drop"))
    _register(bt, _pref(transport="bt_headset"))
    try:
        sent = _run(_send_audio_to_listeners(np.zeros(200, dtype=np.float32), "ja", "studio"))
    finally:
        _cleanup(web, bt)

    assert sent == 1
    assert len(web.frames) == 0
    assert len(bt.frames) == 1


def test_mute_bt_headset_does_not_mute_web_delivery():
    web = FakeListener()
    bt = FakeListener()
    _register(web, _pref(transport="web_browser"))
    _register(bt, _pref(transport="bt_headset", delivery_mode="drop"))
    try:
        sent = _run(_send_audio_to_listeners(np.zeros(200, dtype=np.float32), "ja", "studio"))
    finally:
        _cleanup(web, bt)

    assert sent == 1
    assert len(web.frames) == 1
    assert len(bt.frames) == 0
