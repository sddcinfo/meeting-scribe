from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi import FastAPI
from starlette.testclient import TestClient

from meeting_scribe.runtime import state
from meeting_scribe.server_support.sessions import ClientSession
from meeting_scribe.ws import admin as admin_ws_mod
from meeting_scribe.ws.view_broadcast import router as view_router


def test_view_ws_rejects_mutation_messages() -> None:
    app = FastAPI()
    app.include_router(view_router)
    with (
        TestClient(app, base_url="http://test") as client,
        client.websocket_connect("/api/ws/view") as ws,
    ):
        ws.send_text(json.dumps({"type": "set_interpretation", "enabled": True}))
        body = json.loads(ws.receive_text())
        assert body["code"] == 403


def test_admin_ws_mute_web_does_not_mute_bt(monkeypatch) -> None:
    async def allow(_ws):
        return True

    monkeypatch.setattr(admin_ws_mod, "require_admin_ws", allow)
    app = FastAPI()
    app.include_router(admin_ws_mod.router)

    web = object()
    bt = object()
    state._audio_out_clients.update({web, bt})
    state._audio_out_prefs[web] = ClientSession(transport="web_browser")
    state._audio_out_prefs[bt] = ClientSession(transport="bt_headset")
    try:
        with (
            TestClient(app, base_url="http://test") as client,
            client.websocket_connect("/api/ws/admin") as ws,
        ):
            ws.send_text(json.dumps({"type": "mute_web"}))
            body = json.loads(ws.receive_text())
            assert body["type"] == "interpretation_ack"
        assert state._audio_out_prefs[web].delivery_mode == "drop"
        assert state._audio_out_prefs[bt].delivery_mode == "simultaneous"
    finally:
        state._audio_out_clients.discard(web)
        state._audio_out_clients.discard(bt)
        state._audio_out_prefs.pop(web, None)
        state._audio_out_prefs.pop(bt, None)


def test_admin_set_language_pair_filters_buffer(monkeypatch) -> None:
    class FakeQueue:
        def __init__(self) -> None:
            self.languages = None

        def set_languages(self, languages):
            self.languages = languages

    class FakeAsr(FakeQueue):
        pass

    class FakeBuffer:
        def __init__(self) -> None:
            self.pair = None

        async def filter_for_language_pair(self, pair):
            self.pair = pair
            return 0

    written = []
    old_meeting = state.current_meeting
    old_storage = state.storage
    old_queue = state.translation_queue
    old_asr = state.asr_backend
    old_buffer = state.interpretation_buffer
    state.current_meeting = SimpleNamespace(meeting_id="m1", language_pair=["en", "ja"])
    state.storage = SimpleNamespace(_write_meta=lambda meta: written.append(meta.language_pair))
    state.translation_queue = FakeQueue()
    state.asr_backend = FakeAsr()
    state.interpretation_buffer = FakeBuffer()
    try:
        import asyncio

        result = asyncio.run(admin_ws_mod._set_language_pair(["en", "zh"]))
        assert result["ok"] is True
        assert written == [["en", "zh"]]
        assert state.translation_queue.languages == ["en", "zh"]
        assert state.asr_backend.languages == ["en", "zh"]
        assert state.interpretation_buffer.pair == ["en", "zh"]
    finally:
        state.current_meeting = old_meeting
        state.storage = old_storage
        state.translation_queue = old_queue
        state.asr_backend = old_asr
        state.interpretation_buffer = old_buffer
