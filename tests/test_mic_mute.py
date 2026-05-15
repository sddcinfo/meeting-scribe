"""Soft mic mute — privacy pause at the input boundary.

Pins three contract points:
  1. The POST route flips ``state.mic_input_muted`` and is reflected in
     ``_interpretation_payload`` immediately (so the UI updates without
     a poll round-trip).
  2. ``_handle_audio`` drops every frame while muted — no PCM bytes
     reach the audio writer and no samples reach ASR.
  3. Mute is runtime-only: ``_start_meeting_locked`` resets it on every
     new meeting and nothing is persisted to settings.json.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient


@pytest.fixture(autouse=True)
def restore_state():
    """Capture every piece of mutable runtime state these tests can touch.

    Some tests run ``_start_meeting_locked`` partially and the reconcile
    helper inside it may add listeners or interpretation buffers to
    ``state`` that would leak into adjacent test files (notably the
    ``test_ws_audio_out`` fan-out assertions).
    """
    from meeting_scribe.runtime import state

    saved_muted = state.mic_input_muted
    saved_clients = set(state._audio_out_clients)
    saved_prefs = dict(state._audio_out_prefs)
    saved_buf = state.interpretation_buffer
    saved_server_mic = state.server_mic
    saved_server_mic_active = state.server_mic_active
    yield
    state.mic_input_muted = saved_muted
    state._audio_out_clients = saved_clients
    state._audio_out_prefs = saved_prefs
    state.interpretation_buffer = saved_buf
    state.server_mic = saved_server_mic
    state.server_mic_active = saved_server_mic_active


def _build_app(*, admin_ok: bool) -> FastAPI:
    from meeting_scribe.routes import admin_audio as admin_audio_mod

    if admin_ok:
        admin_audio_mod._require_admin_response = lambda req: None
    else:
        admin_audio_mod._require_admin_response = lambda req: JSONResponse(
            {"error": "admin required"}, status_code=403
        )
    app = FastAPI()
    app.include_router(admin_audio_mod.router)
    return app


def test_mic_route_requires_admin() -> None:
    app = _build_app(admin_ok=False)
    with TestClient(app, base_url="http://test") as client:
        resp = client.post("/api/admin/audio/mic", json={"muted": True})
    assert resp.status_code == 403


def test_mic_route_flips_state_and_payload_reflects_it() -> None:
    from meeting_scribe.runtime import state

    state.mic_input_muted = False
    app = _build_app(admin_ok=True)
    with TestClient(app, base_url="http://test") as client:
        resp = client.post("/api/admin/audio/mic", json={"muted": True})

    assert resp.status_code == 200
    body = resp.json()
    assert body["mic_muted"] is True
    assert state.mic_input_muted is True


def test_mic_route_rejects_missing_field() -> None:
    app = _build_app(admin_ok=True)
    with TestClient(app, base_url="http://test") as client:
        resp = client.post("/api/admin/audio/mic", json={})
    assert resp.status_code == 400


def test_mic_route_rejects_non_bool() -> None:
    app = _build_app(admin_ok=True)
    with TestClient(app, base_url="http://test") as client:
        resp = client.post("/api/admin/audio/mic", json={"muted": "yes"})
    assert resp.status_code == 400


def test_mic_route_does_not_persist_to_settings() -> None:
    """Mute is meeting-scoped runtime state — settings.json must stay clean."""
    saved: dict[str, Any] = {}

    def fake_save(updates: dict[str, Any]) -> None:
        saved.update(updates)

    app = _build_app(admin_ok=True)
    with (
        patch("meeting_scribe.routes.admin_audio._save_settings_override", fake_save),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post("/api/admin/audio/mic", json={"muted": True})

    assert resp.status_code == 200
    # The mute toggle must NOT have called _save_settings_override.
    assert saved == {}


@pytest.mark.asyncio
async def test_handle_audio_drops_frame_when_muted() -> None:
    """Privacy pause: no PCM to disk, no samples to ASR while muted."""
    from meeting_scribe.runtime import state
    from meeting_scribe.ws.audio_input import _handle_audio

    state.mic_input_muted = True

    writer_calls: list[Any] = []
    asr_calls: list[Any] = []

    class _FakeWriter:
        total_bytes = 0

        def write_at(self, *args: Any, **kwargs: Any) -> None:
            writer_calls.append((args, kwargs))

    class _FakeASR:
        async def process_audio_bytes(self, *args: Any, **kwargs: Any) -> None:
            asr_calls.append((args, kwargs))

    saved_writer = state.audio_writer
    saved_asr = state.asr_backend
    saved_dropped = state.metrics.mic_muted_chunks_dropped
    state.audio_writer = _FakeWriter()
    state.asr_backend = _FakeASR()
    try:
        # 1024 bytes of fabricated PCM with a leading rate header.
        rate_header = (16000).to_bytes(4, "little")
        pcm = b"\x00\x00" * 256
        await _handle_audio(rate_header + pcm)
    finally:
        state.audio_writer = saved_writer
        state.asr_backend = saved_asr

    assert writer_calls == [], "writer must not be touched while muted"
    assert asr_calls == [], "ASR must not be touched while muted"
    assert state.metrics.mic_muted_chunks_dropped == saved_dropped + 1


def test_meeting_start_resets_mute_flag(monkeypatch) -> None:
    """A stale mute toggle must not silence a freshly-started meeting.

    The reset lives downstream of the deep-health gate so a 503 refusal
    preserves the operator's pre-start mute. To prove the reset fires
    when health passes, this test patches the health gate to ready and
    lets the handler raise downstream (storage isn't set up); the reset
    line is reached before any downstream attempt.
    """
    import asyncio

    from meeting_scribe.routes import meeting_lifecycle
    from meeting_scribe.runtime import state
    from meeting_scribe.server_support import backend_health

    state.mic_input_muted = True
    state.current_meeting = None

    async def fake_deep_health(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {"asr": {"ready": True}, "translate": {"ready": True}}

    async def fake_preflight() -> None:
        return None

    monkeypatch.setattr(backend_health, "_deep_backend_health", fake_deep_health)
    monkeypatch.setattr(meeting_lifecycle, "_meeting_start_preflight", fake_preflight)

    class _StubRequest:
        async def json(self) -> dict[str, Any]:
            raise ValueError("no body")

    async def runner() -> None:
        # The handler will raise once it tries to touch ``state.storage``
        # (not initialized in tests). The reset runs before any storage
        # access, so we just swallow the downstream exception.
        try:
            await meeting_lifecycle._start_meeting_locked(_StubRequest())
        except Exception:
            pass

    asyncio.run(runner())
    assert state.mic_input_muted is False
