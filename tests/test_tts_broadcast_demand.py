from __future__ import annotations

import time

import pytest

from meeting_scribe.models import TranscriptEvent, TranslationState, TranslationStatus
from meeting_scribe.pipeline import transcript_event
from meeting_scribe.runtime import state
from meeting_scribe.server_support.sessions import ClientSession


def _translated_event(target_language: str) -> TranscriptEvent:
    now = time.monotonic()
    return TranscriptEvent(
        segment_id=f"tts-demand-{target_language}",
        revision=1,
        is_final=True,
        start_ms=0,
        end_ms=1000,
        language="ja" if target_language == "en" else "en",
        text="source",
        translation=TranslationState(
            status=TranslationStatus.DONE,
            text="translated",
            target_language=target_language,
            completed_at=now,
        ),
        utterance_end_at=now,
    )


@pytest.fixture(autouse=True)
def reset_audio_out_state(monkeypatch):
    old_clients = set(state._audio_out_clients)
    old_prefs = dict(state._audio_out_prefs)
    old_tts_backend = state.tts_backend
    old_asr_backend = state.asr_backend

    class TTSBackend:
        available = True

        def cache_voice(self, *_args, **_kwargs):
            return None

    class ASRBackend:
        last_audio_chunk = None

    async def noop_broadcast(_event):
        return None

    import meeting_scribe.audio.local_sink as local_sink

    monkeypatch.setattr(transcript_event, "_broadcast", noop_broadcast)
    monkeypatch.setattr(local_sink, "ensure_local_sink_listener_registered", lambda: None)
    monkeypatch.setattr(state, "tts_backend", TTSBackend(), raising=False)
    monkeypatch.setattr(state, "asr_backend", ASRBackend(), raising=False)
    state._audio_out_clients.clear()
    state._audio_out_prefs.clear()
    yield
    state._audio_out_clients = old_clients
    state._audio_out_prefs = old_prefs
    state.tts_backend = old_tts_backend
    state.asr_backend = old_asr_backend


@pytest.mark.asyncio
async def test_broadcast_translation_skips_tts_enqueue_when_no_listener_wants_target(
    monkeypatch,
):
    listener = object()
    state._audio_out_clients.add(listener)
    state._audio_out_prefs[listener] = ClientSession(
        preferred_language="en",
        delivery_mode="simultaneous",
        send_audio=True,
    )
    enqueued = []
    monkeypatch.setattr(
        transcript_event, "_enqueue_tts", lambda event, speaker_id: enqueued.append(event)
    )

    await transcript_event._broadcast_translation(_translated_event("ja"))

    assert enqueued == []


@pytest.mark.asyncio
async def test_broadcast_translation_enqueues_tts_when_listener_wants_target(monkeypatch):
    listener = object()
    state._audio_out_clients.add(listener)
    state._audio_out_prefs[listener] = ClientSession(
        preferred_language="en",
        delivery_mode="simultaneous",
        send_audio=True,
    )
    enqueued = []
    monkeypatch.setattr(
        transcript_event, "_enqueue_tts", lambda event, speaker_id: enqueued.append(event)
    )

    await transcript_event._broadcast_translation(_translated_event("en"))

    assert [event.segment_id for event in enqueued] == ["tts-demand-en"]
