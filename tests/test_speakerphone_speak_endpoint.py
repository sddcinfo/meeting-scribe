"""Tests for ``apply_speak`` — the speakerphone TTS feedback path.

Covers:

* Fresh ``mapping.load()`` on every call (language/enabled changes
  take effect on the very next call).
* Split ``respect_enabled`` semantics — internal route respects the
  saved flag; preview route bypasses it.
* Token-queue reservation gives real exclusion against translation:
  a drained gate causes feedback to drop without touching TTS.
* Idle-backend regression: full gate + no in-flight translation
  must succeed; this guards against re-introducing the broken
  ``wait_for(acquire(), timeout=0)`` idiom that would drop here.
* Unsaved inline override (from the preview "Test" button) wins
  over the saved override.
* Unknown ``label_id`` raises ``FeedbackError`` → 400 path.

The tests stub ``state.tts_backend`` and ``state._audio_out_clients``
so no real synthesis happens. They use monkeypatch on
``mapping.default_path`` to isolate per-test config writes.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from meeting_scribe.runtime import state
from meeting_scribe.speakerphone import api as sp_api
from meeting_scribe.speakerphone import mapping

# ── Fakes ──────────────────────────────────────────────────────────────


class _FakeTTSBackend:
    """Records every synthesize_stream call + emits a known chunk shape.

    Behaviour controlled by ``self.before_yield_event`` — if set, the
    backend waits on the event before yielding (lets a test simulate
    in-flight translation that holds the gate).
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.sample_rate = 24000
        self.before_yield_event: asyncio.Event | None = None

    async def synthesize_stream(self, *, text, language, **kwargs):
        self.calls.append({"text": text, "language": language})
        if self.before_yield_event is not None:
            await self.before_yield_event.wait()
        # one short chunk so _stream_feedback_to_local_sink has work
        yield np.array([0.1, 0.2, 0.1, 0.0], dtype=np.float32)


class _FakeListener:
    """LocalSinkListener stub — records the WAV bytes it would play."""

    def __init__(self) -> None:
        self.received: list[bytes] = []

    async def send_bytes(self, data: bytes) -> None:
        self.received.append(data)


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def isolated_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    """Redirect mapping default_path to a fresh per-test JSON file.

    Also resets the mtime cache between tests so the fresh-read assertion
    is meaningful.
    """
    target = tmp_path / "speakerphone.json"
    monkeypatch.setattr(mapping, "default_path", lambda: target)
    return target


@pytest.fixture
def stubbed_state(monkeypatch: pytest.MonkeyPatch):
    """Plant a fake TTS backend + listener + initialised dispatch gate.

    Yields a tuple ``(backend, listener)`` for assertions.
    """
    backend = _FakeTTSBackend()
    listener = _FakeListener()
    gate: asyncio.Queue = asyncio.Queue(maxsize=1)
    gate.put_nowait(object())  # one token, mirrors production N=1

    monkeypatch.setattr(state, "tts_backend", backend)
    monkeypatch.setattr(state, "_audio_out_clients", [listener])
    monkeypatch.setattr(state, "tts_dispatch_gate", gate)
    yield backend, listener, gate


# ── Happy path / catalog resolution ────────────────────────────────────


@pytest.mark.asyncio
async def test_speak_returns_resolved_text_and_plays(
    isolated_mapping,
    stubbed_state,
) -> None:
    backend, listener, _gate = stubbed_state
    result = await sp_api.apply_speak(
        label_id="volume_up",
        language="ja",
        overrides_inline=None,
        respect_enabled=True,
    )
    assert result["ok"] is True
    assert result.get("skipped") is None
    assert result["text"] == "音量アップ"
    assert backend.calls and backend.calls[0]["language"] == "ja"
    assert listener.received, "WAV bytes never sent to the listener"
    # Sanity-check WAV header.
    assert listener.received[0][:4] == b"RIFF"


@pytest.mark.asyncio
async def test_speak_returns_to_initial_token_count(
    isolated_mapping,
    stubbed_state,
) -> None:
    """After every speak call, the dispatch gate must be replenished.

    Leaking the token would deadlock every future feedback AND every
    future translation — same gate.
    """
    _backend, _listener, gate = stubbed_state
    initial = gate.qsize()
    await sp_api.apply_speak(
        label_id="volume_up",
        language="en",
        overrides_inline=None,
        respect_enabled=True,
    )
    assert gate.qsize() == initial


# ── respect_enabled split semantics ────────────────────────────────────


@pytest.mark.asyncio
async def test_internal_respects_enabled_false(
    isolated_mapping,
    stubbed_state,
) -> None:
    """When ``enabled=False`` in the saved mapping, internal speak skips."""
    backend, _listener, _gate = stubbed_state
    doc = mapping.default_document()
    doc["button_feedback"]["enabled"] = False
    mapping.save(doc, isolated_mapping)

    result = await sp_api.apply_speak(
        label_id="volume_up",
        language="en",
        overrides_inline=None,
        respect_enabled=True,
    )
    assert result["skipped"] is True
    assert "disabled" in result["reason"]
    assert backend.calls == []  # never reached TTS


@pytest.mark.asyncio
async def test_preview_bypasses_enabled_false(
    isolated_mapping,
    stubbed_state,
) -> None:
    """Same saved mapping; preview path MUST still play (audition use case)."""
    backend, _listener, _gate = stubbed_state
    doc = mapping.default_document()
    doc["button_feedback"]["enabled"] = False
    mapping.save(doc, isolated_mapping)

    result = await sp_api.apply_speak(
        label_id="volume_up",
        language="en",
        overrides_inline=None,
        respect_enabled=False,
    )
    assert result["ok"] is True
    assert result.get("skipped") is None
    assert len(backend.calls) == 1


# ── Fresh mapping reads ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_speak_rereads_mapping_on_every_call(
    isolated_mapping,
    stubbed_state,
) -> None:
    """Operator changes language in the Hardware tab → next speak uses it.

    We don't pass ``language`` here; the server fills it from the
    fresh mapping read on each call.
    """
    backend, _listener, _gate = stubbed_state
    doc = mapping.default_document()
    doc["button_feedback"]["language"] = "en"
    mapping.save(doc, isolated_mapping)
    await sp_api.apply_speak(
        label_id="volume_up",
        language=None,
        overrides_inline=None,
        respect_enabled=True,
    )
    assert backend.calls[-1]["language"] == "en"
    assert backend.calls[-1]["text"] == "Volume up"

    # Operator flips language to Japanese.
    doc = mapping.load(isolated_mapping)
    doc["button_feedback"]["language"] = "ja"
    mapping.save(doc, isolated_mapping)
    await sp_api.apply_speak(
        label_id="volume_up",
        language=None,
        overrides_inline=None,
        respect_enabled=True,
    )
    assert backend.calls[-1]["language"] == "ja"
    assert backend.calls[-1]["text"] == "音量アップ"


# ── Dispatch gate reservation ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_drained_gate_returns_backend_busy(
    isolated_mapping,
    stubbed_state,
) -> None:
    """All tokens held → feedback drops, never invokes the TTS backend."""
    backend, _listener, gate = stubbed_state
    # Drain the gate (simulate translation holding the only token).
    gate.get_nowait()
    result = await sp_api.apply_speak(
        label_id="volume_up",
        language="en",
        overrides_inline=None,
        respect_enabled=True,
    )
    assert result["skipped"] is True
    assert "busy" in result["reason"]
    assert backend.calls == []  # TTS backend NOT touched


@pytest.mark.asyncio
async def test_full_gate_with_no_translation_in_flight_succeeds(
    isolated_mapping,
    stubbed_state,
) -> None:
    """Idle-backend regression test.

    With the gate full and no translation in flight, feedback MUST
    succeed and play. This catches the broken
    ``wait_for(sem.acquire(), timeout=0)`` idiom — that one would
    erroneously time out at t=0 even with permits available.
    """
    backend, listener, gate = stubbed_state
    # Sanity: gate is full from the fixture.
    assert gate.qsize() == 1
    result = await sp_api.apply_speak(
        label_id="volume_up",
        language="en",
        overrides_inline=None,
        respect_enabled=True,
    )
    assert result["ok"] is True
    assert result.get("skipped") is None
    assert len(backend.calls) == 1
    assert listener.received


@pytest.mark.asyncio
async def test_gate_token_released_even_when_synthesis_raises(
    isolated_mapping,
    stubbed_state,
) -> None:
    """Synthesis exception must not leak the token (would deadlock everything)."""
    _backend, _listener, gate = stubbed_state

    class _RaisingBackend(_FakeTTSBackend):
        async def synthesize_stream(self, *, text, language, **kwargs):
            self.calls.append({"text": text, "language": language})
            raise RuntimeError("simulated TTS crash")
            yield  # pragma: no cover — unreachable

    state.tts_backend = _RaisingBackend()
    result = await sp_api.apply_speak(
        label_id="volume_up",
        language="en",
        overrides_inline=None,
        respect_enabled=True,
    )
    assert result["ok"] is False
    assert "synthesis failed" in result["reason"]
    # Token returned to the gate.
    assert gate.qsize() == 1


# ── Unsaved override (preview Test button) ─────────────────────────────


@pytest.mark.asyncio
async def test_preview_inline_override_wins_over_saved(
    isolated_mapping,
    stubbed_state,
) -> None:
    """The Test button's unsaved text takes precedence over the saved one."""
    backend, _listener, _gate = stubbed_state
    doc = mapping.default_document()
    doc["button_feedback"]["overrides"] = {"volume_up": {"en": "Saved text"}}
    mapping.save(doc, isolated_mapping)

    result = await sp_api.apply_speak(
        label_id="volume_up",
        language="en",
        overrides_inline={"volume_up": {"en": "Unsaved test text"}},
        respect_enabled=False,
    )
    assert result["text"] == "Unsaved test text"
    assert backend.calls[-1]["text"] == "Unsaved test text"


@pytest.mark.asyncio
async def test_inline_override_for_unsaved_label(
    isolated_mapping,
    stubbed_state,
) -> None:
    """Preview an override for a label that has NO saved overrides yet."""
    backend, _listener, _gate = stubbed_state
    # default mapping has overrides={}
    result = await sp_api.apply_speak(
        label_id="volume_up",
        language="en",
        overrides_inline={"volume_up": {"en": "Just typed"}},
        respect_enabled=False,
    )
    assert result["text"] == "Just typed"
    assert backend.calls[-1]["text"] == "Just typed"


# ── Unknown label_id ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_label_id_raises_feedback_error(
    isolated_mapping,
    stubbed_state,
) -> None:
    with pytest.raises(sp_api.FeedbackError) as excinfo:
        await sp_api.apply_speak(
            label_id="not_a_real_label",
            language="en",
            overrides_inline=None,
            respect_enabled=True,
        )
    assert "not_a_real_label" in str(excinfo.value)


# ── Missing TTS backend / dispatch gate ────────────────────────────────


@pytest.mark.asyncio
async def test_missing_dispatch_gate_returns_skipped(
    isolated_mapping,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before lifespan runs, the gate is None. Should skip cleanly."""
    monkeypatch.setattr(state, "tts_dispatch_gate", None)
    monkeypatch.setattr(state, "tts_backend", _FakeTTSBackend())
    monkeypatch.setattr(state, "_audio_out_clients", [_FakeListener()])
    result = await sp_api.apply_speak(
        label_id="volume_up",
        language="en",
        overrides_inline=None,
        respect_enabled=True,
    )
    assert result["skipped"] is True


@pytest.mark.asyncio
async def test_missing_tts_backend_returns_not_ok(
    isolated_mapping,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(state, "tts_backend", None)
    gate: asyncio.Queue = asyncio.Queue(maxsize=1)
    gate.put_nowait(object())
    monkeypatch.setattr(state, "tts_dispatch_gate", gate)
    monkeypatch.setattr(state, "_audio_out_clients", [])
    result = await sp_api.apply_speak(
        label_id="volume_up",
        language="en",
        overrides_inline=None,
        respect_enabled=True,
    )
    assert result["ok"] is False
    assert "tts backend unavailable" in result["reason"]
    # Token still returned to the gate.
    assert gate.qsize() == 1


# ── Voice resolution + diagnostic fields (regression for 2026-05-13) ────


@pytest.mark.asyncio
async def test_speak_passes_studio_voice_to_backend(
    isolated_mapping,
    stubbed_state,
) -> None:
    """The Qwen3 backend skips with "no voice specified and no ref_audio"
    when both ``voice_reference`` and ``studio_voice`` are None — that's
    what initially caused a silent 44 ms no-op against a live server on
    2026-05-13. ``_stream_feedback_to_local_sink`` MUST pass the
    language's catalog studio voice via ``studio_voice_for``.
    """
    backend, _listener, _gate = stubbed_state

    # Replace the fake's synthesize_stream with one that captures all
    # kwargs verbatim so we can assert on what the production code sent.
    captured: dict[str, object] = {}

    async def _capture(*, text, language, **kwargs):  # type: ignore[no-untyped-def]
        captured["text"] = text
        captured["language"] = language
        captured.update(kwargs)
        yield np.array([0.1, 0.2], dtype=np.float32)

    backend.synthesize_stream = _capture  # type: ignore[method-assign]

    await sp_api.apply_speak(
        label_id="volume_up",
        language="en",
        overrides_inline=None,
        respect_enabled=True,
    )
    assert "studio_voice" in captured, "apply_speak must pass studio_voice to synthesize_stream"
    assert isinstance(captured["studio_voice"], str)
    assert captured["studio_voice"], "studio_voice must be a non-empty string"


@pytest.mark.asyncio
async def test_speak_surfaces_played_and_listeners(
    isolated_mapping,
    stubbed_state,
) -> None:
    """Successful synth + fan-out reports played=True and listener count."""
    _backend, listener, _gate = stubbed_state
    result = await sp_api.apply_speak(
        label_id="volume_up",
        language="en",
        overrides_inline=None,
        respect_enabled=True,
    )
    assert result["ok"] is True
    assert result.get("played") is True
    assert result.get("listeners") == 1
    assert result.get("bytes", 0) > 44  # RIFF header + at least one PCM frame
    assert len(listener.received) == 1


@pytest.mark.asyncio
async def test_speak_reports_no_listeners(
    isolated_mapping,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When there's no registered local-sink listener (e.g. no meeting
    active and no room-sink configured), apply_speak must report
    played=False with a clear reason instead of silently succeeding."""
    backend = _FakeTTSBackend()
    gate: asyncio.Queue = asyncio.Queue(maxsize=1)
    gate.put_nowait(object())
    monkeypatch.setattr(state, "tts_backend", backend)
    monkeypatch.setattr(state, "_audio_out_clients", [])
    monkeypatch.setattr(state, "tts_dispatch_gate", gate)

    result = await sp_api.apply_speak(
        label_id="volume_up",
        language="en",
        overrides_inline=None,
        respect_enabled=True,
    )
    assert result["ok"] is True
    assert result.get("played") is False
    assert "no local-sink listeners" in (result.get("reason") or "")


@pytest.mark.asyncio
async def test_speak_reports_empty_synthesis(
    isolated_mapping,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the backend yields no chunks (e.g. voice resolution failed
    upstream), the response must reflect that — not lie with played=True."""

    class _SilentBackend:
        sample_rate = 24000

        async def synthesize_stream(self, *, text, language, **kwargs):
            if False:  # pragma: no cover — generator with zero yields
                yield np.array([], dtype=np.float32)

    listener = _FakeListener()
    gate: asyncio.Queue = asyncio.Queue(maxsize=1)
    gate.put_nowait(object())
    monkeypatch.setattr(state, "tts_backend", _SilentBackend())
    monkeypatch.setattr(state, "_audio_out_clients", [listener])
    monkeypatch.setattr(state, "tts_dispatch_gate", gate)

    result = await sp_api.apply_speak(
        label_id="volume_up",
        language="en",
        overrides_inline=None,
        respect_enabled=True,
    )
    assert result["ok"] is True
    assert result.get("played") is False
    assert "no audio" in (result.get("reason") or "").lower()
    assert listener.received == []
