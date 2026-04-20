"""WebSocket audio-out lifecycle tests.

Exercises the extracted lifecycle helpers ``_create_audio_out_session``,
``_unregister_audio_out_client``, and ``_handle_audio_out_message`` plus
the fan-out delivery path with multiple listeners in different states.

Hermetic — uses the same FakeWs stub as ``test_audio_out_protocol.py``.
No TestClient, no lifespan, no backends, no threading.

Tier A: Lifecycle helpers — session creation, teardown, message parsing,
        format negotiation, encoder lifecycle, protocol error handling.
Tier B: Fan-out integration — multi-format delivery, language filtering,
        grace-window flush, passthrough filtering.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("av")

from meeting_scribe import server

SAMPLE_RATE_TTS = 24000
SAMPLE_RATE_PASSTHROUGH = 16000


class FakeWs:
    """Minimal async WebSocket stub — mirrors test_audio_out_protocol.py."""

    def __init__(self, host: str = "10.42.0.99", port: int = 12345) -> None:
        self.binary_frames: list[bytes] = []
        self.text_frames: list[str] = []

        class _Client:
            def __init__(self, h: str, p: int) -> None:
                self.host = h
                self.port = p

        self.client = _Client(host, port)

    async def send_bytes(self, data: bytes) -> None:
        self.binary_frames.append(data)

    async def send_text(self, data: str) -> None:
        self.text_frames.append(data)


def _sine_pcm(duration_s: float, rate: int = SAMPLE_RATE_TTS) -> np.ndarray:
    n = int(duration_s * rate)
    t = np.arange(n, dtype=np.float32) / rate
    return (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_pref(audio_format: str | None = None) -> server.ClientSession:
    return server.ClientSession(
        preferred_language="en",
        send_audio=True,
        interpretation_mode="translation",
        voice_mode="studio",
        audio_format=audio_format,
        grace_deadline=time.monotonic() + server._AUDIO_FORMAT_GRACE_S,
    )


def _register_ws(ws, pref):
    """Helper to populate server globals for a FakeWs (cleanup in finally)."""
    server._audio_out_clients.add(ws)
    server._audio_out_prefs[ws] = pref


def _cleanup_ws(*ws_list):
    """Remove FakeWs entries from server globals."""
    for ws in ws_list:
        server._audio_out_clients.discard(ws)
        server._audio_out_prefs.pop(ws, None)


# ═══════════════════════════════════════════════════════════════════════
# Tier A — Lifecycle helpers
# ═══════════════════════════════════════════════════════════════════════


def test_create_session_adds_to_prefs():
    ws = FakeWs()
    try:
        pref = server._create_audio_out_session(ws)
        assert ws in server._audio_out_prefs
        assert pref.audio_format is None
        assert pref.send_audio is True
        assert pref.grace_deadline > time.monotonic() - 1.0
    finally:
        server._audio_out_prefs.pop(ws, None)


def test_unregister_closes_encoder_cleans_dicts():
    ws = FakeWs()
    pref = _make_pref("mse-fmp4-aac")
    mock_encoder = MagicMock()
    pref.mse_encoder = mock_encoder
    _register_ws(ws, pref)

    server._unregister_audio_out_client(ws)

    assert ws not in server._audio_out_clients
    assert ws not in server._audio_out_prefs
    mock_encoder.close.assert_called_once()


def test_unregister_idempotent():
    ws = FakeWs()
    pref = _make_pref("wav-pcm")
    _register_ws(ws, pref)

    server._unregister_audio_out_client(ws)
    # Second call should not raise
    server._unregister_audio_out_client(ws)

    assert ws not in server._audio_out_clients
    assert ws not in server._audio_out_prefs


def test_handle_message_set_format_mse():
    ws = FakeWs()
    pref = _make_pref(None)
    _register_ws(ws, pref)
    try:
        ack = _run(server._handle_audio_out_message(
            ws, pref, json.dumps({"type": "set_format", "format": "mse-fmp4-aac"})
        ))
        assert ack is not None
        parsed = json.loads(ack)
        assert parsed == {"type": "format_ack", "format": "mse-fmp4-aac"}
        assert pref.audio_format == "mse-fmp4-aac"
    finally:
        _cleanup_ws(ws)


def test_handle_message_set_format_wav():
    ws = FakeWs()
    pref = _make_pref(None)
    _register_ws(ws, pref)
    try:
        ack = _run(server._handle_audio_out_message(
            ws, pref, json.dumps({"type": "set_format", "format": "wav-pcm"})
        ))
        assert ack is not None
        parsed = json.loads(ack)
        assert parsed == {"type": "format_ack", "format": "wav-pcm"}
        assert pref.audio_format == "wav-pcm"
    finally:
        _cleanup_ws(ws)


def test_handle_message_invalid_format():
    ws = FakeWs()
    pref = _make_pref(None)
    _register_ws(ws, pref)
    try:
        ack = _run(server._handle_audio_out_message(
            ws, pref, json.dumps({"type": "set_format", "format": "opus-ogg"})
        ))
        assert ack is None
        assert pref.audio_format is None
    finally:
        _cleanup_ws(ws)


def test_handle_message_set_language():
    ws = FakeWs()
    pref = _make_pref("wav-pcm")
    _register_ws(ws, pref)
    try:
        ack = _run(server._handle_audio_out_message(
            ws, pref, json.dumps({"type": "set_language", "language": "ja"})
        ))
        assert ack is None
        assert pref.preferred_language == "ja"
    finally:
        _cleanup_ws(ws)


def test_handle_message_set_mode_and_voice():
    ws = FakeWs()
    pref = _make_pref("wav-pcm")
    _register_ws(ws, pref)
    try:
        ack = _run(server._handle_audio_out_message(
            ws, pref, json.dumps({"type": "set_mode", "mode": "full"})
        ))
        assert ack is None
        assert pref.interpretation_mode == "full"

        ack = _run(server._handle_audio_out_message(
            ws, pref, json.dumps({"type": "set_voice", "voice": "cloned"})
        ))
        assert ack is None
        assert pref.voice_mode == "cloned"
    finally:
        _cleanup_ws(ws)


def test_handle_message_format_switch_closes_old_encoder():
    ws = FakeWs()
    pref = _make_pref("mse-fmp4-aac")
    mock_encoder = MagicMock()
    pref.mse_encoder = mock_encoder
    _register_ws(ws, pref)
    try:
        ack = _run(server._handle_audio_out_message(
            ws, pref, json.dumps({"type": "set_format", "format": "wav-pcm"})
        ))
        assert ack is not None
        assert pref.audio_format == "wav-pcm"
        mock_encoder.close.assert_called_once()
        assert pref.mse_encoder is None
    finally:
        _cleanup_ws(ws)


def test_handle_message_malformed_json():
    ws = FakeWs()
    pref = _make_pref("wav-pcm")
    original_format = pref.audio_format
    ack = _run(server._handle_audio_out_message(ws, pref, "not json {{{"))
    assert ack is None
    assert pref.audio_format == original_format


def test_handle_message_missing_type():
    ws = FakeWs()
    pref = _make_pref("wav-pcm")
    ack = _run(server._handle_audio_out_message(
        ws, pref, json.dumps({"format": "mse-fmp4-aac"})
    ))
    assert ack is None
    assert pref.audio_format == "wav-pcm"


def test_handle_message_unknown_type():
    ws = FakeWs()
    pref = _make_pref("wav-pcm")
    original_lang = pref.preferred_language
    ack = _run(server._handle_audio_out_message(
        ws, pref, json.dumps({"type": "set_volume", "volume": 0.5})
    ))
    assert ack is None
    assert pref.preferred_language == original_lang


def test_handle_message_wrong_field_types():
    ws = FakeWs()
    pref = _make_pref(None)
    ack = _run(server._handle_audio_out_message(
        ws, pref, json.dumps({"type": "set_format", "format": 42})
    ))
    assert ack is None
    assert pref.audio_format is None


def test_handle_message_missing_payload_fields():
    """Recognized setter messages with missing payload fields must not crash."""
    ws = FakeWs()

    # set_format with no "format" key → .get("format","") → "" not valid → no-op
    pref = _make_pref(None)
    ack = _run(server._handle_audio_out_message(
        ws, pref, json.dumps({"type": "set_format"})
    ))
    assert ack is None
    assert pref.audio_format is None

    # set_language with no "language" key → .get("language","") → _norm_lang("") → falsy → no mutation
    pref = _make_pref("wav-pcm")
    pref.preferred_language = "en"
    ack = _run(server._handle_audio_out_message(
        ws, pref, json.dumps({"type": "set_language"})
    ))
    assert ack is None
    assert pref.preferred_language == "en"

    # set_mode with no "mode" key → .get("mode","translation") → "translation" (valid default)
    pref.interpretation_mode = "full"
    ack = _run(server._handle_audio_out_message(
        ws, pref, json.dumps({"type": "set_mode"})
    ))
    assert ack is None
    assert pref.interpretation_mode == "translation"  # reset to default

    # set_voice with no "voice" key → .get("voice","studio") → "studio" (valid default)
    pref.voice_mode = "cloned"
    ack = _run(server._handle_audio_out_message(
        ws, pref, json.dumps({"type": "set_voice"})
    ))
    assert ack is None
    assert pref.voice_mode == "studio"  # reset to default


# ═══════════════════════════════════════════════════════════════════════
# Tier B — Fan-out integration
# ═══════════════════════════════════════════════════════════════════════


def test_concurrent_mixed_format_fanout():
    """Three listeners (MSE, WAV, grace) receive correct formats."""
    ws_mse = FakeWs(host="10.42.0.1", port=1001)
    ws_wav = FakeWs(host="10.42.0.2", port=1002)
    ws_grace = FakeWs(host="10.42.0.3", port=1003)

    pref_mse = _make_pref("mse-fmp4-aac")
    pref_wav = _make_pref("wav-pcm")
    pref_grace = _make_pref(None)  # still in grace window

    _register_ws(ws_mse, pref_mse)
    _register_ws(ws_wav, pref_wav)
    _register_ws(ws_grace, pref_grace)
    try:
        pcm = _sine_pcm(0.500)

        async def _drive():
            await server._send_audio_to_listeners(pcm, "en", "studio")

        _run(_drive())

        # MSE listener got init (0x49) + at least one frame
        assert len(ws_mse.binary_frames) >= 1
        assert ws_mse.binary_frames[0][:1] == b"\x49"

        # WAV listener got unprefixed RIFF
        assert len(ws_wav.binary_frames) == 1
        assert ws_wav.binary_frames[0][:4] == b"RIFF"

        # Grace listener got nothing on the wire — audio is buffered
        assert len(ws_grace.binary_frames) == 0
        assert len(pref_grace.pending_audio) >= 1
    finally:
        if pref_mse.mse_encoder:
            pref_mse.mse_encoder.close()
        _cleanup_ws(ws_mse, ws_wav, ws_grace)


def test_language_filtering_in_fanout():
    """Delivery with target_language='en' only reaches the en listener."""
    ws_en = FakeWs(host="10.42.0.1", port=2001)
    ws_ja = FakeWs(host="10.42.0.2", port=2002)

    pref_en = _make_pref("wav-pcm")
    pref_en.preferred_language = "en"
    pref_ja = _make_pref("wav-pcm")
    pref_ja.preferred_language = "ja"

    _register_ws(ws_en, pref_en)
    _register_ws(ws_ja, pref_ja)
    try:
        pcm = _sine_pcm(0.100)

        async def _drive():
            sent = await server._send_audio_to_listeners(pcm, "en", "studio")
            return sent

        sent = _run(_drive())

        assert sent == 1
        assert len(ws_en.binary_frames) == 1
        assert ws_en.binary_frames[0][:4] == b"RIFF"
        assert len(ws_ja.binary_frames) == 0
    finally:
        _cleanup_ws(ws_en, ws_ja)


def test_grace_window_buffers_then_flush_delivers():
    """Listener in grace gets buffered; flush after set_format delivers."""
    ws = FakeWs()
    pref = _make_pref(None)
    _register_ws(ws, pref)
    try:
        pcm_a = _sine_pcm(0.100)
        pcm_b = _sine_pcm(0.100)

        async def _drive():
            await server._deliver_audio_to_listener(ws, pref, pcm_a, SAMPLE_RATE_TTS, None)
            await server._deliver_audio_to_listener(ws, pref, pcm_b, SAMPLE_RATE_TTS, None)
            assert len(ws.binary_frames) == 0
            assert len(pref.pending_audio) == 2
            # Resolve format
            pref.audio_format = "wav-pcm"
            await server._flush_pending_audio(ws, pref)

        _run(_drive())
        assert len(ws.binary_frames) == 2
        for f in ws.binary_frames:
            assert f[:4] == b"RIFF"
    finally:
        _cleanup_ws(ws)


def test_passthrough_filters_by_mode_and_language():
    """_send_passthrough_audio only reaches 'full' mode listeners with matching language."""
    ws_full_en = FakeWs(host="10.42.0.1", port=3001)
    ws_trans_en = FakeWs(host="10.42.0.2", port=3002)
    ws_full_ja = FakeWs(host="10.42.0.3", port=3003)

    pref_full_en = _make_pref("wav-pcm")
    pref_full_en.interpretation_mode = "full"
    pref_full_en.preferred_language = "en"

    pref_trans_en = _make_pref("wav-pcm")
    pref_trans_en.interpretation_mode = "translation"
    pref_trans_en.preferred_language = "en"

    pref_full_ja = _make_pref("wav-pcm")
    pref_full_ja.interpretation_mode = "full"
    pref_full_ja.preferred_language = "ja"

    _register_ws(ws_full_en, pref_full_en)
    _register_ws(ws_trans_en, pref_trans_en)
    _register_ws(ws_full_ja, pref_full_ja)
    try:
        pcm = _sine_pcm(0.100, rate=SAMPLE_RATE_PASSTHROUGH)

        async def _drive():
            await server._send_passthrough_audio(pcm, "en")

        _run(_drive())

        # Only full+en should receive it
        assert len(ws_full_en.binary_frames) == 1
        assert ws_full_en.binary_frames[0][:4] == b"RIFF"
        assert len(ws_trans_en.binary_frames) == 0
        assert len(ws_full_ja.binary_frames) == 0
    finally:
        _cleanup_ws(ws_full_en, ws_trans_en, ws_full_ja)
