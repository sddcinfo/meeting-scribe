"""Tests for TTS backend — voice caching, encoding, availability."""

from __future__ import annotations

import numpy as np
import pytest

from meeting_scribe.backends.tts_qwen3 import Qwen3TTSBackend


class TestVoiceCache:
    """Voice reference caching from conversation audio."""

    def test_cache_voice_stores_reference(self):
        backend = Qwen3TTSBackend()
        audio = np.random.randn(48000).astype(np.float32) * 0.5  # 3 seconds at 16kHz
        backend.cache_voice("speaker-1", audio)
        assert backend.has_voice("speaker-1")

    def test_cache_voice_skips_short_audio(self):
        backend = Qwen3TTSBackend()
        audio = np.random.randn(8000).astype(np.float32)  # 0.5 seconds — too short
        backend.cache_voice("speaker-1", audio)
        assert not backend.has_voice("speaker-1")

    def test_cache_voice_skips_silence(self):
        backend = Qwen3TTSBackend()
        audio = np.zeros(48000, dtype=np.float32)  # Silent
        backend.cache_voice("speaker-1", audio)
        assert not backend.has_voice("speaker-1")

    def test_cache_voice_only_first(self):
        backend = Qwen3TTSBackend()
        audio1 = np.random.randn(48000).astype(np.float32) * 0.5
        audio2 = np.random.randn(48000).astype(np.float32) * 0.8
        backend.cache_voice("speaker-1", audio1)
        backend.cache_voice("speaker-1", audio2)
        # Should keep the first one (progressive upgrades are gated by
        # a min-gap + quality margin; a second call right after won't swap)
        assert backend.has_voice("speaker-1")
        cached = backend.get_voice("speaker-1")
        assert cached is not None
        assert len(cached) <= 48000  # Best segment extraction


class TestBestSegment:
    """Extract the highest-energy segment for voice reference."""

    def test_short_audio_returned_as_is(self):
        audio = np.ones(1000, dtype=np.float32)
        result = Qwen3TTSBackend._best_segment(audio, segment_len=2000)
        assert len(result) == 1000

    def test_selects_loudest_segment(self):
        audio = np.zeros(96000, dtype=np.float32)
        # Put energy at samples 48000-72000
        audio[48000:72000] = 0.8
        result = Qwen3TTSBackend._best_segment(audio, segment_len=24000)
        assert float(np.mean(np.abs(result))) > 0.5


class TestVoiceRefEncoding:
    """Base64 WAV encoding for API transport."""

    def test_encode_produces_data_uri(self):
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        uri = Qwen3TTSBackend._encode_voice_ref(audio)
        assert uri.startswith("data:audio/wav;base64,")
        import base64

        raw = base64.b64decode(uri.split(",", 1)[1])
        assert raw[:4] == b"RIFF"

    def test_encode_decode_roundtrip(self):
        import base64
        import io
        import wave

        original = np.random.randn(16000).astype(np.float32) * 0.5
        uri = Qwen3TTSBackend._encode_voice_ref(original, sample_rate=16000)
        raw = base64.b64decode(uri.split(",", 1)[1])
        buf = io.BytesIO(raw)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 16000


class TestLanguageGate:
    """TTS synthesis is gated on tts_native language support."""

    @pytest.mark.asyncio
    async def test_synthesize_stream_skips_non_tts_language(self):
        """Non-TTS languages (e.g. Dutch) yield no audio chunks."""
        backend = Qwen3TTSBackend(vllm_url="http://localhost:8002")
        backend._mode = "vllm"
        backend._urls = ["http://localhost:8002"]
        chunks = []
        async for chunk in backend.synthesize_stream(
            text="Hallo wereld",
            language="nl",
            studio_voice="aiden",
        ):
            chunks.append(chunk)
        assert chunks == []

    @pytest.mark.asyncio
    async def test_synthesize_returns_empty_for_non_tts_language(self):
        """The synthesize wrapper returns empty array for non-TTS languages."""
        backend = Qwen3TTSBackend(vllm_url="http://localhost:8002")
        backend._mode = "vllm"
        backend._urls = ["http://localhost:8002"]
        result = await backend.synthesize(
            text="مرحبا",
            language="ar",
        )
        assert len(result) == 0


class TestAvailability:
    """Backend availability before start."""

    def test_not_available_before_start(self):
        backend = Qwen3TTSBackend()
        assert not backend.available

    @pytest.mark.asyncio
    async def test_disabled_without_url(self):
        backend = Qwen3TTSBackend(vllm_url=None)
        await backend.start()
        assert not backend.available
