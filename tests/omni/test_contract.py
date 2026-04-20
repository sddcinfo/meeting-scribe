"""Contract tests — production-equivalent payloads against the Omni spike.

These tests verify that Qwen3-Omni-30B-A3B speaks the same wire protocol
the existing backends already consume, BEFORE Phase D routing cuts over.
Each test skips cleanly when the Omni container isn't reachable, so the
suite stays green in CI without it.

Set SCRIBE_OMNI_URL=http://localhost:8032 (or similar) to enable these.
"""
from __future__ import annotations

import base64
import io
import os
import wave

import httpx
import numpy as np
import pytest

OMNI_URL = os.environ.get("SCRIBE_OMNI_URL", "")
pytestmark = pytest.mark.skipif(
    not OMNI_URL, reason="Set SCRIBE_OMNI_URL to exercise Omni contract tests"
)


def _short_wav_b64(seconds: float = 1.5, sample_rate: int = 16000) -> str:
    t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _short_wav_data_uri(seconds: float = 6.0, sample_rate: int = 24_000) -> str:
    """Boundary-sized ref_audio — mirrors Qwen3TTSBackend._REF_AUDIO_MAX_*."""
    t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 330 * t).astype(np.float32)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.mark.asyncio
async def test_health_models():
    async with httpx.AsyncClient(timeout=10) as c:
        h = await c.get(f"{OMNI_URL}/health")
        assert h.status_code == 200
        m = await c.get(f"{OMNI_URL}/v1/models")
        assert m.status_code == 200


@pytest.mark.asyncio
async def test_tts_studio_stream_pcm():
    """Studio voice returns int16 PCM stream decoding to valid samples."""
    body = {
        "model": "qwen3-tts",
        "input": "Hello from the contract test.",
        "voice": "aiden",
        "stream": True,
        "response_format": "pcm",
        "priority": -10,
    }
    async with httpx.AsyncClient(timeout=30) as c, c.stream(
        "POST", f"{OMNI_URL}/v1/audio/speech", json=body
    ) as resp:
        resp.raise_for_status()
        bytes_total = 0
        async for chunk in resp.aiter_bytes():
            bytes_total += len(chunk)
        # Should return at least 0.5 s of PCM.
        assert bytes_total >= 24_000 * 2 * 0.5


@pytest.mark.asyncio
async def test_tts_cloned_ref_audio():
    """Boundary-size inline ref_audio data URI is accepted and streamed."""
    body = {
        "model": "qwen3-tts",
        "input": "Cloned voice contract test.",
        "voice": "custom",
        "ref_audio": _short_wav_data_uri(),
        "stream": True,
        "response_format": "pcm",
        "priority": -10,
    }
    async with httpx.AsyncClient(timeout=30) as c, c.stream(
        "POST", f"{OMNI_URL}/v1/audio/speech", json=body
    ) as resp:
        resp.raise_for_status()
        bytes_total = 0
        async for chunk in resp.aiter_bytes():
            bytes_total += len(chunk)
        assert bytes_total > 0


@pytest.mark.asyncio
async def test_asr_chat_completion():
    """OpenAI-compatible /v1/chat/completions input_audio path (matches VllmASRBackend)."""
    body = {
        "model": "qwen3-omni",
        "messages": [
            {"role": "system", "content": "Transcribe in the spoken language."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": _short_wav_b64(), "format": "wav"},
                    }
                ],
            },
        ],
        "max_tokens": 128,
        "temperature": 0.0,
        "priority": -20,
    }
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{OMNI_URL}/v1/chat/completions", json=body)
        assert r.status_code == 200
        data = r.json()
        assert "choices" in data


@pytest.mark.asyncio
async def test_translate_chat_completion():
    body = {
        "model": "qwen3-omni",
        "messages": [
            {"role": "system", "content": "Translate from Japanese to English."},
            {"role": "user", "content": "こんにちは、世界"},
        ],
        "max_tokens": 128,
        "temperature": 0.0,
        "priority": -10,
    }
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{OMNI_URL}/v1/chat/completions", json=body)
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_error_semantics_bad_voice():
    body = {
        "model": "qwen3-tts",
        "input": "bad voice",
        "voice": "not_a_real_voice_xyz",
        "stream": True,
        "response_format": "pcm",
    }
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(f"{OMNI_URL}/v1/audio/speech", json=body)
        # Must be a clean 4xx — not a hang and not a 5xx.
        assert 400 <= r.status_code < 500


@pytest.mark.asyncio
async def test_streaming_cancellation():
    """Client disconnect mid-stream must not wedge the server's next request."""
    body = {
        "model": "qwen3-tts",
        "input": "A reasonably long sentence that streams for a while.",
        "voice": "aiden",
        "stream": True,
        "response_format": "pcm",
    }
    async with httpx.AsyncClient(timeout=30) as c:
        async with c.stream("POST", f"{OMNI_URL}/v1/audio/speech", json=body) as resp:
            resp.raise_for_status()
            got = 0
            async for chunk in resp.aiter_bytes():
                got += len(chunk)
                if got > 4096:
                    break  # disconnect

        # Immediately issue a fresh request — server must still be responsive.
        body2 = dict(body, input="Follow-up after cancellation.")
        r2 = await c.post(f"{OMNI_URL}/v1/audio/speech", json=body2)
        assert r2.status_code == 200
