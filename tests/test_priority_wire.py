"""Verify scribe-main sends the expected `priority` field on outgoing
vLLM requests. Guards against priority scheduling being a no-op on the
consolidated Omni instance, and against the 2026-04-14 regression
where summary.py and refinement.py were sending -5 and -8 respectively
instead of -10 (priority inversion that let refinement block summary).

Priority ladder (lower wins):
  -20  live ASR partials (user hears silence otherwise)
  -10  live translate, live TTS, summary (post-meeting), refinement
    0  reprocess (default — always preempted by live)
   10  autosre coding agent
   20  plan-review runner
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pytest

from meeting_scribe.backends.asr_vllm import VllmASRBackend
from meeting_scribe.backends.translate_vllm import VllmTranslateBackend
from meeting_scribe.backends.tts_qwen3 import Qwen3TTSBackend


def _fake_response(payload: dict):
    m = AsyncMock()
    m.status_code = 200
    m.raise_for_status = lambda: None
    m.json = lambda: payload
    return m


class TestPriorityHeaders:
    @pytest.mark.asyncio
    async def test_asr_sends_priority_minus_20(self):
        be = VllmASRBackend(base_url="http://localhost:8003", languages=("ja", "en"))
        captured: dict = {}

        async def fake_post(path, json, **kw):
            captured.update(json)
            return _fake_response({
                "choices": [{"message": {"content": "language English<asr_text>hello"}}]
            })

        be._client = AsyncMock()
        be._client.post = fake_post
        be._model = "Qwen/Qwen3-ASR-1.7B"
        # Force immediate flush (bypass buffering + VAD gating for the wire test).
        be._buffer_threshold = 0

        # Nonzero audio above VAD threshold so rms check passes.
        audio = np.full(16000, 0.3, dtype=np.float32)
        async for _ in be.process_audio(audio, sample_offset=0):
            break

        assert captured.get("priority") == -20, (
            f"ASR must send priority=-20; got {captured.get('priority')}"
        )

    @pytest.mark.asyncio
    async def test_translate_sends_priority_minus_10(self):
        be = VllmTranslateBackend(base_url="http://localhost:8010", model="qwen3.5-int4")
        captured: dict = {}

        async def fake_post(url, json, **kw):
            captured.update(json)
            return _fake_response({"choices": [{"message": {"content": "hello"}}]})

        be._client = AsyncMock()
        be._client.post = fake_post

        await be.translate("こんにちは", "ja", "en")
        assert captured.get("priority") == -10

    @pytest.mark.asyncio
    async def test_tts_sends_priority_minus_10(self):
        be = Qwen3TTSBackend(vllm_url="http://localhost:8002")
        be._mode = "vllm"

        captured: dict = {}

        class _Streamer:
            async def __aenter__(self_inner):
                return self_inner
            async def __aexit__(self_inner, *a):
                return False
            def raise_for_status(self_inner):
                pass
            async def aiter_bytes(self_inner):
                if False:
                    yield b""
                return

        class _FakeClient:
            is_closed = False
            def stream(self_inner, method, url, json=None, **kw):
                captured.update(json or {})
                return _Streamer()
            async def aclose(self_inner):
                pass

        be._http_client = _FakeClient()
        async for _ in be.synthesize_stream(
            text="hello", language="en", studio_voice="aiden"
        ):
            pass

        assert captured.get("priority") == -10


class TestSummaryPriority:
    """summary.py generates post-meeting summaries via the translate
    vLLM instance. It should run at the same tier as live translation
    (-10) so it preempts the coding agent but never blocks live paths.
    The old -5 value created a priority inversion vs refinement (-8),
    which left summary queued behind refinement calls."""

    def test_summary_source_sends_priority_minus_10(self):
        from pathlib import Path
        src = (Path(__file__).parent.parent / "src" / "meeting_scribe" / "summary.py").read_text()
        # generate_summary defaults to priority=-10 (same tier as live
        # translation). The value is parameterized via _call_vllm_summary
        # so the literal appears as a default argument, not inline JSON.
        assert "priority: int = -10" in src, (
            "summary.py generate_summary must default to priority=-10 — "
            "previous bug was -5 which allowed refinement (-8) to block it"
        )
        assert '"priority": -5' not in src
        assert '"priority": -8' not in src


class TestRefinementPriority:
    """refinement.py runs the trailing re-ASR / re-translate worker
    during a live meeting. It should match live translation (-10) so
    summary (-10) never blocks behind it and coding (10) never blocks
    it. Old values: -8 (ASR pass) and -8 (translate pass)."""

    def test_refinement_source_sends_priority_minus_10(self):
        from pathlib import Path
        src = (Path(__file__).parent.parent / "src" / "meeting_scribe" / "refinement.py").read_text()
        assert '"priority": -10' in src
        assert '"priority": -8' not in src
        # There are TWO priority fields in refinement (ASR pass + translate pass);
        # both must be -10.
        assert src.count('"priority": -10') >= 2, (
            "expected at least 2 priority=-10 fields in refinement.py "
            "(one for ASR, one for translate)"
        )


class TestReprocessNoPriority:
    """Reprocess is the background bulk re-ASR + re-translate of a
    finished meeting. It must send NO priority field (defaults to 0)
    so live paths at -10/-20 always preempt it."""

    def test_reprocess_source_no_priority(self):
        from pathlib import Path
        src = (Path(__file__).parent.parent / "src" / "meeting_scribe" / "reprocess.py").read_text()
        assert '"priority"' not in src, (
            "reprocess.py must NOT set a priority field — it should "
            "run at default (0) so every live path (-10/-20) preempts it"
        )
