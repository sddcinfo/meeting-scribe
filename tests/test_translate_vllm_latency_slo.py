"""Latency-SLO invariants for the live translate backend.

The live-utterance translation path has a strict real-time budget (new
speaker utterance every ~400-600 ms) and cannot afford Qwen's reasoning
mode, which routinely adds 3-5 s per call while the model emits its
``<think>...</think>`` block.  Post-process paths (slide deck
translation during reinsertion, post-meeting summaries, eager summary
pre-compute) can adopt reasoning since they run off the hot path.

This test pins the invariant at the call-site level: every
``VllmTranslateBackend.translate`` request MUST set
``chat_template_kwargs.enable_thinking = False``.  If someone flips it
to ``True`` (or drops the flag, which would inherit the server default
that may flip to True on future nightly images), this test fails loud.

Plan: ``~/.claude/plans/sprightly-prancing-valiant.md``.
Auto-memory: ``project_scribe_reasoning_path_split``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from meeting_scribe.backends.translate_vllm import VllmTranslateBackend


@pytest.fixture
def backend():
    be = VllmTranslateBackend(base_url="http://localhost:8010", model="test")
    be._client = AsyncMock()
    return be


def _fake_ok_response(text: str = "Hello!"):
    """Return a minimal OpenAI-compat chat-completion response.

    httpx's ``raise_for_status`` and ``json`` are sync methods; using
    plain callables here avoids unawaited-AsyncMock warnings.
    """
    resp = AsyncMock()
    resp.raise_for_status = lambda: None
    resp.status_code = 200
    resp.json = lambda: {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    }
    return resp


@pytest.mark.asyncio
async def test_live_translate_disables_reasoning(backend):
    """enable_thinking MUST be False on every live translate call.

    Reasoning mode (enable_thinking=True) adds multi-second latency that
    breaks the real-time meeting SLO.  Regression guard for anyone who
    flips the flag thinking it will improve quality — reasoning
    variants belong on the batch/post-process path instead.
    """
    seen_payloads = []

    async def fake_post(url, json):
        seen_payloads.append(json)
        return _fake_ok_response()

    backend._client.post = fake_post

    await backend.translate("こんにちは", source_language="ja", target_language="en")

    assert seen_payloads, "backend did not call post()"
    payload = seen_payloads[0]
    assert "chat_template_kwargs" in payload, (
        "live translate must pin chat_template_kwargs — "
        "omitting lets the server default (possibly thinking=True) win"
    )
    assert payload["chat_template_kwargs"].get("enable_thinking") is False, (
        "LATENCY SLO VIOLATION: live translate path must send "
        "enable_thinking=False so reasoning doesn't add multi-second "
        "latency to each utterance.  Reasoning variants must only be "
        "used on post-process paths (slides, summaries).  See auto-memory "
        "project_scribe_reasoning_path_split."
    )


@pytest.mark.asyncio
async def test_live_translate_streaming_disabled(backend):
    """Streaming MUST be disabled on the live path — the translate_fn
    awaits the full response; enabling stream would break the JSON
    path in the current backend.

    This is a secondary invariant in the same file because it's part of
    the same "live-path payload contract" that regresses together."""
    seen_payloads = []

    async def fake_post(url, json):
        seen_payloads.append(json)
        return _fake_ok_response()

    backend._client.post = fake_post

    await backend.translate("hi", source_language="en", target_language="ja")

    assert seen_payloads[0].get("stream") is False, (
        "live translate path must send stream=False — the backend's "
        "response handling assumes a single JSON response, not SSE."
    )


@pytest.mark.asyncio
async def test_live_translate_priority_preempts_coding(backend):
    """Priority must be the highest (most-negative) value so translation
    always preempts coding/review workloads on the shared GPU."""
    seen_payloads = []

    async def fake_post(url, json):
        seen_payloads.append(json)
        return _fake_ok_response()

    backend._client.post = fake_post

    await backend.translate("test", source_language="en", target_language="ja")

    priority = seen_payloads[0].get("priority")
    assert priority is not None, "live translate must set an explicit priority"
    # Coding agent runs at +10, plan-review at +20. Translation must be
    # strictly lower (higher priority under vLLM's --scheduling-policy=priority).
    assert priority < 10, (
        f"LATENCY SLO VIOLATION: live translate priority {priority} is not "
        "strictly lower than the coding agent (+10); translation must "
        "preempt coding so real-time audio is never blocked behind a "
        "long coding prefill."
    )


@pytest.mark.asyncio
async def test_en_ja_live_path_passes_no_prior_context(backend):
    """EN→JA live translations MUST leave ``prior_context`` at ``None``.

    The sweep (reports/context_window_sweep/2026-04-19/summary.md:37-42)
    showed JA→EN benefits materially from a rolling context window but
    EN→JA only at a much lower rate. B1 intentionally scopes the
    live-context feature to JA→EN; flipping EN→JA on later would need
    its own measurement. This regression guard fails loud if anyone
    threads prior_context through the EN→JA live path.
    """
    from meeting_scribe.translation.queue import TranslationQueue

    seen_calls = []

    class _RecordingBackend:
        async def translate(
            self,
            text,
            source_language,
            target_language,
            prior_context=None,
            meeting_id=None,
        ):
            seen_calls.append(
                {
                    "source_language": source_language,
                    "target_language": target_language,
                    "prior_context": prior_context,
                }
            )
            return f"{text}::{target_language}"

    from meeting_scribe.models import TranscriptEvent

    queue = TranslationQueue(on_result=None, languages=("en", "ja"), concurrency=1, timeout=2.0)
    await queue.start(_RecordingBackend())
    try:
        # Non-zero maxlen so we would accumulate if the direction-gate broke.
        queue.bind_meeting("mtg-slo", history_maxlen=2)
        for i, text in enumerate(["one", "two", "three"]):
            await queue.submit(
                TranscriptEvent(
                    segment_id=f"s-{i}",
                    revision=0,
                    is_final=True,
                    start_ms=0,
                    end_ms=1000,
                    language="en",
                    text=text,
                )
            )
        await queue.flush_merge_gate()
        # Drain
        for _ in range(100):
            if queue.is_idle():
                break
            import asyncio as _asyncio

            await _asyncio.sleep(0.01)

        assert len(seen_calls) == 3
        for call in seen_calls:
            assert call["source_language"] == "en"
            assert call["target_language"] == "ja"
            assert call["prior_context"] is None, (
                "B1 inverse-direction invariant: EN→JA live path must remain "
                f"stateless. Saw prior_context={call['prior_context']!r}"
            )
    finally:
        await queue.stop()
