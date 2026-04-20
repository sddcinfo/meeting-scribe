"""Tests for VllmTranslateBackend's LRU cache fast-path.

The cache skips the vLLM round-trip when the exact (src, tgt, text)
tuple was just translated. Targets the ~15-20 % repeated-filler phrases
seen in real meeting logs.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from meeting_scribe.backends.translate_vllm import VllmTranslateBackend


@pytest.fixture
def backend():
    be = VllmTranslateBackend(base_url="http://localhost:8010", model="test")
    # Fake client — we never actually hit the wire.
    be._client = AsyncMock()
    return be


@pytest.mark.asyncio
async def test_cache_miss_then_hit(backend):
    payload_calls = 0

    async def fake_post(url, json):
        nonlocal payload_calls
        payload_calls += 1
        m = AsyncMock()
        m.raise_for_status = lambda: None
        m.status_code = 200
        m.json = lambda: {
            "choices": [{"message": {"content": "hello world"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        return m

    backend._client.post = fake_post

    out1 = await backend.translate("こんにちは", "ja", "en")
    out2 = await backend.translate("こんにちは", "ja", "en")
    assert out1 == "hello world"
    assert out2 == "hello world"
    assert payload_calls == 1, "second call should be served from LRU cache"
    assert backend._cache_hits == 1
    assert backend._cache_misses == 1


@pytest.mark.asyncio
async def test_cache_normalizes_whitespace(backend):
    payload_calls = 0

    async def fake_post(url, json):
        nonlocal payload_calls
        payload_calls += 1
        m = AsyncMock()
        m.raise_for_status = lambda: None
        m.status_code = 200
        m.json = lambda: {
            "choices": [{"message": {"content": "yes"}}],
            "usage": {},
        }
        return m

    backend._client.post = fake_post

    out1 = await backend.translate("  はい  ", "ja", "en")
    out2 = await backend.translate("はい", "ja", "en")
    out3 = await backend.translate("はい\t", "ja", "en")
    assert out1 == out2 == out3 == "yes"
    assert payload_calls == 1


@pytest.mark.asyncio
async def test_cache_bounded(backend):
    backend._cache_max = 3
    calls = 0

    async def fake_post(url, json):
        nonlocal calls
        calls += 1
        msg = f"t{calls}"
        m = AsyncMock()
        m.raise_for_status = lambda: None
        m.status_code = 200
        m.json = lambda: {
            "choices": [{"message": {"content": msg}}],
            "usage": {},
        }
        return m

    backend._client.post = fake_post

    # 4 distinct calls — oldest should be evicted.
    for w in ("a", "b", "c", "d"):
        await backend.translate(w, "ja", "en")
    assert len(backend._cache) == 3
    # Cache key gained a context-fingerprint slot in Phase B3; for
    # calls without prior_context it is empty, so the historical
    # 3-tuple shape becomes ``(src, tgt, text, "")``.
    assert ("ja", "en", "a", "") not in backend._cache
    assert ("ja", "en", "d", "") in backend._cache


# ── Phase B3 — context-aware cache key collision guard ─────────


class TestContextFingerprint:
    """Regression guard for B3's context fingerprint.

    The plan's explicit rejection rationale: a naive `|`.join-and-hash
    encoding could collapse two distinct histories to the same bytes
    if an utterance contained the delimiter (or a newline, backslash,
    etc).  Switching to ``json.dumps(..., ensure_ascii=True)`` makes
    every delimiter-class byte a `\\uXXXX` escape, so no such
    collision is possible.  These cases are the ones that would have
    aliased under the rejected naive scheme.
    """

    def test_empty_context_produces_empty_fingerprint(self):
        from meeting_scribe.backends.translate_vllm import _fingerprint_prior_context

        assert _fingerprint_prior_context(None) == ""
        assert _fingerprint_prior_context([]) == ""

    def test_delimiter_characters_do_not_alias_histories(self):
        """Adversarial cases that collide under naive ``"|".join`` encoding."""
        from meeting_scribe.backends.translate_vllm import _fingerprint_prior_context

        a = [("a|b", "x")]
        b = [("a", "b|x")]
        # Naive f"{s}|{t}" would yield "a|b|x" for both.
        assert _fingerprint_prior_context(a) != _fingerprint_prior_context(b)

    def test_newlines_do_not_alias_histories(self):
        from meeting_scribe.backends.translate_vllm import _fingerprint_prior_context

        a = [("line1", "line2"), ("line3", "line4")]
        b = [("line1\nline2", "line3\nline4")]
        assert _fingerprint_prior_context(a) != _fingerprint_prior_context(b)

    def test_order_changes_change_fingerprint(self):
        from meeting_scribe.backends.translate_vllm import _fingerprint_prior_context

        a = [("s1", "t1"), ("s2", "t2")]
        b = [("s2", "t2"), ("s1", "t1")]
        # Order matters — the prompt reads "earlier → newer", so
        # rearranging must produce a different hash.
        assert _fingerprint_prior_context(a) != _fingerprint_prior_context(b)

    def test_identical_contexts_produce_identical_fingerprints(self):
        """The whole point of the fingerprint is that same history → same
        hash → cache hit."""
        from meeting_scribe.backends.translate_vllm import _fingerprint_prior_context

        ctx = [("こんにちは", "Hello"), ("ありがとう", "Thanks")]
        # Two separate list instances with the same content.
        assert _fingerprint_prior_context(ctx) == _fingerprint_prior_context(list(ctx))


@pytest.mark.asyncio
async def test_cache_hits_when_context_matches(backend, monkeypatch):
    """Same text + same prior_context → cache hit (no second backend call)."""
    tmp_log = Path("/tmp/test_cache_hit.jsonl")
    tmp_log.unlink(missing_ok=True)
    monkeypatch.setattr(
        "meeting_scribe.backends.translate_vllm._TRANS_LOG_PATH", tmp_log
    )

    call_count = 0

    async def fake_post(url, json):
        nonlocal call_count
        call_count += 1
        m = AsyncMock()
        m.raise_for_status = lambda: None
        m.status_code = 200
        m.json = lambda: {
            "choices": [{"message": {"content": f"translated-{call_count}"}}],
            "usage": {},
        }
        return m

    backend._client.post = fake_post

    ctx = [("prior-src", "prior-tgt")]
    r1 = await backend.translate("hello", "ja", "en", prior_context=ctx)
    r2 = await backend.translate("hello", "ja", "en", prior_context=ctx)
    assert r1 == r2 == "translated-1"
    assert call_count == 1


@pytest.mark.asyncio
async def test_cache_misses_when_context_differs(backend, monkeypatch):
    """Same text + different prior_context → second call goes to backend."""
    tmp_log = Path("/tmp/test_cache_miss.jsonl")
    tmp_log.unlink(missing_ok=True)
    monkeypatch.setattr(
        "meeting_scribe.backends.translate_vllm._TRANS_LOG_PATH", tmp_log
    )

    call_count = 0

    async def fake_post(url, json):
        nonlocal call_count
        call_count += 1
        m = AsyncMock()
        m.raise_for_status = lambda: None
        m.status_code = 200
        m.json = lambda: {
            "choices": [{"message": {"content": f"translated-{call_count}"}}],
            "usage": {},
        }
        return m

    backend._client.post = fake_post

    r1 = await backend.translate(
        "hello", "ja", "en", prior_context=[("x", "X")]
    )
    r2 = await backend.translate(
        "hello", "ja", "en", prior_context=[("y", "Y")]
    )
    assert r1 == "translated-1"
    assert r2 == "translated-2"
    assert call_count == 2
