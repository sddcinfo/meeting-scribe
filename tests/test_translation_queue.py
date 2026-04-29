"""Tests for translation queue — merge gate, target language routing, cancellation."""

from __future__ import annotations

import asyncio

import pytest

from meeting_scribe.models import TranscriptEvent, TranslationStatus
from meeting_scribe.translation.queue import TranslationQueue, _can_merge, _merge_events


def _make_event(
    seg_id: str = "s1",
    text: str = "Hello",
    language: str = "en",
    start_ms: int = 0,
    end_ms: int = 1000,
    is_final: bool = True,
    cluster_id: int = 0,
) -> TranscriptEvent:
    from meeting_scribe.models import SpeakerAttribution

    return TranscriptEvent(
        segment_id=seg_id,
        text=text,
        language=language,
        is_final=is_final,
        start_ms=start_ms,
        end_ms=end_ms,
        speakers=[SpeakerAttribution(cluster_id=cluster_id, source="test")],
    )


class TestMergeGate:
    """Adjacent segment merging before translation."""

    def test_same_speaker_same_language_merges(self):
        a = _make_event("s1", "Hello", "en", 0, 1000, cluster_id=0)
        b = _make_event("s2", "world", "en", 1000, 2000, cluster_id=0)
        assert _can_merge(a, b) is True

    def test_different_language_no_merge(self):
        a = _make_event("s1", "Hello", "en", 0, 1000)
        b = _make_event("s2", "こんにちは", "ja", 1000, 2000)
        assert _can_merge(a, b) is False

    def test_large_gap_no_merge(self):
        a = _make_event("s1", "Hello", "en", 0, 1000)
        b = _make_event("s2", "world", "en", 5000, 6000)
        assert _can_merge(a, b) is False

    def test_different_speaker_no_merge(self):
        a = _make_event("s1", "Hello", "en", 0, 1000, cluster_id=0)
        b = _make_event("s2", "world", "en", 1000, 2000, cluster_id=1)
        assert _can_merge(a, b) is False

    def test_non_final_no_merge(self):
        a = _make_event("s1", "Hello", "en", 0, 1000, is_final=False)
        b = _make_event("s2", "world", "en", 1000, 2000)
        assert _can_merge(a, b) is False

    def test_merge_concatenates_text(self):
        a = _make_event("s1", "Hello", "en", 0, 1000)
        b = _make_event("s2", "world", "en", 1000, 2000)
        merged = _merge_events(a, b)
        assert merged.text == "Hello world"
        assert merged.start_ms == 0
        assert merged.end_ms == 2000


class TestTranslationTargetRouting:
    """Translation queue respects language pair for target routing."""

    @pytest.mark.asyncio
    async def test_ja_in_ja_en_pair_translates_to_en(self):
        results = []

        async def on_result(event):
            results.append(event)

        class MockBackend:
            async def translate(self, text, source_language, target_language, **kwargs):
                assert source_language == "ja"
                assert target_language == "en"
                return "translated"

        q = TranslationQueue(on_result=on_result, languages=("ja", "en"))
        await q.start(MockBackend())
        await q.submit(_make_event("s1", "テスト", "ja"))
        await q.flush_merge_gate()

        # Wait for processing
        for _ in range(50):
            if any(
                r.translation and r.translation.status == TranslationStatus.DONE for r in results
            ):
                break
            await asyncio.sleep(0.05)

        await q.stop()

        done_events = [
            r for r in results if r.translation and r.translation.status == TranslationStatus.DONE
        ]
        assert len(done_events) == 1
        assert done_events[0].translation.target_language == "en"
        assert done_events[0].translation.text == "translated"

    @pytest.mark.asyncio
    async def test_zh_in_zh_en_pair_translates_to_en(self):
        results = []

        async def on_result(event):
            results.append(event)

        class MockBackend:
            async def translate(self, text, source_language, target_language, **kwargs):
                assert source_language == "zh"
                assert target_language == "en"
                return "translated_zh"

        q = TranslationQueue(on_result=on_result, languages=("zh", "en"))
        await q.start(MockBackend())
        await q.submit(_make_event("s1", "你好", "zh"))
        await q.flush_merge_gate()

        for _ in range(50):
            if any(
                r.translation and r.translation.status == TranslationStatus.DONE for r in results
            ):
                break
            await asyncio.sleep(0.05)

        await q.stop()

        done_events = [
            r for r in results if r.translation and r.translation.status == TranslationStatus.DONE
        ]
        assert len(done_events) == 1
        assert done_events[0].translation.target_language == "en"

    @pytest.mark.asyncio
    async def test_language_not_in_pair_skipped(self):
        results = []

        async def on_result(event):
            results.append(event)

        class MockBackend:
            async def translate(self, **kwargs):
                raise AssertionError("Should not be called")

        q = TranslationQueue(on_result=on_result, languages=("ja", "en"))
        await q.start(MockBackend())
        # French is not in ja/en pair — should be skipped
        await q.submit(_make_event("s1", "Bonjour", "fr"))
        await q.flush_merge_gate()

        for _ in range(20):
            if results:
                break
            await asyncio.sleep(0.05)

        await q.stop()

        skipped = [
            r
            for r in results
            if r.translation and r.translation.status == TranslationStatus.SKIPPED
        ]
        assert len(skipped) == 1


class TestTranslationQueueMonolingual:
    """Monolingual meetings short-circuit every translation request —
    no enqueue, no worker dispatch, and no TranslationQueue-induced
    events. The on_result callback is invoked only via the SKIPPED
    terminal path (if reached), never with a DONE status."""

    @pytest.mark.asyncio
    async def test_monolingual_never_dispatches_to_backend(self):
        translator_calls = 0

        async def on_result(event):
            # No-op; we only assert on the backend-call counter.
            pass

        class MockBackend:
            async def translate(self, *args, **kwargs):
                nonlocal translator_calls
                translator_calls += 1
                return "should-not-happen"

        q = TranslationQueue(on_result=on_result, languages=["en"])
        await q.start(MockBackend())
        # Submit a segment in the meeting's language — a bilingual queue
        # would translate it to the other pair member, a monolingual
        # queue must not.
        await q.submit(_make_event("s1", "Hello", "en"))
        await q.submit(_make_event("s2", "world", "en"))
        await q.flush_merge_gate()
        # Give the worker pool a moment; nothing should happen.
        await asyncio.sleep(0.1)
        await q.stop()

        assert translator_calls == 0, "Monolingual queue must never invoke the translator backend"

    @pytest.mark.asyncio
    async def test_set_languages_flips_to_monolingual(self):
        """``set_languages`` at meeting start must switch behaviour
        cleanly — a queue created with the process-wide ja/en default
        stops translating once a monolingual meeting is started."""
        translator_calls = 0

        async def on_result(event):
            pass

        class MockBackend:
            async def translate(self, *args, **kwargs):
                nonlocal translator_calls
                translator_calls += 1
                return "tr"

        q = TranslationQueue(on_result=on_result, languages=("ja", "en"))
        await q.start(MockBackend())
        # Switch to monolingual before any submit happens.
        q.set_languages(["en"])
        await q.submit(_make_event("s1", "Hello", "en"))
        await q.flush_merge_gate()
        await asyncio.sleep(0.1)
        await q.stop()

        assert translator_calls == 0
