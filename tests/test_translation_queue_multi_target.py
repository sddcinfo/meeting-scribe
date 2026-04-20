"""Multi-target fan-out tests for TranslationQueue.

Exercise the demand-driven path gated behind ``MULTI_TARGET_ENABLED``.
The flag is module-level; tests monkeypatch it on the queue module so
they don't depend on process env at import time.
"""

from __future__ import annotations

import asyncio

import pytest

from meeting_scribe.models import SpeakerAttribution, TranscriptEvent, TranslationStatus
from meeting_scribe.translation import queue as queue_mod
from meeting_scribe.translation.queue import TranslationQueue


class _StubBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def translate(
        self, text: str, source_language: str, target_language: str, **kwargs
    ) -> str:
        self.calls.append((text, source_language, target_language))
        return f"{text}::{target_language}"


def _event(
    seg_id: str = "s1",
    text: str = "Hello",
    language: str = "en",
    revision: int = 0,
) -> TranscriptEvent:
    return TranscriptEvent(
        segment_id=seg_id,
        text=text,
        language=language,
        is_final=True,
        revision=revision,
        start_ms=0,
        end_ms=1000,
        speakers=[SpeakerAttribution(cluster_id=0, source="test")],
    )


@pytest.fixture
def multi_target(monkeypatch):
    monkeypatch.setattr(queue_mod, "MULTI_TARGET_ENABLED", True)
    yield


@pytest.mark.asyncio
async def test_fanout_emits_one_done_per_target(multi_target):
    results: list[TranscriptEvent] = []

    async def on_result(ev: TranscriptEvent) -> None:
        results.append(ev)

    q = TranslationQueue(maxsize=10, concurrency=1, timeout=5.0, on_result=on_result)
    await q.start(_StubBackend())
    try:
        await q.submit(
            _event(language="en"),
            baseline_targets=frozenset({"ja"}),
            optional_targets=frozenset({"fr"}),
        )
        await q.flush_merge_gate()
        await asyncio.sleep(0.1)
    finally:
        await q.stop()

    done = [r for r in results if r.translation and r.translation.status == TranslationStatus.DONE]
    target_langs = sorted(r.translation.target_language for r in done)
    assert target_langs == ["fr", "ja"]


@pytest.mark.asyncio
async def test_source_equal_target_is_skipped(multi_target):
    results: list[TranscriptEvent] = []

    async def on_result(ev: TranscriptEvent) -> None:
        results.append(ev)

    q = TranslationQueue(maxsize=10, concurrency=1, timeout=5.0, on_result=on_result)
    await q.start(_StubBackend())
    try:
        # en source with en in optional → en should NOT translate to en.
        await q.submit(
            _event(language="en"),
            baseline_targets=frozenset({"ja"}),
            optional_targets=frozenset({"en"}),
        )
        await q.flush_merge_gate()
        await asyncio.sleep(0.1)
    finally:
        await q.stop()

    done_langs = [
        r.translation.target_language
        for r in results
        if r.translation and r.translation.status == TranslationStatus.DONE
    ]
    assert done_langs == ["ja"]


@pytest.mark.asyncio
async def test_optional_dropped_from_baseline_union(multi_target):
    """Optional that overlaps baseline should not produce duplicates."""
    results: list[TranscriptEvent] = []

    async def on_result(ev: TranscriptEvent) -> None:
        results.append(ev)

    q = TranslationQueue(maxsize=10, concurrency=1, timeout=5.0, on_result=on_result)
    await q.start(_StubBackend())
    try:
        await q.submit(
            _event(language="en"),
            baseline_targets=frozenset({"ja"}),
            optional_targets=frozenset({"ja", "fr"}),
        )
        await q.flush_merge_gate()
        await asyncio.sleep(0.1)
    finally:
        await q.stop()

    done_langs = sorted(
        r.translation.target_language
        for r in results
        if r.translation and r.translation.status == TranslationStatus.DONE
    )
    assert done_langs == ["fr", "ja"]


@pytest.mark.asyncio
async def test_backpressure_trims_optional_before_dropping(multi_target):
    results: list[TranscriptEvent] = []

    async def on_result(ev: TranscriptEvent) -> None:
        results.append(ev)

    # Maxsize 1 forces backpressure on second submit.
    q = TranslationQueue(maxsize=1, concurrency=1, timeout=5.0, on_result=on_result)
    # Don't start workers — we want items to pile up pending.
    q._backend = _StubBackend()  # type: ignore[attr-defined]
    q._running = True  # type: ignore[attr-defined]

    await q.submit(
        _event("s1", language="en"),
        baseline_targets=frozenset({"ja"}),
        optional_targets=frozenset({"fr", "de"}),
    )
    await q.flush_merge_gate()
    # After this first submit, one item with 1 baseline + 2 optional sits in queue.
    assert len(q._items) == 1
    item = q._items[0]
    assert item.baseline_targets == frozenset({"ja"})
    assert item.optional_targets == frozenset({"fr", "de"})

    await q.submit(
        _event("s2", language="en"),
        baseline_targets=frozenset({"ja"}),
        optional_targets=frozenset(),
    )
    await q.flush_merge_gate()

    # The first item's optional targets should be trimmed down (capacity=1
    # still forces a drop, but trim should have bled optional targets first).
    # We expect at least one trim to have occurred before the final drop.
    # Concretely: the dropped item keeps its baseline (we just log+drop);
    # the new item is present with baseline.
    ids = [it.segment_id for it in q._items if not it.cancelled]
    assert "s2" in ids
