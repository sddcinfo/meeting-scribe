"""Tests for the translation-queue + slide-job drain path.

Covers ``active_count`` / ``pending_count`` / ``merge_gate_held``
introspection that the ``meeting-scribe drain`` CLI gates on, plus the
new ``pause``/``resume`` and ``cancel_all`` additions backing
``pause-translation`` / ``resume-translation`` and ``drain --force``.

Does not spin up the FastAPI server — the HTTP layer is a thin shim
over these queue methods, and the server unit is already covered by
the existing admin-endpoint smoke tests elsewhere in the suite.
"""

from __future__ import annotations

import asyncio

import pytest

from meeting_scribe.models import SpeakerAttribution, TranscriptEvent
from meeting_scribe.translation.queue import TranslationQueue


def _make_event(
    seg_id: str,
    text: str = "Hello",
    language: str = "en",
    start_ms: int = 0,
    end_ms: int = 1000,
    cluster_id: int = 0,
) -> TranscriptEvent:
    return TranscriptEvent(
        segment_id=seg_id,
        text=text,
        language=language,
        is_final=True,
        start_ms=start_ms,
        end_ms=end_ms,
        speakers=[SpeakerAttribution(cluster_id=cluster_id, source="test")],
    )


class _BlockingBackend:
    """Translation backend that blocks on an event until released.

    Lets tests freeze a worker in the middle of a translation so
    ``active_count()`` reports a non-zero value deterministically.
    """

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.entered = asyncio.Event()
        self.calls = 0

    async def translate(self, text, source_language, target_language, **kwargs):
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        return f"{text}:translated"


class _FastBackend:
    """Trivial backend that returns immediately — for merge-gate / pause tests."""

    async def translate(self, text, source_language, target_language, **kwargs):
        return f"{text}:translated"


class TestDrainIntrospection:
    """active_count / pending_count / merge_gate_held shape the drain gate."""

    @pytest.mark.asyncio
    async def test_empty_queue_is_idle(self):
        q = TranslationQueue(on_result=None, languages=("ja", "en"))
        await q.start(_FastBackend())
        try:
            assert q.active_count() == 0
            assert q.pending_count() == 0
            assert q.merge_gate_held() is False
            assert q.is_idle() is True
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_merge_gate_reports_held(self):
        q = TranslationQueue(on_result=None, languages=("ja", "en"))
        await q.start(_FastBackend())
        try:
            await q.submit(_make_event("s1", "テスト", "ja"))
            # A single submit leaves the event in the merge gate —
            # pending_count is 0, merge_gate_held is True.
            assert q.merge_gate_held() is True
            assert q.active_count() == 0
            # is_idle respects the merge gate.
            assert q.is_idle() is False

            await q.flush_merge_gate()
            # After flush, the held event is enqueued (or already done).
            # merge_gate clears immediately regardless.
            assert q.merge_gate_held() is False
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_active_count_during_translation(self):
        backend = _BlockingBackend()
        q = TranslationQueue(
            on_result=None,
            concurrency=1,
            languages=("ja", "en"),
        )
        await q.start(backend)
        try:
            await q.submit(_make_event("s1", "テスト", "ja"))
            await q.flush_merge_gate()

            # Wait for the worker to enter the blocking translate call.
            await asyncio.wait_for(backend.entered.wait(), timeout=2.0)

            assert q.active_count() == 1
            assert q.pending_count() == 0

            # Release the worker so stop() can join cleanly.
            backend.release.set()

            # Poll until idle.
            for _ in range(50):
                if q.is_idle():
                    break
                await asyncio.sleep(0.05)
            assert q.is_idle() is True
        finally:
            backend.release.set()
            await q.stop()


class TestPauseResume:
    """pause() gates submit() without touching in-flight work."""

    @pytest.mark.asyncio
    async def test_paused_queue_drops_new_submits(self):
        backend = _FastBackend()
        q = TranslationQueue(on_result=None, languages=("ja", "en"))
        await q.start(backend)
        try:
            q.pause()
            assert q.is_paused() is True

            await q.submit(_make_event("s1", "テスト", "ja"))
            await q.submit(_make_event("s2", "テスト2", "ja"))

            # Paused submit() silently drops — nothing enters the queue
            # and nothing enters the merge gate.
            assert q.pending_count() == 0
            assert q.active_count() == 0
            assert q.merge_gate_held() is False

            q.resume()
            assert q.is_paused() is False

            await q.submit(_make_event("s3", "テスト3", "ja"))
            # After resume, the merge gate starts holding events again.
            assert q.merge_gate_held() is True
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_pause_resume_are_idempotent(self):
        q = TranslationQueue(on_result=None, languages=("ja", "en"))
        await q.start(_FastBackend())
        try:
            q.pause()
            q.pause()  # second call is a no-op
            assert q.is_paused() is True

            q.resume()
            q.resume()  # second call is a no-op
            assert q.is_paused() is False
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_pause_does_not_affect_in_flight_item(self):
        """Already-queued items continue even after pause() — per the plan's
        drain protocol: pause gates NEW intake, drain waits the rest out."""
        backend = _BlockingBackend()
        q = TranslationQueue(
            on_result=None,
            concurrency=1,
            languages=("ja", "en"),
        )
        await q.start(backend)
        try:
            await q.submit(_make_event("s1", "テスト", "ja"))
            await q.flush_merge_gate()
            await asyncio.wait_for(backend.entered.wait(), timeout=2.0)
            assert q.active_count() == 1

            q.pause()
            # Pause must NOT cancel the in-flight item.
            assert q.active_count() == 1

            backend.release.set()
            for _ in range(50):
                if q.is_idle():
                    break
                await asyncio.sleep(0.05)
            assert q.is_idle() is True
        finally:
            backend.release.set()
            await q.stop()


class TestForceCancel:
    """cancel_all() flips .cancelled on every not-yet-started item."""

    @pytest.mark.asyncio
    async def test_cancel_all_with_pending_items(self):
        # Freeze the worker so enqueued items actually pile up.
        backend = _BlockingBackend()
        q = TranslationQueue(
            on_result=None,
            concurrency=1,
            maxsize=50,
            languages=("ja", "en"),
        )
        await q.start(backend)
        try:
            # Different cluster_ids prevent the merge gate from folding
            # adjacent same-speaker segments into a single item — we
            # want 3 distinct queue entries for this test.
            await q.submit(_make_event("s1", "one", "ja", cluster_id=0))
            await q.submit(_make_event("s2", "two", "ja", start_ms=2000, end_ms=3000, cluster_id=1))
            await q.submit(_make_event("s3", "three", "ja", start_ms=4000, end_ms=5000, cluster_id=2))
            # Flush the merge gate so the last event is also queued.
            await q.flush_merge_gate()

            # Wait for worker to grab one → one active, two pending.
            await asyncio.wait_for(backend.entered.wait(), timeout=2.0)
            assert q.active_count() == 1
            assert q.pending_count() == 2

            cancelled = q.cancel_all()
            assert cancelled == 2
            # Active item stays running — cancel_all only flips not-started.
            assert q.active_count() == 1
            # Pending items are marked cancelled but still count until
            # a worker picks them up; logically they're out of the
            # queue from the drain's perspective (the worker will fire
            # a skip callback and move on quickly).
            assert q.pending_count() == 2

            backend.release.set()
            # Wait until drained.
            for _ in range(100):
                if q.is_idle():
                    break
                await asyncio.sleep(0.05)
            assert q.is_idle() is True
        finally:
            backend.release.set()
            await q.stop()

    @pytest.mark.asyncio
    async def test_cancel_all_on_empty_queue_returns_zero(self):
        q = TranslationQueue(on_result=None, languages=("ja", "en"))
        await q.start(_FastBackend())
        try:
            assert q.cancel_all() == 0
        finally:
            await q.stop()


class TestSlideRunnerCancelHelper:
    """cancel_current_job() returns False when nothing is running."""

    @pytest.mark.asyncio
    async def test_cancel_with_no_job_returns_false(self):
        # Minimal SlideJobRunner stand-in: construct the real class with a
        # tmp meetings dir and verify the cancel helper short-circuits on
        # empty state.  Avoids pulling in a full slide-pipeline fixture.
        import tempfile
        from pathlib import Path

        from meeting_scribe.slides.job import SlideJobRunner

        async def _noop_translate(text, source_lang, target_lang, system_prompt, max_tokens):
            return text

        async def _noop_broadcast(_payload):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            runner = SlideJobRunner(
                meetings_dir=Path(tmp),
                translate_fn=_noop_translate,
                broadcast_fn=_noop_broadcast,
            )
            assert runner.is_running() is False
            assert await runner.cancel_current_job() is False
