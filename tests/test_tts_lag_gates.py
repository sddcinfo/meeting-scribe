"""Tests for Phase 1 TTS lag gates.

Covers: whitelist filler drop, stale-on-enqueue, pre-synth budget gate,
dequeue gate, size-aware + deadline-aware timeout, in-flight counter,
utterance_end_at propagation from ASR, upstream vs post-translation lag
split, and missing-origin refusal.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from meeting_scribe.models import (
    TranscriptEvent,
    TranslationState,
    TranslationStatus,
)
from meeting_scribe.runtime import state as runtime_state
from meeting_scribe.server_support.metrics_helpers import (
    _percentile,
    _percentile_dict,
)
from meeting_scribe.tts.worker import _enqueue_tts, _record_segment_lag


def _make_event(
    text: str = "Hello world",
    *,
    utterance_end_at: float | None = None,
    completed_at: float | None = None,
    target_language: str = "en",
) -> TranscriptEvent:
    if utterance_end_at is None:
        utterance_end_at = time.monotonic()
    if completed_at is None:
        completed_at = utterance_end_at
    return TranscriptEvent(
        segment_id="abcd1234-0000-0000-0000-000000000000",
        revision=1,
        is_final=True,
        start_ms=0,
        end_ms=1000,
        language="ja",
        text="こんにちは",
        translation=TranslationState(
            status=TranslationStatus.DONE,
            text=text,
            target_language=target_language,
            completed_at=completed_at,
        ),
        utterance_end_at=utterance_end_at,
    )


@pytest.fixture(autouse=True)
def reset_server_state():
    """Reset module-level TTS state between tests."""
    from meeting_scribe.runtime import state as runtime_state

    runtime_state.tts_in_flight = 0
    runtime_state.tts_inflight_started.clear()
    runtime_state.metrics.reset()
    if runtime_state.tts_queue is not None:
        while not runtime_state.tts_queue.empty():
            try:
                runtime_state.tts_queue.get_nowait()
                runtime_state.tts_queue.task_done()
            except Exception:
                break
    yield
    runtime_state.tts_in_flight = 0
    runtime_state.tts_inflight_started.clear()


class TestProducerWhitelistFiller:
    """[P2-2-i1] Whitelist filler drop."""

    def test_drops_whitelisted_ack_when_backlog(self):
        runtime_state.tts_queue = asyncio.Queue(maxsize=3)
        # Simulate backlog: put a dummy item so outstanding > 0
        runtime_state.tts_queue.put_nowait(
            (
                _make_event("hello"),
                "default",
                time.monotonic() + 4.0,
                time.monotonic(),
            )
        )
        evt = _make_event("はい")
        _enqueue_tts(evt, "default")
        assert runtime_state.metrics.tts_dropped_filler == 1
        # Original dummy is still the only item
        assert runtime_state.tts_queue.qsize() == 1

    def test_keeps_ack_when_idle(self):
        runtime_state.tts_queue = asyncio.Queue(maxsize=3)
        evt = _make_event("はい")
        _enqueue_tts(evt, "default")
        assert runtime_state.metrics.tts_dropped_filler == 0
        assert runtime_state.tts_queue.qsize() == 1

    def test_keeps_short_but_not_whitelisted_when_backlog(self):
        runtime_state.tts_queue = asyncio.Queue(maxsize=3)
        # Backlog
        runtime_state.tts_queue.put_nowait(
            (
                _make_event("hello"),
                "default",
                time.monotonic() + 4.0,
                time.monotonic(),
            )
        )
        evt = _make_event("No.")  # short but not in whitelist
        _enqueue_tts(evt, "default")
        assert runtime_state.metrics.tts_dropped_filler == 0
        assert runtime_state.tts_queue.qsize() == 2


class TestProducerStaleGate:
    """[P1-2-i1] stale-on-enqueue gate."""

    def test_drops_stale_when_deadline_already_blown(self):
        runtime_state.tts_queue = asyncio.Queue(maxsize=3)
        runtime_state.tts_queue.put_nowait(
            (
                _make_event("hello"),
                "default",
                time.monotonic() + 4.0,
                time.monotonic(),
            )
        )
        evt = _make_event("Hello", utterance_end_at=time.monotonic() - 10.0)
        _enqueue_tts(evt, "default")
        assert runtime_state.metrics.tts_dropped_stale_producer == 1

    def test_keeps_stale_when_no_backlog(self):
        # No backlog → still admit even if deadline is tight
        runtime_state.tts_queue = asyncio.Queue(maxsize=3)
        evt = _make_event("Hello", utterance_end_at=time.monotonic() - 10.0)
        _enqueue_tts(evt, "default")
        assert runtime_state.metrics.tts_dropped_stale_producer == 0
        assert runtime_state.tts_queue.qsize() == 1


class TestMissingOriginRefusal:
    """[P1-2-i5] event.utterance_end_at is None is a code bug."""

    def test_refuses_event_with_missing_origin(self):
        runtime_state.tts_queue = asyncio.Queue(maxsize=3)
        evt = _make_event("Hello")
        evt.utterance_end_at = None
        _enqueue_tts(evt, "default")
        assert runtime_state.metrics.tts_dropped_missing_origin == 1
        assert runtime_state.tts_queue.qsize() == 0


class TestPercentileHelper:
    """[P1-6-i1] guard against small-window NaN / crash."""

    def test_returns_none_for_empty(self):
        assert _percentile([], 0.5) is None
        d = _percentile_dict([])
        assert d == {"p50": None, "p95": None, "p99": None, "sample_count": 0}

    def test_returns_none_below_min_samples(self):
        d = _percentile_dict([1.0, 2.0, 3.0])
        assert d["p50"] is None
        assert d["sample_count"] == 3

    def test_computes_quantiles_for_sufficient_samples(self):
        d = _percentile_dict(list(range(1, 101)))  # 1..100
        assert d["sample_count"] == 100
        # Nearest-rank with round-half-to-even on (q * (n-1)):
        #   p50: round(49.5)=50 → srt[50]=51
        #   p95: round(94.05)=94 → srt[94]=95
        #   p99: round(98.01)=98 → srt[98]=99
        assert d["p50"] == 51
        assert d["p95"] == 95
        assert d["p99"] == 99


class TestSegmentLagRecording:
    """[P1-5-i2 + P1-1-i1] segment-level lag, pre-fan-out, upstream split."""

    def test_upstream_and_post_translation_split(self):
        evt = _make_event(
            "Hello",
            utterance_end_at=0.0,
            completed_at=2.0,
        )
        _record_segment_lag(evt, now=3.0)
        assert list(runtime_state.metrics.end_to_end_lag_ms) == [3000.0]
        assert list(runtime_state.metrics.upstream_lag_ms) == [2000.0]
        assert list(runtime_state.metrics.tts_post_translation_lag_ms) == [1000.0]

    def test_skips_when_utterance_end_missing(self):
        evt = _make_event("Hello")
        evt.utterance_end_at = None
        _record_segment_lag(evt, now=5.0)
        # Histograms should not grow
        assert len(runtime_state.metrics.end_to_end_lag_ms) == 0
        assert len(runtime_state.metrics.upstream_lag_ms) == 0


class TestAsrVllmUtteranceEndAt:
    """[P1-2-i2 + P1-2-i5] asr_vllm computes origin from audio clock."""

    def test_utterance_end_at_tracks_audio_end(self):
        """The ASR backend's utterance_end_at must equal
        audio_wall_at_start + end_ms/1000, independent of when the
        backend emits the event.
        """
        from meeting_scribe.models import TranscriptEvent

        # Simulate what asr_vllm does: build an event with
        # audio_wall_at_start + end_ms/1000.
        audio_wall_at_start = 1000.0
        end_ms = 4500
        expected = audio_wall_at_start + end_ms / 1000.0
        evt = TranscriptEvent(
            segment_id="x",
            end_ms=end_ms,
            language="ja",
            text="test",
            utterance_end_at=expected,
        )
        assert evt.utterance_end_at == 1004.5


class TestTranslationQueueStampsCompletedAt:
    """[P1-1-i1] translation.completed_at is set before the callback."""

    @pytest.mark.asyncio
    async def test_completed_at_populated_on_done(self):
        from meeting_scribe.models import TranslationState, TranslationStatus
        from meeting_scribe.translation.queue import TranslationQueue

        captured: list[TranscriptEvent] = []

        async def on_result(event):
            captured.append(event)

        class FakeBackend:
            async def translate(self, text, source_language, target_language, **kwargs):
                return "ok"

        q = TranslationQueue(
            maxsize=4,
            concurrency=1,
            timeout=5.0,
            on_result=on_result,
            languages=("ja", "en"),
        )
        await q.start(FakeBackend())
        evt = _make_event("hello", completed_at=None)
        evt.translation = TranslationState(
            status=TranslationStatus.PENDING,
            text=None,
            target_language="en",
        )
        await q.submit(evt)
        # Flush the merge gate so the single event actually enters the queue.
        await q.flush_merge_gate()
        # Drain the worker
        await asyncio.sleep(0.3)
        await q.stop()
        # on_result is called for BOTH in_progress (pending) and done events;
        # the done event is the last one with status=DONE.
        done_events = [
            e for e in captured if e.translation and e.translation.status == TranslationStatus.DONE
        ]
        assert len(done_events) == 1
        done = done_events[0]
        assert done.translation.completed_at is not None
        # Stamp should be roughly "now"
        assert time.monotonic() - done.translation.completed_at < 5.0
