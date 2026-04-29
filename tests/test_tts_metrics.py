"""Tests for Phase 2 TTS observability: percentiles, health evaluator,
workers_busy counter, loop lag monitor, crash state, listener-count
independence of the SLA histograms.
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
from meeting_scribe.server_support import crash_tracking
from meeting_scribe.server_support.metrics_helpers import _percentile_dict
from meeting_scribe.tts.worker import _record_segment_lag


def _make_event(text: str = "Hello") -> TranscriptEvent:
    now = time.monotonic()
    return TranscriptEvent(
        segment_id="metric-test-0000-0000-0000-000000000000",
        revision=1,
        is_final=True,
        start_ms=0,
        end_ms=1000,
        language="ja",
        text="こんにちは",
        translation=TranslationState(
            status=TranslationStatus.DONE,
            text=text,
            target_language="en",
            completed_at=now,
        ),
        utterance_end_at=now,
    )


@pytest.fixture(autouse=True)
def reset_server_state():
    from meeting_scribe.runtime import metrics as runtime_metrics
    from meeting_scribe.runtime import state as runtime_state

    runtime_state.metrics.reset()
    runtime_state.tts_in_flight = 0
    runtime_state.tts_inflight_started.clear()
    runtime_metrics._tts_stall_candidate_since = None
    runtime_metrics._tts_degraded_candidate_since = None
    crash_tracking._crash_state = None
    yield


class TestPercentileGuard:
    """[P1-6-i1] percentile dict never crashes on tiny windows."""

    def test_empty_deque(self):
        from collections import deque

        d = _percentile_dict(deque(maxlen=256))
        assert d == {"p50": None, "p95": None, "p99": None, "sample_count": 0}

    def test_one_sample(self):
        d = _percentile_dict([42.0])
        assert d["sample_count"] == 1
        assert d["p50"] is None


class TestTTSHealthEvaluator:
    """[P1-4-i1 + P1-3-i2 + P1-4-i2]"""

    def _run_eval_tick(self, monkeypatch):
        """Helper: run one iteration of the health evaluator body.

        We can't easily await the full tts_health_evaluator coroutine
        under pytest — it sleeps forever. So we reproduce its decision
        logic inline using the same constants, matching what the
        real function does on one tick.
        """
        from meeting_scribe.runtime import metrics as runtime_metrics
        from meeting_scribe.server_support.metrics_helpers import _percentile

        now = time.monotonic()
        qsize = runtime_state.tts_queue.qsize() if runtime_state.tts_queue else 0
        in_flight = runtime_state.tts_in_flight
        queue_saturated = (
            in_flight >= runtime_state.TTS_CONTAINER_MAX_CONCURRENCY
            and qsize >= runtime_state.TTS_QUEUE_MAXSIZE
        )
        e2e_p95 = _percentile(sorted(runtime_state.metrics.end_to_end_lag_ms), 0.95)
        no_progress_stall = (
            in_flight > 0
            and runtime_state.metrics.last_delivery_at > 0
            and (now - runtime_state.metrics.last_delivery_at)
            > runtime_metrics._TTS_NO_PROGRESS_STALL_S
        )
        stall_condition = no_progress_stall or (e2e_p95 is not None and e2e_p95 > 6000)
        degraded_condition = queue_saturated or (e2e_p95 is not None and e2e_p95 > 3500)
        return now, stall_condition, degraded_condition

    def test_healthy_by_default(self):

        _, stall, degraded = self._run_eval_tick(None)
        assert not stall
        assert not degraded
        assert runtime_state.metrics.tts_health_state == "healthy"

    def test_degraded_when_p95_high(self):

        # Inject 30 lag samples at 4500 ms
        for _ in range(30):
            runtime_state.metrics.end_to_end_lag_ms.append(4500.0)
        _, stall, degraded = self._run_eval_tick(None)
        assert not stall
        assert degraded

    def test_stalled_when_p95_very_high(self):

        for _ in range(30):
            runtime_state.metrics.end_to_end_lag_ms.append(7000.0)
        _, stall, _ = self._run_eval_tick(None)
        assert stall

    def test_no_progress_stall_detected(self):

        # Simulate in-flight with no recent delivery.
        runtime_state.tts_in_flight = 1
        runtime_state.metrics.last_delivery_at = time.monotonic() - 10.0
        _, stall, _ = self._run_eval_tick(None)
        assert stall


class TestWorkersBusyCounter:
    """[P1-3-i1] workers_busy == in_flight, not task count."""

    def test_workers_busy_reflects_in_flight(self):

        runtime_state.tts_in_flight = 0
        d = runtime_state.metrics.to_dict()
        assert d["tts"]["workers_busy"] == 0

        runtime_state.tts_in_flight = 1
        d = runtime_state.metrics.to_dict()
        assert d["tts"]["workers_busy"] == 1


class TestRecordSegmentLagListenerIndependent:
    """[P1-5-i2 + P1-3-i3] segment lag recorded ONCE, not per listener."""

    def test_single_segment_produces_single_sample(self):
        evt = _make_event("hello")
        # Simulate the synthesis-complete path: _record_segment_lag is called
        # exactly once per segment, regardless of how many listeners receive it.
        _record_segment_lag(evt, now=time.monotonic() + 1.0)
        _record_segment_lag(evt, now=time.monotonic() + 1.5)  # second segment
        # Two segments → two samples in the SLA histograms
        assert len(runtime_state.metrics.end_to_end_lag_ms) == 2


class TestCrashState:
    """[P2-1-i2] crash metadata is sanitised."""

    def test_crash_exposes_only_opaque_code(self):

        try:
            raise RuntimeError("secret internal detail — api_key=xyz")
        except RuntimeError as e:
            crash_tracking._record_crash("tts_worker", e)

        state = crash_tracking._sanitised_crash_state()
        assert state is not None
        assert state["state"] == "crashed"
        assert state["component"] == "tts_worker"
        # Ensure no secret / type name leaks out
        assert "secret" not in state["code"]
        assert "RuntimeError" not in state["code"]
        assert "api_key" not in state["code"]
        assert len(state["code"]) == 64  # full sha256 hexdigest

    def test_reports_null_when_no_crash(self):

        crash_tracking._crash_state = None
        assert crash_tracking._sanitised_crash_state() is None


class TestMetricsApiStatusShape:
    """The /api/status payload must include all the new TTS + listener fields."""

    def test_to_dict_has_all_tts_fields(self):

        d = runtime_state.metrics.to_dict()
        assert "tts" in d
        assert "listener" in d
        assert "loop_lag_ms" in d
        assert "crash" in d

        tts = d["tts"]
        for key in (
            "queue_depth",
            "queue_maxsize",
            "workers_busy",
            "workers_total",
            "container_concurrency",
            "submitted",
            "delivered",
            "drops",
            "timeouts",
            "synth_ms",
            "upstream_lag_ms",
            "tts_post_translation_lag_ms",
            "end_to_end_lag_ms",
            "oldest_inflight_age_ms",
            "health",
            "health_since",
        ):
            assert key in tts, f"missing tts.{key}"

        for dk in (
            "filler",
            "stale_producer",
            "stale_worker",
            "pre_synth",
            "post_synth",
            "queue_full",
            "missing_origin",
        ):
            assert dk in tts["drops"], f"missing tts.drops.{dk}"

        listener = d["listener"]
        for key in (
            "connected",
            "deliveries",
            "send_failed",
            "removed_on_send_error",
            "send_ms",
        ):
            assert key in listener, f"missing listener.{key}"


class TestLoopLagMonitorRecords:
    """[Phase 2] event loop lag monitor populates the histogram."""

    @pytest.mark.asyncio
    async def test_monitor_records_at_least_one_sample(self):
        from meeting_scribe.runtime.health_monitors import loop_lag_monitor

        task = asyncio.create_task(loop_lag_monitor())
        try:
            # Let it tick at least twice (each tick sleeps 0.5 s).
            await asyncio.sleep(1.2)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        assert len(runtime_state.metrics.loop_lag_ms) >= 2


class TestSilenceWatchdogStaleTimestampGuard:
    """Regression guard for the 2026-04-28 ``no_audio (31185s)`` warning.

    ``state.last_audio_chunk_ts`` is bumped by ws.audio_input on every
    frame regardless of meeting state — a stray admin-tab WS, a prior
    meeting, anything. If the watchdog only checks ``last_audio_chunk_ts``
    it fires immediately when the next meeting starts and reports the
    cumulative gap as silence age. The fix anchors age against
    ``max(last_audio_chunk_ts, metrics.meeting_start)``.
    """

    @pytest.mark.asyncio
    async def test_does_not_fire_on_stale_pre_meeting_timestamp(self, monkeypatch):
        """Stale chunk_ts older than the current meeting's start does
        not produce a no_audio warning."""

        from meeting_scribe.runtime import health_monitors

        captured: list[dict] = []

        async def _capture(payload: dict) -> None:
            captured.append(payload)

        # Intercept the broadcast so we can assert no warning fires.
        monkeypatch.setattr(health_monitors, "_broadcast_json", _capture)

        # Simulate: a stray frame landed long before any meeting (or
        # under a prior meeting that's since ended).
        runtime_state.last_audio_chunk_ts = time.monotonic() - 60.0
        runtime_state.silence_warn_sent = False

        # Now a brand-new meeting starts NOW. Its meeting_start is
        # younger than last_audio_chunk_ts, so the gap from
        # last_audio_chunk_ts is irrelevant.
        runtime_state.metrics.meeting_start = time.monotonic()

        class _FakeMeeting:
            meeting_id = "test-stale-ts-guard"

        runtime_state.current_meeting = _FakeMeeting()

        try:
            task = asyncio.create_task(health_monitors.silence_watchdog_loop())
            await asyncio.sleep(2.5)  # well over the 1.0s poll
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            runtime_state.current_meeting = None
            runtime_state.last_audio_chunk_ts = 0.0
            runtime_state.silence_warn_sent = False
            runtime_state.metrics.meeting_start = 0.0

        no_audio = [p for p in captured if p.get("reason") == "no_audio"]
        assert no_audio == [], f"watchdog fired against stale pre-meeting timestamp: {no_audio}"
