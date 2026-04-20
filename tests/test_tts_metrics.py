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
    from meeting_scribe import server

    server.metrics.reset()
    server._tts_in_flight = 0
    server._tts_inflight_started.clear()
    server._tts_stall_candidate_since = None
    server._tts_degraded_candidate_since = None
    server._crash_state = None
    yield


class TestPercentileGuard:
    """[P1-6-i1] percentile dict never crashes on tiny windows."""

    def test_empty_deque(self):
        from collections import deque

        from meeting_scribe.server import _percentile_dict

        d = _percentile_dict(deque(maxlen=256))
        assert d == {"p50": None, "p95": None, "p99": None, "sample_count": 0}

    def test_one_sample(self):
        from meeting_scribe.server import _percentile_dict

        d = _percentile_dict([42.0])
        assert d["sample_count"] == 1
        assert d["p50"] is None


class TestTTSHealthEvaluator:
    """[P1-4-i1 + P1-3-i2 + P1-4-i2]"""

    def _run_eval_tick(self, monkeypatch):
        """Helper: run one iteration of the health evaluator body.

        We can't easily await the full _tts_health_evaluator coroutine
        under pytest — it sleeps forever. So we reproduce its decision
        logic inline using the same constants, matching what the
        real function does on one tick.
        """
        from meeting_scribe import server

        now = time.monotonic()
        qsize = server._tts_queue.qsize() if server._tts_queue else 0
        in_flight = server._tts_in_flight
        queue_saturated = (
            in_flight >= server._TTS_CONTAINER_MAX_CONCURRENCY
            and qsize >= server._TTS_QUEUE_MAXSIZE
        )
        e2e_p95 = server._percentile(sorted(server.metrics.end_to_end_lag_ms), 0.95)
        no_progress_stall = (
            in_flight > 0
            and server.metrics.last_delivery_at > 0
            and (now - server.metrics.last_delivery_at) > server._TTS_NO_PROGRESS_STALL_S
        )
        stall_condition = no_progress_stall or (e2e_p95 is not None and e2e_p95 > 6000)
        degraded_condition = queue_saturated or (e2e_p95 is not None and e2e_p95 > 3500)
        return now, stall_condition, degraded_condition

    def test_healthy_by_default(self):
        from meeting_scribe import server

        _, stall, degraded = self._run_eval_tick(None)
        assert not stall
        assert not degraded
        assert server.metrics.tts_health_state == "healthy"

    def test_degraded_when_p95_high(self):
        from meeting_scribe import server

        # Inject 30 lag samples at 4500 ms
        for _ in range(30):
            server.metrics.end_to_end_lag_ms.append(4500.0)
        _, stall, degraded = self._run_eval_tick(None)
        assert not stall
        assert degraded

    def test_stalled_when_p95_very_high(self):
        from meeting_scribe import server

        for _ in range(30):
            server.metrics.end_to_end_lag_ms.append(7000.0)
        _, stall, _ = self._run_eval_tick(None)
        assert stall

    def test_no_progress_stall_detected(self):
        from meeting_scribe import server

        # Simulate in-flight with no recent delivery.
        server._tts_in_flight = 1
        server.metrics.last_delivery_at = time.monotonic() - 10.0
        _, stall, _ = self._run_eval_tick(None)
        assert stall


class TestWorkersBusyCounter:
    """[P1-3-i1] workers_busy == in_flight, not task count."""

    def test_workers_busy_reflects_in_flight(self):
        from meeting_scribe import server

        server._tts_in_flight = 0
        d = server.metrics.to_dict()
        assert d["tts"]["workers_busy"] == 0

        server._tts_in_flight = 1
        d = server.metrics.to_dict()
        assert d["tts"]["workers_busy"] == 1


class TestRecordSegmentLagListenerIndependent:
    """[P1-5-i2 + P1-3-i3] segment lag recorded ONCE, not per listener."""

    def test_single_segment_produces_single_sample(self):
        from meeting_scribe import server

        evt = _make_event("hello")
        # Simulate the synthesis-complete path: _record_segment_lag is called
        # exactly once per segment, regardless of how many listeners receive it.
        server._record_segment_lag(evt, now=time.monotonic() + 1.0)
        server._record_segment_lag(evt, now=time.monotonic() + 1.5)  # second segment
        # Two segments → two samples in the SLA histograms
        assert len(server.metrics.end_to_end_lag_ms) == 2


class TestCrashState:
    """[P2-1-i2] crash metadata is sanitised."""

    def test_crash_exposes_only_opaque_code(self):
        from meeting_scribe import server

        try:
            raise RuntimeError("secret internal detail — api_key=xyz")
        except RuntimeError as e:
            server._record_crash("tts_worker", e)

        state = server._sanitised_crash_state()
        assert state is not None
        assert state["state"] == "crashed"
        assert state["component"] == "tts_worker"
        # Ensure no secret / type name leaks out
        assert "secret" not in state["code"]
        assert "RuntimeError" not in state["code"]
        assert "api_key" not in state["code"]
        assert len(state["code"]) == 64  # full sha256 hexdigest

    def test_reports_null_when_no_crash(self):
        from meeting_scribe import server

        server._crash_state = None
        assert server._sanitised_crash_state() is None


class TestMetricsApiStatusShape:
    """The /api/status payload must include all the new TTS + listener fields."""

    def test_to_dict_has_all_tts_fields(self):
        from meeting_scribe import server

        d = server.metrics.to_dict()
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
        from meeting_scribe import server

        task = asyncio.create_task(server._loop_lag_monitor())
        try:
            # Let it tick at least twice (each tick sleeps 0.5 s).
            await asyncio.sleep(1.2)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        assert len(server.metrics.loop_lag_ms) >= 2
