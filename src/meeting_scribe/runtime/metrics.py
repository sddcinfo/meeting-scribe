"""Per-meeting metrics + the TTS health evaluator background loop.

Two pieces:

* ``Metrics`` — server-wide counters and rolling-window latency
  histograms. Reset per meeting via ``Metrics.reset()`` (called from
  ``/api/meeting/start`` so a fresh meeting starts at zero). The class
  inspects ``state.tts_*`` for queue depth / in-flight counts and the
  pipeline-level config constants (``state.TTS_*``) for
  ``queue_maxsize`` / ``workers_total`` / ``container_concurrency``.

* ``tts_health_evaluator`` — long-lived background loop that runs at
  500 ms cadence and mutates ``state.metrics.tts_health_state`` with
  hysteresis between ``healthy`` / ``degraded`` / ``stalled``.

Pulled out of ``server.py`` once the TTS pipeline runtime handles
moved into ``runtime.state`` — both modules now read from a single
canonical state surface, which is the precondition for this
extraction.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

from meeting_scribe.runtime import state
from meeting_scribe.server_support.crash_tracking import _sanitised_crash_state
from meeting_scribe.server_support.metrics_helpers import (
    _percentile,
    _percentile_dict,
)

logger = logging.getLogger(__name__)


# ── Real-time metrics ──────────────────────────────────────────


class Metrics:
    """Server-wide performance metrics, reset per meeting."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.meeting_start: float = 0.0
        self.audio_chunks: int = 0
        self.audio_seconds: float = 0.0
        self.asr_events: int = 0
        self.asr_partials: int = 0
        self.asr_finals: int = 0
        self.asr_finals_filler_dropped: int = 0
        self.asr_finals_deduped: int = 0
        self.translations_submitted: int = 0
        self.translations_completed: int = 0
        self.translations_failed: int = 0
        self.translation_total_ms: float = 0.0
        self.last_asr_event_time: float = 0.0
        self.translate_warmup_ms: float = 0.0
        self.asr_load_ms: float = 0.0

        # ── TTS counters [Phase 1] ──────────────────────────────
        self.tts_submitted: int = 0
        self.tts_delivered: int = 0
        self.tts_dropped_filler: int = 0
        self.tts_dropped_stale_producer: int = 0
        self.tts_dropped_stale_worker: int = 0
        self.tts_dropped_pre_synth: int = 0
        self.tts_dropped_post_synth: int = 0
        self.tts_dropped_queue_full: int = 0
        self.tts_dropped_missing_origin: int = 0
        self.tts_synth_timeouts: int = 0
        self.last_delivery_at: float = 0.0
        self.tts_synth_ms_p95: float | None = None

        # ── TTS latency histograms (256-sample rolling windows) ─
        self.tts_synth_ms: deque[float] = deque(maxlen=256)
        self.upstream_lag_ms: deque[float] = deque(maxlen=256)
        self.tts_post_translation_lag_ms: deque[float] = deque(maxlen=256)
        self.end_to_end_lag_ms: deque[float] = deque(maxlen=256)

        # ── Listener transport metrics (per send_bytes) [P1-5-i2] ─
        self.listener_send_ms: deque[float] = deque(maxlen=256)
        self.listener_deliveries: int = 0
        self.listener_send_failed: int = 0
        self.listener_removed_on_send_error: int = 0

        # ── Event-loop lag monitor [Phase 2] ────────────────────
        self.loop_lag_ms: deque[float] = deque(maxlen=256)

        # ── TTS health state (mutated by tts_health_evaluator) ─
        self.tts_health_state: str = "healthy"
        self.tts_health_since: float = 0.0

        # ── Backend RTT histograms [W5 — reliability dashboard] ────
        # Per-request round-trip time (request build → response received)
        # for each model backend. Distinct from `tts_synth_ms` (TTS-only,
        # end-to-end audio latency) and `end_to_end_lag_ms` (live-path
        # listener latency). These RTT histograms drive (a) the W4
        # adaptive-timeout logic for synthetic probes, and (b) the W5
        # dashboard tile "ASR RTT p95".
        self.asr_request_rtt_ms: deque[float] = deque(maxlen=256)
        self.translate_request_rtt_ms: deque[float] = deque(maxlen=256)
        self.diarize_request_rtt_ms: deque[float] = deque(maxlen=256)

        # ── Watchdog fire counters [W5 — every fire, plus escalations] ──
        # `watchdog_fires_total` is incremented on EVERY ASR watchdog
        # fire (W6b will wire the increment in `backends/asr_vllm.py`).
        # `_watchdog_fire_timestamps` powers the per-minute rate tile.
        # `watchdog_escalations_total` separately counts the >=3
        # consecutive-fires transition into RECOVERY_PENDING (W6b).
        self.watchdog_fires_total: int = 0
        self._watchdog_fire_timestamps: deque[float] = deque(maxlen=256)
        self.watchdog_escalations_total: int = 0

        # ── ASR final-event metrics [W5 — final-latency tile] ──────
        # `utterance_end_to_final_ms` samples the latency from the last
        # audio chunk of an utterance to the moment the ASR final
        # transcript is emitted. `last_final_ts` powers the
        # "Time Since Final" staleness tile.
        self.utterance_end_to_final_ms: deque[float] = deque(maxlen=256)
        self.last_final_ts: float = 0.0

    @property
    def elapsed_seconds(self) -> float:
        if self.meeting_start == 0:
            return 0.0
        return time.monotonic() - self.meeting_start

    @property
    def asr_events_per_second(self) -> float:
        elapsed = self.elapsed_seconds
        return self.asr_events / elapsed if elapsed > 0 else 0.0

    @property
    def avg_translation_ms(self) -> float:
        if self.translations_completed == 0:
            return 0.0
        return self.translation_total_ms / self.translations_completed

    def to_dict(self) -> dict:
        tts_block = {
            "queue_depth": state.tts_queue.qsize() if state.tts_queue else 0,
            "queue_maxsize": state.TTS_QUEUE_MAXSIZE,
            "workers_busy": state.tts_in_flight,
            "workers_total": state.TTS_WORKER_COUNT,
            "container_concurrency": state.TTS_CONTAINER_MAX_CONCURRENCY,
            "submitted": self.tts_submitted,
            "delivered": self.tts_delivered,
            "drops": {
                "filler": self.tts_dropped_filler,
                "stale_producer": self.tts_dropped_stale_producer,
                "stale_worker": self.tts_dropped_stale_worker,
                "pre_synth": self.tts_dropped_pre_synth,
                "post_synth": self.tts_dropped_post_synth,
                "queue_full": self.tts_dropped_queue_full,
                "missing_origin": self.tts_dropped_missing_origin,
            },
            "timeouts": self.tts_synth_timeouts,
            "synth_ms": _percentile_dict(self.tts_synth_ms),
            "upstream_lag_ms": _percentile_dict(self.upstream_lag_ms),
            "tts_post_translation_lag_ms": _percentile_dict(self.tts_post_translation_lag_ms),
            "end_to_end_lag_ms": _percentile_dict(self.end_to_end_lag_ms),
            "oldest_inflight_age_ms": (
                int((time.monotonic() - min(state.tts_inflight_started.values())) * 1000)
                if state.tts_inflight_started
                else 0
            ),
            "last_delivery_age_ms": (
                int((time.monotonic() - self.last_delivery_at) * 1000)
                if self.last_delivery_at
                else None
            ),
            "health": self.tts_health_state,
            "health_since": round(self.tts_health_since, 2),
        }
        listener_block = {
            "connected": len(state._audio_out_clients),
            "deliveries": self.listener_deliveries,
            "send_failed": self.listener_send_failed,
            "removed_on_send_error": self.listener_removed_on_send_error,
            "send_ms": _percentile_dict(self.listener_send_ms),
        }
        # ── W5 — derived reliability fields ───────────────────────
        now = time.monotonic()
        # Watchdog fires per minute: count timestamps within the last 60s
        # (the deque is bounded at 256, so this is at most a 256-element
        # filter — cheap).
        watchdog_fires_per_min = sum(
            1 for ts in self._watchdog_fire_timestamps if now - ts < 60.0
        )
        # Time since the most recent ASR final, in seconds. None if no
        # final has been emitted yet (fresh meeting / boot).
        time_since_last_final_s: float | None = None
        if self.last_final_ts > 0:
            time_since_last_final_s = round(now - self.last_final_ts, 1)

        return {
            "elapsed_s": round(self.elapsed_seconds, 1),
            "audio_chunks": self.audio_chunks,
            "audio_s": round(self.audio_seconds, 1),
            "asr_events": self.asr_events,
            "asr_partials": self.asr_partials,
            "asr_finals": self.asr_finals,
            "asr_eps": round(self.asr_events_per_second, 1),
            "translations_submitted": self.translations_submitted,
            "translations_completed": self.translations_completed,
            "translations_failed": self.translations_failed,
            "avg_translation_ms": round(self.avg_translation_ms),
            "translate_warmup_ms": round(self.translate_warmup_ms),
            "asr_load_ms": round(self.asr_load_ms),
            "tts": tts_block,
            "listener": listener_block,
            "loop_lag_ms": _percentile_dict(self.loop_lag_ms),
            "asr_request_rtt_ms": _percentile_dict(self.asr_request_rtt_ms),
            "translate_request_rtt_ms": _percentile_dict(self.translate_request_rtt_ms),
            "diarize_request_rtt_ms": _percentile_dict(self.diarize_request_rtt_ms),
            "utterance_end_to_final_ms": _percentile_dict(self.utterance_end_to_final_ms),
            "watchdog_fires_total": self.watchdog_fires_total,
            "watchdog_fires_per_min": watchdog_fires_per_min,
            "watchdog_escalations_total": self.watchdog_escalations_total,
            "time_since_last_final_s": time_since_last_final_s,
            "crash": _sanitised_crash_state(),
        }


# ── TTS health evaluator [Phase 2 + P1-3-i2 + P1-4-i2] ────────────

_TTS_NO_PROGRESS_STALL_S = 8.0
_TTS_STALL_DWELL_S = 3.0
_TTS_DEGRADED_DWELL_S = 1.5

# Hysteresis candidate timestamps — module-level so a server restart
# (which re-imports this module) starts from a clean slate but a
# transient ``healthy → degraded`` flap can finish dwelling within
# the lifetime of a single process.
_tts_stall_candidate_since: float | None = None
_tts_degraded_candidate_since: float | None = None


def _commit_tts_health(new_state: str, now: float) -> None:
    if new_state != state.metrics.tts_health_state:
        logger.warning("TTS health: %s → %s", state.metrics.tts_health_state, new_state)
        state.metrics.tts_health_state = new_state
        state.metrics.tts_health_since = now


async def tts_health_evaluator() -> None:
    """Background state machine for TTS health [P1-3-i2 + P1-4-i2].

    Runs every 500 ms and mutates ``state.metrics.tts_health_state``.
    Uses separate candidate timestamps for stall vs degraded so
    hysteresis is honoured (the "committed since" timestamp is NOT
    used for dwell). Reads progress-based signals (last_delivery_at,
    state.tts_inflight_started) as well as percentile and saturation
    signals so a no-progress hang is caught even when percentiles
    stop moving.
    """
    global _tts_stall_candidate_since, _tts_degraded_candidate_since
    while True:
        try:
            await asyncio.sleep(0.5)
            now = time.monotonic()

            qsize = state.tts_queue.qsize() if state.tts_queue else 0
            in_flight = state.tts_in_flight
            queue_saturated = (
                in_flight >= state.TTS_CONTAINER_MAX_CONCURRENCY
                and qsize >= state.TTS_QUEUE_MAXSIZE
            )

            e2e_p95 = _percentile(sorted(state.metrics.end_to_end_lag_ms), 0.95)

            no_progress_stall = (
                in_flight > 0
                and state.metrics.last_delivery_at > 0
                and (now - state.metrics.last_delivery_at) > _TTS_NO_PROGRESS_STALL_S
            )
            oldest_inflight_age = 0.0
            if state.tts_inflight_started:
                oldest_inflight_age = now - min(state.tts_inflight_started.values())
            expected = max(
                state.TTS_EXPECTED_SYNTH_DEFAULT_S,
                (state.metrics.tts_synth_ms_p95 or 0) / 1000.0,
            )
            stuck_request = oldest_inflight_age > (expected + 2.0)

            backend_degraded = bool(
                state.tts_backend and getattr(state.tts_backend, "degraded", False)
            )

            stall_condition = (
                backend_degraded or no_progress_stall or (e2e_p95 is not None and e2e_p95 > 6000)
            )
            degraded_condition = (
                queue_saturated or stuck_request or (e2e_p95 is not None and e2e_p95 > 3500)
            )

            if stall_condition:
                if _tts_stall_candidate_since is None:
                    _tts_stall_candidate_since = now
                if now - _tts_stall_candidate_since >= _TTS_STALL_DWELL_S:
                    _commit_tts_health("stalled", now)
                _tts_degraded_candidate_since = None
            else:
                _tts_stall_candidate_since = None
                if degraded_condition:
                    if _tts_degraded_candidate_since is None:
                        _tts_degraded_candidate_since = now
                    if now - _tts_degraded_candidate_since >= _TTS_DEGRADED_DWELL_S:
                        _commit_tts_health("degraded", now)
                else:
                    _tts_degraded_candidate_since = None
                    _commit_tts_health("healthy", now)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("TTS health evaluator crashed: %s", e)
