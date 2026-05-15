"""Tests for the W5 reliability metrics added to runtime.metrics.Metrics.

The W5 plan added per-backend RTT histograms, watchdog fire counters,
ASR utterance-end-to-final latency, and time-since-last-final to the
Metrics class so the dashboard tiles have a stable signal source.
This module locks the contract: the keys exist in to_dict(), values
follow the percentile-dict shape used elsewhere, watchdog fires count
EVERY fire (not just the >=3 escalations), and time-since-last-final
reports None on a fresh meeting.
"""

from __future__ import annotations

import time
from collections import deque
from typing import ClassVar

import pytest

from meeting_scribe.runtime import state as runtime_state
from meeting_scribe.runtime.metrics import Metrics


@pytest.fixture
def metrics(monkeypatch) -> Metrics:
    """Fresh Metrics instance wired into runtime.state for the duration
    of the test (so to_dict()'s sibling lookups don't blow up on
    None state)."""
    m = Metrics()
    monkeypatch.setattr(runtime_state, "metrics", m, raising=False)
    # Some to_dict() branches read tts_queue / tts_in_flight; default
    # them to safe noops if not already present in test state.
    if not hasattr(runtime_state, "tts_queue") or runtime_state.tts_queue is None:
        monkeypatch.setattr(runtime_state, "tts_queue", None, raising=False)
    return m


class TestRTTHistograms:
    def test_all_three_rtt_histograms_exist(self, metrics):
        assert isinstance(metrics.asr_request_rtt_ms, deque)
        assert isinstance(metrics.translate_request_rtt_ms, deque)
        assert isinstance(metrics.diarize_request_rtt_ms, deque)
        assert metrics.asr_request_rtt_ms.maxlen == 256
        assert metrics.translate_request_rtt_ms.maxlen == 256
        assert metrics.diarize_request_rtt_ms.maxlen == 256

    def test_to_dict_exposes_percentile_shape(self, metrics):
        # Populate ≥10 samples (the _percentile_dict floor) with ramp values.
        for ms in range(10, 110):
            metrics.asr_request_rtt_ms.append(float(ms))

        d = metrics.to_dict()
        assert "asr_request_rtt_ms" in d
        block = d["asr_request_rtt_ms"]
        assert set(block.keys()) >= {"p50", "p95", "p99", "sample_count"}
        assert block["sample_count"] == 100
        assert block["p50"] is not None and 50 <= block["p50"] <= 70
        assert block["p95"] is not None and block["p95"] > block["p50"]


class TestWatchdogFireCounters:
    def test_watchdog_fires_total_starts_at_zero(self, metrics):
        assert metrics.watchdog_fires_total == 0
        assert metrics.watchdog_escalations_total == 0

    def test_per_minute_rate_counts_recent_fires(self, metrics):
        """The dashboard tile must show fires from the FIRST fire,
        not just from the >=3 escalation point. Verifies that
        watchdog_fires_per_min in to_dict() reflects the fire-timestamps
        deque, which W6b will append to on every fire."""
        now = time.monotonic()
        # 3 recent fires + 1 ancient (>60s) fire
        metrics._watchdog_fire_timestamps.extend([now - 5, now - 10, now - 30, now - 120])
        d = metrics.to_dict()
        assert d["watchdog_fires_per_min"] == 3, d

    def test_per_minute_rate_zero_when_no_fires(self, metrics):
        d = metrics.to_dict()
        assert d["watchdog_fires_per_min"] == 0


class TestUtteranceEndToFinalAndTimeSinceFinal:
    def test_utterance_end_to_final_histogram_exists(self, metrics):
        assert isinstance(metrics.utterance_end_to_final_ms, deque)
        assert metrics.utterance_end_to_final_ms.maxlen == 256

    def test_time_since_last_final_is_none_at_fresh_start(self, metrics):
        d = metrics.to_dict()
        assert d["time_since_last_final_s"] is None

    def test_time_since_last_final_grows_after_final(self, metrics):
        metrics.last_final_ts = time.monotonic() - 4.0
        d = metrics.to_dict()
        assert d["time_since_last_final_s"] is not None
        assert 3.5 <= d["time_since_last_final_s"] <= 4.5


class TestStatusFieldShapeStability:
    """Lock down the keys the dashboard JS reads. If any of these
    rename or disappear, the frontend tile silently shows '—' — these
    assertions are the contract-fail signal."""

    EXPECTED_TOP_LEVEL_KEYS: ClassVar[set[str]] = {
        "asr_request_rtt_ms",
        "translate_request_rtt_ms",
        "diarize_request_rtt_ms",
        "utterance_end_to_final_ms",
        "watchdog_fires_total",
        "watchdog_fires_per_min",
        "watchdog_escalations_total",
        "time_since_last_final_s",
    }

    def test_w5_keys_present(self, metrics):
        d = metrics.to_dict()
        missing = self.EXPECTED_TOP_LEVEL_KEYS - set(d.keys())
        assert not missing, f"missing W5 metric keys: {missing}"
