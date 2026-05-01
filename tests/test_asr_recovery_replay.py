"""Tests for the W6a recovery state machine on VllmASRBackend.

Locks down:
- Per-submission offset tracking (success → REMOVE; failure → KEEP).
- Earliest-unresolved-offset capture on `_begin_recovery_pending()`,
  surviving 30 ordinary submission turnover events (Codex iteration-6
  P1 acceptance test).
- Recovery-generation guard on success and failure paths
  (Codex iteration-7 P1 acceptance test).
- Idempotent `_begin_recovery_pending()` (Codex iteration-7 P1).
- Defensive _MAX_UNRESOLVED_SUBMISSIONS ceiling.
- State-machine transitions NORMAL → RECOVERY_PENDING.

The actual replay loop (replay_until_caught_up) reads recording.pcm
from a live meeting directory; it is exercised end-to-end in the
W6b/Tier-3 live drill, not here.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from meeting_scribe.backends import asr_vllm as av
from meeting_scribe.backends.asr_vllm import (
    _MAX_UNRESOLVED_SUBMISSIONS,
    InflightSubmission,
    VllmASRBackend,
)
from meeting_scribe.runtime import state as runtime_state
from meeting_scribe.runtime.metrics import Metrics


@pytest.fixture
def backend(monkeypatch) -> VllmASRBackend:
    """Fresh backend wired to a fresh Metrics instance."""
    m = Metrics()
    monkeypatch.setattr(runtime_state, "metrics", m, raising=False)
    monkeypatch.setattr(runtime_state, "audio_writer", None, raising=False)
    b = VllmASRBackend()
    return b


def _seed_failed(b: VllmASRBackend, audio_start: int, audio_end: int) -> InflightSubmission:
    """Helper: pretend a submission was made + failed."""
    sub = b._track_submission_start(audio_start, audio_end)
    b._track_submission_failed(sub, RuntimeError("seeded fail"))
    return sub


def _seed_inflight(b: VllmASRBackend, audio_start: int, audio_end: int) -> InflightSubmission:
    return b._track_submission_start(audio_start, audio_end)


def _seed_complete(b: VllmASRBackend, audio_start: int, audio_end: int) -> None:
    """Helper: pretend a submission succeeded — entry is REMOVED."""
    sub = b._track_submission_start(audio_start, audio_end)
    b._track_submission_complete(sub)


class TestSubmissionTracking:
    def test_success_evicts_entry(self, backend):
        sub = backend._track_submission_start(100, 200)
        assert sub in backend._submissions
        assert backend._track_submission_complete(sub) is True
        assert sub not in backend._submissions

    def test_failure_retains_entry(self, backend):
        sub = backend._track_submission_start(100, 200)
        assert backend._track_submission_failed(sub, RuntimeError("boom")) is True
        assert sub in backend._submissions
        assert sub.status == "failed"

    def test_request_id_monotonic(self, backend):
        a = backend._track_submission_start(0, 100)
        b = backend._track_submission_start(100, 200)
        assert b.request_id > a.request_id


class TestRecoveryStartOffset:
    """The Codex iteration-4 + iteration-6 P1 acceptance tests."""

    def test_capture_uses_earliest_unresolved_offset(self, backend):
        """3 sequential failed submissions → capture starts at the EARLIEST,
        not the most recent."""
        _seed_failed(backend, 100, 200)
        _seed_failed(backend, 200, 300)
        _seed_failed(backend, 300, 400)

        backend._begin_recovery_pending()

        assert backend._recovery_state == "RECOVERY_PENDING"
        assert backend._recovery_start_offset == 100

    def test_earliest_offset_survives_30_completed_submissions(self, backend):
        """Codex iteration-6 P1 acceptance test: 1 failed at offset 50,
        followed by 30 successful submissions, then escalation. The
        successes are evicted on completion so the failed entry survives
        turnover (the prior maxlen=20 deque would have evicted it)."""
        _seed_failed(backend, 50, 150)
        for i in range(30):
            start = 150 + i * 100
            _seed_complete(backend, start, start + 100)

        # Only the 1 failed entry should remain.
        assert len(backend._submissions) == 1
        assert backend._submissions[0].audio_start_offset == 50

        backend._begin_recovery_pending()
        assert backend._recovery_start_offset == 50

    def test_no_unresolved_falls_back_with_warning(self, backend, caplog):
        """If _submissions is empty when escalation triggers, fall
        back to current_recording_pcm_offset() and log loudly."""
        with patch.object(av.state, "current_recording_pcm_offset", return_value=999):
            backend._begin_recovery_pending()
        assert backend._recovery_start_offset == 999
        assert any(
            "empty in-flight submissions" in r.message
            for r in caplog.records
        )


class TestRecoveryGenerationGuard:
    """The Codex iteration-7 P1 acceptance tests for stale responses."""

    def test_late_response_from_old_generation_is_silently_discarded(self, backend, caplog):
        import logging as _logging

        sub = backend._track_submission_start(100, 200)  # gen=0
        # Recovery escalates → generation bumps to 1.
        backend._begin_recovery_pending()
        assert backend._recovery_generation == 1

        with caplog.at_level(_logging.DEBUG, logger=av.logger.name):
            ok = backend._track_submission_complete(sub)
        assert ok is False  # caller must not process the result

        # Watchdog state, _submissions list, and generation must NOT
        # have been mutated by the stale response.
        assert backend._recovery_generation == 1

        # Exactly one DEBUG log line about the stale response.
        debug_msgs = [
            r for r in caplog.records
            if "stale response" in r.message and r.levelname == "DEBUG"
        ]
        assert len(debug_msgs) == 1, [r.message for r in caplog.records]

    def test_late_failure_from_old_generation_is_silently_discarded(self, backend, caplog):
        sub = backend._track_submission_start(100, 200)
        backend._begin_recovery_pending()
        ok = backend._track_submission_failed(sub, RuntimeError("late"))
        assert ok is False
        # Submission status must not have been changed to "failed".
        assert sub.status == "inflight"

    def test_replay_submissions_pass_guard_under_current_generation(self, backend):
        """Replay path tags submissions with the CURRENT generation,
        so they pass the guard normally."""
        backend._begin_recovery_pending()  # gen=1
        # New submission via the normal start helper picks up gen=1.
        sub = backend._track_submission_start(0, 100)
        assert sub.generation == 1
        assert backend._track_submission_complete(sub) is True


class TestIdempotentRecoveryEscalation:
    def test_double_call_is_noop(self, backend):
        """A stale-task-induced re-entry must not corrupt state."""
        backend._begin_recovery_pending()
        gen_after_first = backend._recovery_generation
        offset_after_first = backend._recovery_start_offset
        escalations_total_first = runtime_state.metrics.watchdog_escalations_total

        # Call again — should be a no-op.
        backend._begin_recovery_pending()

        assert backend._recovery_generation == gen_after_first
        assert backend._recovery_start_offset == offset_after_first
        assert runtime_state.metrics.watchdog_escalations_total == escalations_total_first


class TestStateMachine:
    def test_initial_state_is_normal(self, backend):
        assert backend._recovery_state == "NORMAL"
        assert backend._recovery_start_offset is None
        assert backend._recovery_generation == 0

    def test_begin_recovery_transitions_to_pending(self, backend):
        with patch.object(av.state, "current_recording_pcm_offset", return_value=500):
            backend._begin_recovery_pending()
        assert backend._recovery_state == "RECOVERY_PENDING"
        assert backend._recovery_requested.is_set()


class TestDefensiveCeiling:
    def test_failed_entries_evicted_at_ceiling(self, backend, caplog):
        """At >_MAX_UNRESOLVED_SUBMISSIONS the oldest *failed* entry
        is dropped with an ERROR; inflight entries are never dropped."""
        # 1 inflight at the start (must NOT be dropped).
        inflight_sub = backend._track_submission_start(0, 100)
        # _MAX failed entries. Total is _MAX+1 — one over the ceiling.
        for i in range(_MAX_UNRESOLVED_SUBMISSIONS):
            _seed_failed(backend, (i + 1) * 100, (i + 1) * 100 + 100)

        # The ceiling fires inside _track_submission_start when len >
        # _MAX. Trigger one more entry to push us into the eviction
        # branch — but use a failed seed so the chain is consistent.
        _seed_failed(
            backend,
            (_MAX_UNRESOLVED_SUBMISSIONS + 1) * 100,
            (_MAX_UNRESOLVED_SUBMISSIONS + 1) * 100 + 100,
        )

        # The oldest failed entry (request_id=2, offset=100) was dropped.
        # The inflight entry (request_id=1, offset=0) survives.
        assert inflight_sub in backend._submissions, "inflight entry was dropped (regression)"
        assert any(
            "exceeded" in r.message and r.levelname == "ERROR"
            for r in caplog.records
        ), [r.message for r in caplog.records]
