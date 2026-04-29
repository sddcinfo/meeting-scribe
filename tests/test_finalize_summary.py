"""Regression tests for the summary-status envelope (A1).

Covers the meeting `3db4286e-...` failure mode: ``_parallel_summary_task``
returned ``{"error": ...}`` and the failure was silent — no log, no
``summary.json``, no status artifact.

These tests exercise the helper module directly. The route-level
integration test for the parallel-summary path lives separately because
``_stop_meeting_locked`` requires a fully bound meeting + storage +
diarization pipeline, which is heavier than a unit test should be.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from meeting_scribe.server_support.summary_status import (
    STATUS_FILENAME,
    SummaryErrorCode,
    SummaryStatus,
    classify_summary_error,
    is_stale,
    next_attempt_id,
    read_status,
    transcript_hash,
    write_status,
)

_PII_FRAGMENT = "TRANSCRIPT_LEAK_CANARY_xyz123"


def _journal(tmp_path: Path) -> Path:
    p = tmp_path / "journal.jsonl"
    p.write_text(json.dumps({"is_final": True, "text": "hello", "segment_id": "s1"}) + "\n")
    return p


class TestClassifier:
    def test_timeout_exception(self):
        exc = httpx.ReadTimeout(f"timed out reading {_PII_FRAGMENT}")
        assert classify_summary_error(exc) is SummaryErrorCode.VLLM_TIMEOUT

    def test_5xx_string(self):
        assert classify_summary_error("HTTP 503 from vLLM") is SummaryErrorCode.VLLM_5XX

    def test_unreachable(self):
        exc = httpx.ConnectError("Connection refused")
        assert classify_summary_error(exc) is SummaryErrorCode.VLLM_UNREACHABLE

    def test_empty_transcript(self):
        d = {"error": "No transcript segments"}
        assert classify_summary_error(d) is SummaryErrorCode.TRANSCRIPT_EMPTY

    def test_unknown_falls_through(self):
        assert classify_summary_error(RuntimeError("totally novel")) is SummaryErrorCode.INTERNAL

    def test_classifier_never_returns_input(self):
        # Belt-and-suspenders: enum values are a closed set, none of them
        # contain the canary fragment.
        exc = RuntimeError(_PII_FRAGMENT)
        code = classify_summary_error(exc)
        assert _PII_FRAGMENT not in code.value


class TestStatusWrite:
    def test_generating_envelope(self, tmp_path):
        journal_path = _journal(tmp_path)
        write_status(
            tmp_path,
            SummaryStatus.GENERATING,
            attempt_id=1,
            journal_path=journal_path,
        )
        env = read_status(tmp_path)
        assert env["status"] == "generating"
        assert env["attempt_id"] == 1
        assert env["transcript_hash"].startswith("sha256:")
        assert env["error_code"] is None
        assert env["user_safe_message"] is None

    def test_error_envelope_has_no_pii(self, tmp_path):
        """The CRITICAL P0 test: no exception text leaks to disk."""
        journal_path = _journal(tmp_path)
        exc = httpx.ReadTimeout(_PII_FRAGMENT)
        code = classify_summary_error(exc)
        write_status(
            tmp_path,
            SummaryStatus.ERROR,
            attempt_id=2,
            journal_path=journal_path,
            error_code=code,
        )
        raw = (tmp_path / STATUS_FILENAME).read_text()
        assert _PII_FRAGMENT not in raw, "raw exception text must NOT land on disk"
        env = json.loads(raw)
        assert env["status"] == "error"
        assert env["error_code"] == "vllm_timeout"
        assert env["user_safe_message"] == "Summary backend timed out"
        assert env["retryable"] is True
        assert env["attempt_id"] == 2

    def test_complete_envelope(self, tmp_path):
        journal_path = _journal(tmp_path)
        write_status(
            tmp_path,
            SummaryStatus.COMPLETE,
            attempt_id=3,
            journal_path=journal_path,
        )
        env = read_status(tmp_path)
        assert env["status"] == "complete"
        assert env["error_code"] is None
        assert env["user_safe_message"] is None

    def test_attempt_id_increments(self, tmp_path):
        journal_path = _journal(tmp_path)
        assert next_attempt_id(tmp_path) == 1
        write_status(
            tmp_path,
            SummaryStatus.ERROR,
            attempt_id=1,
            journal_path=journal_path,
            error_code=SummaryErrorCode.VLLM_TIMEOUT,
        )
        assert next_attempt_id(tmp_path) == 2
        write_status(
            tmp_path,
            SummaryStatus.COMPLETE,
            attempt_id=2,
            journal_path=journal_path,
        )
        assert next_attempt_id(tmp_path) == 3


class TestStaleness:
    def test_unchanged_journal_is_fresh(self, tmp_path):
        journal_path = _journal(tmp_path)
        write_status(
            tmp_path,
            SummaryStatus.COMPLETE,
            attempt_id=1,
            journal_path=journal_path,
        )
        assert is_stale(tmp_path, journal_path) is False

    def test_journal_mutation_marks_stale(self, tmp_path):
        journal_path = _journal(tmp_path)
        write_status(
            tmp_path,
            SummaryStatus.COMPLETE,
            attempt_id=1,
            journal_path=journal_path,
        )
        # Append a new event — journal hash must change.
        with journal_path.open("a") as f:
            f.write(json.dumps({"is_final": True, "text": "new", "segment_id": "s2"}) + "\n")
        assert is_stale(tmp_path, journal_path) is True

    def test_missing_status_is_not_stale(self, tmp_path):
        journal_path = _journal(tmp_path)
        assert is_stale(tmp_path, journal_path) is False


class TestTranscriptHash:
    def test_empty_journal(self, tmp_path):
        h = transcript_hash(tmp_path / "absent.jsonl")
        assert h == "sha256:empty"

    def test_deterministic(self, tmp_path):
        journal_path = _journal(tmp_path)
        assert transcript_hash(journal_path) == transcript_hash(journal_path)
