"""Tests for the eager-summary loop telemetry (A2).

The loop is patched to a single iteration via short-circuiting the
``while`` predicate so the test runs in milliseconds. We verify that
``state._eager_summary_metrics`` reflects success and error paths
without leaking raw exception text.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from meeting_scribe.runtime import state

_PII_FRAGMENT = "EAGER_SUMMARY_LEAK_CANARY_xyz"


@pytest.fixture(autouse=True)
def _reset_metrics():
    state._eager_summary_metrics.last_start_at = None
    state._eager_summary_metrics.last_success_at = None
    state._eager_summary_metrics.last_error_code = None
    state._eager_summary_metrics.last_skipped_reason = None
    state._eager_summary_metrics.in_flight = False
    state._eager_summary_metrics.draft_event_count_at_last_run = 0
    state._eager_summary_metrics.runs_total = 0
    state._eager_summary_metrics.errors_total = 0
    state._eager_summary_event_count = 0
    state._eager_summary_cache = None
    yield


def test_metrics_dataclass_defaults():
    m = state._eager_summary_metrics
    assert m.last_start_at is None
    assert m.last_success_at is None
    assert m.last_error_code is None
    assert m.in_flight is False
    assert m.runs_total == 0
    assert m.errors_total == 0


def test_classifier_drives_error_code_no_pii(monkeypatch):
    """Direct test of the metrics-update path: an exception lands as
    enum, not raw text."""
    from meeting_scribe.server_support.summary_status import (
        classify_summary_error,
    )

    exc = httpx.ReadTimeout(_PII_FRAGMENT)
    code = classify_summary_error(exc).value
    state._eager_summary_metrics.last_error_code = code
    state._eager_summary_metrics.errors_total += 1
    assert state._eager_summary_metrics.last_error_code == "vllm_timeout"
    assert _PII_FRAGMENT not in (state._eager_summary_metrics.last_error_code or "")


@pytest.mark.asyncio
async def test_loop_records_success(tmp_path, monkeypatch):
    """Integration: stub generate_draft_summary, run one loop iteration,
    assert metrics reflect a successful draft."""
    from meeting_scribe.models import MeetingMeta, MeetingState
    from meeting_scribe.runtime import meeting_loops as loops

    # Build minimal journal so n_finals >= 50
    meeting_id = "telemetry-success"
    mdir = tmp_path / meeting_id
    mdir.mkdir(parents=True)
    journal = mdir / "journal.jsonl"
    with journal.open("w") as f:
        for i in range(60):
            f.write(
                json.dumps({"is_final": True, "text": f"line {i}", "segment_id": str(i)}) + "\n"
            )

    class _FakeStorage:
        def _meeting_dir(self, mid: str) -> Path:
            return mdir

    class _FakeConfig:
        translate_vllm_url = "http://unused"

    monkeypatch.setattr(state, "storage", _FakeStorage())
    monkeypatch.setattr(state, "config", _FakeConfig())
    monkeypatch.setattr(
        state,
        "current_meeting",
        MeetingMeta(meeting_id=meeting_id, state=MeetingState.RECORDING),
    )

    async def _fake_draft(*args, **kwargs):
        return {"metadata": {"generation_ms": 42.0}, "topics": []}, 60

    monkeypatch.setattr("meeting_scribe.summary.generate_draft_summary", _fake_draft)
    monkeypatch.setattr(loops.asyncio, "sleep", _instant_sleep_then_stop(meeting_id))

    await loops._eager_summary_loop(meeting_id)

    m = state._eager_summary_metrics
    assert m.runs_total == 1
    assert m.last_start_at is not None
    assert m.last_success_at is not None
    assert m.last_error_code is None
    assert m.draft_event_count_at_last_run == 60
    assert m.errors_total == 0


@pytest.mark.asyncio
async def test_loop_records_error_no_pii(tmp_path, monkeypatch):
    """Integration: stub generate_draft_summary to raise, assert
    last_error_code is the enum value and the canary fragment never
    lands in any persisted state."""
    from meeting_scribe.models import MeetingMeta, MeetingState
    from meeting_scribe.runtime import meeting_loops as loops

    meeting_id = "telemetry-error"
    mdir = tmp_path / meeting_id
    mdir.mkdir(parents=True)
    journal = mdir / "journal.jsonl"
    with journal.open("w") as f:
        for i in range(60):
            f.write(json.dumps({"is_final": True, "text": f"x{i}", "segment_id": str(i)}) + "\n")

    class _FakeStorage:
        def _meeting_dir(self, mid: str) -> Path:
            return mdir

    class _FakeConfig:
        translate_vllm_url = "http://unused"

    monkeypatch.setattr(state, "storage", _FakeStorage())
    monkeypatch.setattr(state, "config", _FakeConfig())
    monkeypatch.setattr(
        state,
        "current_meeting",
        MeetingMeta(meeting_id=meeting_id, state=MeetingState.RECORDING),
    )

    async def _raising_draft(*args, **kwargs):
        raise httpx.ReadTimeout(_PII_FRAGMENT)

    monkeypatch.setattr("meeting_scribe.summary.generate_draft_summary", _raising_draft)
    monkeypatch.setattr(loops.asyncio, "sleep", _instant_sleep_then_stop(meeting_id))

    await loops._eager_summary_loop(meeting_id)

    m = state._eager_summary_metrics
    assert m.errors_total == 1
    assert m.last_error_code == "vllm_timeout"
    assert _PII_FRAGMENT not in (m.last_error_code or "")
    assert m.last_success_at is None


def _instant_sleep_then_stop(meeting_id: str):
    """Replacement for asyncio.sleep that lets the warmup + initial
    iteration run, then clears state.current_meeting so the loop exits.

    Uses the unpatched real_sleep captured at module import so the
    patched namespace doesn't recurse.
    """
    real_sleep = asyncio.sleep
    call_count = {"n": 0}

    async def _sleep(_seconds):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            state.current_meeting = None
        await real_sleep(0)

    return _sleep
