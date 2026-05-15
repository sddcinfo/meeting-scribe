"""Phase B progress sidecar — server-side contract.

Pins:
  * In-flight writes carry ``session_uuid`` + ``started_at`` so the
    boot sweep can distinguish live from orphan.
  * Success-terminal removes the sidecar.
  * Failure writes a stable error code + operator-safe message; no raw
    exception text leaks through.
  * Reprocess success and explicit dismiss both clear the sidecar.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from meeting_scribe.config import ServerConfig
from meeting_scribe.storage import MeetingStorage


@pytest.fixture
def storage_with_meeting(tmp_path: Path) -> tuple[MeetingStorage, str]:
    config = ServerConfig()
    config.meetings_dir = tmp_path / "meetings"
    storage = MeetingStorage(config)
    mid = "test-meeting-1"
    (config.meetings_dir / mid).mkdir(parents=True, exist_ok=True)
    return storage, mid


def test_write_phase_b_progress_persists_session_uuid_and_eta(
    storage_with_meeting: tuple[MeetingStorage, str],
) -> None:
    storage, mid = storage_with_meeting
    storage.write_phase_b_progress(
        mid,
        step=4,
        total=7,
        label="Running full-audio diarization...",
        eta_seconds=42,
        session_uuid="abc-123",
        started_at=1700000000.0,
    )

    payload = storage.read_phase_b_progress(mid)
    assert payload is not None
    assert payload["terminal"] is False
    assert payload["step"] == 4
    assert payload["eta_seconds"] == 42
    assert payload["session_uuid"] == "abc-123"
    assert payload["started_at"] == 1700000000.0
    # No raw exception fields can appear on an in-flight write.
    assert "error" not in payload


def test_write_phase_b_failure_uses_stable_code(
    storage_with_meeting: tuple[MeetingStorage, str],
) -> None:
    storage, mid = storage_with_meeting
    storage.write_phase_b_failure(
        mid,
        step=5,
        total=7,
        label="Generating timeline...",
        code="diarize_failed",
        message="Speaker diarization failed — try Reprocess",
    )
    payload = storage.read_phase_b_progress(mid)
    assert payload is not None
    assert payload["terminal"] is True
    assert payload["error"]["code"] == "diarize_failed"
    assert payload["error"]["message"].startswith("Speaker diarization")
    # No session_uuid on failure — any process can mark a meeting failed.
    assert "session_uuid" not in payload


def test_clear_phase_b_progress_removes_sidecar(
    storage_with_meeting: tuple[MeetingStorage, str],
) -> None:
    storage, mid = storage_with_meeting
    storage.write_phase_b_progress(
        mid,
        step=3,
        total=7,
        label="Saving speaker data...",
        eta_seconds=30,
        session_uuid="abc",
        started_at=0.0,
    )
    assert storage.read_phase_b_progress(mid) is not None
    storage.clear_phase_b_progress(mid)
    assert storage.read_phase_b_progress(mid) is None
    # Idempotent — repeated clear must not raise.
    storage.clear_phase_b_progress(mid)


def test_sidecar_survives_simulated_process_restart(
    storage_with_meeting: tuple[MeetingStorage, str], tmp_path: Path
) -> None:
    storage, mid = storage_with_meeting
    storage.write_phase_b_progress(
        mid,
        step=4,
        total=7,
        label="Running diarization...",
        eta_seconds=20,
        session_uuid="proc-A",
        started_at=1000.0,
    )

    # Simulate a process restart by spinning up a fresh storage handle.
    config = ServerConfig()
    config.meetings_dir = storage._meetings_dir
    fresh_storage = MeetingStorage(config)
    payload = fresh_storage.read_phase_b_progress(mid)
    assert payload is not None
    assert payload["session_uuid"] == "proc-A"


def test_read_phase_b_progress_returns_none_for_missing_file(
    storage_with_meeting: tuple[MeetingStorage, str],
) -> None:
    storage, mid = storage_with_meeting
    assert storage.read_phase_b_progress(mid) is None


def test_read_phase_b_progress_returns_none_for_corrupt_file(
    storage_with_meeting: tuple[MeetingStorage, str],
) -> None:
    storage, mid = storage_with_meeting
    sidecar = storage._phase_b_sidecar_path(mid)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("{ not valid json")
    assert storage.read_phase_b_progress(mid) is None
