"""Boot sweep — rewrite orphan Phase B sidecars to ``interrupted``.

A meeting-scribe process crash mid-Phase-B leaves the sidecar in the
in-flight shape with a ``session_uuid`` from the dead process. On the
next boot, ``sweep_orphan_phase_b_sidecars`` must rewrite each one as
a failure-terminal sidecar with ``code="interrupted"`` so the API
never serves ambiguous in-flight from a process that no longer exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meeting_scribe.config import ServerConfig
from meeting_scribe.storage import MeetingStorage


def _seed_sidecar(
    storage: MeetingStorage,
    mid: str,
    *,
    terminal: bool,
    session_uuid: str = "stale",
    step: int = 4,
    label: str = "Running full-audio diarization...",
) -> None:
    meeting_dir = storage._meetings_dir / mid
    meeting_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "terminal": terminal,
        "step": step,
        "total_steps": 7,
        "label": label,
    }
    if terminal:
        payload["error"] = {
            "code": "diarize_failed",
            "message": "Test failure",
            "step": step,
        }
    else:
        payload["eta_seconds"] = 30
        payload["session_uuid"] = session_uuid
        payload["started_at"] = 1700000000.0
    (meeting_dir / "phase_b_progress.json").write_text(json.dumps(payload))


@pytest.fixture
def storage(tmp_path: Path) -> MeetingStorage:
    config = ServerConfig()
    config.meetings_dir = tmp_path / "meetings"
    config.meetings_dir.mkdir(parents=True, exist_ok=True)
    return MeetingStorage(config)


def test_orphan_inflight_sidecars_become_interrupted(storage: MeetingStorage) -> None:
    # Three orphans from two prior processes, plus one fresh in-flight
    # from the "current" process (current_session_uuid below).
    _seed_sidecar(storage, "old-1", terminal=False, session_uuid="proc-A")
    _seed_sidecar(storage, "old-2", terminal=False, session_uuid="proc-A", step=5)
    _seed_sidecar(storage, "old-3", terminal=False, session_uuid="proc-B", step=6)
    _seed_sidecar(storage, "live", terminal=False, session_uuid="current")
    _seed_sidecar(storage, "already-failed", terminal=True)

    rewritten = storage.sweep_orphan_phase_b_sidecars("current")

    # Only the three orphans are rewritten — the live one is untouched
    # and the already-failed one is not re-stamped.
    assert sorted(rewritten) == ["old-1", "old-2", "old-3"]

    for mid in ("old-1", "old-2", "old-3"):
        payload = storage.read_phase_b_progress(mid)
        assert payload is not None
        assert payload["terminal"] is True
        assert payload["error"]["code"] == "interrupted"
        assert "Reprocess" in payload["error"]["message"]

    live = storage.read_phase_b_progress("live")
    assert live is not None
    assert live["terminal"] is False
    assert live["session_uuid"] == "current"

    failed = storage.read_phase_b_progress("already-failed")
    assert failed is not None
    assert failed["error"]["code"] == "diarize_failed"  # unchanged


def test_sweep_handles_missing_meetings_dir(tmp_path: Path) -> None:
    config = ServerConfig()
    # Don't create the directory — sweep must still return cleanly.
    config.meetings_dir = tmp_path / "nope"
    storage = MeetingStorage(config)
    assert storage.sweep_orphan_phase_b_sidecars("current") == []


def test_sweep_ignores_corrupt_sidecar(storage: MeetingStorage) -> None:
    meeting_dir = storage._meetings_dir / "corrupt"
    meeting_dir.mkdir(parents=True, exist_ok=True)
    (meeting_dir / "phase_b_progress.json").write_text("{ not valid")

    # Should not raise, should not rewrite a corrupt sidecar (we can't
    # tell what step it was at).
    rewritten = storage.sweep_orphan_phase_b_sidecars("current")
    assert rewritten == []


def test_sweep_is_idempotent(storage: MeetingStorage) -> None:
    _seed_sidecar(storage, "old-1", terminal=False, session_uuid="proc-A")
    first = storage.sweep_orphan_phase_b_sidecars("current")
    assert first == ["old-1"]
    # Second sweep finds the rewritten failure-terminal sidecar and
    # leaves it alone — no double-rewrite.
    second = storage.sweep_orphan_phase_b_sidecars("current")
    assert second == []
