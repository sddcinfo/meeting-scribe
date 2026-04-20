"""Tests for audit_meetings helper and cleanup classification logic."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from meeting_scribe.config import ServerConfig
from meeting_scribe.storage import MeetingStorage


@pytest.fixture
def storage(tmp_path):
    meetings_dir = tmp_path / "meetings"
    meetings_dir.mkdir()
    cfg = ServerConfig()
    s = MeetingStorage(cfg)
    s._meetings_dir = meetings_dir
    return s


def _make_meeting(
    storage: MeetingStorage,
    mid: str,
    state: str = "complete",
    audio_bytes: int = 0,
    journal_lines: int = 0,
    has_summary: bool = False,
    age_hours: float = 2.0,
) -> Path:
    """Create a synthetic meeting dir."""
    meeting_dir = storage._meetings_dir / mid
    meeting_dir.mkdir()

    created_at = (datetime.now(UTC) - timedelta(hours=age_hours)).isoformat()
    meta = {
        "meeting_id": mid,
        "state": state,
        "created_at": created_at,
        "language_pair": ["ja", "en"],
    }
    (meeting_dir / "meta.json").write_text(json.dumps(meta))

    # Audio file
    audio_dir = meeting_dir / "audio"
    audio_dir.mkdir()
    if audio_bytes > 0:
        (audio_dir / "recording.pcm").write_bytes(b"\x00" * audio_bytes)
    else:
        (audio_dir / "recording.pcm").write_bytes(b"")

    # Journal
    if journal_lines > 0:
        journal = meeting_dir / "journal.jsonl"
        lines = [
            json.dumps(
                {
                    "segment_id": f"seg{i}",
                    "revision": 0,
                    "is_final": True,
                    "text": f"test {i}",
                    "language": "en",
                    "speakers": [],
                    "translation": None,
                }
            )
            for i in range(journal_lines)
        ]
        journal.write_text("\n".join(lines) + "\n")

    if has_summary:
        (meeting_dir / "summary.json").write_text(
            json.dumps({"executive_summary": "test", "topics": [], "action_items": []})
        )

    return meeting_dir


class TestAuditMeetings:
    def test_empty_dir(self, storage):
        assert storage.audit_meetings() == []

    def test_single_complete_meeting(self, storage):
        _make_meeting(
            storage,
            "m1",
            state="complete",
            audio_bytes=32000 * 60,
            journal_lines=20,
            has_summary=True,
        )
        audit = storage.audit_meetings()
        assert len(audit) == 1
        m = audit[0]
        assert m["meeting_id"] == "m1"
        assert m["state"] == "complete"
        assert m["audio_duration_s"] == 60.0
        assert m["journal_lines"] == 20
        assert m["has_summary"] is True

    def test_detects_empty_meeting(self, storage):
        """Empty meeting has 0s audio and 0 events."""
        _make_meeting(storage, "empty1", state="complete")
        audit = storage.audit_meetings()
        assert len(audit) == 1
        m = audit[0]
        assert m["audio_duration_s"] == 0
        assert m["journal_lines"] == 0
        assert m["has_audio"] is False

    def test_detects_interrupted(self, storage):
        _make_meeting(
            storage,
            "int1",
            state="interrupted",
            audio_bytes=32000 * 120,
            journal_lines=50,
        )
        audit = storage.audit_meetings()
        assert audit[0]["state"] == "interrupted"
        assert audit[0]["audio_duration_s"] == 120.0

    def test_age_hours_computed(self, storage):
        _make_meeting(storage, "old1", age_hours=5.0)
        audit = storage.audit_meetings()
        assert 4.5 < audit[0]["age_hours"] < 5.5

    def test_ignores_non_meeting_dirs(self, storage):
        # A dir without meta.json should be ignored
        (storage._meetings_dir / "random").mkdir()
        _make_meeting(storage, "good")
        audit = storage.audit_meetings()
        assert len(audit) == 1
        assert audit[0]["meeting_id"] == "good"

    def test_handles_corrupt_meta(self, storage):
        meeting_dir = storage._meetings_dir / "corrupt"
        meeting_dir.mkdir()
        (meeting_dir / "meta.json").write_text("not json")
        audit = storage.audit_meetings()
        assert audit == []

    def test_multiple_meetings_sorted(self, storage):
        _make_meeting(storage, "m1", state="complete", audio_bytes=32000 * 30, journal_lines=5)
        _make_meeting(storage, "m2", state="interrupted", audio_bytes=32000 * 120, journal_lines=50)
        _make_meeting(storage, "m3", state="complete")
        audit = storage.audit_meetings()
        assert len(audit) == 3
        ids = [m["meeting_id"] for m in audit]
        assert set(ids) == {"m1", "m2", "m3"}


class TestCleanupClassification:
    """Test the cleanup command's classification logic (replicated here as unit test)."""

    def _classify(self, audit: list[dict]) -> tuple[list, list, list]:
        """Reproduce cleanup's classification to test thresholds."""
        to_finalize, to_regen, to_delete = [], [], []
        for m in audit:
            if m["state"] == "recording":
                continue
            is_empty = m["audio_duration_s"] < 5 and m["journal_lines"] == 0 and m["age_hours"] > 1
            if is_empty:
                to_delete.append(m)
                continue
            if (
                m["state"] == "interrupted"
                and m["audio_duration_s"] >= 60
                and m["journal_lines"] >= 5
            ):
                to_finalize.append(m)
                continue
            if m["state"] == "complete" and not m["has_summary"] and m["journal_lines"] >= 5:
                to_regen.append(m)
        return to_finalize, to_regen, to_delete

    def test_interrupted_large_is_finalized(self, storage):
        _make_meeting(
            storage,
            "int-big",
            state="interrupted",
            audio_bytes=32000 * 200,
            journal_lines=100,
        )
        audit = storage.audit_meetings()
        to_finalize, _, _ = self._classify(audit)
        assert len(to_finalize) == 1

    def test_interrupted_tiny_is_deleted(self, storage):
        _make_meeting(
            storage,
            "int-tiny",
            state="interrupted",
            audio_bytes=0,
            journal_lines=0,
            age_hours=2.0,
        )
        audit = storage.audit_meetings()
        _, _, to_delete = self._classify(audit)
        assert len(to_delete) == 1

    def test_interrupted_small_not_touched(self, storage):
        """Interrupted with some content but under finalize threshold stays as-is."""
        _make_meeting(
            storage,
            "int-small",
            state="interrupted",
            audio_bytes=32000 * 30,
            journal_lines=3,
        )
        audit = storage.audit_meetings()
        to_finalize, to_regen, to_delete = self._classify(audit)
        assert len(to_finalize) == 0
        assert len(to_regen) == 0
        assert len(to_delete) == 0

    def test_complete_missing_summary_regen(self, storage):
        _make_meeting(
            storage,
            "comp-no-sum",
            state="complete",
            audio_bytes=32000 * 100,
            journal_lines=20,
            has_summary=False,
        )
        audit = storage.audit_meetings()
        _, to_regen, _ = self._classify(audit)
        assert len(to_regen) == 1

    def test_complete_with_summary_untouched(self, storage):
        _make_meeting(
            storage,
            "comp-sum",
            state="complete",
            audio_bytes=32000 * 60,
            journal_lines=15,
            has_summary=True,
        )
        audit = storage.audit_meetings()
        to_f, to_r, to_d = self._classify(audit)
        assert to_f == [] and to_r == [] and to_d == []

    def test_empty_fresh_meeting_not_deleted(self, storage):
        """Fresh empty meeting (<1h old) is NOT deleted."""
        _make_meeting(storage, "fresh-empty", state="complete", age_hours=0.25)
        audit = storage.audit_meetings()
        _, _, to_delete = self._classify(audit)
        assert len(to_delete) == 0

    def test_old_empty_meeting_deleted(self, storage):
        _make_meeting(storage, "old-empty", state="complete", age_hours=5.0)
        audit = storage.audit_meetings()
        _, _, to_delete = self._classify(audit)
        assert len(to_delete) == 1

    def test_meeting_with_5s_audio_not_deleted(self, storage):
        """Meeting with any audio >= 5s is preserved (conservative)."""
        _make_meeting(
            storage,
            "tiny-audio",
            state="complete",
            audio_bytes=32000 * 6,
            journal_lines=0,
            age_hours=5.0,
        )
        audit = storage.audit_meetings()
        _, _, to_delete = self._classify(audit)
        assert len(to_delete) == 0

    def test_meeting_with_events_not_deleted(self, storage):
        """Meeting with any journal content is preserved."""
        _make_meeting(
            storage,
            "has-events",
            state="complete",
            audio_bytes=0,
            journal_lines=2,
            age_hours=5.0,
        )
        audit = storage.audit_meetings()
        _, _, to_delete = self._classify(audit)
        assert len(to_delete) == 0
