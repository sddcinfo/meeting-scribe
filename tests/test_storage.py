"""Tests for meeting storage — journal replay, metadata persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meeting_scribe.models import MeetingMeta
from meeting_scribe.storage import MeetingStorage


@pytest.fixture
def storage(tmp_path):
    """Create a MeetingStorage with a temp directory."""
    from meeting_scribe.config import ServerConfig

    meetings_dir = tmp_path / "meetings"
    meetings_dir.mkdir()
    cfg = ServerConfig()
    # MeetingStorage expects meetings_dir as a Path-compatible value
    s = MeetingStorage(cfg)
    s._meetings_dir = meetings_dir
    return s


class TestJournalReplay:
    """read_journal_raw for multi-browser sync."""

    def test_read_empty_journal(self, storage):
        meta = storage.create_meeting(MeetingMeta())
        lines = storage.read_journal_raw(meta.meeting_id)
        assert lines == []

    def test_read_journal_with_events(self, storage):
        meta = storage.create_meeting(MeetingMeta())
        mid = meta.meeting_id
        # Write some events
        journal_path = Path(storage._meetings_dir) / mid / "journal.jsonl"
        events = [
            json.dumps({"segment_id": f"s{i}", "text": f"Event {i}", "is_final": True})
            for i in range(5)
        ]
        journal_path.write_text("\n".join(events) + "\n")

        lines = storage.read_journal_raw(mid)
        assert len(lines) == 5
        assert json.loads(lines[0])["segment_id"] == "s0"

    def test_read_journal_max_lines(self, storage):
        meta = storage.create_meeting(MeetingMeta())
        mid = meta.meeting_id
        journal_path = Path(storage._meetings_dir) / mid / "journal.jsonl"
        events = [json.dumps({"segment_id": f"s{i}", "text": f"Event {i}"}) for i in range(100)]
        journal_path.write_text("\n".join(events) + "\n")

        lines = storage.read_journal_raw(mid, max_lines=10)
        assert len(lines) == 10
        # Should be the LAST 10 events
        assert json.loads(lines[0])["segment_id"] == "s90"

    def test_read_journal_nonexistent_meeting(self, storage):
        lines = storage.read_journal_raw("nonexistent-id")
        assert lines == []
