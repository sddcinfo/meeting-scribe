"""Tests for transcript deduplication fixes.

The server broadcasts each segment multiple times:
  1. Raw event (no translation)
  2. Translation in_progress
  3. Translation done

The journal contains duplicate entries per segment_id. read_journal_raw()
must collapse these so late-joining clients see each segment once.
"""

from __future__ import annotations

import json

import pytest

from meeting_scribe.config import ServerConfig
from meeting_scribe.storage import MeetingStorage


@pytest.fixture
def storage(tmp_path):
    """Create MeetingStorage with a temp meetings directory."""
    meetings_dir = tmp_path / "meetings"
    meetings_dir.mkdir()
    cfg = ServerConfig()
    s = MeetingStorage(cfg)
    s._meetings_dir = meetings_dir
    return s


def _make_event(segment_id: str, text: str, translation: dict | None = None) -> str:
    """Build a JSONL event line."""
    e = {
        "segment_id": segment_id,
        "revision": 0,
        "is_final": True,
        "start_ms": 0,
        "end_ms": 1000,
        "language": "ja",
        "text": text,
        "speakers": [],
        "translation": translation,
    }
    return json.dumps(e)


class TestJournalDedup:
    def test_empty_journal(self, storage):
        """Empty journal returns empty list."""
        meeting_dir = storage._meetings_dir / "test-meeting"
        meeting_dir.mkdir()
        (meeting_dir / "journal.jsonl").write_text("")
        assert storage.read_journal_raw("test-meeting") == []

    def test_missing_journal(self, storage):
        """Missing journal file returns empty list."""
        assert storage.read_journal_raw("nonexistent") == []

    def test_single_segment_no_dupe(self, storage):
        """Single segment is returned as-is."""
        meeting_dir = storage._meetings_dir / "m1"
        meeting_dir.mkdir()
        journal = meeting_dir / "journal.jsonl"
        journal.write_text(_make_event("seg1", "hello") + "\n")
        result = storage.read_journal_raw("m1")
        assert len(result) == 1
        assert json.loads(result[0])["text"] == "hello"

    def test_duplicate_segment_prefers_translated(self, storage):
        """When a segment_id appears twice, keep the version with translation."""
        meeting_dir = storage._meetings_dir / "m2"
        meeting_dir.mkdir()
        journal = meeting_dir / "journal.jsonl"

        raw_event = _make_event("seg1", "こんにちは", translation=None)
        translated_event = _make_event(
            "seg1",
            "こんにちは",
            translation={"status": "done", "text": "Hello", "target_language": "en"},
        )
        journal.write_text(raw_event + "\n" + translated_event + "\n")

        result = storage.read_journal_raw("m2")
        assert len(result) == 1, "Duplicate segment should be collapsed"
        data = json.loads(result[0])
        assert data["translation"] is not None
        assert data["translation"]["text"] == "Hello"

    def test_duplicate_segment_translation_first(self, storage):
        """Translation-first order also collapses correctly."""
        meeting_dir = storage._meetings_dir / "m3"
        meeting_dir.mkdir()
        journal = meeting_dir / "journal.jsonl"

        translated = _make_event(
            "seg1", "foo", translation={"status": "done", "text": "bar", "target_language": "ja"}
        )
        raw = _make_event("seg1", "foo", translation=None)
        # Write translated first, then raw — later-seen raw should NOT overwrite
        journal.write_text(translated + "\n" + raw + "\n")

        result = storage.read_journal_raw("m3")
        assert len(result) == 1
        data = json.loads(result[0])
        assert data["translation"] is not None
        assert data["translation"]["text"] == "bar"

    def test_multiple_segments_preserve_order(self, storage):
        """Multiple unique segments preserve chronological order."""
        meeting_dir = storage._meetings_dir / "m4"
        meeting_dir.mkdir()
        journal = meeting_dir / "journal.jsonl"

        lines = [
            _make_event("seg1", "first"),
            _make_event("seg2", "second"),
            _make_event("seg3", "third"),
        ]
        journal.write_text("\n".join(lines) + "\n")

        result = storage.read_journal_raw("m4")
        assert len(result) == 3
        assert json.loads(result[0])["text"] == "first"
        assert json.loads(result[1])["text"] == "second"
        assert json.loads(result[2])["text"] == "third"

    def test_interleaved_duplicates(self, storage):
        """Duplicates interleaved with unique segments — order preserved."""
        meeting_dir = storage._meetings_dir / "m5"
        meeting_dir.mkdir()
        journal = meeting_dir / "journal.jsonl"

        lines = [
            _make_event("seg1", "a"),
            _make_event("seg2", "b"),
            _make_event(
                "seg1", "a", translation={"status": "done", "text": "A", "target_language": "en"}
            ),
            _make_event("seg3", "c"),
            _make_event(
                "seg2", "b", translation={"status": "done", "text": "B", "target_language": "en"}
            ),
        ]
        journal.write_text("\n".join(lines) + "\n")

        result = storage.read_journal_raw("m5")
        assert len(result) == 3
        # Order is first-seen: seg1, seg2, seg3
        data = [json.loads(line) for line in result]
        assert data[0]["segment_id"] == "seg1"
        assert data[0]["translation"]["text"] == "A"
        assert data[1]["segment_id"] == "seg2"
        assert data[1]["translation"]["text"] == "B"
        assert data[2]["segment_id"] == "seg3"
        assert data[2]["translation"] is None

    def test_corrupt_lines_skipped(self, storage):
        """Malformed lines are silently skipped."""
        meeting_dir = storage._meetings_dir / "m6"
        meeting_dir.mkdir()
        journal = meeting_dir / "journal.jsonl"
        journal.write_text(
            _make_event("seg1", "good")
            + "\n"
            + "not valid json\n"
            + _make_event("seg2", "also good")
            + "\n"
        )
        result = storage.read_journal_raw("m6")
        assert len(result) == 2

    def test_max_lines_limit(self, storage):
        """max_lines truncates to the most recent N after dedup."""
        meeting_dir = storage._meetings_dir / "m7"
        meeting_dir.mkdir()
        journal = meeting_dir / "journal.jsonl"

        lines = [_make_event(f"seg{i}", f"text{i}") for i in range(10)]
        journal.write_text("\n".join(lines) + "\n")

        result = storage.read_journal_raw("m7", max_lines=3)
        assert len(result) == 3
        # Last 3 segments
        assert json.loads(result[0])["segment_id"] == "seg7"
        assert json.loads(result[2])["segment_id"] == "seg9"

    def test_empty_segment_id_skipped(self, storage):
        """Entries without segment_id are skipped."""
        meeting_dir = storage._meetings_dir / "m8"
        meeting_dir.mkdir()
        journal = meeting_dir / "journal.jsonl"
        journal.write_text(
            _make_event("seg1", "valid")
            + "\n"
            + json.dumps({"no_segment_id": True, "text": "bad"})
            + "\n"
        )
        result = storage.read_journal_raw("m8")
        assert len(result) == 1
