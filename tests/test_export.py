"""Tests for meeting export — markdown, text, ZIP."""

from __future__ import annotations

import json
import zipfile
from io import BytesIO

from meeting_scribe.export import (
    _format_timestamp,
    _load_events_with_corrections,
    meeting_to_markdown,
    meeting_to_zip,
    transcript_to_text,
)


class TestTimestampFormatting:
    def test_zero(self):
        assert _format_timestamp(0) == "00:00"

    def test_seconds(self):
        assert _format_timestamp(45000) == "00:45"

    def test_minutes(self):
        assert _format_timestamp(125000) == "02:05"

    def test_hours(self):
        assert _format_timestamp(3723000) == "1:02:03"


class TestEventLoading:
    def test_loads_final_events(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        events = [
            {
                "segment_id": "s1",
                "text": "Hello",
                "is_final": True,
                "language": "en",
                "start_ms": 0,
                "end_ms": 1000,
            },
            {
                "segment_id": "s2",
                "text": "World",
                "is_final": True,
                "language": "en",
                "start_ms": 1000,
                "end_ms": 2000,
            },
            {"segment_id": "s3", "text": "partial", "is_final": False},
        ]
        journal.write_text("\n".join(json.dumps(e) for e in events))
        loaded = _load_events_with_corrections(journal)
        assert len(loaded) == 2  # Only finals

    def test_applies_speaker_corrections(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        events = [
            {
                "segment_id": "s1",
                "text": "Hello",
                "is_final": True,
                "language": "en",
                "start_ms": 0,
                "end_ms": 1000,
                "speakers": [{"identity": "Speaker 0"}],
            },
            {"type": "speaker_correction", "segment_id": "s1", "speaker_name": "Brad"},
        ]
        journal.write_text("\n".join(json.dumps(e) for e in events))
        loaded = _load_events_with_corrections(journal)
        assert loaded[0]["speakers"][0]["identity"] == "Brad"

    def test_skips_malformed_lines(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        journal.write_text(
            '{"segment_id":"s1","text":"OK","is_final":true,"language":"en","start_ms":0,"end_ms":1000}\nNOT_JSON\n'
        )
        loaded = _load_events_with_corrections(journal)
        assert len(loaded) == 1

    def test_empty_journal(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        journal.write_text("")
        loaded = _load_events_with_corrections(journal)
        assert loaded == []


class TestMarkdownExport:
    def _make_meeting_dir(self, tmp_path):
        d = tmp_path / "meeting"
        d.mkdir()
        (d / "journal.jsonl").write_text(
            json.dumps(
                {
                    "segment_id": "s1",
                    "text": "Hello world",
                    "is_final": True,
                    "language": "en",
                    "start_ms": 0,
                    "end_ms": 2000,
                    "speakers": [{"identity": "Brad"}],
                }
            )
        )
        (d / "summary.json").write_text(
            json.dumps(
                {
                    "executive_summary": "Test meeting about testing.",
                    "topics": [{"title": "Testing", "description": "Discussed tests"}],
                    "decisions": ["Write more tests"],
                    "action_items": [{"task": "Add tests", "assignee": "Brad"}],
                }
            )
        )
        return d

    def test_markdown_contains_summary(self, tmp_path):
        d = self._make_meeting_dir(tmp_path)
        md = meeting_to_markdown(d)
        assert "Test meeting about testing" in md

    def test_markdown_contains_transcript(self, tmp_path):
        d = self._make_meeting_dir(tmp_path)
        md = meeting_to_markdown(d)
        assert "Hello world" in md

    def test_markdown_contains_speaker(self, tmp_path):
        d = self._make_meeting_dir(tmp_path)
        md = meeting_to_markdown(d)
        assert "Brad" in md


class TestTextExport:
    def test_text_export_all_languages(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        journal.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "segment_id": "s1",
                            "text": "Hello",
                            "is_final": True,
                            "language": "en",
                            "start_ms": 0,
                            "end_ms": 1000,
                        }
                    ),
                    json.dumps(
                        {
                            "segment_id": "s2",
                            "text": "こんにちは",
                            "is_final": True,
                            "language": "ja",
                            "start_ms": 1000,
                            "end_ms": 2000,
                        }
                    ),
                ]
            )
        )
        txt = transcript_to_text(journal)
        assert "Hello" in txt
        assert "こんにちは" in txt

    def test_text_export_filtered(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        journal.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "segment_id": "s1",
                            "text": "Hello",
                            "is_final": True,
                            "language": "en",
                            "start_ms": 0,
                            "end_ms": 1000,
                        }
                    ),
                    json.dumps(
                        {
                            "segment_id": "s2",
                            "text": "こんにちは",
                            "is_final": True,
                            "language": "ja",
                            "start_ms": 1000,
                            "end_ms": 2000,
                        }
                    ),
                ]
            )
        )
        txt = transcript_to_text(journal, lang="en")
        assert "Hello" in txt
        assert "こんにちは" not in txt


class TestZipExport:
    def test_zip_contains_journal(self, tmp_path):
        d = tmp_path / "meeting"
        d.mkdir()
        (d / "journal.jsonl").write_text('{"segment_id":"s1","text":"test","is_final":true}\n')
        (d / "meta.json").write_text('{"meeting_id":"test-123"}')
        data = meeting_to_zip(d)
        assert isinstance(data, bytes)
        buf = BytesIO(data)
        with zipfile.ZipFile(buf, "r") as zf:
            names = zf.namelist()
            assert any("journal" in n for n in names)
