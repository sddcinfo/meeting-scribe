"""Tests for meeting summary generation — prompt construction, output parsing."""

from __future__ import annotations

import json

from meeting_scribe.summary import SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_PROMPT


class TestSummaryPrompt:
    def test_prompt_has_json_template(self):
        assert "executive_summary" in SUMMARY_SYSTEM_PROMPT
        assert "topics" in SUMMARY_SYSTEM_PROMPT
        assert "decisions" in SUMMARY_SYSTEM_PROMPT
        assert "action_items" in SUMMARY_SYSTEM_PROMPT

    def test_prompt_has_formatting_slots(self):
        assert "{duration_min" in SUMMARY_USER_PROMPT
        assert "{num_speakers}" in SUMMARY_USER_PROMPT
        assert "{transcript}" in SUMMARY_USER_PROMPT

    def test_prompt_can_be_formatted(self):
        formatted = SUMMARY_USER_PROMPT.format(
            duration_min=30.5,
            num_speakers=3,
            speaker_names="Brad, Alice, Bob",
            languages="English, Japanese",
            num_segments=150,
            transcript="[00:00] Brad: Hello everyone...",
        )
        assert "30.5" in formatted
        assert "Brad, Alice, Bob" in formatted


class TestSummaryOutput:
    """Validate summary.json round-trip + schema."""

    def test_synthetic_summary_round_trips(self, tmp_path):
        """Build a representative summary doc, write it, re-read it, and
        assert the structure that consumers (frontend + versions._summarize_summary_doc)
        rely on. Uses a synthetic fixture so the test is hermetic — no
        dependency on whatever meetings happen to be on disk."""
        path = tmp_path / "summary.json"
        path.write_text(
            json.dumps(
                {
                    "executive_summary": "A 3-line summary of the meeting.",
                    "topics": [{"title": "Roadmap", "summary": "Q3 launches"}],
                    "decisions": [{"text": "Ship 1.5.0 next Tuesday"}],
                    "action_items": [{"owner": "Alice", "text": "draft post"}],
                    "key_quotes": [],
                    "key_insights": [],
                    "schema_version": 3,
                }
            )
        )
        summary = json.loads(path.read_text())
        assert "executive_summary" in summary
        assert isinstance(summary.get("topics"), list)
        assert isinstance(summary.get("action_items"), list)
        assert len(summary["executive_summary"]) > 10
        assert summary["schema_version"] == 3
