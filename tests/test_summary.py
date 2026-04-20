"""Tests for meeting summary generation — prompt construction, output parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    """Validate summary.json structure from real meetings."""

    def test_real_meeting_summary_structure(self):
        """Check summary structure from the f38d5807 meeting if available."""
        summary_path = (
            Path(__file__).parent.parent
            / "meetings"
            / "f38d5807-bbdf-4c5c-96fb-cb8267e55ed0"
            / "summary.json"
        )
        if not summary_path.exists():
            pytest.skip("Meeting f38d5807 not available")

        summary = json.loads(summary_path.read_text())
        assert "executive_summary" in summary
        assert isinstance(summary.get("topics"), list)
        assert isinstance(summary.get("action_items"), list)

    def test_real_meeting_97cd_summary(self):
        """Check summary from 97cd1d18 meeting."""
        summary_path = (
            Path(__file__).parent.parent
            / "meetings"
            / "97cd1d18-89f9-4d2f-b3d2-3b88a087bb0d"
            / "summary.json"
        )
        if not summary_path.exists():
            pytest.skip("Meeting 97cd1d18 not available")

        summary = json.loads(summary_path.read_text())
        assert "executive_summary" in summary
        assert len(summary["executive_summary"]) > 10
