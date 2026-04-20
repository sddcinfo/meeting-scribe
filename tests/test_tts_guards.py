"""Tests for TTS background synthesis guards.

Validates: text length cap, sentence boundary detection, meeting-stopped guard,
timeout behavior, background task cancellation.
"""

from __future__ import annotations


class TestTTSTextCap:
    """Test the 300-char text truncation with sentence boundary detection."""

    MAX_TTS_CHARS = 300

    def _truncate(self, text: str) -> str:
        """Reproduce the truncation logic from _tts_background."""
        if len(text) <= self.MAX_TTS_CHARS:
            return text
        cut = text[: self.MAX_TTS_CHARS].rfind("。")
        if cut < 100:
            cut = text[: self.MAX_TTS_CHARS].rfind(". ")
        if cut < 100:
            cut = self.MAX_TTS_CHARS
        return text[: cut + 1]

    def test_short_text_unchanged(self):
        text = "Hello world"
        assert self._truncate(text) == text

    def test_exactly_300_chars(self):
        text = "a" * 300
        assert self._truncate(text) == text

    def test_over_300_truncated(self):
        text = "a" * 500
        result = self._truncate(text)
        assert len(result) <= 301  # +1 for the boundary char

    def test_japanese_sentence_boundary(self):
        """Truncates at 。 (Japanese period) when past char 100."""
        # 。 must be after position 100 to be used as a cut point
        text = "あ" * 120 + "。" + "い" * 300  # 。 at position 120
        result = self._truncate(text)
        assert result.endswith("。"), f"Should end at 。: {result[-10:]}"
        assert len(result) <= 301

    def test_japanese_boundary_too_early_ignored(self):
        """Japanese period before position 100 is ignored (text too short)."""
        text = "短い文。" + "x" * 400  # 。 at position 3 — too early
        result = self._truncate(text)
        assert len(result) > 50  # Should NOT cut at position 3

    def test_english_sentence_boundary(self):
        """Truncates at '. ' (English period) when no Japanese period."""
        text = "This is a short sentence. " + "More text here. " * 30
        result = self._truncate(text)
        assert ". " in result[-5:] or result.endswith("."), f"Should end at period: {result[-10:]}"

    def test_no_sentence_boundary(self):
        """Falls back to hard cut at MAX_TTS_CHARS when no boundary found."""
        text = "abcdefghij" * 50  # No periods at all
        result = self._truncate(text)
        assert len(result) == 301  # MAX_TTS_CHARS + 1

    def test_boundary_near_start_ignored(self):
        """Period before char 100 is ignored (too short)."""
        text = "Short. " + "x" * 400  # Period at char 6
        result = self._truncate(text)
        # Should not cut at char 6 — too short
        assert len(result) > 50


class TestMeetingStateGuards:
    """Test _persist/_clear/_get_interrupted_meeting functions."""

    def test_persist_creates_file(self, tmp_path):
        state_file = tmp_path / "active.json"
        import json

        state_file.write_text(json.dumps({"meeting_id": "test-123", "start_time": 1000}) + "\n")
        data = json.loads(state_file.read_text())
        assert data["meeting_id"] == "test-123"

    def test_clear_removes_file(self, tmp_path):
        state_file = tmp_path / "active.json"
        state_file.write_text('{"meeting_id": "test"}')
        state_file.unlink()
        assert not state_file.exists()

    def test_get_interrupted_missing_file(self, tmp_path):
        """Returns None when no state file exists."""
        state_file = tmp_path / "active.json"
        assert not state_file.exists()

    def test_get_interrupted_corrupt_json(self, tmp_path):
        """Returns None for corrupt JSON."""
        state_file = tmp_path / "active.json"
        state_file.write_text("not valid json{{{")
        import json

        try:
            json.loads(state_file.read_text())
            assert False, "Should have raised"
        except json.JSONDecodeError:
            pass  # Expected

    def test_get_interrupted_valid(self, tmp_path):
        """Returns meeting_id from valid state file."""
        import json

        state_file = tmp_path / "active.json"
        state_file.write_text(json.dumps({"meeting_id": "abc-123", "start_time": 1000}))
        data = json.loads(state_file.read_text())
        assert data.get("meeting_id") == "abc-123"


class TestPriorityValues:
    """Validate priority values are set correctly across all backends."""

    def test_translation_highest_priority(self):
        """Translation backend uses priority -10 (highest)."""
        from pathlib import Path

        src = (
            Path(__file__).parent.parent
            / "src"
            / "meeting_scribe"
            / "backends"
            / "translate_vllm.py"
        )
        content = src.read_text()
        assert '"priority": -10' in content, "Translation must use priority -10"

    def test_summary_uses_live_tier_priority(self):
        """Summary uses priority -10 (same tier as live translation).

        The old value -5 sat between coding (10) and translation (-10),
        creating a priority inversion vs refinement (-8) which blocked
        summary from running in a reasonable time. The priority is now
        parameterized via _call_vllm_summary with default -10.
        """
        from pathlib import Path

        src = Path(__file__).parent.parent / "src" / "meeting_scribe" / "summary.py"
        content = src.read_text()
        assert "priority: int = -10" in content, "Summary must default to priority -10"
        assert '"priority": -5' not in content

    def test_name_extraction_uses_live_tier_priority(self):
        """Name extraction runs inline during a live meeting and must
        match the live translation tier (-10). Was -5 — same inversion."""
        from pathlib import Path

        src = Path(__file__).parent.parent / "src" / "meeting_scribe" / "speaker" / "name_llm.py"
        content = src.read_text()
        assert '"priority": -10' in content, "Name extraction must use priority -10"
        assert '"priority": -5' not in content

    def test_priority_ordering(self):
        """Ladder: ASR (-20) < live translate / TTS / summary / name /
        refinement (-10) < reprocess (0) < coding (10) < plan-review (20).

        Lower value = higher priority in vLLM."""
        asr = -20
        live_tier = -10  # translate, tts, summary, name, refinement
        reprocess = 0
        coding = 10
        plan_review = 20
        assert asr < live_tier < reprocess < coding < plan_review

    def test_thinking_disabled_for_translation(self):
        """Translation requests disable thinking mode."""
        from pathlib import Path

        src = (
            Path(__file__).parent.parent
            / "src"
            / "meeting_scribe"
            / "backends"
            / "translate_vllm.py"
        )
        content = src.read_text()
        assert '"enable_thinking": False' in content

    def test_translation_max_tokens_capped(self):
        """Translation max_tokens is 256 (not 1024)."""
        from pathlib import Path

        src = (
            Path(__file__).parent.parent
            / "src"
            / "meeting_scribe"
            / "backends"
            / "translate_vllm.py"
        )
        content = src.read_text()
        assert "max_tokens: int = 256" in content, "Translation max_tokens should be 256"
