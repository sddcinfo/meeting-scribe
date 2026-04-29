"""Unit tests for server-side logic — no running server needed.

Tests pure functions extracted from server.py: name extraction,
hallucination filtering, language normalization, JS syntax checks.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest


class TestNameExtraction:
    """Self-introduction pattern extraction."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from meeting_scribe.speaker.name_extraction import extract_name as _extract_name_from_text

        self.extract = _extract_name_from_text

    def test_english_my_name_is(self):
        assert self.extract("Hi, my name is Brad") == "Brad"
        assert self.extract("Hello, my name is Sarah") == "Sarah"

    def test_english_call_me(self):
        assert self.extract("Call me John") == "John"

    def test_english_false_positives_rejected(self):
        assert self.extract("I'm ready to listen") is None
        assert self.extract("I'm going to the store") is None
        assert self.extract("My name is the best") is None

    def test_japanese_desu(self):
        assert self.extract("田中です") == "Tanaka"
        assert self.extract("私は田中です") == "Tanaka"

    def test_japanese_false_positives_rejected(self):
        assert self.extract("午前中に配達の予定をしていたんですけれども") is None
        assert self.extract("が難しい状況でございます") is None
        assert self.extract("よろしいですか") is None

    def test_no_intro(self):
        assert self.extract("The budget review is next week") is None
        assert self.extract("予算について話しましょう") is None


class TestHallucinationFilter:
    """ASR hallucination detection."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from meeting_scribe.backends.asr_filters import _is_hallucination

        self.is_hallucination = _is_hallucination

    def test_known_phrases(self):
        assert self.is_hallucination("Thank you for watching")
        assert self.is_hallucination("ご視聴ありがとうございました")

    def test_repeated_words(self):
        assert self.is_hallucination("water water water")

    def test_repeated_characters(self):
        assert self.is_hallucination("aaaaaaaaaaa")
        assert self.is_hallucination("狼狼狼狼狼狼狼狼")

    def test_long_single_word(self):
        assert self.is_hallucination(
            "aitititititititititititititititititititititititititititititititititi"
        )

    def test_normal_text_passes(self):
        assert not self.is_hallucination("Hello everyone")
        assert not self.is_hallucination("予算について話しましょう")
        assert not self.is_hallucination("The budget review is next week")
        assert not self.is_hallucination("こんにちは")


class TestJavaScriptSyntax:
    """Basic JS syntax validation."""

    def test_no_orphaned_else(self):
        js = (Path(__file__).parent.parent / "static" / "js" / "scribe-app.js").read_text()
        lines = js.split("\n")
        brace_depth = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            brace_depth += stripped.count("{") - stripped.count("}")
            if stripped == "} else {" and brace_depth < 0:
                raise AssertionError(f"Possible orphaned else at line {i + 1}")

    def test_no_duplicate_top_level_consts(self):
        js = (Path(__file__).parent.parent / "static" / "js" / "scribe-app.js").read_text()
        lines = js.split("\n")
        top_level = []
        for line in lines:
            if not line or line[0] in (" ", "\t"):
                continue
            stripped = line.strip()
            if stripped.startswith("const ") or stripped.startswith("let "):
                name = stripped.split()[1].rstrip(";=,")
                top_level.append(name)
        dupes = {k: v for k, v in Counter(top_level).items() if v > 1}
        assert not dupes, f"Duplicate top-level declarations: {dupes}"
