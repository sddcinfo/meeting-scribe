"""Tests for ASR language detection filters — multi-language script detection."""

from __future__ import annotations

from meeting_scribe.backends.asr_filters import _detect_language_from_text


class TestLanguageDetection:
    """Heuristic language detection from text content."""

    def test_japanese_hiragana(self):
        assert _detect_language_from_text("こんにちは世界") == "ja"

    def test_japanese_katakana(self):
        assert _detect_language_from_text("テスト") == "ja"

    def test_english(self):
        assert _detect_language_from_text("Hello world, this is a test") == "en"

    def test_korean(self):
        assert _detect_language_from_text("안녕하세요 세계") == "ko"

    def test_chinese_simplified(self):
        # Pure CJK without kana → detected as zh or ja depending on implementation
        result = _detect_language_from_text("你好世界测试")
        assert result in ("zh", "ja")  # CJK without kana is ambiguous

    def test_mixed_ja_en(self):
        # Predominantly Japanese
        result = _detect_language_from_text("今日はgood dayですね")
        assert result == "ja"

    def test_empty_string(self):
        assert _detect_language_from_text("") == "unknown"

    def test_whitespace_only(self):
        assert _detect_language_from_text("   ") == "unknown"

    def test_short_text(self):
        # Very short text may return unknown
        result = _detect_language_from_text("a")
        assert result in ("en", "unknown")

    def test_numbers_only(self):
        result = _detect_language_from_text("12345")
        assert result in ("en", "unknown")

    def test_cyrillic_russian(self):
        result = _detect_language_from_text("Привет мир")
        assert result == "ru"

    def test_arabic(self):
        result = _detect_language_from_text("مرحبا بالعالم")
        assert result == "ar"

    def test_thai(self):
        result = _detect_language_from_text("สวัสดีชาวโลก")
        assert result == "th"
