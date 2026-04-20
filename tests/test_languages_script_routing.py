"""Script-based language detection + column routing.

Two separate routing layers share the same invariant — text in a given
script must NEVER land in a column for a different script:

1. Server side: `_detect_language_from_text` in asr_filters.py is used
   as a fallback when Qwen3-ASR doesn't tag the output with a language.

2. Client side: `_routeLangByScript` in static/js/scribe-app.js routes
   an ASR segment to compact-col-a or compact-col-b before rendering.
   This test file mirrors the client expectations as a regression spec;
   the actual JS logic is kept in sync by hand.
"""
from __future__ import annotations

import pytest

from meeting_scribe.backends.asr_filters import _detect_language_from_text


class TestDetectJapanese:
    @pytest.mark.parametrize("text", [
        "こんにちは",                          # hiragana only
        "コンピュータ",                         # katakana only
        "今日は会議です",                        # mixed kanji + kana
        "えっと、ちょっと聞きたいんですが",          # filler + kanji
    ])
    def test_japanese_routed_to_ja(self, text):
        assert _detect_language_from_text(text) == "ja"


class TestDetectChinese:
    @pytest.mark.parametrize("text", [
        "你好世界",
        "我们明天开会",
        "请问您的意见是什么",
    ])
    def test_han_without_kana_routed_to_zh(self, text):
        assert _detect_language_from_text(text) == "zh"


class TestDetectKorean:
    @pytest.mark.parametrize("text", [
        "안녕하세요",
        "오늘 회의가 있습니다",
    ])
    def test_hangul_routed_to_ko(self, text):
        assert _detect_language_from_text(text) == "ko"


class TestDetectLatin:
    @pytest.mark.parametrize("text", [
        "Hello world, this is a meeting.",
        "The quick brown fox jumps over the lazy dog.",
        "OK, let's move on to the next topic.",
    ])
    def test_latin_routed_to_en(self, text):
        assert _detect_language_from_text(text) == "en"


class TestDetectOtherScripts:
    def test_cyrillic_routed_to_ru(self):
        assert _detect_language_from_text("Привет, как дела?") == "ru"

    def test_arabic_routed_to_ar(self):
        assert _detect_language_from_text("مرحبا بالعالم") == "ar"

    def test_thai_routed_to_th(self):
        assert _detect_language_from_text("สวัสดีครับ") == "th"

    def test_devanagari_routed_to_hi(self):
        assert _detect_language_from_text("नमस्ते दुनिया") == "hi"


class TestDetectEdgeCases:
    def test_empty_string_is_unknown(self):
        assert _detect_language_from_text("") == "unknown"

    def test_whitespace_only_is_unknown(self):
        assert _detect_language_from_text("   ") == "unknown"

    def test_single_char_is_unknown(self):
        assert _detect_language_from_text("a") == "unknown"

    def test_numbers_only_defaults_to_en(self):
        # No non-Latin script → treated as Latin → "en"
        assert _detect_language_from_text("12345") == "en"

    def test_mixed_ja_and_en_with_kana_wins(self):
        # Real-world: "about the 会議 tomorrow" — short EN + kanji — but
        # pure kanji without kana trips the CJK-ambiguous branch and
        # may land as zh; with kana present it's definitively ja.
        assert _detect_language_from_text("明日の会議について話しましょう") == "ja"


class TestTranslationTargetRouting:
    """The companion piece: given a detected language, translation
    routes to the OTHER side of the language pair."""

    def test_ja_source_in_ja_en_pair(self):
        from meeting_scribe.languages import get_translation_target
        assert get_translation_target("ja", ("ja", "en")) == "en"

    def test_en_source_in_ja_en_pair(self):
        from meeting_scribe.languages import get_translation_target
        assert get_translation_target("en", ("ja", "en")) == "ja"

    def test_zh_source_in_zh_en_pair(self):
        from meeting_scribe.languages import get_translation_target
        assert get_translation_target("zh", ("zh", "en")) == "en"

    def test_ko_source_in_ko_en_pair(self):
        from meeting_scribe.languages import get_translation_target
        assert get_translation_target("ko", ("ko", "en")) == "en"


class TestClientSideRoutingSpec:
    """Spec tests for the client-side `_routeLangByScript` behaviour.

    These don't run JS — they codify the routing rules that the server
    and client MUST agree on. If any of these expectations change, the
    JS in scribe-app.js must be updated to match (and vice versa).

    Rules:
    - Hangul present + ko in pair → ko
    - Japanese kana present + ja in pair → ja
    - Han only + ja/zh/ko in pair → first-match in that order
    - No CJK at all → non-CJK language in pair (or ASR hint if valid)
    """

    def test_hangul_routes_to_ko_in_pair(self):
        # Mirrors: hasHangul && pair includes 'ko' → 'ko'
        hangul = "안녕"
        pair = ("ko", "en")
        assert self._route(hangul, None, pair) == "ko"

    def test_kana_routes_to_ja_in_pair(self):
        # Mirrors: hasJaKana && pair includes 'ja' → 'ja'
        text = "こんにちは"
        assert self._route(text, None, ("ja", "en")) == "ja"

    def test_han_only_prefers_ja_then_zh(self):
        # Ambiguous Han — prefer ja if in pair, else zh, else ko
        text = "会議"
        assert self._route(text, None, ("ja", "en")) == "ja"
        assert self._route(text, None, ("zh", "en")) == "zh"
        assert self._route(text, None, ("ko", "en")) == "ko"

    def test_latin_with_cjk_pair_goes_to_en(self):
        # No CJK in text → route to the non-CJK side of the pair
        assert self._route("Hello there", None, ("ja", "en")) == "en"

    def test_asr_hint_honored_if_valid_and_no_script_conflict(self):
        assert self._route("Hola", "es", ("es", "en")) == "es"

    def test_asr_hint_overridden_when_script_says_otherwise(self):
        # ASR mislabels Japanese as English → script check wins
        assert self._route("会議です", "en", ("ja", "en")) == "ja"

    @staticmethod
    def _route(text: str, asr_lang: str | None, pair: tuple[str, str]) -> str:
        """Python port of _routeLangByScript from scribe-app.js.
        Kept here as the source of truth for the shared invariant."""
        import re
        has_ja_kana = bool(re.search(r"[\u3040-\u309F\u30A0-\u30FF]", text))
        has_cjk = bool(re.search(r"[\u3400-\u9FFF\uF900-\uFAFF]", text))
        has_hangul = bool(re.search(r"[\uAC00-\uD7AF\u1100-\u11FF]", text))
        if has_hangul and "ko" in pair:
            return "ko"
        if has_ja_kana and "ja" in pair:
            return "ja"
        if has_cjk:
            for c in ("ja", "zh", "ko"):
                if c in pair:
                    return c
        cjk = {"ja", "zh", "ko"}
        if asr_lang and asr_lang in pair and asr_lang not in cjk:
            return asr_lang
        non_cjk = next((l for l in pair if l not in cjk), None)
        return non_cjk or pair[0]
