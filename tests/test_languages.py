"""Tests for multi-language support — language registry, normalization, translation routing."""

from __future__ import annotations

import pytest

from meeting_scribe.languages import (
    DEFAULT_LANGUAGE_PAIR,
    LANGUAGE_REGISTRY,
    get_translation_prompt,
    get_translation_target,
    is_tts_native,
    is_valid_language_pair,
    normalize_language,
    parse_language_pair,
    to_api_response,
)

_TTS_CODES = {"en", "zh", "ja", "ko", "fr", "de", "es", "it", "pt", "ru"}
_ASR_ONLY_CODES = {"nl", "ar", "th", "vi", "id", "ms", "hi", "tr", "pl", "uk"}


class TestLanguageRegistry:
    """Language registry completeness and structure."""

    def test_registry_has_all_supported_languages(self):
        assert set(LANGUAGE_REGISTRY.keys()) == _TTS_CODES | _ASR_ONLY_CODES

    def test_registry_count(self):
        assert len(LANGUAGE_REGISTRY) == 20

    def test_core_languages_present(self):
        for code in ("en", "ja", "zh", "ko", "fr", "de", "es"):
            assert code in LANGUAGE_REGISTRY, f"{code} missing from registry"

    def test_registry_tts_voice_map_is_subset(self):
        # Every TTS voice-map key must be in the registry with tts_native=True.
        from meeting_scribe.backends.tts_voices import all_studio_voices

        voice_codes = set(all_studio_voices().keys())
        assert voice_codes <= set(LANGUAGE_REGISTRY.keys())
        for code in voice_codes:
            assert LANGUAGE_REGISTRY[code].tts_native, f"{code} in voice map but tts_native=False"

    def test_tts_native_flag(self):
        for code in _TTS_CODES:
            assert LANGUAGE_REGISTRY[code].tts_native, f"{code} should be tts_native=True"
        for code in _ASR_ONLY_CODES:
            assert not LANGUAGE_REGISTRY[code].tts_native, f"{code} should be tts_native=False"

    def test_language_has_required_fields(self):
        for code, lang in LANGUAGE_REGISTRY.items():
            assert lang.code == code
            assert len(lang.name) > 0
            assert len(lang.native_name) > 0

    def test_japanese_has_css_class(self):
        ja = LANGUAGE_REGISTRY["ja"]
        assert ja.css_font_class == "ja"

    def test_english_no_css_class(self):
        en = LANGUAGE_REGISTRY["en"]
        assert en.css_font_class == ""


class TestIsTtsNative:
    """Defensive is_tts_native helper."""

    def test_tts_language_returns_true(self):
        assert is_tts_native("en") is True
        assert is_tts_native("ja") is True

    def test_asr_only_language_returns_false(self):
        assert is_tts_native("nl") is False
        assert is_tts_native("ar") is False

    def test_unknown_code_returns_false(self):
        assert is_tts_native("sw") is False
        assert is_tts_native("xx") is False

    def test_empty_string_returns_false(self):
        assert is_tts_native("") is False


class TestNormalizeLanguage:
    """Language code normalization from various input formats."""

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            # TTS-native languages
            ("en", "en"),
            ("eng", "en"),
            ("english", "en"),
            ("English", "en"),
            ("ENGLISH", "en"),
            ("ja", "ja"),
            ("jpn", "ja"),
            ("japanese", "ja"),
            ("zh", "zh"),
            ("zho", "zh"),
            ("chinese", "zh"),
            ("mandarin", "zh"),
            ("ko", "ko"),
            ("korean", "ko"),
            ("fr", "fr"),
            ("french", "fr"),
            ("de", "de"),
            ("german", "de"),
            # ASR-only languages
            ("nl", "nl"),
            ("nld", "nl"),
            ("dut", "nl"),
            ("dutch", "nl"),
            ("ar", "ar"),
            ("ara", "ar"),
            ("arabic", "ar"),
            ("th", "th"),
            ("tha", "th"),
            ("thai", "th"),
            ("vi", "vi"),
            ("vie", "vi"),
            ("vietnamese", "vi"),
            ("id", "id"),
            ("ind", "id"),
            ("indonesian", "id"),
            ("ms", "ms"),
            ("msa", "ms"),
            ("may", "ms"),
            ("malay", "ms"),
            ("hi", "hi"),
            ("hin", "hi"),
            ("hindi", "hi"),
            ("tr", "tr"),
            ("tur", "tr"),
            ("turkish", "tr"),
            ("pl", "pl"),
            ("pol", "pl"),
            ("polish", "pl"),
            ("uk", "uk"),
            ("ukr", "uk"),
            ("ukrainian", "uk"),
        ],
    )
    def test_normalizes_known_languages(self, input_val, expected):
        assert normalize_language(input_val) == expected

    def test_unknown_language_returns_unknown(self):
        assert normalize_language("klingon") == "unknown"

    def test_empty_string_returns_unknown(self):
        assert normalize_language("") == "unknown"

    def test_none_returns_unknown(self):
        assert normalize_language(None) == "unknown"

    def test_whitespace_handled(self):
        assert normalize_language(" en ") == "en"


class TestTranslationTarget:
    """Translation target resolution for arbitrary language pairs."""

    def test_ja_en_pair_ja_detected(self):
        assert get_translation_target("ja", ("ja", "en")) == "en"

    def test_ja_en_pair_en_detected(self):
        assert get_translation_target("en", ("ja", "en")) == "ja"

    def test_zh_en_pair(self):
        assert get_translation_target("zh", ("zh", "en")) == "en"
        assert get_translation_target("en", ("zh", "en")) == "zh"

    def test_ko_ja_pair(self):
        assert get_translation_target("ko", ("ko", "ja")) == "ja"
        assert get_translation_target("ja", ("ko", "ja")) == "ko"

    def test_language_not_in_pair_returns_none(self):
        assert get_translation_target("fr", ("ja", "en")) is None

    def test_unknown_language_returns_none(self):
        assert get_translation_target("unknown", ("ja", "en")) is None


class TestTranslationPrompt:
    """Dynamic translation prompt generation.

    EN↔JA is on a measured few-shot path (see
    ``reports/phase3/prompt_ablation_2026-04-18.md``); every other pair
    returns the generic "professional translator" prompt.
    """

    def test_ja_to_en_prompt(self):
        """JA→EN uses the few-shot meeting prompt (Phase 3 winner)."""
        prompt = get_translation_prompt("ja", "en")
        assert "Japanese" in prompt
        assert "English" in prompt
        assert "translat" in prompt.lower()
        # Few-shot anchor: prompt MUST include at least one of the
        # committed exemplars.  Matches the Phase 3 sweep's winning
        # `fewshot_meeting` variant.
        assert (
            "提案を検討するためのフォローアップ会議を設定しましょう。" in prompt
            or "APIのレスポンスタイムが改善されました。" in prompt
        )

    def test_en_to_ja_prompt(self):
        """EN→JA uses the few-shot meeting prompt (primary scribe direction)."""
        prompt = get_translation_prompt("en", "ja")
        assert "Japanese" in prompt
        assert "English" in prompt
        # Few-shot anchor on the EN→JA side of the corpus.
        assert (
            "Let's schedule a follow-up meeting" in prompt
            or "The API response time has improved." in prompt
        )

    def test_zh_to_en_prompt(self):
        """ZH→EN falls back to the generic translator prompt (no exemplars)."""
        prompt = get_translation_prompt("zh", "en")
        assert "Chinese" in prompt
        assert "English" in prompt
        assert "translator" in prompt.lower()  # generic fallback signature

    def test_ko_to_ja_prompt(self):
        """KO→JA falls back to the generic translator prompt."""
        prompt = get_translation_prompt("ko", "ja")
        assert "Korean" in prompt
        assert "Japanese" in prompt
        assert "translator" in prompt.lower()

    def test_nl_to_en_prompt(self):
        """NL→EN falls back to the generic translator prompt."""
        prompt = get_translation_prompt("nl", "en")
        assert "Dutch" in prompt
        assert "English" in prompt
        assert "translator" in prompt.lower()

    def test_fewshot_only_applies_to_en_ja_pair(self):
        """Non-EN↔JA pairs must NOT accidentally inherit the Japanese
        exemplars (that would leak wrong-language exemplars into other
        pairs' prompts)."""
        for src, tgt in [("zh", "en"), ("en", "zh"), ("fr", "en"), ("de", "en")]:
            prompt = get_translation_prompt(src, tgt)
            assert "提案を検討するためのフォローアップ会議を設定しましょう。" not in prompt
            assert "APIのレスポンスタイムが改善されました。" not in prompt


class TestTranslationPromptPriorContext:
    """Rolling meeting-context window folded into the system prompt.

    Used by the refinement worker (follow-shortly-after path) to anchor
    the LLM on the meeting's ongoing topic/speakers/proper-nouns so
    fragmented utterances don't get hallucinated into free-standing
    sentences.  The live path leaves prior_context=None to keep its
    latency SLO unaffected.
    """

    def test_no_context_is_backwards_compatible(self):
        # prior_context=None must reproduce the exact stateless prompt.
        cold = get_translation_prompt("ja", "en")
        warm = get_translation_prompt("ja", "en", prior_context=None)
        assert cold == warm

    def test_empty_context_is_noop(self):
        # An empty list is also a no-op — keeps the refinement worker's
        # "no prior translations yet" path from injecting an empty
        # context block.
        cold = get_translation_prompt("ja", "en")
        warm = get_translation_prompt("ja", "en", prior_context=[])
        assert cold == warm

    def test_context_appears_in_prompt_order_preserved(self):
        prior = [
            ("Oda-sanから報告がありました。", "Oda-san gave a report."),
            ("TIS社の状況を確認しました。", "We checked the situation at TIS."),
        ]
        prompt = get_translation_prompt("ja", "en", prior_context=prior)
        # Both source strings must be threaded through (oldest first).
        a = prompt.find("Oda-sanから報告がありました。")
        b = prompt.find("TIS社の状況を確認しました。")
        assert 0 <= a < b, "Prior context must preserve oldest→newest order"
        # And both translations must be present.
        assert "Oda-san gave a report." in prompt
        assert "We checked the situation at TIS." in prompt
        # Header line that signals context to the model.
        assert "Earlier in this meeting" in prompt
        # The closing "translate the next" instruction must still be
        # present so the model knows which utterance is the live one.
        assert "translate the next" in prompt

    def test_context_works_without_fewshot_language(self):
        # ZH→EN has no few-shot exemplars.  Context block still threads
        # through so non-EN↔JA pairs benefit too.
        prior = [("测试消息", "Test message")]
        prompt = get_translation_prompt("zh", "en", prior_context=prior)
        assert "测试消息" in prompt
        assert "Test message" in prompt

    def test_context_drops_empty_tuples(self):
        # Real journals sometimes have translations with empty text
        # (retry failed, model returned empty) — filter those so the
        # prompt never contains lone arrows with no counterpart.
        prior = [
            ("valid src", "valid tgt"),
            ("empty_translation", ""),
            ("", "empty source"),
        ]
        prompt = get_translation_prompt("ja", "en", prior_context=prior)
        assert "valid src" in prompt
        assert "valid tgt" in prompt
        assert "empty_translation" not in prompt
        # A stray empty source shouldn't leak either.
        assert "empty source" not in prompt


class TestParseLanguagePair:
    """Language pair string parsing."""

    def test_standard_pair(self):
        assert parse_language_pair("ja,en") == ("ja", "en")

    def test_with_spaces(self):
        assert parse_language_pair("ja , en") == ("ja", "en")

    def test_zh_en(self):
        assert parse_language_pair("zh,en") == ("zh", "en")

    def test_nl_en(self):
        assert parse_language_pair("nl,en") == ("nl", "en")

    def test_invalid_falls_back(self):
        assert parse_language_pair("invalid") == DEFAULT_LANGUAGE_PAIR

    def test_unsupported_code_falls_back(self):
        # Swahili is not in the registry.
        assert parse_language_pair("sw,en") == DEFAULT_LANGUAGE_PAIR

    def test_same_language_twice_falls_back(self):
        assert parse_language_pair("en,en") == DEFAULT_LANGUAGE_PAIR


class TestIsValidLanguagePair:
    """Pair validation used by the meeting-create endpoint."""

    def test_valid_tts_pair(self):
        assert is_valid_language_pair("ja", "en")
        assert is_valid_language_pair("de", "fr")

    def test_valid_mixed_pair(self):
        assert is_valid_language_pair("nl", "en")
        assert is_valid_language_pair("en", "ar")

    def test_valid_asr_only_pair(self):
        assert is_valid_language_pair("nl", "ar")

    def test_rejects_same_code(self):
        assert not is_valid_language_pair("en", "en")

    def test_rejects_unsupported_code(self):
        assert not is_valid_language_pair("sw", "en")
        assert not is_valid_language_pair("en", "sw")

    def test_rejects_empty(self):
        assert not is_valid_language_pair("", "en")


class TestAPIResponse:
    """Language API response format."""

    def test_api_response_has_languages(self):
        resp = to_api_response()
        assert "languages" in resp
        assert len(resp["languages"]) == 20

    def test_api_response_has_default_pair(self):
        resp = to_api_response()
        assert "default_pair" in resp
        a, b = resp["default_pair"]
        assert is_valid_language_pair(a, b)

    def test_api_response_has_no_popular_pairs(self):
        # Removed — the UI now composes pairs from two independent dropdowns.
        resp = to_api_response()
        assert "popular_pairs" not in resp

    def test_language_entry_format(self):
        resp = to_api_response()
        lang = resp["languages"][0]
        assert "code" in lang
        assert "name" in lang
        assert "native_name" in lang

    def test_api_response_includes_tts_supported(self):
        resp = to_api_response()
        for lang in resp["languages"]:
            assert "tts_supported" in lang
            assert isinstance(lang["tts_supported"], bool)

    def test_api_response_tts_supported_matches_registry(self):
        resp = to_api_response()
        for lang in resp["languages"]:
            assert lang["tts_supported"] == LANGUAGE_REGISTRY[lang["code"]].tts_native


# ───────── Monolingual support (single-language meetings) ─────────
# ``language_pair`` now holds 1 or 2 codes — length 1 means the meeting
# is monolingual and every translation path must short-circuit. These
# tests lock the shape contract that the rest of the codebase (model
# validator, API, translation queue, slide job) depends on.


class TestIsValidLanguages:
    """``is_valid_languages`` — authoritative shape check used by the
    Pydantic validator on ``MeetingMeta.language_pair``."""

    def test_length_one_valid(self):
        from meeting_scribe.languages import is_valid_languages

        assert is_valid_languages(["en"]) is True
        assert is_valid_languages(["ja"]) is True

    def test_length_two_valid(self):
        from meeting_scribe.languages import is_valid_languages

        assert is_valid_languages(["ja", "en"]) is True
        assert is_valid_languages(["zh", "ko"]) is True

    def test_length_zero_invalid(self):
        from meeting_scribe.languages import is_valid_languages

        assert is_valid_languages([]) is False

    def test_length_three_invalid(self):
        from meeting_scribe.languages import is_valid_languages

        assert is_valid_languages(["en", "ja", "ko"]) is False

    def test_duplicate_pair_invalid(self):
        from meeting_scribe.languages import is_valid_languages

        assert is_valid_languages(["en", "en"]) is False

    def test_unknown_code_invalid_length_one(self):
        from meeting_scribe.languages import is_valid_languages

        assert is_valid_languages(["sw"]) is False

    def test_unknown_code_invalid_length_two(self):
        from meeting_scribe.languages import is_valid_languages

        assert is_valid_languages(["en", "sw"]) is False

    def test_non_list_invalid(self):
        from meeting_scribe.languages import is_valid_languages

        assert is_valid_languages("en,ja") is False  # string, not list
        assert is_valid_languages(None) is False  # type: ignore[arg-type]


class TestParseLanguagesStrict:
    """Strict parser used for untrusted input (API request bodies).
    Returns ``None`` on invalid input so the caller can surface a 400.
    """

    def test_single_code(self):
        from meeting_scribe.languages import parse_languages_strict

        assert parse_languages_strict("en") == ["en"]
        assert parse_languages_strict("ja") == ["ja"]

    def test_pair(self):
        from meeting_scribe.languages import parse_languages_strict

        assert parse_languages_strict("ja,en") == ["ja", "en"]

    def test_pair_with_whitespace(self):
        from meeting_scribe.languages import parse_languages_strict

        assert parse_languages_strict(" ja , en ") == ["ja", "en"]

    def test_duplicate_pair_returns_none(self):
        from meeting_scribe.languages import parse_languages_strict

        assert parse_languages_strict("en,en") is None

    def test_unknown_code_returns_none(self):
        from meeting_scribe.languages import parse_languages_strict

        assert parse_languages_strict("xx") is None

    def test_empty_returns_none(self):
        from meeting_scribe.languages import parse_languages_strict

        assert parse_languages_strict("") is None

    def test_three_parts_returns_none(self):
        from meeting_scribe.languages import parse_languages_strict

        assert parse_languages_strict("en,ja,ko") is None

    def test_non_string_returns_none(self):
        from meeting_scribe.languages import parse_languages_strict

        assert parse_languages_strict(123) is None  # type: ignore[arg-type]
        assert parse_languages_strict(None) is None  # type: ignore[arg-type]


class TestParseLanguagesLenient:
    """Lenient config-only parser. Invalid input falls back to the
    default pair but emits a loud WARNING log so bad deployments are
    discoverable."""

    def test_valid_single_code(self):
        from meeting_scribe.languages import parse_languages

        assert parse_languages("en") == ["en"]

    def test_valid_pair(self):
        from meeting_scribe.languages import parse_languages

        assert parse_languages("ja,en") == ["ja", "en"]

    def test_invalid_falls_back_to_default(self, caplog):
        import logging

        from meeting_scribe.languages import DEFAULT_LANGUAGE_PAIR, parse_languages

        with caplog.at_level(logging.WARNING, logger="meeting_scribe.languages"):
            result = parse_languages("not_a_pair")
        assert result == list(DEFAULT_LANGUAGE_PAIR)
        # Loud-warning contract: the bad value and the fallback must be
        # named in the log so the operator can find the typo in prod.
        assert any("not_a_pair" in rec.message for rec in caplog.records), (
            "Lenient fallback must name the rejected value in the warning"
        )

    def test_duplicate_pair_falls_back(self, caplog):
        import logging

        from meeting_scribe.languages import DEFAULT_LANGUAGE_PAIR, parse_languages

        with caplog.at_level(logging.WARNING, logger="meeting_scribe.languages"):
            result = parse_languages("en,en")
        assert result == list(DEFAULT_LANGUAGE_PAIR)


class TestGetTranslationTargetMonolingual:
    """Monolingual meetings must never produce a translation target."""

    def test_length_one_returns_none_for_any_lang(self):
        # Regardless of what language was detected, a monolingual
        # meeting has nothing to translate to.
        assert get_translation_target("en", ["en"]) is None
        assert get_translation_target("ja", ["en"]) is None
        assert get_translation_target("unknown", ["en"]) is None

    def test_length_two_list_behaves_like_tuple(self):
        # The function now accepts list or tuple uniformly.
        assert get_translation_target("ja", ["ja", "en"]) == "en"
        assert get_translation_target("en", ["ja", "en"]) == "ja"
