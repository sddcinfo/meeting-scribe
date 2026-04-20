"""Tests for the lingua-backed ASR language post-corrector.

The corrector's contract: never make things worse than the ASR alone.
That means: short text passes through, unknown languages pass through,
weak signals pass through, single-language pairs pass through. Only
confident disagreements between lingua and the ASR tag flip the label.
"""
from __future__ import annotations

import pytest

# Skip the whole module if lingua isn't installed — the post-corrector
# is opt-in and graceful-fallback by design.
pytest.importorskip("lingua")

from meeting_scribe.language_correction import (
    _ISO_TO_LINGUA,
    correct_segment_language,
    correction_stats,
)


@pytest.fixture(autouse=True)
def _reset_stats():
    """Each test starts with a clean stats counter so assertions are scoped."""
    correction_stats.reset()
    yield


class TestCorrectSegmentLanguage:
    def test_short_text_passes_through(self):
        # ``Hi``: 2 chars — below MIN_TEXT_CHARS, lingua never queried
        out = correct_segment_language("Hi", "en", ("nl", "en"))
        assert out == "en"
        snap = correction_stats.snapshot()
        assert snap["skipped_short"] == 1
        assert snap["overridden"] == 0

    def test_single_language_pair_skips(self):
        # Lingua needs ≥ 2 distinct codes to disambiguate
        out = correct_segment_language("Iets in het Nederlands.", "en", ("en",))
        assert out == "en"
        assert correction_stats.snapshot()["skipped_no_detector"] == 1

    def test_dutch_speech_corrected_from_en_to_nl(self):
        # Real-world failure case from meeting e5b376b2 — Qwen3-ASR
        # tagged Dutch as English. Lingua correctly identifies it.
        text = (
            "Het is een vergadering om de strategie te bespreken "
            "en we hebben veel besluiten genomen."
        )
        out = correct_segment_language(text, "en", ("nl", "en"))
        assert out == "nl"
        snap = correction_stats.snapshot()
        assert snap["overridden"] == 1
        assert "en→nl" in snap["override_pairs"]

    def test_english_speech_left_alone(self):
        text = "This is a regular english sentence about quarterly results."
        out = correct_segment_language(text, "en", ("nl", "en"))
        assert out == "en"
        snap = correction_stats.snapshot()
        assert snap["kept"] == 1
        assert snap["overridden"] == 0

    def test_japanese_left_alone_when_correct(self):
        out = correct_segment_language("これは日本語の文章です。", "ja", ("ja", "en"))
        assert out == "ja"

    def test_german_speech_corrected_from_en_to_de(self):
        # Same Germanic→English bias affects German. Confirms the fix
        # generalizes beyond Dutch.
        text = (
            "Das ist eine sehr lange deutsche Aussage über die "
            "Geschäftsstrategie für das nächste Quartal."
        )
        out = correct_segment_language(text, "en", ("de", "en"))
        assert out == "de"

    def test_unknown_meeting_pair_passes_through(self):
        out = correct_segment_language("test", "en", ())
        assert out == "en"

    def test_iso_map_includes_meeting_supported_languages(self):
        # Sanity: every code we hand to the corrector should map to a
        # lingua Language enum member (otherwise lingua silently can't
        # disambiguate that pair). Catches typos in _ISO_TO_LINGUA.
        from lingua import Language

        for iso, name in _ISO_TO_LINGUA.items():
            assert hasattr(Language, name), f"{iso!r} → {name!r} not in Language enum"

    def test_latency_recorded(self):
        # Latency should be tracked on every call — visible to the
        # /api/status endpoint for real-time observability.
        text = "Een Nederlands verhaal over wat we vandaag hebben besproken."
        correct_segment_language(text, "en", ("nl", "en"))
        snap = correction_stats.snapshot()
        assert snap["calls"] == 1
        assert snap["mean_latency_ms"] > 0
        assert snap["max_latency_ms"] > 0
