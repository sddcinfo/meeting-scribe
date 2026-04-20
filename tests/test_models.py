"""Tests for data models — TranscriptEvent, TranslationState, MeetingMeta."""

from __future__ import annotations

from meeting_scribe.models import (
    MeetingMeta,
    MeetingState,
    TranscriptEvent,
    TranslationStatus,
)


class TestTranscriptEventTranslation:
    """with_translation() multi-language support."""

    def _make_event(self, language: str = "ja") -> TranscriptEvent:
        return TranscriptEvent(
            segment_id="test-001",
            text="テスト",
            language=language,
            is_final=True,
            start_ms=0,
            end_ms=1000,
        )

    def test_with_translation_explicit_target(self):
        e = self._make_event("ja")
        t = e.with_translation(TranslationStatus.DONE, text="Test", target_language="en")
        assert t.translation.target_language == "en"
        assert t.translation.text == "Test"
        assert t.translation.status == TranslationStatus.DONE

    def test_with_translation_zh_to_en(self):
        e = self._make_event("zh")
        t = e.with_translation(TranslationStatus.DONE, text="Test", target_language="en")
        assert t.translation.target_language == "en"

    def test_with_translation_ko_to_ja(self):
        e = self._make_event("ko")
        t = e.with_translation(TranslationStatus.DONE, text="テスト", target_language="ja")
        assert t.translation.target_language == "ja"

    def test_with_translation_no_target_defaults_empty(self):
        e = self._make_event("ja")
        t = e.with_translation(TranslationStatus.SKIPPED)
        assert t.translation.target_language == ""

    def test_with_translation_in_progress(self):
        e = self._make_event("ja")
        t = e.with_translation(TranslationStatus.IN_PROGRESS, target_language="en")
        assert t.translation.status == TranslationStatus.IN_PROGRESS
        assert t.translation.text is None

    def test_with_translation_failed(self):
        e = self._make_event("ja")
        t = e.with_translation(TranslationStatus.FAILED, target_language="en")
        assert t.translation.status == TranslationStatus.FAILED

    def test_original_event_unchanged(self):
        e = self._make_event("ja")
        t = e.with_translation(TranslationStatus.DONE, text="Test", target_language="en")
        assert e.translation is None  # Original unchanged
        assert t.translation is not None


class TestMeetingMeta:
    """Meeting metadata defaults."""

    def test_default_language_pair(self):
        m = MeetingMeta()
        assert m.language_pair == ["en", "ja"]

    def test_custom_language_pair(self):
        m = MeetingMeta(language_pair=["zh", "en"])
        assert m.language_pair == ["zh", "en"]

    def test_default_state(self):
        m = MeetingMeta()
        assert m.state == MeetingState.CREATED

    def test_meeting_id_generated(self):
        m1 = MeetingMeta()
        m2 = MeetingMeta()
        assert m1.meeting_id != m2.meeting_id
        assert len(m1.meeting_id) > 8

    def test_is_favorite_defaults_false(self):
        m = MeetingMeta()
        assert m.is_favorite is False

    def test_is_favorite_serialization_roundtrip(self):
        # New field must survive write-then-read of the JSON form, and
        # legacy meta.json files (no is_favorite key) must load with the
        # default False so existing meetings don't break.
        m = MeetingMeta(is_favorite=True)
        raw = m.model_dump_json()
        assert '"is_favorite":true' in raw or '"is_favorite": true' in raw
        m2 = MeetingMeta.model_validate_json(raw)
        assert m2.is_favorite is True

        legacy_no_field = MeetingMeta().model_dump_json().replace(
            '"is_favorite":false', ""
        ).replace(',,', ',').replace('{,', '{').replace(',}', '}')
        # Just verify the standard validator accepts a JSON missing the key
        m3 = MeetingMeta.model_validate_json('{"meeting_id": "abc"}')
        assert m3.is_favorite is False


class TestMeetingMetaLanguagePair:
    """``language_pair`` holds 1 or 2 codes — authoritative Pydantic
    validator. Every code path that constructs a ``MeetingMeta`` (API,
    reload from disk, fixtures) runs through this check, so invalid
    shapes cannot enter the system."""

    def test_bilingual_accepted(self):
        m = MeetingMeta(language_pair=["ja", "en"])
        assert m.language_pair == ["ja", "en"]
        assert m.is_monolingual is False

    def test_monolingual_accepted(self):
        m = MeetingMeta(language_pair=["en"])
        assert m.language_pair == ["en"]
        assert m.is_monolingual is True

    def test_default_is_bilingual(self):
        # Field default is ["ja", "en"] — monolingual is opt-in.
        m = MeetingMeta()
        assert m.is_monolingual is False

    def test_empty_rejected(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MeetingMeta(language_pair=[])

    def test_length_three_rejected(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MeetingMeta(language_pair=["ja", "en", "ko"])

    def test_duplicate_pair_rejected(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MeetingMeta(language_pair=["en", "en"])

    def test_unknown_code_rejected(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MeetingMeta(language_pair=["sw"])

    def test_reload_path_rejects_corrupt_shapes(self):
        """Pydantic's automatic JSON validation must fire on reload so a
        corrupted persisted meta.json fails loudly rather than silently
        starting a meeting in a weird state."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MeetingMeta.model_validate_json(
                '{"meeting_id": "abc", "language_pair": ["en", "en"]}'
            )
        with pytest.raises(ValidationError):
            MeetingMeta.model_validate_json(
                '{"meeting_id": "abc", "language_pair": []}'
            )
        # Valid length-1 round-trips through JSON.
        m = MeetingMeta.model_validate_json(
            '{"meeting_id": "abc", "language_pair": ["en"]}'
        )
        assert m.is_monolingual is True
