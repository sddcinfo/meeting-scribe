"""Persistence round-trip coverage for monolingual meetings.

The ``language_pair`` field on ``MeetingMeta`` now holds 1 or 2 codes.
Length 1 = monolingual (no translation work, no translation entries
in the journal, empty translations in exports). These tests lock the
serializer + loader contract so a monolingual meeting still looks
monolingual after a server restart or a reprocess pass.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from meeting_scribe.models import MeetingMeta


class TestMeetingMetaJSONRoundTrip:
    """``MeetingMeta`` ↔ JSON must preserve monolingual shape."""

    def test_monolingual_meta_roundtrips_through_json(self):
        m = MeetingMeta(meeting_id="mono-1", language_pair=["en"])
        payload = m.model_dump_json()
        # Ensure ``language_pair`` is serialized as a length-1 list, not
        # coerced into a string or padded to length 2.
        parsed = json.loads(payload)
        assert parsed["language_pair"] == ["en"]
        # Reload — Pydantic's field validator runs automatically.
        reloaded = MeetingMeta.model_validate_json(payload)
        assert reloaded.language_pair == ["en"]
        assert reloaded.is_monolingual is True

    def test_bilingual_meta_roundtrips_unchanged(self):
        # Regression guard: the length-2 happy path must not be broken.
        m = MeetingMeta(meeting_id="bi-1", language_pair=["ja", "en"])
        reloaded = MeetingMeta.model_validate_json(m.model_dump_json())
        assert reloaded.language_pair == ["ja", "en"]
        assert reloaded.is_monolingual is False

    def test_reload_rejects_corrupt_length_three(self):
        # Pydantic's model_validate_json runs the field validator, so a
        # corrupt on-disk meta.json fails at load time rather than
        # starting the meeting in a weird state.
        with pytest.raises(ValidationError):
            MeetingMeta.model_validate_json(
                '{"meeting_id": "x", "language_pair": ["en", "ja", "ko"]}'
            )

    def test_reload_rejects_duplicate_pair(self):
        with pytest.raises(ValidationError):
            MeetingMeta.model_validate_json(
                '{"meeting_id": "x", "language_pair": ["en", "en"]}'
            )

    def test_reload_rejects_empty(self):
        with pytest.raises(ValidationError):
            MeetingMeta.model_validate_json(
                '{"meeting_id": "x", "language_pair": []}'
            )

    def test_reload_rejects_unknown_code(self):
        with pytest.raises(ValidationError):
            MeetingMeta.model_validate_json(
                '{"meeting_id": "x", "language_pair": ["xx"]}'
            )


class TestJournalMonolingualShape:
    """Journal events from a monolingual meeting never carry a
    ``translation`` object. Clients (popout, reader, export) must
    treat ``translation`` as an optional field — bilingual meetings
    also have the absent case during the pre-translation window, so
    this is a contract the frontend was already observing."""

    def test_segment_without_translation_serializes_cleanly(self):
        from meeting_scribe.models import SpeakerAttribution, TranscriptEvent

        e = TranscriptEvent(
            segment_id="s1",
            text="Hello",
            language="en",
            is_final=True,
            start_ms=0,
            end_ms=1000,
            speakers=[SpeakerAttribution(cluster_id=0, source="test")],
        )
        blob = e.model_dump_json()
        loaded = json.loads(blob)
        # ``translation`` may be absent OR may be present as a null/empty
        # object — either is acceptable, but any DONE/SKIPPED fields must
        # NOT leak into a monolingual segment.
        tr = loaded.get("translation")
        if tr is not None:
            assert tr.get("status") != "done", (
                "Monolingual segments must not carry a completed translation"
            )


class TestGetTranslationTargetListContract:
    """``get_translation_target`` must accept a length-1 list (monolingual)
    and return ``None`` regardless of detected language — this is what
    the translation queue, refinement pipeline, and slide job all rely
    on for their short-circuit branches."""

    def test_monolingual_list_returns_none(self):
        from meeting_scribe.languages import get_translation_target

        for detected in ("en", "ja", "zh", "unknown"):
            assert get_translation_target(detected, ["en"]) is None
