"""Tests for the speakerphone sidecar JSON store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meeting_scribe.speakerphone import mapping
from meeting_scribe.speakerphone.constants import (
    DEFAULT_LONG_PRESS_MS,
    DEFAULT_MEETING_PROFILE,
    MAPPING_SCHEMA_VERSION,
)


def test_default_document_passes_validation() -> None:
    doc = mapping.default_document()
    mapping.validate(doc)


def test_default_document_includes_sp325() -> None:
    doc = mapping.default_document()
    assert "413c:8223" in doc["devices"]
    sp325 = doc["devices"]["413c:8223"]
    assert sp325["buttons"]["phone"]["short"] == "tts_cycle"
    assert sp325["buttons"]["phone"]["long"] == "interpretation_toggle"
    assert sp325["buttons"]["teams"]["short"] == "meeting_record_toggle"
    assert sp325["buttons"]["phone_mute"]["short"] == "mic_mute_toggle"


def test_default_document_includes_default_meeting_profile() -> None:
    doc = mapping.default_document()
    assert doc["default_meeting_profile"]["name"] == DEFAULT_MEETING_PROFILE["name"]
    assert doc["default_meeting_profile"]["languages"] == ["en", "ja"]


def test_default_long_press_threshold_is_1000ms() -> None:
    # Acceptance criterion 3 explicitly pins this; if someone changes
    # DEFAULT_LONG_PRESS_MS, the UI copy must change too, so we lock
    # it down here.
    assert DEFAULT_LONG_PRESS_MS == 1000
    doc = mapping.default_document()
    assert doc["long_press_ms"] == 1000


def test_load_returns_defaults_when_file_missing(tmp_path: Path) -> None:
    target = tmp_path / "missing.json"
    doc = mapping.load(target)
    mapping.validate(doc)


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    target = tmp_path / "speakerphone.json"
    doc = mapping.default_document()
    mapping.save(doc, target)
    assert target.exists()
    loaded = mapping.load(target)
    assert loaded == doc


def test_save_is_atomic_via_tmp_then_replace(tmp_path: Path) -> None:
    # We can't easily prove atomicity from userspace, but we can prove
    # the temp file isn't left behind on success.
    target = tmp_path / "speakerphone.json"
    mapping.save(mapping.default_document(), target)
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(target.name + ".")]
    assert leftovers == []


def test_load_rejects_wrong_schema_version(tmp_path: Path) -> None:
    target = tmp_path / "speakerphone.json"
    payload = mapping.default_document()
    payload["version"] = MAPPING_SCHEMA_VERSION + 1
    target.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        mapping.load(target)


def test_load_rejects_non_object_top_level(tmp_path: Path) -> None:
    target = tmp_path / "speakerphone.json"
    target.write_text("[1, 2, 3]")
    with pytest.raises(ValueError):
        mapping.load(target)


def test_load_fills_missing_top_level_keys(tmp_path: Path) -> None:
    target = tmp_path / "speakerphone.json"
    # User-written doc missing leds and default_meeting_profile.
    target.write_text(
        json.dumps(
            {
                "version": MAPPING_SCHEMA_VERSION,
                "long_press_ms": 1000,
                "devices": {
                    "413c:8223": {
                        "name": "Dell SP325",
                        "buttons": {
                            "phone": {"short": "noop"},
                        },
                    },
                },
            },
        ),
    )
    loaded = mapping.load(target)
    assert "default_meeting_profile" in loaded
    assert "states" in loaded["leds"]


def test_validate_rejects_unknown_action() -> None:
    doc = mapping.default_document()
    doc["devices"]["413c:8223"]["buttons"]["phone"]["short"] = "summon_the_kraken"
    with pytest.raises(mapping.MappingValidationError):
        mapping.validate(doc)


def test_validate_rejects_consumer_page_button() -> None:
    doc = mapping.default_document()
    doc["devices"]["413c:8223"]["buttons"]["volume_up"] = {"short": "noop"}
    with pytest.raises(mapping.MappingValidationError):
        mapping.validate(doc)


def test_validate_rejects_unknown_led_state() -> None:
    doc = mapping.default_document()
    doc["leds"]["states"]["disco_inferno"] = {
        "enabled": True,
        "pattern": "fast_blink",
    }
    with pytest.raises(mapping.MappingValidationError):
        mapping.validate(doc)


def test_validate_rejects_unknown_pattern() -> None:
    doc = mapping.default_document()
    doc["leds"]["states"]["recording"]["pattern"] = "rainbow"
    with pytest.raises(mapping.MappingValidationError):
        mapping.validate(doc)


def test_validate_rejects_out_of_range_long_press() -> None:
    doc = mapping.default_document()
    doc["long_press_ms"] = 50
    with pytest.raises(mapping.MappingValidationError):
        mapping.validate(doc)


def test_validate_rejects_profile_with_zero_languages() -> None:
    doc = mapping.default_document()
    doc["default_meeting_profile"]["languages"] = []
    with pytest.raises(mapping.MappingValidationError):
        mapping.validate(doc)


def test_validate_rejects_profile_room_tts_outside_meeting_langs() -> None:
    doc = mapping.default_document()
    doc["default_meeting_profile"]["languages"] = ["en", "ja"]
    doc["default_meeting_profile"]["room_tts_language"] = "fr"
    with pytest.raises(mapping.MappingValidationError):
        mapping.validate(doc)


def test_compute_etag_is_stable() -> None:
    doc = mapping.default_document()
    etag1 = mapping.compute_etag(doc)
    etag2 = mapping.compute_etag(doc)
    assert etag1 == etag2
    assert len(etag1) == 32


def test_compute_etag_changes_after_edit() -> None:
    doc = mapping.default_document()
    etag_before = mapping.compute_etag(doc)
    doc["devices"]["413c:8223"]["buttons"]["phone"]["short"] = "noop"
    etag_after = mapping.compute_etag(doc)
    assert etag_before != etag_after


# ── button_feedback schema ──────────────────────────────────────────


def test_default_document_includes_button_feedback() -> None:
    doc = mapping.default_document()
    feedback = doc["button_feedback"]
    assert feedback["enabled"] is True
    assert feedback["language"] == "en"
    assert feedback["overrides"] == {}


def test_validate_accepts_default_button_feedback() -> None:
    doc = mapping.default_document()
    mapping.validate(doc)  # should not raise


def test_validate_rejects_non_bool_enabled() -> None:
    doc = mapping.default_document()
    doc["button_feedback"]["enabled"] = "yes"
    with pytest.raises(mapping.MappingValidationError):
        mapping.validate(doc)


def test_validate_rejects_non_tts_native_language() -> None:
    doc = mapping.default_document()
    # "nl" is in LANGUAGE_REGISTRY but tts_native=False
    doc["button_feedback"]["language"] = "nl"
    with pytest.raises(mapping.MappingValidationError) as excinfo:
        mapping.validate(doc)
    assert "TTS-native" in str(excinfo.value)


def test_validate_rejects_unknown_language_code() -> None:
    doc = mapping.default_document()
    doc["button_feedback"]["language"] = "xx"
    with pytest.raises(mapping.MappingValidationError):
        mapping.validate(doc)


def test_validate_accepts_each_tts_native_language() -> None:
    from meeting_scribe.languages import LANGUAGE_REGISTRY

    for code, lang in LANGUAGE_REGISTRY.items():
        if not lang.tts_native:
            continue
        doc = mapping.default_document()
        doc["button_feedback"]["language"] = code
        mapping.validate(doc)  # should not raise for any TTS-native code


def test_validate_rejects_unknown_label_in_overrides() -> None:
    doc = mapping.default_document()
    doc["button_feedback"]["overrides"] = {
        "fly_to_mars": {"en": "Engage!"},
    }
    with pytest.raises(mapping.MappingValidationError) as excinfo:
        mapping.validate(doc)
    assert "fly_to_mars" in str(excinfo.value)


def test_validate_rejects_non_tts_native_language_in_overrides() -> None:
    doc = mapping.default_document()
    doc["button_feedback"]["overrides"] = {
        "volume_up": {"nl": "Harder"},  # nl is ASR-only, not TTS-native
    }
    with pytest.raises(mapping.MappingValidationError) as excinfo:
        mapping.validate(doc)
    assert "TTS-native" in str(excinfo.value) or "nl" in str(excinfo.value)


def test_validate_rejects_empty_override_text() -> None:
    doc = mapping.default_document()
    doc["button_feedback"]["overrides"] = {
        "volume_up": {"en": "   "},
    }
    with pytest.raises(mapping.MappingValidationError):
        mapping.validate(doc)


def test_validate_accepts_well_formed_overrides() -> None:
    doc = mapping.default_document()
    doc["button_feedback"]["overrides"] = {
        "volume_up": {"en": "Loud", "ja": "うるさい"},
        "mic_muted": {"en": "Hush"},
    }
    mapping.validate(doc)  # should not raise
