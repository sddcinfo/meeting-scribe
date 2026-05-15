"""Coverage + resolution tests for the button-feedback label catalog.

The coverage assertion is the load-bearing one: every TTS-native
language must have an entry for every canonical label_id, otherwise
runtime fallback to English silently degrades the operator's chosen
language. The CI run of these tests is what guarantees a contributor
who adds a new language can't forget any labels.
"""

from __future__ import annotations

import pytest

from meeting_scribe.languages import LANGUAGE_REGISTRY
from meeting_scribe.speakerphone import labels


def _tts_native_codes() -> set[str]:
    return {code for code, lang in LANGUAGE_REGISTRY.items() if lang.tts_native}


# ── Coverage ───────────────────────────────────────────────────────────


def test_every_label_has_every_tts_native_language() -> None:
    """The keystone test — no missed translations.

    Coverage matrix: |CANONICAL_LABELS| × |tts_native_codes|. Every
    cell must be a non-empty string. A new language added to
    LANGUAGE_REGISTRY with tts_native=True forces the contributor to
    add a translation for every label_id, or CI fails here.
    """
    tts_langs = _tts_native_codes()
    missing: list[str] = []
    empty: list[str] = []
    for label_id in labels.CANONICAL_LABELS:
        for lang in tts_langs:
            value = labels.LABELS[label_id].get(lang)
            if value is None:
                missing.append(f"{label_id}[{lang}]")
            elif not isinstance(value, str) or not value.strip():
                empty.append(f"{label_id}[{lang}]")
    assert missing == [], f"missing translations: {missing}"
    assert empty == [], f"empty translations: {empty}"


def test_canonical_labels_includes_all_expected_ids() -> None:
    expected_subset = {
        # Consumer-page labels: volume_up / volume_down still ride the
        # catalog because the admin SPA's "Test feedback" preview button
        # uses ``volume_up`` as its sample. The ``system_*`` mute
        # entries were removed 2026-05-13 when the consumer-page
        # announcements were dropped; only the action sentinel
        # ``system_mute_toggled`` survives in
        # ``KERNEL_KEY_TO_CONSUMER_LABEL`` (no speech translation).
        "volume_up",
        "volume_down",
        "mic_muted",
        "mic_unmuted",
        "tts_dir_en",
        "tts_dir_ja",
        "tts_dir_all",
        "interp_on",
        "interp_off",
        "meeting_started",
        "meeting_stopped",
    }
    assert expected_subset.issubset(labels.CANONICAL_LABELS), (
        f"missing canonical labels: {expected_subset - labels.CANONICAL_LABELS}"
    )


def test_every_tts_native_language_has_a_direction_label() -> None:
    """For every TTS-native language XX, the `tts_dir_XX` label exists.

    The Phone-button cycle announces the new direction by speaking the
    `tts_dir_<new_lang>` label. A missing entry would crash the
    daemon's `_emit_press` resolution.
    """
    tts_langs = _tts_native_codes()
    for lang in tts_langs:
        assert f"tts_dir_{lang}" in labels.CANONICAL_LABELS, (
            f"missing tts_dir_{lang} for TTS-native language {lang}"
        )


# ── Resolve precedence ────────────────────────────────────────────────


def test_resolve_returns_catalog_entry_for_known_label_and_language() -> None:
    assert labels.resolve("volume_up", "en") == "Volume up"
    assert labels.resolve("volume_up", "ja") == "音量アップ"


def test_resolve_falls_back_to_english_for_unknown_language() -> None:
    # "xx" is not a TTS-native language. Resolver falls back to en.
    assert labels.resolve("volume_up", "xx") == "Volume up"


def test_resolve_unknown_label_id_raises_label_not_found() -> None:
    with pytest.raises(labels.LabelNotFoundError) as excinfo:
        labels.resolve("nonexistent_button", "en")
    assert "nonexistent_button" in str(excinfo.value)


def test_resolve_label_not_found_is_subclass_of_key_error() -> None:
    """Catch-everything ``except KeyError`` blocks keep working."""
    with pytest.raises(KeyError):
        labels.resolve("nope", "en")


# ── Override precedence ──────────────────────────────────────────────


def test_resolve_override_wins_over_catalog() -> None:
    overrides = {"volume_up": {"en": "Loud", "ja": "うるさい"}}
    assert labels.resolve("volume_up", "en", overrides) == "Loud"
    assert labels.resolve("volume_up", "ja", overrides) == "うるさい"


def test_resolve_override_only_affects_listed_language() -> None:
    overrides = {"volume_up": {"en": "Loud"}}
    # ja not in overrides → falls through to catalog
    assert labels.resolve("volume_up", "ja", overrides) == "音量アップ"


def test_resolve_empty_override_falls_through_to_catalog() -> None:
    overrides = {"volume_up": {"en": "   "}}  # whitespace-only is "empty"
    assert labels.resolve("volume_up", "en", overrides) == "Volume up"


def test_resolve_unknown_label_in_overrides_does_not_match() -> None:
    overrides = {"different_button": {"en": "Whatever"}}
    assert labels.resolve("volume_up", "en", overrides) == "Volume up"


def test_resolve_with_none_overrides_is_safe() -> None:
    assert labels.resolve("volume_up", "en", None) == "Volume up"
    assert labels.resolve("volume_up", "en", {}) == "Volume up"
