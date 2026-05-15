"""Tests for the persisted ``interpretation_last_room_tts_language`` field.

Acceptance criterion 4: a Phone-button long-press → off → long-press → on
sequence (possibly straddling a meeting-scribe restart) must restore the
exact direction the user had selected. The server-side update happens
inside ``settings_store._effective_interpretation_last_room_tts_language``
+ the ``admin_audio.audio_interpretation_post`` side effect.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from meeting_scribe.server_support import settings_store


@pytest.fixture
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect settings_store to a tmpdir so tests don't touch real config."""
    target = tmp_path / "settings.json"
    monkeypatch.setattr(settings_store, "SETTINGS_OVERRIDE_FILE", target)
    # Invalidate any module-level cache.
    monkeypatch.setattr(settings_store, "_settings_cache", None, raising=False)
    monkeypatch.setattr(settings_store, "_settings_cache_mtime", 0.0, raising=False)
    return target


def test_default_last_direction_is_all(isolated_settings: Path) -> None:
    assert settings_store._effective_interpretation_last_room_tts_language() == "all"


def test_setting_last_direction_persists(isolated_settings: Path) -> None:
    settings_store._save_settings_override(
        {"interpretation_last_room_tts_language": "ja"},
    )
    assert settings_store._effective_interpretation_last_room_tts_language() == "ja"


def test_invalid_last_direction_falls_back_to_all(isolated_settings: Path) -> None:
    settings_store._save_settings_override(
        {"interpretation_last_room_tts_language": 42},
    )
    assert settings_store._effective_interpretation_last_room_tts_language() == "all"


def test_empty_string_last_direction_falls_back_to_all(
    isolated_settings: Path,
) -> None:
    settings_store._save_settings_override(
        {"interpretation_last_room_tts_language": ""},
    )
    assert settings_store._effective_interpretation_last_room_tts_language() == "all"


def test_last_direction_survives_separate_load(isolated_settings: Path) -> None:
    """Writes via _save_settings_override are reloadable from disk.

    A fresh ``_load_settings_override`` call after the write must pick up
    the persisted value (no in-memory state can be the answer here).
    """
    settings_store._save_settings_override(
        {"interpretation_last_room_tts_language": "en"},
    )
    # Force cache invalidation as if the process restarted.
    settings_store._settings_cache = None  # type: ignore[attr-defined]
    settings_store._settings_cache_mtime = 0.0  # type: ignore[attr-defined]
    fresh = settings_store._load_settings_override()
    assert fresh.get("interpretation_last_room_tts_language") == "en"
    assert settings_store._effective_interpretation_last_room_tts_language() == "en"
