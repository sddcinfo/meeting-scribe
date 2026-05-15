"""One-shot legacy-key migration in settings_store.

Covers the rollout pattern: a settings.json file written by a build that
predates the routing-card era carries ``audio_meeting_sink_node``. On
load, ``settings_store`` rewrites that key to
``audio_admin_tts_sink_node`` and leaves a ``..._legacy_backup`` breadcrumb
so a determined operator can hand-restore on rollback. The migration is
single-shot per process and never re-runs once the legacy key is gone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def fresh_settings_store(tmp_path, monkeypatch):
    """Yield a freshly-imported settings_store pinned to ``tmp_path``.

    settings_store caches the loaded dict at module level and tracks
    whether migration has already run; each test wants a clean slate.
    Patching the override file path is more honest than monkeypatching
    the home directory.
    """
    settings_path = tmp_path / "settings.json"
    import meeting_scribe.server_support.settings_store as store

    # Reset module-level state without re-importing.
    monkeypatch.setattr(store, "SETTINGS_OVERRIDE_FILE", settings_path)
    monkeypatch.setattr(store, "_settings_cache", None)
    monkeypatch.setattr(store, "_settings_cache_mtime", 0.0)
    monkeypatch.setattr(store, "_legacy_migration_attempted", False)
    return store, settings_path


def _write_settings(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def test_migration_moves_legacy_when_new_key_empty(fresh_settings_store) -> None:
    store, path = fresh_settings_store
    _write_settings(path, {"audio_meeting_sink_node": "bluez_output.42"})

    result = store._load_settings_override()

    assert result["audio_admin_tts_sink_node"] == "bluez_output.42"
    assert "audio_meeting_sink_node" not in result
    assert result["audio_admin_tts_sink_node_legacy_backup"] == "bluez_output.42"

    # And it's persisted, not just in-memory:
    on_disk = json.loads(path.read_text())
    assert on_disk["audio_admin_tts_sink_node"] == "bluez_output.42"
    assert "audio_meeting_sink_node" not in on_disk


def test_migration_keeps_new_key_when_both_present(fresh_settings_store) -> None:
    store, path = fresh_settings_store
    _write_settings(
        path,
        {
            "audio_meeting_sink_node": "bluez_output.OLD",
            "audio_admin_tts_sink_node": "bluez_output.NEW",
        },
    )

    result = store._load_settings_override()

    assert result["audio_admin_tts_sink_node"] == "bluez_output.NEW"
    assert "audio_meeting_sink_node" not in result
    # The backup captures the legacy value, not the new key.
    assert result["audio_admin_tts_sink_node_legacy_backup"] == "bluez_output.OLD"


def test_migration_is_noop_without_legacy_key(fresh_settings_store) -> None:
    store, path = fresh_settings_store
    _write_settings(path, {"audio_admin_tts_sink_node": "alsa_output.poly"})

    result = store._load_settings_override()

    assert result["audio_admin_tts_sink_node"] == "alsa_output.poly"
    assert "audio_admin_tts_sink_node_legacy_backup" not in result
    # File untouched on no-op:
    on_disk = json.loads(path.read_text())
    assert on_disk == {"audio_admin_tts_sink_node": "alsa_output.poly"}


def test_migration_is_idempotent_within_process(fresh_settings_store) -> None:
    store, path = fresh_settings_store
    _write_settings(path, {"audio_meeting_sink_node": "alsa_output.legacy"})

    first = store._load_settings_override()
    # Force cache miss so we re-enter the load path. The migration flag
    # at module level should prevent a second migration pass.
    store._settings_cache_mtime = -1.0
    second = store._load_settings_override()

    assert first == second
    assert "audio_meeting_sink_node" not in second


def test_migration_preserves_existing_backup(fresh_settings_store) -> None:
    """A backup from a previous migration is the authoritative one."""
    store, path = fresh_settings_store
    _write_settings(
        path,
        {
            "audio_meeting_sink_node": "alsa_output.second_legacy",
            "audio_admin_tts_sink_node": "alsa_output.current",
            "audio_admin_tts_sink_node_legacy_backup": "alsa_output.first_legacy",
        },
    )

    result = store._load_settings_override()

    assert result["audio_admin_tts_sink_node"] == "alsa_output.current"
    # The original backup wins; we don't overwrite it with a fresh one.
    assert result["audio_admin_tts_sink_node_legacy_backup"] == "alsa_output.first_legacy"
    assert "audio_meeting_sink_node" not in result


def test_migration_does_not_create_backup_for_empty_legacy(fresh_settings_store) -> None:
    """An empty legacy value carries no information; skip the breadcrumb."""
    store, path = fresh_settings_store
    _write_settings(path, {"audio_meeting_sink_node": ""})

    result = store._load_settings_override()

    assert "audio_meeting_sink_node" not in result
    # No value to back up means no breadcrumb.
    assert "audio_admin_tts_sink_node_legacy_backup" not in result
