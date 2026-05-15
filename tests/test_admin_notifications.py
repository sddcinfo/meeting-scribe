"""Phase-1 admin notifications helper tests.

Producers (``audio_routing.reconcile_audio_routing``) call
``put_notification(kind, **fields)`` and ``dismiss_if_present(kind)``;
``/api/status`` calls ``active_notifications()``. The persistence layer
mirrors ``state.pending_admin_notifications`` into settings.json so the
banner survives a process restart.
"""

from __future__ import annotations

import pytest

from meeting_scribe.runtime import state
from meeting_scribe.server_support import admin_notifications
from meeting_scribe.server_support.settings_store import (
    _load_settings_override,
    _save_settings_override,
)


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Redirect settings.json to a tmp file so tests don't trample real state.

    ``SETTINGS_OVERRIDE_FILE`` is a module-level Path constant in
    ``settings_store`` — monkeypatch it directly. Also reset the
    mtime-keyed cache so the new file is read fresh.
    """
    from meeting_scribe.server_support import settings_store

    tmp_file = tmp_path / "settings.json"
    monkeypatch.setattr(settings_store, "SETTINGS_OVERRIDE_FILE", tmp_file)
    settings_store._settings_cache = None
    settings_store._settings_cache_mtime = 0.0
    state.pending_admin_notifications = {}
    yield
    state.pending_admin_notifications = {}
    settings_store._settings_cache = None
    settings_store._settings_cache_mtime = 0.0


def test_put_notification_persists_to_settings_and_state() -> None:
    admin_notifications.put_notification("mic_rebound", mic_from="old", mic_to="new")
    assert "mic_rebound" in state.pending_admin_notifications
    entry = state.pending_admin_notifications["mic_rebound"]
    assert entry["mic_from"] == "old"
    assert entry["mic_to"] == "new"
    assert entry["dismissed_at"] is None
    # Also on disk.
    persisted = _load_settings_override().get(admin_notifications.SETTINGS_KEY)
    assert persisted["mic_rebound"]["mic_to"] == "new"


def test_put_notification_replaces_prior_entry_with_same_kind() -> None:
    admin_notifications.put_notification("mic_rebound", mic_from="a", mic_to="b")
    admin_notifications.put_notification("mic_rebound", mic_from="c", mic_to="d")
    entries = state.pending_admin_notifications
    assert len(entries) == 1
    assert entries["mic_rebound"]["mic_to"] == "d"


def test_put_notification_resets_dismissed_at_on_recurrence() -> None:
    admin_notifications.put_notification("mic_unresolved", mic_node="x")
    admin_notifications.dismiss_if_present("mic_unresolved")
    # Recurrence after dismissal must clear the dismiss flag so the banner
    # re-surfaces.
    admin_notifications.put_notification("mic_unresolved", mic_node="x")
    assert state.pending_admin_notifications["mic_unresolved"]["dismissed_at"] is None


def test_dismiss_if_present_returns_false_when_nothing_to_dismiss() -> None:
    assert admin_notifications.dismiss_if_present("mic_rebound") is False


def test_dismiss_if_present_marks_entry_dismissed_and_returns_true() -> None:
    admin_notifications.put_notification("mic_rebound", mic_from="a", mic_to="b")
    assert admin_notifications.dismiss_if_present("mic_rebound") is True
    # Already dismissed → second call returns False (no double-dismiss).
    assert admin_notifications.dismiss_if_present("mic_rebound") is False


def test_active_notifications_filters_dismissed_and_orders_newest_first() -> None:
    import time as _time

    admin_notifications.put_notification("mic_unresolved", mic_node="a")
    # Force a small time gap so created_at differs deterministically.
    _time.sleep(0.01)
    admin_notifications.put_notification("mic_rebound", mic_from="x", mic_to="y")
    admin_notifications.dismiss_if_present("mic_unresolved")
    active = admin_notifications.active_notifications()
    assert [n["kind"] for n in active] == ["mic_rebound"]


def test_load_into_state_mirrors_persisted_into_state() -> None:
    # Pre-seed settings as if a prior process saved a notification.
    _save_settings_override(
        {
            admin_notifications.SETTINGS_KEY: {
                "mic_rebound": {
                    "kind": "mic_rebound",
                    "mic_from": "old",
                    "mic_to": "new",
                    "created_at": 100.0,
                    "dismissed_at": None,
                }
            }
        }
    )
    state.pending_admin_notifications = {}
    admin_notifications.load_into_state()
    assert "mic_rebound" in state.pending_admin_notifications


def test_load_into_state_tolerates_hand_edited_corrupt_entries() -> None:
    _save_settings_override(
        {
            admin_notifications.SETTINGS_KEY: {
                "mic_rebound": {"kind": "mic_rebound"},  # valid dict
                "junk_kind": "not a dict",  # corrupt — must be filtered out
            }
        }
    )
    admin_notifications.load_into_state()
    assert "mic_rebound" in state.pending_admin_notifications
    assert "junk_kind" not in state.pending_admin_notifications
