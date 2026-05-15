"""Regression: ``ensure_local_sink_listener_registered`` must purge stale
listeners when the operator disables every local-sink target.

Without the purge, ``_listener_tts_demand`` keeps returning the leaked
``LocalSinkListener``'s voice mode, the TTS producer keeps enqueueing
work that has no sink to deliver to, and the "TTS stalled" badge fires
while the UI shows outputs off (observed 2026-05-12, this regression).
"""

from __future__ import annotations

import pytest

import meeting_scribe.audio.local_sink as local_sink
from meeting_scribe.audio.local_sink import (
    LOCAL_SINK_ROLE_ROOM,
    LocalSinkListener,
    ensure_local_sink_listener_registered,
)
from meeting_scribe.runtime import state


@pytest.fixture
def isolated_audio_out_state():
    old_clients = set(state._audio_out_clients)
    old_prefs = dict(state._audio_out_prefs)
    state._audio_out_clients.clear()
    state._audio_out_prefs.clear()
    yield
    state._audio_out_clients = old_clients
    state._audio_out_prefs = old_prefs


def _register_room_listener(monkeypatch, settings: dict) -> LocalSinkListener:
    """Force a room-sink listener into the fan-out as if the operator
    had previously configured one."""
    monkeypatch.setattr(local_sink, "_load_settings_override", lambda: settings)
    monkeypatch.setattr(local_sink, "should_enable_local_sink", lambda: True)
    monkeypatch.setattr(
        local_sink,
        "_resolve_sink_target",
        lambda _settings, role: "alsa_output.room-poly" if role == LOCAL_SINK_ROLE_ROOM else "",
    )
    monkeypatch.setattr(
        local_sink,
        "_resolve_safe_room_sink_target",
        lambda _settings: "alsa_output.room-poly",
    )
    listener = ensure_local_sink_listener_registered()
    assert isinstance(listener, LocalSinkListener)
    return listener


def test_disabled_path_unregisters_existing_room_listener(monkeypatch, isolated_audio_out_state):
    listener = _register_room_listener(
        monkeypatch, settings={"room_sink_node": "alsa_output.room-poly"}
    )
    assert listener in state._audio_out_clients
    assert listener in state._audio_out_prefs

    # Operator clears every local-sink target. The next reconcile must
    # not just return early — it must clean up the stale listener so
    # _listener_tts_demand returns an empty set.
    monkeypatch.setattr(local_sink, "should_enable_local_sink", lambda: False)

    assert ensure_local_sink_listener_registered() is None
    assert listener not in state._audio_out_clients
    assert listener not in state._audio_out_prefs


def test_disabled_path_is_noop_when_nothing_registered(monkeypatch, isolated_audio_out_state):
    monkeypatch.setattr(local_sink, "should_enable_local_sink", lambda: False)

    # No LocalSinkListener present; must remain a clean no-op.
    sentinel = object()
    state._audio_out_clients.add(sentinel)
    state._audio_out_prefs[sentinel] = object()

    assert ensure_local_sink_listener_registered() is None
    # Non-LocalSinkListener entries are untouched.
    assert sentinel in state._audio_out_clients
    assert sentinel in state._audio_out_prefs
