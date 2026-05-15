"""Tests for the daemon's feedback dispatcher.

Covers:

* State-reflecting label selection — the announcement matches the
  post-action state, not the pre-action state.
* Phone long-press fires exactly ONE feedback (interp_on/off) — no
  preceding ``phone_hold`` utterance.
* The daemon ALWAYS calls ``client.speak`` regardless of the
  ``button_feedback.enabled`` flag visible to it; the enable gate
  lives exclusively on the server (so language/enable changes apply
  with zero daemon-poll lag).
* Best-effort: a failing speak call doesn't propagate.
* Consumer-page keys (Vol+/Vol-/Mute) are SILENT — daemon drives
  ``wpctl`` but never calls ``client.speak``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from meeting_scribe.speakerphone import daemon as sp_daemon
from meeting_scribe.speakerphone.evdev_listener import PressEvent


class _RecordingClient:
    """MeetingClient stub that records every speak/get_state call."""

    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state
        self.speak_calls: list[str] = []
        self.get_state_calls = 0
        self.report_press_calls: list[tuple[str, str, str]] = []
        self.action_calls: list[str] = []

    async def get_state(self) -> dict[str, Any]:
        self.get_state_calls += 1
        return self._state

    async def set_interpretation(self, **kwargs):
        return {}

    async def toggle_mic_mute(self):
        return {}

    async def toggle_meeting_record(self):
        return {}

    async def speak(self, *, label_id: str, language=None, overrides=None) -> dict[str, Any]:
        self.speak_calls.append(label_id)
        return {"ok": True}

    async def report_press(self, *, device_key, button, press_kind) -> None:
        self.report_press_calls.append((device_key, button, press_kind))


def _make_session(
    client: _RecordingClient,
    mapping_doc: dict | None = None,
) -> sp_daemon.DeviceSession:
    """Build a DeviceSession bound to the recording client.

    We use throwaway evdev/hidraw paths because the test never opens
    them — only the `_emit_press` / `_emit_feedback` / `_on_consumer_key`
    methods are exercised here.
    """
    doc = (
        mapping_doc
        if mapping_doc is not None
        else {
            "version": 1,
            "long_press_ms": 1000,
            "devices": {
                "413c:8223": {
                    "name": "Test SP325",
                    "buttons": {
                        "phone": {"short": "tts_cycle", "long": "interpretation_toggle"},
                        "teams": {"short": "meeting_record_toggle", "long": "noop"},
                        "phone_mute": {"short": "mic_mute_toggle", "long": "noop"},
                    },
                    "leds": {"mute_led": {"state_machine": "default"}},
                },
            },
            "leds": {"states": {}},
            "default_meeting_profile": {
                "name": "Test",
                "languages": ["en", "ja"],
                "interpretation_enabled": True,
                "room_tts_language": "all",
                "admin_tts_language": "en",
            },
            "button_feedback": {"enabled": True, "language": "en", "overrides": {}},
        }
    )
    return sp_daemon.DeviceSession(
        device_key="413c:8223",
        evdev_path=Path("/dev/null"),
        hidraw_path=Path("/dev/null"),
        client=client,
        mapping_doc=doc,
    )


# ── State-reflecting label selection ──────────────────────────────────


@pytest.mark.asyncio
async def test_phone_short_announces_new_tts_direction() -> None:
    client = _RecordingClient(
        {
            "interpretation": {"enabled": True, "room_tts_language": "ja"},
            "meeting": {"recording": False},
        }
    )
    session = _make_session(client)
    label = await session._resolve_telephony_feedback_label(
        PressEvent(button="phone", kind="short"),
    )
    assert label == "tts_dir_ja"


@pytest.mark.asyncio
async def test_phone_short_with_all_direction() -> None:
    client = _RecordingClient(
        {
            "interpretation": {"enabled": True, "room_tts_language": "all"},
            "meeting": {"recording": False},
        }
    )
    session = _make_session(client)
    label = await session._resolve_telephony_feedback_label(
        PressEvent(button="phone", kind="short"),
    )
    assert label == "tts_dir_all"


@pytest.mark.asyncio
async def test_phone_long_announces_interp_state_when_enabled() -> None:
    client = _RecordingClient(
        {
            "interpretation": {"enabled": True, "room_tts_language": "ja"},
            "meeting": {"recording": False},
        }
    )
    session = _make_session(client)
    label = await session._resolve_telephony_feedback_label(
        PressEvent(button="phone", kind="long"),
    )
    assert label == "interp_on"


@pytest.mark.asyncio
async def test_phone_long_announces_interp_off_when_disabled() -> None:
    client = _RecordingClient(
        {
            "interpretation": {"enabled": False, "room_tts_language": "ja"},
            "meeting": {"recording": False},
        }
    )
    session = _make_session(client)
    label = await session._resolve_telephony_feedback_label(
        PressEvent(button="phone", kind="long"),
    )
    assert label == "interp_off"


@pytest.mark.asyncio
async def test_teams_announces_recording_state() -> None:
    client = _RecordingClient(
        {
            "interpretation": {"enabled": True},
            "meeting": {"recording": True, "meeting_id": "abc"},
        }
    )
    session = _make_session(client)
    label = await session._resolve_telephony_feedback_label(
        PressEvent(button="teams", kind="short"),
    )
    assert label == "meeting_started"


@pytest.mark.asyncio
async def test_teams_announces_stopped_when_idle() -> None:
    client = _RecordingClient(
        {
            "interpretation": {"enabled": True},
            "meeting": {"recording": False},
        }
    )
    session = _make_session(client)
    label = await session._resolve_telephony_feedback_label(
        PressEvent(button="teams", kind="short"),
    )
    assert label == "meeting_stopped"


@pytest.mark.asyncio
async def test_phone_mute_announces_mic_state() -> None:
    client = _RecordingClient(
        {
            "interpretation": {"enabled": True, "mic_muted": True},
            "meeting": {"recording": True},
        }
    )
    session = _make_session(client)
    label = await session._resolve_telephony_feedback_label(
        PressEvent(button="phone_mute", kind="short"),
    )
    assert label == "mic_muted"


@pytest.mark.asyncio
async def test_phone_mute_announces_mic_on_when_unmuted() -> None:
    client = _RecordingClient(
        {
            "interpretation": {"enabled": True, "mic_muted": False},
            "meeting": {"recording": True},
        }
    )
    session = _make_session(client)
    label = await session._resolve_telephony_feedback_label(
        PressEvent(button="phone_mute", kind="short"),
    )
    assert label == "mic_unmuted"


@pytest.mark.asyncio
async def test_unknown_button_returns_none() -> None:
    """Buttons without a feedback mapping (e.g. a future phone_mute/long)
    return None so the daemon skips silently."""
    client = _RecordingClient({"interpretation": {}, "meeting": {}})
    session = _make_session(client)
    label = await session._resolve_telephony_feedback_label(
        PressEvent(button="teams", kind="long"),
    )
    assert label is None


# ── Phone long-press fires EXACTLY one feedback ───────────────────────


@pytest.mark.asyncio
async def test_phone_long_press_fires_one_speak_call() -> None:
    """Long-press of Phone must produce one (and only one) speak call.

    No preceding "Phone hold" utterance.
    """
    client = _RecordingClient(
        {
            "interpretation": {"enabled": True, "room_tts_language": "ja"},
            "meeting": {"recording": False},
        }
    )
    session = _make_session(client)
    await session._emit_press(PressEvent(button="phone", kind="long"))
    # interpretation_toggle action triggers — speak called exactly once.
    assert client.speak_calls == ["interp_on"]


# ── Daemon ALWAYS calls speak regardless of local cached `enabled` ────


@pytest.mark.asyncio
async def test_daemon_always_calls_speak_even_when_locally_disabled() -> None:
    """Codex P1 carry-over: enable gate is server-only.

    The daemon must NOT short-circuit when its cached
    ``button_feedback.enabled`` is False. The server side will
    enforce the gate after re-reading the fresh mapping; the daemon
    always sends so a "disable" toggle applies on the very next
    press with zero poll lag.
    """
    client = _RecordingClient(
        {
            "interpretation": {"enabled": True, "room_tts_language": "en"},
            "meeting": {"recording": False},
        }
    )
    # Mapping doc says feedback is DISABLED — daemon must ignore.
    mapping_doc = _make_session(client)._mapping_doc.copy()
    mapping_doc["button_feedback"] = {
        "enabled": False,
        "language": "en",
        "overrides": {},
    }
    session = _make_session(client, mapping_doc=mapping_doc)
    await session._emit_feedback("volume_up")
    assert client.speak_calls == ["volume_up"]


@pytest.mark.asyncio
async def test_emit_feedback_swallows_speak_exceptions() -> None:
    """A flaky TTS backend cannot crash the button-press path."""

    class _RaisingClient(_RecordingClient):
        async def speak(self, **kwargs):
            raise RuntimeError("simulated server 500")

    client = _RaisingClient({})
    session = _make_session(client)
    # Should not raise.
    await session._emit_feedback("volume_up")


# ── Consumer-key callback drives wpctl silently ─────────────────────


class _FakeProc:
    """Minimal subprocess fake — records argv via the surrounding factory."""

    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self):
        return (self._stdout, self._stderr)

    async def wait(self):
        return self.returncode

    def kill(self):  # pragma: no cover
        pass


@pytest.mark.asyncio
async def test_on_consumer_key_mute_drives_wpctl_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KEY_MUTE press: daemon toggles wpctl mute, no TTS announcement.

    Announcing "system muted" out loud after every mute press got
    quickly tiresome (2026-05-13 — the user was hearing a queue full
    of "Volume up" repeats and asked to keep the speech for state
    changes that AREN'T self-evident from the audio).
    """
    client = _RecordingClient({})
    session = _make_session(client)

    calls: list[list[str]] = []

    async def fake_create(*args, **kwargs):
        calls.append(list(args))
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
    await session._on_consumer_key("system_mute_toggled")

    assert any("set-mute" in c for c in calls), (
        "daemon must call wpctl set-mute on a headless system"
    )
    assert client.speak_calls == [], (
        "consumer-page presses are silent — the audio change IS the feedback"
    )


@pytest.mark.asyncio
async def test_on_consumer_key_volume_up_drives_wpctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vol+ must reach ``wpctl set-volume … 5%+`` on a headless box.

    The GB10 has no desktop media-key agent. If the daemon doesn't
    drive wpctl itself, the SP325's volume button does nothing —
    that's the bug reported 2026-05-13 (3× KEY_VOLUMEUP arrived at
    event3, ``wpctl get-volume`` still showed 0.40). NO speech is
    emitted; the volume change itself is the feedback.
    """
    client = _RecordingClient({})
    session = _make_session(client)

    calls: list[list[str]] = []

    async def fake_create(*args, **kwargs):
        calls.append(list(args))
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
    await session._on_consumer_key("volume_up")

    assert len(calls) == 1
    argv = calls[0]
    assert argv[0] == "wpctl"
    assert "set-volume" in argv
    assert "@DEFAULT_AUDIO_SINK@" in argv
    assert "5%+" in argv
    # Hard cap so successive presses can't overshoot 100%.
    assert "-l" in argv and "1.0" in argv
    assert client.speak_calls == [], "Vol+ is silent — sound-level IS the feedback"


@pytest.mark.asyncio
async def test_on_consumer_key_volume_down_drives_wpctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Symmetric: Vol- must call ``wpctl set-volume … 5%-``, no speech."""
    client = _RecordingClient({})
    session = _make_session(client)

    calls: list[list[str]] = []

    async def fake_create(*args, **kwargs):
        calls.append(list(args))
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
    await session._on_consumer_key("volume_down")

    assert len(calls) == 1
    argv = calls[0]
    assert "set-volume" in argv and "5%-" in argv
    assert client.speak_calls == []


@pytest.mark.asyncio
async def test_on_consumer_key_swallows_wpctl_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wpctl missing/erroring must not break the press path. Vol+ is
    silent regardless of whether wpctl succeeded."""
    client = _RecordingClient({})
    session = _make_session(client)

    async def failing_create(*args, **kwargs):
        raise FileNotFoundError("no wpctl")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", failing_create)
    await session._on_consumer_key("volume_up")
    assert client.speak_calls == []
