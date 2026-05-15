"""Tests for the LED priority-ordered state machine."""

from __future__ import annotations

from meeting_scribe.speakerphone import led_state_machine as sm
from meeting_scribe.speakerphone.constants import DEFAULT_LED_STATE_BEHAVIOR


def test_idle_ready_when_nothing_active() -> None:
    res = sm.resolve(sm.SystemSignals(), DEFAULT_LED_STATE_BEHAVIOR)
    assert res.state == "idle_ready"
    assert res.pattern == "off"


def test_error_wins_over_everything() -> None:
    res = sm.resolve(
        sm.SystemSignals(
            error=True,
            backend_unready=True,
            mic_muted=True,
            recording=True,
        ),
        DEFAULT_LED_STATE_BEHAVIOR,
    )
    assert res.state == "error"


def test_backend_unready_outranks_mic_muted_and_recording() -> None:
    res = sm.resolve(
        sm.SystemSignals(backend_unready=True, mic_muted=True, recording=True),
        DEFAULT_LED_STATE_BEHAVIOR,
    )
    assert res.state == "backend_unready"


def test_mic_muted_outranks_recording() -> None:
    res = sm.resolve(
        sm.SystemSignals(mic_muted=True, recording=True),
        DEFAULT_LED_STATE_BEHAVIOR,
    )
    assert res.state == "mic_muted"
    assert res.pattern == "solid"


def test_recording_wins_when_no_higher_state_active() -> None:
    res = sm.resolve(sm.SystemSignals(recording=True), DEFAULT_LED_STATE_BEHAVIOR)
    assert res.state == "recording"
    assert res.pattern == "slow_pulse"


def test_disabled_state_falls_through_to_next() -> None:
    config = dict(DEFAULT_LED_STATE_BEHAVIOR)
    config["mic_muted"] = {"enabled": False, "pattern": "solid"}
    res = sm.resolve(
        sm.SystemSignals(mic_muted=True, recording=True),
        config,
    )
    # mic_muted disabled → fall through to recording.
    assert res.state == "recording"


def test_disabled_idle_state_safely_falls_to_default_off() -> None:
    config = {
        "error": {"enabled": False, "pattern": "very_fast_blink"},
        "backend_unready": {"enabled": False, "pattern": "fast_blink"},
        "mic_muted": {"enabled": False, "pattern": "solid"},
        "recording": {"enabled": False, "pattern": "slow_pulse"},
        "idle_ready": {"enabled": False, "pattern": "off"},
    }
    res = sm.resolve(sm.SystemSignals(), config)
    # Every state disabled — safety net should still produce a
    # resolution rather than crash.
    assert res.state == "idle_ready"
    assert res.pattern == "off"


def test_resolver_uses_user_pattern_when_supplied() -> None:
    config = dict(DEFAULT_LED_STATE_BEHAVIOR)
    config["recording"] = {"enabled": True, "pattern": "blink"}
    res = sm.resolve(sm.SystemSignals(recording=True), config)
    assert res.pattern == "blink"


def test_resolver_falls_back_to_default_pattern_on_invalid_config() -> None:
    config = dict(DEFAULT_LED_STATE_BEHAVIOR)
    # Operator wrote a malformed config: pattern not a string.
    config["recording"] = {"enabled": True, "pattern": 42}
    res = sm.resolve(sm.SystemSignals(recording=True), config)
    assert res.state == "recording"
    assert res.pattern == "slow_pulse"  # default


def test_resolver_handles_missing_config() -> None:
    # state_config=None should still resolve to the defaults.
    res = sm.resolve(sm.SystemSignals(backend_unready=True), None)
    assert res.state == "backend_unready"
    assert res.pattern == "fast_blink"


def test_bootup_sequence_transitions_cleanly() -> None:
    # Simulate boot: backend unready → ready. Single-state resolution
    # per tick is enough to assert the transition happens at the right
    # boundary.
    boot = sm.resolve(
        sm.SystemSignals(backend_unready=True),
        DEFAULT_LED_STATE_BEHAVIOR,
    )
    ready = sm.resolve(sm.SystemSignals(), DEFAULT_LED_STATE_BEHAVIOR)
    assert boot.state == "backend_unready"
    assert ready.state == "idle_ready"
    assert boot.pattern == "fast_blink"
    assert ready.pattern == "off"
