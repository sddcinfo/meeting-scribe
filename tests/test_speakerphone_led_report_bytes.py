"""Tests for LED hidraw report byte assembly + pattern expansion."""

from __future__ import annotations

import pytest

from meeting_scribe.speakerphone import hid_leds


def test_build_led_report_prepends_report_id() -> None:
    report = hid_leds.build_led_report({"mute_led": True})
    assert report[0] == hid_leds.REPORT_ID_TELEPHONY_LED
    assert len(report) == 2


def test_mute_ring_on_sets_only_bit_1() -> None:
    assert hid_leds.mute_ring(True) == bytes([0x05, 0b00000010])


def test_mute_ring_off_clears_all_bits() -> None:
    assert hid_leds.mute_ring(False) == bytes([0x05, 0])


def test_build_led_report_combines_multiple_leds() -> None:
    report = hid_leds.build_led_report(
        {"off_hook_led": True, "mute_led": True, "ring_led": True},
    )
    assert report == bytes([0x05, 0b00000111])


def test_expand_pattern_solid_yields_single_on_step() -> None:
    steps = hid_leds.expand_pattern([(1000, 0)])
    assert len(steps) == 1
    assert steps[0].on is True
    assert steps[0].duration_ms == 1000


def test_expand_pattern_off_yields_single_off_step() -> None:
    steps = hid_leds.expand_pattern([(0, 1000)])
    assert len(steps) == 1
    assert steps[0].on is False


def test_expand_pattern_blink_yields_two_alternating_steps() -> None:
    steps = hid_leds.expand_pattern([(250, 250)])
    assert [s.on for s in steps] == [True, False]
    assert [s.duration_ms for s in steps] == [250, 250]


def test_expand_pattern_double_blink_yields_four_steps() -> None:
    steps = hid_leds.expand_pattern([(125, 125, 125, 625)])
    assert [s.on for s in steps] == [True, False, True, False]
    assert [s.duration_ms for s in steps] == [125, 125, 125, 625]


def test_resolve_pattern_returns_off_for_unknown_name() -> None:
    assert hid_leds.resolve_pattern("nonexistent") == hid_leds.LED_PATTERNS["off"]


def test_pattern_runner_dedups_writes() -> None:
    writes: list[bool] = []
    schedule = [(500, 500)]
    runner = hid_leds.PatternRunner(schedule, writes.append)
    # First tick → on, write True.
    runner.tick()
    # Second tick → off, write False.
    runner.tick()
    # Third tick → on, write True (state changed back).
    runner.tick()
    assert writes == [True, False, True]


def test_pattern_runner_with_solid_writes_once_then_holds() -> None:
    writes: list[bool] = []
    runner = hid_leds.PatternRunner([(1000, 0)], writes.append)
    for _ in range(5):
        runner.tick()
    # Only the first tick should have produced a write since the
    # state never changes.
    assert writes == [True]


def test_pattern_runner_write_off_emits_false_and_resets_cache() -> None:
    writes: list[bool] = []
    runner = hid_leds.PatternRunner([(1000, 0)], writes.append)
    runner.tick()  # writes True
    runner.write_off()  # writes False
    runner.tick()  # writes True again (cache reset by write_off)
    assert writes == [True, False, True]


def test_expand_pattern_rejects_odd_length_slot_tuple() -> None:
    with pytest.raises(ValueError):
        hid_leds.expand_pattern([(100, 100, 100)])
