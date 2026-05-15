"""Tests for the HID descriptor decoder + Report ID 5 codec."""

from __future__ import annotations

import pytest

from meeting_scribe.speakerphone import descriptor as desc


def test_canonical_descriptor_length_matches_live_capture() -> None:
    # The fixture was dumped 2026-05-13 from
    # /sys/class/hidraw/hidraw0/device/report_descriptor on the SP325.
    # If a future descriptor refresh changes the length, this test
    # forces an explicit update rather than silently drifting.
    assert len(desc.SP325_DESCRIPTOR) == 211


def test_descriptor_parses_without_error() -> None:
    entries = desc.parse_descriptor(desc.SP325_DESCRIPTOR)
    assert entries, "descriptor parse returned no entries"
    assert all(e.usage_page > 0 for e in entries)


def test_descriptor_exposes_telephony_input_report_5() -> None:
    entries = desc.parse_descriptor(desc.SP325_DESCRIPTOR)
    telephony_inputs = [
        e
        for e in entries
        if e.usage_page == 0x0B and e.direction == "input" and e.report_id == 0x05
    ]
    assert telephony_inputs, "no telephony input report 5 found"
    # Hook Switch (0x20) and Phone Mute (0x2F) must both be present
    # in the union of telephony input usages so the daemon knows
    # which bits to watch.
    seen_usages = {u for e in telephony_inputs for u in e.usages}
    assert 0x20 in seen_usages, "Hook Switch (0x20) missing"
    assert 0x2F in seen_usages, "Phone Mute (0x2F) missing"


def test_descriptor_exposes_led_output_report_5() -> None:
    entries = desc.parse_descriptor(desc.SP325_DESCRIPTOR)
    led_outputs = [
        e
        for e in entries
        if e.usage_page == 0x08 and e.direction == "output" and e.report_id == 0x05
    ]
    assert led_outputs, "no LED output report 5 found"
    seen_usages = {u for e in led_outputs for u in e.usages}
    # The SP325 only physically exposes the Mute LED, but the descriptor
    # still claims the full HID-telephony LED set. Both the Off-Hook
    # (0x17) and Mute (0x09) usages must be present so the
    # encode_led_output_report bit positions stay correct.
    assert 0x17 in seen_usages, "Off-Hook LED (0x17) missing"
    assert 0x09 in seen_usages, "Mute LED (0x09) missing"


def test_decode_telephony_payload_unpacks_all_known_bits() -> None:
    # bit 0 = Hook Switch, bit 1 = Phone Mute, bit 5 = Speaker
    payload = bytes([0b00100011])  # 0x23
    decoded = desc.decode_telephony_report(payload)
    assert decoded["phone"] is True
    assert decoded["phone_mute"] is True
    assert decoded["speaker"] is True
    assert decoded["redial"] is False
    assert decoded["phone_key"] is False


def test_decode_telephony_payload_handles_zero() -> None:
    decoded = desc.decode_telephony_report(b"\x00")
    assert all(not pressed for pressed in decoded.values())


def test_decode_telephony_payload_rejects_empty() -> None:
    with pytest.raises(ValueError):
        desc.decode_telephony_report(b"")


def test_pressed_buttons_returns_only_set_bits() -> None:
    payload = bytes([0b00000001])  # only Hook Switch
    assert desc.pressed_buttons(payload) == {"phone"}


def test_encode_led_output_sets_mute_bit() -> None:
    out = desc.encode_led_output_report({"mute_led": True})
    # Mute LED is bit 1 of the output byte.
    assert out == bytes([0b00000010])


def test_encode_led_output_clears_unmentioned_bits() -> None:
    out = desc.encode_led_output_report({"mute_led": False})
    assert out == bytes([0])


def test_encode_led_output_combines_multiple_leds() -> None:
    out = desc.encode_led_output_report(
        {"off_hook_led": True, "mute_led": True, "ring_led": True},
    )
    # bits 0, 1, 2 → 0b00000111
    assert out == bytes([0b00000111])


def test_encode_led_output_ignores_unknown_led_name() -> None:
    # A future device that adds a non-standard LED name shouldn't crash
    # the encoder — it should just be dropped.
    out = desc.encode_led_output_report({"future_led": True, "mute_led": True})
    assert out == bytes([0b00000010])


def test_describe_renders_each_entry() -> None:
    entries = desc.parse_descriptor(desc.SP325_DESCRIPTOR)
    rendered = desc.describe(entries)
    # Each line should have report_id, dir, page tokens.
    for line in rendered.splitlines():
        assert "report_id=" in line
        assert "dir=" in line
        assert "page=" in line
