"""Tests for bt.parse_card_profiles + bt.choose_profile.

Two synthetic device fixtures (a Ray-Ban Meta lookalike with mSBC + a2dp,
and an A2DP-only speaker) drive the ranking + capability-classification
paths. Pure-function tests, no subprocess.
"""

from __future__ import annotations

import json

import pytest

from meeting_scribe.bt import (
    BluetoothCapabilityError,
    choose_profile,
    parse_card_profiles,
    parse_node_for_mac,
)


_RAY_BAN_LIKE = json.dumps(
    [
        {
            "name": "bluez_card.AA_BB_CC_DD_EE_FF",
            "properties": {"api.bluez5.address": "AA:BB:CC:DD:EE:FF"},
            "profiles": [
                {
                    "name": "a2dp-sink-sbc",
                    "description": "High Fidelity Playback (SBC)",
                    "available": True,
                },
                {
                    "name": "a2dp-sink-aac",
                    "description": "High Fidelity Playback (AAC)",
                    "available": True,
                },
                {
                    "name": "headset-head-unit-msbc",
                    "description": "Headset Head Unit (mSBC)",
                    "available": True,
                },
                {
                    "name": "headset-head-unit",
                    "description": "Headset Head Unit (CVSD)",
                    "available": True,
                },
                {"name": "off", "description": "Off", "available": True},
            ],
        }
    ]
)

_SPEAKER_ONLY = json.dumps(
    [
        {
            "name": "bluez_card.11_22_33_44_55_66",
            "properties": {"api.bluez5.address": "11:22:33:44:55:66"},
            "profiles": [
                {
                    "name": "a2dp-sink-sbc",
                    "description": "High Fidelity Playback (SBC)",
                    "available": True,
                }
            ],
        }
    ]
)


def test_parse_card_profiles_extracts_for_target_mac() -> None:
    profiles = parse_card_profiles(_RAY_BAN_LIKE, mac="AA:BB:CC:DD:EE:FF")
    assert len(profiles) == 4
    names = {p.name for p in profiles}
    assert "a2dp-sink-sbc" in names
    assert "headset-head-unit-msbc" in names
    # "off" is filtered out — never a useful target.
    assert "off" not in names


def test_parse_card_profiles_returns_empty_for_unknown_mac() -> None:
    assert parse_card_profiles(_RAY_BAN_LIKE, mac="00:11:22:33:44:55") == []


def test_choose_profile_a2dp_prefers_higher_codec() -> None:
    profiles = parse_card_profiles(_RAY_BAN_LIKE, mac="AA:BB:CC:DD:EE:FF")
    chosen = choose_profile(profiles, want="a2dp")
    # AAC > SBC in our ranking.
    assert chosen == "a2dp-sink-aac"


def test_choose_profile_hfp_prefers_msbc_over_cvsd() -> None:
    profiles = parse_card_profiles(_RAY_BAN_LIKE, mac="AA:BB:CC:DD:EE:FF")
    chosen = choose_profile(profiles, want="hfp")
    assert chosen == "headset-head-unit-msbc"


def test_choose_profile_a2dp_only_device_has_no_hfp() -> None:
    profiles = parse_card_profiles(_SPEAKER_ONLY, mac="11:22:33:44:55:66")
    with pytest.raises(BluetoothCapabilityError):
        choose_profile(profiles, want="hfp")


def test_choose_profile_a2dp_only_device_serves_a2dp() -> None:
    profiles = parse_card_profiles(_SPEAKER_ONLY, mac="11:22:33:44:55:66")
    assert choose_profile(profiles, want="a2dp") == "a2dp-sink-sbc"


def test_parse_node_for_mac_filters_by_address() -> None:
    sources = json.dumps(
        [
            {
                "name": "bluez_input.AA_BB_CC_DD_EE_FF.headset-head-unit-msbc",
                "properties": {
                    "device.bus": "bluetooth",
                    "api.bluez5.address": "AA:BB:CC:DD:EE:FF",
                },
            },
            {
                "name": "alsa_input.usb_other_device",
                "properties": {"device.bus": "usb"},
            },
        ]
    )
    sinks = json.dumps(
        [
            {
                "name": "bluez_output.AA_BB_CC_DD_EE_FF.a2dp-sink",
                "properties": {
                    "device.bus": "bluetooth",
                    "api.bluez5.address": "AA:BB:CC:DD:EE:FF",
                },
            }
        ]
    )
    src, sink = parse_node_for_mac(sources, sinks, mac="AA:BB:CC:DD:EE:FF")
    assert src is not None and "AA_BB_CC_DD_EE_FF" in src
    assert sink is not None and sink.startswith("bluez_output.")


def test_parse_node_for_mac_returns_none_when_absent() -> None:
    src, sink = parse_node_for_mac("[]", "[]", mac="AA:BB:CC:DD:EE:FF")
    assert (src, sink) == (None, None)


def test_parse_card_profiles_rejects_invalid_mac() -> None:
    """Validation runs before JSON parse — refuse malformed MACs."""
    from meeting_scribe.bt import BluetoothError

    with pytest.raises(BluetoothError, match="invalid MAC"):
        parse_card_profiles("[]", mac="not-a-mac")


def test_parse_card_profiles_handles_dict_form() -> None:
    """Some pactl versions emit profiles as a dict instead of a list —
    parse both shapes."""
    dict_form = json.dumps(
        [
            {
                "name": "bluez_card.AA_BB_CC_DD_EE_FF",
                "properties": {"api.bluez5.address": "AA:BB:CC:DD:EE:FF"},
                "profiles": {
                    "a2dp-sink": {"description": "A2DP", "available": True},
                    "headset-head-unit": {"description": "HFP", "available": True},
                },
            }
        ]
    )
    profiles = parse_card_profiles(dict_form, mac="AA:BB:CC:DD:EE:FF")
    assert {p.name for p in profiles} == {"a2dp-sink", "headset-head-unit"}
