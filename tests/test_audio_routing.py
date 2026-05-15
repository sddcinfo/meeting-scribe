"""Pure-function tests for ``audio_routing.parse_pw_dump_devices`` +
``parse_wpctl_defaults`` + ``get_routing_settings``.

The fixture mirrors a GB10 with the Dell WL5024 BT headset, the Poly
Sync 20-M USB speakerphone, the built-in NVIDIA HDA HDMI audio sink,
and a stray non-audio video source — exactly the inventory the
``admin-audio-card.js`` UI has to render.
"""

from __future__ import annotations

import json

from meeting_scribe.audio.audio_routing import (
    SETTINGS_AUDIO_ADMIN_TTS_SINK_NODE,
    SETTINGS_AUDIO_MEETING_MIC_ACTIVE,
    SETTINGS_AUDIO_MEETING_MIC_NODE,
    admin_room_sinks_collide,
    audio_nodes_share_physical_device,
    get_routing_settings,
    parse_pw_dump_devices,
    parse_wpctl_defaults,
    parse_wpctl_volume,
)

_PW_DUMP_FIXTURE = json.dumps(
    [
        # BT headset device + nodes (a2dp + hfp variants surfacing one
        # source + one sink each at the time of capture).
        {
            "id": 47,
            "type": "PipeWire:Interface:Device",
            "info": {
                "props": {
                    "device.api": "bluez5",
                    "device.bus": "bluetooth",
                    "device.name": "bluez_card.10_54_D2_C4_76_0B",
                    "api.bluez5.address": "10:54:D2:C4:76:0B",
                }
            },
        },
        {
            "id": 45,
            "type": "PipeWire:Interface:Node",
            "info": {
                "props": {
                    "media.class": "Audio/Source",
                    "node.name": "bluez_input.10_54_D2_C4_76_0B.0",
                    "node.description": "Dell WL5024 Headset",
                    "device.api": "bluez5",
                }
            },
        },
        {
            "id": 43,
            "type": "PipeWire:Interface:Node",
            "info": {
                "props": {
                    "media.class": "Audio/Sink",
                    "node.name": "bluez_output.10_54_D2_C4_76_0B.1",
                    "node.description": "Dell WL5024 Headset",
                    "device.api": "bluez5",
                }
            },
        },
        # USB Poly speakerphone.
        {
            "id": 54,
            "type": "PipeWire:Interface:Node",
            "info": {
                "props": {
                    "media.class": "Audio/Sink",
                    "node.name": "alsa_output.usb-Plantronics_Poly_Sync_20-M_8B33-00.analog-stereo",
                    "node.description": "Poly Sync 20-M Analog Stereo",
                    "device.api": "alsa",
                }
            },
        },
        {
            "id": 55,
            "type": "PipeWire:Interface:Node",
            "info": {
                "props": {
                    "media.class": "Audio/Source",
                    "node.name": "alsa_input.usb-Plantronics_Poly_Sync_20-M_8B33-00.mono-fallback",
                    "node.description": "Poly Sync 20-M Mono",
                    "device.api": "alsa",
                }
            },
        },
        # Built-in HDMI sink.
        {
            "id": 80,
            "type": "PipeWire:Interface:Node",
            "info": {
                "props": {
                    "media.class": "Audio/Sink",
                    "node.name": "alsa_output.pci-0000_03_00.1.hdmi-stereo",
                    "node.description": "NVIDIA HDA HDMI",
                    "device.api": "alsa",
                }
            },
        },
        # Non-audio video node — must be ignored.
        {
            "id": 91,
            "type": "PipeWire:Interface:Node",
            "info": {
                "props": {
                    "media.class": "Video/Source",
                    "node.name": "v4l2_input.usb-Logitech_C920",
                }
            },
        },
    ]
)


_WPCTL_STATUS_FIXTURE = """\
PipeWire 'pipewire-0'

Audio
 ├─ Devices:
 │
 ├─ Sinks:
 │      43. Dell WL5024 Headset                 [vol: 0.40]
 │  *   54. Poly Sync 20-M Analog Stereo        [vol: 0.37]
 │      80. NVIDIA HDA HDMI                     [vol: 1.00]
 │
 ├─ Sink endpoints:
 │
 ├─ Sources:
 │      45. Dell WL5024 Headset                 [vol: 0.00]
 │  *   55. Poly Sync 20-M Mono                 [vol: 1.00]
 │
 ├─ Source endpoints:
 │
 └─ Streams:

Video
"""


def test_parse_pw_dump_devices_groups_sinks_and_sources() -> None:
    devices = parse_pw_dump_devices(_PW_DUMP_FIXTURE)
    assert len(devices["sinks"]) == 3
    assert len(devices["sources"]) == 2


def test_parse_pw_dump_devices_classifies_usb_bt_hdmi() -> None:
    devices = parse_pw_dump_devices(_PW_DUMP_FIXTURE)
    by_node = {s["node_name"]: s for s in devices["sinks"]}
    assert by_node["bluez_output.10_54_D2_C4_76_0B.1"]["device_class"] == "bluetooth"
    assert (
        by_node["alsa_output.usb-Plantronics_Poly_Sync_20-M_8B33-00.analog-stereo"]["device_class"]
        == "usb"
    )
    assert by_node["alsa_output.pci-0000_03_00.1.hdmi-stereo"]["device_class"] == "hdmi"


def test_parse_pw_dump_devices_ignores_video_nodes() -> None:
    devices = parse_pw_dump_devices(_PW_DUMP_FIXTURE)
    for entry in devices["sources"] + devices["sinks"]:
        assert "Logitech" not in (entry.get("description") or "")


def test_parse_pw_dump_devices_handles_garbage_input() -> None:
    assert parse_pw_dump_devices("not json") == {"sources": [], "sinks": []}
    # A bare dict (not wrapped in a list) is treated as a single object
    # snapshot — caught one edge case from real pw-dump.
    assert parse_pw_dump_devices('{"not": "node"}') == {"sources": [], "sinks": []}


def test_parse_pw_dump_devices_handles_concatenated_arrays() -> None:
    """pw-dump (without ``-m``) sometimes emits two snapshots in one
    stdout when PipeWire activity races the dump call (e.g. a parallel
    ``wpctl status`` triggering a Client added/removed event). The
    parser must accept multiple concatenated JSON arrays — bug surfaced
    live on a GB10 where the in-process probe got "Extra data: line
    4684" against valid pw-dump output and the routing UI flagged
    real devices as ``(missing)``."""
    first_snapshot = json.dumps(
        [
            {
                "id": 100,
                "type": "PipeWire:Interface:Node",
                "info": {
                    "props": {
                        "media.class": "Audio/Source",
                        "node.name": "src.first.snapshot",
                        "node.description": "First",
                        "device.api": "alsa",
                    }
                },
            }
        ]
    )
    second_snapshot = json.dumps(
        [
            {
                "id": 200,
                "type": "PipeWire:Interface:Node",
                "info": {
                    "props": {
                        "media.class": "Audio/Sink",
                        "node.name": "sink.second.snapshot",
                        "node.description": "Second",
                        "device.api": "alsa",
                    }
                },
            }
        ]
    )
    devices = parse_pw_dump_devices(first_snapshot + "\n" + second_snapshot)
    assert {s["node_name"] for s in devices["sources"]} == {"src.first.snapshot"}
    assert {s["node_name"] for s in devices["sinks"]} == {"sink.second.snapshot"}


def test_parse_pw_dump_devices_later_snapshot_wins_on_id_collision() -> None:
    """When the same ``id`` appears in multiple snapshots, the freshest
    state (last snapshot) is retained — matches operator intuition that
    pw-dump's later events reflect the current live state."""
    earlier = json.dumps(
        [
            {
                "id": 42,
                "type": "PipeWire:Interface:Node",
                "info": {
                    "props": {
                        "media.class": "Audio/Sink",
                        "node.name": "stale.name",
                        "node.description": "Stale",
                        "device.api": "alsa",
                    }
                },
            }
        ]
    )
    later = json.dumps(
        [
            {
                "id": 42,
                "type": "PipeWire:Interface:Node",
                "info": {
                    "props": {
                        "media.class": "Audio/Sink",
                        "node.name": "fresh.name",
                        "node.description": "Fresh",
                        "device.api": "alsa",
                    }
                },
            }
        ]
    )
    devices = parse_pw_dump_devices(earlier + later)
    assert len(devices["sinks"]) == 1
    assert devices["sinks"][0]["node_name"] == "fresh.name"


def test_parse_pw_dump_devices_uses_node_name_when_description_missing() -> None:
    raw = json.dumps(
        [
            {
                "id": 1,
                "type": "PipeWire:Interface:Node",
                "info": {
                    "props": {
                        "media.class": "Audio/Sink",
                        "node.name": "fallback_only",
                    }
                },
            }
        ]
    )
    devices = parse_pw_dump_devices(raw)
    assert devices["sinks"][0]["description"] == "fallback_only"


def test_parse_wpctl_defaults_extracts_starred_ids() -> None:
    defaults = parse_wpctl_defaults(_WPCTL_STATUS_FIXTURE)
    assert defaults == {"source": 55, "sink": 54}


def test_parse_wpctl_defaults_returns_none_when_no_default() -> None:
    txt = "Audio\n ├─ Sinks:\n │      43. Foo\n ├─ Sink endpoints:\n"
    defaults = parse_wpctl_defaults(txt)
    assert defaults["sink"] is None
    assert defaults["source"] is None


def test_get_routing_settings_normalizes_empty_strings() -> None:
    settings = {
        SETTINGS_AUDIO_MEETING_MIC_NODE: "",
        SETTINGS_AUDIO_ADMIN_TTS_SINK_NODE: None,
        SETTINGS_AUDIO_MEETING_MIC_ACTIVE: True,
    }
    out = get_routing_settings(settings)
    assert out == {
        "mic_node": "",
        "admin_sink_node": "",
        "room_sink_node": "",
        "mic_active": True,
        "mic_stable_id": "",
        "mic_discriminator": None,
    }


def test_get_routing_settings_defaults_missing_keys() -> None:
    out = get_routing_settings({})
    assert out == {
        "mic_node": "",
        "admin_sink_node": "",
        "room_sink_node": "",
        "mic_active": False,
        "mic_stable_id": "",
        "mic_discriminator": None,
    }


def test_audio_nodes_share_physical_device_for_poly_usb_source_and_sink() -> None:
    source = "alsa_input.usb-Plantronics_Poly_Sync_20-M_8B33ABCDEF-00.mono-fallback"
    sink = "alsa_output.usb-Plantronics_Poly_Sync_20-M_8B33ABCDEF-00.analog-stereo"

    assert audio_nodes_share_physical_device(source, sink)


def test_audio_nodes_do_not_match_unrelated_devices() -> None:
    source = "alsa_input.pci-0000_00_1f.3.analog-stereo"
    sink = "alsa_output.usb-Plantronics_Poly_Sync_20-M_8B33ABCDEF-00.analog-stereo"

    assert not audio_nodes_share_physical_device(source, sink)


def test_parse_wpctl_volume_unmuted() -> None:
    parsed = parse_wpctl_volume("Volume: 0.85\n")
    assert parsed == {"volume": 0.85, "muted": False}


def test_parse_wpctl_volume_muted() -> None:
    parsed = parse_wpctl_volume("Volume: 1.00 [MUTED]\n")
    assert parsed == {"volume": 1.0, "muted": True}


def test_parse_wpctl_volume_handles_extra_whitespace_and_no_decimals() -> None:
    parsed = parse_wpctl_volume("Volume:  1\n")
    assert parsed == {"volume": 1.0, "muted": False}


def test_parse_wpctl_volume_returns_none_on_unparseable_input() -> None:
    assert parse_wpctl_volume("") is None
    assert parse_wpctl_volume("Node 99 not found\n") is None


_POLY_SINK = "alsa_output.usb-Plantronics_Poly_Sync_20-M_8B33-00.analog-stereo"
_POLY_HEADSET_SINK = "bluez_output.10_54_D2_C4_76_0B.1"


def test_admin_room_collide_when_same_device_and_room_all() -> None:
    assert admin_room_sinks_collide(
        admin_sink=_POLY_SINK,
        room_sink=_POLY_SINK,
        admin_lang="en",
        room_lang="all",
    )


def test_admin_room_collide_when_same_device_and_same_lang() -> None:
    assert admin_room_sinks_collide(
        admin_sink=_POLY_SINK,
        room_sink=_POLY_SINK,
        admin_lang="ja",
        room_lang="ja",
    )


def test_admin_room_no_collision_when_room_lang_differs_from_admin_lang() -> None:
    # Same speaker, different content — intentional bilingual output, not echo.
    assert not admin_room_sinks_collide(
        admin_sink=_POLY_SINK,
        room_sink=_POLY_SINK,
        admin_lang="en",
        room_lang="ja",
    )


def test_admin_room_no_collision_when_sinks_are_different_physical_devices() -> None:
    assert not admin_room_sinks_collide(
        admin_sink=_POLY_HEADSET_SINK,
        room_sink=_POLY_SINK,
        admin_lang="en",
        room_lang="all",
    )


def test_admin_room_no_collision_when_either_sink_unset() -> None:
    assert not admin_room_sinks_collide(
        admin_sink="",
        room_sink=_POLY_SINK,
        admin_lang="en",
        room_lang="all",
    )
    assert not admin_room_sinks_collide(
        admin_sink=_POLY_SINK,
        room_sink="",
        admin_lang="en",
        room_lang="all",
    )


def test_admin_room_collide_defaults_treat_blank_room_lang_as_all() -> None:
    # Pre-2026-04 settings.json may omit room_tts_language entirely;
    # absent value resolves to the "all" default.
    assert admin_room_sinks_collide(
        admin_sink=_POLY_SINK,
        room_sink=_POLY_SINK,
        admin_lang="en",
        room_lang=None,
    )


# ──────────────────────────────────────────────────────────────
# Phase 1 — stable_id extraction + resolver tagged-union tests
# ──────────────────────────────────────────────────────────────


from meeting_scribe.audio.audio_routing import (
    Ambiguous,
    NotFound,
    Resolved,
    _discriminator_matches,
    _physical_audio_device_id,
    resolve_node_for_stable_id,
)


def test_stable_id_extracts_sp325_usb_serial() -> None:
    # The 2026-05-14 demo failure binding. ``.2`` instance suffix is
    # stripped by the regex anchor.
    sid = _physical_audio_device_id(
        "alsa_input.usb-Dell_Inc._Dell_SP325_Speakerphone_0000000000000000-00.pro-input-0.2"
    )
    assert sid.startswith("usb:dell_inc._dell_sp325_speakerphone_")
    assert sid != ""


def test_stable_id_extracts_poly_sync_usb_serial() -> None:
    sid = _physical_audio_device_id(
        "alsa_input.usb-Plantronics_Poly_Sync_20-M_ABC123-00.mono-fallback"
    )
    assert sid.startswith("usb:plantronics_poly_sync_20-m_")
    assert sid != ""


def test_stable_id_extracts_bluetooth_mac() -> None:
    sid = _physical_audio_device_id("bluez_input.10_54_D2_C4_76_0B.0")
    assert sid == "bluez:10:54:d2:c4:76:0b"


def test_stable_id_returns_empty_for_unrecognized_synthetic_names() -> None:
    # Pure synthetic test-only name. The fallback regex chops off the
    # alsa_input./alsa_output. prefix but cannot extract a stable id
    # from a free-form string. Empty return = "unstable" = caller must
    # not auto-rebind.
    assert _physical_audio_device_id("") == ""
    assert _physical_audio_device_id(None) == ""


def test_discriminator_matches_empty_want_matches_anything() -> None:
    # Legacy bindings (no discriminator persisted) should match any
    # live entry — empty / None ``want`` doesn't constrain.
    assert _discriminator_matches(None, {"port": "pro-input-0", "nick": "Pro"})
    assert _discriminator_matches({}, {"port": "pro-input-0"})


def test_discriminator_matches_field_by_field() -> None:
    want = {"port": "pro-input-0"}
    assert _discriminator_matches(want, {"port": "pro-input-0", "nick": "Pro"})
    assert not _discriminator_matches(want, {"port": "headset-input", "nick": "Pro"})


def test_discriminator_matches_ignores_empty_want_fields() -> None:
    # Want fields that are empty strings or None don't constrain.
    want = {"port": "pro-input-0", "nick": ""}
    assert _discriminator_matches(want, {"port": "pro-input-0", "nick": "any"})


def test_resolver_returns_not_found_for_empty_stable_id() -> None:
    result = resolve_node_for_stable_id("", None, [{"stable_id": "usb:foo", "node_name": "x"}])
    assert isinstance(result, NotFound)


def test_resolver_returns_resolved_on_unique_match() -> None:
    devices = [
        {
            "node_name": "alsa_input.usb-foo-00.pro-input-0.2",
            "stable_id": "usb:foo",
            "discriminator": {"port": "pro-input-0"},
        },
        {"node_name": "other", "stable_id": "usb:bar", "discriminator": {"port": "input"}},
    ]
    result = resolve_node_for_stable_id("usb:foo", None, devices)
    assert isinstance(result, Resolved)
    assert result.node_name == "alsa_input.usb-foo-00.pro-input-0.2"


def test_resolver_returns_not_found_when_device_absent() -> None:
    devices = [{"node_name": "x", "stable_id": "usb:other", "discriminator": None}]
    result = resolve_node_for_stable_id("usb:foo", None, devices)
    assert isinstance(result, NotFound)


def test_resolver_returns_ambiguous_when_multiple_stable_id_matches_without_discriminator() -> None:
    devices = [
        {
            "node_name": "alsa_input.usb-foo-00.input-A.2",
            "stable_id": "usb:foo",
            "discriminator": {"port": "input-A"},
        },
        {
            "node_name": "alsa_input.usb-foo-00.input-B.2",
            "stable_id": "usb:foo",
            "discriminator": {"port": "input-B"},
        },
    ]
    result = resolve_node_for_stable_id("usb:foo", None, devices)
    assert isinstance(result, Ambiguous)
    assert len(result.candidates) == 2
    assert {c["node_name"] for c in result.candidates} == {
        "alsa_input.usb-foo-00.input-A.2",
        "alsa_input.usb-foo-00.input-B.2",
    }


def test_resolver_narrows_with_discriminator_to_single_match() -> None:
    devices = [
        {
            "node_name": "alsa_input.usb-foo-00.input-A.2",
            "stable_id": "usb:foo",
            "discriminator": {"port": "input-A"},
        },
        {
            "node_name": "alsa_input.usb-foo-00.input-B.2",
            "stable_id": "usb:foo",
            "discriminator": {"port": "input-B"},
        },
    ]
    result = resolve_node_for_stable_id("usb:foo", {"port": "input-B"}, devices)
    assert isinstance(result, Resolved)
    assert result.node_name == "alsa_input.usb-foo-00.input-B.2"


def test_resolver_ambiguous_when_discriminator_matches_multiple() -> None:
    # Two identical USB headsets on a hub — same VID:PID, same descriptor,
    # same port. Stable-id alone can't distinguish; discriminator can't
    # either. Resolver must fail closed.
    devices = [
        {
            "node_name": "alsa_input.usb-headset-00.input.2",
            "stable_id": "usb:headset",
            "discriminator": {"port": "input"},
        },
        {
            "node_name": "alsa_input.usb-headset-01.input.2",
            "stable_id": "usb:headset",
            "discriminator": {"port": "input"},
        },
    ]
    result = resolve_node_for_stable_id("usb:headset", {"port": "input"}, devices)
    assert isinstance(result, Ambiguous)
    assert len(result.candidates) == 2


def test_parse_pw_dump_extracts_stable_id_and_discriminator() -> None:
    # Verify the enumeration pipeline puts stable_id + discriminator on
    # each device entry so the resolver has what it needs.
    devices = parse_pw_dump_devices(_PW_DUMP_FIXTURE)
    sources = devices["sources"]
    assert len(sources) >= 1
    for entry in sources:
        assert "stable_id" in entry
        assert "discriminator" in entry
        # Every fixture node has a real device — stable_id non-empty.
        assert entry["stable_id"] != ""
        assert isinstance(entry["discriminator"], dict)
        assert entry["discriminator"]["direction"] == "source"


# ──────────────────────────────────────────────────────────────
# Phase 1.2 — reconcile_audio_routing path coverage
# (success-with-recovery + clear-on-recovery contract)
# ──────────────────────────────────────────────────────────────


import pytest

from meeting_scribe.audio.audio_routing import (
    SETTINGS_AUDIO_MEETING_MIC_DISCRIMINATOR,
    SETTINGS_AUDIO_MEETING_MIC_STABLE_ID,
    _reconcile_mic_with_stable_id,
)


@pytest.fixture
def _isolated_settings(tmp_path, monkeypatch):
    """Redirect ``settings.json`` to a tmp file so tests don't fight the
    real GB10 state. ``SETTINGS_OVERRIDE_FILE`` is a module-level Path
    constant; patch it directly + clear the mtime-keyed cache."""
    from meeting_scribe.runtime import state as _state
    from meeting_scribe.server_support import settings_store

    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(settings_store, "SETTINGS_OVERRIDE_FILE", settings_path)
    settings_store._settings_cache = None
    settings_store._settings_cache_mtime = 0.0
    _state.audio_route_status = "ok"
    _state.pending_admin_notifications = {}
    _state.server_mic_active = False
    yield settings_path
    settings_store._settings_cache = None
    settings_store._settings_cache_mtime = 0.0
    _state.audio_route_status = "ok"
    _state.pending_admin_notifications = {}
    _state.server_mic_active = False


def _make_reconcile_stub(captured: list) -> object:
    """Build an awaitable that records calls + flips ``server_mic_active``."""

    from meeting_scribe.runtime import state as _state

    async def _stub(*, mic_node: str, mic_active: bool) -> None:
        captured.append({"mic_node": mic_node, "mic_active": mic_active})
        _state.server_mic_active = bool(mic_active and mic_node)

    return _stub


@pytest.mark.asyncio
async def test_path_a_clears_failure_state_on_recovery(_isolated_settings, monkeypatch) -> None:
    """A device that came back under the same node name must clear any
    latched ``audio_route_status`` from a prior NotFound/Ambiguous run."""
    from meeting_scribe.runtime import state as _state
    from meeting_scribe.server_support import admin_notifications

    # Seed a stale failure state as if a prior reconcile had run.
    _state.audio_route_status = "unresolved"
    admin_notifications.put_notification("mic_unresolved", mic_node="alsa_input.foo")

    captured: list = []

    async def fake_enumerate() -> dict:
        return {
            "sources": [
                {
                    "node_name": "alsa_input.usb-foo-00.input-0.2",
                    "stable_id": "usb:foo",
                    "discriminator": {"port": "input-0"},
                }
            ],
            "sinks": [],
        }

    monkeypatch.setattr(
        "meeting_scribe.audio.audio_routing.enumerate_audio_devices",
        fake_enumerate,
    )

    result = await _reconcile_mic_with_stable_id(
        mic_node="alsa_input.usb-foo-00.input-0.2",
        mic_active=True,
        mic_stable_id="usb:foo",
        mic_discriminator={"port": "input-0"},
        reconcile_server_mic_fn=_make_reconcile_stub(captured),
    )

    assert result["status"] == "ok"
    assert _state.audio_route_status == "ok"
    # The stale mic_unresolved notification must be dismissed on recovery.
    active_kinds = [n["kind"] for n in admin_notifications.active_notifications()]
    assert "mic_unresolved" not in active_kinds


@pytest.mark.asyncio
async def test_path_b_rebinds_and_dismisses_stale_failure_notifications(
    _isolated_settings, monkeypatch
) -> None:
    """A successful rebind must also clear stale ambiguous/unresolved
    notifications even though it puts its own ``mic_rebound`` row."""
    from meeting_scribe.runtime import state as _state
    from meeting_scribe.server_support import admin_notifications

    _state.audio_route_status = "ambiguous"
    admin_notifications.put_notification(
        "mic_ambiguous", stable_id="usb:foo", candidates=[{"node_name": "a"}, {"node_name": "b"}]
    )

    captured: list = []

    async def fake_enumerate() -> dict:
        # mic_node stored as the old name; live device exposes the new ".3"
        # suffix. Stable_id resolves uniquely → Resolved → rebind.
        return {
            "sources": [
                {
                    "node_name": "alsa_input.usb-foo-00.input-0.3",
                    "stable_id": "usb:foo",
                    "discriminator": {"port": "input-0"},
                }
            ],
            "sinks": [],
        }

    monkeypatch.setattr(
        "meeting_scribe.audio.audio_routing.enumerate_audio_devices",
        fake_enumerate,
    )

    result = await _reconcile_mic_with_stable_id(
        mic_node="alsa_input.usb-foo-00.input-0.2",  # stale .2 suffix
        mic_active=True,
        mic_stable_id="usb:foo",
        mic_discriminator={"port": "input-0"},
        reconcile_server_mic_fn=_make_reconcile_stub(captured),
    )

    assert result["status"] == "rebound"
    assert result["rebound_from"] == "alsa_input.usb-foo-00.input-0.2"
    assert result["rebound_to"] == "alsa_input.usb-foo-00.input-0.3"
    assert _state.audio_route_status == "ok"
    active_kinds = {n["kind"] for n in admin_notifications.active_notifications()}
    # mic_rebound is the new informational row.
    assert "mic_rebound" in active_kinds
    # mic_ambiguous from before must be gone.
    assert "mic_ambiguous" not in active_kinds


@pytest.mark.asyncio
async def test_path_c_ambiguous_keeps_persisted_binding(_isolated_settings, monkeypatch) -> None:
    """Ambiguous resolution must NOT clear persisted mic_node / stable_id —
    a transient hub state shouldn't destroy the operator's configuration."""
    from meeting_scribe.runtime import state as _state
    from meeting_scribe.server_support import admin_notifications
    from meeting_scribe.server_support.settings_store import _save_settings_override

    captured: list = []

    async def fake_enumerate() -> dict:
        return {
            "sources": [
                {"node_name": "a", "stable_id": "usb:headset", "discriminator": {"port": "input"}},
                {"node_name": "b", "stable_id": "usb:headset", "discriminator": {"port": "input"}},
            ],
            "sinks": [],
        }

    monkeypatch.setattr(
        "meeting_scribe.audio.audio_routing.enumerate_audio_devices",
        fake_enumerate,
    )

    # Persist a starting binding so we can assert it survives.
    _save_settings_override(
        {
            "audio_meeting_mic_node": "old-node",
            SETTINGS_AUDIO_MEETING_MIC_STABLE_ID: "usb:headset",
            SETTINGS_AUDIO_MEETING_MIC_DISCRIMINATOR: {"port": "input"},
        }
    )

    result = await _reconcile_mic_with_stable_id(
        mic_node="old-node",
        mic_active=True,
        mic_stable_id="usb:headset",
        mic_discriminator={"port": "input"},
        reconcile_server_mic_fn=_make_reconcile_stub(captured),
    )

    assert result["status"] == "ambiguous"
    assert _state.audio_route_status == "ambiguous"
    active = admin_notifications.active_notifications()
    assert any(n["kind"] == "mic_ambiguous" for n in active)
    # Persisted binding still intact.
    from meeting_scribe.server_support.settings_store import _load_settings_override

    persisted = _load_settings_override()
    assert persisted["audio_meeting_mic_node"] == "old-node"
    assert persisted[SETTINGS_AUDIO_MEETING_MIC_STABLE_ID] == "usb:headset"


@pytest.mark.asyncio
async def test_path_d_not_found_keeps_persisted_for_transient_disconnect(
    _isolated_settings, monkeypatch
) -> None:
    """Transient USB disconnect: device gone, binding survives, reconciles
    on next call when device returns."""
    from meeting_scribe.runtime import state as _state
    from meeting_scribe.server_support import admin_notifications

    captured: list = []

    async def fake_enumerate_empty() -> dict:
        return {"sources": [], "sinks": []}

    monkeypatch.setattr(
        "meeting_scribe.audio.audio_routing.enumerate_audio_devices",
        fake_enumerate_empty,
    )

    result = await _reconcile_mic_with_stable_id(
        mic_node="alsa_input.usb-foo-00.input-0.2",
        mic_active=True,
        mic_stable_id="usb:foo",
        mic_discriminator={"port": "input-0"},
        reconcile_server_mic_fn=_make_reconcile_stub(captured),
    )

    assert result["status"] == "unresolved"
    assert _state.audio_route_status == "unresolved"
    assert any(n["kind"] == "mic_unresolved" for n in admin_notifications.active_notifications())


@pytest.mark.asyncio
async def test_path_e_capture_start_failed_when_reconcile_raises(
    _isolated_settings, monkeypatch
) -> None:
    """Routing resolved but the capture process refused to start. Must
    set ``capture_failed`` status and surface the detail."""
    from meeting_scribe.runtime import state as _state
    from meeting_scribe.server_support import admin_notifications

    async def fake_enumerate() -> dict:
        return {
            "sources": [
                {
                    "node_name": "alsa_input.usb-foo-00.input-0.2",
                    "stable_id": "usb:foo",
                    "discriminator": {"port": "input-0"},
                }
            ],
            "sinks": [],
        }

    monkeypatch.setattr(
        "meeting_scribe.audio.audio_routing.enumerate_audio_devices",
        fake_enumerate,
    )

    async def boom(*, mic_node: str, mic_active: bool) -> None:
        raise RuntimeError("pw-record exec failed: ENOENT")

    result = await _reconcile_mic_with_stable_id(
        mic_node="alsa_input.usb-foo-00.input-0.2",
        mic_active=True,
        mic_stable_id="usb:foo",
        mic_discriminator={"port": "input-0"},
        reconcile_server_mic_fn=boom,
    )

    assert result["status"] == "capture_failed"
    assert "ENOENT" in result.get("detail", "")
    assert _state.audio_route_status == "capture_failed"
    assert any(
        n["kind"] == "mic_capture_failed" for n in admin_notifications.active_notifications()
    )


@pytest.mark.asyncio
async def test_path_a_capture_start_failed_when_server_mic_active_does_not_flip(
    _isolated_settings, monkeypatch
) -> None:
    """``reconcile_server_mic`` returns without raising, but
    ``state.server_mic_active`` never flips to True. Should fall to path E."""
    from meeting_scribe.runtime import state as _state

    async def fake_enumerate() -> dict:
        return {
            "sources": [
                {
                    "node_name": "alsa_input.usb-foo-00.input-0.2",
                    "stable_id": "usb:foo",
                    "discriminator": None,
                }
            ],
            "sinks": [],
        }

    monkeypatch.setattr(
        "meeting_scribe.audio.audio_routing.enumerate_audio_devices",
        fake_enumerate,
    )

    # Stub the verify helper so the test doesn't wait the full 2 s.
    async def never_flips(*, timeout_s: float = 2.0) -> bool:
        return False

    monkeypatch.setattr(
        "meeting_scribe.audio.audio_routing._verify_server_mic_started",
        never_flips,
    )

    async def silent_reconcile(*, mic_node: str, mic_active: bool) -> None:
        _state.server_mic_active = False  # capture process refused to start

    result = await _reconcile_mic_with_stable_id(
        mic_node="alsa_input.usb-foo-00.input-0.2",
        mic_active=True,
        mic_stable_id="usb:foo",
        mic_discriminator=None,
        reconcile_server_mic_fn=silent_reconcile,
    )

    assert result["status"] == "capture_failed"
    assert _state.audio_route_status == "capture_failed"


@pytest.mark.asyncio
async def test_disabled_path_clears_failure_state_and_returns_ok(
    _isolated_settings, monkeypatch
) -> None:
    """``mic_active=False`` is a deliberate disable — it should clear any
    latched failure state, not preserve it."""
    from meeting_scribe.runtime import state as _state
    from meeting_scribe.server_support import admin_notifications

    _state.audio_route_status = "unresolved"
    admin_notifications.put_notification("mic_unresolved", mic_node="x")

    captured: list = []
    result = await _reconcile_mic_with_stable_id(
        mic_node="x",
        mic_active=False,
        mic_stable_id="usb:foo",
        mic_discriminator=None,
        reconcile_server_mic_fn=_make_reconcile_stub(captured),
    )

    assert result["status"] == "ok"
    assert _state.audio_route_status == "ok"
    assert not any(
        n["kind"] == "mic_unresolved" for n in admin_notifications.active_notifications()
    )
    # Reconcile was still called so the capture process stops cleanly.
    assert captured == [{"mic_node": "x", "mic_active": False}]
