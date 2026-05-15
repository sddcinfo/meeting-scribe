"""WAN-feature settings schema, validators, and CRUD helpers.

See ``docs/plans/wifi-wan-gateway.md`` Phase A for the contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def fresh_settings_store(tmp_path, monkeypatch):
    """Yield a freshly-pinned settings_store + the settings.json path."""
    settings_path = tmp_path / "settings.json"
    import meeting_scribe.server_support.settings_store as store

    monkeypatch.setattr(store, "SETTINGS_OVERRIDE_FILE", settings_path)
    monkeypatch.setattr(store, "_settings_cache", None)
    monkeypatch.setattr(store, "_settings_cache_mtime", 0.0)
    monkeypatch.setattr(store, "_legacy_migration_attempted", False)
    return store, settings_path


def _write_settings(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


_VALID_UUID = "12345678-1234-4567-8901-123456789abc"
_VALID_PROFILE = {
    "id": _VALID_UUID,
    "ssid": "Yunomotocho",
    "bssid": None,
    "psk_ref": "YUNOMOTOCHO_PSK",
    "regdomain": "JP",
    "last_seen": None,
}


# ─── Validators ────────────────────────────────────────────────


def test_is_valid_egress_mode_accepts_block_gateway_captive(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    assert store._is_valid_egress_mode("block") is True
    assert store._is_valid_egress_mode("gateway") is True
    assert store._is_valid_egress_mode("captive") is True


def test_is_valid_egress_mode_rejects_other(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    assert store._is_valid_egress_mode("off") is False
    assert store._is_valid_egress_mode("") is False
    assert store._is_valid_egress_mode(None) is False
    assert store._is_valid_egress_mode(0) is False


def test_is_valid_egress_mode_source_accepts_default_operator(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    assert store._is_valid_egress_mode_source("default") is True
    assert store._is_valid_egress_mode_source("operator") is True


def test_is_valid_egress_mode_source_rejects_other(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    for bad in ("", "user", None, 0, True):
        assert store._is_valid_egress_mode_source(bad) is False


def test_uuid4_validator_accepts_canonical(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    assert store._is_valid_uuid4_str(_VALID_UUID) is True


def test_uuid4_validator_rejects_uuid1_and_truncated(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    # uuid1 form (version digit != 4)
    assert store._is_valid_uuid4_str("12345678-1234-1234-8901-123456789abc") is False
    # truncated
    assert store._is_valid_uuid4_str("12345678-1234-4567-8901-123456789ab") is False
    # uppercase (we require lowercase canonical)
    assert store._is_valid_uuid4_str(_VALID_UUID.upper()) is False
    # garbage
    assert store._is_valid_uuid4_str("not-a-uuid") is False


def test_psk_ref_validator(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    assert store._is_valid_psk_ref("YUNOMOTOCHO_PSK") is True
    assert store._is_valid_psk_ref("GH_TOKEN") is True
    assert store._is_valid_psk_ref("KEY_WITH_DIGITS_123") is True
    # Disallowed: lowercase, leading digit, spaces, length cap
    assert store._is_valid_psk_ref("yunomotocho_psk") is False
    assert store._is_valid_psk_ref("1NOT_LEADING_DIGIT") is False
    assert store._is_valid_psk_ref("HAS SPACE") is False
    assert store._is_valid_psk_ref("") is False
    assert store._is_valid_psk_ref("X" * 65) is False


def test_bssid_validator(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    assert store._is_valid_bssid("aa:bb:cc:dd:ee:ff") is True
    assert store._is_valid_bssid("AA:BB:CC:DD:EE:FF") is True
    assert store._is_valid_bssid("a:b:c:d:e:f") is False
    assert store._is_valid_bssid("aabbccddeeff") is False
    assert store._is_valid_bssid("") is False


def test_ssid_validator(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    assert store._is_valid_ssid("Yunomotocho") is True
    assert store._is_valid_ssid("a") is True
    assert store._is_valid_ssid("x" * 32) is True
    # Exceeds 32 octets
    assert store._is_valid_ssid("x" * 33) is False
    # Empty
    assert store._is_valid_ssid("") is False
    # Embedded null
    assert store._is_valid_ssid("foo\x00bar") is False
    # Non-ASCII multi-byte chars inside the 32-octet UTF-8 budget
    assert store._is_valid_ssid("ゆのもとちょう") is True  # 21 utf-8 bytes


def test_wan_profile_validator_accepts_minimal(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    assert store._is_valid_wan_profile(_VALID_PROFILE) is True


def test_wan_profile_validator_rejects_missing_required_fields(fresh_settings_store) -> None:
    """``id`` and ``ssid`` are mandatory; ``psk_ref`` is optional (OPEN
    networks) — exercised separately by ``test_wan_profile_accepts_missing_psk_ref``."""
    store, _ = fresh_settings_store
    for missing in ("id", "ssid"):
        bad = dict(_VALID_PROFILE)
        del bad[missing]
        assert store._is_valid_wan_profile(bad) is False, f"missing {missing} should fail"


def test_wan_profile_validator_rejects_bad_bssid(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    bad = dict(_VALID_PROFILE)
    bad["bssid"] = "not-a-mac"
    assert store._is_valid_wan_profile(bad) is False


def test_wan_profile_validator_rejects_bad_psk_ref(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    bad = dict(_VALID_PROFILE)
    bad["psk_ref"] = "lower_case"
    assert store._is_valid_wan_profile(bad) is False


# ─── Readers + writers (file round-trips) ─────────────────────


def test_egress_mode_defaults_to_block_when_unset(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    assert store._effective_wan_egress_mode() == "block"


def test_egress_mode_round_trip(fresh_settings_store) -> None:
    store, path = fresh_settings_store
    store._set_wan_egress_mode("gateway")
    assert store._effective_wan_egress_mode() == "gateway"
    on_disk = json.loads(path.read_text())
    assert on_disk[store.SETTINGS_WAN_EGRESS_MODE] == "gateway"


def test_set_egress_mode_rejects_invalid(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    with pytest.raises(ValueError):
        store._set_wan_egress_mode("off")


def test_egress_mode_ignores_corrupt_on_disk_value(fresh_settings_store) -> None:
    store, path = fresh_settings_store
    _write_settings(path, {store.SETTINGS_WAN_EGRESS_MODE: "freeforall"})
    # Defensive read: invalid persisted value falls back to "block".
    assert store._effective_wan_egress_mode() == "block"


def test_egress_mode_source_defaults_to_default_when_unset(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    assert store._wan_egress_mode_source() == "default"


def test_set_egress_mode_stamps_operator_source_by_default(fresh_settings_store) -> None:
    """External callers (CLI/REST/UI) end up marked as ``operator`` so the
    migration ladder never overwrites their pick."""
    store, path = fresh_settings_store
    store._set_wan_egress_mode("captive")
    assert store._effective_wan_egress_mode() == "captive"
    assert store._wan_egress_mode_source() == "operator"
    on_disk = json.loads(path.read_text())
    assert on_disk[store.SETTINGS_WAN_EGRESS_MODE_SOURCE] == "operator"


def test_set_egress_mode_with_default_source_stays_default(fresh_settings_store) -> None:
    """The migration ladder uses ``source="default"`` so a future
    migration tick can still operate. External callers should NOT pass
    this — the test just locks the API contract."""
    store, _ = fresh_settings_store
    store._set_wan_egress_mode("captive", source="default")
    assert store._wan_egress_mode_source() == "default"


def test_set_egress_mode_rejects_invalid_source(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    with pytest.raises(ValueError):
        store._set_wan_egress_mode("captive", source="admin")


def test_egress_mode_source_ignores_corrupt_on_disk_value(fresh_settings_store) -> None:
    store, path = fresh_settings_store
    _write_settings(path, {store.SETTINGS_WAN_EGRESS_MODE_SOURCE: "junk"})
    assert store._wan_egress_mode_source() == "default"


def test_save_and_load_wan_profile(fresh_settings_store) -> None:
    store, path = fresh_settings_store
    store._save_wan_profile(_VALID_PROFILE)
    loaded = store._find_wan_profile_by_id(_VALID_UUID)
    assert loaded is not None
    assert loaded["ssid"] == "Yunomotocho"
    assert loaded["psk_ref"] == "YUNOMOTOCHO_PSK"
    # Round-trip via the persisted file.
    on_disk = json.loads(path.read_text())
    assert any(p["id"] == _VALID_UUID for p in on_disk[store.SETTINGS_WAN_PROFILES])


def test_save_wan_profile_rejects_malformed(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    bad = dict(_VALID_PROFILE)
    bad["id"] = "not-a-uuid"
    with pytest.raises(ValueError):
        store._save_wan_profile(bad)


def test_save_wan_profile_overwrites_by_id(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    store._save_wan_profile(_VALID_PROFILE)
    updated = dict(_VALID_PROFILE)
    updated["ssid"] = "RenamedSSID"
    store._save_wan_profile(updated)
    all_profiles = store._load_wan_profiles()
    assert len(all_profiles) == 1
    assert all_profiles[0]["ssid"] == "RenamedSSID"


def test_delete_wan_profile_removes(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    store._save_wan_profile(_VALID_PROFILE)
    assert store._delete_wan_profile(_VALID_UUID) is True
    assert store._find_wan_profile_by_id(_VALID_UUID) is None
    # second delete is a no-op
    assert store._delete_wan_profile(_VALID_UUID) is False


def test_load_wan_profiles_filters_malformed_silently(fresh_settings_store) -> None:
    """Hand-edited settings.json with one bad entry should not poison reads."""
    store, path = fresh_settings_store
    _write_settings(
        path,
        {
            store.SETTINGS_WAN_PROFILES: [
                _VALID_PROFILE,
                {"id": "not-uuid", "ssid": "Bad"},  # malformed
                {"id": "abcdef01-2345-4789-89ab-cdef01234567", "ssid": "X", "psk_ref": "X_PSK"},
            ]
        },
    )
    profiles = store._load_wan_profiles()
    assert len(profiles) == 2  # bad one dropped
    assert {p["ssid"] for p in profiles} == {"Yunomotocho", "X"}


def test_find_wan_profiles_by_ssid_returns_all_matches(fresh_settings_store) -> None:
    """Multi-BSSID environments may save two profiles under the same SSID."""
    store, _ = fresh_settings_store
    store._save_wan_profile(_VALID_PROFILE)
    second = dict(_VALID_PROFILE)
    second["id"] = "abcdef01-2345-4789-89ab-cdef01234567"
    second["bssid"] = "aa:bb:cc:dd:ee:ff"
    store._save_wan_profile(second)
    matches = store._find_wan_profiles_by_ssid("Yunomotocho")
    assert len(matches) == 2
    assert {p["id"] for p in matches} == {_VALID_UUID, second["id"]}


def test_set_active_profile_rejects_unknown_id(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    with pytest.raises(ValueError):
        store._set_wan_active_profile_id("abcdef01-2345-4789-89ab-cdef01234567")


def test_set_active_profile_accepts_existing(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    store._save_wan_profile(_VALID_PROFILE)
    store._set_wan_active_profile_id(_VALID_UUID)
    assert store._effective_wan_active_profile_id() == _VALID_UUID


def test_set_active_profile_can_clear(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    store._save_wan_profile(_VALID_PROFILE)
    store._set_wan_active_profile_id(_VALID_UUID)
    store._set_wan_active_profile_id(None)
    assert store._effective_wan_active_profile_id() is None


def test_active_profile_id_ignores_garbage_on_disk(fresh_settings_store) -> None:
    store, path = fresh_settings_store
    _write_settings(path, {store.SETTINGS_WAN_ACTIVE_PROFILE_ID: "not-a-uuid"})
    assert store._effective_wan_active_profile_id() is None


# ─── Wired-state persistence ───────────────────────────────────


def test_wired_state_round_trip(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    name, metric = store._load_wan_wired_state()
    assert name is None and metric is None
    store._save_wan_wired_state("Wired connection 3", 600)
    name, metric = store._load_wan_wired_state()
    assert name == "Wired connection 3"
    assert metric == 600


def test_wired_state_accepts_unknown_prior_metric(fresh_settings_store) -> None:
    """NM reports route-metric=-1 when unset; we represent that as None."""
    store, _ = fresh_settings_store
    store._save_wan_wired_state("Wired connection 3", None)
    name, metric = store._load_wan_wired_state()
    assert name == "Wired connection 3"
    assert metric is None


def test_wired_state_rejects_empty_name(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    with pytest.raises(ValueError):
        store._save_wan_wired_state("", 100)


# ─── Captive state cache ───────────────────────────────────────


def test_captive_state_empty_by_default(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    assert store._load_wan_captive_state() == {}


def test_captive_state_round_trip_per_iface(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    store._save_wan_captive_state(
        "wlan_sta",
        connectivity="portal",
        portal_url="https://venue.example/login",
        probed_at="2026-05-12T10:00:00Z",
    )
    state = store._load_wan_captive_state()
    assert state["wlan_sta"]["connectivity"] == "portal"
    assert state["wlan_sta"]["portal_url"] == "https://venue.example/login"
    assert state["wlan_sta"]["probed_at"] == "2026-05-12T10:00:00Z"


def test_captive_state_normalizes_corrupt_entries(fresh_settings_store) -> None:
    """A hand-edited bad portal_url should not raise — defensive default."""
    store, path = fresh_settings_store
    _write_settings(
        path,
        {
            store.SETTINGS_WAN_CAPTIVE_STATE: {
                "wlan_sta": {"connectivity": "full", "portal_url": 42, "probed_at": None},
            }
        },
    )
    state = store._load_wan_captive_state()
    assert state["wlan_sta"]["connectivity"] == "full"
    assert state["wlan_sta"]["portal_url"] is None


# ─── Band preference ──────────────────────────────────────────


def test_is_valid_wan_band_accepts_codes(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    for ok in ("auto", "a", "bg"):
        assert store._is_valid_wan_band(ok) is True


def test_is_valid_wan_band_rejects_other(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    for bad in ("", "5", "2.4", "any", "ax", None, 5):
        assert store._is_valid_wan_band(bad) is False


def test_wan_profile_accepts_band_field(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    base = dict(_VALID_PROFILE)
    for ok in ("auto", "a", "bg"):
        candidate = dict(base, band=ok)
        assert store._is_valid_wan_profile(candidate) is True


def test_wan_profile_rejects_invalid_band(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    base = dict(_VALID_PROFILE)
    bad = dict(base, band="anyband")
    assert store._is_valid_wan_profile(bad) is False


def test_wan_profile_band_optional_for_legacy(fresh_settings_store) -> None:
    """Older settings files predate the band field — they must still load."""
    store, _ = fresh_settings_store
    legacy = {k: v for k, v in _VALID_PROFILE.items() if k != "band"}
    assert "band" not in legacy
    assert store._is_valid_wan_profile(legacy) is True


# ─── Open network profile (psk_ref optional) ──────────────────


def test_wan_profile_accepts_missing_psk_ref(fresh_settings_store) -> None:
    """OPEN networks have no PSK. psk_ref must be optional (None or absent)."""
    store, _ = fresh_settings_store
    minus_psk = {k: v for k, v in _VALID_PROFILE.items() if k != "psk_ref"}
    assert store._is_valid_wan_profile(minus_psk) is True
    none_psk = dict(_VALID_PROFILE, psk_ref=None)
    assert store._is_valid_wan_profile(none_psk) is True


def test_wan_profile_rejects_bad_psk_ref_even_when_present(fresh_settings_store) -> None:
    """A psk_ref that is present but malformed must still be rejected."""
    store, _ = fresh_settings_store
    bad = dict(_VALID_PROFILE, psk_ref="lower_case_invalid")
    assert store._is_valid_wan_profile(bad) is False


def test_save_open_profile_round_trips(fresh_settings_store) -> None:
    store, _ = fresh_settings_store
    open_profile = dict(_VALID_PROFILE, ssid="DellEfficiency_Guest", psk_ref=None)
    store._save_wan_profile(open_profile)
    loaded = store._find_wan_profile_by_id(_VALID_UUID)
    assert loaded is not None
    assert loaded["psk_ref"] is None
