"""Phase D — ``meeting-scribe wifi wan ...`` Click CLI surface.

Tests use the Click ``CliRunner`` with subprocess/IO seams stubbed.
Profiles are persisted to a per-test ``settings.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

_VALID_UUID = "12345678-1234-4567-8901-123456789abc"
_VALID_UUID_2 = "abcdef01-2345-4789-89ab-cdef01234567"


def _build_profile(profile_id: str = _VALID_UUID, **overrides: Any) -> dict:
    base = {
        "id": profile_id,
        "ssid": "Yunomotocho",
        "bssid": None,
        "psk_ref": "YUNOMOTOCHO_PSK",
        "regdomain": None,
        "last_seen": None,
    }
    base.update(overrides)
    return base


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch):
    """Fresh settings_store + a clean CliRunner."""
    settings_path = tmp_path / "settings.json"
    import meeting_scribe.server_support.settings_store as store

    monkeypatch.setattr(store, "SETTINGS_OVERRIDE_FILE", settings_path)
    monkeypatch.setattr(store, "_settings_cache", None)
    monkeypatch.setattr(store, "_settings_cache_mtime", 0.0)
    monkeypatch.setattr(store, "_legacy_migration_attempted", False)
    return CliRunner(), store


def _invoke(runner: CliRunner, *args: str):
    from meeting_scribe.cli import cli

    return runner.invoke(cli, list(args), catch_exceptions=False)


# ─── profiles add / ls / rm / set-active ───────────────────────


def test_profiles_add_validates_psk_ref_by_default(cli_env, monkeypatch) -> None:
    runner, _ = cli_env
    # PSK ref does NOT exist.
    monkeypatch.setattr("meeting_scribe.server_support.secrets.psk_ref_exists", lambda ref: False)
    result = _invoke(
        runner,
        "wifi",
        "wan",
        "profiles",
        "add",
        "--ssid",
        "Yunomotocho",
        "--psk-ref",
        "BOGUS_PSK",
    )
    assert result.exit_code != 0
    assert "not found in the age store" in result.output


def test_profiles_add_bypass_validation_flag(cli_env, monkeypatch) -> None:
    runner, store = cli_env
    monkeypatch.setattr("meeting_scribe.server_support.secrets.psk_ref_exists", lambda ref: False)
    result = _invoke(
        runner,
        "wifi",
        "wan",
        "profiles",
        "add",
        "--ssid",
        "Yunomotocho",
        "--psk-ref",
        "YUNOMOTOCHO_PSK",
        "--no-verify-psk-ref",
    )
    assert result.exit_code == 0, result.output
    profiles = store._load_wan_profiles()
    assert len(profiles) == 1
    assert profiles[0]["ssid"] == "Yunomotocho"
    # ID printed full, never truncated.
    assert profiles[0]["id"] in result.output
    assert "…" not in result.output  # no ellipsis


def test_profiles_ls_shows_full_ids(cli_env, monkeypatch) -> None:
    runner, store = cli_env
    store._save_wan_profile(_build_profile())
    store._save_wan_profile(_build_profile(_VALID_UUID_2, bssid="aa:bb:cc:dd:ee:ff"))
    result = _invoke(runner, "wifi", "wan", "profiles", "ls")
    assert result.exit_code == 0, result.output
    assert _VALID_UUID in result.output
    assert _VALID_UUID_2 in result.output
    # No truncation.
    assert "…" not in result.output


def test_profiles_rm_removes_by_id(cli_env) -> None:
    runner, store = cli_env
    store._save_wan_profile(_build_profile())
    result = _invoke(runner, "wifi", "wan", "profiles", "rm", "--id", _VALID_UUID)
    assert result.exit_code == 0, result.output
    assert store._load_wan_profiles() == []


def test_profiles_rm_unknown_id_errors(cli_env) -> None:
    runner, _ = cli_env
    result = _invoke(runner, "wifi", "wan", "profiles", "rm", "--id", _VALID_UUID)
    assert result.exit_code != 0
    assert "no profile" in result.output


def test_profiles_set_active(cli_env) -> None:
    runner, store = cli_env
    store._save_wan_profile(_build_profile())
    result = _invoke(runner, "wifi", "wan", "profiles", "set-active", "--id", _VALID_UUID)
    assert result.exit_code == 0
    assert store._effective_wan_active_profile_id() == _VALID_UUID


# ─── wan up — id / ssid / ambiguity ──────────────────────────


def test_wan_up_by_id(cli_env, monkeypatch) -> None:
    runner, store = cli_env
    store._save_wan_profile(_build_profile())

    called: dict[str, Any] = {}

    async def _stub_up(profile_id: str) -> None:
        called["id"] = profile_id

    monkeypatch.setattr("meeting_scribe.wifi_wan.wan_up", _stub_up)
    result = _invoke(runner, "wifi", "wan", "up", "--id", _VALID_UUID)
    assert result.exit_code == 0, result.output
    assert called["id"] == _VALID_UUID


def test_wan_up_by_ssid_unique_succeeds(cli_env, monkeypatch) -> None:
    runner, store = cli_env
    store._save_wan_profile(_build_profile())

    called: dict[str, Any] = {}

    async def _stub_up(profile_id: str) -> None:
        called["id"] = profile_id

    monkeypatch.setattr("meeting_scribe.wifi_wan.wan_up", _stub_up)
    result = _invoke(runner, "wifi", "wan", "up", "--ssid", "Yunomotocho")
    assert result.exit_code == 0, result.output
    assert called["id"] == _VALID_UUID


def test_wan_up_by_ssid_ambiguous_errors_with_full_ids(cli_env, monkeypatch) -> None:
    runner, store = cli_env
    store._save_wan_profile(_build_profile())
    store._save_wan_profile(_build_profile(_VALID_UUID_2, bssid="aa:bb:cc:dd:ee:ff"))

    async def _stub_up(profile_id: str) -> None:
        raise AssertionError("must not be called")

    monkeypatch.setattr("meeting_scribe.wifi_wan.wan_up", _stub_up)
    result = _invoke(runner, "wifi", "wan", "up", "--ssid", "Yunomotocho")
    assert result.exit_code != 0
    assert "ambiguous" in result.output
    # Both full ids printed for the operator to pick from.
    assert _VALID_UUID in result.output
    assert _VALID_UUID_2 in result.output


def test_wan_up_missing_id_and_ssid_errors(cli_env) -> None:
    runner, _ = cli_env
    result = _invoke(runner, "wifi", "wan", "up")
    assert result.exit_code != 0
    assert "specify --id or --ssid" in result.output


def test_wan_up_unknown_id_errors(cli_env, monkeypatch) -> None:
    runner, _ = cli_env

    async def _stub_up(profile_id: str) -> None:
        raise AssertionError("must not be called")

    monkeypatch.setattr("meeting_scribe.wifi_wan.wan_up", _stub_up)
    result = _invoke(runner, "wifi", "wan", "up", "--id", _VALID_UUID)
    assert result.exit_code != 0
    assert "no wan profile" in result.output


# ─── wan down ────────────────────────────────────────────────


def test_wan_down_invokes_wifi_wan(cli_env, monkeypatch) -> None:
    runner, _ = cli_env
    called = {"count": 0}

    async def _stub_down() -> None:
        called["count"] += 1

    monkeypatch.setattr("meeting_scribe.wifi_wan.wan_down", _stub_down)
    result = _invoke(runner, "wifi", "wan", "down")
    assert result.exit_code == 0, result.output
    assert called["count"] == 1


# ─── wan status ──────────────────────────────────────────────


def test_wan_status_renders_per_iface(cli_env, monkeypatch) -> None:
    runner, _ = cli_env

    async def _stub_status() -> dict:
        return {
            "wired": {
                "iface": "enP7s7",
                "up": True,
                "lease": "192.168.8.153",
                "profile_name": "Wired connection 3",
                "route_metric": 100,
                "default_route": True,
            },
            "wifi": {
                "iface": "wlan_sta",
                "up": True,
                "lease": "10.0.0.42",
                "profile_name": "meeting-scribe-sta-" + _VALID_UUID,
                "ssid": "Yunomotocho",
                "bssid": "aa:bb:cc:dd:ee:ff",
                "signal_dbm": -55,
                "route_metric": 600,
                "default_route": False,
                "connectivity": "portal",
                "portal_url": "https://venue.example/login",
            },
            "active_default": "enP7s7",
            "egress_mode": "gateway",
        }

    monkeypatch.setattr("meeting_scribe.wifi_wan.wan_status", _stub_status)
    result = _invoke(runner, "wifi", "wan", "status")
    assert result.exit_code == 0, result.output
    # Both ifaces present; both leases shown.
    assert "192.168.8.153" in result.output
    assert "10.0.0.42" in result.output
    # Per-iface captive state surfaced.
    assert "portal" in result.output
    assert "https://venue.example/login" in result.output
    # Active default + egress mode.
    assert "enP7s7" in result.output
    assert "gateway" in result.output
    # Full ssid not truncated.
    assert "Yunomotocho" in result.output


# ─── wan scan ────────────────────────────────────────────────


def test_wan_scan_empty(cli_env, monkeypatch) -> None:
    runner, _ = cli_env

    async def _stub_scan():
        return []

    monkeypatch.setattr("meeting_scribe.wifi_wan.scan_upstream", _stub_scan)
    result = _invoke(runner, "wifi", "wan", "scan")
    assert result.exit_code == 0
    assert "no APs visible" in result.output


def test_wan_scan_lists_results(cli_env, monkeypatch) -> None:
    runner, _ = cli_env
    from meeting_scribe import wifi_sta

    async def _stub_scan():
        return [
            wifi_sta.ScanEntry(
                bssid="aa:bb:cc:dd:ee:ff",
                ssid="Yunomotocho",
                channel=36,
                signal_dbm=-55.0,
                rsn_present=True,
            ),
            wifi_sta.ScanEntry(
                bssid="11:22:33:44:55:66",
                ssid="OpenNet",
                channel=6,
                signal_dbm=-70.0,
                rsn_present=False,
            ),
        ]

    monkeypatch.setattr("meeting_scribe.wifi_wan.scan_upstream", _stub_scan)
    # Consolidated default: SSID + band + count, no BSSID.
    result = _invoke(runner, "wifi", "wan", "scan")
    assert result.exit_code == 0, result.output
    assert "Yunomotocho" in result.output
    assert "OpenNet" in result.output
    assert "5 GHz" in result.output
    assert "2.4 GHz" in result.output
    assert "WPA" in result.output
    assert "OPEN" in result.output
    # BSSID is deliberately hidden in the consolidated view.
    assert "aa:bb:cc:dd:ee:ff" not in result.output

    # --raw exposes the BSS-level detail for advanced operators.
    raw_result = _invoke(runner, "wifi", "wan", "scan", "--raw")
    assert raw_result.exit_code == 0, raw_result.output
    assert "aa:bb:cc:dd:ee:ff" in raw_result.output
    assert "11:22:33:44:55:66" in raw_result.output


# ─── band preference ──────────────────────────────────────────


def test_profiles_add_default_band_is_auto(cli_env, monkeypatch) -> None:
    runner, store = cli_env
    monkeypatch.setattr("meeting_scribe.server_support.secrets.psk_ref_exists", lambda ref: True)
    result = _invoke(
        runner,
        "wifi",
        "wan",
        "profiles",
        "add",
        "--ssid",
        "Yunomotocho",
        "--psk-ref",
        "YUNOMOTOCHO_PSK",
    )
    assert result.exit_code == 0, result.output
    profile = store._load_wan_profiles()[0]
    assert profile["band"] == "auto"


def test_profiles_add_band_5_maps_to_a(cli_env, monkeypatch) -> None:
    runner, store = cli_env
    monkeypatch.setattr("meeting_scribe.server_support.secrets.psk_ref_exists", lambda ref: True)
    result = _invoke(
        runner,
        "wifi",
        "wan",
        "profiles",
        "add",
        "--ssid",
        "Yunomotocho",
        "--psk-ref",
        "YUNOMOTOCHO_PSK",
        "--band",
        "5",
    )
    assert result.exit_code == 0, result.output
    assert store._load_wan_profiles()[0]["band"] == "a"


def test_profiles_add_band_2_4_maps_to_bg(cli_env, monkeypatch) -> None:
    runner, store = cli_env
    monkeypatch.setattr("meeting_scribe.server_support.secrets.psk_ref_exists", lambda ref: True)
    result = _invoke(
        runner,
        "wifi",
        "wan",
        "profiles",
        "add",
        "--ssid",
        "Yunomotocho",
        "--psk-ref",
        "YUNOMOTOCHO_PSK",
        "--band",
        "2.4",
    )
    assert result.exit_code == 0, result.output
    assert store._load_wan_profiles()[0]["band"] == "bg"


def test_profiles_ls_shows_band_column(cli_env) -> None:
    runner, store = cli_env
    store._save_wan_profile(
        {
            "id": _VALID_UUID,
            "ssid": "Yunomotocho",
            "bssid": None,
            "band": "a",
            "psk_ref": "YUNOMOTOCHO_PSK",
            "regdomain": None,
            "last_seen": None,
        }
    )
    result = _invoke(runner, "wifi", "wan", "profiles", "ls")
    assert result.exit_code == 0, result.output
    assert "Band" in result.output
    assert "5" in result.output  # 5 GHz selection


# ─── Open networks (no PSK_REF) ─────────────────────────────


def test_profiles_add_open_succeeds_without_psk_ref(cli_env) -> None:
    runner, store = cli_env
    result = _invoke(
        runner,
        "wifi",
        "wan",
        "profiles",
        "add",
        "--ssid",
        "DellEfficiency_Guest",
        "--open",
    )
    assert result.exit_code == 0, result.output
    profiles = store._load_wan_profiles()
    assert len(profiles) == 1
    assert profiles[0]["psk_ref"] is None
    assert profiles[0]["ssid"] == "DellEfficiency_Guest"
    assert "OPEN" in result.output


def test_profiles_add_rejects_open_with_psk_ref(cli_env) -> None:
    """--open and --psk-ref are mutually exclusive (foot-gun prevention)."""
    runner, _ = cli_env
    result = _invoke(
        runner,
        "wifi",
        "wan",
        "profiles",
        "add",
        "--ssid",
        "Foo",
        "--open",
        "--psk-ref",
        "Y_PSK",
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_profiles_add_rejects_neither_psk_nor_open(cli_env) -> None:
    """Missing both must error so a forgotten psk_ref isn't silently OPEN."""
    runner, _ = cli_env
    result = _invoke(
        runner,
        "wifi",
        "wan",
        "profiles",
        "add",
        "--ssid",
        "Foo",
    )
    assert result.exit_code != 0
    assert "--psk-ref" in result.output or "open" in result.output


def test_profiles_ls_renders_open_label(cli_env) -> None:
    runner, store = cli_env
    store._save_wan_profile(
        {
            "id": _VALID_UUID,
            "ssid": "DellEfficiency_Guest",
            "bssid": None,
            "band": "auto",
            "psk_ref": None,
            "regdomain": None,
            "last_seen": None,
        }
    )
    result = _invoke(runner, "wifi", "wan", "profiles", "ls")
    assert result.exit_code == 0, result.output
    assert "(OPEN)" in result.output
