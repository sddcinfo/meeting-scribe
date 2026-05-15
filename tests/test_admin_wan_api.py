"""Phase E — ``/api/admin/wan/...`` REST surface.

Validates admin gating, profile CRUD shape, PSK secrecy (never returned
in any response), per-iface status passthrough.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient

_VALID_UUID = "12345678-1234-4567-8901-123456789abc"
_VALID_UUID_2 = "abcdef01-2345-4789-89ab-cdef01234567"


def _build_app(*, admin_ok: bool) -> FastAPI:
    """FastAPI app with admin_wan routes registered + admin gate stubbed."""
    from meeting_scribe.routes import admin_wan as admin_wan_mod

    if admin_ok:
        admin_wan_mod._require_admin_response = lambda req: None
    else:
        admin_wan_mod._require_admin_response = lambda req: JSONResponse(
            {"error": "admin required"}, status_code=403
        )
    app = FastAPI()
    app.include_router(admin_wan_mod.router)
    return app


@pytest.fixture
def admin_app(tmp_path: Path, monkeypatch):
    """Admin-OK app + per-test settings_store pinned to tmp_path."""
    import meeting_scribe.server_support.settings_store as store

    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(store, "SETTINGS_OVERRIDE_FILE", settings_path)
    monkeypatch.setattr(store, "_settings_cache", None)
    monkeypatch.setattr(store, "_settings_cache_mtime", 0.0)
    monkeypatch.setattr(store, "_legacy_migration_attempted", False)
    return _build_app(admin_ok=True), store


# ─── Admin gating ─────────────────────────────────────────────


def test_all_routes_require_admin() -> None:
    app = _build_app(admin_ok=False)
    with TestClient(app, base_url="http://test") as client:
        for verb, path in (
            ("get", "/api/admin/wan/status"),
            ("get", "/api/admin/wan/profiles"),
            ("get", "/api/admin/wan/scan"),
            ("post", "/api/admin/wan/up"),
            ("post", "/api/admin/wan/down"),
            ("delete", f"/api/admin/wan/profiles/{_VALID_UUID}"),
            ("post", f"/api/admin/wan/profiles/{_VALID_UUID}/set-active"),
        ):
            fn = getattr(client, verb)
            resp = fn(path) if verb in ("get", "delete") else fn(path, json={})
            assert resp.status_code == 403, f"{verb.upper()} {path} not gated"


# ─── status / scan ────────────────────────────────────────────


def test_status_calls_wifi_wan(admin_app) -> None:
    app, _ = admin_app
    sample = {"wired": {"up": True}, "wifi": {"connectivity": "full"}, "egress_mode": "block"}
    with (
        patch("meeting_scribe.wifi_wan.wan_status", new=AsyncMock(return_value=sample)),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.get("/api/admin/wan/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["wired"]["up"] is True
    assert body["wifi"]["connectivity"] == "full"


def test_scan_passes_through_entries(admin_app) -> None:
    app, _ = admin_app
    from meeting_scribe.wifi_sta import ScanEntry

    entries = [
        ScanEntry(
            bssid="aa:bb:cc:dd:ee:ff",
            ssid="Yunomotocho",
            channel=36,
            signal_dbm=-55.0,
            rsn_present=True,
        ),
    ]
    with (
        patch("meeting_scribe.wifi_wan.scan_upstream", new=AsyncMock(return_value=entries)),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.get("/api/admin/wan/scan")
    assert resp.status_code == 200
    body = resp.json()
    assert body["entries"][0]["ssid"] == "Yunomotocho"
    assert body["entries"][0]["security"] == "wpa"


# ─── profiles CRUD ────────────────────────────────────────────


def test_profiles_create_returns_full_id_no_psk_field(admin_app) -> None:
    """The PSK plaintext must NEVER appear in any response."""
    app, store = admin_app
    with (
        patch("meeting_scribe.server_support.secrets.psk_ref_exists", lambda ref: True),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post(
            "/api/admin/wan/profiles",
            json={"ssid": "Yunomotocho", "psk_ref": "YUNOMOTOCHO_PSK"},
        )
    assert resp.status_code == 201, resp.text
    profile = resp.json()["profile"]
    assert profile["ssid"] == "Yunomotocho"
    assert profile["psk_ref"] == "YUNOMOTOCHO_PSK"
    assert "psk" not in profile  # never plaintext
    # Full id round-trips.
    assert len(profile["id"]) == 36
    # Persisted.
    assert store._find_wan_profile_by_id(profile["id"]) is not None


def test_profiles_create_rejects_invalid_ssid(admin_app) -> None:
    app, _ = admin_app
    with (
        patch("meeting_scribe.server_support.secrets.psk_ref_exists", lambda ref: True),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post(
            "/api/admin/wan/profiles",
            json={"ssid": "", "psk_ref": "YUNOMOTOCHO_PSK"},
        )
    assert resp.status_code == 400
    assert "ssid" in resp.json()["error"]


def test_profiles_create_rejects_invalid_psk_ref(admin_app) -> None:
    app, _ = admin_app
    with (
        patch("meeting_scribe.server_support.secrets.psk_ref_exists", lambda ref: True),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post(
            "/api/admin/wan/profiles",
            json={"ssid": "Y", "psk_ref": "lower_case_invalid"},
        )
    assert resp.status_code == 400
    assert "psk_ref" in resp.json()["error"]


def test_profiles_create_validates_psk_ref_exists(admin_app) -> None:
    """422 when psk_ref looks syntactically valid but isn't in the age store."""
    app, _ = admin_app
    with (
        patch(
            "meeting_scribe.server_support.secrets.psk_ref_exists",
            lambda ref: False,
        ),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post(
            "/api/admin/wan/profiles",
            json={"ssid": "Y", "psk_ref": "GENUINE_LOOKING_PSK"},
        )
    assert resp.status_code == 422
    assert resp.json()["psk_ref"] == "GENUINE_LOOKING_PSK"


def test_profiles_list_omits_psk_field_always(admin_app) -> None:
    """No PSK leakage even if settings.json was hand-edited with a `psk` key."""
    app, _store = admin_app
    profile = {
        "id": _VALID_UUID,
        "ssid": "Yunomotocho",
        "bssid": None,
        "psk_ref": "YUNOMOTOCHO_PSK",
        "regdomain": None,
        "last_seen": None,
        # An attacker writes plaintext here; the public projection must omit it.
        "psk": "should-never-be-returned",
    }
    # Bypass the validator's strictness — write directly.
    from meeting_scribe.server_support.settings_store import (
        SETTINGS_WAN_PROFILES,
        _save_settings_override,
    )

    _save_settings_override({SETTINGS_WAN_PROFILES: [profile]})

    with TestClient(app, base_url="http://test") as client:
        resp = client.get("/api/admin/wan/profiles")
    assert resp.status_code == 200
    body_str = resp.text
    assert "should-never-be-returned" not in body_str


def test_profiles_delete_by_uuid(admin_app) -> None:
    app, store = admin_app
    store._save_wan_profile(
        {
            "id": _VALID_UUID,
            "ssid": "Y",
            "bssid": None,
            "psk_ref": "Y_PSK",
            "regdomain": None,
            "last_seen": None,
        }
    )
    with TestClient(app, base_url="http://test") as client:
        resp = client.delete(f"/api/admin/wan/profiles/{_VALID_UUID}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == _VALID_UUID
    assert store._find_wan_profile_by_id(_VALID_UUID) is None


def test_profiles_delete_unknown_404(admin_app) -> None:
    app, _ = admin_app
    with TestClient(app, base_url="http://test") as client:
        resp = client.delete(f"/api/admin/wan/profiles/{_VALID_UUID}")
    assert resp.status_code == 404


def test_profiles_delete_rejects_non_uuid(admin_app) -> None:
    app, _ = admin_app
    with TestClient(app, base_url="http://test") as client:
        resp = client.delete("/api/admin/wan/profiles/not-a-uuid")
    assert resp.status_code == 400


def test_set_active_round_trip(admin_app) -> None:
    app, store = admin_app
    store._save_wan_profile(
        {
            "id": _VALID_UUID,
            "ssid": "Y",
            "bssid": None,
            "psk_ref": "Y_PSK",
            "regdomain": None,
            "last_seen": None,
        }
    )
    with TestClient(app, base_url="http://test") as client:
        resp = client.post(f"/api/admin/wan/profiles/{_VALID_UUID}/set-active")
    assert resp.status_code == 200
    assert resp.json()["active_id"] == _VALID_UUID
    assert store._effective_wan_active_profile_id() == _VALID_UUID


def test_set_active_unknown_id_404(admin_app) -> None:
    app, _ = admin_app
    with TestClient(app, base_url="http://test") as client:
        resp = client.post(f"/api/admin/wan/profiles/{_VALID_UUID}/set-active")
    assert resp.status_code == 404


# ─── up / down ────────────────────────────────────────────────


def test_wan_up_calls_wifi_wan(admin_app) -> None:
    app, store = admin_app
    store._save_wan_profile(
        {
            "id": _VALID_UUID,
            "ssid": "Y",
            "bssid": None,
            "psk_ref": "Y_PSK",
            "regdomain": None,
            "last_seen": None,
        }
    )
    fake_up = AsyncMock()
    with (
        patch("meeting_scribe.wifi_wan.wan_up", new=fake_up),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post("/api/admin/wan/up", json={"id": _VALID_UUID})
    assert resp.status_code == 200
    fake_up.assert_awaited_once_with(_VALID_UUID)


def test_wan_up_rejects_invalid_id(admin_app) -> None:
    app, _ = admin_app
    with TestClient(app, base_url="http://test") as client:
        resp = client.post("/api/admin/wan/up", json={"id": "not-a-uuid"})
    assert resp.status_code == 400


def test_wan_up_translates_psk_decrypt_error_to_422(admin_app) -> None:
    app, _ = admin_app
    from meeting_scribe.server_support.secrets import SecretDecryptError

    async def _fail(profile_id: str) -> None:
        raise SecretDecryptError("decrypt failed in test")

    with (
        patch("meeting_scribe.wifi_wan.wan_up", new=_fail),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post("/api/admin/wan/up", json={"id": _VALID_UUID})
    assert resp.status_code == 422
    # Don't leak the underlying exception text.
    assert "decrypt failed in test" not in resp.text


def test_wan_down_calls_wifi_wan(admin_app) -> None:
    app, _ = admin_app
    fake_down = AsyncMock()
    with (
        patch("meeting_scribe.wifi_wan.wan_down", new=fake_down),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post("/api/admin/wan/down")
    assert resp.status_code == 200
    fake_down.assert_awaited_once()


# ─── Band preference ─────────────────────────────────────────


def test_profiles_create_accepts_band(admin_app) -> None:
    app, store = admin_app
    with (
        patch("meeting_scribe.server_support.secrets.psk_ref_exists", lambda ref: True),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post(
            "/api/admin/wan/profiles",
            json={"ssid": "Yunomotocho", "psk_ref": "YUNOMOTOCHO_PSK", "band": "a"},
        )
    assert resp.status_code == 201, resp.text
    profile = resp.json()["profile"]
    assert profile["band"] == "a"
    persisted = store._load_wan_profiles()[0]
    assert persisted["band"] == "a"


def test_profiles_create_default_band_is_auto(admin_app) -> None:
    app, _store = admin_app
    with (
        patch("meeting_scribe.server_support.secrets.psk_ref_exists", lambda ref: True),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post(
            "/api/admin/wan/profiles",
            json={"ssid": "Yunomotocho", "psk_ref": "YUNOMOTOCHO_PSK"},
        )
    assert resp.status_code == 201
    assert resp.json()["profile"]["band"] == "auto"


def test_profiles_create_rejects_invalid_band(admin_app) -> None:
    app, _ = admin_app
    with (
        patch("meeting_scribe.server_support.secrets.psk_ref_exists", lambda ref: True),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post(
            "/api/admin/wan/profiles",
            json={"ssid": "Y", "psk_ref": "X_PSK", "band": "garbage"},
        )
    assert resp.status_code == 400
    assert "band" in resp.json()["error"]


def test_scan_returns_consolidated_networks_and_raw_entries(admin_app) -> None:
    """REST scan exposes both ``networks`` (consolidated) and ``entries`` (raw)."""
    app, _ = admin_app
    from meeting_scribe.wifi_sta import ScanEntry

    entries = [
        ScanEntry(
            bssid="aa:00:00:00:00:01",
            ssid="DellEfficiency_Guest",
            channel=36,
            signal_dbm=-50.0,
            rsn_present=False,
        ),
        ScanEntry(
            bssid="aa:00:00:00:00:02",
            ssid="DellEfficiency_Guest",
            channel=1,
            signal_dbm=-72.0,
            rsn_present=False,
        ),
        ScanEntry(
            bssid="bb:00:00:00:00:01",
            ssid="DellEfficiency_Corp",
            channel=36,
            signal_dbm=-58.0,
            rsn_present=True,
        ),
    ]
    with (
        patch("meeting_scribe.wifi_wan.scan_upstream", new=AsyncMock(return_value=entries)),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.get("/api/admin/wan/scan")
    assert resp.status_code == 200
    body = resp.json()
    # Consolidated form: one row per (SSID, security).
    networks = {(n["ssid"], n["security"]): n for n in body["networks"]}
    guest = networks[("DellEfficiency_Guest", "open")]
    assert guest["ap_count"] == 2
    assert sorted(guest["bands"]) == ["a", "bg"]
    assert guest["best_signal_band"] == "a"  # strongest BSS is on ch 36

    # Raw form preserved for advanced callers.
    raw_bssids = {e["bssid"] for e in body["entries"]}
    assert raw_bssids == {"aa:00:00:00:00:01", "aa:00:00:00:00:02", "bb:00:00:00:00:01"}


# ─── Open networks via REST ───────────────────────────────────


def test_profiles_create_open_network_with_flag(admin_app) -> None:
    """``open: true`` lets the operator create an OPEN profile without a psk_ref."""
    app, store = admin_app
    with TestClient(app, base_url="http://test") as client:
        resp = client.post(
            "/api/admin/wan/profiles",
            json={"ssid": "DellEfficiency_Guest", "open": True},
        )
    assert resp.status_code == 201, resp.text
    profile = resp.json()["profile"]
    assert profile["psk_ref"] is None
    assert profile["ssid"] == "DellEfficiency_Guest"
    assert store._load_wan_profiles()[0]["psk_ref"] is None


def test_profiles_create_rejects_open_with_psk_ref(admin_app) -> None:
    app, _ = admin_app
    with TestClient(app, base_url="http://test") as client:
        resp = client.post(
            "/api/admin/wan/profiles",
            json={
                "ssid": "Foo",
                "open": True,
                "psk_ref": "Y_PSK",
            },
        )
    assert resp.status_code == 400
    assert "open" in resp.json()["error"]


def test_profiles_create_rejects_neither_open_nor_psk_ref(admin_app) -> None:
    app, _ = admin_app
    with TestClient(app, base_url="http://test") as client:
        resp = client.post(
            "/api/admin/wan/profiles",
            json={"ssid": "Foo"},
        )
    assert resp.status_code == 400
    assert "psk_ref" in resp.json()["error"] or "open" in resp.json()["error"]
