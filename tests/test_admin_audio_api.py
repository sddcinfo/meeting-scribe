"""Tests for routes/admin_audio.py — admin gating, request validation,
side-effect verification (settings persistence + ServerMicCapture
reconcile + LocalSinkListener retarget).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient


@pytest.fixture(autouse=True)
def restore_audio_state():
    from meeting_scribe.runtime import state

    clients = set(state._audio_out_clients)
    prefs = dict(state._audio_out_prefs)
    interpretation_buffer = state.interpretation_buffer
    yield
    state._audio_out_clients = clients
    state._audio_out_prefs = prefs
    state.interpretation_buffer = interpretation_buffer


def _build_app(*, admin_ok: bool) -> FastAPI:
    """FastAPI app with admin_audio routes registered.

    Mirrors ``test_admin_bt_api._build_app`` — patches
    ``_require_admin_response`` so the gate can be exercised without a
    real cookie.
    """
    from meeting_scribe.routes import admin_audio as admin_audio_mod

    if admin_ok:
        admin_audio_mod._require_admin_response = lambda req: None
    else:
        admin_audio_mod._require_admin_response = lambda req: JSONResponse(
            {"error": "admin required"}, status_code=403
        )
    app = FastAPI()
    app.include_router(admin_audio_mod.router)
    return app


def test_devices_requires_admin() -> None:
    app = _build_app(admin_ok=False)
    with TestClient(app, base_url="http://test") as client:
        resp = client.get("/api/admin/audio/devices")
    assert resp.status_code == 403


def test_devices_returns_enumerated_payload() -> None:
    app = _build_app(admin_ok=True)
    fake_devices = {
        "sources": [
            {
                "node_id": 1,
                "node_name": "src1",
                "description": "S1",
                "kind": "source",
                "device_class": "usb",
                "is_default": True,
            }
        ],
        "sinks": [
            {
                "node_id": 2,
                "node_name": "snk1",
                "description": "K1",
                "kind": "sink",
                "device_class": "bluetooth",
                "is_default": False,
            }
        ],
    }
    with (
        patch(
            "meeting_scribe.routes.admin_audio.enumerate_audio_devices",
            new=AsyncMock(return_value=fake_devices),
        ),
        patch(
            "meeting_scribe.routes.admin_audio._load_settings_override",
            return_value={"audio_meeting_mic_node": "src1", "audio_meeting_mic_active": True},
        ),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.get("/api/admin/audio/devices")
    assert resp.status_code == 200
    body = resp.json()
    assert body["devices"] == fake_devices
    assert body["selection"]["mic_node"] == "src1"
    assert body["selection"]["mic_active"] is True


def test_route_post_persists_and_reconciles() -> None:
    """POST patches the settings file and triggers ServerMicCapture +
    LocalSinkListener reconciliation through the audio-routing self-heal
    surface."""
    captured: dict[str, Any] = {}

    def fake_save(overrides: dict[str, Any]) -> None:
        captured.update(overrides)

    fake_outcome = {
        "ok": True,
        "mic": {"ok": True, "error_message": None, "status": "ok"},
        "sink": {"ok": True, "error_message": None},
    }
    fake_reconcile = AsyncMock(return_value=fake_outcome)

    async def fake_derive(_node: str) -> tuple[str, dict[str, Any] | None]:
        return "usb:dell_inc._dell_sp325_speakerphone_0000000000000000", {"port": "pro-input-0"}

    app = _build_app(admin_ok=True)
    with (
        patch(
            "meeting_scribe.routes.admin_audio._save_settings_override",
            fake_save,
        ),
        patch(
            "meeting_scribe.routes.admin_audio._load_settings_override",
            return_value={
                "audio_meeting_mic_node": "alsa_input.poly",
                "audio_admin_tts_sink_node": "alsa_output.poly",
                "audio_meeting_mic_active": True,
            },
        ),
        patch(
            "meeting_scribe.audio.audio_routing.reconcile_audio_routing",
            fake_reconcile,
        ),
        patch(
            "meeting_scribe.audio.audio_routing.derive_stable_identity_for_node",
            fake_derive,
        ),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post(
            "/api/admin/audio/route",
            json={
                "mic_node": "alsa_input.poly",
                "admin_sink_node": "alsa_output.poly",
                "mic_active": True,
            },
        )
    assert resp.status_code == 200, resp.text
    assert captured["audio_meeting_mic_node"] == "alsa_input.poly"
    assert captured["audio_admin_tts_sink_node"] == "alsa_output.poly"
    assert captured["audio_meeting_mic_active"] is True
    # Phase 1 derives the stable identity at persist time.
    assert captured["audio_meeting_mic_stable_id"].startswith("usb:dell_inc._dell_sp325")
    fake_reconcile.assert_awaited_once()
    body = resp.json()
    assert body["reconcile_outcome"]["mic"]["status"] == "ok"


def test_route_post_rejects_non_string_mic_node() -> None:
    app = _build_app(admin_ok=True)
    with TestClient(app, base_url="http://test") as client:
        resp = client.post("/api/admin/audio/route", json={"mic_node": 42})
    assert resp.status_code == 400


def test_route_post_rejects_non_bool_mic_active() -> None:
    app = _build_app(admin_ok=True)
    with TestClient(app, base_url="http://test") as client:
        resp = client.post("/api/admin/audio/route", json={"mic_active": "true"})
    assert resp.status_code == 400


def test_route_post_rejects_empty_body() -> None:
    app = _build_app(admin_ok=True)
    with TestClient(app, base_url="http://test") as client:
        resp = client.post("/api/admin/audio/route", json={})
    assert resp.status_code == 400


def test_route_post_supports_partial_patch() -> None:
    """The UI can change mic_active alone without re-sending mic_node /
    sink_node. The unchanged fields stay untouched in the persisted
    settings."""
    captured: dict[str, Any] = {}

    def fake_save(overrides: dict[str, Any]) -> None:
        captured.update(overrides)

    app = _build_app(admin_ok=True)
    with (
        patch(
            "meeting_scribe.routes.admin_audio._save_settings_override",
            fake_save,
        ),
        patch(
            "meeting_scribe.routes.admin_audio._load_settings_override",
            return_value={"audio_meeting_mic_active": False},
        ),
        patch(
            "meeting_scribe.audio.audio_routing.reconcile_audio_routing",
            new=AsyncMock(
                return_value={
                    "ok": True,
                    "mic": {"ok": True, "error_message": None, "status": "ok"},
                    "sink": {"ok": True, "error_message": None},
                }
            ),
        ),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post("/api/admin/audio/route", json={"mic_active": False})
    assert resp.status_code == 200
    assert captured == {"audio_meeting_mic_active": False}


def test_route_post_rejects_room_sink_without_active_server_mic() -> None:
    app = _build_app(admin_ok=True)
    with (
        patch(
            "meeting_scribe.routes.admin_audio._load_settings_override",
            return_value={
                "audio_meeting_mic_node": "",
                "audio_meeting_mic_active": False,
            },
        ),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post(
            "/api/admin/audio/route",
            json={
                "room_sink_node": (
                    "alsa_output.usb-Plantronics_Poly_Sync_20-M_8B33ABCDEF-00.analog-stereo"
                )
            },
        )

    assert resp.status_code == 400
    assert "matching server mic" in resp.json()["error"]


def test_route_post_allows_room_sink_with_matching_server_mic() -> None:
    captured: dict[str, Any] = {}

    def fake_save(overrides: dict[str, Any]) -> None:
        captured.update(overrides)

    fake_outcome = {
        "ok": True,
        "mic": {"ok": True, "error_message": None, "status": "ok"},
        "sink": {"ok": True, "error_message": None},
    }
    fake_reconcile = AsyncMock(return_value=fake_outcome)

    async def fake_derive(_node: str) -> tuple[str, dict[str, Any] | None]:
        return "usb:plantronics_poly_sync_20-m_8b33abcdef", {"port": "mono-fallback"}

    poly_mic = "alsa_input.usb-Plantronics_Poly_Sync_20-M_8B33ABCDEF-00.mono-fallback"
    poly_sink = "alsa_output.usb-Plantronics_Poly_Sync_20-M_8B33ABCDEF-00.analog-stereo"
    app = _build_app(admin_ok=True)
    with (
        patch("meeting_scribe.routes.admin_audio._save_settings_override", fake_save),
        patch(
            "meeting_scribe.routes.admin_audio._load_settings_override",
            return_value={
                "audio_meeting_mic_node": poly_mic,
                "audio_meeting_mic_active": True,
                "audio_room_tts_sink_node": poly_sink,
            },
        ),
        patch(
            "meeting_scribe.audio.audio_routing.reconcile_audio_routing",
            fake_reconcile,
        ),
        patch(
            "meeting_scribe.audio.audio_routing.derive_stable_identity_for_node",
            fake_derive,
        ),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post(
            "/api/admin/audio/route",
            json={
                "mic_node": poly_mic,
                "mic_active": True,
                "room_sink_node": poly_sink,
            },
        )

    assert resp.status_code == 200
    assert captured["audio_meeting_mic_node"] == poly_mic
    assert captured["audio_meeting_mic_active"] is True
    assert captured["audio_room_tts_sink_node"] == poly_sink
    fake_reconcile.assert_awaited_once()


def test_route_post_clears_room_sink_when_server_mic_is_disabled() -> None:
    captured: dict[str, Any] = {}

    def fake_save(overrides: dict[str, Any]) -> None:
        captured.update(overrides)

    app = _build_app(admin_ok=True)
    with (
        patch("meeting_scribe.routes.admin_audio._save_settings_override", fake_save),
        patch(
            "meeting_scribe.routes.admin_audio._load_settings_override",
            return_value={
                "audio_meeting_mic_node": "alsa_input.usb-Plantronics_Poly_Sync_20-M_8B33-00.mono-fallback",
                "audio_meeting_mic_active": False,
                "audio_room_tts_sink_node": "alsa_output.usb-Plantronics_Poly_Sync_20-M_8B33-00.analog-stereo",
            },
        ),
        patch(
            "meeting_scribe.audio.audio_routing.reconcile_audio_routing",
            new=AsyncMock(
                return_value={
                    "ok": True,
                    "mic": {"ok": True, "error_message": None, "status": "ok"},
                    "sink": {"ok": True, "error_message": None},
                }
            ),
        ),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post("/api/admin/audio/route", json={"mic_active": False})

    assert resp.status_code == 200
    assert captured["audio_meeting_mic_active"] is False
    assert captured["audio_room_tts_sink_node"] == ""


def test_interpretation_requires_admin() -> None:
    app = _build_app(admin_ok=False)
    with TestClient(app, base_url="http://test") as client:
        resp = client.get("/api/admin/audio/interpretation")
    assert resp.status_code == 403


def test_interpretation_get_reconciles_configured_local_sink() -> None:
    app = _build_app(admin_ok=True)
    with (
        patch(
            "meeting_scribe.audio.local_sink.ensure_local_sink_listener_registered",
            return_value=None,
        ) as ensure,
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.get("/api/admin/audio/interpretation")

    assert resp.status_code == 200
    ensure.assert_called_once()


def test_interpretation_post_persists_enable_and_pause() -> None:
    captured: dict[str, Any] = {}

    def fake_save(overrides: dict[str, Any]) -> None:
        captured.update(overrides)

    app = _build_app(admin_ok=True)
    with (
        patch("meeting_scribe.routes.admin_audio._save_settings_override", fake_save),
        patch(
            "meeting_scribe.routes.admin_audio._effective_interpretation_enabled",
            return_value=True,
        ),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post(
            "/api/admin/audio/interpretation",
            json={"enabled": True, "pause_flush_ms": 1700, "idle_drain_ms": 5100},
        )
    assert resp.status_code == 200
    assert captured["interpretation_enabled"] is True
    assert captured["interpretation_pause_flush_ms"] == 1700
    assert captured["interpretation_idle_drain_ms"] == 5100


def test_interpretation_enable_keeps_room_sink_target_language() -> None:
    import meeting_scribe.audio.local_sink as local_sink
    from meeting_scribe.audio.local_sink import LocalSinkListener
    from meeting_scribe.runtime import state
    from meeting_scribe.server_support.sessions import ClientSession

    mic_node = "alsa_input.usb-Plantronics_Poly_Sync_20-M_8B33ABCDEF-00.mono-fallback"
    sink_node = "alsa_output.usb-Plantronics_Poly_Sync_20-M_8B33ABCDEF-00.analog-stereo"
    settings = {
        "interpretation_enabled": True,
        "room_tts_language": "en",
        "audio_meeting_mic_active": True,
        "audio_meeting_mic_node": mic_node,
        "audio_room_tts_sink_node": sink_node,
    }
    room = LocalSinkListener(
        target_node=sink_node,
        role=local_sink.LOCAL_SINK_ROLE_ROOM,
        echo_guard=True,
    )
    state._audio_out_clients.add(room)
    state._audio_out_prefs[room] = ClientSession(
        transport="room_sink",
        preferred_language="en",
        delivery_mode="simultaneous",
    )
    app = _build_app(admin_ok=True)
    try:
        with (
            patch("meeting_scribe.routes.admin_audio._save_settings_override", lambda _: None),
            patch(
                "meeting_scribe.routes.admin_audio._load_settings_override",
                return_value=settings,
            ),
            patch(
                "meeting_scribe.audio.local_sink._load_settings_override",
                return_value=settings,
            ),
            patch(
                "meeting_scribe.audio.local_sink._effective_interpretation_enabled",
                return_value=True,
            ),
            patch(
                "meeting_scribe.routes.admin_audio._effective_interpretation_enabled",
                return_value=True,
            ),
            TestClient(app, base_url="http://test") as client,
        ):
            resp = client.post("/api/admin/audio/interpretation", json={"enabled": True})
        assert resp.status_code == 200
        pref = state._audio_out_prefs[room]
        assert pref.delivery_mode == "consecutive"
        assert pref.preferred_language == "en"
    finally:
        room.shutdown()
        state._audio_out_clients.discard(room)
        state._audio_out_prefs.pop(room, None)


def test_interpretation_post_persists_room_sink_language() -> None:
    captured: dict[str, Any] = {}

    def fake_save(overrides: dict[str, Any]) -> None:
        captured.update(overrides)

    app = _build_app(admin_ok=True)
    with (
        patch("meeting_scribe.routes.admin_audio._save_settings_override", fake_save),
        patch(
            "meeting_scribe.routes.admin_audio._effective_interpretation_enabled",
            return_value=True,
        ),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post(
            "/api/admin/audio/interpretation",
            json={"local_sink_language": "ja"},
        )
    assert resp.status_code == 200
    assert captured["admin_tts_language"] == "ja"


def test_interpretation_post_persists_tts_voice_mode() -> None:
    """The audio popover / room-setup voice-mode select rides the same
    endpoint as the global settings panel — locks the parity in.
    """
    captured: dict[str, Any] = {}

    def fake_save(overrides: dict[str, Any]) -> None:
        captured.update(overrides)

    app = _build_app(admin_ok=True)
    with (
        patch("meeting_scribe.routes.admin_audio._save_settings_override", fake_save),
        patch(
            "meeting_scribe.routes.admin_audio._effective_interpretation_enabled",
            return_value=True,
        ),
        TestClient(app, base_url="http://test") as client,
    ):
        resp_ok = client.post(
            "/api/admin/audio/interpretation",
            json={"tts_voice_mode": "cloned"},
        )
        resp_bad = client.post(
            "/api/admin/audio/interpretation",
            json={"tts_voice_mode": "nonsense"},
        )
    assert resp_ok.status_code == 200
    assert captured["tts_voice_mode"] == "cloned"
    assert resp_bad.status_code == 400


def test_interpretation_get_exposes_voice_mode_options() -> None:
    """GET /api/admin/audio/interpretation includes the same voice_mode
    surface as /api/admin/settings so the in-meeting popover + setup
    screen can render their select without a second fetch.
    """
    app = _build_app(admin_ok=True)
    with (
        patch(
            "meeting_scribe.server_support.settings_store._effective_tts_voice_mode",
            return_value="cloned",
        ),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.get("/api/admin/audio/interpretation")
    assert resp.status_code == 200
    body = resp.json()
    assert body["tts_voice_mode"] == "cloned"
    codes = {opt["code"] for opt in body["tts_voice_mode_options"]}
    assert codes == {"studio", "cloned"}


def test_interpretation_mute_web_does_not_mute_bt() -> None:
    from meeting_scribe.runtime import state
    from meeting_scribe.server_support.sessions import ClientSession

    web = object()
    bt = object()
    state._audio_out_clients.update({web, bt})
    state._audio_out_prefs[web] = ClientSession(transport="web_browser")
    state._audio_out_prefs[bt] = ClientSession(transport="bt_headset")
    app = _build_app(admin_ok=True)
    try:
        with TestClient(app, base_url="http://test") as client:
            resp = client.post("/api/admin/audio/interpretation", json={"mute": "mute_web"})
        assert resp.status_code == 200
        assert state._audio_out_prefs[web].delivery_mode == "drop"
        assert state._audio_out_prefs[bt].delivery_mode == "simultaneous"
    finally:
        state._audio_out_clients.discard(web)
        state._audio_out_clients.discard(bt)
        state._audio_out_prefs.pop(web, None)
        state._audio_out_prefs.pop(bt, None)


def test_interpretation_rejects_bad_mute_command() -> None:
    app = _build_app(admin_ok=True)
    with TestClient(app, base_url="http://test") as client:
        resp = client.post("/api/admin/audio/interpretation", json={"mute": "all"})
    assert resp.status_code == 400


# ── W3: failure envelopes ─────────────────────────────────────────────
#
# The new admin_audio routes must never silently swallow reconcile
# failures (the prior pattern returned 200 with the success payload,
# making the UI think a mic switch had taken effect when it hadn't).
# These tests pin three things:
#   1. failures surface as 500
#   2. the envelope shape is ``{error: operation_code, message: ...}``
#   3. the response body never contains substrings from the underlying
#      exception — operators see a stable code, full traceback only
#      goes to the log.


_FINGERPRINT = "RECONCILE_INTERNAL_BACKTRACE_FINGERPRINT_8E4A"


def _assert_no_exception_leak(body: dict[str, Any]) -> None:
    payload_text = str(body)
    assert _FINGERPRINT not in payload_text, f"exception text leaked into response: {payload_text}"


def test_route_post_returns_capture_failed_outcome_when_server_mic_reconcile_fails() -> None:
    """A pw-record spawn failure now surfaces as a structured outcome
    (``mic.status="capture_failed"``) rather than a 500. The UI consumes
    this from ``reconcile_outcome`` and shows an actionable banner."""

    fake_outcome = {
        "ok": False,
        "mic": {
            "ok": False,
            "error_message": "mic capture failed to start: pw-record exec failed",
            "status": "capture_failed",
            "detail": "pw-record exec failed",
        },
        "sink": {"ok": True, "error_message": None},
    }
    fake_reconcile = AsyncMock(return_value=fake_outcome)

    app = _build_app(admin_ok=True)
    with (
        patch("meeting_scribe.routes.admin_audio._save_settings_override", lambda _u: None),
        patch(
            "meeting_scribe.routes.admin_audio._load_settings_override",
            return_value={
                "audio_meeting_mic_node": "src1",
                "audio_meeting_mic_active": True,
            },
        ),
        patch(
            "meeting_scribe.audio.audio_routing.reconcile_audio_routing",
            fake_reconcile,
        ),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post("/api/admin/audio/route", json={"mic_active": True})

    assert resp.status_code == 200
    body = resp.json()
    outcome = body["reconcile_outcome"]
    assert outcome["mic"]["status"] == "capture_failed"
    assert outcome["mic"]["ok"] is False
    assert "capture" in outcome["mic"]["error_message"]


def test_route_post_reports_500_when_reconcile_audio_routing_raises() -> None:
    """Only an unexpected exception inside ``reconcile_audio_routing`` (not
    a normal failure path) produces a 500. The route still returns
    a stable error label so the UI can match on it."""

    async def boom() -> dict[str, Any]:
        raise RuntimeError(_FINGERPRINT)

    app = _build_app(admin_ok=True)
    with (
        patch("meeting_scribe.routes.admin_audio._save_settings_override", lambda _u: None),
        patch(
            "meeting_scribe.routes.admin_audio._load_settings_override",
            return_value={
                "audio_admin_tts_sink_node": "alsa_output.poly",
            },
        ),
        patch(
            "meeting_scribe.audio.audio_routing.reconcile_audio_routing",
            boom,
        ),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post(
            "/api/admin/audio/route",
            json={"admin_sink_node": "alsa_output.poly"},
        )

    assert resp.status_code == 500
    body = resp.json()
    assert body["error"] == "reconcile_audio_routing"
    _assert_no_exception_leak(body)


def test_interpretation_get_reports_500_when_reconcile_fails() -> None:
    def boom() -> None:
        raise RuntimeError(_FINGERPRINT)

    app = _build_app(admin_ok=True)
    with (
        patch(
            "meeting_scribe.audio.local_sink.ensure_local_sink_listener_registered",
            boom,
        ),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.get("/api/admin/audio/interpretation")

    assert resp.status_code == 500
    body = resp.json()
    assert body["error"] == "ensure_local_sink_listener"
    _assert_no_exception_leak(body)


def _poly_devices_payload() -> dict[str, Any]:
    return {
        "sources": [
            {
                "node_id": 49,
                "node_name": "alsa_input.usb-Plantronics_Poly.mono-fallback",
                "description": "Poly Sync 20-M Mono",
                "kind": "source",
                "device_class": "usb",
                "is_default": True,
                "volume": 1.0,
                "muted": False,
            }
        ],
        "sinks": [
            {
                "node_id": 48,
                "node_name": "alsa_output.usb-Plantronics_Poly.analog-stereo",
                "description": "Poly Sync 20-M Analog Stereo",
                "kind": "sink",
                "device_class": "usb",
                "is_default": True,
                "volume": 1.0,
                "muted": False,
            }
        ],
    }


def test_volume_post_requires_admin() -> None:
    app = _build_app(admin_ok=False)
    with TestClient(app, base_url="http://test") as client:
        resp = client.post("/api/admin/audio/volume", json={"node_name": "x", "volume": 0.5})
    assert resp.status_code == 403


def test_volume_post_rejects_missing_node_name() -> None:
    app = _build_app(admin_ok=True)
    with TestClient(app, base_url="http://test") as client:
        resp = client.post("/api/admin/audio/volume", json={"volume": 0.5})
    assert resp.status_code == 400


def test_volume_post_rejects_out_of_range_volume() -> None:
    app = _build_app(admin_ok=True)
    with TestClient(app, base_url="http://test") as client:
        resp = client.post(
            "/api/admin/audio/volume",
            json={"node_name": "alsa_input.foo", "volume": 2.5},
        )
    assert resp.status_code == 400


def test_volume_post_rejects_unknown_node() -> None:
    app = _build_app(admin_ok=True)
    with (
        patch(
            "meeting_scribe.routes.admin_audio.enumerate_audio_devices",
            new=AsyncMock(return_value={"sources": [], "sinks": []}),
        ),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post(
            "/api/admin/audio/volume",
            json={"node_name": "alsa_input.gone", "volume": 0.5},
        )
    assert resp.status_code == 404


def test_volume_post_resolves_node_name_and_calls_set_device_volume() -> None:
    captured: dict[str, Any] = {}

    async def fake_set(
        node_ref: int | str, *, volume: float | None = None, muted: bool | None = None
    ) -> bool:
        captured["node_ref"] = node_ref
        captured["volume"] = volume
        captured["muted"] = muted
        return True

    app = _build_app(admin_ok=True)
    with (
        patch(
            "meeting_scribe.routes.admin_audio.enumerate_audio_devices",
            new=AsyncMock(return_value=_poly_devices_payload()),
        ),
        patch("meeting_scribe.routes.admin_audio.set_device_volume", new=fake_set),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post(
            "/api/admin/audio/volume",
            json={
                "node_name": "alsa_output.usb-Plantronics_Poly.analog-stereo",
                "volume": 0.7,
                "muted": False,
            },
        )

    assert resp.status_code == 200
    # Numeric node_id from the pw-dump enumeration is what wpctl needs.
    assert captured["node_ref"] == 48
    assert captured["volume"] == 0.7
    assert captured["muted"] is False
    body = resp.json()
    assert body["node_name"] == "alsa_output.usb-Plantronics_Poly.analog-stereo"
    assert body["volume"] == 1.0  # echoed from the post-write enumerate stub
    assert body["muted"] is False


_POLY_MIC = "alsa_input.usb-Plantronics_Poly_Sync_20-M_8B33-00.mono-fallback"
_POLY_SINK = "alsa_output.usb-Plantronics_Poly_Sync_20-M_8B33-00.analog-stereo"


def test_route_post_rejects_admin_sink_that_collides_with_room_sink() -> None:
    """Operator explicitly tries to set admin sink to the same Poly the room
    sink already uses → 400 instead of silently saving the dual-render echo."""
    app = _build_app(admin_ok=True)
    with (
        patch(
            "meeting_scribe.routes.admin_audio._load_settings_override",
            return_value={
                "audio_meeting_mic_node": _POLY_MIC,
                "audio_meeting_mic_active": True,
                "audio_room_tts_sink_node": _POLY_SINK,
                "room_tts_language": "all",
                "admin_tts_language": "en",
            },
        ),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post(
            "/api/admin/audio/route",
            json={"admin_sink_node": _POLY_SINK},
        )

    assert resp.status_code == 400
    assert "echo" in resp.json()["error"].lower()


def test_route_post_auto_clears_admin_sink_when_room_sink_change_creates_collision() -> None:
    """Operator changes the room sink onto the same Poly the admin sink uses;
    the admin sink is auto-cleared in the same patch (silent fix, not 400)."""
    captured: dict[str, Any] = {}

    def fake_save(overrides: dict[str, Any]) -> None:
        captured.update(overrides)

    app = _build_app(admin_ok=True)
    with (
        patch("meeting_scribe.routes.admin_audio._save_settings_override", fake_save),
        patch(
            "meeting_scribe.routes.admin_audio._load_settings_override",
            return_value={
                "audio_meeting_mic_node": _POLY_MIC,
                "audio_meeting_mic_active": True,
                "audio_admin_tts_sink_node": _POLY_SINK,
                "room_tts_language": "all",
                "admin_tts_language": "en",
            },
        ),
        patch(
            "meeting_scribe.audio.audio_routing.reconcile_audio_routing",
            new=AsyncMock(
                return_value={
                    "ok": True,
                    "mic": {"ok": True, "error_message": None, "status": "ok"},
                    "sink": {"ok": True, "error_message": None},
                }
            ),
        ),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post(
            "/api/admin/audio/route",
            json={"room_sink_node": _POLY_SINK},
        )

    assert resp.status_code == 200
    assert captured["audio_room_tts_sink_node"] == _POLY_SINK
    assert captured["audio_admin_tts_sink_node"] == ""  # auto-cleared


def test_interpretation_post_auto_clears_admin_sink_when_room_lang_creates_collision() -> None:
    """Changing room TTS language to ``all`` while admin sink == room sink
    triggers the same auto-clear via the interpretation endpoint."""
    captured: dict[str, Any] = {}

    def fake_save(overrides: dict[str, Any]) -> None:
        captured.update(overrides)

    app = _build_app(admin_ok=True)
    with (
        patch("meeting_scribe.routes.admin_audio._save_settings_override", fake_save),
        patch(
            "meeting_scribe.routes.admin_audio._load_settings_override",
            return_value={
                "audio_admin_tts_sink_node": _POLY_SINK,
                "audio_room_tts_sink_node": _POLY_SINK,
                "admin_tts_language": "en",
                "room_tts_language": "ja",  # was non-overlapping
            },
        ),
        patch(
            "meeting_scribe.routes.admin_audio._reconcile_local_sink_or_log",
            return_value=True,
        ),
        patch(
            "meeting_scribe.routes.admin_audio._apply_interpretation_live",
            new=AsyncMock(),
        ),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post(
            "/api/admin/audio/interpretation",
            json={"room_tts_language": "all"},
        )

    assert resp.status_code == 200
    assert captured["room_tts_language"] == "all"
    assert captured["audio_admin_tts_sink_node"] == ""  # silently cleared


def test_route_post_allows_admin_sink_when_room_lang_does_not_overlap() -> None:
    """admin_lang=en, room_lang=ja, same physical Poly → bilingual output,
    not a collision. Save should go through cleanly."""
    captured: dict[str, Any] = {}

    def fake_save(overrides: dict[str, Any]) -> None:
        captured.update(overrides)

    app = _build_app(admin_ok=True)
    with (
        patch("meeting_scribe.routes.admin_audio._save_settings_override", fake_save),
        patch(
            "meeting_scribe.routes.admin_audio._load_settings_override",
            return_value={
                "audio_meeting_mic_node": _POLY_MIC,
                "audio_meeting_mic_active": True,
                "audio_room_tts_sink_node": _POLY_SINK,
                "room_tts_language": "ja",
                "admin_tts_language": "en",
            },
        ),
        patch(
            "meeting_scribe.audio.audio_routing.reconcile_audio_routing",
            new=AsyncMock(
                return_value={
                    "ok": True,
                    "mic": {"ok": True, "error_message": None, "status": "ok"},
                    "sink": {"ok": True, "error_message": None},
                }
            ),
        ),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post(
            "/api/admin/audio/route",
            json={"admin_sink_node": _POLY_SINK},
        )

    assert resp.status_code == 200
    assert captured["audio_admin_tts_sink_node"] == _POLY_SINK


def test_volume_post_returns_500_when_set_device_volume_fails() -> None:
    async def fake_set(*_args: Any, **_kwargs: Any) -> bool:
        return False

    app = _build_app(admin_ok=True)
    with (
        patch(
            "meeting_scribe.routes.admin_audio.enumerate_audio_devices",
            new=AsyncMock(return_value=_poly_devices_payload()),
        ),
        patch("meeting_scribe.routes.admin_audio.set_device_volume", new=fake_set),
        TestClient(app, base_url="http://test") as client,
    ):
        resp = client.post(
            "/api/admin/audio/volume",
            json={
                "node_name": "alsa_output.usb-Plantronics_Poly.analog-stereo",
                "volume": 0.5,
            },
        )

    assert resp.status_code == 500
    body = resp.json()
    assert body["error"] == "apply_device_volume"
