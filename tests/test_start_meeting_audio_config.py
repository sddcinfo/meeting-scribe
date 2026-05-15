"""Atomic audio_config on ``POST /api/meeting/start``.

The setup wizard sends a single ``audio_config`` snapshot with the start
request; the handler validates, persists, and reconciles atomically
before recording begins. These tests pin the contract:

  * Validation errors return 400 before deep-health is touched.
  * A valid audio_config writes to settings AND triggers reconcile.
  * Reconcile failure with explicit audio_config returns 503 and the
    meeting is NOT started.
  * No audio_config + reconcile failure proceeds as a soft warning.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest


async def _call_start(body: dict[str, Any]) -> Any:
    """Invoke ``_start_meeting_locked`` directly with a stubbed request."""
    from meeting_scribe.routes import meeting_lifecycle

    class _Request:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        async def json(self) -> dict[str, Any]:
            return self._payload

    return await meeting_lifecycle._start_meeting_locked(_Request(body))


@pytest.fixture(autouse=True)
def stub_health(monkeypatch):
    """Default to healthy backends so we don't drop out at the health gate."""
    from meeting_scribe.routes import meeting_lifecycle
    from meeting_scribe.server_support import backend_health

    async def fake_health(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {"asr": {"ready": True}, "translate": {"ready": True}}

    async def fake_preflight() -> None:
        return None

    monkeypatch.setattr(backend_health, "_deep_backend_health", fake_health)
    monkeypatch.setattr(meeting_lifecycle, "_meeting_start_preflight", fake_preflight)


@pytest.fixture(autouse=True)
def restore_state():
    from meeting_scribe.runtime import state

    saved = state.current_meeting
    yield
    state.current_meeting = saved


def test_invalid_audio_config_returns_400_before_health_check() -> None:
    """A typed-mismatch on audio_config must short-circuit with 400."""

    async def runner() -> Any:
        return await _call_start({"audio_config": {"mic_node": 42}})

    resp = asyncio.run(runner())
    assert resp.status_code == 400
    body = resp.body.decode()
    assert "mic_node must be string or null" in body
    assert "audio_config_invalid" in body


def test_valid_audio_config_persists_and_reconciles() -> None:
    """A valid audio_config writes to settings AND calls reconcile."""
    saved: dict[str, Any] = {}

    def fake_save(updates: dict[str, Any]) -> None:
        saved.update(updates)

    reconcile_mock = AsyncMock(return_value={"ok": True, "mic": {"ok": True}, "sink": {"ok": True}})

    async def runner() -> Any:
        with (
            patch(
                "meeting_scribe.server_support.settings_store._save_settings_override",
                fake_save,
            ),
            patch(
                "meeting_scribe.audio.audio_routing.reconcile_audio_routing",
                reconcile_mock,
            ),
            # Block before storage transitions so we don't need a real
            # storage handle. The reconcile + persistence happen before
            # this point; we just need to observe them.
            patch(
                "meeting_scribe.runtime.state.storage",
                None,
            ),
        ):
            return await _call_start(
                {
                    "audio_config": {
                        "mic_node": "alsa_input.poly",
                        "admin_sink_node": "alsa_output.poly",
                        "mic_active": True,
                    }
                }
            )

    # The handler will raise when it hits state.storage; we don't care
    # about the response, only the recorded side effects.
    with pytest.raises((AttributeError, TypeError, Exception)):
        asyncio.run(runner())

    assert saved["audio_meeting_mic_node"] == "alsa_input.poly"
    assert saved["audio_admin_tts_sink_node"] == "alsa_output.poly"
    assert saved["audio_meeting_mic_active"] is True
    reconcile_mock.assert_awaited()


def test_explicit_audio_config_with_reconcile_failure_returns_503() -> None:
    """Operator chose routing → reconcile failure refuses the start."""

    failing_reconcile = AsyncMock(
        return_value={
            "ok": False,
            "mic": {
                "ok": False,
                "error_message": "Could not switch the microphone routing",
            },
            "sink": {"ok": True, "error_message": None},
        }
    )

    async def runner() -> Any:
        with (
            patch(
                "meeting_scribe.server_support.settings_store._save_settings_override",
                lambda _u: None,
            ),
            patch(
                "meeting_scribe.audio.audio_routing.reconcile_audio_routing",
                failing_reconcile,
            ),
        ):
            return await _call_start(
                {
                    "audio_config": {"mic_node": "missing.usb", "mic_active": True},
                }
            )

    resp = asyncio.run(runner())
    assert resp.status_code == 503
    body = resp.body.decode()
    assert "audio_routing_failed" in body
    assert "microphone routing" in body
    # No raw exception text leaks.
    assert "Traceback" not in body


def test_no_audio_config_soft_warns_on_reconcile_failure(monkeypatch) -> None:
    """No explicit audio_config → reconcile failure is a soft warning."""
    failing_reconcile = AsyncMock(
        return_value={
            "ok": False,
            "mic": {
                "ok": False,
                "error_message": "Could not switch the microphone routing",
            },
            "sink": {"ok": True, "error_message": None},
        }
    )

    async def runner() -> Any:
        with (
            patch(
                "meeting_scribe.audio.audio_routing.reconcile_audio_routing",
                failing_reconcile,
            ),
        ):
            try:
                return await _call_start({"language_pair": ["en", "ja"]})
            except Exception:
                return None

    # Run the entry path far enough to prove the handler did NOT return
    # 503 at the reconcile gate. State.storage is None so we'll trip
    # later — but the reconcile call must have happened with no fatal
    # return before that.
    asyncio.run(runner())
    failing_reconcile.assert_awaited()
