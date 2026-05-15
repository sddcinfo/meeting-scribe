"""UDS-transport HTTP client used by the daemon.

Thin wrapper around ``httpx.AsyncClient`` with the UDS transport. The
daemon's ``actions.MeetingClient`` protocol is satisfied here.

The client owns the socket path resolution (same logic as
``speakerphone/uds.py``) so the daemon and server agree on where to
look without any wiring config.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx

from meeting_scribe.speakerphone.uds import default_uds_path

logger = logging.getLogger(__name__)


class UdsMeetingClient:
    """``MeetingClient`` implementation that talks over the UDS.

    Used by the daemon's action dispatcher. Designed to be cheap to
    instantiate per device task; reuses a single ``AsyncClient`` for
    the duration of the daemon process.
    """

    def __init__(self, path: Path | None = None, *, timeout: float = 5.0) -> None:
        self._path = path or default_uds_path()
        transport = httpx.AsyncHTTPTransport(uds=str(self._path))
        # ``base_url`` value is irrelevant for UDS transport — but
        # httpx requires a host so use a sentinel that never gets DNS'd.
        self._client = httpx.AsyncClient(
            transport=transport,
            base_url="http://meeting-scribe.sock",
            timeout=timeout,
        )

    @property
    def path(self) -> Path:
        return self._path

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── MeetingClient protocol ─────────────────────────────────────────

    async def get_state(self) -> dict[str, Any]:
        resp = await self._client.get("/api/internal/speakerphone/state")
        resp.raise_for_status()
        return resp.json()

    async def set_interpretation(
        self,
        *,
        enabled: bool | None = None,
        room_tts_language: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if enabled is not None:
            body["enabled"] = enabled
        if room_tts_language is not None:
            body["room_tts_language"] = room_tts_language
        resp = await self._client.post(
            "/api/internal/speakerphone/interpretation",
            json=body,
        )
        resp.raise_for_status()
        return resp.json()

    async def toggle_mic_mute(self) -> dict[str, Any]:
        resp = await self._client.post("/api/internal/speakerphone/mic-mute")
        resp.raise_for_status()
        return resp.json()

    async def toggle_meeting_record(self) -> dict[str, Any]:
        resp = await self._client.post("/api/internal/speakerphone/meeting-toggle")
        resp.raise_for_status()
        return resp.json()

    async def report_press(
        self,
        *,
        device_key: str,
        button: str,
        press_kind: str,
    ) -> None:
        body = {"device_key": device_key, "button": button, "press_kind": press_kind}
        resp = await self._client.post("/api/internal/speakerphone/press", json=body)
        resp.raise_for_status()

    async def speak(
        self,
        *,
        label_id: str,
        language: str | None = None,
        overrides: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Request synthesis + playback of a button-feedback label.

        Daemon physical-press path: server reads the mapping fresh and
        resolves ``language`` / ``overrides`` itself, so the daemon
        normally passes only ``label_id``. The optional kwargs exist
        for the test client and for a future per-device override path.
        """
        body: dict[str, Any] = {"label_id": label_id}
        if language is not None:
            body["language"] = language
        if overrides is not None:
            body["overrides"] = overrides
        resp = await self._client.post(
            "/api/internal/speakerphone/speak",
            json=body,
        )
        resp.raise_for_status()
        return resp.json()
