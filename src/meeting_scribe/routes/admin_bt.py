"""Admin Bluetooth API endpoints — Plan §B.7b.

Wires :mod:`meeting_scribe.bt` (control plane primitives) into typed
admin routes that the BT card calls. Every route here is gated by
``_require_admin_response`` (cookie auth) and CSRF-protected by the
Origin allowlist middleware. Cache-headers middleware applies
``Cache-Control: no-store, private`` because every path lives under
``/api/admin/``.

Endpoints:

* ``GET /api/admin/bt/status``       — bt_status_sync snapshot.
* ``POST /api/admin/bt/connect``     — connect by MAC.
* ``POST /api/admin/bt/disconnect``  — disconnect (or all).
* ``POST /api/admin/bt/forget``      — remove paired device.
* ``POST /api/admin/bt/pair``        — pair → trust → connect.
* ``POST /api/admin/bt/mic``         — drives the bridge state machine
                                        toward MicLive (HFP) or Idle.
* ``GET /api/admin/diag/audio``      — pactl/pipewire reachability.
"""

from __future__ import annotations

import logging
from typing import Any

import fastapi
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from meeting_scribe import bt as bt_mod
from meeting_scribe.server_support.admin_guard import _require_admin_response

logger = logging.getLogger(__name__)
router = APIRouter()


def _bad_request(detail: str, code: int = 400) -> JSONResponse:
    return JSONResponse({"error": detail}, status_code=code)


@router.get("/api/admin/bt/status")
async def bt_status_endpoint(request: Request) -> JSONResponse:
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    snap = await bt_mod.bt_status_sync()
    return JSONResponse(snap)


@router.post("/api/admin/bt/connect")
async def bt_connect_endpoint(request: Request) -> JSONResponse:
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    body = await _read_json(request)
    mac = body.get("mac") if isinstance(body, dict) else None
    if not isinstance(mac, str):
        return _bad_request("mac required")
    try:
        await bt_mod.bt_connect(mac)
    except bt_mod.BluetoothError as exc:
        return _bad_request(str(exc), code=502)
    return JSONResponse({"ok": True, "mac": mac})


@router.post("/api/admin/bt/disconnect")
async def bt_disconnect_endpoint(request: Request) -> JSONResponse:
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    body = await _read_json(request)
    mac = body.get("mac") if isinstance(body, dict) else None
    try:
        await bt_mod.bt_disconnect(mac, user_initiated=True)
    except bt_mod.BluetoothError as exc:
        return _bad_request(str(exc), code=502)
    return JSONResponse({"ok": True, "mac": mac or None})


@router.post("/api/admin/bt/forget")
async def bt_forget_endpoint(request: Request) -> JSONResponse:
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    body = await _read_json(request)
    mac = body.get("mac") if isinstance(body, dict) else None
    if not isinstance(mac, str):
        return _bad_request("mac required")
    try:
        await bt_mod.bt_forget(mac)
    except bt_mod.BluetoothError as exc:
        return _bad_request(str(exc), code=502)
    return JSONResponse({"ok": True, "mac": mac})


@router.post("/api/admin/bt/pair")
async def bt_pair_endpoint(request: Request) -> JSONResponse:
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    body = await _read_json(request)
    mac = body.get("mac") if isinstance(body, dict) else None
    if not isinstance(mac, str):
        return _bad_request("mac required")
    try:
        result = await bt_mod.bt_pair(mac)
    except bt_mod.BluetoothError as exc:
        # Passkey-required failures often surface here; the UI surfaces
        # the CLI-fallback instruction in the modal.
        return _bad_request(str(exc), code=502)
    return JSONResponse(result)


@router.post("/api/admin/bt/mic")
async def bt_mic_endpoint(request: Request) -> JSONResponse:
    """Toggle the bridge between Idle (A2DP) and MicLive (HFP).

    Body: ``{"enabled": true|false}``. Persisted to settings store so
    the bridge resumes the operator's last toggle on reconnect.
    """
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    body = await _read_json(request)
    enabled = body.get("enabled") if isinstance(body, dict) else None
    if not isinstance(enabled, bool):
        return _bad_request("enabled bool required")
    # The actual state-machine call lives in audio/bt_bridge.py and is
    # plumbed through state when the bridge is active. Without the
    # bridge wired in we simply persist the toggle so the next
    # bridge.start() picks it up.
    from meeting_scribe.server_support.settings_store import (
        _save_settings_override,
    )
    from meeting_scribe.bt import SETTINGS_BT_INPUT_ACTIVE

    _save_settings_override({SETTINGS_BT_INPUT_ACTIVE: enabled})
    return JSONResponse({"ok": True, "enabled": enabled})


@router.get("/api/admin/diag/audio")
async def diag_audio_endpoint(request: Request) -> JSONResponse:
    """Plan §B.9: live in-process pactl/pipewire reachability.

    The healthcheck script fronts this for ``meeting-scribe doctor``
    and the BT card's "Bluetooth stack unavailable" banner.
    """
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    out: dict[str, Any] = {"pipewire": "unknown", "pactl": "unknown"}
    proc_pw = await bt_mod._run(["pw-cli", "info", "0"])
    out["pipewire"] = "ok" if proc_pw.returncode == 0 else "fail"
    proc_pa = await bt_mod._run(["pactl", "info"])
    out["pactl"] = "ok" if proc_pa.returncode == 0 else "fail"
    return JSONResponse(out)


async def _read_json(request: Request) -> Any:
    try:
        return await request.json()
    except Exception:  # noqa: BLE001 — return None so the route emits 400
        return None
