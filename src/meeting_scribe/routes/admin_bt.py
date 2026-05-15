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
* ``GET /api/admin/diag/audio``      — pipewire (pw-cli + wpctl) reachability.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from meeting_scribe import bt as bt_mod
from meeting_scribe.server_support.admin_guard import _require_admin_response

logger = logging.getLogger(__name__)
router = APIRouter()


def _bad_request(detail: str, code: int = 400) -> JSONResponse:
    return JSONResponse({"error": detail}, status_code=code)


def _bt_failure(operation: str, exc: BaseException) -> JSONResponse:
    """Surface a bluetooth backend failure without leaking the
    underlying exception's str() to the wire. The detailed message
    goes to the server log; the response carries a stable error
    code the admin UI maps to copy. CodeQL flagged the previous
    ``_bad_request(str(exc), ...)`` pattern as information exposure
    through an exception (the str includes filesystem paths +
    bluetoothctl-internal detail)."""
    logger.warning("bt %s failed: %s", operation, exc, exc_info=True)
    return JSONResponse(
        {"error": "bt_backend_failed", "operation": operation},
        status_code=502,
    )


@router.get("/api/admin/bt/status")
async def bt_status_endpoint(request: Request) -> JSONResponse:
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    snap = await bt_mod.bt_status_sync()
    return JSONResponse(snap)


@router.get("/api/admin/bt/scan")
async def bt_scan_endpoint(request: Request) -> JSONResponse:
    """Scan for nearby BT devices (~10s) and return non-paired candidates.

    Purpose: drive the BT pairing flow from the admin UI so operators
    don't have to drop into the CLI to get a MAC.

    Returns ``{"devices": [{"mac", "name"}, ...]}``. Already-paired
    devices are filtered out (they're surfaced via
    ``GET /api/admin/bt/status``). Devices that haven't yet broadcast
    a friendly name are filtered too — the UI re-runs the scan a few
    seconds later to pick them up.
    """
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    timeout_raw = request.query_params.get("timeout", "10")
    try:
        timeout = float(timeout_raw)
    except ValueError:
        timeout = 10.0
    try:
        devices = await bt_mod.bt_scan(timeout=timeout)
    except bt_mod.BluetoothError as exc:
        return _bt_failure("scan", exc)
    return JSONResponse({"devices": devices})


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
        return _bt_failure("connect", exc)
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
        return _bt_failure("disconnect", exc)
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
        return _bt_failure("forget", exc)
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
        # Passkey-required failures often surface here; the UI maps the
        # ``pair`` operation code in _bt_failure's response to the
        # CLI-fallback instruction shown in the modal.
        return _bt_failure("pair", exc)
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
    # Persist the toggle FIRST so a flapping radio (profile-switch
    # raises) still leaves the operator's intent recorded — the next
    # boot reconciliation in lifespan picks up the persisted value
    # and re-applies. Then drive the BT card profile to match: HFP
    # (mic + degraded playback) when enabled, A2DP (high-fidelity
    # playback only) when disabled.
    from meeting_scribe.bt import SETTINGS_BT_INPUT_ACTIVE, apply_bt_input_state
    from meeting_scribe.server_support.settings_store import (
        _save_settings_override,
    )

    _save_settings_override({SETTINGS_BT_INPUT_ACTIVE: enabled})
    try:
        reconcile = await apply_bt_input_state(enabled)
    except bt_mod.BluetoothError as exc:
        return _bt_failure("mic", exc)
    return JSONResponse({"ok": True, "enabled": enabled, "reconcile": reconcile})


@router.get("/api/admin/diag/audio")
async def diag_audio_endpoint(request: Request) -> JSONResponse:
    """Plan §B.9: live in-process pipewire reachability.

    The healthcheck script fronts this for ``meeting-scribe doctor``
    and the BT card's "Bluetooth stack unavailable" banner. The control
    plane talks pipewire natively (pw-dump + wpctl) — the legacy
    ``pulseaudio-utils`` package is not required.
    """
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    out: dict[str, Any] = {"pipewire": "unknown", "wpctl": "unknown"}
    proc_pw = await bt_mod._run(["pw-cli", "info", "0"])
    out["pipewire"] = "ok" if proc_pw.returncode == 0 else "fail"
    proc_wp = await bt_mod._run(["wpctl", "status"])
    out["wpctl"] = "ok" if proc_wp.returncode == 0 else "fail"
    return JSONResponse(out)


async def _read_json(request: Request) -> Any:
    try:
        return await request.json()
    except Exception:
        return None
