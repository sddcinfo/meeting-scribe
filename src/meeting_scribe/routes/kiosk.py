"""``/api/kiosk/*`` - read-only endpoints the kiosk role consumes.

All routes are guarded by ``require_kiosk_listener`` so they 404 from
the canonical HTTPS listener, plus ``require_role(Role.ADMIN, Role.KIOSK)``
so an admin can call them too for debugging.

Endpoints:
  * ``GET /api/kiosk/settings``      - narrow projection of admin settings
                                       (appliance_pin, hdmi_*, popout_layout,
                                       active_meeting_id only).
  * ``GET /api/kiosk/popout-layout`` - the server-authoritative popout layout
                                       blob (or ``null`` if no laptop admin
                                       has saved one yet).
"""

from __future__ import annotations

import fastapi
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from meeting_scribe.auth import Role, require_role
from meeting_scribe.auth.guards import require_kiosk_listener
from meeting_scribe.kiosk.hdmi_status import read_status as _read_hdmi_status
from meeting_scribe.runtime import state

router = APIRouter()


def _appliance_pin() -> str:
    """Resolve the 4-digit appliance PIN. Lazy import to avoid early-
    boot circulars during route registration."""
    from meeting_scribe.cli._common import appliance_pin

    return appliance_pin()


def _kiosk_settings_projection() -> dict:
    """Return only the keys the kiosk role is allowed to read."""
    from meeting_scribe.server_support.settings_store import _load_settings_override

    overrides = _load_settings_override() or {}
    return {
        "appliance_pin": _appliance_pin(),
        "hdmi_enabled": bool(overrides.get("hdmi_enabled", True)),
        "hdmi_mode": str(overrides.get("hdmi_mode", "auto")),
        "hdmi_rotation": int(overrides.get("hdmi_rotation", 0)),
        "hdmi_idle_sleep_minutes": int(overrides.get("hdmi_idle_sleep_minutes", 0)),
        "popout_layout": overrides.get("popout_layout"),
        "hdmi_status": _read_hdmi_status(),
        "active_meeting_id": (
            state.current_meeting.meeting_id
            if getattr(state, "current_meeting", None) is not None
            else None
        ),
    }


@router.get(
    "/api/kiosk/settings",
    dependencies=[
        Depends(require_kiosk_listener),
        Depends(require_role(Role.ADMIN, Role.KIOSK)),
    ],
)
async def kiosk_settings(request: fastapi.Request) -> JSONResponse:
    """Narrow projection for the kiosk-runtime + kiosk browser."""
    _ = request  # quiet ruff F841 — fastapi requires the param
    return JSONResponse(_kiosk_settings_projection())


@router.get(
    "/api/kiosk/popout-layout",
    dependencies=[
        Depends(require_kiosk_listener),
        Depends(require_role(Role.ADMIN, Role.KIOSK)),
    ],
)
async def kiosk_popout_layout(request: fastapi.Request) -> JSONResponse:
    """Return the server-authoritative popout layout or ``null``."""
    _ = request
    from meeting_scribe.server_support.settings_store import _load_settings_override

    overrides = _load_settings_override() or {}
    return JSONResponse(
        {
            "layout": overrides.get("popout_layout"),
            "version": int(overrides.get("popout_layout_version", 0)),
        }
    )
