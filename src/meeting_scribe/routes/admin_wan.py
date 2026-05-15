"""Admin WAN management API endpoints.

Wires :mod:`meeting_scribe.wifi_wan` into typed admin routes that the
admin-UI WAN tab calls. Every route here is gated by
``_require_admin_response`` (cookie auth) and CSRF-protected by the
Origin allowlist middleware.

Profile resources are addressed by stable uuid4 ``id`` — SSID is
display-only. Full ids are returned unsanitized so the admin UI can
copy/paste them verbatim (per ``feedback_no_id_truncation``).

Endpoints:

* ``GET    /api/admin/wan/status``        — per-iface state + active default
* ``GET    /api/admin/wan/profiles``      — list (PSK is NEVER returned)
* ``POST   /api/admin/wan/profiles``      — add; body ``{ssid, psk_ref, bssid?}``
* ``DELETE /api/admin/wan/profiles/{id}`` — remove by uuid4
* ``POST   /api/admin/wan/profiles/{id}/set-active`` — mark active
* ``POST   /api/admin/wan/up``            — body ``{id}``; bring up STA
* ``POST   /api/admin/wan/down``          — tear down STA
* ``GET    /api/admin/wan/scan``          — iw scan results
"""

from __future__ import annotations

import logging
import uuid as _uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from meeting_scribe.server_support import secrets, settings_store
from meeting_scribe.server_support.admin_guard import _require_admin_response

logger = logging.getLogger(__name__)
router = APIRouter()


def _bad_request(detail: str, *, code: int = 400, **extra) -> JSONResponse:
    payload = {"error": detail}
    payload.update(extra)
    return JSONResponse(payload, status_code=code)


def _public_profile(profile: dict) -> dict:
    """Strip any non-public fields. PSK refs are public; PSK plaintext never is."""
    return {
        "id": profile["id"],
        "ssid": profile["ssid"],
        "bssid": profile.get("bssid"),
        "band": profile.get("band") or "auto",
        "psk_ref": profile["psk_ref"],
        "regdomain": profile.get("regdomain"),
        "last_seen": profile.get("last_seen"),
    }


# ─── status / scan ─────────────────────────────────────────────


@router.get("/api/admin/wan/status")
async def get_wan_status(request: Request) -> JSONResponse:
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    from meeting_scribe import wifi_wan

    return JSONResponse(await wifi_wan.wan_status())


@router.get("/api/admin/wan/mode")
async def get_wan_mode(request: Request) -> JSONResponse:
    """Return the current egress mode + its provenance."""
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    return JSONResponse(
        {
            "mode": settings_store._effective_wan_egress_mode(),
            "source": settings_store._wan_egress_mode_source(),
        }
    )


@router.put("/api/admin/wan/mode")
async def put_wan_mode(request: Request) -> JSONResponse:
    """Set the egress mode + trigger a firewall reconcile.

    ``PUT`` always stamps source as ``"operator"`` so the value sticks
    against future ``default`` migrations.
    """
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    try:
        body = await request.json()
    except ValueError, RuntimeError:
        return _bad_request("body must be JSON")
    if not isinstance(body, dict):
        return _bad_request("body must be a JSON object")
    mode = body.get("mode")
    if not settings_store._is_valid_egress_mode(mode):
        return _bad_request("mode must be block|gateway|captive")
    try:
        settings_store._set_wan_egress_mode(mode)  # source="operator"
    except ValueError as exc:
        return _bad_request(str(exc))
    # Apply the new posture immediately.
    try:
        from meeting_scribe.wifi import reconcile_network_state

        await reconcile_network_state()
    except Exception as exc:
        return JSONResponse(
            {
                "mode": mode,
                "source": "operator",
                "warning": f"reconcile failed: {exc!s}",
            },
            status_code=200,
        )
    return JSONResponse({"mode": mode, "source": "operator"})


@router.get("/api/admin/wan/scan")
async def get_wan_scan(request: Request) -> JSONResponse:
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    from meeting_scribe import wifi_wan

    entries = await wifi_wan.scan_upstream()
    groups = wifi_wan.consolidate_scan(entries)
    return JSONResponse(
        {
            # Consolidated form is what the admin UI renders — one row
            # per (SSID, security). Bands tells the UI which radios
            # are available for the band-preference radio.
            "networks": [
                {
                    "ssid": g.ssid,
                    "security": "wpa" if g.rsn_present else "open",
                    "bands": list(g.bands),
                    "best_signal_dbm": g.best_signal_dbm,
                    "best_signal_band": g.best_signal_band,
                    "ap_count": g.ap_count,
                    "channels": list(g.channels),
                }
                for g in groups
            ],
            # Raw BSSes for the few callers that want BSS-level detail
            # (CLI ``wifi wan scan --raw``, future advanced tooling).
            "entries": [
                {
                    "ssid": e.ssid,
                    "bssid": e.bssid,
                    "channel": e.channel,
                    "signal_dbm": e.signal_dbm,
                    "security": "wpa" if e.rsn_present else "open",
                }
                for e in entries
            ],
        }
    )


# ─── profiles CRUD ─────────────────────────────────────────────


@router.get("/api/admin/wan/profiles")
async def get_wan_profiles(request: Request) -> JSONResponse:
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    return JSONResponse(
        {
            "profiles": [_public_profile(p) for p in settings_store._load_wan_profiles()],
            "active_id": settings_store._effective_wan_active_profile_id(),
        }
    )


@router.post("/api/admin/wan/profiles")
async def post_wan_profile(request: Request) -> JSONResponse:
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    try:
        body = await request.json()
    except ValueError, RuntimeError:
        return _bad_request("body must be JSON")
    if not isinstance(body, dict):
        return _bad_request("body must be a JSON object")
    ssid = body.get("ssid")
    psk_ref = body.get("psk_ref")
    bssid = body.get("bssid")
    band = body.get("band") or "auto"
    is_open = bool(body.get("open"))
    if not settings_store._is_valid_ssid(ssid):
        return _bad_request("invalid ssid")
    # Open networks: open=true, psk_ref omitted. The flag is required
    # so a forgotten psk_ref can never become an accidental open profile.
    if is_open and psk_ref:
        return _bad_request("open=true cannot be combined with psk_ref")
    if not is_open and not psk_ref:
        return _bad_request("psk_ref is required (or set open=true for an open network)")
    if psk_ref and not settings_store._is_valid_psk_ref(psk_ref):
        return _bad_request("invalid psk_ref")
    if bssid is not None and not settings_store._is_valid_bssid(bssid):
        return _bad_request("invalid bssid")
    if not settings_store._is_valid_wan_band(band):
        return _bad_request("invalid band; expected auto|a|bg")
    # Validate that psk_ref actually exists in the age store (skip for
    # OPEN profiles and for explicit setup-page opt-outs).
    if psk_ref and not body.get("skip_psk_validation") and not secrets.psk_ref_exists(psk_ref):
        return _bad_request(
            "psk_ref not present in credentials store",
            code=422,
            psk_ref=psk_ref,
        )
    profile = {
        "id": str(_uuid.uuid4()),
        "ssid": ssid,
        "bssid": bssid,
        "band": band,
        "psk_ref": psk_ref,  # None for open networks
        "regdomain": None,
        "last_seen": None,
    }
    settings_store._save_wan_profile(profile)
    return JSONResponse({"profile": _public_profile(profile)}, status_code=201)


@router.delete("/api/admin/wan/profiles/{profile_id}")
async def delete_wan_profile(profile_id: str, request: Request) -> JSONResponse:
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    if not settings_store._is_valid_uuid4_str(profile_id):
        return _bad_request("invalid profile id")
    if not settings_store._delete_wan_profile(profile_id):
        return _bad_request("no profile with that id", code=404)
    return JSONResponse({"deleted": profile_id})


@router.post("/api/admin/wan/profiles/{profile_id}/set-active")
async def post_wan_profile_set_active(profile_id: str, request: Request) -> JSONResponse:
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    if not settings_store._is_valid_uuid4_str(profile_id):
        return _bad_request("invalid profile id")
    try:
        settings_store._set_wan_active_profile_id(profile_id)
    except ValueError as exc:
        return _bad_request(str(exc), code=404)
    return JSONResponse({"active_id": profile_id})


# ─── up / down ─────────────────────────────────────────────────


@router.post("/api/admin/wan/up")
async def post_wan_up(request: Request) -> JSONResponse:
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    try:
        body = await request.json()
    except ValueError, RuntimeError:
        return _bad_request("body must be JSON")
    if not isinstance(body, dict):
        return _bad_request("body must be a JSON object")
    profile_id = body.get("id")
    if not settings_store._is_valid_uuid4_str(profile_id):
        return _bad_request("invalid id")
    from meeting_scribe import wifi_wan

    try:
        await wifi_wan.wan_up(profile_id)
    except (secrets.SecretNotFoundError, secrets.SecretDecryptError) as exc:
        logger.warning("wan_up: PSK resolve failed for %s: %s", profile_id, exc)
        return _bad_request("psk_resolve_failed", code=422)
    except ValueError as exc:
        return _bad_request(str(exc), code=404)
    except RuntimeError as exc:
        logger.warning("wan_up failed for %s: %s", profile_id, exc)
        return _bad_request("wan_up_failed", code=500)
    return JSONResponse({"active_id": profile_id})


@router.post("/api/admin/wan/down")
async def post_wan_down(request: Request) -> JSONResponse:
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    from meeting_scribe import wifi_wan

    await wifi_wan.wan_down()
    return JSONResponse({"ok": True})
