"""Admin-scope routes — drain, pause/resume translation, and admin
settings GET/PUT.

The settings PUT handler is by far the largest — it's the
single-source-of-truth flow for runtime regdomain / timezone /
tts-voice-mode / wifi-mode / admin-ssid + password changes.
WiFi mode changes go down a 202 Accepted async path because the
cutover takes 5–10 s; everything else applies synchronously and
returns the new effective payload.
"""

from __future__ import annotations

import asyncio
import logging

import fastapi
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from meeting_scribe.runtime import state
from meeting_scribe.server_support.regdomain import (
    _current_regdomain,
    _ensure_regdomain,
    _ensure_regdomain_persistent,
)
from meeting_scribe.server_support.settings_store import (
    _WIFI_REGDOMAIN_OPTIONS,
    _effective_hdmi_enabled,
    _effective_hdmi_idle_sleep_minutes,
    _effective_hdmi_mode,
    _effective_hdmi_rotation,
    _effective_interpretation_enabled,
    _effective_interpretation_idle_drain_ms,
    _effective_interpretation_pause_flush_ms,
    _effective_regdomain,
    _effective_timezone,
    _effective_tts_voice_mode,
    _is_valid_hdmi_idle_sleep,
    _is_valid_hdmi_mode,
    _is_valid_hdmi_rotation,
    _is_valid_regdomain,
    _save_settings_override,
)
from meeting_scribe.server_support.timezone import _is_valid_timezone, _timezone_options

logger = logging.getLogger(__name__)

router = APIRouter()


def _admin_settings_payload() -> dict:
    """Build the GET /api/admin/settings response body.

    Includes current values AND the option lists, so the UI can render
    dropdowns without a second round-trip.
    """
    from meeting_scribe.server_support.settings_store import (
        _load_settings_override as _wifi_load_settings,
    )
    from meeting_scribe.wifi import (
        _load_hotspot_state,
        _nmcli_ap_is_active,
        _nmcli_read_live_ap_credentials,
        _wpa_supplicant_ap_security,
    )

    wifi_settings = _wifi_load_settings()
    wifi_mode = wifi_settings.get("wifi_mode", "admin")
    from meeting_scribe.languages import LANGUAGE_REGISTRY

    local_sink_language_options = [
        {"code": code, "name": lang.name}
        for code, lang in LANGUAGE_REGISTRY.items()
        if lang.tts_native
    ]

    # Live AP state from nmcli/wpa_cli — NOT just the state file
    wifi_active = _nmcli_ap_is_active()
    live_creds = _nmcli_read_live_ap_credentials() if wifi_active else None
    live_security = _wpa_supplicant_ap_security() if wifi_active else None

    # Live mode: derived from /tmp/meeting-hotspot.json which the AP
    # lifecycle writes after every up/down. The persisted ``wifi_mode``
    # is what the operator chose in Settings; the *live* mode is what
    # the box is actually broadcasting right now (could differ if the
    # box is stuck in setup mode, or if the AP came up via a different
    # code path). Surface both so the UI can show a divergence note
    # instead of silently displaying a stale dropdown — the 2026-05-07
    # "the dropdown shows off but Live: Dell Meeting 1618" report.
    hotspot_state = _load_hotspot_state() if wifi_active else None
    wifi_mode_live = (hotspot_state or {}).get("mode") if wifi_active else "off"

    payload = {
        "wifi_regdomain": _effective_regdomain(),
        "wifi_regdomain_current": _current_regdomain(),
        "wifi_regdomain_options": [
            {"code": code, "name": name} for code, name in _WIFI_REGDOMAIN_OPTIONS
        ],
        "wifi_mode": wifi_mode,
        "wifi_mode_live": wifi_mode_live,
        "wifi_mode_options": [
            {"code": "off", "name": "Off"},
            {"code": "meeting", "name": "Meeting (rotating SSID, captive portal)"},
            {"code": "admin", "name": "Admin (fixed SSID, admin UI over WiFi)"},
        ],
        "wifi_active": wifi_active,
        "wifi_ssid": live_creds[0] if live_creds else None,
        "wifi_security": live_security,
        "admin_ssid": wifi_settings.get("admin_ssid", ""),
        "admin_password_set": bool(wifi_settings.get("admin_password")),
        "timezone": _effective_timezone(),
        "timezone_options": _timezone_options(),
        "tts_voice_mode": _effective_tts_voice_mode(),
        "tts_voice_mode_options": [
            {"code": "studio", "name": "Studio voice (Qwen3-TTS, studio quality)"},
            {"code": "cloned", "name": "Participant voice (clone each speaker)"},
        ],
        "admin_tts_language": wifi_settings.get(
            "admin_tts_language", wifi_settings.get("local_sink_language", "en")
        ),
        "room_tts_language": wifi_settings.get("room_tts_language", "all"),
        "local_sink_language": wifi_settings.get(
            "admin_tts_language", wifi_settings.get("local_sink_language", "en")
        ),
        "local_sink_language_options": local_sink_language_options,
        "interpretation_enabled": _effective_interpretation_enabled(),
        "interpretation_pause_flush_ms": _effective_interpretation_pause_flush_ms(),
        "interpretation_idle_drain_ms": _effective_interpretation_idle_drain_ms(),
    }
    # Appliance identity (device pin + full id). Both are stable per-device
    # and safe to surface in the admin UI: the 4-digit pin is already
    # public via the SSID suffix, and the 16-hex id is the unique-per-fleet
    # tag used to derive certs / mDNS names. The Settings panel renders
    # them as a read-only "Device" row so operators can correlate the
    # box they're configuring with stickers / fleet inventory.
    try:
        from meeting_scribe.cli._common import (
            _read_or_mint_appliance_id,
            appliance_pin,
        )

        payload["appliance_pin"] = appliance_pin()
        payload["appliance_id"] = _read_or_mint_appliance_id()
    except Exception:
        # Appliance id not yet minted (fresh tree before ``meeting-scribe
        # setup``). The UI must tolerate ``None`` here.
        payload["appliance_pin"] = None
        payload["appliance_id"] = None

    # HDMI kiosk display settings: current values plus the live
    # connector status (mode list, connected bool) read from
    # ``/run/meeting-scribe/hdmi-status.json``. The admin "HDMI Display"
    # tab uses both halves to render dropdowns and a status indicator
    # without a second round-trip.
    try:
        from meeting_scribe.kiosk.hdmi_status import read_status as _read_hdmi_status

        payload["hdmi_enabled"] = _effective_hdmi_enabled()
        payload["hdmi_mode"] = _effective_hdmi_mode()
        payload["hdmi_rotation"] = _effective_hdmi_rotation()
        payload["hdmi_idle_sleep_minutes"] = _effective_hdmi_idle_sleep_minutes()
        payload["hdmi_rotation_options"] = [0, 90, 180, 270]
        payload["hdmi_status"] = _read_hdmi_status()
    except Exception:
        logger.exception("admin settings: hdmi status read failed")
        payload["hdmi_enabled"] = True
        payload["hdmi_mode"] = "auto"
        payload["hdmi_rotation"] = 0
        payload["hdmi_idle_sleep_minutes"] = 0
        payload["hdmi_rotation_options"] = [0, 90, 180, 270]
        payload["hdmi_status"] = {
            "connected": False,
            "current_mode": None,
            "available_modes": [],
            "rotation": 0,
            "enabled": False,
            "edid_name": None,
            "updated_at": None,
            "source": "sentinel",
        }
    return payload


@router.get("/api/admin/finalize/status")
async def get_admin_finalize_status() -> JSONResponse:
    """Operator-facing snapshot of in-flight Phase B finalize work.

    Surfaces what ``meeting-scribe finalize status`` displays:

    * ``gpu_lease_holder`` — ``"idle"``, ``"recording"``, or ``"phase_b"``.
      Tells the operator whether a meeting is currently recording (the
      GPU is sovereign for that meeting) or whether a Phase B is using
      the GPU for finalize.
    * ``phase_b_tasks`` — list of ``{meeting_id, name, state}`` for
      every Phase B task tracked by the lifecycle module. ``state`` is
      one of ``"running"``, ``"done"``, ``"cancelled"``.
    """
    from meeting_scribe.routes.meeting_lifecycle import _phase_b_tasks
    from meeting_scribe.runtime.gpu_lease import gpu_lease

    tasks_payload = []
    for mid, task in list(_phase_b_tasks.items()):
        if task.cancelled():
            task_state = "cancelled"
        elif task.done():
            task_state = "done"
        else:
            task_state = "running"
        tasks_payload.append(
            {
                "meeting_id": mid,
                "name": task.get_name(),
                "state": task_state,
            }
        )
    return JSONResponse(
        {
            "gpu_lease_holder": gpu_lease().holder,
            "phase_b_tasks": tasks_payload,
        }
    )


@router.post("/api/admin/factory-reset")
async def post_admin_factory_reset(request: fastapi.Request) -> JSONResponse:
    """Wipe setup state + admin credentials so the box re-enters first-touch.

    Calls ``setup_state.factory_reset``: bumps ``auth-version`` so the
    caller's own admin cookie stops working immediately, removes the
    LIVE HMACs, and clears the AP password from settings_store. The
    next ``meeting-scribe restart`` lands in setup mode.

    Admin-cookie-protected — the caller's cookie was valid the moment
    this handler ran but is invalidated before the response returns.
    """
    from meeting_scribe.server_support.admin_guard import _require_admin_response

    blocked = _require_admin_response(request)
    if blocked is not None:
        return blocked

    from meeting_scribe import setup_state
    from meeting_scribe.server_support.settings_store import _save_settings_override

    try:
        _save_settings_override({"admin_password": "", "wifi_mode": "off"})
    except Exception:
        logger.exception("factory_reset: clearing settings_store failed (non-fatal)")

    setup_state.factory_reset()
    logger.warning(
        "admin factory_reset triggered via /api/admin/factory-reset — "
        "next restart re-enters setup mode",
    )
    return JSONResponse(
        {
            "ok": True,
            "message": "Factory reset complete. Restart meeting-scribe to enter setup mode.",
        }
    )


@router.post("/api/admin/drain")
async def post_admin_drain(request: fastapi.Request) -> JSONResponse:
    """Wait up to ``timeout`` seconds for translation + slide work to idle.

    Query params:
      - ``timeout`` (float, default 1.0) — max seconds to wait on this call.
      - ``force`` (bool, default false) — cancel every not-yet-started
        translation item AND abort the in-flight slide pipeline before
        returning.  Used by ``meeting-scribe drain --force`` for incidents
        where an operator chooses to drop in-flight work rather than wait.

    Response body:
      ``{"idle": bool, "translation_active": int, "translation_pending": int,
         "merge_gate_held": bool, "slide_in_flight": {...},
         "force_cancelled": {"translation": int, "slide": bool}?}``

    Callers (``meeting-scribe drain``) poll in a loop until ``idle=true`` or
    their own overall timeout expires.  The endpoint flushes the merge gate
    once at entry so a held final event doesn't hide behind "empty queue".
    """
    try:
        timeout_s = float(request.query_params.get("timeout", "1.0"))
    except ValueError:
        timeout_s = 1.0
    timeout_s = max(0.1, min(timeout_s, 10.0))  # clamp

    force_raw = request.query_params.get("force", "").lower()
    force = force_raw in ("1", "true", "yes")

    # Flush the merge gate so a held final event becomes visible as pending.
    if state.translation_queue is not None:
        await state.translation_queue.flush_merge_gate()

    force_cancelled: dict[str, int | bool] = {}
    if force:
        t_cancelled = state.translation_queue.cancel_all() if state.translation_queue else 0
        s_cancelled = (
            await state.slide_job_runner.cancel_current_job()
            if state.slide_job_runner is not None
            else False
        )
        force_cancelled = {"translation": t_cancelled, "slide": s_cancelled}
        if t_cancelled or s_cancelled:
            logger.warning(
                "drain --force: cancelled %d translation item(s), slide_aborted=%s",
                t_cancelled,
                s_cancelled,
            )

    deadline = asyncio.get_event_loop().time() + timeout_s
    while True:
        t_active = state.translation_queue.active_count() if state.translation_queue else 0
        t_pending = state.translation_queue.pending_count() if state.translation_queue else 0
        t_held = state.translation_queue.merge_gate_held() if state.translation_queue else False
        slide_info = (
            state.slide_job_runner.in_flight()
            if state.slide_job_runner is not None
            else {"running": False}
        )
        idle = (
            t_active == 0 and t_pending == 0 and not t_held and not slide_info.get("running", False)
        )
        if idle or asyncio.get_event_loop().time() >= deadline:
            body: dict[str, object] = {
                "idle": idle,
                "translation_active": t_active,
                "translation_pending": t_pending,
                "merge_gate_held": t_held,
                "slide_in_flight": slide_info,
            }
            if force:
                body["force_cancelled"] = force_cancelled
            return JSONResponse(body)
        await asyncio.sleep(0.1)


@router.post("/api/admin/pause-translation")
async def post_admin_pause_translation() -> JSONResponse:
    """Gate new translation intake.  Already-queued + in-flight items continue.

    Paired with ``/api/admin/resume-translation`` for operator-driven
    model-swap windows.  Called by ``meeting-scribe pause-translation``
    before a drain so new ASR output doesn't pile up against an
    about-to-unload backend.  Idempotent.
    """
    if state.translation_queue is None:
        return JSONResponse({"error": "Translation queue not initialised"}, status_code=503)
    state.translation_queue.pause()
    return JSONResponse({"paused": state.translation_queue.is_paused()})


@router.post("/api/admin/resume-translation")
async def post_admin_resume_translation() -> JSONResponse:
    """Re-open translation intake.  Idempotent."""
    if state.translation_queue is None:
        return JSONResponse({"error": "Translation queue not initialised"}, status_code=503)
    state.translation_queue.resume()
    return JSONResponse({"paused": state.translation_queue.is_paused()})


@router.get("/api/admin/eager-summary-status")
async def get_eager_summary_status() -> JSONResponse:
    """Return live telemetry from the eager-summary loop.

    Lets an operator (or test) prove the loop is alive (``last_start_at``
    recent), making progress (``last_success_at`` within the loop's
    cadence), and not stuck on a backend error (``last_error_code``
    null AFTER ``last_success_at``).

    Never proxies raw exception text — ``last_error_code`` is enum-only.
    """
    m = state._eager_summary_metrics
    return JSONResponse(
        {
            "last_start_at": m.last_start_at,
            "last_success_at": m.last_success_at,
            "last_error_code": m.last_error_code,
            "last_skipped_reason": m.last_skipped_reason,
            "in_flight": m.in_flight,
            "draft_event_count_at_last_run": m.draft_event_count_at_last_run,
            "runs_total": m.runs_total,
            "errors_total": m.errors_total,
            "current_meeting": (
                state.current_meeting.meeting_id if state.current_meeting else None
            ),
        }
    )


@router.get("/api/admin/admin-password")
async def get_admin_password(request: fastapi.Request) -> JSONResponse:
    """Return the appliance admin login password.

    Deterministic per-appliance (``DellMeetingAdmin<NNNN>`` where ``<NNNN>``
    is the 4-digit appliance pin baked into the SSID). The PIN itself is
    public via the SSID, so the password is closer to a memorability
    helper than a secret — real protection is the AP-only network surface
    plus the admin cookie that gates this endpoint.

    Surfaced from the Settings panel Credentials tab so operators can
    recover the password after first-touch setup without resorting to
    ``scripts/show_sidecar_creds.py``. Fetched only on explicit Reveal
    click, never on tab activation — see ``/api/admin/guest-pin`` for the
    always-fetched-on-tab-open companion.
    """
    from meeting_scribe.server_support.admin_guard import _require_admin_response

    blocked = _require_admin_response(request)
    if blocked is not None:
        return blocked

    from meeting_scribe.setup_state import _mint_admin_password

    resp = JSONResponse({"password": _mint_admin_password()})
    resp.headers["Cache-Control"] = "no-store, private"
    resp.headers["Pragma"] = "no-cache"
    return resp


@router.get("/api/admin/guest-pin")
async def get_guest_pin(request: fastapi.Request) -> JSONResponse:
    """Return the 4-digit appliance guest PIN.

    Same 4 digits as the last block of the Admin SSID (already public
    in WiFi range). Fetched on Credentials-tab activation; safe to leave
    in any intermediate cache because it's not a secret.
    """
    from meeting_scribe.server_support.admin_guard import _require_admin_response

    blocked = _require_admin_response(request)
    if blocked is not None:
        return blocked

    from meeting_scribe.setup_state import _mint_guest_pin

    return JSONResponse({"pin": _mint_guest_pin()})


@router.get("/api/admin/settings")
async def get_admin_settings(request: fastapi.Request) -> JSONResponse:
    """Return admin-configurable runtime settings + selectable options.

    Fields:
      - ``wifi_regdomain``     — effective 2-letter country code in force
      - ``wifi_regdomain_current`` — live ``iw reg get`` value
      - ``wifi_regdomain_options`` — [{code, name}] supported-country list
      - ``timezone``           — effective IANA tz (empty = local time)
      - ``timezone_options``   — sorted IANA tz list

    The effective values come from config > persisted override > default,
    so the admin UI always shows the value actually in force.
    """
    return JSONResponse(_admin_settings_payload())


@router.put("/api/admin/settings")
async def put_admin_settings(request: fastapi.Request) -> JSONResponse:
    """Update admin-configurable runtime settings.

    Body accepts any subset of:
      - ``wifi_regdomain``: 2-letter ISO country code from the supported
        list. Persisted, applied via ``iw reg set`` immediately, and
        written to ``/etc/modprobe.d/cfg80211-<code>.conf`` for boot.
      - ``timezone``: IANA timezone name (e.g. ``Asia/Tokyo``) or empty
        string to clear and use the server's local time. Validated via
        ``zoneinfo.ZoneInfo`` before persisting.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be a JSON object"}, status_code=400)

    updates: dict = {}

    if "wifi_regdomain" in body:
        raw_code = body.get("wifi_regdomain")
        if not isinstance(raw_code, str):
            return JSONResponse({"error": "wifi_regdomain must be a string"}, status_code=400)
        code = raw_code.strip().upper()
        if len(code) != 2 or not code.isalpha():
            return JSONResponse(
                {"error": "wifi_regdomain must be a 2-letter ISO country code"},
                status_code=400,
            )
        if not _is_valid_regdomain(code):
            return JSONResponse(
                {"error": (f"wifi_regdomain {code!r} is not in the supported country list")},
                status_code=400,
            )
        updates["wifi_regdomain"] = code

    if "timezone" in body:
        raw_tz = body.get("timezone")
        if not isinstance(raw_tz, str):
            return JSONResponse({"error": "timezone must be a string"}, status_code=400)
        tz_name = raw_tz.strip()
        if tz_name and not _is_valid_timezone(tz_name):
            return JSONResponse(
                {"error": f"timezone {tz_name!r} is not a valid IANA name"},
                status_code=400,
            )
        updates["timezone"] = tz_name

    # HDMI kiosk display settings. Kiosk-runtime picks these up via
    # inotify on the settings file; no live-apply round-trip from this
    # handler. ``hdmi_mode`` is validated against the live wlr-randr
    # mode list cached at ``/run/meeting-scribe/hdmi-status.json``.
    if "hdmi_enabled" in body:
        raw = body.get("hdmi_enabled")
        if not isinstance(raw, bool):
            return JSONResponse({"error": "hdmi_enabled must be boolean"}, status_code=400)
        updates["hdmi_enabled"] = raw

    if "hdmi_mode" in body:
        raw_mode = body.get("hdmi_mode")
        if not isinstance(raw_mode, str):
            return JSONResponse({"error": "hdmi_mode must be a string"}, status_code=400)
        mode = raw_mode.strip()
        if not _is_valid_hdmi_mode(mode):
            return JSONResponse(
                {"error": f"hdmi_mode {mode!r} is not in the current connector's mode list"},
                status_code=400,
            )
        updates["hdmi_mode"] = mode

    if "hdmi_rotation" in body:
        raw_rot = body.get("hdmi_rotation")
        if not _is_valid_hdmi_rotation(raw_rot):
            return JSONResponse(
                {"error": "hdmi_rotation must be one of 0, 90, 180, 270"},
                status_code=400,
            )
        updates["hdmi_rotation"] = raw_rot

    if "hdmi_idle_sleep_minutes" in body:
        raw_sleep = body.get("hdmi_idle_sleep_minutes")
        if not _is_valid_hdmi_idle_sleep(raw_sleep):
            return JSONResponse(
                {"error": "hdmi_idle_sleep_minutes must be int in [0, 240]"},
                status_code=400,
            )
        updates["hdmi_idle_sleep_minutes"] = raw_sleep

    if "tts_voice_mode" in body:
        raw_mode = body.get("tts_voice_mode")
        if not isinstance(raw_mode, str) or raw_mode not in ("studio", "cloned"):
            return JSONResponse(
                {"error": "tts_voice_mode must be 'studio' or 'cloned'"},
                status_code=400,
            )
        updates["tts_voice_mode"] = raw_mode

    for sink_lang_key in ("local_sink_language", "admin_tts_language", "room_tts_language"):
        if sink_lang_key not in body:
            continue
        from meeting_scribe.languages import LANGUAGE_REGISTRY

        raw_lang = body.get(sink_lang_key)
        allowed = {code for code, lang in LANGUAGE_REGISTRY.items() if lang.tts_native}
        if sink_lang_key == "room_tts_language":
            allowed = allowed | {"all"}
        if not isinstance(raw_lang, str) or raw_lang not in allowed:
            return JSONResponse(
                {"error": f"{sink_lang_key} must be a TTS-supported language code"},
                status_code=400,
            )
        updates[
            "admin_tts_language" if sink_lang_key == "local_sink_language" else sink_lang_key
        ] = raw_lang

    if "interpretation_enabled" in body:
        raw_enabled = body.get("interpretation_enabled")
        if not isinstance(raw_enabled, bool):
            return JSONResponse(
                {"error": "interpretation_enabled must be a boolean"},
                status_code=400,
            )
        updates["interpretation_enabled"] = raw_enabled

    for key in ("interpretation_pause_flush_ms", "interpretation_idle_drain_ms"):
        if key in body:
            raw_ms = body.get(key)
            if not isinstance(raw_ms, int) or raw_ms < 100 or raw_ms > 60000:
                return JSONResponse(
                    {"error": f"{key} must be an integer between 100 and 60000"},
                    status_code=400,
                )
            updates[key] = raw_ms

    # ── WiFi fields ──────────────────────────────────────────
    wifi_mode_changed = False
    new_wifi_mode: str | None = None

    if "wifi_mode" in body:
        raw_wm = body.get("wifi_mode")
        if not isinstance(raw_wm, str) or raw_wm not in ("off", "meeting", "admin"):
            return JSONResponse(
                {"error": "wifi_mode must be 'off', 'meeting', or 'admin'"},
                status_code=400,
            )
        from meeting_scribe.server_support.settings_store import (
            _load_settings_override as _wifi_settings,
        )

        # Defaults to ``admin`` to match the boot lifespan default (a fresh
        # box with no persisted ``wifi_mode`` boots into admin, so a Settings
        # save must treat "no prior value" as admin too — otherwise the
        # first save quietly looked like "admin→off" and the rotation logic
        # tore the AP down).
        current_mode = _wifi_settings().get("wifi_mode", "admin")
        if raw_wm != current_mode:
            wifi_mode_changed = True
            new_wifi_mode = raw_wm
        updates["wifi_mode"] = raw_wm

    if "admin_ssid" in body:
        raw_ssid = body.get("admin_ssid")
        if not isinstance(raw_ssid, str):
            return JSONResponse({"error": "admin_ssid must be a string"}, status_code=400)
        ssid = raw_ssid.strip()
        ssid_bytes = ssid.encode("utf-8")
        if not (1 <= len(ssid_bytes) <= 32):
            return JSONResponse({"error": "admin_ssid must be 1-32 bytes"}, status_code=400)
        if not all(0x20 <= ord(c) <= 0x7E for c in ssid):
            return JSONResponse({"error": "admin_ssid must be printable ASCII"}, status_code=400)
        updates["admin_ssid"] = ssid

    if "admin_password" in body:
        raw_pw = body.get("admin_password")
        if not isinstance(raw_pw, str):
            return JSONResponse({"error": "admin_password must be a string"}, status_code=400)
        if not (8 <= len(raw_pw) <= 63):
            return JSONResponse(
                {"error": "admin_password must be 8-63 characters (WPA2/WPA3 constraint)"},
                status_code=400,
            )
        if not all(0x20 <= ord(c) <= 0x7E for c in raw_pw):
            return JSONResponse(
                {"error": "admin_password must be printable ASCII"}, status_code=400
            )
        updates["admin_password"] = raw_pw

    if not updates:
        return JSONResponse({"error": "no recognized settings in body"}, status_code=400)

    # ── WiFi mode change → async cutover with 202 ───────────
    if wifi_mode_changed and new_wifi_mode is not None:
        from meeting_scribe.wifi import build_config, wifi_switch

        # Validate the new config BEFORE committing anything
        try:
            if new_wifi_mode == "off":
                new_cfg = None
            else:
                # Use updates for admin_ssid/admin_password if they were
                # in this same PUT, otherwise build_config reads settings.
                new_cfg = build_config(
                    new_wifi_mode,
                    updates.get("admin_ssid") if new_wifi_mode == "admin" else None,
                    updates.get("admin_password") if new_wifi_mode == "admin" else None,
                    "a",
                    36,
                )
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        # Save non-wifi settings immediately (timezone, regdomain, etc.)
        non_wifi = {
            k: v
            for k, v in updates.items()
            if k not in ("wifi_mode", "admin_ssid", "admin_password")
        }
        if non_wifi:
            _save_settings_override(non_wifi)

        async def _do_switch():
            from meeting_scribe.wifi import wifi_down as _wd

            if new_cfg is None:
                await _wd()
            else:
                await wifi_switch(new_cfg)

        asyncio.create_task(_do_switch())
        return JSONResponse(
            {"status": "switching", "wifi_mode": new_wifi_mode},
            status_code=202,
        )

    # ── Non-wifi-mode-change path (immediate apply) ──────────
    _save_settings_override(updates)

    if (
        any(k.startswith("interpretation_") for k in updates)
        or "local_sink_language" in updates
        or "admin_tts_language" in updates
        or "room_tts_language" in updates
    ):
        try:
            from meeting_scribe.audio.interpretation_buffer import InterpretationBuffer
            from meeting_scribe.audio.local_sink import (
                ensure_local_sink_listener_registered,
            )

            if _effective_interpretation_enabled():
                if state.interpretation_buffer is None:
                    state.interpretation_buffer = InterpretationBuffer(
                        pause_flush_ms=_effective_interpretation_pause_flush_ms(),
                        idle_drain_ms=_effective_interpretation_idle_drain_ms(),
                    )
                else:
                    state.interpretation_buffer.pause_flush_ms = (
                        _effective_interpretation_pause_flush_ms()
                    )
                    state.interpretation_buffer.idle_drain_ms = (
                        _effective_interpretation_idle_drain_ms()
                    )
            elif state.interpretation_buffer is not None:
                await state.interpretation_buffer.set_enabled(False)
                state.interpretation_buffer = None

            ensure_local_sink_listener_registered()
        except Exception:
            logger.exception("interpretation settings live-apply failed")

    for key in ("wifi_regdomain", "timezone"):
        if key in updates:
            try:
                setattr(state.config, key, updates[key])
            except Exception:
                pass

    persistent_ok = True
    runtime_ok = True
    if "wifi_regdomain" in updates:
        loop = asyncio.get_event_loop()
        persistent_ok = await loop.run_in_executor(None, _ensure_regdomain_persistent)
        runtime_ok = await loop.run_in_executor(None, _ensure_regdomain)

    response = _admin_settings_payload()
    response["persistent_ok"] = persistent_ok
    response["runtime_ok"] = runtime_ok
    return JSONResponse(response)


# ── Kiosk integration ────────────────────────────────────────────
#
# Three endpoints power the GB10 cage / chromium HDMI mirror:
#
#   * ``POST /api/admin/kiosk/mint-nonce``    - admin mints a
#       single-use bootstrap nonce that the kiosk-runtime hands to
#       ``/kiosk-bootstrap`` on the loopback listener. The actual
#       cookie issue happens there; this endpoint only stamps the
#       nonce.
#   * ``GET /api/admin/popout-layout``        - read the
#       server-authoritative popout layout the kiosk mirror follows.
#   * ``PUT /api/admin/popout-layout``        - persist + broadcast
#       a new layout. Any admin-cookie tab can write; broadcast
#       fires the ``popout_layout_changed`` WS event so the kiosk
#       (and every other popout tab) mirrors within ~500 ms.
#
# All three are admin-only by virtue of the existing hotspot_guard
# cookie gate; an explicit ``require_role(Role.ADMIN)`` dependency
# layered on top would be redundant but cheap to add later.


@router.post("/api/admin/kiosk/mint-nonce")
async def post_kiosk_mint_nonce(request: fastapi.Request) -> JSONResponse:
    """Mint a single-use bootstrap nonce for the kiosk-runtime.

    The kiosk-runtime calls this endpoint at startup (authenticated as
    admin via the deterministic local admin password) and feeds the
    returned nonce to ``GET http://127.0.0.1:8444/kiosk-bootstrap?nonce=X``
    which exchanges it for a ``scribe_kiosk`` cookie.

    Nonces are 32 random bytes (hex-encoded), 60 s TTL, single-use,
    held in process memory. See ``meeting_scribe.kiosk.nonces``.
    """
    _ = request  # quiet F841
    from meeting_scribe.kiosk.nonces import mint_nonce

    return JSONResponse({"nonce": mint_nonce(), "ttl_seconds": 60})


@router.get("/api/admin/popout-layout")
async def get_admin_popout_layout(request: fastapi.Request) -> JSONResponse:
    """Return the server-authoritative popout layout + monotonic version.

    Shape mirrors the browser-side ``popout_layout_v2`` localStorage
    object plus a ``version`` integer that increments on every PUT.
    Returns ``layout: null, version: 0`` when no layout has been
    persisted yet.
    """
    _ = request
    from meeting_scribe.server_support.settings_store import _load_settings_override

    overrides = _load_settings_override() or {}
    return JSONResponse(
        {
            "layout": overrides.get("popout_layout"),
            "version": int(overrides.get("popout_layout_version", 0)),
        }
    )


@router.put("/api/admin/popout-layout")
async def put_admin_popout_layout(request: fastapi.Request) -> JSONResponse:
    """Persist a new popout layout + broadcast to every subscriber.

    Validation is shape-only: ``layout`` must be a JSON object (or
    ``null`` to clear). Fine-grained schema validation lives in the
    client (popout-layout-storage.js); the server's job is to be the
    single arbiter of "latest version" via the monotonic version
    counter and to fan out the change.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "body_must_be_object"}, status_code=400)

    layout = body.get("layout")
    if layout is not None and not isinstance(layout, dict):
        return JSONResponse({"error": "layout_must_be_object_or_null"}, status_code=400)
    source_tab_id = body.get("source_tab_id")
    if source_tab_id is not None and not isinstance(source_tab_id, str):
        return JSONResponse({"error": "source_tab_id_must_be_string"}, status_code=400)

    from meeting_scribe.server_support.broadcast import _broadcast_json
    from meeting_scribe.server_support.settings_store import (
        _load_settings_override,
    )

    current = _load_settings_override() or {}
    version = int(current.get("popout_layout_version", 0)) + 1
    updates = {"popout_layout": layout, "popout_layout_version": version}
    _save_settings_override(updates)

    await _broadcast_json(
        {
            "type": "popout_layout_changed",
            "layout": layout,
            "version": version,
            "source_tab_id": source_tab_id or "",
        }
    )

    return JSONResponse({"layout": layout, "version": version})
