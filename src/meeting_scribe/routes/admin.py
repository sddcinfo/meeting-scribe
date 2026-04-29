"""Admin-scope routes — drain, pause/resume translation, refinement
stats, polished status, and admin settings GET/PUT.

The settings PUT handler is by far the largest — it's the
single-source-of-truth flow for runtime regdomain / timezone /
dev-mode / tts-voice-mode / wifi-mode / admin-ssid + password
changes. WiFi mode changes go down a 202 Accepted async path
because the cutover takes 5–10 s; everything else applies
synchronously and returns the new effective payload.
"""

from __future__ import annotations

import asyncio
import logging

import fastapi
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from meeting_scribe.runtime import state
from meeting_scribe.server_support.refinement_drains import (
    _drain_entry_to_dict,
    _find_drains_by_meeting,
    _refinement_drains,
)
from meeting_scribe.server_support.regdomain import (
    _current_regdomain,
    _ensure_regdomain,
    _ensure_regdomain_persistent,
)
from meeting_scribe.server_support.settings_store import (
    _WIFI_REGDOMAIN_OPTIONS,
    _effective_regdomain,
    _effective_timezone,
    _effective_tts_voice_mode,
    _is_dev_mode,
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
        _nmcli_ap_is_active,
        _nmcli_read_live_ap_credentials,
        _wpa_supplicant_ap_security,
    )

    wifi_settings = _wifi_load_settings()
    wifi_mode = wifi_settings.get("wifi_mode", "off")

    # Live AP state from nmcli/wpa_cli — NOT just the state file
    wifi_active = _nmcli_ap_is_active()
    live_creds = _nmcli_read_live_ap_credentials() if wifi_active else None
    live_security = _wpa_supplicant_ap_security() if wifi_active else None

    payload = {
        "wifi_regdomain": _effective_regdomain(),
        "wifi_regdomain_current": _current_regdomain(),
        "wifi_regdomain_options": [
            {"code": code, "name": name} for code, name in _WIFI_REGDOMAIN_OPTIONS
        ],
        "wifi_mode": wifi_mode,
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
        "dev_mode": _is_dev_mode(),
        "tts_voice_mode": _effective_tts_voice_mode(),
        "tts_voice_mode_options": [
            {"code": "studio", "name": "Studio voice (Qwen3-TTS, studio quality)"},
            {"code": "cloned", "name": "Participant voice (clone each speaker)"},
        ],
    }
    return payload


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


@router.get("/api/admin/refinement-stats")
async def get_admin_refinement_stats(
    drain_id: int | None = None,
    meeting_id: str | None = None,
) -> JSONResponse:
    """Return refinement counter snapshots for validation.

    Resolution order:
      1. ``drain_id`` (unambiguous) — returns the exact entry if present.
      2. ``meeting_id`` — returns the most recent drain entry for that
         meeting, plus ``other_drain_ids`` if earlier drains exist, plus
         a ``live`` block if the worker for that meeting is still
         running.
      3. Otherwise 404 — caller is asking about a meeting that never
         started the worker.
    """
    if drain_id is not None:
        entry = next((e for e in _refinement_drains if e.drain_id == drain_id), None)
        if entry is None:
            return JSONResponse(
                {"error": f"no drain entry for drain_id={drain_id}"},
                status_code=404,
            )
        return JSONResponse(_drain_entry_to_dict(entry))

    if meeting_id:
        entries = _find_drains_by_meeting(meeting_id)
        live_block: dict | None = None
        if (
            state.refinement_worker is not None
            and getattr(state.refinement_worker, "_meeting_id", None) == meeting_id
        ):
            live_block = {
                "meeting_id": meeting_id,
                "translate_calls": state.refinement_worker.translate_call_count,
                "asr_calls": state.refinement_worker.asr_call_count,
                "errors_at_stop": state.refinement_worker.last_error_count,
            }
        if not entries and live_block is None:
            return JSONResponse(
                {"error": (f"no active or recent refinement for meeting_id={meeting_id}")},
                status_code=404,
            )
        # Most recent = last entry with matching meeting_id (list is append-only).
        newest = entries[-1] if entries else None
        others = [e.drain_id for e in entries[:-1]] if entries else []
        return JSONResponse(
            {
                "drain": _drain_entry_to_dict(newest) if newest else None,
                "live": live_block,
                "other_drain_ids": others,
            }
        )

    return JSONResponse(
        {"error": "must pass drain_id or meeting_id"},
        status_code=400,
    )


@router.get("/api/meeting/{meeting_id}/polished-status")
async def get_polished_status(meeting_id: str) -> JSONResponse:
    """Polish-drain progress for a meeting.

    Returns the latest drain entry for the meeting (most-recent-wins)
    with the counter snapshot inlined so a polling client fetches state
    + counters in one round-trip.  If no drain entry exists but
    ``polished.json`` is on disk (post-restart state), returns
    ``state=complete`` with the file's mtime so the harness doesn't
    false-alarm after a server restart.
    """
    entries = _find_drains_by_meeting(meeting_id)
    if entries:
        newest = entries[-1]
        body = _drain_entry_to_dict(newest)
        polished_path = state.storage._meeting_dir(meeting_id) / "polished.json"
        body["polished_json_mtime"] = (
            polished_path.stat().st_mtime if polished_path.is_file() else None
        )
        return JSONResponse(body)

    # No registry entry — but the file may be on disk from a previous
    # process. Treat that as "complete" from disk.
    polished_path = state.storage._meeting_dir(meeting_id) / "polished.json"
    if polished_path.is_file():
        return JSONResponse(
            {
                "meeting_id": meeting_id,
                "state": "complete",
                "polished_json_mtime": polished_path.stat().st_mtime,
                "drain_id": None,
                "note": "read from disk — no in-memory drain entry",
            }
        )

    return JSONResponse(
        {
            "meeting_id": meeting_id,
            "state": "absent",
            "polished_json_mtime": None,
            "drain_id": None,
        },
        status_code=404,
    )


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

    if "dev_mode" in body:
        raw_dev = body.get("dev_mode")
        if not isinstance(raw_dev, bool):
            return JSONResponse({"error": "dev_mode must be a boolean"}, status_code=400)
        updates["dev_mode"] = raw_dev

    if "tts_voice_mode" in body:
        raw_mode = body.get("tts_voice_mode")
        if not isinstance(raw_mode, str) or raw_mode not in ("studio", "cloned"):
            return JSONResponse(
                {"error": "tts_voice_mode must be 'studio' or 'cloned'"},
                status_code=400,
            )
        updates["tts_voice_mode"] = raw_mode

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

        current_mode = _wifi_settings().get("wifi_mode", "off")
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
