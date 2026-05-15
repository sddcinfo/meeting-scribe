"""Public admin endpoints for the speakerphone subsystem.

Cookie-session admin auth on every endpoint (same guard the existing
audio/BT/WAN admin routes use). Exposes the Hardware tab's read/write
surface to the GUI:

* ``GET    /api/admin/speakerphone/state`` — live device + interpretation snapshot
* ``GET    /api/admin/speakerphone/mapping`` — full sidecar document + ETag
* ``PUT    /api/admin/speakerphone/mapping`` — full-document replace with If-Match
* ``PATCH  /api/admin/speakerphone/mapping`` — RFC 6902 JSON-Patch single edits
* ``POST   /api/admin/speakerphone/led-test`` — preview LED patterns
* ``POST   /api/admin/speakerphone/reset-defaults`` — restore canonical mapping

Mutating routes never bypass the admin guard. The matching internal
namespace (``/api/internal/speakerphone/*``) is for the daemon and is
only served over the Unix-domain socket.
"""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from meeting_scribe.server_support.admin_guard import _require_admin_response
from meeting_scribe.speakerphone import api as sp_api
from meeting_scribe.speakerphone import mapping

logger = logging.getLogger(__name__)

router = APIRouter()


# ── compliance cache ─────────────────────────────────────────────────
# Compliance does a 3-5 second pw-record + FFT. Caching the last
# result for ``_COMPLIANCE_CACHE_SECONDS`` keeps the admin SPA chip
# cheap to poll (one capture per minute, not per request).
_COMPLIANCE_CACHE_SECONDS = 60.0
_compliance_cache: dict | None = None
_compliance_cache_ts: float = 0.0
_compliance_lock = asyncio.Lock()


def _bad_request(msg: str) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=400)


@router.get("/api/admin/speakerphone/state")
async def state_get(request: Request) -> JSONResponse:
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    return JSONResponse(sp_api.build_state_payload())


@router.get("/api/admin/speakerphone/wideband-status")
async def wideband_status_get(request: Request) -> JSONResponse:
    """Last-known SP325 wideband compliance state.

    Surfaces a small payload the admin SPA polls every ~minute to
    render a status chip in the header. Visibility-only — does not
    gate any meeting flow. Result is cached for 60 s.

    Response::

        {
          "status": "pass|warn|fail|unavailable",
          "high_band_pct": 2.5,
          "rolloff_3400hz_pct": 2.8,
          "reason": "…",
          "fetched_at": 1715623012.4,
          "age_seconds": 12.0,
          "cached": true
        }
    """
    deny = _require_admin_response(request)
    if deny is not None:
        return deny

    global _compliance_cache, _compliance_cache_ts

    now = time.time()
    cached = (
        _compliance_cache is not None and (now - _compliance_cache_ts) < _COMPLIANCE_CACHE_SECONDS
    )
    if cached:
        return JSONResponse(
            {
                **_compliance_cache,  # type: ignore[arg-type]
                "fetched_at": _compliance_cache_ts,
                "age_seconds": round(now - _compliance_cache_ts, 1),
                "cached": True,
            }
        )

    # Run a fresh probe under a lock so concurrent requests don't
    # stampede pw-record.
    async with _compliance_lock:
        # Double-check after acquiring the lock — another request may
        # have populated the cache while we waited.
        if (
            _compliance_cache is not None
            and (time.time() - _compliance_cache_ts) < _COMPLIANCE_CACHE_SECONDS
        ):
            return JSONResponse(
                {
                    **_compliance_cache,
                    "fetched_at": _compliance_cache_ts,
                    "age_seconds": round(time.time() - _compliance_cache_ts, 1),
                    "cached": True,
                }
            )

        try:
            from meeting_scribe.speakerphone import compliance
            from meeting_scribe.speakerphone.daemon import _guess_pipewire_source_name

            device_key = "413c:8223"  # SP325 default; future: walk known catalog
            node = _guess_pipewire_source_name(device_key)
            if not node:
                payload = {
                    "status": "unavailable",
                    "reason": "no PipeWire source node for SP325 — device not connected?",
                }
            else:
                min_hbp, min_rolloff = compliance.expected_thresholds(device_key)
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: compliance.probe_device(
                        node,
                        capture_seconds=3.0,
                        min_high_band_pct=min_hbp,
                        min_rolloff_pct=min_rolloff,
                    ),
                )
                payload = {
                    "status": result.status,
                    "high_band_pct": result.high_band_pct,
                    "rolloff_3400hz_pct": result.rolloff_3400hz_pct,
                    "reason": getattr(result, "reason", None),
                    "device_key": device_key,
                }
        except Exception as e:
            logger.exception("wideband-status probe failed")
            payload = {"status": "unavailable", "reason": str(e)}

        _compliance_cache = payload
        _compliance_cache_ts = time.time()

    return JSONResponse(
        {
            **payload,
            "fetched_at": _compliance_cache_ts,
            "age_seconds": 0.0,
            "cached": False,
        }
    )


@router.post("/api/admin/speakerphone/wideband-apply")
async def wideband_apply_post(request: Request) -> JSONResponse:
    """Operator-initiated wideband re-apply.

    Wraps ``Sp325HidClient.apply_wideband_good`` so the admin can flip
    the SP325 from narrowband back to wideband on demand. Same recipe
    the daemon runs at attach time, but accessible from the GUI's
    Audio popover.

    Returns the applied command set on success. Does not include the
    post-settle compliance verification — the SPA chip polls
    ``wideband-status`` separately for that.
    """
    deny = _require_admin_response(request)
    if deny is not None:
        return deny

    global _compliance_cache, _compliance_cache_ts

    def _apply() -> dict:
        from meeting_scribe.speakerphone.sp325_hid import (
            Sp325Error,
            Sp325HidClient,
        )

        try:
            with Sp325HidClient.open_default() as cli:
                return {"ok": True, "applied": cli.apply_wideband_good(settle_seconds=15.0)}
        except Sp325Error as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _apply)

    # Invalidate the compliance cache so the next status poll re-probes.
    _compliance_cache = None
    _compliance_cache_ts = 0.0

    status = 200 if result.get("ok") else 500
    return JSONResponse(result, status_code=status)


@router.get("/api/admin/speakerphone/mapping")
async def mapping_get(request: Request) -> JSONResponse:
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    doc = mapping.load()
    etag = mapping.compute_etag(doc)
    resp = JSONResponse(doc)
    resp.headers["ETag"] = f'"{etag}"'
    return resp


@router.put("/api/admin/speakerphone/mapping")
async def mapping_put(request: Request) -> Response:
    """Full-document replace with required ``If-Match`` ETag check."""
    deny = _require_admin_response(request)
    if deny is not None:
        return deny

    if_match = request.headers.get("If-Match", "").strip().strip('"')
    if not if_match:
        return _bad_request("If-Match header is required for PUT")

    try:
        new_doc = await request.json()
    except Exception:
        return _bad_request("JSON object expected")
    if not isinstance(new_doc, dict):
        return _bad_request("JSON object expected")

    current = mapping.load()
    try:
        merged = mapping.replace_full(current, new_doc, etag=if_match)
    except mapping.StaleEtagError as exc:
        return JSONResponse(
            {
                "error": "etag mismatch",
                "expected_etag": exc.expected_etag,
                "actual_etag": exc.actual_etag,
            },
            status_code=412,
        )
    except mapping.MappingValidationError as exc:
        return _bad_request(str(exc))

    mapping.save(merged)
    new_etag = mapping.compute_etag(merged)
    resp = JSONResponse(merged)
    resp.headers["ETag"] = f'"{new_etag}"'
    return resp


@router.patch("/api/admin/speakerphone/mapping")
async def mapping_patch(request: Request) -> Response:
    """Apply RFC 6902 JSON-Patch ops against the mapping document.

    Path allow-list lives in :func:`mapping.apply_patch`; an op against
    any other path returns 400 with the offending path echoed back.
    """
    deny = _require_admin_response(request)
    if deny is not None:
        return deny

    try:
        ops = await request.json()
    except Exception:
        return _bad_request("JSON array of ops expected")
    if not isinstance(ops, list):
        return _bad_request("PATCH body must be a list of ops")

    current = mapping.load()
    try:
        merged = mapping.apply_patch(current, ops)
    except mapping.MappingValidationError as exc:
        return _bad_request(str(exc))

    mapping.save(merged)
    new_etag = mapping.compute_etag(merged)
    resp = JSONResponse(merged)
    resp.headers["ETag"] = f'"{new_etag}"'
    return resp


@router.post("/api/admin/speakerphone/reset-defaults")
async def reset_defaults(request: Request) -> JSONResponse:
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    doc = mapping.default_document()
    mapping.save(doc)
    new_etag = mapping.compute_etag(doc)
    resp = JSONResponse(doc)
    resp.headers["ETag"] = f'"{new_etag}"'
    return resp


@router.post("/api/admin/speakerphone/speak/preview")
async def speak_preview(request: Request) -> JSONResponse:
    """Preview a button-feedback label from the Hardware tab.

    Body: ``{label_id: str, language?: str, overrides?: dict}``.

    Calls ``apply_speak(..., respect_enabled=False)`` so the audition
    works even when ``button_feedback.enabled`` is False — the
    operator needs to hear a label BEFORE deciding to turn feedback
    on. The Test buttons in the Hardware tab pass the current
    (possibly unsaved) text input value as an inline ``overrides``
    entry so the audition reflects what's actually typed.

    All other validation and the TTS-backend reservation logic are
    identical to the internal route.
    """
    from meeting_scribe.speakerphone import api as sp_api

    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    try:
        body = await request.json()
    except Exception:
        return _bad_request("JSON object expected")
    if not isinstance(body, dict):
        return _bad_request("JSON object expected")

    label_id = body.get("label_id")
    if not isinstance(label_id, str) or not label_id:
        return _bad_request("label_id (string) is required")
    language = body.get("language")
    if language is not None and not isinstance(language, str):
        return _bad_request("language must be a string")
    overrides = body.get("overrides")
    if overrides is not None and not isinstance(overrides, dict):
        return _bad_request("overrides must be an object")

    try:
        payload = await sp_api.apply_speak(
            label_id=label_id,
            language=language,
            overrides_inline=overrides,
            respect_enabled=False,
        )
    except sp_api.FeedbackError as exc:
        return JSONResponse(
            {"error": str(exc), "label_id": label_id},
            status_code=400,
        )
    return JSONResponse(payload)


@router.post("/api/admin/speakerphone/led-test")
async def led_test(request: Request) -> JSONResponse:
    deny = _require_admin_response(request)
    if deny is not None:
        return deny
    # The GUI's LED test is a no-op on the server: the daemon receives
    # the request via the mapping document polling and plays the
    # patterns on the actual hardware. We respond with the patterns
    # the daemon will cycle so the GUI can show inline "preview"
    # buttons. See speakerphone/api.py:play_led_test for the actual
    # cadence the daemon uses.
    from meeting_scribe.speakerphone.constants import LED_PATTERNS

    return JSONResponse(
        {
            "patterns": list(LED_PATTERNS.keys()),
            "note": "Daemon plays each pattern for ~1s. Watch the Mute LED ring.",
        },
    )
