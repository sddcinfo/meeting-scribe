"""Diagnostics routes — listener telemetry, log tail/stream, issues feed.

Powers the in-app Diagnostics view (system health + log tail +
issues feed) so users don't need to drop into a terminal to
investigate problems.

* ``/api/diag/listener`` and ``/api/diag/listeners`` — ring-buffer
  of audio-listener telemetry pushed by browser tabs every ~2 s.
  Public (no auth gate) because it's the only way the server learns
  what the listener actually sees on a guest device — the alternative
  is asking users to read banners off their phones.

* ``/api/diag/log/server`` and ``/api/diag/log/server/stream`` — admin-
  gated text + SSE views over the in-process server log.

* ``/api/diag/issues`` — admin-gated WARNING+ event feed from the
  ring buffer (used by the on-page issues badge).

* ``/api/debug/diarize`` — diarization backend introspection. Names
  the snapshot dict ``info`` (not ``state``) so it doesn't shadow
  the ``runtime.state`` module — pre-extraction the inner scope
  raised ``UnboundLocalError`` because Python's name resolution made
  the local ``state`` shadow the module on the RHS too. Fixed during
  extraction.
"""

from __future__ import annotations

import logging
import time

import fastapi
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from meeting_scribe import diagnostics as _diag
from meeting_scribe.runtime import state
from meeting_scribe.server_support.admin_guard import _require_admin_response

logger = logging.getLogger(__name__)

router = APIRouter()


# Live telemetry from each Listen client, pushed by the browser every ~2s.
# We can then query GET /api/diag/listeners from the trace tool to see what
# is actually happening on the user's device — context state, queue depth,
# decode counts, errors — without asking them to read a banner off a phone.
_listener_diag: dict[str, dict] = {}  # client_id → state dict (in-memory only)


@router.post("/api/diag/listener")
async def diag_listener_post(request: fastapi.Request) -> JSONResponse:
    """Accept a tiny JSON ping from a browser-side audio listener.

    Body shape (all optional, free-form):
      {
        "client_id":   "<stable per-tab uuid>",
        "page":        "admin" | "guest" | "reader",
        "ctx_state":   "running" | "suspended" | "closed" | "null",
        "ctx_rate":    48000,
        "ws_state":    "OPEN" | "CONNECTING" | "CLOSING" | "CLOSED" | "NULL",
        "primed":      true,
        "queue":       0,
        "bytes_in":    12345,
        "blobs_in":    7,
        "decoded":     7,
        "decode_err":  0,
        "played":      6,
        "last_err":    "",
        "ua_short":    "Safari/iPhone",
      }

    Stored in a tiny in-memory ring (last 16 distinct client_ids). The trace
    tool reads /api/diag/listeners to surface the latest snapshot per client.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "expected object"}, status_code=400)
    client_id = str(body.get("client_id") or "anon")[:64]
    body["_received_at"] = time.time()
    try:
        body["_peer"] = f"{request.client.host}" if request.client else "?"
    except Exception:
        body["_peer"] = "?"
    _listener_diag[client_id] = body
    # Bound the dict so a noisy client can't OOM the server.
    if len(_listener_diag) > 16:
        oldest = sorted(_listener_diag.items(), key=lambda kv: kv[1].get("_received_at", 0))
        for k, _ in oldest[:-16]:
            _listener_diag.pop(k, None)
    return JSONResponse({"ok": True})


@router.get("/api/diag/listeners")
async def diag_listeners_get() -> JSONResponse:
    """Return the latest audio-listener telemetry from every recent client."""
    now = time.time()
    out = []
    for cid, snapshot in _listener_diag.items():
        out.append(
            {
                **snapshot,
                "client_id": cid,
                "age_s": round(now - snapshot.get("_received_at", now), 1),
            }
        )
    out.sort(key=lambda r: r.get("age_s", 0))
    return JSONResponse({"listeners": out})


@router.get("/api/debug/diarize")
async def debug_diarize() -> JSONResponse:
    """Debug endpoint: inspect the diarization backend state.

    Returns the result cache size, time range, pending catch-up queue,
    and other diagnostics for investigating why speakers aren't being
    attributed to events.
    """
    info: dict = {
        "diarize_backend_exists": state.diarize_backend is not None,
        "pending_catchup_events": len(state._pending_speaker_events),
        "catchup_task_running": (
            state._speaker_catchup_task is not None and not state._speaker_catchup_task.done()
        ),
    }
    if state.diarize_backend is not None:
        info["base_offset_samples"] = getattr(state.diarize_backend, "_base_offset", None)
        info["buffer_samples"] = getattr(state.diarize_backend, "_buffer_samples", None)
        info["buffer_threshold_samples"] = getattr(state.diarize_backend, "_buffer_threshold", None)
        cache = getattr(state.diarize_backend, "_result_cache", None)
        if cache is not None:
            info["result_cache_size"] = len(cache)
            if cache:
                items = list(cache.values())
                first = items[0]
                last = items[-1]
                info["result_cache_range_ms"] = {
                    "first_start": first.start_ms,
                    "first_end": first.end_ms,
                    "last_start": last.start_ms,
                    "last_end": last.end_ms,
                }
                # Unique global cluster IDs present
                info["unique_cluster_ids"] = sorted({dr.cluster_id for dr in items})
        # Global centroid state
        gc = getattr(state.diarize_backend, "_global_centroids", None)
        if gc is not None:
            info["global_centroids_count"] = len(gc)
            info["next_global_id"] = getattr(state.diarize_backend, "_next_global_id", None)
        # Failure state
        info["degraded"] = getattr(state.diarize_backend, "degraded", False)
        info["consecutive_failures"] = getattr(state.diarize_backend, "_consecutive_failures", 0)
        info["last_error"] = getattr(state.diarize_backend, "last_error", None)

    # Sample a few pending events so we can see their time ranges
    if state._pending_speaker_events:
        samples = []
        for sid in list(state._pending_speaker_events.keys())[:5]:
            e = state._pending_speaker_events[sid]
            samples.append(
                {
                    "segment_id": sid,
                    "start_ms": getattr(e, "start_ms", None),
                    "end_ms": getattr(e, "end_ms", None),
                    "text_preview": getattr(e, "text", "") or "",
                }
            )
        info["pending_samples"] = samples

    return JSONResponse(info)


@router.get("/api/diag/log/server")
async def get_diag_server_log(
    request: fastapi.Request,
    lines: int = 500,
    level: str | None = None,
    search: str | None = None,
):
    """Tail of the in-process server log.

    ``level`` (info|warning|error|critical) is a *minimum* level filter;
    ``search`` is a case-insensitive substring filter applied after the
    tail so the most-recent matches are always returned.
    """
    blocked = _require_admin_response(request)
    if blocked is not None:
        return blocked
    lines = max(1, min(lines, 5000))
    body = "\n".join(_diag.tail_log_lines(max_lines=lines, level=level, search=search))
    return fastapi.Response(content=body + "\n", media_type="text/plain; charset=utf-8")


@router.get("/api/diag/log/server/stream")
async def stream_diag_server_log(request: fastapi.Request):
    """SSE stream that tails the server log live.

    Each new log line is sent as one ``data:`` event. The stream ends if
    the client disconnects.
    """
    blocked = _require_admin_response(request)
    if blocked is not None:
        return blocked

    from starlette.responses import StreamingResponse

    async def _events():
        # Initial flush: a small tail so the viewer has context the moment
        # the stream opens (otherwise it'd look empty until the next log
        # line lands).
        for line in _diag.tail_log_lines(max_lines=200):
            yield f"data: {line}\n\n"
        async for line in _diag.stream_log_lines():
            if await request.is_disconnected():
                break
            yield f"data: {line}\n\n"

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/diag/issues")
async def get_diag_issues(
    request: fastapi.Request,
    since_id: int | None = None,
    since_ts: float | None = None,
    level: str | None = None,
    component: str | None = None,
    limit: int = 200,
):
    """Recent WARNING+ events from the in-memory ring buffer.

    Pass ``since_id`` from the previous response to get only new events
    (efficient polling). ``component`` matches a logger-name substring,
    e.g. ``translation`` or ``slides``.
    """
    blocked = _require_admin_response(request)
    if blocked is not None:
        return blocked
    ring = _diag.get_ring_buffer()
    items = (
        ring.snapshot(
            since_id=since_id,
            since_ts=since_ts,
            level=level,
            component=component,
            limit=max(1, min(limit, 1000)),
        )
        if ring
        else []
    )
    return JSONResponse({"events": items})
