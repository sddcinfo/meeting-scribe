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


# Client-side error ring-buffer. Browser tabs POST to this endpoint
# from window.onerror / unhandledrejection / explicit reportClientError()
# calls so we can investigate UI bugs without asking the operator to
# open DevTools. Bounded ring so a buggy tab in a tight loop can't
# OOM the server. Read by ``GET /api/diag/client-errors`` and by the
# admin diagnostics view.
_client_errors: list[dict] = []
_CLIENT_ERROR_RING_SIZE = 256


@router.post("/api/diag/client-error")
async def diag_client_error_post(request: fastapi.Request) -> JSONResponse:
    """Accept a single browser-side error report.

    Body shape (all optional, free-form):
      {
        "client_id":  "<stable per-tab uuid>",
        "page":       "admin" | "guest" | "reader" | "popout" | …,
        "kind":       "uncaught" | "unhandled-rejection" | "console-error"
                      | "manual",
        "message":    "Uncaught TypeError: …",
        "stack":      "TypeError: x …\n    at f (admin-boot.js:1234)",
        "url":        "https://192.168.1.168/#meeting/…",
        "user_agent": "…",
        "viewport":   {"w": 1920, "h": 1080},
        "context":    {…free-form per-call payload…},
      }

    No auth gate (intentionally) — admin-cookie requirement would block
    pre-sign-in errors (the captive portal, the wizard) which are exactly
    the cases where headless debugging is most useful. The sender writes
    the page identity so the receiver knows where the report came from.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "expected object"}, status_code=400)
    body.setdefault("kind", "unknown")
    body["received_at"] = time.time()
    try:
        body["peer"] = f"{request.client.host}" if request.client else "?"
    except Exception:
        body["peer"] = "?"
    # Truncate any oversized strings before storage so a stack trace
    # loop doesn't grow per-entry without bound.
    for k, v in list(body.items()):
        if isinstance(v, str) and len(v) > 8192:
            body[k] = v[:8192] + "…[truncated]"
    _client_errors.append(body)
    # Surface to the server log so the issues feed picks it up too;
    # WARNING level keeps it visible without spamming on every routine
    # console.error from third-party scripts.
    logger.warning(
        "client-error %s/%s on %s: %s",
        body.get("kind"),
        body.get("page", "?"),
        body.get("url", "?"),
        (body.get("message") or "")[:300],
    )
    if len(_client_errors) > _CLIENT_ERROR_RING_SIZE:
        # Drop the oldest 25% in one slice, not one-at-a-time, so a
        # noisy client doesn't pay the O(N) cost on every POST.
        del _client_errors[: len(_client_errors) - _CLIENT_ERROR_RING_SIZE]
    return JSONResponse({"ok": True, "stored": len(_client_errors)})


@router.get("/api/diag/client-errors")
async def diag_client_errors_get(request: fastapi.Request) -> JSONResponse:
    """Return the most recent browser-side error reports.

    Admin-gated read because some payloads contain URL/page paths that
    leak meeting ids. Writes are unauthenticated (see POST handler).
    """
    blocked = _require_admin_response(request)
    if blocked is not None:
        return blocked
    now = time.time()
    out = [
        {**err, "age_s": round(now - err.get("received_at", now), 1)}
        for err in _client_errors[-128:]
    ]
    return JSONResponse({"errors": out, "total": len(_client_errors)})


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


# ── GPU topology ────────────────────────────────────────────────────────
#
# Shell out to ``sddc gpu top --json`` so this server has zero
# vLLM-metrics / nvidia-smi parsing logic of its own — sddc-cli is the
# single source of truth.  The 2 s SWR cache mirrors the pattern in
# gpu_monitor._get_container_stats: serve stale immediately, refresh in
# the background. Without it a stuck ``--query-compute-apps`` call would
# block every diagnostics-panel poll.
import asyncio as _gpu_asyncio
import json as _gpu_json
import os as _gpu_os
import shutil as _gpu_shutil
import threading as _gpu_threading
from pathlib import Path as _GpuPath

_GPU_CACHE: dict | None = None
_GPU_CACHE_TS: float = 0.0
_GPU_CACHE_TTL = 2.0
_GPU_REFRESH_LOCK = _gpu_threading.Lock()
_GPU_REFRESH_IN_FLIGHT = False


def _resolve_sddc_cli() -> str | None:
    """Find the ``sddc`` binary, honoring PATH first then well-known
    install locations.

    The systemd user unit runs with a minimal ``PATH`` that doesn't
    include ``~/.local/share/mise/installs/python/*/bin``, so a plain
    ``shutil.which("sddc")`` lookup misses it even when sddc-cli is
    installed and works fine from an interactive shell. Probe the
    common alternatives so the diagnostics panel doesn't go dark just
    because the launcher is on a path systemd doesn't inherit.
    """
    found = _gpu_shutil.which("sddc")
    if found:
        return found
    candidates = [
        _GpuPath.home() / ".local" / "bin" / "sddc",
        _GpuPath("/usr/local/bin/sddc"),
    ]
    mise_python = _GpuPath.home() / ".local" / "share" / "mise" / "installs" / "python"
    if mise_python.is_dir():
        for ver_dir in sorted(mise_python.iterdir(), reverse=True):
            candidates.append(ver_dir / "bin" / "sddc")
    for c in candidates:
        if c.is_file() and _gpu_os.access(c, _gpu_os.X_OK):
            return str(c)
    return None


def _gpu_topology_blocking() -> dict | None:
    import subprocess

    sddc = _resolve_sddc_cli()
    if not sddc:
        return {"error": "sddc CLI not on PATH"}
    try:
        res = subprocess.run(
            [sddc, "gpu", "top", "--json"],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except subprocess.TimeoutExpired:
        return {"error": "sddc gpu top timed out"}
    except Exception as e:
        return {"error": f"sddc gpu top failed: {e}"}
    if res.returncode != 0:
        return {"error": (res.stderr or "non-zero exit").strip()[:500]}
    try:
        return _gpu_json.loads(res.stdout)
    except _gpu_json.JSONDecodeError as e:
        return {"error": f"bad json: {e}"}


def _refresh_gpu_cache_async() -> None:
    global _GPU_CACHE, _GPU_CACHE_TS, _GPU_REFRESH_IN_FLIGHT
    try:
        fresh = _gpu_topology_blocking()
        if fresh is not None:
            _GPU_CACHE = fresh
            _GPU_CACHE_TS = time.monotonic()
    finally:
        with _GPU_REFRESH_LOCK:
            _GPU_REFRESH_IN_FLIGHT = False


@router.get("/api/gpu/topology")
async def get_gpu_topology(request: fastapi.Request):
    """Per-process + per-engine GPU snapshot, sourced from `sddc gpu top --json`.

    Cached 2 s with stale-while-revalidate so the diagnostics panel can
    poll on its existing 5 s tick without paying the nvidia-smi +
    vLLM-scrape cost on the request path.
    """
    blocked = _require_admin_response(request)
    if blocked is not None:
        return blocked

    global _GPU_CACHE, _GPU_CACHE_TS, _GPU_REFRESH_IN_FLIGHT
    now = time.monotonic()
    fresh_enough = _GPU_CACHE is not None and (now - _GPU_CACHE_TS) < _GPU_CACHE_TTL
    if fresh_enough:
        return JSONResponse(_GPU_CACHE)

    if _GPU_CACHE is not None:
        with _GPU_REFRESH_LOCK:
            if not _GPU_REFRESH_IN_FLIGHT:
                _GPU_REFRESH_IN_FLIGHT = True
                _gpu_threading.Thread(
                    target=_refresh_gpu_cache_async,
                    name="gpu-topology-refresh",
                    daemon=True,
                ).start()
        return JSONResponse(_GPU_CACHE)

    fresh = await _gpu_asyncio.get_running_loop().run_in_executor(None, _gpu_topology_blocking)
    if fresh is None:
        return JSONResponse({"error": "topology unavailable"}, status_code=503)
    _GPU_CACHE = fresh
    _GPU_CACHE_TS = now
    return JSONResponse(fresh)
