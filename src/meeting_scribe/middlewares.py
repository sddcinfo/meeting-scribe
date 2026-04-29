"""HTTP middlewares for the meeting-scribe FastAPI app.

Three pure middlewares, all installed at module import time:

* ``hotspot_guard`` — restricts guest-scope requests (hotspot subnet
  clients + plain-HTTP requests) to a small allowlist; everything
  else 403s. Active meetings widen the allowlist to include the live
  view + audio output WS + slide reads.
* ``request_timing`` — logs any HTTP request that took > 500 ms
  end-to-end, with the URL and elapsed time. Lets us tie slow-callback
  warnings back to the specific handler.
* ``static_cache_headers`` — sets ``Cache-Control`` on ``/static/*``
  responses so JS/HTML/CSS revalidate every time but media assets get
  long max-age.

Pulled out of ``server.py`` so the FastAPI app construction stays a
thin ``include_router`` shell. The hotspot allowlists live here too —
they're only consumed by ``hotspot_guard``.
"""

from __future__ import annotations

import logging
import time

import fastapi
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from meeting_scribe.server_support.request_scope import (
    _has_active_meeting,
    _is_guest_scope,
)

logger = logging.getLogger(__name__)


# Paths hotspot clients can access (only during an active meeting,
# except / and captive probes which are always allowed).
_HOTSPOT_ALWAYS_ALLOWED = frozenset(
    (
        "/",
        "/reader",
        "/api/status",
        "/api/languages",
        "/api/captive",  # RFC 8910 captive-portal probe
        "/hotspot-detect.html",
        "/generate_204",
        "/gen_204",
        "/canonical.html",
        "/connecttest.txt",
        "/ncsi.txt",
        "/success.txt",
        "/redirect",
    )
)

_HOTSPOT_MEETING_ALLOWED = (
    "/api/status",
    "/api/languages",
    "/api/ws/view",
    "/api/ws/audio-out",
    # Guest-side audio-chain diagnostics. Without this, hotspot clients
    # POSTing their AudioContext / WS / decode state back to us got 403
    # and we had zero visibility into why playback was silent — exactly
    # the hole that hid the client-side bug 2026-04-15.
    "/api/diag/listener",
)


def register_middlewares(app: FastAPI) -> None:
    """Install all HTTP middlewares on ``app``.

    Registration order matters: in Starlette, the LAST middleware
    registered wraps every prior one — i.e. it runs FIRST on the way
    in and LAST on the way out. The original server.py registers
    ``hotspot_guard`` first, ``request_timing`` second,
    ``static_cache_headers`` last; we preserve that exact order here.
    """

    @app.middleware("http")
    async def hotspot_guard(request: fastapi.Request, call_next):
        """Restrict guest-scope requests to the guest live view.

        A request is "guest-scope" when it comes from a hotspot-subnet
        IP OR it arrived over plain HTTP. Both are routed through the
        same guest allowlist so the HTTP-only guest listener on port
        80 and the hotspot-subnet IP check stay in lockstep.
        """
        if not _is_guest_scope(request):
            return await call_next(request)

        path = request.url.path

        # Static files are always allowed for guest-scope requests —
        # guest.html is self-contained but we want /static/*.html and
        # captive-portal probes served for other OS probes.
        if path.startswith("/static/"):
            return await call_next(request)

        # Always allow: guest index page + captive portal probes
        if path in _HOTSPOT_ALWAYS_ALLOWED:
            return await call_next(request)

        # Meeting-gated: only during active recording
        if _has_active_meeting() and any(path.startswith(p) for p in _HOTSPOT_MEETING_ALLOWED):
            return await call_next(request)

        # Slide viewer: guests can GET slide metadata and images during
        # a meeting. Write endpoints (POST upload, PUT advance) are
        # blocked here AND guarded by _require_admin() defense-in-depth.
        if (
            _has_active_meeting()
            and request.method == "GET"
            and "/slides" in path
            and path.startswith("/api/meetings/")
        ):
            return await call_next(request)

        # Block everything else — admin API, past meetings, controls.
        return JSONResponse(
            {"error": "Not available on guest WiFi (use https://<gb10>:8080/ for admin)"},
            status_code=403,
        )

    @app.middleware("http")
    async def request_timing(request: fastapi.Request, call_next):
        """Log any HTTP request that takes > 500 ms end-to-end.

        Added 2026-04-15 to pinpoint which handler was causing periodic
        ~2.5 s event-loop stalls that kept killing the audio-out WS. The
        asyncio slow-callback warning told us "a handler" was slow but
        not which one. This middleware closes that gap — the next slow
        request lands in the log with its exact URL + duration, no more
        guessing.
        """
        t0 = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        if elapsed_ms > 500:
            logger.warning(
                "SLOW HTTP %s %s — %.0f ms (client=%s)",
                request.method,
                request.url.path,
                elapsed_ms,
                request.client.host if request.client else "?",
            )
        return response

    @app.middleware("http")
    async def static_cache_headers(request: fastapi.Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            # JS/HTML/CSS: must-revalidate so browsers don't serve stale
            # scribe-app.js / popout-*.js after a server push. ETag stays
            # intact so 304s still short-circuit the body transfer when
            # nothing changed. Media assets (fonts, vendor bundles) keep
            # the long max-age since they're content-hashed or static.
            path = request.url.path
            needs_revalidation = path.endswith((".js", ".mjs", ".html", ".css"))
            if needs_revalidation:
                response.headers["Cache-Control"] = "no-cache, must-revalidate"
            else:
                response.headers["Cache-Control"] = "public, max-age=3600"
        return response
