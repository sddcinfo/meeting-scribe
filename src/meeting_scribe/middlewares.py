"""HTTP middlewares for the meeting-scribe FastAPI app.

Seven middlewares for the unified-hotspot v1.0 trust model. Listed
**outermost-first** — they run top-down on the way in (host canonicalization
runs first), and bottom-up on the way out:

* ``host_canonicalization`` — outermost. Forces the appliance origin to a
  single canonical host (env ``SCRIBE_CANONICAL_HOST``, e.g. ``10.42.0.1``).
  Any HTTPS request whose ``Host`` header doesn't match → 308 to
  ``https://<canonical>{path}{?query}``. Disabled when env var is unset (dev
  / unit tests).
* ``origin_allowlist`` — fail-closed CSRF guard. State-changing methods
  (POST/PUT/PATCH/DELETE) on ``/api/admin/*``, ``/admin/bootstrap``,
  ``/api/meeting/*`` MUST carry an ``Origin`` header in the configured
  allowlist (env ``SCRIBE_CANONICAL_ORIGINS``). Disabled when unset.
* ``hotspot_guard`` — restricts guest-scope requests (no admin cookie) to a
  small allowlist; everything else 403s. Active meetings widen the allowlist
  to include the live view + audio output WS + slide reads.
* ``csp_injector`` — sets ``Content-Security-Policy`` on every
  ``Content-Type: text/html`` response. Default-on; future templates inherit
  the policy. Includes ``require-trusted-types-for 'script'`` for
  defense-in-depth.
* ``cache_headers`` — sets ``Cache-Control: no-store, private; Vary: Cookie``
  on the auth-sensitive paths (exact ``/`` and ``/admin/bootstrap``, prefix
  ``/api/admin/``).
* ``request_timing`` — logs any HTTP request that took > 500 ms end-to-end.
* ``static_cache_headers`` — sets cache headers on ``/static/*``.

Pulled out of ``server.py`` so the FastAPI app construction stays a thin
``include_router`` shell. The hotspot allowlists live here too — they're
only consumed by ``hotspot_guard``.
"""

from __future__ import annotations

import logging
import os
import time
from urllib.parse import urlsplit

import fastapi
from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse

from meeting_scribe.server_support.request_scope import (
    _has_active_meeting,
    _is_guest_scope,
)

logger = logging.getLogger(__name__)


def _split_csv(value: str) -> tuple[str, ...]:
    """Tokenize a comma-separated env var, trimming whitespace and dropping blanks."""
    return tuple(token.strip() for token in value.split(",") if token.strip())


def _canonical_host() -> str | None:
    """Return the configured canonical host, or None to disable canonicalization.

    Read from ``SCRIBE_CANONICAL_HOST``. Production sets this to ``10.42.0.1``
    so non-canonical Host headers receive a 308; tests and dev runs leave it
    unset and the middleware no-ops.
    """
    value = os.environ.get("SCRIBE_CANONICAL_HOST", "").strip()
    return value or None


def _origin_allowlist() -> frozenset[str]:
    """Return the configured Origin allowlist (set of ``scheme://host[:port]``).

    Read from ``SCRIBE_CANONICAL_ORIGINS`` (comma-separated). Empty / unset
    → empty set, which the middleware interprets as "fail open" (no Origin
    enforcement). Production sets this to
    ``"https://10.42.0.1,https://10.42.0.1:443"``.
    """
    raw = os.environ.get("SCRIBE_CANONICAL_ORIGINS", "")
    return frozenset(_split_csv(raw))


def _origin_allowed(origin: str | None) -> bool:
    """Match ``origin`` against the configured allowlist on scheme + host + port.

    The allowlist is exact-match: ``https://10.42.0.1`` and
    ``https://10.42.0.1:443`` are listed separately if both should be
    accepted. ``None`` / empty is rejected by callers that need a state-
    changing-request guarantee; permissive callers can short-circuit it.
    """
    allow = _origin_allowlist()
    if not allow:
        # No allowlist configured → middleware short-circuits "fail open".
        return True
    if not origin:
        return False
    return origin in allow


# Paths where state-changing requests MUST carry an allowed Origin. Read-only
# methods (GET, HEAD, OPTIONS) are exempt because they cannot mutate state.
_ORIGIN_GUARDED_PREFIXES: tuple[str, ...] = (
    "/api/admin/",
    "/admin/bootstrap",
    "/api/meeting/",
)

# Paths receiving the auth-sensitive cache-control headers. Exact-match on
# ``/`` and ``/admin/bootstrap`` (so longer paths like ``/admin/bootstrap.css``
# are NOT matched), plus prefix-match on ``/api/admin/``.
_CACHE_NO_STORE_EXACT: frozenset[str] = frozenset({"/", "/admin/bootstrap"})
_CACHE_NO_STORE_PREFIX: tuple[str, ...] = ("/api/admin/",)

# Default Content-Security-Policy. Single line, concatenated from logical
# directives below for readability. Updated alongside any change in the
# allowed JS/audio/WS surface — see Plan 1 §A.6.4.
_CSP_DIRECTIVES: tuple[str, ...] = (
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self'",
    "img-src 'self' data:",
    # Existing MSE/fMP4 audio output attaches blob: URLs to <audio>.
    "media-src 'self' blob:",
    # AudioWorklet served from /static/js/audio-worklet.js — no blob: needed.
    "worker-src 'self'",
    # Explicit wss URL because Safari/iOS doesn't always resolve 'self' for
    # WebSocket schemes; see https://github.com/w3c/webappsec-csp/issues/7.
    "connect-src 'self' wss://10.42.0.1",
    "object-src 'none'",
    "base-uri 'none'",
    "frame-ancestors 'none'",
    "require-trusted-types-for 'script'",
)
_CSP_HEADER_VALUE = "; ".join(_CSP_DIRECTIVES)


# Paths hotspot clients can access (only during an active meeting,
# except / and captive probes which are always allowed).
#
# Sign-in surface (``/admin/bootstrap`` GET form, ``/api/admin/authorize``
# POST, ``/api/admin/deauthorize``, ``/api/admin/logout``) is included so
# operators on the open AP can always reach the bootstrap form even before
# they have the admin cookie. The auth rate-limiter (Plan §Guest Admission
# Control) still throttles abuse on these paths.
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
        # Sign-in surface — accessible pre-auth so operators without a
        # cookie can complete /admin/bootstrap → /api/admin/authorize.
        "/admin/bootstrap",
        "/api/admin/authorize",
        "/api/admin/deauthorize",
        "/api/admin/logout",
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

    Registration order matters: in Starlette, the LAST middleware registered
    wraps every prior one — it runs FIRST on the way in. We register
    innermost-first so the final outermost layer (host canonicalization)
    runs before any other middleware sees a non-canonical Host.

    Order on the wire (outermost → innermost):
        host_canonicalization → origin_allowlist → hotspot_guard
        → csp_injector → cache_headers → request_timing → static_cache_headers
    """

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

    @app.middleware("http")
    async def request_timing(request: fastapi.Request, call_next):
        """Log any HTTP request that takes > 500 ms end-to-end."""
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
    async def cache_headers(request: fastapi.Request, call_next):
        """Auth-sensitive cache headers on ``/`` (exact), ``/admin/bootstrap``
        (exact), and any path under ``/api/admin/``.

        ``Cache-Control: no-store, private`` + ``Vary: Cookie`` together
        prevent the browser from caching admin responses across logout, and
        prevent any intermediate from sharing them between users. Static
        assets are NOT touched here (their cache is governed by the
        ``static_cache_headers`` middleware).
        """
        response = await call_next(request)
        path = request.url.path
        if path in _CACHE_NO_STORE_EXACT or any(
            path.startswith(prefix) for prefix in _CACHE_NO_STORE_PREFIX
        ):
            response.headers["Cache-Control"] = "no-store, private"
            response.headers["Vary"] = "Cookie"
        return response

    @app.middleware("http")
    async def csp_injector(request: fastapi.Request, call_next):
        """Apply the strict CSP to every ``Content-Type: text/html`` response.

        Default-on so future templates and framework error pages inherit the
        policy automatically. Non-HTML responses (JSON, audio binary, captive
        ``/generate_204``) are left untouched.
        """
        response = await call_next(request)
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type.lower():
            response.headers["Content-Security-Policy"] = _CSP_HEADER_VALUE
        return response

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
            {"error": "Not available on guest WiFi (use https://10.42.0.1/ for admin)"},
            status_code=403,
        )

    @app.middleware("http")
    async def origin_allowlist(request: fastapi.Request, call_next):
        """Fail-closed CSRF guard for state-changing admin/meeting requests.

        Read-only methods (GET/HEAD/OPTIONS) are exempt — they cannot mutate
        state. Every POST/PUT/PATCH/DELETE on a guarded path must carry an
        ``Origin`` header that exactly matches one of the configured
        canonical origins (env ``SCRIBE_CANONICAL_ORIGINS``). Missing or
        disallowed Origin → ``HTTP 403`` with a structured error body.

        When the env var is unset the allowlist is empty and the middleware
        short-circuits ``True`` (dev / unit tests). Production sets the env
        var to the canonical appliance origin so this becomes load-bearing.
        """
        method = request.method.upper()
        path = request.url.path
        if method in ("GET", "HEAD", "OPTIONS"):
            return await call_next(request)
        if not any(path.startswith(prefix) for prefix in _ORIGIN_GUARDED_PREFIXES):
            return await call_next(request)
        # When the allowlist is empty the helper returns True → no-op.
        if not _origin_allowlist():
            return await call_next(request)
        origin = request.headers.get("origin")
        if not _origin_allowed(origin):
            return JSONResponse(
                {"error": "csrf_origin_disallowed", "origin": origin or ""},
                status_code=403,
            )
        return await call_next(request)

    @app.middleware("http")
    async def host_canonicalization(request: fastapi.Request, call_next):
        """Force-308 any HTTPS request to the canonical appliance host.

        Production binds a single TLS listener at ``10.42.0.1:443`` with a
        leaf cert whose only SAN is ``IP:10.42.0.1``. Hosts that don't match
        the cert can't even establish TLS, so the only way a non-canonical
        Host header reaches this middleware is through a misbehaving client
        on the canonical socket. We 308 them to the canonical URL and
        preserve the path + query verbatim.

        When ``SCRIBE_CANONICAL_HOST`` is unset (dev / unit tests) the
        middleware no-ops so existing test clients hitting ``127.0.0.1`` keep
        working.
        """
        canonical = _canonical_host()
        if canonical is None:
            return await call_next(request)
        # Only canonicalize HTTPS — the captive HTTP sub-app on port 80 has
        # its own catch-all 308 to canonical. The TLS app's redirect target
        # is always https.
        scheme = request.url.scheme.lower()
        if scheme not in ("https", "wss"):
            return await call_next(request)
        host_header = (request.headers.get("host") or "").strip()
        # Strip port if present — canonicalization is host-only at this layer.
        host_only = host_header.split(":", 1)[0].lower()
        if host_only == canonical.lower():
            return await call_next(request)
        target = urlsplit(str(request.url))
        query = f"?{target.query}" if target.query else ""
        return RedirectResponse(
            url=f"https://{canonical}{target.path}{query}",
            status_code=308,
        )
