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
  (POST/PUT/PATCH/DELETE) on ``/api/admin/*`` and ``/api/meeting/*``
  MUST carry an ``Origin`` header in the configured allowlist (env
  ``SCRIBE_CANONICAL_ORIGINS``). Disabled when unset.
* ``hotspot_guard`` — restricts guest-scope requests (no admin cookie) to a
  small allowlist; everything else 403s. Active meetings widen the allowlist
  to include the live view + audio output WS + slide reads.
* ``csp_injector`` — sets ``Content-Security-Policy`` on every
  ``Content-Type: text/html`` response. Default-on; future templates inherit
  the policy. ``require-trusted-types-for 'script'`` is intentionally
  NOT set — several admin panels still assign ``element.innerHTML = ...``
  directly without a Trusted Types policy, which would break dynamic UI
  renders. Enabling Trusted Types requires a per-policy migration first.
* ``cache_headers`` — sets ``Cache-Control: no-store, private; Vary: Cookie``
  on the auth-sensitive paths (exact ``/`` and ``/auth``, prefix
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

from meeting_scribe.server_support.request_scope import _has_active_meeting

# Bind port for the loopback-only kiosk HTTP listener. Matched in the
# ``listener_tagging`` middleware to mark ``request.state.via_kiosk_listener``.
# The listener is wired in ``server.main()`` and serves only ``/kiosk``,
# ``/kiosk-bootstrap``, and ``/api/kiosk/*``.
_KIOSK_LISTENER_PORT = int(os.environ.get("SCRIBE_KIOSK_LISTENER_PORT", "8444"))

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
    "/api/meeting/",
)

# Paths receiving the auth-sensitive cache-control headers. Exact-match on
# ``/`` and ``/auth`` plus prefix-match on ``/api/admin/`` so the auth form
# can't be replayed from a back-button cache and admin responses can't be
# cached across logout.
_CACHE_NO_STORE_EXACT: frozenset[str] = frozenset({"/", "/auth"})
_CACHE_NO_STORE_PREFIX: tuple[str, ...] = ("/api/admin/",)

# Default Content-Security-Policy. Single line, concatenated from logical
# directives below for readability. Updated alongside any change in the
# allowed JS/audio/WS surface — see Plan 1 §A.6.4.
#
# Threat model: this is an operator-facing admin tool served from the
# appliance itself (no internet route in production). The XSS concern
# is "can an attacker get arbitrary JS to run as admin" — that's
# defended by ``script-src 'self'`` (no inline scripts, no eval, no
# external script sources). ``style-src`` is the secondary lever
# protecting against CSS-based exfiltration; for this surface
# ``'unsafe-inline'`` is the right tradeoff because the bundled JS
# uses CSSOM (``el.style.display = 'none'``) and inline-style HTML
# attributes throughout. Refactoring all of that to nonces or class
# toggles is a Q3-sized task; in the meantime the strict policy
# produced ~30 CSP violation errors per page load on every viewport,
# drowning the operator console and making real client-side bugs
# invisible.
_CSP_DIRECTIVES: tuple[str, ...] = (
    "default-src 'self'",
    "script-src 'self'",
    # ``'unsafe-inline'`` allows both inline ``style="..."`` attributes
    # and ``el.style.X = Y`` JS assignments. See threat-model note above.
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data:",
    # Existing MSE/fMP4 audio output attaches blob: URLs to <audio>.
    "media-src 'self' blob:",
    # AudioWorklet served from /static/js/audio-worklet.js — no blob: needed.
    "worker-src 'self'",
    # Explicit ``font-src 'self'`` (was implicit via default-src). All fonts
    # used in production are operating-system fonts referenced by name in
    # CSS ``font-family``; no remote font-CDN — the appliance has no
    # internet route. An explicit directive surfaces a clearer console
    # error if something ever tries to load fonts.gstatic.com or similar.
    "font-src 'self'",
    # Explicit wss URL because Safari/iOS doesn't always resolve 'self' for
    # WebSocket schemes; see https://github.com/w3c/webappsec-csp/issues/7.
    "connect-src 'self' wss://10.42.0.1",
    "object-src 'none'",
    "base-uri 'none'",
    "frame-ancestors 'none'",
    # require-trusted-types-for 'script' would break the bundled JS's
    # innerHTML assignments — see module docstring.
)
_CSP_HEADER_VALUE = "; ".join(_CSP_DIRECTIVES)


# Paths hotspot clients can access (only during an active meeting,
# except / and captive probes which are always allowed).
#
# Sign-in surface (``/auth`` GET form, ``/api/admin/authorize``
# POST, ``/api/admin/deauthorize``, ``/api/admin/logout``) is included so
# operators on the open AP can always reach the auth form even before
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
        # cookie can complete /auth → /api/admin/authorize.
        "/auth",
        "/api/admin/authorize",
        "/api/admin/deauthorize",
        "/api/admin/logout",
        # Phase F setup wizard — reachable from any interface. AP
        # clients ARE guest-scope by definition; without this allowlist
        # the hotspot_guard 403's the wizard before the route handler
        # ever sees it. Race between simultaneous claimants is handled
        # by the cookie-bound session in setup_state.
        "/setup",
    )
)

# Prefix allowlist for guest-scope requests. The wizard's XHR endpoints
# (POST /api/setup/begin, GET /api/setup/credentials, etc.) are too
# numerous for the exact-match set; they all share the /api/setup/
# prefix so a single check covers them.
_HOTSPOT_GUEST_ALLOWED_PREFIXES: tuple[str, ...] = ("/api/setup/",)


# Captive-portal probe paths — phones MUST reach these without auth
# so the OS can detect the captive sheet correctly. Independent of
# pre/post-setup state.
_HOTSPOT_CAPTIVE_PROBES = frozenset(
    (
        "/api/captive",
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


# Post-setup, before the operator has signed in, these paths are
# reachable so they CAN sign in. Everything else 302s to /auth.
_HOTSPOT_AUTH_GATE_ALLOWED = frozenset(
    (
        "/auth",
        "/api/admin/authorize",
        "/api/admin/deauthorize",
        "/api/admin/logout",
    )
)
_HOTSPOT_AUTH_GATE_PREFIXES: tuple[str, ...] = ("/api/auth/",)

# Public editorial pages. These are the marketing / documentation
# surface that ships to GitHub Pages too, so the same content is
# already public — auth-gating them on the captive-portal AP is just
# friction. The trailing-slash forms are canonical; the no-slash
# variants 308-redirect to them via the route handlers in routes/views.py.
_EDITORIAL_PUBLIC_PATHS: frozenset[str] = frozenset(
    (
        "/how-it-works",
        "/how-it-works/",
        "/benchmarking",
        "/benchmarking/",
        "/hardware-scaling",
        "/hardware-scaling/",
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


# Paths the setup-mode redirect lets through (everything else GET-redirects to
# ``/setup`` until the appliance completes first-touch). Listed as exact paths
# (``setup_allowlist_exact``) and prefixes (``setup_allowlist_prefix``).
_SETUP_ALLOWLIST_EXACT: frozenset[str] = frozenset(
    (
        "/setup",
        # Read-only health endpoint. Without this, the
        # ``meeting-scribe start`` wait-loop's poller follows the
        # 302 to /setup, fails JSON-decode, and (pre-self-loop-guard)
        # auto-claimed the wizard from the AP gateway IP.
        "/api/status",
    )
)
_SETUP_ALLOWLIST_PREFIX: tuple[str, ...] = (
    "/api/setup/",
    "/static/js/setup-",
    "/static/css/",
    # Captive-portal probes — let the browser/OS perform its own
    # detection without bouncing it through /setup.
    "/hotspot-detect.html",
    "/generate_204",
    "/gen_204",
    "/canonical.html",
    "/connecttest.txt",
    "/ncsi.txt",
    "/success.txt",
    "/redirect",
    "/api/captive",
)


def _is_setup_allowlisted(path: str) -> bool:
    if path in _SETUP_ALLOWLIST_EXACT:
        return True
    return any(path.startswith(prefix) for prefix in _SETUP_ALLOWLIST_PREFIX)


def register_middlewares(app: FastAPI) -> None:
    """Install all HTTP middlewares on ``app``.

    Registration order matters: in Starlette, the LAST middleware registered
    wraps every prior one — it runs FIRST on the way in. We register
    innermost-first so the final outermost layer (host canonicalization)
    runs before any other middleware sees a non-canonical Host.

    Order on the wire (outermost → innermost):
        listener_tagging → role_injector → host_canonicalization
        → setup_mode_redirect → origin_allowlist → hotspot_guard
        → csp_injector → cache_headers → request_timing → static_cache_headers
    """

    @app.middleware("http")
    async def static_cache_headers(request: fastapi.Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            # JS/HTML/CSS: must-revalidate so browsers don't serve stale
            # admin SPA bundles after a server push. ETag stays intact
            # so 304s still short-circuit the body transfer when nothing
            # changed. Media assets (fonts, vendor bundles) keep the
            # long max-age since they're content-hashed or static.
            path = request.url.path
            needs_revalidation = path.endswith((".js", ".mjs", ".html", ".css"))
            if needs_revalidation:
                response.headers["Cache-Control"] = "no-cache, must-revalidate"
            else:
                response.headers["Cache-Control"] = "public, max-age=3600"
        return response

    @app.middleware("http")
    async def request_timing(request: fastapi.Request, call_next):
        """Log any HTTP request that takes > 500 ms end-to-end.

        When the runtime-config knob ``debug_audio_timing`` is set to a
        truthy value, also log every ``/api/meetings/<id>/audio`` request
        regardless of duration — this is the diagnostic path for "playback
        feels slow but the server log is empty" reports. The knob is
        per-request so flipping it via ``meeting-scribe config set`` /
        SIGHUP is enough; no restart, no static-asset cache invalidation.
        """
        from meeting_scribe import runtime_config

        path = request.url.path
        debug_audio = bool(runtime_config.get("debug_audio_timing", False))
        is_audio = debug_audio and path.startswith("/api/meetings/") and path.endswith("/audio")

        t0 = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        if is_audio:
            logger.info(
                "AUDIO %s %s — %.0f ms code=%s ctype=%s len=%s req_range=%s qs=%s (client=%s)",
                request.method,
                path,
                elapsed_ms,
                response.status_code,
                response.headers.get("content-type", "-"),
                response.headers.get("content-length", "-"),
                request.headers.get("range", "-"),
                request.url.query or "-",
                request.client.host if request.client else "?",
            )
        elif elapsed_ms > 500:
            logger.warning(
                "SLOW HTTP %s %s — %.0f ms (client=%s)",
                request.method,
                path,
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
        """Post-setup auth gate for AP clients.

        ``SCRIBE_HOTSPOT_GUARD_BYPASS=1`` short-circuits the gate
        entirely — used by unit tests that exercise OTHER
        middlewares (CSP, cache headers, origin allowlist) and
        don't care about the auth gate. Production never sets it.

        Requests that arrived on the kiosk loopback listener
        (``127.0.0.1:8444``) are also exempt: that listener serves
        only the ``/kiosk`` mirror and the ``/api/kiosk/*`` reads,
        which have their own ``require_kiosk_listener`` + ``require_role``
        gates. Reaching the loopback listener at all requires being
        the GB10-local kiosk-runtime process, so the hotspot guard
        adds nothing of value here.
        """
        if os.environ.get("SCRIBE_HOTSPOT_GUARD_BYPASS") == "1":
            return await call_next(request)
        if getattr(request.state, "via_kiosk_listener", False):
            return await call_next(request)
        # Once ``setup-complete`` is on disk, the AP stays open
        # (OWE) but every guest-scope request needs ONE of:
        #   * ``scribe_admin`` cookie  (admin sign-in)
        #   * ``ms_guest`` cookie      (4-digit PIN sign-in)
        # Otherwise → redirect to ``/auth``.
        #
        # Pre-setup (no setup-complete marker), the wizard surface
        # at /setup + /api/setup/* is reachable.
        #
        # Captive-portal probes + static assets are always exempt —
        # phones rely on the captive probes to detect the captive
        # sheet, and the auth page itself loads its CSS from /static.
        # In v1.0 every client lives on the same HTTPS origin
        # (10.42.0.1:443), so the old ``_is_guest_scope`` short-circuit
        # never applied — the gate runs unconditionally now.

        path = request.url.path

        # Static + captive probes + public editorial pages are always
        # exempt. Editorial pages (how-it-works, benchmarking,
        # hardware-scaling) are the public docs surface — same content
        # is mirrored to GitHub Pages and rendered unauthenticated
        # there, so gating them on the local AP is just friction.
        if path.startswith("/static/"):
            return await call_next(request)
        if path in _HOTSPOT_CAPTIVE_PROBES:
            return await call_next(request)
        if path in _EDITORIAL_PUBLIC_PATHS:
            return await call_next(request)

        # Pre-setup: only the wizard surface is reachable. Read-only
        # JSON status endpoints (``/api/status``, etc.) are allowed
        # through with their normal handlers so ``meeting-scribe
        # start``'s wait-loop can detect a healthy server even before
        # first-touch — without this the poller follows the 302 to
        # /setup, gets HTML, fails JSON-decode, retries every 250 ms,
        # and floods the log AND (pre-fix) auto-claimed the wizard
        # before the operator's AP client could reach it.
        from meeting_scribe import setup_state

        if not setup_state.is_setup_complete():
            if path == "/setup" or path.startswith("/api/setup/"):
                return await call_next(request)
            if path in _HOTSPOT_MEETING_ALLOWED:
                return await call_next(request)
            return RedirectResponse(url="/setup", status_code=302)

        # Post-setup: /auth + /api/auth/* + the admin sign-in
        # surface are reachable without a cookie so the operator
        # can sign in.
        if path in _HOTSPOT_AUTH_GATE_ALLOWED or any(
            path.startswith(prefix) for prefix in _HOTSPOT_AUTH_GATE_PREFIXES
        ):
            return await call_next(request)

        # Otherwise the request needs an admin cookie OR a guest
        # cookie. Both are HMAC-signed; verifiers live in their
        # respective modules.
        from meeting_scribe.routes.guest_auth import (
            GUEST_COOKIE,
            GUEST_COOKIE_TTL,
            _admin_secret_store,
            _sign_guest_cookie,
            verify_guest_cookie,
        )
        from meeting_scribe.runtime import state as _state

        admin_cookie = request.cookies.get("scribe_admin")
        guest_cookie = request.cookies.get("ms_guest")
        admin_ok = getattr(
            _state, "_terminal_cookie_signer", None
        ) is not None and _state._terminal_cookie_signer.verify(admin_cookie)
        guest_ok = verify_guest_cookie(guest_cookie)

        # Phase H captive-gateway bridge: a client who signed in
        # through the port-80 captive sub-app has their IP in the
        # ``ms-allowed-guests`` (or ``ms-allowed-admins``) ipset but
        # NO cookie in the canonical HTTPS browser context (iOS CNA
        # sandbox isolates cookies from Safari). Treat ipset
        # membership as a valid auth signal and mint a cookie inline
        # so subsequent requests on this browser session short-circuit
        # the ipset check.
        minted_guest_cookie: str | None = None
        if not (admin_ok or guest_ok):
            try:
                from meeting_scribe.server_support import firewall_allowlist

                client = getattr(request, "client", None)
                ip = client.host if client is not None else ""
                if ip and firewall_allowlist.is_admin(ip):
                    # No safe path to mint an admin cookie outside the
                    # /api/admin/authorize flow (needs the live admin
                    # secret rotation tracker). Let admin paths fall
                    # through to the explicit /auth redirect — admin's
                    # browser still has the admin cookie from the
                    # canonical-HTTPS sign-in.
                    pass
                elif ip and firewall_allowlist.is_guest(ip):
                    minted_guest_cookie = _sign_guest_cookie(_admin_secret_store().secret)
                    guest_ok = True
            except Exception:
                logger.exception("hotspot_guard: ipset bridge lookup failed")

        if admin_ok or guest_ok:
            response = await call_next(request)
            if minted_guest_cookie is not None:
                response.set_cookie(
                    GUEST_COOKIE,
                    minted_guest_cookie,
                    max_age=GUEST_COOKIE_TTL,
                    httponly=True,
                    secure=True,
                    samesite="strict",
                    path="/",
                )
            return response

        # No valid cookie — bounce GETs to /auth, refuse mutations.
        if request.method.upper() in ("GET", "HEAD"):
            return RedirectResponse(url="/auth", status_code=302)
        return JSONResponse({"error": "auth_required", "auth_url": "/auth"}, status_code=401)

        # (legacy paths below kept for any future fallthrough — but
        # the cookie check above terminates every request.)
        if path in _HOTSPOT_ALWAYS_ALLOWED or any(
            path.startswith(prefix) for prefix in _HOTSPOT_GUEST_ALLOWED_PREFIXES
        ):
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
    async def setup_mode_redirect(request: fastapi.Request, call_next):
        """When the appliance has not completed first-touch setup,
        redirect every GET that isn't on the setup allowlist back to
        ``/setup``.

        Active during setup-pending OR committing state. Once
        ``setup-complete`` exists this middleware is a no-op.

        Setup-route responses pick up no-store cache headers
        (``Cache-Control: no-store, max-age=0``, ``Pragma: no-cache``,
        ``Vary: Cookie``) so the credentials page can't be replayed
        from any cache layer.
        """
        from meeting_scribe import setup_state

        if setup_state.is_setup_complete():
            return await call_next(request)

        path = request.url.path
        method = request.method.upper()
        if method in ("GET", "HEAD") and not _is_setup_allowlisted(path):
            return RedirectResponse(url="/setup", status_code=302)

        response = await call_next(request)
        if path == "/setup" or any(
            path.startswith(prefix) for prefix in ("/setup/", "/api/setup/")
        ):
            response.headers["Cache-Control"] = "no-store, max-age=0"
            response.headers["Pragma"] = "no-cache"
            existing_vary = response.headers.get("Vary")
            response.headers["Vary"] = f"{existing_vary}, Cookie" if existing_vary else "Cookie"
        return response

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

    @app.middleware("http")
    async def role_injector(request: fastapi.Request, call_next):
        """Resolve ``request.state.role`` from cookies.

        Priority: ``scribe_admin`` > ``scribe_kiosk`` > ``ms_guest`` >
        ``Role.GUEST``. Each cookie is HMAC-verified against its
        ``CookieSigner``; an invalid cookie is treated as "not present"
        and falls through. Highest privilege wins so an operator who
        happens to also have a kiosk cookie is still treated as admin.

        Pure read-side: never mints or rotates cookies — that's the
        job of ``/api/admin/authorize`` (admin) and ``/kiosk-bootstrap``
        (kiosk). The middleware just inspects + tags.
        """
        from meeting_scribe.auth.roles import Role
        from meeting_scribe.runtime import state as _state
        from meeting_scribe.terminal.auth import COOKIE_NAME, KIOSK_COOKIE_NAME

        role = Role.GUEST

        admin_signer = getattr(_state, "_terminal_cookie_signer", None)
        kiosk_signer = getattr(_state, "_kiosk_cookie_signer", None)

        admin_cookie = request.cookies.get(COOKIE_NAME)
        if admin_signer is not None and admin_signer.verify(admin_cookie):
            role = Role.ADMIN
        else:
            kiosk_cookie = request.cookies.get(KIOSK_COOKIE_NAME)
            if kiosk_signer is not None and kiosk_signer.verify(kiosk_cookie):
                role = Role.KIOSK
            else:
                # Guest cookie verification lives in routes.guest_auth;
                # importing it here would create a cycle, so we treat
                # a guest cookie's mere presence as a hint and let the
                # legacy code in hotspot_guard do the full HMAC check.
                # The role distinction we care about for new RBAC is
                # ADMIN vs KIOSK vs OTHER; all OTHER collapses to GUEST.
                role = Role.GUEST

        request.state.role = role
        return await call_next(request)

    @app.middleware("http")
    async def listener_tagging(request: fastapi.Request, call_next):
        """Tag ``request.state.via_kiosk_listener`` based on bind port.

        The kiosk loopback listener is bound to ``127.0.0.1:8444``
        (plain HTTP). Any request arriving there gets the flag set so
        ``require_kiosk_listener`` lets through and downstream
        middlewares (``hotspot_guard``) bypass their cookie gate.

        The port is read from ``request.scope["server"]`` which Starlette
        populates with ``(host, port)``. Falls back to ``False`` if the
        scope is missing the field (test client, unusual transport).
        """
        server = request.scope.get("server") if isinstance(request.scope, dict) else None
        port = None
        if server and len(server) >= 2:
            port = server[1]
        request.state.via_kiosk_listener = port == _KIOSK_LISTENER_PORT
        return await call_next(request)
