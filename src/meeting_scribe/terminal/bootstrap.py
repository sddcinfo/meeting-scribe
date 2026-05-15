"""Admin-authorize flow: post the wizard-minted password → receive a signed cookie.

The wizard mints a memorable admin password (``DellMeetingAdmin<NNNN>``
where ``NNNN`` is the SSID's 4-digit suffix) and persists its HMAC at
``admin-password-hmac``. This module verifies the candidate against
that stored HMAC and issues the ``scribe_admin`` cookie that gates
every admin endpoint.

The pre-wizard ``BOOTSTRAP_HTML`` form + the master-secret-leaking
``/api/admin/terminal-access`` GET endpoint were removed when the
wizard became the single source of admin credentials. ``/auth`` is
now the operator-visible sign-in surface.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from meeting_scribe import setup_state
from meeting_scribe.runtime import state
from meeting_scribe.terminal.auth import (
    COOKIE_NAME,
    AdminSecretStore,
    CookieSigner,
    decode_verified_cookie,
    revoke_session,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


@dataclass
class BootstrapConfig:
    admin_secret: AdminSecretStore
    cookie_signer: CookieSigner


def _close_admin_ws_for_session(session_id: str) -> None:
    """Best-effort: close every admin WS bound to ``session_id``.

    Logout / re-auth call this so an outstanding privileged WS from the
    revoked session can't keep streaming. Schedules close coroutines
    via the running event loop; never raises (the cookie revocation
    must remain idempotent).
    """
    bucket = state._admin_ws_by_session.pop(session_id, None)
    if not bucket:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Not in an async context — best-effort no-op; the WS handler's
        # finally block will clean up when its connection eventually
        # closes naturally.
        return
    for ws in bucket:
        close = getattr(ws, "close", None)
        if close is None:
            continue
        coro = close(code=1008, reason="revoked")
        if asyncio.iscoroutine(coro):
            loop.create_task(coro)


async def _logout(request: Request, cfg: BootstrapConfig) -> JSONResponse:
    """Revoke the cookie's session_id, close active admin WS, delete the
    cookie. Idempotent — works even when the cookie is missing or
    invalid (no error path leaks state to unauthenticated callers).
    """
    cookie_value = request.cookies.get(COOKIE_NAME)
    ok, session_id, issued_at = decode_verified_cookie(cfg.cookie_signer, cookie_value)
    if ok and session_id is not None and issued_at is not None:
        revoke_session(
            session_id,
            expiry_epoch=issued_at + cfg.cookie_signer.max_age_seconds,
            revoked_sessions=state._revoked_sessions,
        )
        _close_admin_ws_for_session(session_id)
    # Captive-gateway: drop the caller's IP from the admin allowlist so
    # they lose WAN forwarding on the next packet. Best-effort: a
    # firewall_allowlist failure must NOT fail the logout response.
    await _captive_remove_admin(request)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(key=COOKIE_NAME, path="/")
    resp.headers["Cache-Control"] = "no-store, private"
    return resp


def _client_ip(request: Request) -> str | None:
    """Return the AP-side client IP for the captive allowlist.

    The HTTPS listener binds directly to ``10.42.0.1`` (no proxy), so
    ``request.client.host`` is the correct source. None when the
    request has no client (test/internal).
    """
    client = getattr(request, "client", None)
    return client.host if client is not None else None


async def _captive_add_admin(request: Request) -> None:
    """Best-effort: add the caller's IP to ``ms-allowed-admins``.

    Wrapped in try/except so a firewall_allowlist failure (binary
    missing, ipset not loaded) doesn't fail the HTTP authorize. The
    GC tick reconciles missed entries later.
    """
    ip = _client_ip(request)
    if not ip:
        return
    try:
        from meeting_scribe.server_support import firewall_allowlist

        await firewall_allowlist.add_admin(ip)
    except Exception:
        logger.exception("captive: add_admin hook failed for %s", ip)


async def _captive_remove_admin(request: Request) -> None:
    """Best-effort opposite of :func:`_captive_add_admin`."""
    ip = _client_ip(request)
    if not ip:
        return
    try:
        from meeting_scribe.server_support import firewall_allowlist

        await firewall_allowlist.remove_admin(ip)
    except Exception:
        logger.exception("captive: remove_admin hook failed for %s", ip)


def _media_type(request: Request) -> str:
    """Return the lowercased media type from ``Content-Type`` with any
    parameters stripped. ``application/json; charset=UTF-8`` and
    ``application/x-www-form-urlencoded; charset=UTF-8`` both
    normalize to the bare media type."""
    raw = request.headers.get("content-type", "")
    return raw.split(";", 1)[0].strip().lower()


async def _read_password_candidate(
    request: Request,
) -> tuple[str | None, JSONResponse | None, bool]:
    """Parse the wizard password out of the request. Returns
    ``(password, error_response, is_form_post)``.

    Form posts (``application/x-www-form-urlencoded``) → 303 redirects
    on completion; XHR posts (``application/json``) → JSON. Unknown
    media types get a 415 the form path can't reach (browsers always
    send a known type for ``<form method=POST>``)."""
    media = _media_type(request)
    if media == "application/x-www-form-urlencoded":
        try:
            form = await request.form()
        except Exception:  # malformed form body
            return None, JSONResponse({"error": "invalid_body"}, status_code=400), True
        candidate = form.get("password", "")
        return str(candidate or ""), None, True
    if media == "application/json":
        try:
            body = await request.json()
        except Exception:
            return None, JSONResponse({"error": "invalid_body"}, status_code=400), False
        password = (body or {}).get("password", "") if isinstance(body, dict) else ""
        return (password if isinstance(password, str) else ""), None, False
    return None, JSONResponse({"error": "unsupported_media_type"}, status_code=415), False


def register_bootstrap_routes(app: FastAPI, cfg: BootstrapConfig) -> None:
    @app.post("/api/admin/authorize")
    async def authorize(request: Request) -> Response:
        # Cookie-only admin model: no AP-subnet gate here. The
        # password-HMAC check below is the credential gate; the
        # ``Origin`` allowlist middleware + the cookie's HttpOnly +
        # SameSite=Strict attributes are the CSRF defense.
        password, parse_err, is_form_post = await _read_password_candidate(request)
        if parse_err is not None:
            return parse_err

        # Verify against the wizard-persisted ``admin-password-hmac``.
        # Returns False pre-finish so a half-completed setup can't
        # mint cookies.
        admin_secret = cfg.admin_secret.secret
        ok = bool(password) and setup_state.verify_admin_password(
            password or "", admin_secret=admin_secret
        )
        if not ok:
            # Jittered delay on failure (carry-over from the bootstrap
            # form). Mitigates online brute force without enabling
            # lockout-based DoS of the legitimate admin user.
            await asyncio.sleep(0.2 + random.random() * 0.4)
            peer = getattr(getattr(request, "client", None), "host", "?")
            logger.info("admin authorize failed (peer=%s)", peer)
            if is_form_post:
                resp = Response(status_code=303, headers={"Location": "/auth?err=1"})
                resp.headers["Cache-Control"] = "no-store, max-age=0"
                return resp
            return JSONResponse({"error": "invalid_password"}, status_code=401)

        # Re-auth on the same browser revokes the prior session_id +
        # closes any privileged WS still alive under it before minting
        # the new cookie.
        prior_cookie = request.cookies.get(COOKIE_NAME)
        if prior_cookie:
            ok_prior, prior_sid, prior_issued = decode_verified_cookie(
                cfg.cookie_signer, prior_cookie
            )
            if ok_prior and prior_sid is not None and prior_issued is not None:
                revoke_session(
                    prior_sid,
                    expiry_epoch=prior_issued + cfg.cookie_signer.max_age_seconds,
                    revoked_sessions=state._revoked_sessions,
                )
                _close_admin_ws_for_session(prior_sid)

        cookie_value = cfg.cookie_signer.issue()
        success_resp: Response
        if is_form_post:
            success_resp = Response(status_code=303, headers={"Location": "/"})
        else:
            success_resp = JSONResponse({"ok": True})
        resp = success_resp
        resp.set_cookie(
            key=COOKIE_NAME,
            value=cookie_value,
            max_age=cfg.cookie_signer.max_age_seconds,
            path="/",
            secure=True,
            httponly=True,
            samesite="strict",
        )
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        # Captive-gateway hook: now that the cookie is set, mirror the
        # caller's IP into the admins ipset so FORWARD AP→WAN starts
        # allowing their packets. Best-effort — see helper docstring.
        await _captive_add_admin(request)
        return resp

    @app.post("/api/admin/deauthorize")
    async def deauthorize(request: Request) -> JSONResponse:
        # Backwards-compatible name kept for older clients; new code calls
        # /api/admin/logout. Both perform the same revoke-session-and-
        # close-WS dance.
        return await _logout(request, cfg)

    @app.post("/api/admin/logout")
    async def logout(request: Request) -> JSONResponse:
        return await _logout(request, cfg)
