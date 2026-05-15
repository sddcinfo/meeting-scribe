"""Post-setup AP-client auth gate (Phase J, simplification).

After ``setup-complete`` is written, the AP stays open (OWE) but
every guest-scope request lands here unless it carries one of:

  * ``scribe_admin`` cookie  — issued via the existing admin
    sign-in flow (``/admin/bootstrap`` → ``POST /api/admin/authorize``).
  * ``ms_guest`` cookie     — issued by ``POST /api/auth/guest-pin``
    after a constant-time match against the persisted guest-pin
    HMAC.

The page shows two simple forms:

  1. Guest — enter the 4-digit PIN the operator displayed.
  2. Admin — enter the admin password.

The wizard's ``GET /setup`` page is an exception (operator
provisions credentials there before any auth exists). Captive
probes (``/hotspot-detect.html``, ``/api/captive``, etc.) are
also exempt so the OS captive sheet keeps detecting correctly.

The middleware in ``middlewares.py`` does the redirect routing;
this module owns the gate page + the ``ms_guest`` cookie.
"""

from __future__ import annotations

import hmac as _hmac
import logging
import secrets
import time

import fastapi
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse, Response

from meeting_scribe import setup_state
from meeting_scribe.runtime import state
from meeting_scribe.terminal.auth import AdminSecretStore, derive_cookie_subkey

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Cookie ───────────────────────────────────────────────────────


GUEST_COOKIE = "ms_guest"
GUEST_COOKIE_TTL = 8 * 3600  # 8 hours — fits a workday


def _admin_secret_store() -> AdminSecretStore:
    store = getattr(state, "admin_secret_store", None)
    if isinstance(store, AdminSecretStore):
        return store
    return AdminSecretStore.load_or_create()


def _guest_subkey(secret: bytes) -> bytes:
    """HKDF-derive the guest cookie subkey. ``boot_session_id=None``
    so a regular service restart leaves the signer intact (guest
    sessions persist). ``auth_version`` mixes in so a factory_reset
    bump invalidates every guest cookie alongside the admin ones."""
    return derive_cookie_subkey(
        secret,
        boot_session_id=None,
        auth_version=setup_state.auth_version(),
        info=b"scribe-guest-cookie-v1",
    )


def _sign_guest_cookie(secret: bytes) -> str:
    """``<issued_unix>.<nonce>.<hex-hmac>``. Same shape as the admin
    cookie. Signed with ``_guest_subkey``: persists across restart,
    rotates on factory_reset (auth_version bump)."""
    issued = str(int(time.time()))
    nonce = secrets.token_hex(8)
    payload = f"{issued}|{nonce}".encode()
    subkey = _guest_subkey(secret)
    sig = _hmac.new(subkey, payload, "sha256").hexdigest()
    return f"{issued}.{nonce}.{sig}"


def verify_guest_cookie(value: str | None, *, max_age: int = GUEST_COOKIE_TTL) -> bool:
    """Validate a ``ms_guest`` cookie's HMAC + age.

    Used by ``middlewares.hotspot_guard`` to gate guest-scope
    requests. Constant-time compare; rejects expired or
    malformed cookies silently.
    """
    if not value:
        return False
    parts = value.split(".")
    if len(parts) != 3:
        return False
    issued_s, nonce, sig = parts
    if not issued_s or not nonce or not sig:
        return False
    try:
        issued = int(issued_s)
    except ValueError:
        return False
    now = int(time.time())
    if now - issued > max_age:
        return False
    if issued - now > 5:  # clock-skew tolerance, no future cookies
        return False
    secret = _admin_secret_store().secret
    subkey = _guest_subkey(secret)
    expected = _hmac.new(subkey, f"{issued_s}|{nonce}".encode(), "sha256").hexdigest()
    return _hmac.compare_digest(sig, expected)


# ── /auth — gate page ───────────────────────────────────────────


@router.get("/auth", response_class=HTMLResponse)
async def get_auth_page(request: fastapi.Request) -> Response:
    """The post-setup landing page. Two forms: guest PIN + admin
    password. Submitting either sets the appropriate cookie and
    redirects to ``/`` (the live UI) or ``/admin`` (admin UI).

    ``?err=1`` (set by ``/api/admin/authorize`` form-redirect on a
    bad password) surfaces an inline error above the admin form so
    the operator sees what happened without losing the sign-in
    surface."""
    if not setup_state.is_setup_complete():
        # Pre-setup: redirect to the wizard.
        return Response(
            status_code=302,
            headers={"Location": "/setup", "Cache-Control": "no-store"},
        )
    show_admin_err = request.query_params.get("err") == "1"
    body = _AUTH_PAGE_HTML.replace(
        "<!--ADMIN_ERR_SLOT-->",
        '<p class="error" role="alert">Incorrect password — try again.</p>'
        if show_admin_err
        else "",
    )
    response = HTMLResponse(body)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@router.post("/api/auth/guest-pin")
async def post_guest_pin(request: fastapi.Request) -> Response:
    """Verify the 4-digit PIN, set ``ms_guest`` cookie on match."""
    body = await request.form()
    pin_value = body.get("pin") or ""
    candidate = (pin_value if isinstance(pin_value, str) else "").strip()
    secret = _admin_secret_store().secret
    if not setup_state.verify_guest_pin(candidate, admin_secret=secret):
        return JSONResponse({"error": "pin_mismatch"}, status_code=401)
    cookie = _sign_guest_cookie(secret)
    response = Response(status_code=303, headers={"Location": "/"})
    response.set_cookie(
        GUEST_COOKIE,
        cookie,
        max_age=GUEST_COOKIE_TTL,
        httponly=True,
        secure=True,
        samesite="strict",
        path="/",
    )
    response.headers["Cache-Control"] = "no-store, max-age=0"
    # Captive-gateway: mirror the caller's IP into the guest ipset so
    # OS captive sheets dismiss for this client. Guests still don't
    # get WAN — only admins do.
    await _captive_add_guest(request)
    return response


@router.post("/api/auth/guest-logout")
async def post_guest_logout(request: fastapi.Request) -> Response:
    response = JSONResponse({"ok": True})
    response.delete_cookie(GUEST_COOKIE, path="/")
    await _captive_remove_guest(request)
    return response


def _client_ip(request: fastapi.Request) -> str | None:
    client = getattr(request, "client", None)
    return client.host if client is not None else None


async def _captive_add_guest(request: fastapi.Request) -> None:
    """Best-effort: add the caller's IP to ``ms-allowed-guests``.

    Wrapped in try/except so a firewall_allowlist failure (binary
    missing, ipset not loaded) doesn't fail the PIN verify. The GC
    tick reconciles missed entries on the next read of the lease file.
    """
    ip = _client_ip(request)
    if not ip:
        return
    try:
        from meeting_scribe.server_support import firewall_allowlist

        await firewall_allowlist.add_guest(ip)
    except Exception:
        logger.exception("captive: add_guest hook failed for %s", ip)


async def _captive_remove_guest(request: fastapi.Request) -> None:
    ip = _client_ip(request)
    if not ip:
        return
    try:
        from meeting_scribe.server_support import firewall_allowlist

        await firewall_allowlist.remove_guest(ip)
    except Exception:
        logger.exception("captive: remove_guest hook failed for %s", ip)


# ── Page template ───────────────────────────────────────────────


_AUTH_PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="theme-color" content="#f8f7f4">
  <title>Meeting Scribe — Sign in</title>
  <link rel="stylesheet" href="/static/css/dist/setup.css">
</head>
<body>
<main class="setup">
  <header class="step-indicator">
    <span class="product">Meeting Scribe</span>
    <span class="step">Sign in</span>
  </header>
  <h1>Sign in to continue</h1>

  <h2>Guest</h2>
  <p>Enter the 4-digit PIN the meeting host shared with you.</p>
  <form method="POST" action="/api/auth/guest-pin">
    <label>Guest PIN
      <input
        name="pin"
        type="text"
        inputmode="numeric"
        pattern="[0-9]{4}"
        maxlength="4"
        minlength="4"
        autocomplete="off"
        required
        placeholder="0000">
    </label>
    <div class="button-row">
      <button type="submit" class="primary">Join as guest</button>
    </div>
  </form>

  <h2>Admin</h2>
  <p>Sign in with the admin password to manage the appliance.</p>
  <!--ADMIN_ERR_SLOT-->
  <form method="POST" action="/api/admin/authorize">
    <label>Admin password
      <input
        name="password"
        type="password"
        autocomplete="current-password"
        required>
    </label>
    <div class="button-row">
      <button type="submit" class="secondary">Sign in as admin</button>
    </div>
  </form>
</main>
</body>
</html>
"""
