"""Setup wizard endpoints (Phase F, simplified).

Single-page first-touch flow. Operator joins the open OWE Wi-Fi
("Dell Meeting <ID4>"), captive sheet pops, deep-links to the
wizard. Wizard:

  1. ``GET /setup``  — claim, mint, render
     • Mint deterministic admin password + 4-digit guest PIN
       (cookie-bound so two phones don't race).
     • Show both on one page with autofill hooks for the
       admin password.
     • "Done" button.

  2. ``POST /api/setup/finish``  — ack + mark complete
     Persists the LIVE HMACs (admin-password, guest-pin), writes
     setup-complete marker. No AP rotation; no reconnect proof;
     no commit-status polling. AP stays OWE forever.

The fingerprint, bootstrap-secret, printed-card, AP-rotation, and
recovery-code machinery were dropped in the v1.0 simplification —
admin password is deterministic (``DellMeetingAdmin<NNNN>``) so an
operator who forgets it can rederive from the SSID, or run
``meeting-scribe factory-reset --yes`` to start over. Anyone
joining the AP after setup-complete hits ``/auth`` (gate page) and
signs in as guest (4-digit PIN) or admin (password).
"""

from __future__ import annotations

import hmac as _hmac
import logging

import fastapi
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse, Response

from meeting_scribe import setup_state
from meeting_scribe.runtime import state
from meeting_scribe.terminal.auth import AdminSecretStore

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Session cookie ───────────────────────────────────────────────


_SID_COOKIE = "ms_setup_sid"
_SID_TTL = 1800  # matches SETUP_EXPIRY_SECONDS


def _no_store(response: Response) -> Response:
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Vary"] = "Cookie"
    return response


def _set_sid_cookie(response: Response, sid: str) -> None:
    response.set_cookie(
        _SID_COOKIE,
        sid,
        max_age=_SID_TTL,
        httponly=True,
        secure=True,
        samesite="strict",
        path="/",
    )


def _read_sid_cookie(request: fastapi.Request) -> bytes | None:
    raw = request.cookies.get(_SID_COOKIE, "")
    return raw.encode("utf-8") if raw else None


def _verify_session(request: fastapi.Request) -> bool:
    cookie_sid = _read_sid_cookie(request)
    if cookie_sid is None:
        return False
    on_disk = setup_state._read_pending_session_id()
    if on_disk is None:
        return False
    return _hmac.compare_digest(cookie_sid, on_disk)


# ── GET /setup — single landing page ────────────────────────────


@router.get("/setup", response_class=HTMLResponse)
async def get_setup_page(request: fastapi.Request) -> Response:
    """The wizard's only operator-visible page."""
    if request.client is None:
        return JSONResponse({"error": "no_client"}, status_code=403)
    if setup_state.is_setup_complete():
        # Setup is one-shot. Once done, ``/setup`` has no further job;
        # bounce the operator straight to the sign-in page instead of
        # showing a "setup is complete" landing page that requires
        # another click.
        return Response(
            status_code=302,
            headers={"Location": "/auth", "Cache-Control": "no-store"},
        )

    cookie_sid = _read_sid_cookie(request)
    client_ip = request.client.host if request.client else "unknown"
    result = setup_state.claim_setup(client_ip, request_cookie_sid=cookie_sid)
    if result.status == "in_progress":
        return _no_store(HTMLResponse(_SETUP_PAGE_IN_PROGRESS_HTML, status_code=409))

    creds = setup_state.read_credentials()
    body = _SETUP_PAGE_HTML.format(
        ssid=_setup_ssid(),
        admin_password=creds.admin_password,
        guest_pin=creds.guest_pin,
    )
    response = HTMLResponse(body)
    if result.status == "created":
        _set_sid_cookie(response, result.session_id)
    return _no_store(response)


def _setup_ssid() -> str:
    """Look up the AP's per-device SSID for the wizard banner.
    Imported lazily so the wifi module isn't loaded for unit tests
    that don't need it."""
    try:
        from meeting_scribe.wifi import setup_ssid

        return setup_ssid()
    except Exception:
        return "Dell Meeting"


# ── POST /api/setup/finish — ack + mark complete ────────────────


@router.post("/api/setup/finish")
async def post_setup_finish(request: fastapi.Request) -> Response:
    """Single-shot finish: persist HMACs + write setup-complete.

    No AP rotation. The OWE AP stays up under the same per-device
    SSID. After this call the operator is "done" — they can sign
    in to the admin UI or share the guest PIN with attendees.
    """
    if not _verify_session(request):
        return JSONResponse({"error": "no_session"}, status_code=403)
    secret_store = _admin_secret_store()
    try:
        setup_state.ack_credentials_saved(admin_secret=secret_store.secret)
    except setup_state.NoPendingSetup as exc:
        # Detail goes to the server log only — CodeQL flagged returning
        # ``str(exc)`` as info exposure. The UI maps the stable
        # ``no_pending_setup`` code to the "session expired" copy.
        logger.info("ack rejected — no pending setup: %s", exc)
        return JSONResponse({"error": "no_pending_setup"}, status_code=410)
    setup_state.mark_setup_complete()
    response = JSONResponse({"state": "complete"})
    response.delete_cookie(_SID_COOKIE, path="/")
    return _no_store(response)


@router.post("/api/setup/cancel")
async def post_setup_cancel(request: fastapi.Request) -> Response:
    if not _verify_session(request):
        return JSONResponse({"error": "no_session"}, status_code=403)
    setup_state.cancel_setup()
    response = JSONResponse({"ok": True})
    response.delete_cookie(_SID_COOKIE, path="/")
    return _no_store(response)


# ── Helpers ─────────────────────────────────────────────────────


def _admin_secret_store() -> AdminSecretStore:
    store = getattr(state, "admin_secret_store", None)
    if isinstance(store, AdminSecretStore):
        return store
    return AdminSecretStore.load_or_create()


# ── Page templates ──────────────────────────────────────────────


_SETUP_HEAD = (
    "<!doctype html>"
    '<html lang="en">'
    "<head>"
    '<meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
    '<meta name="theme-color" content="#f8f7f4">'
    "<title>{title}</title>"
    '<link rel="stylesheet" href="/static/css/dist/setup.css">'
    "</head>"
    "<body>"
)
_SETUP_FOOT = "</body></html>"


_SETUP_PAGE_HTML = (
    _SETUP_HEAD.replace("{title}", "Meeting Scribe — First-time setup")
    + '<main class="setup">'
    + '<header class="step-indicator">'
    + '<span class="product">Meeting Scribe</span>'
    + '<span class="step">First-time setup</span>'
    + "</header>"
    + '<p class="eyebrow">Welcome</p>'
    + "<h1>Two numbers, one phrase.</h1>"
    + '<p class="lede">'
    + "Your demo device is ready. Below are the two credentials you "
    "need &mdash; designed to be memorable on purpose. Tap to copy "
    "either one, or save them to your password manager."
    + "</p>"
    + '<div class="creds-panel">'
    + '<div class="cred-row pin">'
    + '<div class="label">Guest PIN</div>'
    + '<div class="value copy-target" data-copy="{guest_pin}">{guest_pin}</div>'
    + '<div class="caption">Same number as the Wi-Fi name &mdash; '
    + 'tell guests <em>"join the Wi-Fi and use the same number"</em>.</div>'
    + "</div>"
    + '<div class="cred-row admin">'
    + '<div class="label">Admin password</div>'
    + '<div class="value copy-target mono" data-copy="{admin_password}">{admin_password}</div>'
    + '<div class="caption">Memorable format &mdash; <em>DellMeetingAdmin</em> '
    + "plus the same four digits.</div>"
    + "</div>"
    + "</div>"
    + '<form id="autofill-form" class="visually-hidden" action="#" '
    + 'autocomplete="on" onsubmit="return false;" aria-hidden="true">'
    + '<input name="username" type="text" autocomplete="username" '
    + 'value="admin" readonly>'
    + '<input id="admin-pw-field" type="password" '
    + 'autocomplete="new-password" value="{admin_password}" readonly>'
    + "</form>"
    + '<p class="hint">'
    + "You're connected to <strong>{ssid}</strong>. The Wi-Fi stays "
    "open after setup &mdash; the guest PIN is what gates the live "
    "transcript view."
    + "</p>"
    + '<div class="button-row">'
    + '<button id="cancel" type="button" class="secondary">Cancel</button>'
    + '<button id="finish" type="button" class="primary">'
    + "<span>I've saved them</span>"
    + '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">'
    + '<path d="M2 7l3.5 3.5L12 3.5" fill="none" stroke="currentColor" '
    + 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
    + "</svg>"
    + "</button>"
    + "</div>"
    + '<div id="status" role="status" aria-live="polite"></div>'
    + '<script src="/static/js/setup-wizard.js"></script>'
    + "</main>"
    + _SETUP_FOOT
)


_SETUP_PAGE_IN_PROGRESS_HTML = (
    _SETUP_HEAD.replace("{title}", "Meeting Scribe — Setup in progress")
    + '<main class="setup">'
    + "<h1>Setup is already in progress</h1>"
    + "<p>Another device on this network has started the setup wizard. "
    "Wait for them to finish, or run "
    "<code>meeting-scribe factory-reset --yes</code> on the appliance "
    "to start over.</p>" + "</main>" + _SETUP_FOOT
)


_SETUP_PAGE_COMPLETE_HTML = (
    _SETUP_HEAD.replace("{title}", "Meeting Scribe — Setup complete")
    + '<main class="setup">'
    + "<h1>Setup is already complete on this device.</h1>"
    + '<p>Sign in via <a href="/auth">/auth</a> or run '
    "<code>meeting-scribe factory-reset --yes</code> to start over.</p>" + "</main>" + _SETUP_FOOT
)
