"""``/kiosk-bootstrap`` — exchange a single-use nonce for a kiosk cookie.

Loopback-only path: the route is guarded by ``require_kiosk_listener``
so it 404s on the canonical HTTPS listener. The cage chromium session
on the GB10 hits it via ``http://127.0.0.1:8444/kiosk-bootstrap?nonce=X``;
on a valid nonce the server sets the ``scribe_kiosk`` cookie and 302s
to ``/kiosk``.

Nonces are minted via ``POST /api/admin/kiosk/mint-nonce`` (admin-auth
required); the kiosk-runtime calls that endpoint at startup using the
deterministic local admin password helper that the CLI already uses.
A separate ``/api/admin/kiosk/mint-nonce`` route is defined in
``routes/admin.py`` so it inherits the admin write-surface gates.

The cookie shape mirrors the admin cookie (HMAC over
``<issued>.<sid>``); ``CookieSigner`` is reused with a kiosk-specific
``hmac_info`` so a kiosk cookie cannot be replayed as an admin cookie
even though they share the boot-derived subkey.
"""

from __future__ import annotations

import logging

import fastapi
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, RedirectResponse

from meeting_scribe.auth.guards import require_kiosk_listener
from meeting_scribe.kiosk.nonces import consume_nonce
from meeting_scribe.runtime import state
from meeting_scribe.terminal.auth import KIOSK_COOKIE_NAME

logger = logging.getLogger(__name__)

router = APIRouter()


# Kiosk cookie TTL: 90 days. The chromium profile lives in
# ``~/snap/chromium/common/kiosk-profile`` and the cookie persists
# across reboots so the kiosk doesn't have to re-bootstrap every
# session start. Revocation is via factory_reset (rotates the secret)
# OR by removing the chromium profile.
_KIOSK_COOKIE_TTL_SECONDS = 90 * 24 * 3600


@router.get(
    "/kiosk-bootstrap",
    response_model=None,
    dependencies=[Depends(require_kiosk_listener)],
)
async def kiosk_bootstrap(
    request: fastapi.Request, nonce: str = ""
) -> RedirectResponse | JSONResponse:
    """Consume a single-use nonce and mint a ``scribe_kiosk`` cookie.

    The endpoint is GET because chromium ``--app=<url>`` issues a GET
    navigation; the redirect-with-Set-Cookie pattern is OAuth-implicit-
    flow shape. CSRF is irrelevant here because:

    * the listener is bound to ``127.0.0.1`` only,
    * the nonce is single-use with a 60 s TTL,
    * the nonce was minted by an admin-authenticated POST.

    On success: 302 → ``/kiosk`` with ``Set-Cookie: scribe_kiosk=...``.
    On any failure (missing/garbage/expired/consumed nonce): 401.
    """
    signer = getattr(state, "_kiosk_cookie_signer", None)
    if signer is None:
        # Server is initializing or the kiosk signer wasn't wired.
        # Fail closed.
        logger.error("kiosk_bootstrap: kiosk cookie signer not initialized")
        return JSONResponse({"error": "kiosk_unavailable"}, status_code=503)

    if not consume_nonce(nonce):
        # Defence in depth: also re-check that the request truly
        # arrived on 127.0.0.1 in case a future middleware change
        # leaks the route to the canonical listener. require_kiosk_listener
        # already 404s in that case; this is belt-and-suspenders.
        client = getattr(request, "client", None)
        client_host = client.host if client is not None else ""
        if client_host not in {"127.0.0.1", "::1"}:
            logger.warning(
                "kiosk_bootstrap: non-loopback client %s reached endpoint",
                client_host,
            )
        return JSONResponse(
            {"error": "invalid_nonce"},
            status_code=401,
            headers={"Cache-Control": "no-store"},
        )

    cookie_value = signer.issue()
    # Redirect with ``?popout=view`` so the existing client-side
    # check (which activates ``body.popout-view`` whenever the URL
    # carries a ``popout`` query) fires for the kiosk too. Without
    # this query, /kiosk renders the full admin SPA (just with admin
    # chrome hidden by the data-role="kiosk" cascade) instead of the
    # popout layout.
    response = RedirectResponse(url="/kiosk?popout=view", status_code=302)
    response.set_cookie(
        key=KIOSK_COOKIE_NAME,
        value=cookie_value,
        max_age=_KIOSK_COOKIE_TTL_SECONDS,
        httponly=True,
        # The loopback listener is plain HTTP, so ``secure=True`` would
        # make chromium drop the cookie immediately. Loopback is treated
        # as a secure context by every modern browser regardless.
        secure=False,
        samesite="strict",
        path="/",
    )
    logger.info("kiosk_bootstrap: minted kiosk cookie for loopback client")
    return response
