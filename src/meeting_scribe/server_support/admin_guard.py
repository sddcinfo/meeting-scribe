"""Admin-scope authentication helpers.

Admin auth is **cookie-only**. The HTTPS listener binds to all
interfaces on port 443 so admin can reach the appliance from the AP
(``https://10.42.0.1``) OR from the LAN
(``https://<lan-ip>``) — same port, no special variables, no
listener-per-interface dance. The ``scribe_admin`` cookie's HMAC is
the gate; ``Origin`` allowlist + the cookie's HttpOnly+SameSite
attributes provide the CSRF defense-in-depth on top.

The wizard at ``/setup`` is also reachable from any interface — the
v1.0 simplification dropped the AP-subnet origin gate so a freshly
imaged appliance can be claimed from the LAN side as well. Race
between two simultaneous claimants is handled by the cookie-bound
session in setup_state (first writer wins, second client gets the
"already in progress" page).
"""

from __future__ import annotations

from typing import Any

import fastapi
from fastapi import Request, WebSocket
from fastapi.responses import JSONResponse

from meeting_scribe.runtime import state


def has_admin_session(request_or_ws: Any) -> bool:
    """Verify the ``scribe_admin`` cookie's HMAC. Returns False if the
    cookie is missing, malformed, expired, or signed with a stale
    subkey (post-restart, post-factory-reset)."""
    cookie = (
        request_or_ws.cookies.get("scribe_admin") if hasattr(request_or_ws, "cookies") else None
    )
    signer = getattr(state, "_terminal_cookie_signer", None)
    if signer is None:
        return False
    return bool(signer.verify(cookie))


def require_admin(request: Request) -> None:
    """FastAPI dependency: ``scribe_admin`` cookie gate.

    Use as ``dependencies=[Depends(require_admin)]`` on every admin
    HTTP route. Returns 401 ``admin_session_required`` when the
    cookie is missing or invalid. ``Origin`` allowlist (run by the
    middleware) + the cookie's HttpOnly+SameSite attributes are
    the CSRF defense; subnet enforcement was removed in v1.0 to
    let admin reach the appliance from the LAN as well as the AP."""
    if not has_admin_session(request):
        raise fastapi.HTTPException(401, "admin_session_required")


async def require_admin_ws(ws: WebSocket) -> bool:
    """Same gate as ``require_admin`` for WebSockets. Returns True
    when accepted; on rejection, closes the WS with code 4401 and
    returns False so the caller can short-circuit cleanly."""
    if not has_admin_session(ws):
        await ws.close(code=4401, reason="admin_session_required")
        return False
    return True


def _require_admin_response(request: Any) -> JSONResponse | None:
    """Return a 403 ``JSONResponse`` if the caller has no valid admin
    cookie, else ``None``. Caller pattern::

        blocked = _require_admin_response(request)
        if blocked is not None:
            return blocked

    The 403 is preserved (rather than 401) because every existing
    JSON client mapped the old ``forbidden`` code to "not signed in"
    copy. Switching to 401 would invalidate that mapping for no
    benefit — both codes signal the same operator action ("sign in
    again")."""
    if not has_admin_session(request):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    return None


def _require_admin_or_raise(request: Any) -> None:
    """Raise ``HTTPException(403)`` if the caller has no valid admin
    cookie. Same predicate as ``require_admin`` /
    ``_require_admin_response`` — just the raise-shape variant for
    handlers that prefer fail-fast over return-value contracts."""
    if not has_admin_session(request):
        raise fastapi.HTTPException(403, "Admin access required")
