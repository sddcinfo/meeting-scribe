"""Admin-scope authentication helpers.

Two flavours of guard, used by different route groups:

* ``_require_admin_response(request)`` returns a 403
  ``JSONResponse`` (or ``None`` if the caller is admin). Used by
  routes that prefer an explicit ``if blocked is not None: return
  blocked`` check — diagnostics, terminal log tail, etc.

* ``_require_admin_or_raise(request)`` raises
  ``fastapi.HTTPException(403)`` if the caller is guest-scope. Used
  by routes that call it for its side-effect (defense-in-depth) and
  expect any failure to short-circuit via the FastAPI exception
  handler.

Pulled out of ``server.py`` because the previous in-module
definitions collided — there were two ``def _require_admin``
declarations in ``server.py``, and the second silently shadowed the
first. The shadow turned the slides-upload guard into "log a
JSONResponse and proceed", letting guest clients upload slides.
Splitting the two flavours into distinct names eliminates the
shadow and makes each call site explicit about which gate it wants.
"""

from __future__ import annotations

from typing import Any

import fastapi
from fastapi.responses import JSONResponse

from meeting_scribe.runtime import state
from meeting_scribe.server_support.request_scope import _is_guest_scope


def _require_admin_response(request: Any) -> JSONResponse | None:
    """Return a 403 ``JSONResponse`` if the caller is not admin, else ``None``.

    Caller pattern::

        blocked = _require_admin_response(request)
        if blocked is not None:
            return blocked
    """
    if _is_guest_scope(request) or not state._terminal_cookie_signer.verify(
        request.cookies.get("scribe_admin")
    ):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    return None


def _require_admin_or_raise(request: Any) -> None:
    """Raise ``HTTPException(403)`` if the caller is guest-scope.

    Used by handlers that treat admin auth as a fail-fast precondition
    rather than a return-value contract.
    """
    if _is_guest_scope(request):
        raise fastapi.HTTPException(403, "Admin access required")
