"""FastAPI dependency factories for role-gated routes.

Use :func:`require_role` as a ``Depends(...)`` on any handler that
must restrict by role. ``request.state.role`` is set by the
``role_injector`` middleware (see :mod:`meeting_scribe.middlewares`).

Example::

    from meeting_scribe.auth import Role, require_role

    @router.put(
        "/api/admin/settings",
        dependencies=[Depends(require_role(Role.ADMIN))],
    )
    async def update_settings(...): ...

The dependency raises :class:`fastapi.HTTPException` with status 403
when the request's role is not in the allowed set. UI flags are NEVER
the authorization boundary; this guard is.
"""

from __future__ import annotations

from collections.abc import Callable

import fastapi
from fastapi import HTTPException

from meeting_scribe.auth.roles import Role


def require_role(*allowed: Role) -> Callable[[fastapi.Request], None]:
    """Build a FastAPI dependency that 403s unless ``request.state.role``
    is in ``allowed``.

    Empty ``allowed`` is a misconfiguration (would 403 every request);
    we raise ``ValueError`` at import time to catch the bug early.
    """
    if not allowed:
        raise ValueError("require_role() needs at least one allowed Role")
    allowed_set = frozenset(allowed)

    def _dep(request: fastapi.Request) -> None:
        role = getattr(request.state, "role", Role.GUEST)
        if role not in allowed_set:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "forbidden",
                    "required_roles": sorted(r.value for r in allowed_set),
                    "actual_role": role.value if isinstance(role, Role) else str(role),
                },
            )

    return _dep


def require_kiosk_listener(request: fastapi.Request) -> None:
    """Return 404 when the request did NOT arrive on the loopback
    kiosk listener (``127.0.0.1:8444``).

    The kiosk listener is plain HTTP, bound to ``127.0.0.1`` only, and
    serves only ``/kiosk``, ``/kiosk-bootstrap``, and ``/api/kiosk/*``.
    Those routes are deliberately invisible from the canonical HTTPS
    listener: even with a valid ``scribe_kiosk`` cookie, a request
    arriving on ``:443`` gets a 404 here, so the kiosk attack surface
    can never be reached from off-host.

    The ``via_kiosk_listener`` flag is set by the
    ``listener_tagging`` middleware.
    """
    via = getattr(request.state, "via_kiosk_listener", False)
    if not via:
        # 404 not 403 - the route should appear to not exist outside
        # the kiosk listener.
        raise HTTPException(status_code=404, detail={"error": "not_found"})
