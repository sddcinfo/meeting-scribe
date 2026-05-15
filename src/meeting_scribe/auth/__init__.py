"""Auth package: role model + dependency guards for the FastAPI app.

The principals in meeting-scribe are:
  * Role.ADMIN  - laptop operator with the ``scribe_admin`` cookie.
  * Role.KIOSK  - GB10-local cage chromium session that mints its
                  ``scribe_kiosk`` cookie via the loopback-only
                  ``/kiosk-bootstrap`` endpoint. Read-only role.
  * Role.GUEST  - anyone else (no cookie or only an ``ms_guest`` cookie).

The role is attached to ``request.state.role`` by the
``role_injector`` middleware in :mod:`meeting_scribe.middlewares` so
every dependency / handler can read it without re-validating cookies.

Use :func:`meeting_scribe.auth.guards.require_role` as a FastAPI
``Depends(...)`` on every route that must restrict by role. UI flags
are NEVER the authorization boundary; the server-side guard is.
"""

from meeting_scribe.auth.guards import require_role
from meeting_scribe.auth.roles import Role

__all__ = ["Role", "require_role"]
