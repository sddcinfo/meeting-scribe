"""Role enum for the meeting-scribe authorization model.

Three principals, attached to ``request.state.role`` by the
``role_injector`` middleware.
"""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    """Authorization principal.

    Ordering of validation (in ``role_injector`` middleware) is:
    ADMIN cookie > KIOSK cookie > GUEST cookie > no cookie. Highest
    privilege wins so an operator browsing on a kiosk-cookied browser
    profile is still treated as admin.
    """

    ADMIN = "admin"
    KIOSK = "kiosk"
    GUEST = "guest"
