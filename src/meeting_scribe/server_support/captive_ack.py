"""Captive-portal acknowledgement set.

Tracks hotspot client IPs that have already loaded the portal page,
so subsequent OS captive-portal probes (Apple
``/hotspot-detect.html``, Android ``/generate_204``, Windows
``/connecttest.txt``, Firefox ``/success.txt``, RFC 8910
``/api/captive``) flip from "redirect to portal" to "you're online" and
the CNA / sign-in notification dismisses.

Without the ack set, iOS keeps polling ``hotspot-detect.html`` forever
because it never sees the Success body, leaving the captive-portal
sheet open even after the user clicked through.

Pulled out of ``server.py`` so ``routes/views.py`` (which calls
``_captive_ack`` from the index route) and the upcoming
``routes/captive_probes.py`` module can share the same set.
"""

from __future__ import annotations

from typing import Any

from meeting_scribe.server_support.request_scope import HOTSPOT_SUBNET

# Portal redirect target. The hotspot AP serves DHCP option 3 = 10.42.0.1
# and DNS option 6 = 10.42.0.1, so this is always reachable from a
# hotspot client. http:// (not https://) is intentional — captive-portal
# probes do not follow TLS during the handshake on most platforms.
_PORTAL_URL = "http://10.42.0.1/"

_captive_acked: set[str] = set()


def _captive_ack(request: Any) -> None:
    """Mark the requesting hotspot client IP as having seen the portal."""
    client = getattr(request, "client", None)
    ip = client.host if client else ""
    if ip and ip.startswith(HOTSPOT_SUBNET):
        _captive_acked.add(ip)


def _is_captive_acked(request: Any) -> bool:
    client = getattr(request, "client", None)
    ip = client.host if client else ""
    return bool(ip) and ip in _captive_acked
