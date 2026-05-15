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

from meeting_scribe.server_support.request_scope import _is_hotspot_client

# Portal redirect target. The hotspot AP serves DHCP option 3 = 10.42.0.1
# and DNS option 6 = 10.42.0.1, so this is always reachable from a
# hotspot client. http:// (not https://) is intentional — captive-portal
# probes do not follow TLS during the handshake on most platforms.
_PORTAL_URL = "http://10.42.0.1/"

_captive_acked: set[str] = set()


def _captive_ack(request: Any) -> None:
    """Mark the requesting hotspot client IP as having seen the portal."""
    if not _is_hotspot_client(request):
        return
    client = getattr(request, "client", None)
    ip = client.host if client else ""
    if ip:
        _captive_acked.add(ip)


def _is_captive_acked(request: Any) -> bool:
    """Return True when the caller's IP has either seen the portal
    page (legacy ack-set membership) OR is in either captive ipset.

    Phase H adds the ipset check so an admin/guest who authenticated
    via the captive flow doesn't keep getting nagged by the OS captive
    sheet — the next probe sees Success and the CNA dismisses. Lookup
    is sync subprocess (``ipset list``), which is fine for probe
    handlers (a few hits per device per join, not per packet).
    """
    client = getattr(request, "client", None)
    ip = client.host if client else ""
    if not ip:
        return False
    if ip in _captive_acked:
        return True
    # Captive-gateway: authorized clients (admin or guest) count as
    # acked. Import lazily so this module stays import-light for tests
    # that don't touch the captive path.
    try:
        from meeting_scribe.server_support import firewall_allowlist

        if firewall_allowlist.is_admin(ip) or firewall_allowlist.is_guest(ip):
            return True
    except Exception:
        pass  # firewall module unavailable in this test/unit context — treat as not-acked
    return False
