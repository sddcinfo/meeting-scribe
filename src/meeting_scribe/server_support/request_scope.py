"""Request-scope classifiers (hotspot vs LAN, guest vs admin, meeting state).

These predicates are read by the ``hotspot_guard`` middleware and by
several route handlers to decide whether a request is allowed at all
and, if so, how restrictive the response should be. Pulled out of
``server.py`` so the upcoming hotspot route modules can import them
without dragging the full server graph in.
"""

from __future__ import annotations

import os

from meeting_scribe.runtime import state

# Clients on the hotspot subnet (default 10.42.0.x) are restricted to
# the guest live-view page only. Override with ``SCRIBE_HOTSPOT_SUBNET``
# at server start.
HOTSPOT_SUBNET = os.environ.get("SCRIBE_HOTSPOT_SUBNET", "10.42.0.")


def _is_hotspot_client(request_or_ws) -> bool:
    """Check if the request comes from the hotspot WiFi subnet."""
    client = getattr(request_or_ws, "client", None)
    client_ip = client.host if client else ""
    return client_ip.startswith(HOTSPOT_SUBNET)


def _is_guest_scope(request_or_ws) -> bool:
    """Return True if this request should be treated as a guest request.

    A request is in guest scope if EITHER:
      - the client IP is on the hotspot subnet (existing check), OR
      - the request arrived over plain HTTP (scheme == 'http' or 'ws').

    The second condition hardens the HTTP-only guest listener on port 80:
    even a LAN user who hits ``http://<gb10-lan-ip>:80/`` by mistake gets
    the guest-restricted view rather than admin. Admin is reachable only
    via the HTTPS listener (https://<gb10-lan-ip>:8080/).
    """
    if _is_hotspot_client(request_or_ws):
        return True
    url = getattr(request_or_ws, "url", None)
    scheme = getattr(url, "scheme", "") if url is not None else ""
    return scheme in ("http", "ws")


def _has_active_meeting() -> bool:
    return state.current_meeting is not None and state.current_meeting.state.value == "recording"
