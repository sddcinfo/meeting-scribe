"""Request-shape helpers used by handlers + the captive-ack tracker.

Pre-v1.0 this module was the auth-scope classifier (hotspot vs LAN,
guest vs admin). v1.0 unified the listener at ``10.42.0.1:443``, made
admin a cookie-only distinction, and routed every guest-vs-admin
decision through ``server_support.admin_guard`` (subnet + cookie).
The leftover here is:

* ``_is_hotspot_client`` — IP-keying for the captive-ack tracker.
  Whether the request happens to come from the AP subnet is a
  presentation concern (used to decide whether to send the OS
  captive-portal Success body), not an auth one.
* ``_has_active_meeting`` — used by the hotspot allowlist
  middleware to decide which paths to widen during recording.
"""

from __future__ import annotations

import os

from meeting_scribe.runtime import state

# IP-prefix shorthand used by ``_is_hotspot_client``. Configurable via
# ``SCRIBE_HOTSPOT_SUBNET`` for in-place testing on a non-default
# bridge IP. Used as a presentation signal only — the admin gate is
# the ``scribe_admin`` cookie, not the source subnet.
_HOTSPOT_PREFIX = os.environ.get("SCRIBE_HOTSPOT_SUBNET", "10.42.0.")


def _is_hotspot_client(request_or_ws) -> bool:
    """True when the client IP is inside the AP DHCP pool. Used as a
    cheap presentation signal — whether to ACK the caller for OS
    captive-portal probes — NOT as an auth gate."""
    client = getattr(request_or_ws, "client", None)
    client_ip = client.host if client else ""
    return client_ip.startswith(_HOTSPOT_PREFIX)


def _has_active_meeting() -> bool:
    return state.current_meeting is not None and state.current_meeting.state.value == "recording"
