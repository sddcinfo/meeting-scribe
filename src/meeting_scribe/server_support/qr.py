"""QR generation stubs.

The product no longer renders QR codes anywhere in the UI: the open
hotspot means scan-to-join is unnecessary, and a 4-digit appliance
PIN ("1618") replaces every prior QR surface. These helpers stay as
empty-string-returning stubs so the two remaining callers don't 500:

* ``hotspot/captive_portal.py`` - emits ``wifi_qr_svg`` and
  ``url_qr_svg`` keys in its JSON response.
* ``validate_customer._phase_meeting_qr`` - smoke test asserts the
  keys exist.
"""

from __future__ import annotations


def _qr_svg(data: str) -> str:
    """Stubbed QR generator. Returns an empty SVG so callers that
    interpolate this into HTML do not break, but no QR is drawn."""
    _ = data
    return "<svg></svg>"


def _wifi_qr_escape(value: str) -> str:
    """Escape WIFI: QR reserved characters. Kept for API compatibility
    with the legacy ``_wifi_qr_svg``; harmless on the stub path."""
    out: list[str] = []
    for ch in value:
        if ch in ("\\", ";", ",", ":", '"'):
            out.append("\\")
        out.append(ch)
    return "".join(out)


def _wifi_qr_svg(ssid: str, password: str, *, auth: str = "WPA2") -> str:
    """Stubbed WiFi QR generator. Returns an empty SVG."""
    _ = ssid, password, auth
    return "<svg></svg>"
