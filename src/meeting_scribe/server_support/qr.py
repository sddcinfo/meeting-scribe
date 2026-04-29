"""QR-code SVG generation for the captive-portal landing page.

These helpers exist outside ``server.py`` so the captive-portal route
modules and the hotspot subsystem can both reach them once routes
are extracted. The ``qrcode`` import is module-local so callers that
never invoke a QR helper don't pay the cost of importing PIL.
"""

from __future__ import annotations


def _qr_svg(data: str) -> str:
    """Generate an SVG QR code for the given data string."""
    import io

    import qrcode  # type: ignore[import-untyped]
    import qrcode.image.svg  # type: ignore[import-untyped]

    img = qrcode.make(data, image_factory=qrcode.image.svg.SvgPathImage, box_size=10)
    buf = io.BytesIO()
    img.save(buf)
    return buf.getvalue().decode()


def _wifi_qr_svg(ssid: str, password: str) -> str:
    """Generate an SVG QR code for WiFi auto-join."""
    return _qr_svg(f"WIFI:T:WPA;S:{ssid};P:{password};;")
