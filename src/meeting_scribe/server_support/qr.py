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


def _wifi_qr_escape(value: str) -> str:
    """Escape WIFI: QR code reserved characters per the de-facto spec.

    The format is ``WIFI:T:<auth>;S:<ssid>;P:<password>;H:<hidden>;;``
    where ``;``, ``,``, ``:``, ``\\``, and ``"`` MUST be backslash-
    escaped inside SSID / password. Spaces and most punctuation are
    fine bare. Without escaping, an SSID like ``Dell;Demo`` or a
    password containing ``;`` produces a corrupt QR that phones
    silently fail to join.
    """
    out = []
    for ch in value:
        if ch in ('\\', ';', ',', ':', '"'):
            out.append('\\')
        out.append(ch)
    return ''.join(out)


def _wifi_qr_svg(ssid: str, password: str, *, auth: str = "WPA2") -> str:
    """Generate an SVG QR code for WiFi auto-join.

    Format: ``WIFI:T:<auth>;S:<ssid>;P:<password>;H:false;;``.

    ``auth`` defaults to ``WPA2``; pass ``WPA3`` only when you have
    confirmed the target client base supports it (older Android <10
    rejects unknown auth types). ``WPA2`` is recognized by every iOS
    Camera, Android Camera, and modern browser, AND the access point
    can still negotiate WPA3-SAE on actual association — the QR's
    auth field is a hint, not a binding constraint.

    SSID + password are properly escaped via :func:`_wifi_qr_escape`
    so passwords containing ``;``, ``:``, ``\\``, ``,``, or ``"`` no
    longer corrupt the QR payload.
    """
    s = _wifi_qr_escape(ssid)
    p = _wifi_qr_escape(password)
    return _qr_svg(f"WIFI:T:{auth};S:{s};P:{p};H:false;;")
