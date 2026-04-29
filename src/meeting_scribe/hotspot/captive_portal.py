"""Captive-portal probe routes + ``/api/meeting/wifi`` info endpoint.

Two-phase captive-portal behavior keyed on client IP (state lives
in ``server_support.captive_ack``):

* **Unacknowledged** — every OS probe returns a 302 redirect (or
  ``captive: true`` for RFC 8910) so the device's captive-portal
  assistant opens automatically on WiFi association and shows the
  guest portal.

* **Acknowledged** — once the client has actually loaded the portal
  page at ``/``, their IP is added to ``_captive_acked``. Subsequent
  probes return the platform-specific "not captive, you're online"
  response so the CNA sheet dismisses and the blue tick appears.

Without the second phase iOS stays stuck in CNA forever — the OS
keeps polling ``/hotspot-detect.html`` and never sees the
``Success`` body it wants, so it never marks the network as
"internet ready". The set is IP-keyed (not cookie-keyed) because
iOS CNA is a separate WebKit context from Safari and does not share
cookies. State is process-local (cleared on meeting-scribe restart)
— fine for a demo AP where the SSID rotates per meeting anyway.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import fastapi
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from meeting_scribe.server_support.captive_ack import (
    _PORTAL_URL,
    _is_captive_acked,
)
from meeting_scribe.server_support.qr import _qr_svg, _wifi_qr_svg

logger = logging.getLogger(__name__)

router = APIRouter()


def _load_hotspot_state() -> dict | None:
    """Read hotspot state written by the AP-bring-up path.

    Pure file read; no nmcli / sudo / dbus.
    """
    state_file = Path("/tmp/meeting-hotspot.json")
    if not state_file.exists():
        return None
    try:
        return json.loads(state_file.read_text())
    except Exception:
        return None


@router.get("/api/meeting/wifi")
async def get_wifi_info(request: fastapi.Request) -> JSONResponse:
    """Return WiFi hotspot info + QR code SVG for client auto-join.

    Pure read from the hotspot state file. No nmcli, no sudo, no dbus.

    The state file is the single source of truth — it's written atomically
    only from ``_start_wifi_ap()`` on meeting-start / rotation and from
    ``_start_captive_portal()`` on hotspot-up, both inside this process.
    Nothing else mutates it. The old version of this handler called
    ``_write_hotspot_state_sync()`` (one `sudo nmcli --show-secrets` + one
    `sudo nmcli con show --active`) on every GET "just in case the state
    file drifted" — observed 2026-04-15, this burned ~1.5 s per request on
    this box because sudo+PAM+dbus is slow, and the guest page polls this
    endpoint every ~10 s. 26 polls × 1.5 s blew past the asyncio executor
    thread pool and showed up as recurring 2.5 s event-loop stalls that
    killed the audio-out WS ping. There was no real drift scenario — just
    defensive code from a bug fear that doesn't actually exist in the
    single-writer world.
    """
    hotspot_state = _load_hotspot_state()
    if not hotspot_state:
        return JSONResponse({"error": "Hotspot not active"}, status_code=503)

    ssid = hotspot_state["ssid"]
    password = hotspot_state["password"]
    ap_ip = hotspot_state["ap_ip"]

    wifi_qr_svg = _wifi_qr_svg(ssid, password)
    # Guest portal is served over plain HTTP on port 80 bound to the hotspot
    # gateway IP. Building the QR URL as http:// (no port) means:
    #   - No self-signed cert warnings when guests scan and join
    #   - The captive portal mini-browser on iOS actually follows the URL
    #   - All guest traffic flows through the HTTP listener, which the
    #     scheme-based guest_scope check already locks down.
    # Admin stays on https://<lan>:8080/; nothing guest-facing references it.
    meeting_url = f"http://{ap_ip}/"
    url_qr_svg = _qr_svg(meeting_url)

    # Silence the unused-variable warning now that we no longer need the
    # caller's URL scheme (the meeting_url is hard-coded to http).
    del request

    return JSONResponse(
        {
            "ssid": ssid,
            "password": password,
            "ap_ip": ap_ip,
            "meeting_url": meeting_url,
            "session_id": hotspot_state.get("session_id", ""),
            "wifi_qr_svg": wifi_qr_svg,
            "url_qr_svg": url_qr_svg,
        }
    )


@router.get("/hotspot-detect.html")
async def captive_apple(request: fastapi.Request) -> fastapi.responses.Response:
    """Apple iOS/macOS probe.

    Unacknowledged → 302 to portal → CNA opens.
    Acknowledged   → exact ``Success`` HTML → CNA dismisses, blue tick.
    """
    if _is_captive_acked(request):
        return fastapi.responses.HTMLResponse(
            "<HTML><HEAD><TITLE>Success</TITLE></HEAD><BODY>Success</BODY></HTML>"
        )
    return fastapi.responses.RedirectResponse(_PORTAL_URL, status_code=302)


@router.get("/generate_204")
@router.get("/gen_204")
@router.get("/canonical.html")
@router.get("/redirect")
async def captive_204(request: fastapi.Request) -> fastapi.responses.Response:
    """Android/ChromeOS/Firefox probes.

    Unacknowledged → 302 to portal → captive-portal sign-in UI.
    Acknowledged   → HTTP 204 No Content → network marked online.
    """
    if _is_captive_acked(request):
        return fastapi.responses.Response(status_code=204)
    return fastapi.responses.RedirectResponse(_PORTAL_URL, status_code=302)


@router.get("/connecttest.txt")
async def captive_windows(request: fastapi.Request) -> fastapi.responses.Response:
    """Windows NCSI probe."""
    if _is_captive_acked(request):
        return fastapi.responses.PlainTextResponse("Microsoft Connect Test")
    return fastapi.responses.RedirectResponse(_PORTAL_URL, status_code=302)


@router.get("/ncsi.txt")
async def captive_ncsi(request: fastapi.Request) -> fastapi.responses.Response:
    """Windows NCSI secondary probe."""
    if _is_captive_acked(request):
        return fastapi.responses.PlainTextResponse("Microsoft NCSI")
    return fastapi.responses.RedirectResponse(_PORTAL_URL, status_code=302)


@router.get("/success.txt")
async def captive_firefox(request: fastapi.Request) -> fastapi.responses.Response:
    """Firefox captive-portal probe."""
    if _is_captive_acked(request):
        return fastapi.responses.PlainTextResponse("success\n")
    return fastapi.responses.RedirectResponse(_PORTAL_URL, status_code=302)


@router.get("/api/captive")
async def captive_rfc8910(request: fastapi.Request) -> JSONResponse:
    """RFC 8910 captive-portal API.

    Modern OSes (iOS 14+, Android 11+) poll this after learning the
    URL from DHCP option 114. Flips from ``captive: true`` to
    ``captive: false`` once the client has loaded the portal page.
    """
    if _is_captive_acked(request):
        return JSONResponse(
            {"captive": False},
            headers={"Content-Type": "application/captive+json"},
        )
    return JSONResponse(
        {
            "captive": True,
            "user-portal-url": _PORTAL_URL,
        },
        headers={"Content-Type": "application/captive+json"},
    )
