"""Captive-portal redirect-chain tests — pure-Python, no iptables.

The bug class this catches: real iOS / Android / Windows captive-portal
probe protocols, where wrong status codes leave the OS captive-portal
notification stuck for 30+ seconds. Historical fixes:

  - Apple's ``/hotspot-detect.html`` must return Success literal HTML
    once the client has acknowledged, otherwise the captive sheet
    never dismisses.
  - Android's ``/generate_204`` must return 204 (not 302 again) once
    acknowledged, or the network stays "limited connectivity".
  - Windows NCSI must respond with the literal ``Microsoft Connect Test``
    body for ``connecttest.txt``.

Real-device verification (real iPhone joining the hotspot, etc.) lives
in ``tests/manual/captive_portal.md`` with a 30-day staleness alarm.
This test layer is the unit / logic-level guard that catches regressions
in the redirect-chain decision function before they hit a device.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from meeting_scribe.hotspot.captive_portal import router as captive_router
from meeting_scribe.server_support.captive_ack import (
    _captive_ack,
    _captive_acked,
    _is_captive_acked,
)


@pytest.fixture(autouse=True)
def _clear_ack_set():
    """Each test starts with an empty captive-ack set."""
    _captive_acked.clear()
    yield
    _captive_acked.clear()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(captive_router)
    # The TestClient's default client IP is 'testclient'; the captive-ack
    # set checks against HOTSPOT_SUBNET ('10.42.0.'), so we override the
    # client IP via headers when needed. For unacked tests, we just go
    # with 'testclient'.
    return TestClient(app, client=("10.42.0.123", 12345))


# ── Unacked client (initial probe — must be redirected to portal) ─────


def test_apple_unacked_redirects_to_portal(client):
    """iOS CNA: first probe to /hotspot-detect.html → 302 to portal."""
    resp = client.get("/hotspot-detect.html", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"].startswith("http://")


def test_android_unacked_redirects_to_portal(client):
    """Android: first probe to /generate_204 → 302 to portal."""
    resp = client.get("/generate_204", follow_redirects=False)
    assert resp.status_code == 302


def test_chromeos_unacked_redirects_to_portal(client):
    """ChromeOS: first probe to /gen_204 → 302 to portal."""
    resp = client.get("/gen_204", follow_redirects=False)
    assert resp.status_code == 302


def test_windows_ncsi_unacked_redirects_to_portal(client):
    """Windows NCSI: first probe to /connecttest.txt → 302 to portal."""
    resp = client.get("/connecttest.txt", follow_redirects=False)
    assert resp.status_code == 302


def test_windows_ncsi2_unacked_redirects_to_portal(client):
    """Windows NCSI: secondary probe /ncsi.txt → 302 to portal."""
    resp = client.get("/ncsi.txt", follow_redirects=False)
    assert resp.status_code == 302


def test_firefox_unacked_redirects_to_portal(client):
    """Firefox: first probe to /success.txt → 302 to portal."""
    resp = client.get("/success.txt", follow_redirects=False)
    assert resp.status_code == 302


def test_rfc8910_unacked_returns_captive_true(client):
    """RFC 8910: first poll to /api/captive → captive: true + portal URL."""
    resp = client.get("/api/captive")
    assert resp.status_code == 200
    body = resp.json()
    assert body["captive"] is True
    assert body["user-portal-url"].startswith("http://")


# ── Acked client (post-portal — must report online) ───────────────────


def _ack_client(client: TestClient) -> None:
    """Mark the test client's IP as having seen the portal."""
    # The TestClient uses 10.42.0.123 (configured above); _captive_ack
    # adds based on request.client.host.
    _captive_acked.add("10.42.0.123")


def test_apple_acked_returns_success_literal(client):
    """iOS CNA: post-ack probe → exact 'Success' literal so the sheet
    dismisses and the blue tick appears.

    The literal MUST contain '<TITLE>Success</TITLE>' AND '<BODY>Success</BODY>' —
    iOS pattern-matches both. Older fixes regressed this once.
    """
    _ack_client(client)
    resp = client.get("/hotspot-detect.html")
    assert resp.status_code == 200
    body = resp.text
    assert "<TITLE>Success</TITLE>" in body, (
        f"iOS CNA expects the literal '<TITLE>Success</TITLE>'; got: {body!r}"
    )
    assert "<BODY>Success</BODY>" in body


def test_android_acked_returns_204(client):
    """Android: post-ack probe to /generate_204 → 204 No Content.

    A non-204 response leaves Android in 'limited connectivity' state.
    """
    _ack_client(client)
    resp = client.get("/generate_204")
    assert resp.status_code == 204


def test_windows_acked_returns_microsoft_connect_test(client):
    """Windows NCSI: post-ack /connecttest.txt → literal body."""
    _ack_client(client)
    resp = client.get("/connecttest.txt")
    assert resp.status_code == 200
    assert resp.text == "Microsoft Connect Test"


def test_windows_ncsi2_acked_returns_microsoft_ncsi(client):
    """Windows NCSI secondary: post-ack /ncsi.txt → literal body."""
    _ack_client(client)
    resp = client.get("/ncsi.txt")
    assert resp.status_code == 200
    assert resp.text == "Microsoft NCSI"


def test_firefox_acked_returns_success_lf(client):
    """Firefox: post-ack /success.txt → 'success\\n' literal."""
    _ack_client(client)
    resp = client.get("/success.txt")
    assert resp.status_code == 200
    assert resp.text == "success\n"


def test_rfc8910_acked_returns_captive_false(client):
    """RFC 8910: post-ack /api/captive → captive: false."""
    _ack_client(client)
    resp = client.get("/api/captive")
    assert resp.status_code == 200
    assert resp.json() == {"captive": False}
    assert resp.headers["content-type"].startswith("application/captive+json")


# ── Acknowledgement bookkeeping ───────────────────────────────────────


def test_ack_only_tracks_hotspot_subnet():
    """``_captive_ack`` only adds IPs in the hotspot subnet; off-subnet
    requests are no-ops (defends against stale-state spillover)."""

    class FakeRequest:
        def __init__(self, host: str) -> None:
            self.client = type("C", (), {"host": host})()

    _captive_acked.clear()
    _captive_ack(FakeRequest("10.42.0.5"))
    assert "10.42.0.5" in _captive_acked
    _captive_ack(FakeRequest("192.168.1.1"))
    assert "192.168.1.1" not in _captive_acked


def test_is_captive_acked_returns_false_for_unknown_ip():
    class FakeRequest:
        def __init__(self, host: str) -> None:
            self.client = type("C", (), {"host": host})()

    _captive_acked.clear()
    assert not _is_captive_acked(FakeRequest("10.42.0.99"))
    _captive_acked.add("10.42.0.99")
    assert _is_captive_acked(FakeRequest("10.42.0.99"))
