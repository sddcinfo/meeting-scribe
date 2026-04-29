"""Playwright end-to-end for the settings drawer.

Mounts a synthetic FastAPI stack that serves the real ``index.html`` +
``scribe-app.js``, plus the bootstrap/terminal routes and a minimal
``/api/admin/settings`` stub so the drawer can load without pulling in
the full meeting-scribe app. Gives us real rendering of the settings
drawer + terminal-access card, independent of what's running on :8080.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from meeting_scribe.terminal.auth import AdminSecretStore, CookieSigner, TicketStore
from meeting_scribe.terminal.bootstrap import BootstrapConfig, register_bootstrap_routes
from meeting_scribe.terminal.registry import ActiveTerminals
from meeting_scribe.terminal.router import TerminalRouterConfig, register_terminal_routes

pytestmark = pytest.mark.browser


STATIC_DIR = Path(__file__).resolve().parents[2] / "static"

_STUB_SETTINGS = {
    "wifi_mode": "meeting",
    "wifi_mode_options": [
        {"code": "off", "name": "Off"},
        {"code": "meeting", "name": "Meeting"},
        {"code": "admin", "name": "Admin"},
    ],
    "admin_ssid": "Dell Admin",
    "admin_password_set": True,
    "wifi_active": True,
    "wifi_ssid": "Dell Demo 0C31",
    "wifi_security": {"key_mgmt": "SAE"},
    "wifi_regdomain": "JP",
    "wifi_regdomain_current": "JP",
    "wifi_regdomain_options": [
        {"code": "JP", "name": "Japan"},
        {"code": "US", "name": "United States"},
    ],
    "timezone": "Asia/Tokyo",
    "timezone_options": ["Asia/Tokyo", "UTC", "America/Los_Angeles"],
    "dev_mode": False,
    "tts_voice_mode": "studio",
    "tts_voice_mode_options": [
        {"code": "studio", "name": "Studio voice"},
        {"code": "participant", "name": "Participant voice"},
    ],
}


@pytest.fixture
def scribe_settings_server(tmp_path, monkeypatch) -> Generator[dict[str, Any], None, None]:
    secret_path = tmp_path / "admin-secret"
    monkeypatch.setenv("SCRIBE_ADMIN_SECRET_FILE", str(secret_path))

    admin_secret = AdminSecretStore.load_or_create(secret_path)
    cookie_signer = CookieSigner(admin_secret.secret)
    ticket_store = TicketStore(admin_secret.secret)
    registry = ActiveTerminals(max_concurrent=4)

    app = FastAPI()
    register_bootstrap_routes(
        app,
        BootstrapConfig(
            admin_secret=admin_secret,
            cookie_signer=cookie_signer,
            is_guest_scope=lambda _r: False,
        ),
    )
    register_terminal_routes(
        app,
        TerminalRouterConfig(
            registry=registry,
            cookie_signer=cookie_signer,
            ticket_store=ticket_store,
            is_guest_scope=lambda _r: False,
        ),
    )

    # Settings stub — just enough for the drawer to populate.
    @app.get("/api/admin/settings")
    async def get_settings(_r: Request) -> JSONResponse:
        return JSONResponse(_STUB_SETTINGS)

    @app.put("/api/admin/settings")
    async def put_settings(_r: Request) -> JSONResponse:
        return JSONResponse(_STUB_SETTINGS)

    # Other endpoints scribe-app.js probes on boot that aren't critical
    # for this test — return minimally-shaped payloads so fetches don't
    # throw and pollute the console.
    @app.get("/api/status")
    async def status() -> JSONResponse:
        return JSONResponse({"terminal": {"count": 0, "available": 4, "max": 4}})

    @app.get("/api/meetings")
    async def meetings() -> JSONResponse:
        return JSONResponse({"meetings": []})

    @app.get("/api/languages")
    async def langs() -> JSONResponse:
        return JSONResponse({"languages": []})

    # Serve the real static tree.
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    import time

    deadline = time.monotonic() + 5.0
    while (not server.started or not server.servers) and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("synthetic settings server failed to start")
    port = server.servers[0].sockets[0].getsockname()[1]

    yield {
        "base_url": f"http://127.0.0.1:{port}",
        "admin_secret": admin_secret.secret.decode(),
    }

    server.should_exit = True
    thread.join(timeout=3.0)


def _open_settings(page) -> None:
    # Gear icon lives in the app header.
    page.click("#btn-settings")
    page.wait_for_selector(".settings-panel.open", timeout=3000)
    # Wait for the WiFi data to populate (the Mode dropdown starts with
    # "Loading…" — we wait for it to get a real option.)
    page.wait_for_function(
        "() => document.querySelector('#setting-wifi-mode option:not([value=\"\"])') !== null",
        timeout=3000,
    )


def test_settings_drawer_opens_and_populates(page, scribe_settings_server):
    srv = scribe_settings_server
    page.goto(srv["base_url"])
    _open_settings(page)

    # Every section heading is visible.
    for title in ["WiFi Hotspot", "Development", "Voice output", "Display", "Terminal access"]:
        assert page.locator(f".settings-section h4:has-text('{title}')").count() == 1

    # The WiFi mode dropdown has the stub's selection.
    assert page.locator("#setting-wifi-mode").input_value() == "meeting"


def test_settings_admin_password_show_toggle(page, scribe_settings_server):
    """The Show/Hide pill flips the input type and its own label."""
    srv = scribe_settings_server
    page.goto(srv["base_url"])
    _open_settings(page)

    pw = page.locator("#setting-admin-password")
    btn = page.locator("#btn-toggle-admin-pw")

    assert pw.get_attribute("type") == "password"
    assert btn.inner_text().strip().lower() == "show"

    btn.click()
    page.wait_for_function(
        "() => document.getElementById('setting-admin-password').type === 'text'",
        timeout=500,
    )
    assert btn.inner_text().strip().lower() == "hide"

    btn.click()
    page.wait_for_function(
        "() => document.getElementById('setting-admin-password').type === 'password'",
        timeout=500,
    )
    assert btn.inner_text().strip().lower() == "show"


def test_settings_admin_password_row_styled(page, scribe_settings_server):
    """Regression: the Show/Hide button used to rely on inline styles; the
    new .btn-icon rule must produce a pill that's at least 40px wide and
    sits on the same baseline as the input.
    """
    srv = scribe_settings_server
    page.goto(srv["base_url"])
    _open_settings(page)

    row = page.locator(".settings-admin-pw-row")
    assert row.count() == 1
    # Input and button live on the same row — their y offsets should line up
    # within a couple of pixels.
    box_input = page.locator("#setting-admin-password").bounding_box()
    box_btn = page.locator("#btn-toggle-admin-pw").bounding_box()
    assert box_input and box_btn
    assert abs(box_input["y"] - box_btn["y"]) < 4, (
        f"input and button misaligned: {box_input} vs {box_btn}"
    )
    # Button must have real CSS width + height.
    assert box_btn["width"] >= 50
    assert box_btn["height"] >= 28


def test_terminal_access_card_loads_secret_and_toggles(page, scribe_settings_server):
    """The terminal-access card fetches the real admin secret, shows an
    unauthorized dot + card state, and the Show/Copy buttons work.
    """
    srv = scribe_settings_server
    page.goto(srv["base_url"])
    _open_settings(page)

    # Wait for the card to leave the 'loading' state.
    page.wait_for_function(
        "() => document.getElementById('term-access-card')?.dataset.state !== 'loading'",
        timeout=3000,
    )
    state = page.get_attribute("#term-access-card", "data-state")
    assert state == "unauthorized", f"expected 'unauthorized' got {state!r}"

    # Secret input pre-populated and masked.
    val = page.locator("#term-access-secret-input").input_value()
    assert len(val) == 64, f"expected 64-hex secret, got {len(val)}"
    assert page.get_attribute("#term-access-secret-input", "type") == "password"

    # Show / hide.
    page.click("#btn-term-secret-reveal")
    assert page.get_attribute("#term-access-secret-input", "type") == "text"
    page.click("#btn-term-secret-reveal")
    assert page.get_attribute("#term-access-secret-input", "type") == "password"


def test_terminal_access_authorize_transitions_state(page, scribe_settings_server):
    """Clicking 'Authorize this browser' mints the cookie and moves the
    card from 'unauthorized' to 'authorized'.
    """
    srv = scribe_settings_server
    page.goto(srv["base_url"])
    _open_settings(page)
    page.wait_for_function(
        "() => document.getElementById('term-access-card')?.dataset.state === 'unauthorized'",
        timeout=3000,
    )

    page.click("#btn-term-authorize")
    page.wait_for_function(
        "() => document.getElementById('term-access-card')?.dataset.state === 'authorized'",
        timeout=3000,
    )
    # The Deauthorize button un-hides.
    assert page.locator("#btn-term-deauthorize").is_visible()
    # The main Save-status line confirmed success.
    status = page.text_content("#settings-status") or ""
    assert "cookie" in status.lower() or "authorized" in status.lower()

    # Deauthorize round-trips the state.
    page.click("#btn-term-deauthorize")
    page.wait_for_function(
        "() => document.getElementById('term-access-card')?.dataset.state === 'unauthorized'",
        timeout=3000,
    )


def test_terminal_font_slider_persists(page, scribe_settings_server):
    """The Font size slider updates localStorage and the live display."""
    srv = scribe_settings_server
    page.goto(srv["base_url"])
    _open_settings(page)

    slider = page.locator("#setting-term-font-size")
    # Use JS to set value because Playwright's fill() on range inputs is unreliable.
    page.evaluate(
        """(v) => {
            const el = document.getElementById('setting-term-font-size');
            el.value = String(v);
            el.dispatchEvent(new Event('input'));
        }""",
        18,
    )
    assert slider.input_value() == "18"
    assert page.text_content("#term-font-value") == "18px"
    assert page.evaluate("() => localStorage.getItem('terminal_font_size')") == "18"


def test_settings_field_vertical_rhythm(page, scribe_settings_server):
    """Regression: stacked .settings-field blocks used to touch each other
    (no margin between Admin SSID / Admin password / Regulatory domain).
    Verify there's at least 10px of breathing room between them now.
    """
    srv = scribe_settings_server
    page.goto(srv["base_url"])
    _open_settings(page)

    fields = page.locator(".settings-section:first-child .settings-field")
    n = fields.count()
    assert n >= 3, f"expected >= 3 WiFi fields, got {n}"

    prev_bottom = None
    for i in range(n):
        box = fields.nth(i).bounding_box()
        if prev_bottom is not None and box:
            gap = box["y"] - prev_bottom
            assert gap >= 10, f"field {i} too close to previous: only {gap:.1f}px of gap"
        prev_bottom = box["y"] + box["height"] if box else prev_bottom
