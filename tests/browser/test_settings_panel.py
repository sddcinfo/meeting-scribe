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

from meeting_scribe.runtime import state as runtime_state
from meeting_scribe.terminal.auth import (
    COOKIE_NAME,
    AdminSecretStore,
    CookieSigner,
    TicketStore,
)
from meeting_scribe.terminal.registry import ActiveTerminals
from meeting_scribe.terminal.router import TerminalRouterConfig, register_terminal_routes

pytestmark = [pytest.mark.browser]


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
    "tts_voice_mode": "studio",
    "tts_voice_mode_options": [
        {"code": "studio", "name": "Studio voice"},
        {"code": "participant", "name": "Participant voice"},
    ],
}


@pytest.fixture
def scribe_settings_server(tmp_path, monkeypatch) -> Generator[dict[str, Any]]:
    secret_path = tmp_path / "admin-secret"
    monkeypatch.setenv("SCRIBE_ADMIN_SECRET_FILE", str(secret_path))

    admin_secret = AdminSecretStore.load_or_create(secret_path)
    cookie_signer = CookieSigner(admin_secret.secret)
    ticket_store = TicketStore(admin_secret.secret)
    registry = ActiveTerminals(max_concurrent=4)

    # admin_guard.has_admin_session reads the signer off this global
    # singleton (set by server.py at production boot). Tests skip
    # server.py's boot path, so install it here for the fixture's
    # lifetime.
    prior_signer = getattr(runtime_state, "_terminal_cookie_signer", None)
    runtime_state._terminal_cookie_signer = cookie_signer

    app = FastAPI()
    # /admin/bootstrap was removed in v1.1; mint the cookie directly +
    # inject it via the playwright context (see _authorize_settings).
    register_terminal_routes(
        app,
        TerminalRouterConfig(
            registry=registry,
            cookie_signer=cookie_signer,
            ticket_store=ticket_store,
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
        "cookie_signer": cookie_signer,
    }

    server.should_exit = True
    thread.join(timeout=3.0)
    runtime_state._terminal_cookie_signer = prior_signer


def _authorize_settings(page, srv) -> None:
    """Inject the production-shaped admin cookie before any navigation."""
    cookie_value = srv["cookie_signer"].issue()
    page.context.add_cookies(
        [
            {
                "name": COOKIE_NAME,
                "value": cookie_value,
                "domain": "127.0.0.1",
                "path": "/",
                "secure": False,
                "httpOnly": True,
                "sameSite": "Strict",
            }
        ]
    )


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
    _authorize_settings(page, srv)
    page.goto(srv["base_url"])
    _open_settings(page)

    # Phase S split the old single-pane drawer into 5 tabs. The
    # section headings live behind their respective `.settings-tab-pane`
    # ancestors and only render once the tab is activated. Surface the
    # ones we still rely on from the default (Network) pane.
    assert page.locator("#settings-pane-network h4:has-text('WiFi Hotspot')").count() == 1

    # The WiFi mode dropdown has the stub's selection.
    assert page.locator("#setting-wifi-mode").input_value() == "meeting"


def test_settings_admin_password_show_toggle(page, scribe_settings_server):
    """The Show/Hide pill flips the input type and its own label."""
    srv = scribe_settings_server
    _authorize_settings(page, srv)
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
    _authorize_settings(page, srv)
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


# The in-drawer terminal-access card (#term-access-card) was deleted by
# Phase S — admin auth moved to /auth + wizard-password, and the
# settings panel no longer carries an "Authorize this browser" flow.
# Two former tests (loads_secret_and_toggles, authorize_transitions_state)
# were removed with the DOM they exercised. Equivalent contract coverage
# lives in tests/test_admin_auth_matrix.py + tests/test_terminal_ws.py.


def test_terminal_font_slider_persists(page, scribe_settings_server):
    """The Font size slider updates localStorage and the live display."""
    srv = scribe_settings_server
    _authorize_settings(page, srv)
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
    _authorize_settings(page, srv)
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
