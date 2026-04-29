"""Resize + font-zoom behavior for the embedded terminal.

Verifies that:
  · Dragging the resize handle (or otherwise changing the body height)
    propagates rows to the server.
  · Changing font-size changes the underlying character metrics and
    propagates new cols/rows to the server.
  · The xterm grid stays geometrically consistent with the container
    (cols × cellW ≈ mount width, within padding slack).

Runs against the minimal synthetic harness in ``test_terminal_panel.py``
so it doesn't touch the user's live -L scribe socket.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse

from meeting_scribe.terminal.auth import AdminSecretStore, CookieSigner, TicketStore
from meeting_scribe.terminal.bootstrap import BootstrapConfig, register_bootstrap_routes
from meeting_scribe.terminal.registry import ActiveTerminals
from meeting_scribe.terminal.router import TerminalRouterConfig, register_terminal_routes

pytestmark = pytest.mark.browser


STATIC_DIR = Path(__file__).resolve().parents[2] / "static"

_HOST = """<!doctype html>
<html><head>
<meta charset="utf-8"><title>resize harness</title>
<link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
<main id="transcript" style="height:80px;background:#f2f1ee;"></main>
<script>
  window.API = window.location.origin;
  window._popoutMinimal = true;
</script>
</body></html>
"""


@pytest.fixture
def scribe_resize_server(tmp_path, monkeypatch) -> Generator[dict[str, Any], None, None]:
    secret_path = tmp_path / "admin-secret"
    monkeypatch.setenv("SCRIBE_ADMIN_SECRET_FILE", str(secret_path))
    monkeypatch.setenv("SCRIBE_TERM_SHELL", "/bin/sh")

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

    @app.get("/")
    async def index():
        return HTMLResponse(_HOST)

    @app.get("/static/js/terminal-panel.js")
    async def panel_js():
        return FileResponse(
            STATIC_DIR / "js" / "terminal-panel.js", media_type="application/javascript"
        )

    @app.get("/static/css/style.css")
    async def style_css():
        return FileResponse(STATIC_DIR / "css" / "style.css", media_type="text/css")

    @app.get("/static/vendor/xterm/{fname}")
    async def vendor(fname: str):
        path = STATIC_DIR / "vendor" / "xterm" / fname
        if not path.exists():
            return HTMLResponse(status_code=404, content="")
        media = "text/css" if fname.endswith(".css") else "application/javascript"
        return FileResponse(path, media_type=media)

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    import time

    deadline = time.monotonic() + 5.0
    while (not server.started or not server.servers) and time.monotonic() < deadline:
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield {
        "base_url": f"http://127.0.0.1:{port}",
        "admin_secret": admin_secret.secret.decode(),
        "registry": registry,
    }
    server.should_exit = True
    thread.join(timeout=3.0)


def _authorize_and_mount(page, srv) -> None:
    page.goto(f"{srv['base_url']}/admin/bootstrap")
    page.fill("#secret", srv["admin_secret"])
    page.click("#btn")
    page.wait_for_url("**/*popout=view*", timeout=3000)
    page.add_script_tag(url="/static/js/terminal-panel.js")
    page.evaluate(
        """async () => {
            const p = new window.TerminalPanel({
                apiBase: window.location.origin,
                wsBase: window.location.origin.replace(/^http/, 'ws'),
            });
            window._panel = p;
            await p.mount();
            p.show();
        }"""
    )
    page.wait_for_selector(".terminal-panel .xterm-screen", timeout=5000)
    page.wait_for_function(
        "() => document.querySelector('.term-status')?.dataset.state === 'live'",
        timeout=10000,
    )


def _measure(page) -> dict:
    return page.evaluate(
        """() => {
            const p = window._panel;
            const t = p && p._term;
            if (!t) return null;
            const mount = document.querySelector('.term-mount');
            const body  = document.querySelector('.term-body');
            const rs = t._core?._renderService;
            const cell = rs?.dimensions?.css?.cell || {};
            return {
                cols: t.cols, rows: t.rows,
                fontSize: t.options.fontSize,
                cellW: cell.width || null, cellH: cell.height || null,
                mountW: mount?.getBoundingClientRect().width,
                mountH: mount?.getBoundingClientRect().height,
                bodyH: body?.getBoundingClientRect().height,
                dimsLabel: document.querySelector('.term-dims')?.textContent,
            };
        }"""
    )


def test_resize_handle_drag_shrinks_rows(page, scribe_resize_server):
    """Shrinking the body height must reduce rows and the status-bar label."""
    _authorize_and_mount(page, scribe_resize_server)
    before = _measure(page)
    assert before["rows"] >= 5
    # Simulate the resize-handle drop by setting body height directly —
    # the ResizeObserver watches this and triggers _scheduleRefit.
    page.evaluate("document.querySelector('.term-body').style.height = '180px'")
    page.wait_for_timeout(400)
    after = _measure(page)
    assert after["rows"] < before["rows"], (
        f"rows should drop on shrink: {before['rows']} → {after['rows']}"
    )
    # Status-bar label reflects the new rows.
    assert after["dimsLabel"] == f"{after['cols']}×{after['rows']}"


def test_font_zoom_propagates_to_new_grid(page, scribe_resize_server):
    """Font zoom must change both cell metrics AND cols/rows."""
    _authorize_and_mount(page, scribe_resize_server)
    before = _measure(page)
    page.evaluate("() => window._panel.setFontSize(22)")
    page.wait_for_timeout(500)
    after = _measure(page)

    # Font size got through to xterm.
    assert after["fontSize"] == 22
    # Bigger font → fatter cells.
    assert after["cellH"] > before["cellH"], (
        f"cellH should grow with font: {before['cellH']} → {after['cellH']}"
    )
    # Bigger cells in the same container → fewer cols + rows.
    assert after["cols"] < before["cols"]
    assert after["rows"] <= before["rows"]
    # Status-bar label is in sync.
    assert after["dimsLabel"] == f"{after['cols']}×{after['rows']}"


def test_fit_stays_consistent_across_font_and_resize(page, scribe_resize_server):
    """After every font change and body resize, cols×cellW is within
    one cell of the mount width (minus its CSS padding).
    """
    _authorize_and_mount(page, scribe_resize_server)

    stages = []
    stages.append(_measure(page))

    for size in (11, 18, 13):
        page.evaluate(f"() => window._panel.setFontSize({size})")
        page.wait_for_timeout(350)
        stages.append(_measure(page))

    page.evaluate("document.querySelector('.term-body').style.height = '220px'")
    page.wait_for_timeout(400)
    stages.append(_measure(page))

    for s in stages:
        assert s["cellW"] and s["mountW"]
        expected = s["cols"] * s["cellW"]
        # Container has 12px left+right padding → ~24px of slack.
        assert abs(expected - s["mountW"]) <= s["cellW"] + 28, (
            f"cols*cellW={expected:.0f} vs mountW={s['mountW']:.0f} (cell={s['cellW']})"
        )


def test_x_button_kills_session_and_closes_panel(page, scribe_resize_server):
    """Clicking the X on the status bar fires the reset endpoint and
    hides the panel. Re-showing creates a fresh session.
    """
    srv = scribe_resize_server
    _authorize_and_mount(page, srv)
    # Sanity: one live session registered.
    assert len(srv["registry"].items) == 1

    # Wire a fetch spy so we can verify the reset endpoint is hit.
    page.evaluate(
        """() => {
            const orig = window.fetch;
            window._resetCalls = [];
            window.fetch = function(u, o) {
                if (typeof u === 'string' && u.includes('/reset')) {
                    window._resetCalls.push({u, method: (o||{}).method});
                }
                return orig.apply(this, arguments);
            };
        }"""
    )

    page.click(".term-close")
    page.wait_for_timeout(500)
    # Panel is now hidden and WS closed → registry empty.
    assert page.evaluate("() => document.querySelector('.terminal-panel').hidden")
    assert len(srv["registry"].items) == 0
    # The reset endpoint was called once with POST.
    calls = page.evaluate("() => window._resetCalls")
    assert calls and calls[0]["u"].endswith("/scribe/reset")
    assert calls[0]["method"] == "POST"


def test_alt_x_only_hides_and_keeps_session(page, scribe_resize_server):
    """Alt+X soft-hides without killing — the WS drops, registry clears,
    but tmux session / shell state persists at the server level.
    """
    srv = scribe_resize_server
    _authorize_and_mount(page, srv)
    page.evaluate(
        """() => {
            const orig = window.fetch;
            window._resetCalls = [];
            window.fetch = function(u, o) {
                if (typeof u === 'string' && u.includes('/reset')) {
                    window._resetCalls.push(u);
                }
                return orig.apply(this, arguments);
            };
        }"""
    )
    # Dispatch a click with altKey=true.
    page.evaluate(
        """() => {
            const btn = document.querySelector('.term-close');
            btn.dispatchEvent(new MouseEvent('click', { altKey: true, bubbles: true }));
        }"""
    )
    page.wait_for_timeout(400)
    assert page.evaluate("() => document.querySelector('.terminal-panel').hidden")
    # No reset endpoint call — we took the hide-only branch.
    assert page.evaluate("() => window._resetCalls") == []
