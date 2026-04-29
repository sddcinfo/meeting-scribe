"""Playwright end-to-end for the in-browser terminal panel.

Runs against a minimal synthetic FastAPI server built in-fixture so
we don't depend on the full meeting-scribe stack. Marked ``browser``
so the default test run skips it — exercise via ``sddc test -m browser``.
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

_TERMINAL_HOST_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>scribe terminal harness</title></head>
<body>
<main id="transcript" style="height:80px;background:#f2f1ee;"></main>
<script>
  window.API = window.location.origin;
  window._popoutMinimal = true;
</script>
</body></html>
"""


@pytest.fixture
def scribe_terminal_server(tmp_path, monkeypatch) -> Generator[dict[str, Any]]:
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
    async def index() -> HTMLResponse:
        return HTMLResponse(_TERMINAL_HOST_PAGE)

    @app.get("/static/js/terminal-panel.js")
    async def panel_js():
        return FileResponse(
            STATIC_DIR / "js" / "terminal-panel.js", media_type="application/javascript"
        )

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
    # Wait for startup.
    import time

    deadline = time.monotonic() + 5.0
    while (not server.started or not server.servers) and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("synthetic scribe server failed to start")
    port = server.servers[0].sockets[0].getsockname()[1]

    yield {
        "base_url": f"http://127.0.0.1:{port}",
        "ws_url": f"ws://127.0.0.1:{port}",
        "admin_secret": admin_secret.secret.decode(),
        "registry": registry,
    }

    server.should_exit = True
    thread.join(timeout=3.0)


def _authorize(page, base_url: str, secret: str) -> None:
    page.goto(f"{base_url}/admin/bootstrap")
    page.wait_for_selector("#secret")
    page.fill("#secret", secret)
    page.click("#btn")
    # The bootstrap page redirects to /?popout=view on success; we tolerate
    # either the success toast OR the redirect.
    page.wait_for_url(
        lambda url: url.endswith("/?popout=view") or "/admin/bootstrap" in url, timeout=3000
    )


def test_bootstrap_authorize_sets_cookie(page, scribe_terminal_server):
    srv = scribe_terminal_server
    _authorize(page, srv["base_url"], srv["admin_secret"])
    # Cookie should now be visible to page context.
    cookies = page.context.cookies()
    assert any(c["name"] == "scribe_admin" for c in cookies)


def test_bootstrap_wrong_secret_shows_error(page, scribe_terminal_server):
    srv = scribe_terminal_server
    page.goto(f"{srv['base_url']}/admin/bootstrap")
    page.wait_for_selector("#secret")
    page.fill("#secret", "this-is-wrong")
    page.click("#btn")
    page.wait_for_selector(".status.err", timeout=3000)
    text = page.text_content(".status")
    assert text is not None and "invalid" in text.lower()


def test_terminal_panel_echoes_input(page, scribe_terminal_server):
    srv = scribe_terminal_server
    page.goto(f"{srv['base_url']}/admin/bootstrap")
    page.fill("#secret", srv["admin_secret"])
    page.click("#btn")
    # The bootstrap page tries to redirect to /?popout=view; our harness's
    # root returns a minimal page where we manually load the terminal panel.
    page.wait_for_url("**/*popout=view*", timeout=3000)

    # Inject the terminal panel into the minimal harness.
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
    # Wait for xterm to open AND for the WS to be live (status dot → amber).
    page.wait_for_selector(".terminal-panel .xterm-screen", timeout=5000)
    try:
        page.wait_for_function(
            """() => {
                const s = document.querySelector('.term-status');
                return s && s.dataset.state === 'live';
            }""",
            timeout=10000,
        )
    except Exception as err:
        state = page.evaluate("() => document.querySelector('.term-status')?.dataset.state")
        label = page.evaluate("() => document.querySelector('.term-dot-label')?.textContent")
        raise AssertionError(
            f"terminal never reached 'live'; state={state!r} label={label!r}"
        ) from err

    # Type a command, assert it echoes in the xterm buffer. Scan both the
    # normal and alternate buffers so this still works if the shell (or
    # tmux, when we switch) is rendering into the alt buffer.
    page.focus(".term-mount .xterm-helper-textarea")
    page.keyboard.type("echo HELLO-FROM-BROWSER\n")
    page.wait_for_function(
        """() => {
            const t = window._panel && window._panel._term;
            if (!t) return false;
            for (const key of ['active', 'normal', 'alternate']) {
                const buf = t.buffer[key];
                if (!buf) continue;
                for (let i = 0; i < buf.length; i++) {
                    const line = buf.getLine(i);
                    if (line && line.translateToString(true).includes('HELLO-FROM-BROWSER')) return true;
                }
            }
            return false;
        }""",
        timeout=3000,
    )


def test_terminal_panel_shows_auth_overlay_without_cookie(page, scribe_terminal_server):
    srv = scribe_terminal_server
    page.goto(f"{srv['base_url']}/")
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
    # No cookie → /api/terminal/ticket returns 403 → auth overlay shown.
    page.wait_for_selector(".term-auth-overlay:not([hidden])", timeout=4000)
    # CTA points at /admin/bootstrap
    href = page.get_attribute(".term-auth-cta", "href")
    assert href and href.endswith("/admin/bootstrap")


def test_terminal_panel_no_doubled_input_on_rapid_remount(page, scribe_terminal_server):
    """Regression: show() called twice rapidly must spawn ONE ws, not two.

    The bug was in _connect(): the early-return guard checked `this._ws`,
    but that was only set after an async ticket fetch. So two back-to-back
    show() calls raced past the guard and spawned two WebSockets, each
    attached to the same tmux session. tmux mirrored shell echo to both,
    so every keystroke rendered doubled in xterm.
    """
    srv = scribe_terminal_server
    _authorize(page, srv["base_url"], srv["admin_secret"])

    # Fire mount + two show()s in the same microtask — simulates the race
    # between mount()'s internal auto-show and scribe-app.js's explicit
    # show(). If the guard works, only one attach reaches the server.
    page.add_script_tag(url="/static/js/terminal-panel.js")
    page.evaluate(
        """async () => {
            const p = new window.TerminalPanel({
                apiBase: window.location.origin,
                wsBase: window.location.origin.replace(/^http/, 'ws'),
            });
            window._panel = p;
            await p.mount();
            // Two show() calls without awaiting in between — recreate the
            // original race.
            p.show();
            p.show();
        }"""
    )
    # Give the ticket fetch + ws.open race window a moment to settle.
    page.wait_for_function(
        """() => {
            const s = document.querySelector('.term-status');
            return s && s.dataset.state === 'live';
        }""",
        timeout=10000,
    )

    # Capacity still reflects ONE reservation.
    registry = srv["registry"]
    assert len(registry.items) == 1, (
        f"expected 1 live session, got {len(registry.items)} — doubled-input race still present"
    )


def test_terminal_panel_font_zoom_persists(page, scribe_terminal_server):
    """Ctrl+= / Ctrl+- live-resize AND persist in localStorage."""
    srv = scribe_terminal_server
    _authorize(page, srv["base_url"], srv["admin_secret"])

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

    initial = page.evaluate("() => window._panel.getFontSize()")
    assert initial == 13

    bigger = page.evaluate("() => window._panel.adjustFontSize(3)")
    assert bigger == 16
    assert page.evaluate("() => window._panel._term.options.fontSize") == 16
    assert page.evaluate("() => localStorage.getItem('terminal_font_size')") == "16"

    # Clamp at max.
    clamped_hi = page.evaluate("() => window._panel.setFontSize(999)")
    assert clamped_hi == 26
    # Clamp at min.
    clamped_lo = page.evaluate("() => window._panel.setFontSize(1)")
    assert clamped_lo == 9


def test_terminal_panel_ctrl_plus_keyboard_zoom(page, scribe_terminal_server):
    """Pressing Ctrl+= inside the xterm bumps the font."""
    srv = scribe_terminal_server
    _authorize(page, srv["base_url"], srv["admin_secret"])

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
    page.focus(".term-mount .xterm-helper-textarea")

    before = page.evaluate("() => window._panel.getFontSize()")
    # Ctrl+= — xterm.attachCustomKeyEventHandler returns false to swallow.
    page.keyboard.press("Control+=")
    page.wait_for_timeout(80)
    after = page.evaluate("() => window._panel.getFontSize()")
    assert after == before + 1, f"Ctrl+= should bump font: {before} → {after}"

    # Ctrl+0 resets to default.
    page.keyboard.press("Control+0")
    page.wait_for_timeout(80)
    assert page.evaluate("() => window._panel.getFontSize()") == 13
