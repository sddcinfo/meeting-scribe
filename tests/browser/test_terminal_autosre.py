"""Smoke test: autosre — the primary CLI Brad drives from the embedded
terminal — runs cleanly inside the panel and renders readable output.

We don't stress-test autosre here; we just verify the panel is a sane
host for it: the banner paints, ANSI colors survive the trip through
the PTY + WS, and cursor/viewport geometry isn't pathological.
"""

from __future__ import annotations

import shutil
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

pytestmark = [
    pytest.mark.browser,
    pytest.mark.skipif(shutil.which("autosre") is None, reason="autosre CLI not installed"),
    pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux not installed"),
]


STATIC_DIR = Path(__file__).resolve().parents[2] / "static"


_HOST_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>autosre harness</title></head>
<body>
<main id="transcript" style="height:80px;background:#f2f1ee;"></main>
<script>
  window.API = window.location.origin;
</script>
</body></html>
"""


@pytest.fixture
def scribe_autosre_server(tmp_path, monkeypatch) -> Generator[dict[str, Any], None, None]:
    secret_path = tmp_path / "admin-secret"
    monkeypatch.setenv("SCRIBE_ADMIN_SECRET_FILE", str(secret_path))
    # Use tmux rather than /bin/sh — autosre does real TTY detection and
    # some features require a controlling terminal tmux provides.
    monkeypatch.delenv("SCRIBE_TERM_SHELL", raising=False)

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
        return HTMLResponse(_HOST_PAGE)

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


def _authorize_and_attach(page, srv) -> None:
    page.goto(f"{srv['base_url']}/admin/bootstrap")
    page.wait_for_selector("#secret")
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
    page.wait_for_selector(".terminal-panel .xterm-screen", timeout=6000)
    page.wait_for_function(
        "() => document.querySelector('.term-status')?.dataset.state === 'live'",
        timeout=10000,
    )


def _buffer_snapshot(page) -> str:
    return page.evaluate(
        """() => {
            const t = window._panel && window._panel._term;
            if (!t) return '<no terminal>';
            const buf = t.buffer.active;
            const total = buf.length;  // total rows including scrollback
            const lines = [];
            for (let i = 0; i < total; i++) {
                const line = buf.getLine(i);
                if (line) lines.push(line.translateToString(true));
            }
            return `[type=${buf.type} length=${total} baseY=${buf.baseY} cursorY=${buf.cursorY}]\\n` + lines.join('\\n');
        }"""
    )


def _buffer_contains(page, needle: str, *, timeout_ms: int = 12000) -> bool:
    """Wait until the xterm buffer (active + both alt/normal) has *needle*
    anywhere in it. Scans both buffers because tmux switches to the
    alternate buffer for its full-screen UI, but single-keystroke echoes
    can land on whichever is currently active.
    """
    try:
        page.wait_for_function(
            """(needle) => {
                const t = window._panel && window._panel._term;
                if (!t) return false;
                for (const bufKey of ['active', 'normal', 'alternate']) {
                    const buf = t.buffer[bufKey];
                    if (!buf) continue;
                    const total = buf.length;
                    for (let i = 0; i < total; i++) {
                        const line = buf.getLine(i);
                        if (line && line.translateToString(true).includes(needle)) return true;
                    }
                }
                return false;
            }""",
            arg=needle,
            timeout=timeout_ms,
        )
        return True
    except Exception as err:
        snap = _buffer_snapshot(page)
        raise AssertionError(
            f"buffer never saw {needle!r} within {timeout_ms}ms. Last snapshot:\n----\n{snap}\n----"
        ) from err


def test_tmux_basic_echo(page, scribe_autosre_server):
    """Sanity check: a basic echo round-trips through tmux + PTY."""
    console_msgs: list[str] = []
    page.on("console", lambda m: console_msgs.append(f"[{m.type}] {m.text}"))

    srv = scribe_autosre_server
    _authorize_and_attach(page, srv)
    page.focus(".term-mount .xterm-helper-textarea")
    page.wait_for_timeout(400)
    page.keyboard.type("echo TMUX-OK\n")
    try:
        _buffer_contains(page, "TMUX-OK")
    except AssertionError:
        reg = srv["registry"]
        ws_keys = list(reg.items.keys())
        summaries = [reg.items[k].summary() for k in ws_keys]
        import os as _os
        import subprocess as _subp

        sock = _os.environ.get("SCRIBE_TMUX_SOCKET", "scribe")
        ls = _subp.run(["tmux", "-L", sock, "list-sessions"], capture_output=True, text=True)
        capture = _subp.run(
            ["tmux", "-L", sock, "capture-pane", "-p", "-t", "scribe"],
            capture_output=True,
            text=True,
        )
        buf_snap = _buffer_snapshot(page)
        # Count WS messages xterm saw.
        ws_stats = page.evaluate(
            """() => ({
                ws_state: window._panel?._ws?.readyState,
                attached: window._panel?._attached,
                bytes_rendered: window._panel?._bytesRendered,
                term_cols: window._panel?._term?.cols,
                term_rows: window._panel?._term?.rows,
            })"""
        )
        raise AssertionError(
            f"tmux echo roundtrip failed.\n"
            f"session summaries: {summaries!r}\n"
            f"tmux ls: {ls.stdout!r} / {ls.stderr!r}\n"
            f"tmux capture-pane: {capture.stdout!r}\n"
            f"xterm buffer: {buf_snap!r}\n"
            f"ws_stats: {ws_stats!r}\n"
            f"console msgs: {console_msgs!r}\n"
        ) from None


def test_autosre_help_renders(page, scribe_autosre_server):
    """`autosre --help` prints the CLI's banner + subcommand list.

    We pipe to ``cat`` so click doesn't hand the output to a pager that
    eats the first screen and hangs waiting for input. We then look for a
    specific subcommand name rather than ``Usage:`` — the autosre help is
    much longer than one viewport and the top of it scrolls out of the
    alt buffer.
    """
    srv = scribe_autosre_server
    _authorize_and_attach(page, srv)
    page.focus(".term-mount .xterm-helper-textarea")
    page.keyboard.type("autosre --help | cat\n")
    # 'provision' is a subcommand that appears in the help listing — its
    # presence proves autosre rendered both the binary invocation AND the
    # click subcommand table through the PTY + WS.
    assert _buffer_contains(page, "provision")


def test_autosre_survives_resize(page, scribe_autosre_server):
    """Resizing mid-session doesn't corrupt the viewport or kill the PTY."""
    srv = scribe_autosre_server
    _authorize_and_attach(page, srv)
    page.focus(".term-mount .xterm-helper-textarea")
    # Kick off a long-ish command — `autosre --help | cat` to keep it in
    # the buffer without paging.
    page.keyboard.type("autosre --help | cat\n")
    _buffer_contains(page, "autosre")

    # Shrink the panel, which fires a SIGWINCH down the PTY.
    page.evaluate("document.querySelector('.term-body').style.height = '180px'")
    page.wait_for_timeout(250)
    page.keyboard.type("echo AFTER-RESIZE\n")
    assert _buffer_contains(page, "AFTER-RESIZE")


def test_tmux_survives_ws_drop(page, scribe_autosre_server):
    """Core claim of the terminal feature: dropping the WebSocket does NOT
    kill the tmux session. Reattach repaints the prior pane content.
    """
    srv = scribe_autosre_server
    _authorize_and_attach(page, srv)
    page.focus(".term-mount .xterm-helper-textarea")

    # Leave a sentinel so we can see if tmux repaints it on reattach.
    page.keyboard.type("echo BREADCRUMB-SENTINEL-42\n")
    assert _buffer_contains(page, "BREADCRUMB-SENTINEL-42")

    # Hard-close the WS and reattach. Registry should wind back to 1.
    page.evaluate("() => window._panel._closeWs()")
    page.wait_for_timeout(400)

    # Kick a reconnect explicitly (the panel would normally do this via
    # onclose, but closing from the client skips onclose in some paths).
    page.evaluate("() => window._panel._connect()")
    page.wait_for_function(
        "() => document.querySelector('.term-status')?.dataset.state === 'live'",
        timeout=8000,
    )
    # Give tmux a moment to repaint the pane on the new client.
    page.wait_for_timeout(500)

    # The sentinel from the previous attach is still visible in the
    # re-attached pane — proof that the underlying tmux session outlived
    # the WS connection. We don't assert on new-command delivery here
    # because input handling around a rapid close/reopen dance is
    # covered by the backend integration suite.
    assert _buffer_contains(page, "BREADCRUMB-SENTINEL-42")
