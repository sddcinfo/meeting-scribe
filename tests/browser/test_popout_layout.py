"""End-to-end browser tests for the popout layout system.

Mounts a synthetic FastAPI stack that serves the real index.html +
popout-layout modules + terminal panel, then exercises preset switch,
state persistence, availability pruning, edit-menu mutations, and
keyboard shortcuts. Uses the tmux-socket isolation fixture in
``conftest.py`` — never touches the user's live ``-L scribe`` socket.

Tests target the current 3-preset registry
(``translate / translator / triple``); ``triple`` is the only preset
that includes a terminal pane.
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


@pytest.fixture
def scribe_layout_server(tmp_path, monkeypatch) -> Generator[dict[str, Any]]:
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

    @app.get("/api/admin/settings")
    async def s(_r: Request):
        return JSONResponse(
            {
                "wifi_mode_options": [{"code": "meeting", "name": "Meeting"}],
                "wifi_mode": "meeting",
                "wifi_regdomain_options": [{"code": "JP", "name": "Japan"}],
                "wifi_regdomain": "JP",
                "timezone_options": ["Asia/Tokyo"],
                "timezone": "Asia/Tokyo",
                "tts_voice_mode_options": [{"code": "studio", "name": "Studio"}],
                "tts_voice_mode": "studio",
                "dev_mode": False,
                "admin_ssid": "Dell Admin",
                "admin_password_set": True,
                "wifi_active": True,
                "wifi_ssid": "Dell Demo",
                "wifi_security": {"key_mgmt": "SAE"},
            }
        )

    @app.get("/api/status")
    async def status(_r: Request):
        return JSONResponse({"terminal": {"count": 0, "available": 4, "max": 4}})

    @app.get("/api/meetings")
    async def meetings(_r: Request):
        return JSONResponse({"meetings": []})

    @app.get("/api/languages")
    async def langs(_r: Request):
        return JSONResponse({"languages": []})

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

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


def _authorize_popout(page, srv) -> None:
    page.goto(f"{srv['base_url']}/admin/bootstrap", wait_until="networkidle")
    page.fill("#secret", srv["admin_secret"])
    page.click("#btn")
    page.wait_for_url("**/*popout=view*", timeout=5000)
    page.wait_for_function(
        "() => typeof window._popoutLayoutState === 'function' && !!window._popoutLayoutState()",
        timeout=8000,
    )


def _terminal_buffer_contains(page, sentinel: str) -> bool:
    return page.evaluate(
        """(needle) => {
            const t = window._terminalPanel && window._terminalPanel._term;
            if (!t) return false;
            for (const k of ['active','normal','alternate']) {
                const buf = t.buffer[k]; if (!buf) continue;
                for (let i = 0; i < buf.length; i++) {
                    const line = buf.getLine(i);
                    if (line && line.translateToString(true).includes(needle)) return true;
                }
            }
            return false;
        }""",
        sentinel,
    )


# ── Tests ─────────────────────────────────────────────────────────


def test_popout_picker_renders_three_presets(page, scribe_layout_server):
    _authorize_popout(page, scribe_layout_server)
    state = page.evaluate(
        """() => ({
            pickerExists: !!document.getElementById('popout-layout-picker'),
            termBtnGone: !document.getElementById('popout-term-btn'),
            options: [...document.querySelectorAll('#popout-layout-picker option')].map(o => o.value),
        })"""
    )
    assert state["pickerExists"]
    assert state["termBtnGone"]
    assert state["options"] == ["translate", "translator", "triple"]


def test_switch_to_triple_attaches_terminal(page, scribe_layout_server):
    """Triple is the only preset with a terminal pane — switching to it
    should mount + go live exactly one PTY session.
    """
    srv = scribe_layout_server
    _authorize_popout(page, srv)
    page.select_option("#popout-layout-picker", "triple")
    page.wait_for_function(
        "() => document.querySelector('.lyt-slot[data-panel=terminal] .terminal-panel') !== null",
        timeout=10000,
    )
    page.wait_for_function(
        "() => document.querySelector('.term-status')?.dataset.state === 'live'",
        timeout=10000,
    )
    assert len(srv["registry"].items) == 1
    assert page.evaluate("() => window._popoutLayoutState().preset") == "triple"


def test_terminal_state_survives_preset_swap(page, scribe_layout_server):
    """Cached-root re-parenting: switching away from ``triple`` removes
    the terminal slot from the DOM but the registry keeps the xterm
    instance alive. Switching back must re-parent the same instance,
    not rebuild it — the buffer's previous output must still be there.
    """
    srv = scribe_layout_server
    _authorize_popout(page, srv)
    page.select_option("#popout-layout-picker", "triple")
    page.wait_for_function(
        "() => document.querySelector('.term-status')?.dataset.state === 'live'",
        timeout=10000,
    )
    page.focus(".term-mount .xterm-helper-textarea")
    page.wait_for_timeout(300)
    page.keyboard.type("echo LAYOUT-STATE-42\n")
    page.wait_for_function(
        """() => {
            const t = window._terminalPanel && window._terminalPanel._term;
            if (!t) return false;
            for (const k of ['active','normal','alternate']) {
                const buf = t.buffer[k]; if (!buf) continue;
                for (let i = 0; i < buf.length; i++) {
                    const line = buf.getLine(i);
                    if (line && line.translateToString(true).includes('LAYOUT-STATE-42')) return true;
                }
            }
            return false;
        }""",
        timeout=5000,
    )

    # Switch away to translator (no terminal) — terminal slot leaves the
    # DOM. The cached xterm root in PopoutPanelRegistry survives.
    page.select_option("#popout-layout-picker", "translator")
    page.wait_for_function(
        "() => window._popoutLayoutState().preset === 'translator'",
        timeout=3000,
    )
    page.wait_for_timeout(400)
    # Sanity: the terminal is no longer in the rendered tree.
    assert (
        page.evaluate(
            "() => document.querySelector('.lyt-slot[data-panel=terminal]')"
        )
        is None
    )

    # Back to triple — the same xterm instance must be re-parented and
    # the sentinel from before the swap must still be in the buffer.
    page.select_option("#popout-layout-picker", "triple")
    page.wait_for_function(
        "() => document.querySelector('.lyt-slot[data-panel=terminal] .terminal-panel') !== null",
        timeout=8000,
    )
    page.wait_for_timeout(400)
    assert _terminal_buffer_contains(page, "LAYOUT-STATE-42"), (
        "terminal state was lost on preset swap"
    )


def test_keyboard_ctrl_shift_digit_picks_preset(page, scribe_layout_server):
    """Ctrl+Shift+1..3 jump to ``translate / translator / triple``
    respectively (per ``PRESET_ORDER``).
    """
    _authorize_popout(page, scribe_layout_server)
    page.evaluate("() => document.body.focus()")
    page.keyboard.press("Control+Shift+1")
    page.wait_for_function(
        "() => window._popoutLayoutState().preset === 'translate'",
        timeout=3000,
    )
    page.keyboard.press("Control+Shift+3")
    page.wait_for_function(
        "() => window._popoutLayoutState().preset === 'triple'",
        timeout=3000,
    )
    page.keyboard.press("Control+Shift+2")
    page.wait_for_function(
        "() => window._popoutLayoutState().preset === 'translator'",
        timeout=3000,
    )


def test_keyboard_ctrl_shift_T_toggles_terminal(page, scribe_layout_server):
    """Ctrl+Shift+T flips between ``lastTermPreset`` and
    ``lastNoTermPreset``. With the default state, that's
    ``translator`` ↔ ``triple``.
    """
    _authorize_popout(page, scribe_layout_server)
    page.select_option("#popout-layout-picker", "translator")
    page.wait_for_function(
        "() => window._popoutLayoutState().preset === 'translator'",
        timeout=3000,
    )
    page.evaluate("() => document.body.focus()")
    page.keyboard.press("Control+Shift+T")
    page.wait_for_function(
        "() => window._popoutLayoutState().preset === 'triple'",
        timeout=3000,
    )
    page.keyboard.press("Control+Shift+T")
    page.wait_for_function(
        "() => window._popoutLayoutState().preset === 'translator'",
        timeout=3000,
    )


def test_ratio_persists_per_preset(page, scribe_layout_server):
    """Setting a ratio for ``triple`` does NOT touch ``translator``'s
    bucket — per-preset namespacing is enforced.
    """
    _authorize_popout(page, scribe_layout_server)
    page.select_option("#popout-layout-picker", "triple")
    page.wait_for_function(
        "() => window._popoutLayoutState().preset === 'triple'", timeout=3000
    )
    page.evaluate(
        """() => {
            const S = window.PopoutLayoutStorage;
            let st = S.load();
            st = S.setPreset(st, 'triple');
            st = S.setRatio(st, 'triple:bottom', 0.3);
        }"""
    )
    stored = page.evaluate("() => JSON.parse(localStorage.getItem('popout_layout_v2'))")
    assert stored["ratiosByPreset"]["triple"]["triple:bottom"] == 0.3
    assert "translator" not in stored["ratiosByPreset"]


def test_translate_preset_is_transcript_only(page, scribe_layout_server):
    """``translate`` is a single-leaf preset — only ``transcript`` should
    be on screen, no slides or terminal.
    """
    _authorize_popout(page, scribe_layout_server)
    page.select_option("#popout-layout-picker", "translate")
    page.wait_for_function(
        """() => {
            if (window._popoutLayoutState().preset !== 'translate') return false;
            const panels = [...document.querySelectorAll('.lyt-slot-leaf')].map(s => s.dataset.panel);
            return panels.length === 1 && panels[0] === 'transcript';
        }""",
        timeout=5000,
    )
    panels = page.evaluate(
        "() => [...document.querySelectorAll('.lyt-slot-leaf')].map(s => s.dataset.panel)"
    )
    assert panels == ["transcript"]


def test_transcript_stays_visible_after_preset_swap(page, scribe_layout_server):
    """Regression: an earlier bug where ``display:flex`` on
    ``#transcript-grid`` broke its internal scrolling layout and made
    the translated view invisible after a preset swap. Verify the
    element is in the tree, has size, and stays scrollable.
    """
    _authorize_popout(page, scribe_layout_server)
    page.select_option("#popout-layout-picker", "triple")
    page.wait_for_function("() => window._popoutLayoutState().preset === 'triple'", timeout=4000)
    page.select_option("#popout-layout-picker", "translator")
    page.wait_for_function(
        "() => window._popoutLayoutState().preset === 'translator'", timeout=4000
    )
    page.wait_for_timeout(400)
    info = page.evaluate(
        """() => {
            const g = document.getElementById('transcript-grid');
            if (!g) return null;
            const r = g.getBoundingClientRect();
            const cs = getComputedStyle(g);
            return {
                connected: g.isConnected,
                w: Math.round(r.width), h: Math.round(r.height),
                display: cs.display,
                overflowY: cs.overflowY,
            };
        }"""
    )
    assert info is not None and info["connected"] is True
    assert info["w"] > 100 and info["h"] > 100
    assert info["display"] != "flex"
    assert info["overflowY"] == "auto"


def test_empty_panels_show_placeholder(page, scribe_layout_server):
    """Triple has all three panels; with no meeting running the
    transcript slot should be marked empty so the CSS overlay can
    render the ``waiting for audio`` hint.
    """
    _authorize_popout(page, scribe_layout_server)
    page.select_option("#popout-layout-picker", "triple")
    page.wait_for_function(
        "() => window._popoutLayoutState().preset === 'triple'", timeout=4000
    )
    page.wait_for_timeout(500)
    slots = page.evaluate(
        """() => [...document.querySelectorAll('.lyt-slot-leaf')].map(s => ({
            panel: s.dataset.panel,
            empty: s.dataset.empty,
            text: s.dataset.emptyText,
        }))"""
    )
    trans = next((s for s in slots if s["panel"] == "transcript"), None)
    assert trans and trans["empty"] == "true" and "audio" in (trans["text"] or "").lower()


def test_edit_menu_opens_and_lists_actions(page, scribe_layout_server):
    """Hover a leaf slot, click the ⋮ menu button, assert the 4 actions
    (split-right, split-down, change, remove) render.
    """
    _authorize_popout(page, scribe_layout_server)
    page.select_option("#popout-layout-picker", "triple")
    page.wait_for_function(
        "() => document.querySelectorAll('.lyt-slot-leaf').length >= 2", timeout=4000
    )
    page.evaluate(
        """() => {
            const chrome = document.querySelector('.lyt-slot-leaf .lyt-leaf-chrome');
            chrome.style.opacity = '1';
            chrome.style.pointerEvents = 'auto';
            chrome.querySelector('.lyt-leaf-menu-btn').click();
        }"""
    )
    page.wait_for_selector(".lyt-menu", timeout=2000)
    items = page.evaluate(
        "() => [...document.querySelectorAll('.lyt-menu-item[data-action]')].map(b => b.dataset.action)"
    )
    assert items == ["split-right", "split-down", "change", "remove"]


def test_edit_menu_split_right_creates_custom_tree(page, scribe_layout_server):
    """Clicking ``split-right`` on a leaf transitions the layout to a
    custom tree with that leaf horizontally split against a new panel.
    """
    _authorize_popout(page, scribe_layout_server)
    page.select_option("#popout-layout-picker", "translator")
    page.wait_for_function(
        "() => window._popoutLayoutState().preset === 'translator'", timeout=4000
    )
    page.wait_for_timeout(300)
    page.evaluate(
        """() => {
            const chrome = document.querySelector('.lyt-slot-leaf[data-panel="transcript"] .lyt-leaf-chrome');
            chrome.querySelector('.lyt-leaf-menu-btn').click();
        }"""
    )
    page.wait_for_selector(".lyt-menu", timeout=2000)
    page.click(".lyt-menu-item[data-action='split-right']")
    page.wait_for_function(
        """() => window._popoutLayoutState().preset === 'custom'
             && document.querySelector('.lyt-split[data-dir="h"]') !== null""",
        timeout=5000,
    )
    info = page.evaluate(
        """() => {
            const split = document.querySelector('.lyt-split[data-dir="h"]');
            return {
                hasHorizontalSplit: !!split,
                panelsInDOM: [...document.querySelectorAll('.lyt-slot-leaf')].map(s => s.dataset.panel).sort(),
            };
        }"""
    )
    assert info["hasHorizontalSplit"]
    assert "transcript" in info["panelsInDOM"]
    # ``_suggestPanel`` prefers a panel not already in the tree —
    # translator has transcript+slides, so it suggests terminal.
    assert "terminal" in info["panelsInDOM"]


def test_edit_menu_remove_collapses_split(page, scribe_layout_server):
    """Removing a leaf from a binary split promotes its sibling up."""
    _authorize_popout(page, scribe_layout_server)
    page.select_option("#popout-layout-picker", "triple")
    page.wait_for_function("() => window._popoutLayoutState().preset === 'triple'", timeout=4000)
    page.wait_for_timeout(400)
    page.evaluate(
        """() => {
            const chrome = document.querySelector('.lyt-slot-leaf[data-panel="terminal"] .lyt-leaf-chrome');
            chrome.querySelector('.lyt-leaf-menu-btn').click();
        }"""
    )
    page.wait_for_selector(".lyt-menu", timeout=2000)
    page.click(".lyt-menu-item[data-action='remove']")
    page.wait_for_function("() => window._popoutLayoutState().preset === 'custom'", timeout=4000)
    panels = sorted(
        page.evaluate(
            "() => [...document.querySelectorAll('.lyt-slot-leaf')].map(s => s.dataset.panel)"
        )
    )
    assert panels == ["slides", "transcript"], (
        f"terminal should be gone; got {panels}"
    )


def test_migration_from_terminal_visible_lands_on_triple(page, scribe_layout_server):
    """With only ``terminal_visible=1`` in localStorage, the first load
    infers ``triple`` (the only current preset with a terminal pane).
    Migration path from the old per-key boolean.
    """
    srv = scribe_layout_server
    page.goto(f"{srv['base_url']}/admin/bootstrap", wait_until="networkidle")
    page.evaluate("() => { localStorage.clear(); localStorage.setItem('terminal_visible', '1'); }")
    page.fill("#secret", srv["admin_secret"])
    page.click("#btn")
    page.wait_for_url("**/*popout=view*", timeout=5000)
    page.wait_for_function(
        "() => typeof window._popoutLayoutState === 'function' && !!window._popoutLayoutState()",
        timeout=8000,
    )
    assert page.evaluate("() => window._popoutLayoutState().preset") == "triple"
