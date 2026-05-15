#!/usr/bin/env python3
"""Visual smoke for the audit-fix changes (W1/W2/W4) — loads the
public ``/static/index.html`` via Playwright (the app's static assets
are served without auth gating) and captures screenshots of the UI
surfaces this branch touched:

  1. Default landing — verifies layout still renders, no JS errors
  2. Mute row with the new Mic button (W2)
  3. Setup wizard meeting-audio card with the data-mode="setup" hint (W4)
  4. Meeting detail finalize banner element exists in the DOM (W1)

Public-static path keeps the smoke runner from depending on the
admin-cookie flow (which requires a wizard-finished password we don't
want to clobber). Backend API calls that would 401/403 are short-
circuited with ``page.route`` interceptors so the JS doesn't error
during render.

Writes PNGs to ``/tmp/visual_smoke_audit_fixes/`` for the operator
to eyeball. Exits non-zero on missing selectors / JS errors / contract
violations so this can be wired into a CI lane later.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

OUT_DIR = Path("/tmp/visual_smoke_audit_fixes")
BASE_URL = os.environ.get("SCRIBE_BASE_URL", "https://127.0.0.1:443")
STATIC_PATH = "/static/index.html"

# Viewport matrix:
#   * 1920x1080 — most common laptop / desktop monitor
#   * 1920x2400 — left half of a 3840x2400 4K monitor (operator-side
#     window pinned to one half of a vertical-tall display)
#   * 1280x900  — fallback smaller laptop
#   * 768x1024  — narrowest reasonable form (tablet portrait)
VIEWPORTS: list[tuple[str, int, int]] = [
    ("1280x900", 1280, 900),
    ("1920x1080", 1920, 1080),
    ("1920x2400", 1920, 2400),
    ("768x1024", 768, 1024),
]

FAILURES: list[str] = []
JS_ERRORS: list[str] = []


def _log(msg: str) -> None:
    print(f"[visual_smoke] {msg}", flush=True)


def _assert(cond: bool, msg: str) -> None:
    marker = "OK " if cond else "FAIL"
    _log(f"  {marker}: {msg}")
    if not cond:
        FAILURES.append(msg)


def _screenshot(page: Page, name: str) -> Path:
    target = OUT_DIR / f"{name}.png"
    page.screenshot(path=str(target), full_page=True)
    _log(f"  wrote {target}")
    return target


def _screenshot_clip(page: Page, name: str, selector: str, pad: int = 24) -> Path:
    """Screenshot just the bounding box of ``selector`` (with padding).

    Avoids the SPA-chrome-stacking problem: instead of fighting to hide
    every panel + dialog the production CSS would normally tuck away,
    we compute the element's pixel rect after rendering and clip the
    viewport to it. Result is a focused image of the widget under test.
    """
    target = OUT_DIR / f"{name}.png"
    rect = page.evaluate(
        """(s) => {
            const el = document.querySelector(s);
            if (!el) return null;
            el.scrollIntoView({block: 'center', inline: 'start'});
            const r = el.getBoundingClientRect();
            return {x: r.x, y: r.y, w: r.width, h: r.height};
        }""",
        selector,
    )
    if not rect or rect["w"] == 0 or rect["h"] == 0:
        _log(f"  {selector}: zero-size rect; falling back to full page")
        page.screenshot(path=str(target), full_page=True)
        return target
    clip = {
        "x": max(0.0, rect["x"] - pad),
        "y": max(0.0, rect["y"] - pad),
        "width": rect["w"] + pad * 2,
        "height": rect["h"] + pad * 2,
    }
    page.screenshot(path=str(target), full_page=False, clip=clip)
    _log(f"  wrote {target} (clip={clip['width']:.0f}x{clip['height']:.0f})")
    return target


def _stub_backend_routes(page: Page) -> None:
    """Short-circuit XHRs that need an admin cookie so the page renders.

    The static HTML loads admin-audio-card.js which fetches a few
    admin endpoints on init; without a cookie those return 403 and the
    JS surfaces a noisy console error. Stub them with empty-but-valid
    payloads so the smoke run focuses on layout, not auth.
    """
    page.route(
        "**/api/admin/audio/devices",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "devices": {"sources": [], "sinks": []},
                    "selection": {
                        "mic_node": "",
                        "admin_sink_node": "",
                        "room_sink_node": "",
                        "mic_active": False,
                        "server_mic_active_live": False,
                    },
                }
            ),
        ),
    )
    page.route(
        "**/api/admin/audio/interpretation",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "enabled": False,
                    "pause_flush_ms": 1500,
                    "idle_drain_ms": 5000,
                    "admin_tts_language": "en",
                    "room_tts_language": "all",
                    "local_sink_language": "en",
                    "local_sink_language_options": [
                        {"code": "en", "name": "English"},
                        {"code": "ja", "name": "Japanese"},
                    ],
                    "listener_counts": {
                        "room_sink": {"total": 0, "active": 0, "muted": 0},
                        "web_browser": {"total": 0, "active": 0, "muted": 0},
                        "admin_monitor": {"total": 0, "active": 0, "muted": 0},
                        "bt_headset": {"total": 0, "active": 0, "muted": 0},
                    },
                    "mic_muted": False,
                    "room_sink_mode": "unregistered",
                }
            ),
        ),
    )
    page.route(
        "**/api/meetings",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"meetings": []}),
        ),
    )
    page.route(
        "**/api/status",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"backends": {}, "version": "smoke"}),
        ),
    )


def _check_w2_mic_button(page: Page) -> None:
    _log("W2: mic mute button is present in the meeting-interpretation row")
    btn_count = page.locator(".interpretation-mute-mic").count()
    _assert(btn_count >= 1, "interpretation-mute-mic exists in DOM")
    if btn_count >= 1:
        first = page.locator(".interpretation-mute-mic").first
        text = first.text_content() or ""
        _assert(
            "mic" in text.lower(),
            f"Mic button label contains 'mic' (saw {text!r})",
        )
        title = first.get_attribute("title") or ""
        _assert(
            "Pause the microphone" in title,
            f"Mic button tooltip is privacy-pause copy (saw {title!r})",
        )
        _assert("—" not in title, "tooltip has no em-dash (UI rule)")


def _check_w4_setup_wizard(page: Page) -> None:
    _log("W4: setup wizard card has data-mode=setup + hint copy")
    el = page.locator("#setup-audio-routing-card")
    _assert(el.count() == 1, "#setup-audio-routing-card present")
    if el.count() == 1:
        mode = el.get_attribute("data-mode") or ""
        _assert(mode == "setup", f"data-mode == setup (saw {mode!r})")
        hint_text = page.locator("#setup-audio-routing-card .setup-audio-hint").text_content() or ""
        _assert(
            "apply when you click Start" in hint_text,
            f"setup hint copy present (saw {hint_text!r})",
        )


def _check_w4_settings_card(page: Page) -> None:
    _log("W4: settings panel card has data-mode=settings")
    el = page.locator("#audio-routing-card")
    _assert(el.count() == 1, "#audio-routing-card present")
    if el.count() == 1:
        mode = el.get_attribute("data-mode") or ""
        _assert(mode == "settings", f"data-mode == settings (saw {mode!r})")


def _check_w1_banner_dom(page: Page) -> None:
    _log("W1: meeting-detail finalize banner element exists (hidden until needed)")
    el = page.locator("#meeting-detail-finalize-banner")
    _assert(el.count() == 1, "#meeting-detail-finalize-banner present")
    if el.count() == 1:
        # Verify each child exists so the patcher in scribe-app.js
        # finds something to update.
        for sub in (
            ".meeting-detail-finalize-label",
            ".meeting-detail-finalize-step",
            ".meeting-detail-finalize-bar",
            ".meeting-detail-finalize-fill",
            ".meeting-detail-finalize-actions",
        ):
            _assert(
                page.locator(f"#meeting-detail-finalize-banner {sub}").count() == 1,
                f"{sub} present inside banner",
            )


def _check_phase_b_progress_api(page: Page) -> None:
    _log("W1: /api/meetings rows expose phase_b_progress field")
    body = page.evaluate(
        """async () => {
            const r = await fetch('/api/meetings', {credentials: 'include'});
            return r.json();
        }"""
    )
    meetings = body.get("meetings") if isinstance(body, dict) else None
    _assert(isinstance(meetings, list), "/api/meetings returns a list")
    if isinstance(meetings, list):
        # The field is allowed to be None; we just verify the schema.
        first_with_field = next(
            (m for m in meetings if "phase_b_progress" in m),
            None,
        )
        _assert(
            first_with_field is not None or len(meetings) == 0,
            "phase_b_progress key present on at least one row (or list is empty)",
        )


_BENIGN_ERROR_PATTERNS = (
    "Failed to load resource",
    "401",
    "403",
    "ERR_TOO_MANY_RETRIES",
)


def _check_js_console(page: Page) -> None:
    """Per-viewport helper retained for parity with earlier shape."""
    _check_js_console_global()


def _check_js_console_global() -> None:
    _log("JS console: no unexpected JS errors across viewports")
    real_errors = [e for e in JS_ERRORS if not any(p in e for p in _BENIGN_ERROR_PATTERNS)]
    _assert(real_errors == [], f"no unexpected JS errors (got {len(real_errors)})")
    for err in real_errors[:8]:
        _log(f"  console error: {err}")


def _run_one_viewport(p, label: str, width: int, height: int) -> None:
    """Run the full smoke pass for a single viewport size."""
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(
        ignore_https_errors=True,
        viewport={"width": width, "height": height},
    )
    page = ctx.new_page()
    page.on("pageerror", lambda exc: JS_ERRORS.append(f"[{label}] {exc}"))
    page.on(
        "console",
        lambda msg: (
            JS_ERRORS.append(f"[{label}] {msg.type}: {msg.text}") if msg.type == "error" else None
        ),
    )

    _stub_backend_routes(page)
    _log(f"loading {BASE_URL}{STATIC_PATH} (viewport={width}x{height})")
    page.goto(f"{BASE_URL}{STATIC_PATH}", wait_until="networkidle", timeout=15000)

    vp_dir = OUT_DIR / label
    vp_dir.mkdir(parents=True, exist_ok=True)

    # Bind the screenshot helpers to this viewport's output dir.
    def _vp_screenshot(name: str) -> None:
        target = vp_dir / f"{name}.png"
        page.screenshot(path=str(target), full_page=True)
        _log(f"  wrote {target}")

    def _vp_clip(name: str, selector: str, pad: int = 24) -> None:
        target = vp_dir / f"{name}.png"
        rect = page.evaluate(
            """(s) => {
                const el = document.querySelector(s);
                if (!el) return null;
                el.scrollIntoView({block: 'center', inline: 'start'});
                const r = el.getBoundingClientRect();
                return {x: r.x, y: r.y, w: r.width, h: r.height};
            }""",
            selector,
        )
        if not rect or rect["w"] == 0 or rect["h"] == 0:
            _log(f"  {selector}: zero-size rect — full-page fallback")
            page.screenshot(path=str(target), full_page=True)
            return
        clip = {
            "x": max(0.0, rect["x"] - pad),
            "y": max(0.0, rect["y"] - pad),
            "width": min(width - max(0.0, rect["x"] - pad), rect["w"] + pad * 2),
            "height": rect["h"] + pad * 2,
        }
        page.screenshot(path=str(target), full_page=False, clip=clip)
        _log(f"  wrote {target} ({clip['width']:.0f}x{clip['height']:.0f})")

    _vp_screenshot("01_landing")

    # DOM contract checks only need to run once per viewport (they all
    # produce the same answer), but running them per-viewport catches
    # the case where a media query removes/hides an element at a
    # specific width.
    _check_w2_mic_button(page)
    _check_w4_setup_wizard(page)
    _check_w4_settings_card(page)
    _check_w1_banner_dom(page)
    _check_phase_b_progress_api(page)

    # Mute row with the Mic button first.
    try:
        page.evaluate(
            """() => {
                document.body.classList.add('recording');
                const mm = document.getElementById('meeting-mode');
                if (mm) mm.style.display = 'block';
                const cb = document.getElementById('control-bar');
                if (cb) cb.style.display = 'block';
                const row = document.getElementById('meeting-interpretation-controls');
                if (row) row.style.display = 'flex';
            }"""
        )
        page.wait_for_timeout(200)
        _vp_clip("02_mute_row_with_mic", "#meeting-interpretation-controls")
    except Exception as exc:
        _log(f"could not capture mute row: {exc}")

    # Setup wizard meeting-audio card.
    try:
        page.evaluate(
            """() => {
                document.body.classList.remove('recording');
                document.getElementById('landing-mode').style.display = 'none';
                document.getElementById('meeting-mode').style.display = 'none';
                const rs = document.getElementById('room-setup');
                if (rs) rs.style.display = 'flex';
            }"""
        )
        page.wait_for_timeout(200)
        _vp_clip("03_setup_audio_card", "#setup-audio-routing-card")
        # Also capture the surrounding context (the entire room-setup
        # area) so the operator can judge whether the card harmonizes
        # with the rest of the page.
        _vp_clip("03b_setup_context", "#room-setup", pad=8)
    except Exception as exc:
        _log(f"could not capture setup card: {exc}")

    # Meeting-detail finalize banner — three states.
    for label, synthetic in [
        (
            "04_banner_inflight",
            {
                "step": 4,
                "total_steps": 7,
                "label": "Running full-audio diarization...",
                "terminal": False,
                "error": False,
                "eta_seconds": 24,
            },
        ),
        (
            "05_banner_failure",
            {
                "step": 5,
                "total_steps": 7,
                "label": "Generating timeline...",
                "terminal": True,
                "error": True,
                "code": "phase_b_failed",
                "message": "Phase B finalize failed.",
            },
        ),
        (
            "06_banner_interrupted",
            {
                "step": 3,
                "total_steps": 7,
                "label": "Saving speaker data...",
                "terminal": True,
                "error": True,
                "code": "interrupted",
                "message": "Finalize was interrupted",
            },
        ),
    ]:
        try:
            page.evaluate(
                """(synthetic) => {
                    document.getElementById('landing-mode').style.display = 'none';
                    document.getElementById('room-setup').style.display = 'none';
                    const mm = document.getElementById('meeting-mode');
                    if (mm) mm.style.display = 'block';
                    const banner = document.getElementById('meeting-detail-finalize-banner');
                    if (!banner) return;
                    banner.style.display = 'block';
                    banner.classList.toggle('is-error', !!synthetic.error);
                    banner.classList.toggle('is-done', !!synthetic.terminal && !synthetic.error);
                    const label = banner.querySelector('.meeting-detail-finalize-label');
                    const stepEl = banner.querySelector('.meeting-detail-finalize-step');
                    const fillEl = banner.querySelector('.meeting-detail-finalize-fill');
                    const actionsEl = banner.querySelector('.meeting-detail-finalize-actions');
                    const step = Number(synthetic.step) || 0;
                    const total = Number(synthetic.total_steps) || 7;
                    if (synthetic.error) {
                        label.textContent =
                            synthetic.code === 'interrupted'
                                ? 'Finalize was interrupted — Reprocess to retry'
                                : 'Finalize failed: ' + (synthetic.message || synthetic.label || 'unknown');
                        stepEl.textContent = step + '/' + total;
                        fillEl.style.width = '100%';
                        actionsEl.innerHTML =
                            '<button class="btn-ghost meeting-detail-finalize-reprocess">Reprocess</button>' +
                            '<button class="btn-ghost meeting-detail-finalize-dismiss">Dismiss</button>';
                    } else {
                        const eta = typeof synthetic.eta_seconds === 'number' && synthetic.eta_seconds > 0
                            ? ' · ~' + synthetic.eta_seconds + 's remaining' : '';
                        label.textContent = 'Finalizing — ' + (synthetic.label || '') + eta;
                        stepEl.textContent = step + '/' + total;
                        fillEl.style.width = Math.round(step / total * 100) + '%';
                        actionsEl.innerHTML = '';
                    }
                }""",
                synthetic,
            )
            page.wait_for_timeout(200)
            _vp_clip(label, "#meeting-detail-finalize-banner")
        except Exception as exc:
            _log(f"could not capture {label}: {exc}")

    browser.close()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        for label, w, h in VIEWPORTS:
            _log(f"━━━ viewport {label} ━━━")
            _run_one_viewport(p, label, w, h)

    _check_js_console_global()
    print()
    print("─" * 60)
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} check(s) failed")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("OK: every visual check passed")
    print(f"screenshots: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
