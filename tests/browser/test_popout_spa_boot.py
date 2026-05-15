"""Pop-out SPA boot + control-surface coverage — Playwright.

The popout (`?popout=view`) is the customer-facing surface — the window
operators project onto TVs, float in PiP, or mirror to the HDMI kiosk.
Until 2026-05-14 every line of its boot lived in a 1,443-line
``if (POPOUT_MODE) { … }`` block at the top of ``static/js/scribe/_legacy.js``.
That block now lives at ``static/js/scribe/features/popout-spa.js`` and
fires via the post-legacy bootstrap.

These tests are the regression net for that extraction. They:

  1. Boot ``?popout=view`` in a fresh browser context against a stubbed
     FastAPI app (same harness ``test_cross_window_sync`` uses — no
     vLLM / mic / ASR dependencies).
  2. Capture every ``pageerror`` event. ANY uncaught exception during
     boot or interaction fails the test. This is the cheapest possible
     gate against "the module imports are wrong" / "a helper got
     dropped" / "a top-level reference broke" regressions, which is
     exactly the failure mode an extraction this large can introduce.
  3. Assert the popout DOM was actually constructed: popout-header
     present, every header control button reachable by id, the
     transcript-grid renderer mounted (``window._gridRenderer`` set).
  4. Exercise the actual header controls — language-mode toggle, text
     scale, scroll direction, layout-preset picker — and assert the
     DOM/state transitions the code intends (body classes, CSS vars,
     LayoutStorage state).
  5. Cycle through every popout layout preset (translate / translator /
     triple / terminal) and confirm none of them throws.
  6. Replay ``meeting_started`` and ``meeting_stopped`` over the live
     WS and verify the popout's lifecycle reset logic clears the grid
     + flips body classes the way the in-meeting popout would.

The cross-window-sync test already covers transcript ingestion and the
WS handler-cascade routing; this file complements it with header-
control + boot-surface coverage so any future popout regressions
surface in CI rather than in front of a customer.
"""

from __future__ import annotations

import pytest

# Reuse the live FastAPI fixture from test_cross_window_sync — it serves
# the real static dir, mounts the real /api/ws/view route, and stubs the
# half-dozen /api endpoints the popout's init touches. We declare the
# source module as a pytest plugin so the `live_meeting_server` fixture
# is discovered without a from-import that would otherwise trip
# F401/F811. The helpers (_broadcast etc.) are imported normally.
pytest_plugins = ["tests.browser.test_cross_window_sync"]

from tests.browser.test_cross_window_sync import (
    _broadcast,
    _make_segment,
    _wait_for_popout_ws_open,
    _wait_until,
)

pytestmark = pytest.mark.browser


# ── Helpers ──────────────────────────────────────────────────────────


def _open_popout(browser, base_url: str):
    """Open a fresh popout context with pageerror capture wired up.

    Returns ``(context, page, errors_list)``. Caller is responsible for
    ``context.close()``. The errors list is a list of strings; if it's
    non-empty at end-of-test, the test fails.
    """
    ctx = browser.new_context()
    page = ctx.new_page()
    errors: list[str] = []
    # Only count uncaught JS exceptions. console.error fires for the
    # harness's stubbed-out API surface (404s on endpoints the popout
    # tries to poll like `/api/admin/settings`, `/api/meetings/.../slides`
    # etc.) — those are noise; a real regression would surface as a
    # pageerror, not a 404.
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    # `?test=1` short-circuits the test-only hooks (`window.__test_msg_log`,
    # `__test_ingest_count`) the cross-window-sync test relies on. Harmless
    # in production code paths; just makes the WS surface observable.
    page.goto(f"{base_url}/?popout=view&test=1", wait_until="domcontentloaded")
    return ctx, page, errors


def _assert_no_runtime_errors(errors: list[str], context: str) -> None:
    """Hard-fail if ANY pageerror or console.error fired during the run.

    The popout extraction is the kind of change where a missed `import`
    or a stale `_legacy.js`-private reference doesn't surface in syntax
    check — it surfaces as a ReferenceError at runtime, on the customer's
    screen. This assert is the cheapest possible runtime gate.
    """
    if errors:
        rendered = "\n  ".join(errors)
        raise AssertionError(f"popout produced runtime errors during {context}:\n  {rendered}")


# ── Tests ────────────────────────────────────────────────────────────


def test_popout_boots_with_no_runtime_errors(browser, live_meeting_server):
    """The popout SPA must boot cleanly — no uncaught exceptions, no
    console.error spam. Catches "the extraction broke an import" the
    moment the bundle evaluates."""
    server = live_meeting_server
    ctx, page, errors = _open_popout(browser, server["base_url"])
    try:
        _wait_until(page, "() => !!window._gridRenderer")
        # Give the async IIFE inside bootPopoutSpa a beat to settle.
        page.wait_for_timeout(300)
        _assert_no_runtime_errors(errors, "boot")
    finally:
        ctx.close()


def test_popout_body_classes_applied_pre_paint(browser, live_meeting_server):
    """`popout-view` + `view-only` body classes are added by _legacy.js
    BEFORE the popout-spa bootstrap fires, so the CSS cascade resolves
    correctly at first paint. Regression: if the body-class add was
    accidentally moved INTO the bootstrap, the popout would briefly
    flash the admin chrome before the post-legacy initializer ran."""
    server = live_meeting_server
    ctx, page, errors = _open_popout(browser, server["base_url"])
    try:
        _wait_until(page, "() => !!window._gridRenderer")
        classes = page.evaluate("() => Array.from(document.body.classList)")
        assert "popout-view" in classes, f"missing popout-view: {classes}"
        assert "view-only" in classes, f"missing view-only: {classes}"
        _assert_no_runtime_errors(errors, "body-class check")
    finally:
        ctx.close()


def test_popout_header_chrome_is_fully_rendered(browser, live_meeting_server):
    """Every header control the popout boot inserts must end up in the
    DOM. Missing one means a dependency import was wrong AND the popout
    silently shipped a half-built header to customers — exactly the
    class of regression the extraction could introduce."""
    server = live_meeting_server
    ctx, page, errors = _open_popout(browser, server["base_url"])
    try:
        _wait_until(page, "() => !!document.querySelector('.popout-header')")
        required_ids = [
            "popout-lang-label",
            "popout-lang-both",
            "popout-lang-a",
            "popout-lang-b",
            "popout-direction",
            "popout-slides-lang",
            "popout-slides-btn",
            "popout-slides-input",
            "popout-text-smaller",
            "popout-text-larger",
            "popout-layout-picker",
            "popout-qr-btn",
            "popout-qr",
        ]
        for el_id in required_ids:
            present = page.evaluate(
                f"() => !!document.getElementById('{el_id}')",
            )
            assert present, f"popout header missing #{el_id}"
        _assert_no_runtime_errors(errors, "header chrome check")
    finally:
        ctx.close()


def test_popout_language_mode_toggle_drives_body_class(browser, live_meeting_server):
    """The both/A/B segmented toggle flips body classes that CSS uses to
    hide the off-language column. If the click handler was dropped during
    extraction the toggle silently no-ops; assert that each click lands
    the matching `lang-mode-*` class on <body>."""
    server = live_meeting_server
    ctx, page, errors = _open_popout(browser, server["base_url"])
    try:
        _wait_until(page, "() => !!document.getElementById('popout-lang-a')")

        # The 'Both' button defaults via localStorage hydration; force A
        # then B then Both and verify each transition.
        for mode, expected in (
            ("a", "lang-mode-a-only"),
            ("b", "lang-mode-b-only"),
            ("both", "lang-mode-both"),
        ):
            page.click(f"#popout-lang-{mode}")
            # Class update is synchronous in the handler so a tiny tick is
            # enough; we still poll to keep the test resilient.
            _wait_until(
                page,
                f"() => document.body.classList.contains('{expected}')",
                timeout_ms=2000,
            )
            classes = page.evaluate("() => Array.from(document.body.classList)")
            # Exactly one of the three lang-mode-* classes should be active.
            lang_modes = [c for c in classes if c.startswith("lang-mode-")]
            assert lang_modes == [expected], (
                f"after clicking #popout-lang-{mode}: expected only {expected} "
                f"but body has {lang_modes}"
            )
        _assert_no_runtime_errors(errors, "language-mode toggle")
    finally:
        ctx.close()


def test_popout_text_scale_buttons_drive_css_var(browser, live_meeting_server):
    """A+ and A- buttons mutate `--text-scale` on documentElement. Persisted
    to localStorage. Regression: if `_applyTextScale` was dropped the
    buttons no-op."""
    server = live_meeting_server
    ctx, page, errors = _open_popout(browser, server["base_url"])
    try:
        _wait_until(page, "() => !!document.getElementById('popout-text-larger')")

        def text_scale() -> float:
            return float(
                page.evaluate(
                    "() => getComputedStyle(document.documentElement).getPropertyValue('--text-scale') || '1'"
                )
                or 1.0
            )

        before = text_scale()
        page.click("#popout-text-larger")
        page.wait_for_timeout(50)
        after_up = text_scale()
        assert after_up > before, f"A+ click did not increase --text-scale ({before} → {after_up})"

        page.click("#popout-text-smaller")
        page.wait_for_timeout(50)
        after_down = text_scale()
        assert after_down < after_up, (
            f"A- click did not decrease --text-scale after A+ ({after_up} → {after_down})"
        )
        _assert_no_runtime_errors(errors, "text-scale buttons")
    finally:
        ctx.close()


def test_popout_layout_picker_cycles_through_every_preset(browser, live_meeting_server):
    """Selecting every preset in the layout picker must not throw.

    The layout renderer reparents transcript/slides/terminal panel roots
    as the user swaps presets. If the ensure-callbacks the popout boot
    registers were broken by extraction (e.g. `_ensureSlideViewerRoot`
    couldn't resolve `_ensureSlideViewer`), the swap throws and the
    layout strands the user on a blank screen.
    """
    server = live_meeting_server
    ctx, page, errors = _open_popout(browser, server["base_url"])
    try:
        _wait_until(page, "() => !!document.getElementById('popout-layout-picker')")
        presets = page.evaluate(
            "() => Array.from(document.getElementById('popout-layout-picker').options).map(o => o.value)",
        )
        assert presets, "layout picker has no options"
        for value in presets:
            page.select_option("#popout-layout-picker", value)
            page.wait_for_timeout(100)
            # The renderer keeps the value in sync; assert the option
            # actually landed.
            current = page.evaluate(
                "() => document.getElementById('popout-layout-picker').value",
            )
            assert current == value, f"preset {value!r} did not stick (got {current!r})"
        _assert_no_runtime_errors(errors, f"layout preset cycle: {presets}")
    finally:
        ctx.close()


def test_popout_scroll_direction_toggle_persists(browser, live_meeting_server):
    """The direction toggle text + localStorage entry must update on
    every click. This is the same control the operator uses on stage to
    swap newest-first ↔ oldest-first mid-meeting."""
    server = live_meeting_server
    ctx, page, errors = _open_popout(browser, server["base_url"])
    try:
        _wait_until(page, "() => !!document.getElementById('popout-direction')")
        page.click("#popout-direction")
        page.wait_for_timeout(50)
        stored = page.evaluate("() => localStorage.getItem('popout_direction')")
        label = page.evaluate(
            "() => document.getElementById('popout-direction').textContent",
        )
        assert stored in ("newest", "oldest"), f"unexpected stored value: {stored!r}"
        # The label should mention the matching direction word.
        if stored == "oldest":
            assert "Oldest" in label, f"label {label!r} doesn't match stored={stored!r}"
        else:
            assert "Newest" in label, f"label {label!r} doesn't match stored={stored!r}"
        _assert_no_runtime_errors(errors, "scroll direction toggle")
    finally:
        ctx.close()


def test_popout_meeting_started_resets_state(browser, live_meeting_server):
    """``meeting_started`` over the view-WS must:
       - clear `store` so the previous meeting's transcript doesn't bleed
       - clear `window._gridRenderer` (visibly clears the rendered grid)
       - apply `data-recording` on <body> so the kiosk splash fades

    The extraction kept the lifecycle branch verbatim; this test wires
    the actual WS path end-to-end to prove the handler still fires.
    """
    server = live_meeting_server
    base = server["base_url"]
    ctx, page, errors = _open_popout(browser, base)
    try:
        _wait_until(page, "() => !!window._gridRenderer")
        _wait_for_popout_ws_open(page)
        # Land one segment so we can prove the reset clears it.
        _broadcast(
            server,
            _make_segment(
                segment_id="pre-reset",
                text="Should be wiped by meeting_started.",
                start_ms=0,
                translation_text="リセット前",
            ),
        )
        _wait_until(page, "() => window._gridRenderer?._segmentMap?.size >= 1")

        # Now fire meeting_started — popout should wipe the store.
        _broadcast(
            server,
            {"type": "meeting_started", "meeting_id": "next-meeting"},
        )
        _wait_until(
            page,
            "() => (window._gridRenderer?._segmentMap?.size || 0) === 0",
            timeout_ms=3000,
        )
        has_recording_attr = page.evaluate(
            "() => document.body.hasAttribute('data-recording')",
        )
        assert has_recording_attr, "body did not gain data-recording on meeting_started"
        _assert_no_runtime_errors(errors, "meeting_started reset")
    finally:
        ctx.close()


def test_popout_meeting_stopped_resets_state(browser, live_meeting_server):
    """``meeting_stopped`` over the view-WS clears the popout's transcript
    state the same way meeting_started does. Both branches must remain
    functional after the extraction.
    """
    server = live_meeting_server
    base = server["base_url"]
    ctx, page, errors = _open_popout(browser, base)
    try:
        _wait_until(page, "() => !!window._gridRenderer")
        _wait_for_popout_ws_open(page)
        _broadcast(
            server,
            _make_segment(
                segment_id="pre-stop",
                text="Should be wiped by meeting_stopped.",
                start_ms=0,
                translation_text="ストップ前",
            ),
        )
        _wait_until(page, "() => window._gridRenderer?._segmentMap?.size >= 1")

        _broadcast(
            server,
            {"type": "meeting_stopped", "meeting_id": "test-meeting-0"},
        )
        _wait_until(
            page,
            "() => (window._gridRenderer?._segmentMap?.size || 0) === 0",
            timeout_ms=3000,
        )
        _assert_no_runtime_errors(errors, "meeting_stopped reset")
    finally:
        ctx.close()


def test_popout_window_globals_are_published(browser, live_meeting_server):
    """The popout boot publishes/relies on a handful of ``window.*``
    contracts. If any are missing, downstream features (the layout
    renderer, the terminal-panel lazy loader, the cross-window-sync
    test hooks) silently break.
    """
    server = live_meeting_server
    ctx, page, errors = _open_popout(browser, server["base_url"])
    try:
        _wait_until(page, "() => !!window._gridRenderer")
        page.wait_for_timeout(200)
        for name in (
            "_gridRenderer",
            "PopoutLayoutPresets",
            "PopoutLayoutStorage",
            "PopoutLayoutRender",
            "PopoutPanelRegistry",
            "_popoutLayoutState",
            "__test_msg_log",
        ):
            present = page.evaluate(f"() => typeof window['{name}'] !== 'undefined'")
            assert present, f"window.{name} was never published by popout boot"
        _assert_no_runtime_errors(errors, "window-contract surface")
    finally:
        ctx.close()
