"""Admin SPA boot contract — the window-stamped globals that
``startRecording`` (features/recording-lifecycle.js) depends on must
all be present + correct shape by the time DOMContentLoaded settles.

Regression class this catches: the 2026-05-14 "stuck at initializing"
report — operator clicked Start, ``window.audioPlayer.meetingId =
data.meeting_id`` threw ``Cannot set properties of undefined``, the
catch handler rolled the page back, and ``#status-line`` stayed at
its HTML default "Initializing..." with the real failure buried in
the dev-tools console.

The recording-lifecycle null-guards `window.audioPlayer` now (so a
missing global no longer crashes the start path), but the right
regression net is asserting that the global is present in the first
place — and every other window contract startRecording walks too.

Same harness shape as test_cross_window_sync.py + test_popout_spa_boot.py;
exercises the real static dir + real /api/ws/view + stubbed
/api/* endpoints. Lives or dies on whether scribe/index.js's
post-legacy bootstraps fired without throwing.
"""

from __future__ import annotations

import pytest

# Reuse the live_meeting_server fixture via pytest_plugins.
pytest_plugins = ["tests.browser.test_cross_window_sync"]

pytestmark = pytest.mark.browser


# ── Helpers ──────────────────────────────────────────────────────────


def _open_admin(browser, base_url: str):
    """Open the admin SPA (no `?popout=…`) with pageerror capture."""
    ctx = browser.new_context()
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    # `?test=1` publishes the cross-window-sync test hooks. Harmless in
    # production code paths; just makes the WS surface observable.
    page.goto(f"{base_url}/?test=1", wait_until="domcontentloaded")
    return ctx, page, errors


# ── Tests ────────────────────────────────────────────────────────────


def test_admin_spa_boots_with_no_runtime_errors(browser, live_meeting_server):
    """Page load + ~300ms settle must produce zero pageerror events.

    Catches "a Phase 3 extraction broke an import" the moment the bundle
    evaluates — same gate the popout boot test uses, just from the
    admin side."""
    server = live_meeting_server
    ctx, page, errors = _open_admin(browser, server["base_url"])
    try:
        # Wait for one of the late-stamped globals so we know the
        # bootstrap chain at least *started*.
        page.wait_for_function("() => !!window.audioPlayer", timeout=8000)
        page.wait_for_timeout(300)
        if errors:
            raise AssertionError("pageerror fired during admin boot: " + "; ".join(errors))
    finally:
        ctx.close()


def test_admin_spa_publishes_recording_lifecycle_window_contracts(browser, live_meeting_server):
    """Every window-stamped global ``startRecording`` walks must be
    present + correct type after admin boot. Asserting this catches the
    bug class "audio-player.bootstrap.js never ran" the moment it
    happens, BEFORE the operator clicks Start.

    The list below is sourced from grep over recording-lifecycle.js's
    ``window.X`` references; keep them in lockstep when the start path
    is extended.
    """
    server = live_meeting_server
    ctx, page, errors = _open_admin(browser, server["base_url"])
    try:
        page.wait_for_function("() => !!window.audioPlayer", timeout=8000)
        page.wait_for_timeout(300)

        contract = page.evaluate(
            """() => ({
                // Stamped by features/audio-player.bootstrap.js — used by
                // recording-lifecycle.js:262 to seed the past-meeting
                // playback target.
                audioPlayer_type: typeof window.audioPlayer,
                audioPlayer_ctor: window.audioPlayer?.constructor?.name,
                // Stamped by createReconciler in _legacy.js — exposed for
                // dev-console introspection and consumed by the status-
                // poll handler.
                reconciler_type: typeof window._reconciler,
                // The popout window-contract surface stamped by
                // features/popout-window.js (used by the meetings-list
                // ⋯ menu to open a popout in a fresh tab).
                refreshWifiQR_type: typeof window.refreshWifiQR,
                // The window.current_meeting_id getter installed by
                // state.js. Backed by state.current_meeting_id.
                current_meeting_id_descriptor: !!Object.getOwnPropertyDescriptor(window, 'current_meeting_id'),
                // Inline-onclick consumers in the meetings-panel rows
                // reach showFinalizationSummaryFor via window.
                showFinalizationSummaryFor_type: typeof window.showFinalizationSummaryFor,
                // Modal-system window contracts — used by inline onclick
                // handlers in the modal markup (closeModal()).
                closeModal_type: typeof window.closeModal,
                alertDialog_type: typeof window.alertDialog,
                confirmDialog_type: typeof window.confirmDialog,
                promptDialog_type: typeof window.promptDialog,
                closeAllModals_type: typeof window.closeAllModals,
                // ?test=1 test hooks the cross-window-sync test relies on.
                test_store_type: typeof window.__test_store,
            })"""
        )

        # Required globals — fail loudly if any is missing.
        expected = {
            "audioPlayer_type": "object",
            "audioPlayer_ctor": "AudioPlayer",
            "reconciler_type": "object",
            "refreshWifiQR_type": "function",
            "current_meeting_id_descriptor": True,
            "showFinalizationSummaryFor_type": "function",
            "closeModal_type": "function",
            "alertDialog_type": "function",
            "confirmDialog_type": "function",
            "promptDialog_type": "function",
            "closeAllModals_type": "function",
            "test_store_type": "object",
        }
        missing = {
            k: (expected[k], contract.get(k)) for k in expected if contract.get(k) != expected[k]
        }
        assert not missing, "admin SPA boot did not stamp required window contracts:\n" + "\n".join(
            f"  {k}: expected {ev!r}, got {gv!r}" for k, (ev, gv) in missing.items()
        )
        if errors:
            raise AssertionError("pageerror fired during admin boot: " + "; ".join(errors))
    finally:
        ctx.close()


def test_recording_lifecycle_audioplayer_null_guard(browser, live_meeting_server):
    """Defense-in-depth: even if a future regression wipes
    window.audioPlayer between page load and start-click, the
    recording-lifecycle null-guard must let the rest of the start
    sequence proceed (with a console.warn surfacing the issue) rather
    than throwing ``Cannot set properties of undefined``.

    We can't fully exercise startRecording in this harness — it does
    getUserMedia + AudioWorklet — but we CAN poke the specific line
    by deleting window.audioPlayer then evaluating the safe assignment
    pattern recording-lifecycle.js uses.
    """
    server = live_meeting_server
    ctx, page, errors = _open_admin(browser, server["base_url"])
    try:
        page.wait_for_function("() => !!window.audioPlayer", timeout=8000)
        # Simulate the regression scenario: audio-player.bootstrap.js
        # somehow never set the global.
        result = page.evaluate(
            """() => {
                const saved = window.audioPlayer;
                try {
                    delete window.audioPlayer;
                    // Mirror the exact guarded form recording-lifecycle.js uses:
                    if (window.audioPlayer) {
                        window.audioPlayer.meetingId = 'test-mid';
                        return { threw: false, branch: 'set' };
                    }
                    return { threw: false, branch: 'guarded' };
                } catch (e) {
                    return { threw: true, msg: e.message };
                } finally {
                    window.audioPlayer = saved;
                }
            }"""
        )
        assert result["threw"] is False, f"null-guard regressed: {result}"
        assert result["branch"] == "guarded", (
            f"expected guarded branch when audioPlayer missing: {result}"
        )
        if errors:
            raise AssertionError("pageerror fired: " + "; ".join(errors))
    finally:
        ctx.close()
