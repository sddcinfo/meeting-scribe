"""Back-fill regression tests for the last ~15 fix(*) commits.

Per the QA uplift plan (Phase 1.E), every fix that previously shipped
without a regression test gets one here. Commits already covered by
the per-class test layers are listed in the docstring of the matching
test file (1.A, 1.B, 1.C, 1.D, captive_portal, iptables_rules) and
are NOT duplicated here.

Coverage status of the last ~15 fix commits:

  e5db1c1 fix(popout): auto-reconnect WS + connection-state pill
    → tests/browser/test_cross_window_sync.py::test_popout_reconnect_replays_without_duplicates

  2fcce56 fix(pipeline): don't run script-router on same-script language pairs
    → tests/browser/test_cross_window_sync.py::test_cross_language_pair_renders_in_correct_columns

  65c5476 fix(popout): hide 'Translating…' overlay until a slide is actually loading
    → covered below in test_translating_overlay_starts_hidden_with_no_deck

  c2eee60 fix(popout): unbreak slide panel + collapse to 3 presets
    → tests/browser/test_popout_presentation.py::test_only_three_presets

  ba2ade1 fix(setup): start button stuck on 'Starting...' after failed start
    → covered below in test_start_button_id_matches_html

  7b9c7c5 fix(preflight,server): honor --wait budget, stop set-iter crash
    → covered below in test_preflight_set_iteration_safe

  popout-clear-on-pulse (this PR's motivating regression)
    → tests/browser/test_cross_window_sync.py::test_speaker_pulse_does_not_clear_popout_grid
    → tests/js/segment-store.test.mjs::regression: control messages without segment_id are ignored

  listener-fan-out isolation (found while fixing the popout regression)
    → tests/js/segment-store.test.mjs::regression: a throwing listener does not block subsequent listeners

NOT YET BACK-FILLED (lower-priority / require heavier fixtures):
  - 21678df TTS pipeline resume — requires real TTS backend on GB10
  - meeting-scribe big batches that bundle many fixes together

These remain on the to-do list. The bug-class dashboard
(``scripts/bug_class_report.py``) shows the live distribution.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIBE_APP_JS = REPO_ROOT / "static" / "js" / "scribe-app.js"
INDEX_HTML = REPO_ROOT / "static" / "index.html"


def test_start_button_id_matches_html():
    """Regression for ba2ade1: the JS error handler used a selector
    that didn't match the actual button id, so after any async failure
    the button stayed disabled with 'Starting...' text — user could
    never retry. The fix was renaming the JS selector to match the HTML.

    This test asserts that every selector for the start button in
    scribe-app.js corresponds to an element id that exists in
    index.html. A future rename of one without the other gets caught
    here.
    """
    js = SCRIBE_APP_JS.read_text()
    html = INDEX_HTML.read_text()

    # Collect every `getElementById('btn-...')` and `getElementById("btn-...")`
    # whose id starts with btn-start in scribe-app.js.
    js_ids = set()
    for m in re.finditer(r'getElementById\([\'"]([a-z0-9-]*btn-start[a-z0-9-]*)[\'"]\)', js):
        js_ids.add(m.group(1))

    # Also catch landing/start IDs.
    for m in re.finditer(r'getElementById\([\'"](landing-start-btn|btn-start-meeting)[\'"]\)', js):
        js_ids.add(m.group(1))

    assert js_ids, "couldn't find any start-button getElementById in scribe-app.js"

    # Each must appear in index.html as an id.
    missing = []
    for jid in js_ids:
        if not re.search(rf'id="{re.escape(jid)}"', html):
            missing.append(jid)
    assert not missing, (
        f"scribe-app.js references button ids that don't exist in index.html: "
        f"{missing}. This is the ba2ade1 regression class — selector "
        f"mismatch silently leaves the UI stuck."
    )


def test_translating_overlay_starts_hidden_with_no_deck():
    """Regression for 65c5476: the `#sv-trans-status` 'Translating…'
    overlay was visible by default, so it appeared on the popout slide
    pane even when no slide deck was loaded. The fix was setting
    initial display: none and only flipping it on when a translated
    slide load was actually in flight.

    Static check: the inline style at the overlay's construction site
    must include ``display: none`` so the default-hidden invariant is
    preserved across future refactors.
    """
    js = SCRIBE_APP_JS.read_text()
    # Find the inline construction of the #sv-trans-status overlay.
    # The construction is in _ensureSlideViewer.
    m = re.search(r'id="sv-trans-status"[^>]*style="([^"]+)"', js)
    assert m, "couldn't find #sv-trans-status construction in scribe-app.js"
    style = m.group(1)
    assert "display:none" in style.replace(" ", ""), (
        f"#sv-trans-status overlay must start with display:none — "
        f"otherwise the 'Translating…' chip shows on every popout open. "
        f"Found inline style: {style}"
    )


def test_preflight_set_iteration_safe():
    """Regression for 7b9c7c5: ``state.ws_connections`` was being
    mutated during iteration in the preflight wait loop, raising
    ``RuntimeError: Set changed size during iteration`` and crashing
    the server.

    The fix was wrapping every iteration with ``list(state.ws_connections)``
    so the snapshot is stable even if the underlying set mutates from
    a concurrent disconnect. This test asserts that idiom is in place
    at every iteration site.
    """
    src_dir = REPO_ROOT / "src" / "meeting_scribe"
    bad_sites: list[str] = []
    for path in src_dir.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text()
        # Naive but effective: any `for ws in state.ws_connections:`
        # that ISN'T wrapped in `list(...)` is suspicious.
        for m in re.finditer(r"for\s+ws\s+in\s+([^:]+):", text):
            target = m.group(1).strip()
            if "ws_connections" in target and "list(" not in target:
                line_no = text[: m.start()].count("\n") + 1
                bad_sites.append(f"{path.relative_to(REPO_ROOT)}:{line_no} — `for ws in {target}:`")

    assert not bad_sites, (
        f"unguarded iteration over ws_connections (the 7b9c7c5 regression class):\n"
        + "\n".join(f"  {s}" for s in bad_sites)
        + f"\nWrap with `for ws in list(state.ws_connections):` to snapshot."
    )


def test_segment_count_listener_guards_null_in_popout():
    """Regression for the popout-empty-grid bug we found while writing
    the cross-window sync tests: the `#segment-count` updater listener
    in scribe-app.js crashed in popout mode (no such element), and
    SegmentStore's listener fan-out aborted before
    `_gridRenderer.update` ran — popout rendered nothing.

    Two-layer fix:
      1. SegmentStore now isolates listener errors (covered by
         tests/js/segment-store.test.mjs).
      2. The listener itself guards against null so it doesn't throw
         in the first place (this test).

    Scope this static check to the listener registration site, not
    every getElementById('segment-count') call — the others (e.g. in
    `updateMeter`) only run during admin-side recording where the
    element is guaranteed present.
    """
    js = SCRIBE_APP_JS.read_text()
    # Find the store.subscribe callback that touches segment-count.
    # The callback is a multi-line arrow function ending with });
    m = re.search(
        r"store\.subscribe\(\s*\(\)\s*=>\s*\{(.*?)\}\s*\)\s*;",
        js,
        flags=re.DOTALL,
    )
    assert m, "couldn't find the segment-count store.subscribe callback in scribe-app.js"
    body = m.group(1)
    # The callback must touch segment-count (otherwise we matched the wrong subscribe).
    assert "segment-count" in body, (
        f"matched a store.subscribe that doesn't reference segment-count; "
        f"adjust the regex. Body: {body!r}"
    )
    # And it must guard null.
    has_guard = (
        "if (el)" in body
        or "if(el)" in body
        or "?.textContent" in body
    )
    assert has_guard, (
        "the #segment-count store.subscribe callback must guard against "
        "null (popout DOM doesn't render that element). Without the guard, "
        "this listener throws and (before SegmentStore's per-listener "
        f"try/catch) aborted the fan-out. Body: {body!r}"
    )
