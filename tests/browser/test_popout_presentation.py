"""Unit-ish browser test for the popout "presentation" view (D1).

Verifies the contract pieces independently — full live-meeting +
slide-upload e2e is intentionally out of scope (significantly heavier
fixture rig). What's covered here:

  * The renamed `translator` preset (now labeled "Presentation")
    actually has the shape the user wants: vertical split, transcript
    on top, slides on the bottom.
  * SlideViewer constructs both an original AND a translated `<img>`
    element side-by-side and exposes navigation methods.
  * The slide URLs follow the canonical `/api/meetings/{id}/slides/{i}/{original|translated}`
    pattern that the popout consumes.

For the live e2e version (start a meeting, upload PPTX, advance slides,
assert both images load via the network), see the manual checklist in
`tests/browser/MANUAL_TEST_CARD.md`. Browser-driven full-stack tests
of the slide pipeline run in `test_slides_convert.py` against the
real backend.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, Response

pytestmark = pytest.mark.browser


STATIC_DIR = Path(__file__).resolve().parents[2] / "static"

# 1×1 PNG so the slide-viewer onload handlers fire deterministically
# without needing real slide-render infrastructure on the test path.
_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c63000100000005000100c1d70d0a0000000049454e44ae426082"
)


_HOST = """<!doctype html>
<html><head><meta charset="utf-8"><title>popout presentation</title></head>
<body>
<div id="slides-host" style="width:600px;height:400px"></div>
<script src="/static/js/popout-layout.js"></script>
<script src="/static/js/popout-layout-presets.js"></script>
<script src="/static/js/popout-layout-storage.js"></script>
<script src="/static/js/slide-viewer.js"></script>
</body></html>
"""


@pytest.fixture
def presentation_server() -> Generator[dict[str, Any]]:
    app = FastAPI()

    @app.get("/")
    async def index():
        return HTMLResponse(_HOST)

    @app.get("/static/js/{fname}")
    async def js(fname: str):
        path = STATIC_DIR / "js" / fname
        if not path.exists():
            return HTMLResponse(status_code=404, content="")
        return FileResponse(path, media_type="application/javascript")

    @app.get("/api/meetings/{mid}/slides/{idx}/{kind}")
    async def slide(mid: str, idx: int, kind: str):
        return Response(content=_TINY_PNG, media_type="image/png")

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    import time

    deadline = time.monotonic() + 5.0
    while (not server.started or not server.servers) and time.monotonic() < deadline:
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield {"base_url": f"http://127.0.0.1:{port}"}
    server.should_exit = True
    thread.join(timeout=3.0)


def _load(page, url: str) -> None:
    page.goto(url, wait_until="networkidle")
    page.wait_for_function(
        "() => window.PopoutLayout && window.PopoutLayoutPresets && window.PopoutLayoutStorage && SlideViewer",
        timeout=3000,
    )


def test_presentation_preset_shape(page, presentation_server):
    """The renamed 'translator' preset must keep its vertical-split
    layout (transcript top, slides bottom) so the popout still renders
    a presentation view as the user expects."""
    _load(page, presentation_server["base_url"])
    shape = page.evaluate(
        """() => {
            const p = window.PopoutLayoutPresets.PRESETS.translator;
            const t = p.tree;
            return {
                slug: p.slug,
                label: p.label,
                description: p.description,
                kind: t.kind,
                dir: t.dir,
                ratio: t.ratio,
                a_panel: t.a && t.a.panel,
                b_panel: t.b && t.b.panel,
            };
        }"""
    )
    assert shape["slug"] == "translator"
    assert shape["label"] == "Presentation"
    assert "Translation top" in shape["description"]
    assert "side-by-side" in shape["description"]
    assert shape["kind"] == "split"
    assert shape["dir"] == "v"
    assert shape["a_panel"] == "transcript"
    assert shape["b_panel"] == "slides"


def test_slide_viewer_renders_both_panes(page, presentation_server):
    """SlideViewer mounts BOTH an original and a translated <img> as
    siblings — that's the side-by-side render. We don't wait for the
    img onload (which depends on the meeting upload pipeline); only
    that the DOM elements are constructed and the URLs are correctly
    formed."""
    _load(page, presentation_server["base_url"])
    out = page.evaluate(
        """() => {
            const host = document.getElementById('slides-host');
            const v = new SlideViewer(host, 'mtg-test', { isAdmin: false });
            v.show(3, 'deck-abc');
            v._loadSlide(0);
            const orig = host.querySelector('img.sv-orig, img.sv-original, .sv-pane-orig img, [class*="orig"] img');
            const trans = host.querySelector('img.sv-trans, img.sv-translated, .sv-pane-trans img, [class*="trans"] img');
            return {
                orig_present: !!v._origImg,
                trans_present: !!v._transImg,
                orig_src: v._origImg && v._origImg.src,
                trans_src: v._transImg && v._transImg.src,
                has_nav: !!v._navBar,
                has_prev: typeof v.prev === 'function',
                has_next: typeof v.next === 'function',
                total_slides: v._totalSlides,
            };
        }"""
    )
    assert out["orig_present"] is True
    assert out["trans_present"] is True
    assert out["orig_src"].endswith("/api/meetings/mtg-test/slides/0/original?d=deck-abc")
    assert out["trans_src"].endswith("/api/meetings/mtg-test/slides/0/translated?d=deck-abc")
    assert out["has_nav"] is True
    assert out["has_prev"] is True
    assert out["has_next"] is True
    assert out["total_slides"] == 3


def test_only_three_presets(page, presentation_server):
    """The popout offers exactly three layouts: Translate, Presentation
    (translator), Triple stack. Removed: developer, fullstack,
    sidebyside, demo."""
    _load(page, presentation_server["base_url"])
    presets = page.evaluate(
        """() => {
            const P = window.PopoutLayoutPresets;
            return {
                order: P.PRESET_ORDER,
                labels: Object.fromEntries(
                    P.PRESET_ORDER.map(s => [s, P.PRESETS[s].label])
                ),
            };
        }"""
    )
    assert presets["order"] == ["translate", "translator", "triple"]
    assert presets["labels"] == {
        "translate": "Translate",
        "translator": "Presentation",
        "triple": "Triple stack",
    }


def test_translate_preset_is_transcript_only(page, presentation_server):
    """Translate preset is a single transcript leaf — no slides, no terminal."""
    _load(page, presentation_server["base_url"])
    shape = page.evaluate(
        """() => {
            const t = window.PopoutLayoutPresets.PRESETS.translate.tree;
            return {
                kind: t.kind,
                panel: t.panel,
                panels: window.PopoutLayout.collectPanels(t),
            };
        }"""
    )
    assert shape["kind"] == "leaf"
    assert shape["panel"] == "transcript"
    assert shape["panels"] == ["transcript"]


def test_legacy_preset_slugs_migrate(page, presentation_server):
    """Stale localStorage from before the 6→3 collapse must migrate
    to a current preset rather than leaving the picker broken."""
    _load(page, presentation_server["base_url"])
    migrated = page.evaluate(
        """() => {
            // Seed localStorage with a v2 state that uses a removed slug
            localStorage.setItem('popout_layout_v2', JSON.stringify({
                version: 2,
                preset: 'fullstack',
                lastTermPreset: 'developer',
                lastNoTermPreset: 'demo',
                ratiosByPreset: {},
                customTree: null,
            }));
            return window.PopoutLayoutStorage.load();
        }"""
    )
    # fullstack → triple, developer → translate, demo → translator
    assert migrated["preset"] == "triple"
    assert migrated["lastTermPreset"] == "translate"
    assert migrated["lastNoTermPreset"] == "translator"


def test_slide_viewer_navigation_updates_both_imgs(page, presentation_server):
    """Advancing slides must update BOTH images, not just the original.
    Regression guard: if anyone refactors `_loadSlide` to skip the
    translated update, the popout silently shows a stale translation."""
    _load(page, presentation_server["base_url"])
    urls = page.evaluate(
        """() => {
            const host = document.getElementById('slides-host');
            const v = new SlideViewer(host, 'mtg-x', { isAdmin: false });
            v.show(5, 'deck-d');
            v._loadSlide(2);
            const orig2 = v._origImg.src;
            const trans2 = v._transImg.src;
            v._loadSlide(3);
            return {
                orig2, trans2,
                orig3: v._origImg.src,
                trans3: v._transImg.src,
            };
        }"""
    )
    assert "/slides/2/original" in urls["orig2"]
    assert "/slides/2/translated" in urls["trans2"]
    assert "/slides/3/original" in urls["orig3"]
    assert "/slides/3/translated" in urls["trans3"]
    assert urls["orig2"] != urls["orig3"]
    assert urls["trans2"] != urls["trans3"]
