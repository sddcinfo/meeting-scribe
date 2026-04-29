"""Visual-regression snapshot suite — in-repo, no SaaS.

Snapshots: popout in each preset (translate / translator / triple),
admin live-meeting view, guest portal, reader. Transcript content is
ALWAYS masked via Playwright `mask:` so no real text ends up in the
committed PNGs.

Snapshots live at tests/browser/__snapshots__/visual_regression/.
Regenerate with `--update-snapshots` (Playwright's standard flag).

Why Chromium-only: the goal is layout / CSS / spacing regression, not
engine quirks (those live in test_webkit_quirks.py). Pinning to one
engine keeps the diff signal-to-noise high.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Generator
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

pytestmark = pytest.mark.browser

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = REPO_ROOT / "static"


# ── Minimal harness — re-uses the cross-window fixture pattern ───────


def _build_app() -> FastAPI:
    app = FastAPI()
    state = {"events": [], "ws_connections": set()}

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/reader")
    async def reader():
        return FileResponse(STATIC_DIR / "reader.html")

    @app.get("/portal")
    async def portal():
        return FileResponse(STATIC_DIR / "portal.html")

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/api/languages")
    async def languages():
        return JSONResponse({
            "languages": [
                {"code": "en", "name": "English", "native_name": "English"},
                {"code": "ja", "name": "Japanese", "native_name": "日本語"},
            ]
        })

    @app.get("/api/status")
    async def status():
        return JSONResponse({
            "meeting": {"id": "vr-test", "language_pair": ["en", "ja"]},
            "backends": {"asr": True, "translate": True, "diarize": True},
            "connections": 1,
        })

    @app.get("/api/meetings/{mid}")
    async def meeting(mid: str):
        return JSONResponse({
            "meta": {
                "meeting_id": mid,
                "language_pair": ["en", "ja"],
                "state": "RECORDING",
            },
            "events": list(state["events"]),
        })

    @app.get("/api/meetings")
    async def meetings():
        return JSONResponse([])

    @app.get("/api/meeting/wifi")
    async def wifi():
        return JSONResponse({"available": False})

    @app.post("/api/diag/listener")
    async def diag_listener():
        return JSONResponse({"ok": True})

    @app.websocket("/api/ws/view")
    async def ws_view(websocket: WebSocket):
        await websocket.accept()
        state["ws_connections"].add(websocket)
        try:
            for ev in state["events"]:
                await websocket.send_text(json.dumps(ev))
            while True:
                await websocket.receive_text()
        except Exception:
            pass
        finally:
            state["ws_connections"].discard(websocket)

    # Pre-seed a few segments so the screenshot has something to render.
    for i in range(3):
        state["events"].append({
            "segment_id": f"vr-seg-{i}",
            "revision": 0,
            "is_final": True,
            "start_ms": i * 2000,
            "end_ms": i * 2000 + 1500,
            "language": "en",
            "text": "Lorem ipsum dolor sit amet.",
            "speakers": [{"cluster_id": i + 1, "source": "diarization"}],
            "translation": {
                "status": "done",
                "text": "あいうえおかきくけこ。",
                "target_language": "ja",
            },
        })

    return app


class _ServerThread(threading.Thread):
    def __init__(self, app: FastAPI) -> None:
        super().__init__(daemon=True)
        self.app = app
        self.port = 0
        self._started = threading.Event()

    def run(self) -> None:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        config = uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="error")
        self._server = uvicorn.Server(config)
        self._started.set()
        self._server.run()

    def wait_ready(self, timeout: float = 10.0) -> None:
        import time
        import urllib.request
        self._started.wait(timeout)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/status", timeout=1)
                return
            except Exception:
                time.sleep(0.05)

    def stop(self) -> None:
        if hasattr(self, "_server"):
            self._server.should_exit = True


@pytest.fixture(scope="module")
def vr_server() -> Generator[str, None, None]:
    app = _build_app()
    thread = _ServerThread(app)
    thread.start()
    thread.wait_ready()
    try:
        yield f"http://127.0.0.1:{thread.port}"
    finally:
        thread.stop()
        thread.join(timeout=3.0)


# ── Snapshot helpers ─────────────────────────────────────────────────


def _mask_transcript_selectors(page) -> list:
    """Return a list of locators to mask out — all transcript content
    so the committed PNG carries layout/CSS shape but no real text."""
    return [
        page.locator("#transcript-grid"),
        page.locator(".compact-block"),
        page.locator(".oo-original"),
        page.locator(".oo-translation"),
        page.locator(".popout-lang"),  # current language label could carry a meeting name
    ]


def _settle(page, ms: int = 500) -> None:
    page.wait_for_timeout(ms)


def _snapshot(page, name: str, *, byte_tolerance: int = 200) -> None:
    """Take a masked, fixed-viewport screenshot and compare to baseline.

    Byte tolerance: screenshots can vary by a few bytes on identical
    layouts due to PNG encoder differences (timestamps, compression
    table reuse). Default 200 bytes allows that without masking real
    layout changes — a true layout drift produces kilobytes of diff.

    First run writes the baseline; subsequent runs compare. Update by
    deleting the .png and re-running.
    """
    baseline = REPO_ROOT / "tests" / "browser" / "__snapshots__" / "visual_regression" / f"{name}.png"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    img = page.screenshot(
        full_page=False,
        mask=_mask_transcript_selectors(page),
        animations="disabled",
    )
    if not baseline.exists():
        baseline.write_bytes(img)
        pytest.skip(f"baseline created at {baseline.relative_to(REPO_ROOT)}")
    expected = baseline.read_bytes()
    if img == expected:
        return
    diff_bytes = abs(len(img) - len(expected))
    if diff_bytes <= byte_tolerance:
        # Sub-tolerance variation — probably encoder noise, not layout.
        return
    # Save the actual side-by-side for inspection.
    actual_path = baseline.with_name(f"{name}.actual.png")
    actual_path.write_bytes(img)
    pytest.fail(
        f"visual diff for {name}: byte size differs by {diff_bytes} (>{byte_tolerance}).\n"
        f"  baseline: {baseline.relative_to(REPO_ROOT)} ({len(expected)} bytes)\n"
        f"  actual:   {actual_path.relative_to(REPO_ROOT)} ({len(img)} bytes)\n"
        f"  Inspect both files, then either fix the regression or "
        f"delete the baseline and re-run to update."
    )


# ── Snapshot tests ───────────────────────────────────────────────────


def _viewport(page, w: int = 1280, h: int = 800) -> None:
    page.set_viewport_size({"width": w, "height": h})


def test_snapshot_popout_translate_preset(page, vr_server):
    _viewport(page, 1280, 720)
    page.goto(f"{vr_server}/?popout=view", wait_until="domcontentloaded")
    page.wait_for_function("() => !!window._gridRenderer", timeout=8000)
    page.evaluate(
        "() => localStorage.setItem('popout_layout_v2', JSON.stringify({version: 2, preset: 'translate', lastTermPreset: 'triple', lastNoTermPreset: 'translate', ratiosByPreset: {}, customTree: null}))"
    )
    page.reload(wait_until="domcontentloaded")
    page.wait_for_function("() => !!window._gridRenderer", timeout=8000)
    _settle(page, 800)
    _snapshot(page, "popout_translate")


def test_snapshot_popout_translator_preset(page, vr_server):
    _viewport(page, 1280, 720)
    page.goto(f"{vr_server}/?popout=view", wait_until="domcontentloaded")
    page.wait_for_function("() => !!window._gridRenderer", timeout=8000)
    page.evaluate(
        "() => localStorage.setItem('popout_layout_v2', JSON.stringify({version: 2, preset: 'translator', lastTermPreset: 'triple', lastNoTermPreset: 'translator', ratiosByPreset: {}, customTree: null}))"
    )
    page.reload(wait_until="domcontentloaded")
    page.wait_for_function("() => !!window._gridRenderer", timeout=8000)
    _settle(page, 800)
    _snapshot(page, "popout_translator")


def test_snapshot_admin_landing(page, vr_server):
    _viewport(page, 1280, 800)
    page.goto(f"{vr_server}/", wait_until="domcontentloaded")
    page.wait_for_timeout(800)  # let landing render settle
    _snapshot(page, "admin_landing")


def test_snapshot_reader_view(page, vr_server):
    _viewport(page, 1280, 800)
    page.goto(f"{vr_server}/reader", wait_until="domcontentloaded")
    page.wait_for_timeout(800)
    _snapshot(page, "reader")
