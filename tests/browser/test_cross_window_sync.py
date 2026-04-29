"""Cross-window transcript sync (admin ↔ popout) — Playwright tests.

The bug class this catches:

  Brad opens the live meeting in the admin window AND the pop-out window
  via `?popout=view&test=1`. Admin shows the full transcript; popout shows
  fragments / nothing / wrong content. Server is broadcasting the same
  events to both — the divergence is in client-side WS handling and
  SegmentStore state.

  Most recently bit: the popout's view-WS handler had a catch-all `else`
  that funneled every non-segment control message (`speaker_pulse`,
  `seat_update`, etc.) into `store.ingest()`, which then fired listeners
  with `segment_id=undefined`, which CompactGridRenderer interpreted as
  "store cleared" — wiping the popout grid every 200 ms during a meeting.

  Admin doesn't have this problem because its audio-WS handler enumerates
  every control type explicitly. The bug lives in the seam, not the data.

What we test:

  1. Both contexts connect to the same /api/ws/view + show same segments.
  2. `speaker_pulse` ticks from the test harness DO NOT clear the popout
     grid — its child count must monotonically increase when transcript
     events arrive between pulses.
  3. After a popout WS disconnect+reconnect, the popout catches up to
     the same set of segment_ids the admin has, with no duplicates.
  4. Cross-language pair: same routing across (en, ja), (en, de),
     (en, fr) — guards against the same-script-router class.

Why a custom harness, not the real meeting_scribe app: the real lifespan
boots vLLM backends. CI doesn't have GPUs. We mount the *real* `view_broadcast`
router (so the WS shape, journal replay, and ws_connections registry are
exercised) and stub the rest of the surface the popout init touches.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

pytestmark = pytest.mark.browser

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = REPO_ROOT / "static"


# ── Test harness: real view_broadcast router + stubbed surface ──────────


class _MeetingState:
    """Mirror the subset of `meeting_scribe.runtime.state` the popout reads."""

    def __init__(self) -> None:
        self.ws_connections: set[WebSocket] = set()
        self.meeting_id = "test-meeting-0"
        self.language_pair: list[str] = ["en", "ja"]
        self.events: list[dict] = []  # journal


def _build_app(harness: _MeetingState) -> FastAPI:
    app = FastAPI()

    # ── Static + page routes ──────────────────────────────────────────
    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ── Stubbed read surface the popout init fetches ─────────────────
    @app.get("/api/languages")
    async def languages():
        # Minimal subset — popout indexes by code, falls back to UPPERCASE
        # for missing entries.
        return JSONResponse(
            {
                "languages": [
                    {"code": "en", "name": "English", "native_name": "English"},
                    {"code": "ja", "name": "Japanese", "native_name": "日本語"},
                    {"code": "de", "name": "German", "native_name": "Deutsch"},
                    {"code": "fr", "name": "French", "native_name": "Français"},
                ]
            }
        )

    @app.get("/api/status")
    async def status():
        return JSONResponse(
            {
                "meeting": {
                    "id": harness.meeting_id,
                    "language_pair": harness.language_pair,
                },
                "backends": {"asr": True, "translate": True, "diarize": False},
                "connections": len(harness.ws_connections),
            }
        )

    @app.get("/api/meetings/{mid}")
    async def meeting(mid: str):
        if mid != harness.meeting_id:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(
            {
                "meta": {
                    "meeting_id": harness.meeting_id,
                    "language_pair": harness.language_pair,
                    "state": "RECORDING",
                },
                "events": list(harness.events),
            }
        )

    @app.get("/api/meetings")
    async def meetings():
        return JSONResponse([])

    @app.get("/api/meeting/wifi")
    async def wifi():
        return JSONResponse({"available": False})

    @app.post("/api/diag/listener")
    async def diag_listener():
        return JSONResponse({"ok": True})

    # ── /api/ws/view — the route the popout connects to ──────────────
    # Reproduces the relevant behavior of meeting_scribe.ws.view_broadcast
    # without depending on the runtime.state singleton.
    @app.websocket("/api/ws/view")
    async def ws_view(websocket: WebSocket) -> None:
        await websocket.accept()
        harness.ws_connections.add(websocket)
        try:
            # Replay journal so late-joining clients catch up — same
            # contract as the real view_broadcast handler.
            for ev in list(harness.events):
                await websocket.send_text(json.dumps(ev))
            while True:
                # The popout pings periodically; we just drain.
                await websocket.receive_text()
        except Exception:
            pass
        finally:
            harness.ws_connections.discard(websocket)

    # ── /api/ws/audio — admin's audio WS, stubbed (no real ASR) ──────
    # Admin connects here when starting a meeting. We don't process
    # audio bytes — we just register the connection so it receives
    # broadcasts, then echo control messages.
    @app.websocket("/api/ws")
    async def ws_audio(websocket: WebSocket) -> None:
        await websocket.accept()
        harness.ws_connections.add(websocket)
        try:
            for ev in list(harness.events):
                await websocket.send_text(json.dumps(ev))
            while True:
                msg = await websocket.receive()
                # Drain text and bytes; the test never plays back.
                if "text" not in msg and "bytes" not in msg:
                    break
        except Exception:
            pass
        finally:
            harness.ws_connections.discard(websocket)

    # ── Test-only: force-disconnect all WS clients ───────────────────
    @app.post("/test/disconnect_all")
    async def disconnect_all():
        # Close every connected WS so the popout's auto-reconnect kicks in.
        # Done from inside the running event loop — `await ws.close()` is
        # the supported path.
        closed = 0
        for ws in list(harness.ws_connections):
            try:
                await ws.close(code=1001, reason="test-disconnect")
                closed += 1
            except Exception:
                pass
            harness.ws_connections.discard(ws)
        return JSONResponse({"closed": closed})

    # ── Test injection endpoint ──────────────────────────────────────
    # POST a JSON payload here and the harness broadcasts it verbatim
    # to every connected WS — the same code path the real server uses
    # via `_broadcast` / `_broadcast_json`.
    @app.post("/test/broadcast")
    async def broadcast(request_payload: dict):
        text = json.dumps(request_payload)
        # If it's a transcript event (has segment_id), persist to journal
        # so reconnect-replay catches up.
        if request_payload.get("segment_id"):
            harness.events.append(request_payload)
        dead: list[WebSocket] = []
        sent = 0
        for ws in list(harness.ws_connections):
            try:
                await ws.send_text(text)
                sent += 1
            except Exception as e:
                print(f"[harness] send failed: {e}", flush=True)
                dead.append(ws)
        for ws in dead:
            harness.ws_connections.discard(ws)
        return JSONResponse({"delivered": sent, "dead": len(dead)})

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
        config = uvicorn.Config(
            self.app, host="127.0.0.1", port=self.port, log_level="error"
        )
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
                urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/api/status", timeout=1
                )
                return
            except Exception:
                time.sleep(0.05)
        raise RuntimeError(f"live_meeting_server did not start on port {self.port}")

    def stop(self) -> None:
        if hasattr(self, "_server"):
            self._server.should_exit = True


@pytest.fixture
def live_meeting_server() -> Generator[dict[str, Any]]:
    """Start a live FastAPI app with stubbed backends + real WS routing.

    The popout and admin pages connect to the *same* server instance,
    receive broadcasts via the same `ws_connections` set, and exercise
    the same client-side scribe-app.js code path that runs in production.
    """
    harness = _MeetingState()
    app = _build_app(harness)
    thread = _ServerThread(app)
    thread.start()
    thread.wait_ready()
    try:
        yield {
            "base_url": f"http://127.0.0.1:{thread.port}",
            "harness": harness,
        }
    finally:
        thread.stop()
        thread.join(timeout=3.0)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_segment(
    *,
    segment_id: str,
    text: str,
    start_ms: int,
    language: str = "en",
    target_language: str = "ja",
    translation_text: str | None = None,
    is_final: bool = True,
    revision: int = 0,
) -> dict:
    """Build a TranscriptEvent payload matching the production wire shape."""
    payload = {
        "segment_id": segment_id,
        "revision": revision,
        "is_final": is_final,
        "start_ms": start_ms,
        "end_ms": start_ms + 1500,
        "language": language,
        "text": text,
        "speakers": [{"cluster_id": 1, "source": "diarize"}],
    }
    if translation_text is not None:
        payload["translation"] = {
            "status": "done",
            "text": translation_text,
            "target_language": target_language,
        }
    return payload


def _broadcast(server, payload: dict, *, allow_zero: bool = False) -> None:
    """POST to /test/broadcast and synchronously deliver to all WS clients.

    If `allow_zero=False` (default) and the server reports zero
    recipients, raises AssertionError — indicates a setup race. The
    reconnect test uses `allow_zero=True` for broadcasts during the
    intentional disconnect window where we EXPECT zero recipients
    (events should land in the journal for replay).
    """
    import urllib.error
    import urllib.request

    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{server['base_url']}/test/broadcast",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            if data.get("delivered", 0) == 0 and not allow_zero:
                raise AssertionError(
                    f"broadcast had zero recipients (no WS connected). "
                    f"Connection setup likely raced ahead of the broadcast. "
                    f"Server response: {data}"
                )
    except urllib.error.URLError as e:  # pragma: no cover — server crashed
        raise AssertionError(f"broadcast failed: {e}") from e


def _capture_console(page) -> list[str]:
    """Attach listeners that record console output + WS frames into a list.

    Returns the list. Caller can inspect it after a failure for context.
    """
    log: list[str] = []

    def _on_console(msg) -> None:
        try:
            log.append(f"[console.{msg.type}] {msg.text}")
        except Exception as e:  # pragma: no cover
            log.append(f"[console-listener-err] {e}")

    def _on_pageerror(exc) -> None:
        log.append(f"[pageerror] {exc}")

    def _on_ws(ws) -> None:
        log.append(f"[ws.open] {ws.url}")

        def _frame_recv(payload) -> None:
            # Playwright passes the payload string/bytes directly to the
            # framereceived/framesent listeners — there's no `.payload`
            # attribute on the argument.
            try:
                if isinstance(payload, bytes):
                    payload = payload.decode("utf-8", errors="replace")
                log.append(f"[ws<] {str(payload)[:240]}")
            except Exception as e:  # pragma: no cover
                log.append(f"[ws<-listener-err] {e}")

        def _frame_sent(payload) -> None:
            try:
                if isinstance(payload, bytes):
                    payload = payload.decode("utf-8", errors="replace")
                log.append(f"[ws>] {str(payload)[:240]}")
            except Exception as e:  # pragma: no cover
                log.append(f"[ws>-listener-err] {e}")

        ws.on("framereceived", _frame_recv)
        ws.on("framesent", _frame_sent)
        ws.on("close", lambda: log.append(f"[ws.close] {ws.url}"))
        ws.on(
            "socketerror",
            lambda err: log.append(f"[ws.error] {ws.url} {err}"),
        )

    def _on_response(resp) -> None:
        if resp.status >= 400:
            log.append(f"[http {resp.status}] {resp.url}")

    page.on("console", _on_console)
    page.on("pageerror", _on_pageerror)
    page.on("websocket", _on_ws)
    page.on("response", _on_response)
    return log


def _wait_for_popout_ws_open(page, timeout_ms: int = 8000) -> None:
    """Wait until the popout's view-WS connection-state pill reads 'open'.

    The popout init does several async fetches (status, meeting, languages)
    before opening the WS; broadcasting before this returns drops events
    on the floor.
    """
    page.wait_for_function(
        "() => document.querySelector('.popout-dot')?.dataset?.connState === 'open'",
        timeout=timeout_ms,
    )


def _wait_until(page, expr_js: str, timeout_ms: int = 5000) -> None:
    try:
        page.wait_for_function(expr_js, timeout=timeout_ms)
    except Exception as e:
        # Surface page state on timeout so failures are debuggable instead
        # of "Timeout 5000ms exceeded" with zero context.
        try:
            grid = page.evaluate(
                "() => ({"
                "  popout_mode: new URLSearchParams(location.search).get('popout'),"
                "  has_grid_el: !!document.getElementById('transcript-grid'),"
                "  has_renderer: !!window._gridRenderer,"
                "  renderer_segments: window._gridRenderer?._segmentMap?.size ?? null,"
                "  renderer_current_block: window._gridRenderer?._currentBlock ? 'set' : 'null',"
                "  block_count: document.querySelectorAll('#transcript-grid .compact-block').length,"
                "  body_classes: document.body.className,"
                "  current_lang_pair: window.currentLanguagePair ?? null,"
                "  test_ingest_count: window.__test_ingest_count ?? null,"
                "  test_store_count: window.__test_store?.count ?? null,"
                "  test_msg_log: window.__test_msg_log ? window.__test_msg_log.slice(-5) : null"
                "})"
            )
        except Exception:
            grid = {"error": "page.evaluate failed"}
        raise AssertionError(
            f"wait_for_function timed out: {expr_js!r}\n"
            f"page state: {grid}\n"
            f"original error: {e}"
        ) from e


# ── Tests ────────────────────────────────────────────────────────────────


def test_two_popouts_render_same_segments(browser, live_meeting_server):
    """Two independent popout viewers ingest the same broadcast events
    and end up with the same set of segment_ids.

    Why two popouts and not admin+popout: the admin page's audio-WS only
    connects after the user starts a meeting (room-setup flow), which
    requires real mic + the room-setup UI machinery. The bug class we're
    catching (popout-WS handler asymmetry) lives entirely in the popout
    code path, so two popout windows is the cleanest observation. Full
    admin↔popout coverage requires the start-meeting fixture and lives
    in the nightly lane.
    """
    server = live_meeting_server
    base = server["base_url"]

    ctx_a = browser.new_context()
    ctx_b = browser.new_context()
    page_a = ctx_a.new_page()
    page_b = ctx_b.new_page()

    try:
        page_a.goto(f"{base}/?popout=view&test=1", wait_until="domcontentloaded")
        page_b.goto(f"{base}/?popout=view&test=1", wait_until="domcontentloaded")
        _wait_until(page_a, "() => !!window._gridRenderer")
        _wait_until(page_b, "() => !!window._gridRenderer")
        _wait_for_popout_ws_open(page_a)
        _wait_for_popout_ws_open(page_b)

        for i, text in enumerate(["Hello world.", "Test one two.", "Final segment."]):
            seg = _make_segment(
                segment_id=f"seg-{i}",
                text=text,
                start_ms=i * 2000,
                translation_text=f"訳{i}",
            )
            # Different speakers per segment so block-merging doesn't
            # collapse them into a single block; we want to observe each
            # segment in the rendered grid independently.
            seg["speakers"] = [{"cluster_id": i + 1, "source": "diarize"}]
            _broadcast(server, seg)

        _wait_until(page_a, "() => window._gridRenderer?._segmentMap?.size >= 3")
        _wait_until(page_b, "() => window._gridRenderer?._segmentMap?.size >= 3")

        ids_a = page_a.evaluate(
            "() => Array.from(window._gridRenderer._segmentMap.keys()).sort()"
        )
        ids_b = page_b.evaluate(
            "() => Array.from(window._gridRenderer._segmentMap.keys()).sort()"
        )

        assert ids_a == ids_b, (
            f"two popouts diverged on the same broadcast:\n"
            f"  page_a = {ids_a}\n"
            f"  page_b = {ids_b}"
        )
        assert len(ids_a) == 3, f"expected 3 segments, got {ids_a}"
    finally:
        ctx_a.close()
        ctx_b.close()


def test_speaker_pulse_does_not_clear_popout_grid(browser, live_meeting_server):
    """Regression for the popout-clear-on-pulse bug.

    speaker_pulse fires every 200 ms during a meeting. If the popout's
    catch-all WS branch funnels it through `store.ingest()` with no
    segment_id, CompactGridRenderer interprets the falsy id as "store
    cleared" and wipes the grid — so the popout only ever shows the
    sliver of utterances received between pulses.

    The fix lives in segment-store.js (early-return when no segment_id).
    This test fails BEFORE the fix and passes after.
    """
    server = live_meeting_server
    base = server["base_url"]

    popout_ctx = browser.new_context()
    popout_page = popout_ctx.new_page()

    # Surface page console + WS frames so test failures are debuggable.
    log = _capture_console(popout_page)

    try:
        popout_page.goto(f"{base}/?popout=view&test=1", wait_until="domcontentloaded")
        _wait_until(popout_page, "() => !!window._gridRenderer")
        _wait_for_popout_ws_open(popout_page)

        # Land 2 transcript segments — different speakers so they don't
        # merge into a single block (CompactGridRenderer collapses same-
        # speaker turns; we want to observe per-segment state survival,
        # not block-merge behavior).
        _broadcast(
            server,
            _make_segment(
                segment_id="seg-A",
                text="Anchor segment one.",
                start_ms=0,
                translation_text="アンカー1",
            ),
        )
        seg_b = _make_segment(
            segment_id="seg-B",
            text="Anchor segment two.",
            start_ms=2000,
            translation_text="アンカー2",
        )
        seg_b["speakers"] = [{"cluster_id": 2, "source": "diarize"}]
        _broadcast(server, seg_b)

        try:
            _wait_until(
                popout_page,
                "() => window._gridRenderer?._segmentMap?.size >= 2",
            )
        except AssertionError as e:
            raise AssertionError(
                f"{e}\n\n--- popout console + WS frames (last 60) ---\n"
                + "\n".join(log[-60:])
            ) from e

        before = popout_page.evaluate(
            "() => window._gridRenderer._segmentMap.size"
        )
        assert before == 2, f"expected 2 segments before pulses, got {before}"

        # Fire 10 speaker_pulse events in rapid succession — same cadence
        # the real server emits during a meeting.
        for i in range(10):
            _broadcast(
                server,
                {
                    "type": "speaker_pulse",
                    "active_speakers": [{"cluster_id": 1, "confidence": 0.9}],
                    "timestamp_ms": 1000 + i * 200,
                },
            )

        # Give the popout a beat to process them.
        popout_page.wait_for_timeout(200)

        after = popout_page.evaluate(
            "() => window._gridRenderer._segmentMap.size"
        )
        text_in_grid = popout_page.evaluate(
            "() => document.getElementById('transcript-grid').textContent"
        )
        assert after == before, (
            f"speaker_pulse shrank the popout's _segmentMap: was {before}, "
            f"now {after}. The popout's WS handler is funneling control "
            f"events through store.ingest() with segment_id=undefined "
            f"(or some other listener-fanout path is wiping state)."
        )
        assert "Anchor segment one." in text_in_grid, (
            f"popout grid lost segment text after speaker_pulse storm. "
            f"grid text: {text_in_grid[:200]!r}"
        )
    finally:
        popout_ctx.close()


def test_popout_reconnect_replays_without_duplicates(browser, live_meeting_server):
    """Closing and reopening the popout's view-WS mid-meeting must
    rehydrate the transcript with the same segment_ids the admin sees,
    with no duplicates."""
    server = live_meeting_server
    base = server["base_url"]
    harness: _MeetingState = server["harness"]

    popout_ctx = browser.new_context()
    popout_page = popout_ctx.new_page()

    try:
        popout_page.goto(f"{base}/?popout=view&test=1", wait_until="domcontentloaded")
        _wait_until(popout_page, "() => !!window._gridRenderer")
        _wait_for_popout_ws_open(popout_page)

        # Land 2 segments — different speakers so blocks don't merge.
        for i in range(2):
            seg = _make_segment(
                segment_id=f"seg-pre-{i}",
                text=f"Pre-disconnect segment {i}.",
                start_ms=i * 2000,
                translation_text=f"切断前{i}",
            )
            seg["speakers"] = [{"cluster_id": i + 1, "source": "diarize"}]
            _broadcast(server, seg)

        _wait_until(
            popout_page,
            "() => window._gridRenderer?._segmentMap?.size >= 2",
        )

        # Force-close the popout's view-WS by hitting the test endpoint;
        # the popout's auto-reconnect kicks in.
        import urllib.request

        urllib.request.urlopen(
            urllib.request.Request(
                f"{base}/test/disconnect_all", method="POST"
            ),
            timeout=2,
        ).read()

        # During the disconnect, broadcast 2 more segments. They land in
        # the harness journal but the popout is offline. allow_zero=True
        # because the popout is intentionally disconnected here.
        for i in range(2, 4):
            seg = _make_segment(
                segment_id=f"seg-pre-{i}",
                text=f"During-disconnect segment {i}.",
                start_ms=i * 2000,
                translation_text=f"切断中{i}",
            )
            seg["speakers"] = [{"cluster_id": i + 1, "source": "diarize"}]
            _broadcast(server, seg, allow_zero=True)

        # Wait for the popout to reconnect (auto-reconnect backoff is
        # ~1 s on first retry) and replay the journal.
        _wait_until(
            popout_page,
            "() => window._gridRenderer?._segmentMap?.size >= 4",
            timeout_ms=15000,
        )

        ids = popout_page.evaluate(
            "() => Array.from(window._gridRenderer._segmentMap.keys())"
        )
        # No duplicates after replay (Map keys are inherently unique;
        # we still assert against the expected set so a partial replay
        # surfaces clearly).
        assert sorted(ids) == [f"seg-pre-{i}" for i in range(4)], (
            f"expected seg-pre-0..3 after reconnect, got {sorted(ids)}"
        )
    finally:
        popout_ctx.close()


@pytest.mark.parametrize(
    ("source_lang", "target_lang", "src_text", "tgt_text"),
    [
        ("en", "ja", "Hello world.", "こんにちは世界。"),
        ("en", "de", "Hello world.", "Hallo Welt."),
        ("en", "fr", "Hello world.", "Bonjour le monde."),
    ],
)
def test_cross_language_pair_renders_in_correct_columns(
    browser, live_meeting_server, source_lang, target_lang, src_text, tgt_text
):
    """The script-router previously force-relabeled all Latin text to 'en'
    even for same-script pairs (en↔de, en↔fr). That made German source
    text leak into the English column for de→en meetings.

    This test parametrizes the language pair and asserts both source and
    translation land in the correct columns. Regression coverage for the
    `fix(pipeline): don't run script-router on same-script language pairs`
    commit class.
    """
    server = live_meeting_server
    server["harness"].language_pair = [source_lang, target_lang]

    popout_ctx = browser.new_context()
    popout_page = popout_ctx.new_page()

    try:
        popout_page.goto(
            f"{server['base_url']}/?popout=view&test=1", wait_until="domcontentloaded"
        )
        _wait_until(popout_page, "() => !!window._gridRenderer")
        _wait_for_popout_ws_open(popout_page)

        _broadcast(
            server,
            _make_segment(
                segment_id="seg-pair",
                text=src_text,
                start_ms=0,
                language=source_lang,
                target_language=target_lang,
                translation_text=tgt_text,
            ),
        )

        _wait_until(
            popout_page,
            "() => document.querySelectorAll('#transcript-grid .compact-block').length >= 1",
        )

        # _routeLangByScript should put source in column-A, translation
        # in column-B (since the harness sets language_pair to [src, tgt]
        # so langA=src, langB=tgt).
        cols = popout_page.evaluate(
            """() => {
                const block = document.querySelector('#transcript-grid .compact-block');
                if (!block) return null;
                return {
                    a: block.querySelector('.compact-col-a')?.textContent || '',
                    b: block.querySelector('.compact-col-b')?.textContent || '',
                };
            }"""
        )
        assert cols is not None
        assert src_text.split(".")[0] in cols["a"], (
            f"source text not in column A for {source_lang}↔{target_lang}: "
            f"colA={cols['a']!r}"
        )
        assert tgt_text.split(".")[0].split("。")[0] in cols["b"], (
            f"translation not in column B for {source_lang}↔{target_lang}: "
            f"colB={cols['b']!r}"
        )
    finally:
        popout_ctx.close()
