"""Shared fixtures for Playwright browser tests.

Provides:
  - Chromium launch args with autoplay policy disabled (deterministic audio)
  - generate_test_fmp4() — pre-encoded fMP4 init + fragments from a sine wave
  - synthetic_mse_server — lightweight FastAPI + WS serving pre-encoded fMP4
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from meeting_scribe.backends.mse_encoder import SAMPLE_RATE_OUT, Fmp4AacEncoder

STATIC_DIR = Path(__file__).resolve().parents[2] / "static"


# ── Tmux socket isolation ────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_tmux_socket(monkeypatch) -> Generator[str]:
    """Pin every browser test to a per-run tmux socket.

    Without this, tests that spawn real tmux (autosre suite, anything
    that leaves SCRIBE_TERM_SHELL unset) attach to the user's live
    ``-L scribe`` socket. Commands typed in a test would then echo into
    the user's real terminal window — exactly what the user reported.
    """
    import os as _os
    import shutil as _shutil
    import subprocess as _subp

    sock = f"scribe-test-{_os.getpid()}"
    monkeypatch.setenv("SCRIBE_TMUX_SOCKET", sock)
    yield sock
    # Best-effort teardown via a plain subprocess — avoids spinning up a
    # second event loop from inside a fixture that pytest-asyncio already
    # manages. The socket is disposable by design, so "no server" is a
    # success state.
    if _shutil.which("tmux"):
        try:
            _subp.run(
                ["tmux", "-L", sock, "kill-server"],
                capture_output=True,
                timeout=3,
                check=False,
            )
        except Exception:
            pass


# ── Chromium launch configuration ────────────────────────────────────


@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args):
    """Override pytest-playwright defaults for deterministic audio playback."""
    return {
        **browser_type_launch_args,
        "args": [
            "--autoplay-policy=no-user-gesture-required",
            "--use-fake-ui-for-media-stream",
        ],
    }


# ── Test audio generation ────────────────────────────────────────────


def generate_test_fmp4(duration_s: float = 2.0, freq: float = 440.0) -> tuple[bytes, list[bytes]]:
    """Generate (init_bytes, [fragment_bytes, ...]) using Fmp4AacEncoder.

    Feeds a sine wave in 50ms chunks to match realistic TTS delivery cadence.
    """
    enc = Fmp4AacEncoder()
    try:
        init = enc.init_segment()
        fragments: list[bytes] = []
        chunk_s = 0.050
        n_chunks = int(duration_s / chunk_s)
        for _ in range(n_chunks):
            n = int(chunk_s * SAMPLE_RATE_OUT)
            t = np.arange(n, dtype=np.float32) / SAMPLE_RATE_OUT
            pcm = (0.8 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            frag = enc.encode(pcm, SAMPLE_RATE_OUT)
            if frag:
                fragments.append(frag)
        tail = enc.flush()
        if tail:
            fragments.append(tail)
    finally:
        enc.close()
    return init, fragments


# ── Synthetic MSE test server ────────────────────────────────────────


def _build_synthetic_app(
    init: bytes,
    fragments: list[bytes],
    *,
    corrupt_init: bool = False,
) -> FastAPI:
    """Build a minimal FastAPI app that serves guest.html + MSE WS."""
    app = FastAPI()

    # Serve static files (guest.html, js/, etc.)
    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "guest.html")

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.websocket("/api/ws/audio-out")
    async def ws_audio_out(websocket: WebSocket):
        await websocket.accept()
        # Wait for set_format
        while True:
            text = await websocket.receive_text()
            try:
                msg = json.loads(text)
                if msg.get("type") == "set_format":
                    fmt = msg.get("format", "")
                    await websocket.send_text(json.dumps({"type": "format_ack", "format": fmt}))
                    break
            except Exception:
                pass

        # Send init frame
        init_payload = b"\x00" * 100 if corrupt_init else init
        await websocket.send_bytes(b"\x49" + init_payload)

        # Send fragments with small delays
        for frag in fragments:
            await asyncio.sleep(0.05)
            try:
                await websocket.send_bytes(b"\x46" + frag)
            except Exception:
                break

        # Keep connection alive
        try:
            while True:
                await websocket.receive_text()
        except Exception:
            pass

    # Stub endpoints that guest.html may probe
    @app.post("/api/diag/listener")
    async def diag_listener():
        return {"ok": True}

    @app.get("/api/diag/listeners")
    async def diag_listeners():
        return []

    return app


class _ServerThread(threading.Thread):
    """Run uvicorn in a daemon thread with a free-port server."""

    def __init__(self, app: FastAPI) -> None:
        super().__init__(daemon=True)
        self.app = app
        self.port: int = 0
        self._started = threading.Event()

    def run(self) -> None:
        import socket

        # Find a free port
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]

        config = uvicorn.Config(
            self.app,
            host="127.0.0.1",
            port=self.port,
            log_level="error",
        )
        self._server = uvicorn.Server(config)
        self._started.set()
        self._server.run()

    def wait_ready(self, timeout: float = 10.0) -> None:
        self._started.wait(timeout)
        # Wait for the server to actually be serving
        import time
        import urllib.request

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=1)
                return
            except Exception:
                time.sleep(0.1)
        raise RuntimeError(f"synthetic server did not start on port {self.port}")

    def stop(self) -> None:
        if hasattr(self, "_server"):
            self._server.should_exit = True


@pytest.fixture(scope="session")
def synthetic_mse_server() -> Generator[int]:
    """Start a synthetic MSE server and yield its port."""
    init, fragments = generate_test_fmp4(duration_s=3.0)
    app = _build_synthetic_app(init, fragments)
    thread = _ServerThread(app)
    thread.start()
    thread.wait_ready()
    yield thread.port
    thread.stop()


@pytest.fixture(scope="session")
def corrupt_init_server() -> Generator[int]:
    """Start a server that sends a corrupt init segment."""
    init, fragments = generate_test_fmp4(duration_s=1.0)
    app = _build_synthetic_app(init, fragments, corrupt_init=True)
    thread = _ServerThread(app)
    thread.start()
    thread.wait_ready()
    yield thread.port
    thread.stop()
