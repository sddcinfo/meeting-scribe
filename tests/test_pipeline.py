"""End-to-end integration tests for meeting-scribe.

Runs against an ISOLATED test server (port 9080) with its own meetings directory.
Production server on port 8080 is never touched.

Tests are split into tiers:
- Fast API tests: HTTP requests only, no audio, ~1s each
- ASR pipeline tests: send real audio, wait for transcription, ~10s each

Run:
    pytest -m integration              # All integration tests
    pytest -m "integration and not slow"  # Fast API tests only
    pytest                             # Unit tests only (default)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import ssl
import time

import httpx
import pytest
import websockets

# All tests in this module are integration tests
pytestmark = pytest.mark.integration

# SSL context for self-signed certs
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE

# Set by fixture
BASE = ""
WS = ""


@pytest.fixture(autouse=True, scope="module")
def _use_test_server(test_server):
    """Wire all tests to the isolated test server."""
    global BASE, WS
    BASE = test_server["base_url"]
    WS = test_server["ws_url"]
    yield
    with contextlib.suppress(Exception):
        httpx.post(f"{BASE}/api/meeting/stop", verify=False, timeout=5)


def _ws_ssl():
    return _ssl_ctx if WS.startswith("wss") else None


# ═══════════════════════════════════════════════════════════════
# FAST API TESTS — no audio, sub-second each
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_server_status():
    async with httpx.AsyncClient(timeout=5, verify=False) as c:
        r = await c.get(f"{BASE}/api/status")
        assert r.status_code == 200
        data = r.json()
        assert data["backends"]["asr"] is True
        assert data["backends"]["translate"] is True
        assert data["backends"]["diarize"] is True
        assert data["backends"]["tts"] is True


@pytest.mark.asyncio
async def test_page_loads():
    async with httpx.AsyncClient(timeout=5, verify=False) as c:
        r = await c.get(f"{BASE}/")
        assert r.status_code == 200
        assert "Meeting Scribe" in r.text


@pytest.mark.asyncio
async def test_languages_api():
    async with httpx.AsyncClient(timeout=5, verify=False) as c:
        r = await c.get(f"{BASE}/api/languages")
        assert r.status_code == 200
        data = r.json()
        assert len(data["languages"]) >= 10
        codes = [l["code"] for l in data["languages"]]
        assert "en" in codes
        assert "ja" in codes
        assert "zh" in codes


@pytest.mark.asyncio
async def test_room_layout_crud():
    async with httpx.AsyncClient(timeout=5, verify=False) as c:
        layout = {
            "preset": "rectangle",
            "tables": [
                {
                    "table_id": "t1",
                    "x": 50,
                    "y": 50,
                    "width": 40,
                    "height": 20,
                    "border_radius": 5,
                    "label": "",
                }
            ],
            "seats": [
                {"seat_id": "s1", "x": 30, "y": 50, "speaker_name": "Alice"},
                {"seat_id": "s2", "x": 70, "y": 50, "speaker_name": "Bob"},
            ],
        }
        r = await c.put(f"{BASE}/api/room/layout", json=layout)
        assert r.status_code == 200

        r = await c.get(f"{BASE}/api/room/layout")
        assert r.status_code == 200
        got = r.json()
        assert got["preset"] == "rectangle"
        assert len(got["seats"]) == 2


@pytest.mark.asyncio
async def test_meeting_lifecycle():
    """Start, check status, stop, verify complete."""
    async with httpx.AsyncClient(timeout=60, verify=False) as c:
        # Start
        r = await c.post(f"{BASE}/api/meeting/start")
        assert r.status_code == 200
        mid = r.json()["meeting_id"]

        # Status shows recording
        r = await c.get(f"{BASE}/api/status")
        assert r.json()["meeting"]["state"] == "recording"

        # Stop
        r = await c.post(f"{BASE}/api/meeting/stop")
        assert r.status_code == 200

        # Shows in history
        r = await c.get(f"{BASE}/api/meetings")
        ids = [m["meeting_id"] for m in r.json()["meetings"]]
        assert mid in ids

        # Delete
        r = await c.delete(f"{BASE}/api/meetings/{mid}")
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_room_layout_persisted_with_meeting():
    async with httpx.AsyncClient(timeout=60, verify=False) as c:
        await c.put(
            f"{BASE}/api/room/layout",
            json={
                "preset": "round",
                "tables": [
                    {
                        "table_id": "t1",
                        "x": 50,
                        "y": 50,
                        "width": 30,
                        "height": 30,
                        "border_radius": 50,
                        "label": "",
                    }
                ],
                "seats": [{"seat_id": "s1", "x": 50, "y": 20}],
            },
        )
        r = await c.post(f"{BASE}/api/meeting/start")
        mid = r.json()["meeting_id"]
        await c.post(f"{BASE}/api/meeting/stop")

        r = await c.get(f"{BASE}/api/meetings/{mid}")
        assert r.json()["room"]["preset"] == "round"

        await c.delete(f"{BASE}/api/meetings/{mid}")


@pytest.mark.asyncio
async def test_websocket_view_connects():
    """View-only WebSocket connects successfully."""
    ws_view = WS.replace("/api/ws", "/api/ws/view")
    async with websockets.connect(ws_view, ssl=_ws_ssl()) as ws:
        # Connection established — send a language preference to verify it's live
        await ws.send(json.dumps({"type": "set_language", "language": "en"}))
        # No error = success


# ═══════════════════════════════════════════════════════════════
# ASR PIPELINE TESTS — send real audio, verify transcription
# ═══════════════════════════════════════════════════════════════


async def _send_audio_chunks(ws, audio_bytes: bytes, chunk_size: int = 8000):
    """Send audio in chunks without unnecessary delays."""
    for i in range(0, len(audio_bytes), chunk_size):
        await ws.send(audio_bytes[i : i + chunk_size])
        await asyncio.sleep(0.05)  # Minimal delay to avoid flooding


async def _collect_finals(ws, timeout: float = 10.0) -> list[dict]:
    """Collect final transcript events until timeout."""
    events = []
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            msg = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 2.0))
            e = json.loads(msg)
            if e.get("is_final") and e.get("text"):
                events.append(e)
    except TimeoutError, websockets.exceptions.ConnectionClosedOK:
        pass
    return events


@pytest.mark.slow
@pytest.mark.asyncio
async def test_asr_english(audio_en_bytes):
    """English audio → ASR produces text events."""
    async with httpx.AsyncClient(timeout=60, verify=False) as c:
        r = await c.post(f"{BASE}/api/meeting/start")
        mid = r.json()["meeting_id"]

    try:
        async with websockets.connect(WS, ssl=_ws_ssl()) as ws:
            await _send_audio_chunks(ws, audio_en_bytes)
            events = await _collect_finals(ws, timeout=10)

        assert len(events) > 0, "No final transcript events received"
        all_text = " ".join(e["text"] for e in events)
        assert len(all_text) > 5, f"Text too short: {all_text}"
    finally:
        async with httpx.AsyncClient(timeout=60, verify=False) as c:
            await c.post(f"{BASE}/api/meeting/stop")
            await c.delete(f"{BASE}/api/meetings/{mid}")


@pytest.mark.slow
@pytest.mark.asyncio
async def test_asr_with_translation(audio_en_bytes):
    """Audio → ASR → translation pipeline produces translated text."""
    async with httpx.AsyncClient(timeout=60, verify=False) as c:
        r = await c.post(f"{BASE}/api/meeting/start")
        mid = r.json()["meeting_id"]

    try:
        translated = []
        async with websockets.connect(WS, ssl=_ws_ssl()) as ws:
            await _send_audio_chunks(ws, audio_en_bytes)
            deadline = time.monotonic() + 20
            try:
                while time.monotonic() < deadline:
                    msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    e = json.loads(msg)
                    tr = e.get("translation") or {}
                    if tr.get("text") and tr.get("status") == "done":
                        translated.append(e)
                        break  # Got one translation, that's enough
            except TimeoutError, websockets.exceptions.ConnectionClosedOK:
                pass

        assert len(translated) > 0, "No translation received"
        assert translated[0]["translation"]["target_language"] != ""
    finally:
        async with httpx.AsyncClient(timeout=60, verify=False) as c:
            await c.post(f"{BASE}/api/meeting/stop")
            await c.delete(f"{BASE}/api/meetings/{mid}")


@pytest.mark.slow
@pytest.mark.asyncio
async def test_audio_recording(audio_en_bytes):
    """Audio is saved to recording.pcm and playable after meeting ends."""
    async with httpx.AsyncClient(timeout=60, verify=False) as c:
        r = await c.post(f"{BASE}/api/meeting/start")
        mid = r.json()["meeting_id"]

    try:
        async with websockets.connect(WS, ssl=_ws_ssl()) as ws:
            await _send_audio_chunks(ws, audio_en_bytes)
            await asyncio.sleep(1)

        async with httpx.AsyncClient(timeout=60, verify=False) as c:
            await c.post(f"{BASE}/api/meeting/stop")
            r = await c.get(f"{BASE}/api/meetings/{mid}/audio?start_ms=0&end_ms=2000")
            assert r.status_code == 200
            assert len(r.content) > 44  # More than just WAV header
            assert r.content[:4] == b"RIFF"
    finally:
        async with httpx.AsyncClient(timeout=60, verify=False) as c:
            await c.delete(f"{BASE}/api/meetings/{mid}")
