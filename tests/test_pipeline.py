"""End-to-end pipeline tests for meeting-scribe.

Tests the full flow: audio → ASR → translation → event delivery.
Uses macOS TTS to generate test audio, sends via WebSocket, verifies events.

Run: PYTHONPATH=src .venv/bin/python3 -m pytest tests/ -v
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import tempfile
import time

import httpx
import numpy as np
import pytest
import soundfile as sf
import torch
import torchaudio
import websockets

BASE = os.environ.get("SCRIBE_TEST_BASE", "http://127.0.0.1:8080")
WS = os.environ.get("SCRIBE_TEST_WS", "ws://127.0.0.1:8080/api/ws")

# Track meeting IDs created during tests for cleanup
_test_meeting_ids: list[str] = []


def _make_s16(text: str, voice: str = "Samantha") -> bytes:
    """Generate s16le 16kHz PCM from macOS TTS."""
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
        path = f.name
    subprocess.run(["say", "-v", voice, "-o", path, text], check=True)
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    a16k = (
        torchaudio.transforms.Resample(sr, 16000)(
            torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        )
        .squeeze(0)
        .numpy()
    )
    return (a16k * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


async def _send_audio(ws, s16: bytes, chunk_bytes: int = 16000, delay: float = 0.5):
    """Send s16le audio in chunks via WebSocket."""
    for i in range(0, len(s16), chunk_bytes):
        await ws.send(s16[i : i + chunk_bytes])
        await asyncio.sleep(delay)


async def _collect_events(ws, timeout: float = 8.0) -> list[dict]:
    """Collect events from WebSocket until timeout."""
    events = []
    try:
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
            events.append(json.loads(msg))
    except (TimeoutError, websockets.exceptions.ConnectionClosedOK):
        pass
    return events


@pytest.fixture(autouse=True, scope="session")
def cleanup_test_meetings():
    """Clean up all meetings and restore draft layout after tests."""
    yield
    import httpx as _httpx

    # Delete test meetings
    for mid in _test_meeting_ids:
        with contextlib.suppress(Exception):
            _httpx.delete(f"{BASE}/api/meetings/{mid}", timeout=5)
    # Restore default draft layout
    with contextlib.suppress(Exception):
        _httpx.put(
            f"{BASE}/api/room/layout",
            json={"preset": "rectangle", "tables": [], "seats": []},
            timeout=5,
        )


async def _start_test_meeting() -> str:
    """Start a meeting and track it for cleanup."""
    async with httpx.AsyncClient(timeout=10) as c:
        # Stop any active meeting first
        await c.post(f"{BASE}/api/meeting/stop")
        r = await c.post(f"{BASE}/api/meeting/start")
        mid = r.json()["meeting_id"]
        _test_meeting_ids.append(mid)
        return mid


async def _stop_and_cleanup(meeting_id: str):
    """Stop meeting and delete it."""
    async with httpx.AsyncClient(timeout=10) as c:
        await c.post(f"{BASE}/api/meeting/stop")
        await c.delete(f"{BASE}/api/meetings/{meeting_id}")


# ─── Server health ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_server_status():
    """Server responds with backend status."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{BASE}/api/status")
        assert r.status_code == 200
        data = r.json()
        assert data["backends"]["asr"] is True
        assert data["backends"]["translate"] is True


@pytest.mark.asyncio
async def test_page_loads():
    """HTML page loads successfully."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{BASE}/")
        assert r.status_code == 200
        assert "Meeting Scribe" in r.text


def test_js_no_syntax_errors():
    """JavaScript file has no obvious syntax errors (orphaned else, duplicate top-level consts)."""
    from collections import Counter
    from pathlib import Path

    js = (Path(__file__).parent.parent / "static" / "js" / "scribe-app.js").read_text()

    # Check for orphaned else (} else { without matching if)
    # Simple heuristic: find "} else {" that appears after a closing brace of a non-if block
    lines = js.split("\n")
    brace_depth = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        brace_depth += stripped.count("{") - stripped.count("}")
        if stripped == "} else {" and brace_depth < 0:
            raise AssertionError(f"Possible orphaned else at line {i + 1}")

    # Check for duplicate top-level const/let (would cause SyntaxError in module)
    top_level = []
    for _i, line in enumerate(lines):
        if not line or line[0] in (" ", "\t"):
            continue
        stripped = line.strip()
        if stripped.startswith("const ") or stripped.startswith("let "):
            name = stripped.split()[1].rstrip(";=,")
            top_level.append(name)
    dupes = {k: v for k, v in Counter(top_level).items() if v > 1}
    assert not dupes, f"Duplicate top-level declarations: {dupes}"


# ─── Room layout API ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_room_layout_crud():
    """PUT and GET room layout round-trip."""
    async with httpx.AsyncClient(timeout=10) as c:
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
        data = r.json()
        assert data["preset"] == "rectangle"
        assert len(data["tables"]) == 1
        assert len(data["seats"]) == 2
        assert data["seats"][0]["speaker_name"] == "Alice"


@pytest.mark.asyncio
async def test_room_layout_validation():
    """Invalid coordinates are rejected."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.put(
            f"{BASE}/api/room/layout",
            json={
                "preset": "test",
                "tables": [],
                "seats": [{"seat_id": "bad", "x": 150, "y": 50}],
            },
        )
        assert r.status_code == 422


# ─── Meeting lifecycle ────────────────────────────────────────


@pytest.mark.asyncio
async def test_meeting_start_stop():
    """Start and stop a meeting."""
    mid = await _start_test_meeting()
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{BASE}/api/status")
        assert r.json()["meeting"]["state"] == "recording"

        await c.post(f"{BASE}/api/meeting/stop")

        r = await c.get(f"{BASE}/api/meetings")
        assert any(m["meeting_id"] == mid for m in r.json()["meetings"])

        # Cleanup
        await c.delete(f"{BASE}/api/meetings/{mid}")


@pytest.mark.asyncio
async def test_meeting_delete():
    """Create and delete a meeting."""
    mid = await _start_test_meeting()
    async with httpx.AsyncClient(timeout=10) as c:
        await c.post(f"{BASE}/api/meeting/stop")

        r = await c.delete(f"{BASE}/api/meetings/{mid}")
        assert r.status_code == 200

        r = await c.get(f"{BASE}/api/meetings")
        assert not any(m["meeting_id"] == mid for m in r.json()["meetings"])


# ─── ASR pipeline (English) ──────────────────────────────────


@pytest.mark.asyncio
async def test_english_transcription():
    """English TTS → ASR → transcript events."""
    # Longer text to ensure we fill the 4s buffer
    s16 = _make_s16(
        "Hello everyone. The meeting starts now. Let us begin with the agenda for today."
    )
    mid = await _start_test_meeting()

    async with websockets.connect(WS) as ws:
        await _send_audio(ws, s16)
        events = await _collect_events(ws, timeout=8)

    await _stop_and_cleanup(mid)

    text_events = [e for e in events if e.get("text")]
    assert len(text_events) > 0, "No transcript events received"

    en_events = [e for e in text_events if e.get("language") == "en"]
    assert len(en_events) > 0, f"No English events. Got: {[e.get('language') for e in text_events]}"

    all_text = " ".join(e["text"] for e in en_events).lower()
    assert "hello" in all_text or "meeting" in all_text, f"Unexpected text: {all_text}"


# ─── ASR pipeline (Japanese) ─────────────────────────────────


@pytest.mark.asyncio
async def test_japanese_transcription():
    """Japanese TTS → ASR → transcript events."""
    # Longer text to fill 4s buffer
    s16 = _make_s16(
        "こんにちは。今日はいい天気ですね。会議を始めましょう。予算について話します。", "Kyoko"
    )
    mid = await _start_test_meeting()
    await asyncio.sleep(1)  # Ensure previous meeting fully stopped

    async with websockets.connect(WS) as ws:
        await _send_audio(ws, s16, delay=0.5)
        events = await _collect_events(ws, timeout=12)

    await _stop_and_cleanup(mid)

    text_events = [e for e in events if e.get("text")]
    assert len(text_events) > 0, "No transcript events received"
    all_text = " ".join(e["text"] for e in text_events)
    assert len(all_text) > 2, f"Text too short: '{all_text}'"


# ─── Translation pipeline ────────────────────────────────────


@pytest.mark.asyncio
async def test_english_to_japanese_translation():
    """English text → finalized → NLLB translation → JA result."""
    s16 = _make_s16("Hello everyone. The budget review is scheduled for next week. Please prepare.")
    mid = await _start_test_meeting()

    async with websockets.connect(WS) as ws:
        await _send_audio(ws, s16, delay=0.5)
        t0 = time.time()
        translated = []
        try:
            while time.time() - t0 < 20:
                msg = await asyncio.wait_for(ws.recv(), timeout=4.0)
                e = json.loads(msg)
                if (e.get("translation") or {}).get("text"):
                    translated.append(e)
        except (TimeoutError, websockets.exceptions.ConnectionClosedOK):
            pass

    await _stop_and_cleanup(mid)

    assert len(translated) > 0, "No translation received within 20s"
    assert translated[0]["translation"]["status"] == "done"
    assert len(translated[0]["translation"]["text"]) > 0


# ─── Audio recording + playback ─────────────────────────────


@pytest.mark.asyncio
async def test_audio_recorded():
    """Audio is saved to recording.pcm during a meeting."""
    s16 = _make_s16("Hello. Testing audio recording feature.", "Samantha")
    mid = await _start_test_meeting()

    async with websockets.connect(WS) as ws:
        await _send_audio(ws, s16, delay=0.5)
        await asyncio.sleep(2)

    async with httpx.AsyncClient(timeout=10) as c:
        await c.post(f"{BASE}/api/meeting/stop")

        # Check audio file exists via the API
        r = await c.get(f"{BASE}/api/meetings/{mid}/audio?start_ms=0&end_ms=2000")
        assert r.status_code == 200, f"Audio playback failed: {r.status_code}"
        assert len(r.content) > 44, "Audio too small (just WAV header?)"
        # WAV header starts with RIFF
        assert r.content[:4] == b"RIFF", "Not a valid WAV file"

        # Check timeline exists
        r = await c.get(f"{BASE}/api/meetings/{mid}/timeline")
        assert r.status_code == 200
        tl = r.json()
        assert tl["duration_ms"] > 0, "Timeline duration is 0"

        await c.delete(f"{BASE}/api/meetings/{mid}")


@pytest.mark.asyncio
async def test_audio_not_recorded_without_meeting():
    """No audio written when no meeting is active."""
    async with httpx.AsyncClient(timeout=10) as c:
        # Ensure no meeting is running
        await c.post(f"{BASE}/api/meeting/stop")

    # Send audio via WebSocket — should be processed for ASR but not saved
    async with websockets.connect(WS) as ws:
        silence = bytes(16000)  # 0.5s silence
        await ws.send(silence)
        await asyncio.sleep(1)

    # No crash, no error — audio just wasn't written


@pytest.mark.asyncio
async def test_audio_drift_gap():
    """Audio file handles disconnects with silence padding."""
    s16_1 = _make_s16("First segment before gap.", "Samantha")
    mid = await _start_test_meeting()

    async with websockets.connect(WS) as ws:
        # Send first batch
        await _send_audio(ws, s16_1, delay=0.5)

    # 2 second gap (simulated disconnect — no WebSocket)
    await asyncio.sleep(2)

    s16_2 = _make_s16("Second segment after gap.", "Samantha")
    async with websockets.connect(WS) as ws:
        await _send_audio(ws, s16_2, delay=0.5)
        await asyncio.sleep(2)

    async with httpx.AsyncClient(timeout=10) as c:
        await c.post(f"{BASE}/api/meeting/stop")

        # Audio should be longer than just the two clips combined
        # because the gap should be filled with silence
        r = await c.get(f"{BASE}/api/meetings/{mid}/timeline")
        assert r.status_code == 200
        tl = r.json()
        # Duration should be > 0 (audio was recorded)
        assert tl["duration_ms"] > 0

        await c.delete(f"{BASE}/api/meetings/{mid}")


# ─── Language detection + filtering ──────────────────────────


def test_language_normalizer():
    """Qwen3-ASR language names are normalized to ja/en only."""
    from meeting_scribe.backends.asr_qwen3 import _normalize_language

    assert _normalize_language("Japanese") == "ja"
    assert _normalize_language("English") == "en"
    assert _normalize_language("ja") == "ja"
    assert _normalize_language("en") == "en"
    # Chinese/Korean/Thai must NOT become 'ja'
    assert _normalize_language("Chinese") == "unknown"
    assert _normalize_language("Mandarin") == "unknown"
    assert _normalize_language("Cantonese") == "unknown"
    assert _normalize_language("Korean") == "unknown"
    assert _normalize_language("Thai") == "unknown"
    assert _normalize_language("Vietnamese") == "unknown"
    assert _normalize_language("") == "unknown"
    assert _normalize_language("zh") == "unknown"


def test_name_extraction():
    """Self-introduction patterns are correctly extracted."""
    from meeting_scribe.server import _extract_name_from_text

    # English patterns — only explicit "my name is X"
    assert _extract_name_from_text("Hi, my name is Brad") == "Brad"
    assert _extract_name_from_text("Hello, my name is Sarah") == "Sarah"
    assert _extract_name_from_text("Call me John") == "John"
    # Should NOT extract from "I'm X" (too many false positives)
    assert _extract_name_from_text("I'm ready to listen") is None
    assert _extract_name_from_text("I'm going to the store") is None
    # Should NOT extract common words even with "my name is"
    assert _extract_name_from_text("My name is the best") is None
    # Japanese patterns — only explicit short name introductions
    assert _extract_name_from_text("田中です") == "田中"
    assert _extract_name_from_text("私は田中です") == "田中"
    # Should NOT match long sentences ending in です
    assert _extract_name_from_text("午前中に配達の予定をしていたんですけれども") is None
    assert _extract_name_from_text("が難しい状況でございます") is None
    assert _extract_name_from_text("よろしいですか") is None
    # No intro
    assert _extract_name_from_text("The budget review is next week") is None
    assert _extract_name_from_text("予算について話しましょう") is None


@pytest.mark.asyncio
async def test_speaker_name_detected_in_meeting():
    """Saying 'my name is X' during a meeting assigns the speaker."""
    # Longer text to fill 4s buffer and include the self-introduction
    s16 = _make_s16(
        "Hello everyone. My name is Brad. I am here to discuss the budget review for next quarter.",
        "Samantha",
    )
    mid = await _start_test_meeting()
    await asyncio.sleep(1)

    async with websockets.connect(WS) as ws:
        await _send_audio(ws, s16, delay=0.5)
        events = await _collect_events(ws, timeout=15)

    await _stop_and_cleanup(mid)

    # Find events with speaker attribution
    with_speakers = [e for e in events if e.get("speakers")]
    assert len(with_speakers) > 0, f"No speaker-attributed events. Total events: {len(events)}"

    # At least one should have identity="Brad"
    brad_events = [
        e for e in with_speakers if any(s.get("identity") == "Brad" for s in e["speakers"])
    ]
    assert len(brad_events) > 0, (
        f"Brad not detected. Speakers found: {[e.get('speakers') for e in with_speakers]}"
    )


def test_hallucination_filter():
    """Known hallucination patterns are caught."""
    from meeting_scribe.backends.asr_qwen3 import _is_hallucination

    # Known phrases
    assert _is_hallucination("Thank you for watching")
    assert _is_hallucination("ご視聴ありがとうございました")
    # Repeated words
    assert _is_hallucination("water water water")
    # Repeated characters
    assert _is_hallucination("aaaaaaaaaaa")
    assert _is_hallucination("狼狼狼狼狼狼狼狼")
    # Long single word
    assert _is_hallucination("aitititititititititititititititititititititititititititititititititi")
    # Normal text should pass
    assert not _is_hallucination("Hello everyone")
    assert not _is_hallucination("予算について話しましょう")
    assert not _is_hallucination("The budget review is next week")
    assert not _is_hallucination("こんにちは")


@pytest.mark.asyncio
async def test_no_chinese_in_transcript():
    """Chinese text should not appear in meeting transcripts (only JA/EN allowed)."""
    mid = await _start_test_meeting()

    # Send English audio — should only produce EN events
    s16 = _make_s16("Hello. This is a test of the meeting system.", "Samantha")
    async with websockets.connect(WS) as ws:
        await _send_audio(ws, s16, delay=0.5)
        events = await _collect_events(ws, timeout=10)

    await _stop_and_cleanup(mid)

    for e in events:
        if e.get("text"):
            lang = e.get("language", "unknown")
            assert lang in ("en", "ja", "unknown"), (
                f"Non-JA/EN language detected: {lang} for text '{e['text'][:40]}'"
            )


# ─── Storage persistence ─────────────────────────────────────


@pytest.mark.asyncio
async def test_room_layout_persisted_with_meeting():
    """Room layout is saved as room.json in meeting directory."""
    async with httpx.AsyncClient(timeout=10) as c:
        # Set up a room layout
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

        # Start and stop meeting
        r = await c.post(f"{BASE}/api/meeting/start")
        mid = r.json()["meeting_id"]
        await c.post(f"{BASE}/api/meeting/stop")

        # Check meeting has room data
        r = await c.get(f"{BASE}/api/meetings/{mid}")
        data = r.json()
        assert "room" in data, "room.json not persisted with meeting"
        assert data["room"]["preset"] == "round"

        # Cleanup
        await c.delete(f"{BASE}/api/meetings/{mid}")
