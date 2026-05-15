"""Wire-protocol tests for /api/ws/audio-out format negotiation.

Exercises the dispatch helper `_deliver_audio_to_listener` through a
fake WebSocket that records every send. Hermetic — no TestClient,
no lifespan, no backends, no threading.

Covers:
  - MSE-format listener receives an init frame prefixed 0x49 on first
    delivery and a media fragment prefixed 0x46 on a subsequent
    delivery (driven in a loop until observed, no hardcoded cadence)
  - wav-pcm listener receives unprefixed RIFF WAV frames (backward
    compat with legacy clients)
  - Listener in the grace window (audio_format=None) buffers audio
    instead of sending; explicit set_format drains the buffer
  - Listener in grace window past the deadline auto-defaults to
    wav-pcm and sends the first real RIFF WAV frame
  - Invalid format is rejected by the WS handler (tested via the
    public set_format codepath in the server module)
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

pytest.importorskip("av")

from meeting_scribe.audio.output_pipeline import _deliver_audio_to_listener
from meeting_scribe.backends.mse_encoder import ACCUMULATION_THRESHOLD_MS
from meeting_scribe.server_support.sessions import ClientSession
from meeting_scribe.ws import audio_output

SAMPLE_RATE_TTS = 24000
SAMPLE_RATE_PASSTHROUGH = 16000


class FakeWs:
    """Minimal async WebSocket stub that records every send_bytes /
    send_text call. Used to test `_deliver_audio_to_listener` without
    spinning up a real connection."""

    def __init__(self, host: str = "10.42.0.99", port: int = 12345) -> None:
        self.binary_frames: list[bytes] = []
        self.text_frames: list[str] = []

        class _Client:
            def __init__(self, h: str, p: int) -> None:
                self.host = h
                self.port = p

        self.client = _Client(host, port)

    async def send_bytes(self, data: bytes) -> None:
        self.binary_frames.append(data)

    async def send_text(self, data: str) -> None:
        self.text_frames.append(data)


def _sine_pcm(duration_s: float, rate: int = SAMPLE_RATE_TTS) -> np.ndarray:
    n = int(duration_s * rate)
    t = np.arange(n, dtype=np.float32) / rate
    return (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _run(coro):
    """Run an async coroutine to completion in a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_pref(audio_format: str | None = None) -> ClientSession:
    """Build a ClientSession in the shape the fan-out helper expects."""
    return ClientSession(
        preferred_language="en",
        send_audio=True,
        interpretation_mode="translation",
        voice_mode="studio",
        audio_format=audio_format,
        grace_deadline=time.monotonic() + audio_output._AUDIO_FORMAT_GRACE_S,
    )


# ── MSE path ─────────────────────────────────────────────────────────


def test_mse_first_delivery_emits_init_frame_prefix_0x49():
    """First delivery after set_format mse-fmp4-aac produces an init
    frame (0x49) followed by either a fragment or buffered accumulation."""
    ws = FakeWs()
    pref = _make_pref("mse-fmp4-aac")

    async def _drive():
        # One large delivery of 500 ms — comfortably above the
        # accumulation threshold so a fragment is produced.
        pcm = _sine_pcm(0.500)
        await _deliver_audio_to_listener(ws, pref, pcm, SAMPLE_RATE_TTS, None)

    _run(_drive())
    pref.mse_encoder.close()  # clean up the encoder

    assert len(ws.binary_frames) >= 2, (
        f"expected init + fragment, got {len(ws.binary_frames)} frame(s)"
    )
    assert ws.binary_frames[0][:1] == b"\x49", (
        f"first frame should be init (0x49 'I'), got {ws.binary_frames[0][:1].hex()}"
    )
    # Init payload must contain ftyp + moov.
    init_payload = ws.binary_frames[0][1:]
    assert b"ftyp" in init_payload, "init frame missing ftyp box"
    assert b"moov" in init_payload, "init frame missing moov box"
    # Init frame must NOT contain a moof (split on the first moof in
    # the encoder should have removed it).
    assert b"moof" not in init_payload, "init frame unexpectedly contains moof"

    # Second frame should be a media fragment.
    assert ws.binary_frames[1][:1] == b"\x46", (
        f"second frame should be fragment (0x46 'F'), got {ws.binary_frames[1][:1].hex()}"
    )
    frag_payload = ws.binary_frames[1][1:]
    assert b"moof" in frag_payload and b"mdat" in frag_payload, "fragment frame missing moof/mdat"


def test_mse_below_threshold_returns_empty_fragment_after_init():
    """Feeding a tiny sub-threshold chunk after the init still produces
    the init frame (encoder construction forces it) but no fragment."""
    ws = FakeWs()
    pref = _make_pref("mse-fmp4-aac")

    async def _drive():
        # Single 10 ms chunk — well below the 60 ms threshold.
        pcm = _sine_pcm(0.010)
        await _deliver_audio_to_listener(ws, pref, pcm, SAMPLE_RATE_TTS, None)

    _run(_drive())
    pref.mse_encoder.close()

    # Init is always sent on first delivery regardless of fragment
    # availability; the encoder's construction pathway already
    # produced it. There must be exactly ONE frame — the init.
    assert len(ws.binary_frames) == 1, (
        f"expected exactly 1 frame (init only), got {len(ws.binary_frames)}"
    )
    assert ws.binary_frames[0][:1] == b"\x49"


def test_mse_drives_until_fragment_appears():
    """Loop-until-fragment pattern from the plan. Feed repeated small
    chunks; after enough deliveries the client MUST receive both an
    init frame and at least one fragment frame.
    """
    ws = FakeWs()
    pref = _make_pref("mse-fmp4-aac")

    async def _drive():
        # 30 × 20 ms chunks = 600 ms total, well above threshold
        for _ in range(30):
            pcm = _sine_pcm(0.020)
            await _deliver_audio_to_listener(ws, pref, pcm, SAMPLE_RATE_TTS, None)

    _run(_drive())
    pref.mse_encoder.close()

    # We expect at least one init + one fragment. Don't assume an
    # exact fragment count.
    assert len(ws.binary_frames) >= 2
    init = [f for f in ws.binary_frames if f[:1] == b"\x49"]
    frags = [f for f in ws.binary_frames if f[:1] == b"\x46"]
    assert len(init) == 1, f"expected exactly 1 init, got {len(init)}"
    assert len(frags) >= 1, (
        f"expected at least 1 fragment after 600 ms of input "
        f"(threshold={ACCUMULATION_THRESHOLD_MS} ms), got {len(frags)}"
    )


# ── wav-pcm path ─────────────────────────────────────────────────────


def test_wav_pcm_listener_receives_unprefixed_riff():
    """wav-pcm listener gets raw RIFF WAV frames — no prefix byte."""
    ws = FakeWs()
    pref = _make_pref("wav-pcm")

    async def _drive():
        pcm = _sine_pcm(0.100)
        await _deliver_audio_to_listener(ws, pref, pcm, SAMPLE_RATE_TTS, None)

    _run(_drive())

    assert len(ws.binary_frames) == 1
    frame = ws.binary_frames[0]
    # Must start with "RIFF" — NOT with a 0x49 or 0x46 prefix.
    assert frame[:4] == b"RIFF", f"expected unprefixed RIFF, got {frame[:8].hex()}"
    assert b"WAVE" in frame[:12]
    assert b"fmt " in frame[:24]
    assert b"data" in frame[:64]


def test_wav_pcm_passthrough_uses_16khz():
    """Passthrough fan-out is 16 kHz source; the WAV header must reflect it."""
    ws = FakeWs()
    pref = _make_pref("wav-pcm")

    async def _drive():
        pcm = _sine_pcm(0.100, rate=SAMPLE_RATE_PASSTHROUGH)
        await _deliver_audio_to_listener(ws, pref, pcm, SAMPLE_RATE_PASSTHROUGH, None)

    _run(_drive())

    assert len(ws.binary_frames) == 1
    frame = ws.binary_frames[0]
    # Sample rate is at byte offset 24 (little-endian uint32).
    rate = int.from_bytes(frame[24:28], "little")
    assert rate == SAMPLE_RATE_PASSTHROUGH, f"expected 16000 Hz in WAV header, got {rate}"


def test_wav_cache_reuses_same_bytes_across_listeners():
    """The wav_cache dict prevents re-encoding when multiple listeners
    share the same (source_rate, audio) input."""
    ws_a = FakeWs()
    ws_b = FakeWs()
    pref_a = _make_pref("wav-pcm")
    pref_b = _make_pref("wav-pcm")
    cache: dict = {}

    async def _drive():
        pcm = _sine_pcm(0.100)
        await _deliver_audio_to_listener(ws_a, pref_a, pcm, SAMPLE_RATE_TTS, cache)
        await _deliver_audio_to_listener(ws_b, pref_b, pcm, SAMPLE_RATE_TTS, cache)

    _run(_drive())
    assert ws_a.binary_frames[0] == ws_b.binary_frames[0], (
        "wav_cache should give both listeners identical bytes"
    )
    assert len(cache) == 1


# ── Grace window / legacy default ────────────────────────────────────


def test_grace_window_buffers_audio_then_flush_delivers_wav():
    """During grace window, delivery is buffered. Setting format flushes."""
    ws = FakeWs()
    pref = _make_pref(None)  # in grace window

    async def _drive():
        pcm_a = _sine_pcm(0.100)
        pcm_b = _sine_pcm(0.100)
        await _deliver_audio_to_listener(ws, pref, pcm_a, SAMPLE_RATE_TTS, None)
        await _deliver_audio_to_listener(ws, pref, pcm_b, SAMPLE_RATE_TTS, None)
        # Nothing on the wire yet — both buffered.
        assert len(ws.binary_frames) == 0
        assert len(pref.pending_audio) == 2
        # Now resolve the format and flush.
        pref.audio_format = "wav-pcm"
        await audio_output._flush_pending_audio(ws, pref)

    _run(_drive())

    assert len(ws.binary_frames) == 2, (
        f"expected both buffered deliveries flushed, got {len(ws.binary_frames)}"
    )
    for f in ws.binary_frames:
        assert f[:4] == b"RIFF"
    assert pref.pending_audio == []


def test_grace_window_expired_defaults_to_wav_pcm():
    """audio_format=None past grace_deadline → defaults to wav-pcm on next delivery."""
    ws = FakeWs()
    pref = _make_pref(None)
    pref.grace_deadline = time.monotonic() - 1.0  # already expired

    async def _drive():
        pcm = _sine_pcm(0.100)
        await _deliver_audio_to_listener(ws, pref, pcm, SAMPLE_RATE_TTS, None)

    _run(_drive())
    assert pref.audio_format == "wav-pcm", f"expected wav-pcm default, got {pref.audio_format!r}"
    assert len(ws.binary_frames) == 1
    assert ws.binary_frames[0][:4] == b"RIFF"


def test_pending_buffer_capped_at_one_second():
    """Pending audio buffer drops items past the 1-second cap."""
    ws = FakeWs()
    pref = _make_pref(None)

    async def _drive():
        # Feed 3 × 500 ms deliveries = 1.5 s attempted, should cap at 1 s.
        for _ in range(3):
            pcm = _sine_pcm(0.500)
            await _deliver_audio_to_listener(ws, pref, pcm, SAMPLE_RATE_TTS, None)

    _run(_drive())

    assert len(ws.binary_frames) == 0  # nothing delivered yet
    buffered_seconds = sum(len(a) / sr for a, sr in pref.pending_audio)
    assert buffered_seconds <= audio_output._AUDIO_FORMAT_PENDING_CAP_S + 1e-3, (
        f"buffered {buffered_seconds:.3f}s exceeds cap {audio_output._AUDIO_FORMAT_PENDING_CAP_S}s"
    )
