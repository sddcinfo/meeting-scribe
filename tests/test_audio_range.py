"""Tests for HTTP Range support on ``GET /api/meetings/<id>/audio``.

Background: pre-fix the audio route declared ``Accept-Ranges: bytes``
but ignored the ``Range:`` header entirely, so seek-driven
``<audio>`` element fetches on a 1+ hour meeting (~200 MB PCM) caused
the browser to repeatedly download the full file. The route now
honours ``Range`` and streams from disk in 64 KiB chunks.

These tests cover the parser corner cases + the endpoint's response
status / headers / body shape across the three ranges that matter:
header-only, PCM-only, and crossing the header/PCM boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meeting_scribe.routes.meeting_crud import (
    _AUDIO_BYTES_PER_SEC,
    _AUDIO_WAV_HEADER_BYTES,
    _build_wav_header,
    _parse_byte_range,
)


class TestParseByteRange:
    def test_explicit_start_end(self) -> None:
        assert _parse_byte_range("bytes=0-99", total_size=200) == (0, 99)

    def test_open_ended_start(self) -> None:
        assert _parse_byte_range("bytes=50-", total_size=200) == (50, 199)

    def test_suffix_range_returns_last_n_bytes(self) -> None:
        assert _parse_byte_range("bytes=-32", total_size=200) == (168, 199)

    def test_suffix_larger_than_file_clamps_to_zero(self) -> None:
        assert _parse_byte_range("bytes=-500", total_size=200) == (0, 199)

    def test_zero_suffix_is_invalid(self) -> None:
        assert _parse_byte_range("bytes=-0", total_size=200) is None

    def test_end_past_file_is_invalid(self) -> None:
        assert _parse_byte_range("bytes=0-200", total_size=200) is None

    def test_start_greater_than_end_is_invalid(self) -> None:
        assert _parse_byte_range("bytes=100-50", total_size=200) is None

    def test_negative_start_is_invalid(self) -> None:
        assert _parse_byte_range("bytes=-1-50", total_size=200) is None

    def test_missing_bytes_prefix_is_invalid(self) -> None:
        assert _parse_byte_range("0-99", total_size=200) is None

    def test_garbage_is_invalid(self) -> None:
        assert _parse_byte_range("bytes=abc", total_size=200) is None
        assert _parse_byte_range("bytes=0-abc", total_size=200) is None

    def test_only_first_range_is_honoured(self) -> None:
        # Multipart ranges (rare for <audio>) — accept the first spec.
        assert _parse_byte_range("bytes=0-9, 20-29", total_size=200) == (0, 9)


class TestBuildWavHeader:
    def test_header_is_44_bytes(self) -> None:
        assert len(_build_wav_header(0)) == 44
        assert len(_build_wav_header(1024)) == 44

    def test_header_starts_with_riff_wave(self) -> None:
        h = _build_wav_header(1000)
        assert h[:4] == b"RIFF"
        assert h[8:12] == b"WAVE"
        assert h[36:40] == b"data"


def _make_meeting_with_pcm(meetings_dir: Path, mid: str, pcm_bytes: bytes) -> Path:
    """Materialise a finalized meeting on disk so the audio endpoint
    can stream from it. Mirrors the helper in ``test_audio_never_deleted``."""
    md = meetings_dir / mid
    md.mkdir(parents=True)
    (md / "meta.json").write_text(
        json.dumps(
            {
                "meeting_id": mid,
                "state": "complete",
                "created_at": "2026-05-07T00:00:00+00:00",
                "language_pair": ["ja", "en"],
                "max_attendees": 10,
                "audio_sample_rate": 16000,
                "enrolled_speakers": {},
                "recording_started_epoch_ms": 0,
            }
        )
    )
    (md / "audio").mkdir()
    (md / "audio" / "recording.pcm").write_bytes(pcm_bytes)
    return md


@pytest.fixture()
def _audio_test_client(monkeypatch, tmp_path):
    """Build a minimal FastAPI app with just the meeting CRUD routes,
    pointed at a fresh on-disk meetings dir. Avoids ``importlib.reload``
    on the full server module (which contaminates other tests' global
    state) and skips the heavy lifespan startup (vLLM probes, TTS
    warmup, etc.) we don't need for endpoint behaviour tests."""
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from meeting_scribe.config import ServerConfig
    from meeting_scribe.routes import meeting_crud
    from meeting_scribe.runtime import state
    from meeting_scribe.storage import MeetingStorage

    meetings_dir = tmp_path / "meetings"
    meetings_dir.mkdir()

    cfg = ServerConfig()
    cfg.meetings_dir = meetings_dir
    fresh_storage = MeetingStorage(cfg)
    monkeypatch.setattr(state, "storage", fresh_storage)

    app = FastAPI()
    app.include_router(meeting_crud.router)
    with TestClient(app) as client:
        yield client, meetings_dir


class TestAudioRangeEndpoint:
    """Endpoint-level: status code, headers, and body shape for the
    three byte ranges that matter (header-only, PCM-only, boundary)."""

    # 5 seconds of zeroed PCM at 16 kHz mono 16-bit → 160 000 bytes.
    PCM_BYTES = b"\x00" * (5 * _AUDIO_BYTES_PER_SEC)
    EXPECTED_TOTAL = _AUDIO_WAV_HEADER_BYTES + len(PCM_BYTES)

    def test_no_range_returns_200_full_wav(self, _audio_test_client) -> None:
        client, meetings_dir = _audio_test_client
        _make_meeting_with_pcm(meetings_dir, "m-fullread", self.PCM_BYTES)

        r = client.get("/api/meetings/m-fullread/audio")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("audio/wav")
        assert r.headers["accept-ranges"] == "bytes"
        assert "content-range" not in {k.lower() for k in r.headers}
        assert len(r.content) == self.EXPECTED_TOTAL
        assert r.content[:4] == b"RIFF"

    def test_range_returns_206_with_content_range(self, _audio_test_client) -> None:
        client, meetings_dir = _audio_test_client
        _make_meeting_with_pcm(meetings_dir, "m-range", self.PCM_BYTES)

        r = client.get(
            "/api/meetings/m-range/audio",
            headers={"Range": "bytes=0-99"},
        )
        assert r.status_code == 206
        assert r.headers["content-range"] == f"bytes 0-99/{self.EXPECTED_TOTAL}"
        assert len(r.content) == 100
        # First 44 bytes are the WAV header; the rest is PCM zeros.
        assert r.content[:4] == b"RIFF"

    def test_range_inside_header_only(self, _audio_test_client) -> None:
        """Range that lives entirely within the 44-byte WAV header."""
        client, meetings_dir = _audio_test_client
        _make_meeting_with_pcm(meetings_dir, "m-header", self.PCM_BYTES)

        r = client.get(
            "/api/meetings/m-header/audio",
            headers={"Range": "bytes=0-43"},
        )
        assert r.status_code == 206
        assert len(r.content) == _AUDIO_WAV_HEADER_BYTES
        assert r.content[:4] == b"RIFF"

    def test_range_inside_pcm_only(self, _audio_test_client) -> None:
        """Range past the header — should yield only PCM bytes."""
        client, meetings_dir = _audio_test_client
        _make_meeting_with_pcm(meetings_dir, "m-pcm-only", self.PCM_BYTES)

        # bytes=44-143 = first 100 bytes of PCM data
        r = client.get(
            "/api/meetings/m-pcm-only/audio",
            headers={"Range": "bytes=44-143"},
        )
        assert r.status_code == 206
        assert len(r.content) == 100
        assert r.content == b"\x00" * 100

    def test_suffix_range(self, _audio_test_client) -> None:
        client, meetings_dir = _audio_test_client
        _make_meeting_with_pcm(meetings_dir, "m-suffix", self.PCM_BYTES)

        r = client.get(
            "/api/meetings/m-suffix/audio",
            headers={"Range": "bytes=-100"},
        )
        assert r.status_code == 206
        assert len(r.content) == 100
        expected_start = self.EXPECTED_TOTAL - 100
        assert r.headers["content-range"] == (
            f"bytes {expected_start}-{self.EXPECTED_TOTAL - 1}/{self.EXPECTED_TOTAL}"
        )

    def test_unsatisfiable_range_returns_416(self, _audio_test_client) -> None:
        client, meetings_dir = _audio_test_client
        _make_meeting_with_pcm(meetings_dir, "m-bad-range", self.PCM_BYTES)

        r = client.get(
            "/api/meetings/m-bad-range/audio",
            headers={"Range": f"bytes=0-{self.EXPECTED_TOTAL + 1000}"},
        )
        assert r.status_code == 416
        assert r.headers["content-range"] == f"bytes */{self.EXPECTED_TOTAL}"
