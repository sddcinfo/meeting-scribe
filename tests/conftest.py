"""Shared test fixtures for GB10 production testing.

Provides pre-recorded audio fixtures so tests can run without external
TTS dependencies.

Fixtures:
    - audio_en_bytes: English s16le 16kHz PCM (from recorded meeting or synthetic)
    - audio_ja_bytes: Japanese s16le 16kHz PCM (from recorded meeting or synthetic)
    - recorded_meeting: Full 87min meeting fixture (PCM + journal + metadata)
    - gb10_available: Skip marker for tests requiring GB10 hardware
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# Test fixture directory — configurable via env var, with sensible search paths
_FIXTURE_SEARCH = [
    Path(os.environ["SCRIBE_TEST_FIXTURES"]) if os.environ.get("SCRIBE_TEST_FIXTURES") else None,
    Path(__file__).parent.parent / "test-fixtures",  # Inside meeting-scribe repo
    Path(__file__).parent.parent.parent / "test-fixtures",  # One level up (monorepo root)
    Path(__file__).parent.parent.parent.parent / "test-fixtures",  # Two levels up
]
FIXTURE_DIR = next((p for p in _FIXTURE_SEARCH if p and p.exists()), Path("/nonexistent"))
MEETING_FIXTURE = FIXTURE_DIR / "90min_english_2026-04-07"

# Server connection defaults — use HTTPS if certs exist (GB10 production)
_certs_exist = (Path(__file__).parent.parent / "certs" / "cert.pem").exists()
_default_scheme = "https" if _certs_exist else "http"
_default_ws_scheme = "wss" if _certs_exist else "ws"
BASE_URL = os.environ.get("SCRIBE_TEST_BASE", f"{_default_scheme}://127.0.0.1:8080")
WS_URL = os.environ.get("SCRIBE_TEST_WS", f"{_default_ws_scheme}://127.0.0.1:8080/api/ws")


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "gb10: requires GB10 hardware with vLLM backends")
    config.addinivalue_line("markers", "slow: long-running tests (>30s)")
    config.addinivalue_line("markers", "system: full-stack system tests (requires autosre start)")
    config.addinivalue_line("markers", "integration: tests against real running services")


@pytest.fixture(autouse=True)
def _enable_meeting_scribe_logger_propagation():
    """Enable log propagation for meeting_scribe during tests.

    In production, ``meeting_scribe/server.py`` attaches its own handler
    and sets ``propagate=False`` so records don't double-emit through the
    root logger's lastResort handler. Pytest's ``caplog`` fixture installs
    its capture on the root logger by default, so propagation must be on
    for tests that inspect log records via ``caplog``.
    """
    import logging

    logger_ms = logging.getLogger("meeting_scribe")
    original = logger_ms.propagate
    logger_ms.propagate = True
    try:
        yield
    finally:
        logger_ms.propagate = original


def _extract_segment_audio(pcm_path: Path, start_ms: int, end_ms: int) -> bytes:
    """Extract a segment of s16le PCM by time range."""
    sample_rate = 16000
    bytes_per_sample = 2  # s16le
    start_byte = int(start_ms / 1000 * sample_rate) * bytes_per_sample
    end_byte = int(end_ms / 1000 * sample_rate) * bytes_per_sample

    with open(pcm_path, "rb") as f:
        f.seek(start_byte)
        return f.read(end_byte - start_byte)


@pytest.fixture(scope="session")
def audio_en_bytes() -> bytes:
    """English audio as s16le 16kHz PCM (~4 seconds) from recorded meeting.

    Requires real recorded meeting fixture. Tests that use this fixture
    will be skipped if the fixture is not available — we never fall back
    to synthetic audio because ASR models can't transcribe sine waves.
    """
    if not MEETING_FIXTURE.exists():
        pytest.skip(
            f"Recorded meeting fixture not found (searched: {FIXTURE_DIR}). Set SCRIBE_TEST_FIXTURES env var."
        )

    journal = MEETING_FIXTURE / "journal.jsonl"
    pcm = MEETING_FIXTURE / "audio" / "recording.pcm"
    if not journal.exists() or not pcm.exists():
        pytest.skip(f"Fixture incomplete: journal={journal.exists()}, pcm={pcm.exists()}")

    with open(journal) as f:
        for line in f:
            seg = json.loads(line)
            if (
                seg.get("language") == "en"
                and seg.get("is_final")
                and len(seg.get("text", "")) > 20
            ):
                audio = _extract_segment_audio(pcm, seg["start_ms"], seg["end_ms"])
                if len(audio) > 16000:
                    return audio

    pytest.skip("No suitable English audio segment found in fixture")


@pytest.fixture(scope="session")
def audio_ja_bytes() -> bytes:
    """Japanese audio as s16le 16kHz PCM (~4 seconds) from recorded meeting.

    Requires real recorded meeting fixture. Skips if not available.
    """
    if not MEETING_FIXTURE.exists():
        pytest.skip(f"Recorded meeting fixture not found (searched: {FIXTURE_DIR})")

    journal = MEETING_FIXTURE / "journal.jsonl"
    pcm = MEETING_FIXTURE / "audio" / "recording.pcm"
    if not journal.exists() or not pcm.exists():
        pytest.skip("Fixture incomplete")

    with open(journal) as f:
        for line in f:
            seg = json.loads(line)
            if seg.get("language") == "ja" and seg.get("is_final"):
                audio = _extract_segment_audio(pcm, seg["start_ms"], seg["end_ms"])
                if len(audio) > 8000:
                    return audio

    pytest.skip("No suitable Japanese audio segment found in fixture")


@pytest.fixture(scope="session")
def recorded_meeting() -> dict | None:
    """Full 87-minute recorded meeting fixture.

    Returns dict with:
        - pcm_path: Path to recording.pcm (s16le 16kHz mono)
        - journal: list of TranscriptEvent dicts
        - meta: meeting metadata dict
        - segments: list of (start_ms, end_ms, text, language) tuples
        - duration_ms: total duration in milliseconds

    Returns None if fixture not available.
    """
    if not MEETING_FIXTURE.exists():
        return None

    pcm_path = MEETING_FIXTURE / "audio" / "recording.pcm"
    journal_path = MEETING_FIXTURE / "journal.jsonl"
    meta_path = MEETING_FIXTURE / "meta.json"

    if not pcm_path.exists() or not journal_path.exists():
        return None

    with open(journal_path) as f:
        journal = [json.loads(line) for line in f]

    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    segments = [
        (s["start_ms"], s["end_ms"], s["text"], s["language"]) for s in journal if s["is_final"]
    ]

    duration_ms = max(s["end_ms"] for s in journal) if journal else 0

    return {
        "pcm_path": pcm_path,
        "journal": journal,
        "meta": meta,
        "segments": segments,
        "duration_ms": duration_ms,
    }


@pytest.fixture
def gb10_available() -> bool:
    """Check if GB10 hardware is available for testing."""
    host = os.environ.get("SCRIBE_GB10_HOST", "")
    if not host:
        pytest.skip("SCRIBE_GB10_HOST not set")
    return True


# ── Integration Test Server ──────────────────────────────────────
# Import the test_server fixture so it's available to all test modules


@pytest.fixture(scope="session")
def test_server():
    """Session-scoped fixture: starts isolated test server, yields config, stops on teardown.

    The test server:
    - Runs on port 9080 (separate from production on 8080)
    - Uses a temporary meetings directory (cleaned up after tests)
    - Connects to the SAME model backends (ports 8000-8003)
    """
    from tests.test_server import TEST_PORT, start_test_server, stop_test_server

    proc = start_test_server()

    scheme = "https" if (Path(__file__).parent.parent / "certs" / "cert.pem").exists() else "http"
    ws_scheme = "wss" if scheme == "https" else "ws"

    yield {
        "base_url": f"{scheme}://127.0.0.1:{TEST_PORT}",
        "ws_url": f"{ws_scheme}://127.0.0.1:{TEST_PORT}/api/ws",
        "port": TEST_PORT,
        "pid": proc.pid,
    }

    stop_test_server()


# ── Flaky-test quarantine ─────────────────────────────────────────────
# Wires in tests/pytest_flaky_skip.py — entries in tests/.flaky.toml are
# skipped (smoke lane) or xfailed (nightly lane) at collection time.
from tests.pytest_flaky_skip import (
    pytest_collection_modifyitems as _flaky_collection_modifyitems,
)

_orig_collection_modifyitems = globals().get("pytest_collection_modifyitems")


def pytest_collection_modifyitems(config, items):
    if _orig_collection_modifyitems is not None:
        _orig_collection_modifyitems(config, items)
    _flaky_collection_modifyitems(config, items)
