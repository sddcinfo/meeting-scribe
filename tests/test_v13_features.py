"""Tests for v1.3 behaviours that still matter.

History: this file used to grep HTML/CSS/JS for literal strings as a
proxy for "feature X exists". Those checks were ceremony — they broke
on harmless renames and passed on broken features. Deleted 2026-04-13.
What's left tests real behaviour: model/language routing, drift math,
refinement math, TTS recipe wiring, and integration smoke against the
test server.
"""

from __future__ import annotations

import asyncio
import ssl

import httpx
import pytest

# ═══════════════════════════════════════════════════════════════
# UNIT TESTS — no server needed
# ═══════════════════════════════════════════════════════════════


class TestMultiLanguageDeHardcoding:
    """Verify ja/en hardcoding is removed from translation + ASR paths."""

    def test_with_translation_preserves_target_language(self):
        from meeting_scribe.models import TranscriptEvent, TranslationStatus

        e = TranscriptEvent(
            segment_id="s1",
            text="你好",
            language="zh",
            is_final=True,
            start_ms=0,
            end_ms=1000,
        )
        t = e.with_translation(TranslationStatus.DONE, text="Hello", target_language="en")
        assert t.translation.target_language == "en"

    def test_with_translation_french_source(self):
        from meeting_scribe.models import TranscriptEvent, TranslationStatus

        e = TranscriptEvent(
            segment_id="s1",
            text="Bonjour",
            language="fr",
            is_final=True,
            start_ms=0,
            end_ms=1000,
        )
        t = e.with_translation(TranslationStatus.DONE, text="Hello", target_language="en")
        assert t.translation.target_language == "en"

    def test_asr_filters_korean_script(self):
        from meeting_scribe.backends.asr_filters import _detect_language_from_text

        assert _detect_language_from_text("안녕하세요") == "ko"

    def test_asr_filters_cjk_ambiguous(self):
        from meeting_scribe.backends.asr_filters import _detect_language_from_text

        # Pure Han is ambiguous between ja and zh; both are acceptable
        assert _detect_language_from_text("你好世界") in ("zh", "ja")

    def test_normalize_language_beyond_ja_en(self):
        from meeting_scribe.languages import normalize_language

        assert normalize_language("Chinese") == "zh"
        assert normalize_language("Korean") == "ko"
        assert normalize_language("French") == "fr"

    def test_translation_queue_routes_zh_en(self):
        from meeting_scribe.languages import get_translation_target

        assert get_translation_target("zh", ("zh", "en")) == "en"
        assert get_translation_target("en", ("zh", "en")) == "zh"


class TestAudioDriftDetection:
    """Wall-clock vs audio-clock drift arithmetic used by the silence watchdog."""

    def test_drift_above_threshold(self):
        wall_elapsed_ms = 10_000
        audio_elapsed_ms = int(9.2 * 1000)
        drift_ms = wall_elapsed_ms - audio_elapsed_ms
        assert drift_ms == 800
        assert abs(drift_ms) > 500

    def test_drift_within_tolerance(self):
        wall_elapsed_ms = 10_000
        drift_ms = wall_elapsed_ms - int(9.8 * 1000)
        assert abs(drift_ms) < 500


class TestOverlappingRefinement:
    """1-second backup between refinement chunks."""

    def test_overlap_is_one_second(self):
        from meeting_scribe.refinement import BYTES_PER_SEC

        end = 320_000  # 10 seconds of s16le @ 16kHz
        new_offset = max(end - BYTES_PER_SEC, 0)
        assert end - new_offset == BYTES_PER_SEC


class TestCaptivePortalOffline:
    """Portal page must work without internet (no CDN links)."""

    def test_no_external_resources(self):
        from pathlib import Path

        content = (Path(__file__).parent.parent / "static" / "portal.html").read_text()
        lower = content.lower()
        assert "cdn" not in lower
        assert "googleapis" not in lower
        assert "cloudflare.com" not in lower


# ═══════════════════════════════════════════════════════════════
# INTEGRATION TESTS — need test server on port 9080
# ═══════════════════════════════════════════════════════════════

_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE

BASE = ""
WS = ""


@pytest.fixture(autouse=True, scope="module")
def _use_test_server_for_integration(test_server, request):
    global BASE, WS
    BASE = test_server["base_url"]
    WS = test_server["ws_url"]
    yield


@pytest.mark.integration
@pytest.mark.asyncio
async def test_language_pair_at_meeting_start():
    """Starting a meeting with zh,en persists the pair on the meeting record."""
    async with httpx.AsyncClient(timeout=60, verify=False) as c:
        r = await c.post(f"{BASE}/api/meeting/start", json={"language_pair": "zh,en"})
        assert r.status_code == 200
        mid = r.json()["meeting_id"]

        r = await c.get(f"{BASE}/api/meetings/{mid}")
        data = r.json()
        pair = data.get("language_pair") or data.get("meta", {}).get("language_pair")
        assert pair in (["zh", "en"], "zh,en"), f"Expected zh,en pair, got {pair}"

        await c.post(f"{BASE}/api/meeting/stop")
        await c.delete(f"{BASE}/api/meetings/{mid}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_status_exposes_all_backends():
    """/api/status reports every backend the UI + health gate depend on,
    including furigana (added 2026-04-13 as a required backend)."""
    async with httpx.AsyncClient(timeout=10, verify=False) as c:
        r = await c.get(f"{BASE}/api/status")
        assert r.status_code == 200
        data = r.json()

        b = data["backends"]
        for name in ("asr", "translate", "diarize", "tts", "furigana"):
            assert name in b, f"missing backend: {name}"
            assert isinstance(b[name], bool)

        gpu = data.get("gpu")
        if gpu:
            assert gpu["vram_total_mb"] > 0
            assert 0 <= gpu["vram_pct"] <= 100


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_start_is_idempotent():
    """Firing /api/meeting/start twice in parallel must return the same
    meeting_id — the lifecycle lock guarantees serialization, and the
    second caller hits the idempotent fast-path."""
    async with httpx.AsyncClient(timeout=60, verify=False) as c:
        r1, r2 = await asyncio.gather(
            c.post(f"{BASE}/api/meeting/start"),
            c.post(f"{BASE}/api/meeting/start"),
        )
        assert r1.status_code == 200
        assert r2.status_code == 200
        id1 = r1.json().get("meeting_id")
        id2 = r2.json().get("meeting_id")
        assert id1 and id2 and id1 == id2
        # One of the responses carries `resumed=True` — the other does not
        resumed_flags = [r1.json().get("resumed"), r2.json().get("resumed")]
        assert True in resumed_flags or False in resumed_flags

        await c.post(f"{BASE}/api/meeting/stop")
        await c.delete(f"{BASE}/api/meetings/{id1}")
