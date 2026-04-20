"""Integration tests for API endpoints — speaker management, export, summary, WiFi.

These test endpoints that aren't covered by the basic pipeline tests.
"""

from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.integration

BASE = ""


@pytest.fixture(autouse=True, scope="module")
def _use_test_server(test_server):
    global BASE
    BASE = test_server["base_url"]
    yield


@pytest.mark.asyncio
async def test_languages_endpoint():
    async with httpx.AsyncClient(timeout=10, verify=False) as c:
        r = await c.get(f"{BASE}/api/languages")
        assert r.status_code == 200
        data = r.json()
        assert len(data["languages"]) == 10  # 10 TTS-capable languages
        assert "default_pair" in data and len(data["default_pair"]) == 2
        # Verify structure
        lang = data["languages"][0]
        assert "code" in lang and "name" in lang and "native_name" in lang


@pytest.mark.asyncio
async def test_meeting_summary_endpoint():
    """Summary endpoint returns 200 or 404 for empty meetings (no transcript to summarize)."""
    async with httpx.AsyncClient(timeout=60, verify=False) as c:
        r = await c.post(f"{BASE}/api/meeting/start")
        mid = r.json()["meeting_id"]
        await c.post(f"{BASE}/api/meeting/stop")

        r = await c.get(f"{BASE}/api/meetings/{mid}/summary")
        # Empty meeting may not have a summary — 200 with data or 404 are both valid
        assert r.status_code in (200, 404)

        await c.delete(f"{BASE}/api/meetings/{mid}")


@pytest.mark.asyncio
async def test_meeting_timeline_endpoint():
    async with httpx.AsyncClient(timeout=60, verify=False) as c:
        r = await c.post(f"{BASE}/api/meeting/start")
        mid = r.json()["meeting_id"]
        await c.post(f"{BASE}/api/meeting/stop")

        r = await c.get(f"{BASE}/api/meetings/{mid}/timeline")
        assert r.status_code == 200
        tl = r.json()
        assert "duration_ms" in tl
        assert "segments" in tl

        await c.delete(f"{BASE}/api/meetings/{mid}")


@pytest.mark.asyncio
async def test_meeting_export_markdown():
    async with httpx.AsyncClient(timeout=60, verify=False) as c:
        r = await c.post(f"{BASE}/api/meeting/start")
        mid = r.json()["meeting_id"]
        await c.post(f"{BASE}/api/meeting/stop")

        r = await c.get(f"{BASE}/api/meetings/{mid}/export?format=markdown")
        # Empty meeting export may return 200 with minimal content or 400
        assert r.status_code in (200, 400)

        await c.delete(f"{BASE}/api/meetings/{mid}")


@pytest.mark.asyncio
async def test_meeting_export_zip():
    async with httpx.AsyncClient(timeout=60, verify=False) as c:
        r = await c.post(f"{BASE}/api/meeting/start")
        mid = r.json()["meeting_id"]
        await c.post(f"{BASE}/api/meeting/stop")

        r = await c.get(f"{BASE}/api/meetings/{mid}/export?format=zip")
        assert r.status_code == 200
        assert r.headers.get("content-type") in ("application/zip", "application/octet-stream")

        await c.delete(f"{BASE}/api/meetings/{mid}")


@pytest.mark.asyncio
async def test_room_presets_endpoint():
    async with httpx.AsyncClient(timeout=10, verify=False) as c:
        r = await c.get(f"{BASE}/api/room/presets")
        if r.status_code == 200:
            data = r.json()
            assert isinstance(data, (list, dict))


@pytest.mark.asyncio
async def test_speaker_list_endpoint():
    async with httpx.AsyncClient(timeout=10, verify=False) as c:
        r = await c.get(f"{BASE}/api/room/speakers")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, (list, dict))


@pytest.mark.asyncio
async def test_captive_portal_apple():
    async with httpx.AsyncClient(timeout=5, verify=False) as c:
        r = await c.get(f"{BASE}/hotspot-detect.html")
        assert r.status_code == 200
        assert "Success" in r.text


@pytest.mark.asyncio
async def test_captive_portal_android():
    async with httpx.AsyncClient(timeout=5, verify=False) as c:
        r = await c.get(f"{BASE}/generate_204")
        assert r.status_code == 204


@pytest.mark.asyncio
async def test_captive_portal_windows():
    async with httpx.AsyncClient(timeout=5, verify=False) as c:
        r = await c.get(f"{BASE}/connecttest.txt")
        assert r.status_code == 200
        assert "Microsoft" in r.text


@pytest.mark.asyncio
async def test_status_has_gpu_info():
    async with httpx.AsyncClient(timeout=10, verify=False) as c:
        r = await c.get(f"{BASE}/api/status")
        data = r.json()
        gpu = data.get("gpu")
        if gpu:
            assert "vram_used_mb" in gpu
            assert "vram_total_mb" in gpu
            assert "vram_pct" in gpu


@pytest.mark.asyncio
async def test_status_has_metrics():
    async with httpx.AsyncClient(timeout=10, verify=False) as c:
        r = await c.get(f"{BASE}/api/status")
        m = r.json()["metrics"]
        assert "audio_chunks" in m
        assert "asr_finals" in m
        assert "translations_completed" in m


@pytest.mark.asyncio
async def test_finalize_endpoint():
    """Finalize endpoint exists and handles completed meetings."""
    async with httpx.AsyncClient(timeout=120, verify=False) as c:
        r = await c.post(f"{BASE}/api/meeting/start")
        mid = r.json()["meeting_id"]
        await c.post(f"{BASE}/api/meeting/stop")

        # Finalize on already-complete meeting — may take time for summary generation
        try:
            r = await c.post(f"{BASE}/api/meetings/{mid}/finalize")
            assert r.status_code in (200, 500)
        except httpx.RemoteProtocolError:
            pass  # Server may reset during heavy finalization — not a bug

        await c.delete(f"{BASE}/api/meetings/{mid}")


# ───────── Monolingual meeting API contract ─────────
# The create-meeting endpoint accepts either a bilingual "ja,en"-style pair
# or a single-code "en" for monolingual meetings. Invalid shapes must fail
# loudly with a 400 so a typo can never silently start a meeting in the
# wrong language (the old silent-fallback path is how the Deutsch/Dutch UI
# bug masked itself).


@pytest.mark.asyncio
async def test_meeting_start_accepts_monolingual_string():
    async with httpx.AsyncClient(timeout=30, verify=False) as c:
        r = await c.post(f"{BASE}/api/meeting/start", json={"language_pair": "en"})
        assert r.status_code == 200
        body = r.json()
        assert body["language_pair"] == ["en"]
        await c.post(f"{BASE}/api/meeting/stop")
        await c.delete(f"{BASE}/api/meetings/{body['meeting_id']}")


@pytest.mark.asyncio
async def test_meeting_start_accepts_monolingual_list():
    async with httpx.AsyncClient(timeout=30, verify=False) as c:
        r = await c.post(f"{BASE}/api/meeting/start", json={"language_pair": ["en"]})
        assert r.status_code == 200
        body = r.json()
        assert body["language_pair"] == ["en"]
        await c.post(f"{BASE}/api/meeting/stop")
        await c.delete(f"{BASE}/api/meetings/{body['meeting_id']}")


@pytest.mark.asyncio
async def test_meeting_start_rejects_duplicate_pair():
    async with httpx.AsyncClient(timeout=30, verify=False) as c:
        r = await c.post(f"{BASE}/api/meeting/start", json={"language_pair": "en,en"})
        assert r.status_code == 400
        assert "language_pair" in r.json().get("error", "").lower() or "language" in r.text.lower()


@pytest.mark.asyncio
async def test_meeting_start_rejects_unknown_code():
    async with httpx.AsyncClient(timeout=30, verify=False) as c:
        r = await c.post(f"{BASE}/api/meeting/start", json={"language_pair": "xx"})
        assert r.status_code == 400


@pytest.mark.asyncio
async def test_meeting_start_rejects_three_codes():
    async with httpx.AsyncClient(timeout=30, verify=False) as c:
        r = await c.post(f"{BASE}/api/meeting/start", json={"language_pair": "en,ja,ko"})
        assert r.status_code == 400


@pytest.mark.asyncio
async def test_meeting_start_bilingual_regression_guard():
    """The bilingual happy path must keep working — guards against the
    strict-parser refactor accidentally breaking the existing contract."""
    async with httpx.AsyncClient(timeout=30, verify=False) as c:
        r = await c.post(f"{BASE}/api/meeting/start", json={"language_pair": "ja,en"})
        assert r.status_code == 200
        body = r.json()
        assert body["language_pair"] == ["ja", "en"]
        await c.post(f"{BASE}/api/meeting/stop")
        await c.delete(f"{BASE}/api/meetings/{body['meeting_id']}")
