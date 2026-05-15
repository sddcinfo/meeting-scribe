"""Security tests — path traversal, input validation, access control."""

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
async def test_path_traversal_meeting_id():
    """Meeting ID with .. should be rejected."""
    async with httpx.AsyncClient(timeout=10, verify=False) as c:
        r = await c.get(f"{BASE}/api/meetings/../../etc/passwd")
        assert r.status_code in (400, 404)

        r = await c.get(f"{BASE}/api/meetings/../../../etc/passwd/audio")
        assert r.status_code in (400, 404)


@pytest.mark.asyncio
async def test_path_traversal_tts_segment_id():
    """Segment ID with .. should be rejected."""
    async with httpx.AsyncClient(timeout=10, verify=False) as c:
        r = await c.get(f"{BASE}/api/meetings/test-id/tts/../../etc/passwd")
        assert r.status_code in (400, 404)


@pytest.mark.asyncio
async def test_path_traversal_slash_in_id():
    """Slash in meeting ID should be rejected."""
    async with httpx.AsyncClient(timeout=10, verify=False) as c:
        r = await c.get(f"{BASE}/api/meetings/valid-id/../../../etc/passwd/summary")
        assert r.status_code in (400, 404)


@pytest.mark.asyncio
async def test_empty_meeting_finalize_no_crash():
    """Finalizing an empty meeting (no audio) should not crash the server."""
    async with httpx.AsyncClient(timeout=60, verify=False) as c:
        # Start and immediately stop — no audio
        r = await c.post(f"{BASE}/api/meeting/start")
        mid = r.json()["meeting_id"]
        await c.post(f"{BASE}/api/meeting/stop")

        # Finalize should succeed without crash
        r = await c.post(f"{BASE}/api/meetings/{mid}/finalize")
        assert r.status_code == 200

        # Server should still be alive
        r = await c.get(f"{BASE}/api/status")
        assert r.status_code == 200

        await c.delete(f"{BASE}/api/meetings/{mid}")


@pytest.mark.asyncio
async def test_nonexistent_meeting_returns_404():
    async with httpx.AsyncClient(timeout=10, verify=False) as c:
        r = await c.get(f"{BASE}/api/meetings/nonexistent-meeting-id-xyz")
        assert r.status_code in (400, 404)

        r = await c.get(f"{BASE}/api/meetings/nonexistent-meeting-id-xyz/summary")
        assert r.status_code in (400, 404)

        r = await c.get(f"{BASE}/api/meetings/nonexistent-meeting-id-xyz/timeline")
        assert r.status_code in (400, 404)
