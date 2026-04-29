"""Tests for ``/api/admin/refinement-stats`` + ``/api/meeting/{id}/polished-status``.

Covers:
  * ``drain_id`` lookup returns exact entry.
  * ``meeting_id`` lookup returns most-recent-wins with ``other_drain_ids``.
  * Live worker counters surface under a ``live`` key when the worker is
    still running.
  * 404 when there is no drain and no live worker.
  * Polished-status reads ``polished.json`` mtime from disk when no
    registry entry exists (post-restart recovery path).

These tests poke at server-module globals directly — the drain
registry is intentionally process-global in production, so that is the
only way to exercise the endpoint without standing up a full server.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from meeting_scribe.routes import admin as admin_routes
from meeting_scribe.runtime import state as runtime_state


class _StubWorker:
    """Minimal stand-in for a running RefinementWorker."""

    def __init__(self, meeting_id: str):
        self._meeting_id = meeting_id
        self.translate_call_count = 42
        self.asr_call_count = 7
        self.last_error_count = 0


def _fresh_registry():
    from meeting_scribe.server_support import refinement_drains as registry

    registry._refinement_drains.clear()
    registry._drain_seq = 0
    return registry


class _FakeCompletedWorker:
    _meeting_id = ""
    translate_call_count = 10
    asr_call_count = 3
    last_error_count = 0

    async def stop(self):
        pass


async def _kickoff_completed_drain(registry, meeting_id: str) -> int:
    """Helper: register and immediately complete one drain entry."""
    worker = _FakeCompletedWorker()
    worker._meeting_id = meeting_id
    registry._drain_seq += 1
    drain_id = registry._drain_seq
    entry = registry._DrainEntry(
        drain_id=drain_id,
        meeting_id=meeting_id,
        task=asyncio.create_task(registry._drain_refinement(worker, meeting_id, drain_id)),
        state="draining",
        started_at=time.time(),
        translate_calls=worker.translate_call_count,
        asr_calls=worker.asr_call_count,
    )
    registry._refinement_drains.append(entry)
    await entry.task
    return drain_id


class TestRefinementStatsEndpoint:
    @pytest.mark.asyncio
    async def test_drain_id_returns_exact_entry(self):
        registry = _fresh_registry()
        drain_id = await _kickoff_completed_drain(registry, "mtg-exact")
        response = await admin_routes.get_admin_refinement_stats(drain_id=drain_id)
        # FastAPI JSONResponse stringifies body to bytes on send but we can
        # inspect it directly via the .body attribute (bytes).
        import json

        body = json.loads(response.body.decode())
        assert body["drain_id"] == drain_id
        assert body["meeting_id"] == "mtg-exact"
        assert body["state"] == "complete"

    @pytest.mark.asyncio
    async def test_drain_id_missing_returns_404(self):
        registry = _fresh_registry()
        response = await admin_routes.get_admin_refinement_stats(drain_id=999_999)
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_meeting_id_returns_most_recent_plus_others(self):
        registry = _fresh_registry()
        first = await _kickoff_completed_drain(registry, "mtg-repeat")
        second = await _kickoff_completed_drain(registry, "mtg-repeat")

        import json

        response = await admin_routes.get_admin_refinement_stats(meeting_id="mtg-repeat")
        body = json.loads(response.body.decode())
        assert body["drain"]["drain_id"] == second
        assert body["other_drain_ids"] == [first]
        assert body["live"] is None

    @pytest.mark.asyncio
    async def test_meeting_id_surfaces_live_worker_when_running(self):
        registry = _fresh_registry()
        stub = _StubWorker(meeting_id="mtg-live")
        # Pretend this worker is the current refinement_worker.
        runtime_state.refinement_worker = stub
        try:
            import json

            response = await admin_routes.get_admin_refinement_stats(meeting_id="mtg-live")
            body = json.loads(response.body.decode())
            assert body["drain"] is None
            assert body["live"] is not None
            assert body["live"]["translate_calls"] == 42
            assert body["live"]["asr_calls"] == 7
            assert body["live"]["meeting_id"] == "mtg-live"
        finally:
            runtime_state.refinement_worker = None

    @pytest.mark.asyncio
    async def test_no_drain_no_worker_returns_404(self):
        registry = _fresh_registry()
        response = await admin_routes.get_admin_refinement_stats(meeting_id="mtg-ghost")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_no_params_returns_400(self):
        registry = _fresh_registry()
        response = await admin_routes.get_admin_refinement_stats()
        assert response.status_code == 400


class TestPolishedStatusEndpoint:
    @staticmethod
    def _install_fake_storage(_registry, root):
        class _FakeStorage:
            def _meeting_dir(self, meeting_id):
                return root / meeting_id

        runtime_state.storage = _FakeStorage()

    @pytest.mark.asyncio
    async def test_drain_entry_is_returned(self, tmp_path):
        registry = _fresh_registry()
        self._install_fake_storage(registry, tmp_path)
        drain_id = await _kickoff_completed_drain(registry, "mtg-poll")

        import json

        response = await admin_routes.get_polished_status(meeting_id="mtg-poll")
        body = json.loads(response.body.decode())
        assert body["drain_id"] == drain_id
        assert body["state"] == "complete"
        assert "polished_json_mtime" in body

    @pytest.mark.asyncio
    async def test_reads_from_disk_when_no_registry_entry(self, tmp_path):
        registry = _fresh_registry()
        fake_meetings = tmp_path / "meetings"
        (fake_meetings / "mtg-disk").mkdir(parents=True)
        polished = fake_meetings / "mtg-disk" / "polished.json"
        polished.write_text("{}")

        self._install_fake_storage(registry, fake_meetings)

        import json

        response = await admin_routes.get_polished_status(meeting_id="mtg-disk")
        body = json.loads(response.body.decode())
        assert body["state"] == "complete"
        assert body["drain_id"] is None
        assert body["polished_json_mtime"] is not None
        assert "read from disk" in (body.get("note") or "")

    @pytest.mark.asyncio
    async def test_absent_meeting_returns_404(self, tmp_path):
        registry = _fresh_registry()
        self._install_fake_storage(registry, tmp_path / "meetings")

        response = await admin_routes.get_polished_status(meeting_id="mtg-missing")
        assert response.status_code == 404
