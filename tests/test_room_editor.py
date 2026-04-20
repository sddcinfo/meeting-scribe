"""Tests for the rich virtual table editor — pre/mid/post meeting.

Covers:
- storage.save_room_layout / load_room_layout round trip
- storage.update_journal_speaker_identity: rewrites events matching cluster_id
- RoomLayout model validation via Pydantic
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from meeting_scribe.config import ServerConfig
from meeting_scribe.models import RoomLayout, SeatPosition, TableObject
from meeting_scribe.storage import MeetingStorage


@pytest.fixture
def storage(tmp_path):
    meetings_dir = tmp_path / "meetings"
    meetings_dir.mkdir()
    cfg = ServerConfig()
    s = MeetingStorage(cfg)
    s._meetings_dir = meetings_dir
    return s


def _make_meeting_dir(storage: MeetingStorage, mid: str) -> None:
    """Create a minimal meeting directory with meta.json."""
    d = storage._meetings_dir / mid
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(
        json.dumps(
            {
                "meeting_id": mid,
                "state": "complete",
                "created_at": datetime.now(UTC).isoformat(),
                "language_pair": ["ja", "en"],
            }
        )
    )


class TestRoomLayoutPersistence:
    def test_save_and_load_roundtrip(self, storage):
        _make_meeting_dir(storage, "m1")
        layout = RoomLayout(
            preset="rectangle",
            tables=[TableObject(x=50, y=50, width=40, height=20, border_radius=5)],
            seats=[
                SeatPosition(x=20, y=50, speaker_name="Alice"),
                SeatPosition(x=80, y=50, speaker_name="Bob"),
            ],
        )
        storage.save_room_layout("m1", layout)

        loaded = storage.load_room_layout("m1")
        assert loaded is not None
        assert loaded.preset == "rectangle"
        assert len(loaded.tables) == 1
        assert len(loaded.seats) == 2
        assert loaded.seats[0].speaker_name == "Alice"
        assert loaded.seats[1].speaker_name == "Bob"

    def test_load_missing_returns_none(self, storage):
        _make_meeting_dir(storage, "empty")
        assert storage.load_room_layout("empty") is None

    def test_load_corrupt_returns_none(self, storage):
        _make_meeting_dir(storage, "corrupt")
        (storage._meetings_dir / "corrupt" / "room.json").write_text("not json{{")
        assert storage.load_room_layout("corrupt") is None

    def test_save_atomic_replaces_existing(self, storage):
        _make_meeting_dir(storage, "m2")
        layout_a = RoomLayout(preset="round", tables=[], seats=[])
        storage.save_room_layout("m2", layout_a)

        layout_b = RoomLayout(
            preset="boardroom",
            tables=[TableObject(x=50, y=50)],
            seats=[SeatPosition(x=30, y=30, speaker_name="New Speaker")],
        )
        storage.save_room_layout("m2", layout_b)

        loaded = storage.load_room_layout("m2")
        assert loaded.preset == "boardroom"
        assert len(loaded.seats) == 1
        assert loaded.seats[0].speaker_name == "New Speaker"


def _journal_event(segment_id: str, cluster_id: int, identity: str | None = None) -> dict:
    return {
        "segment_id": segment_id,
        "revision": 0,
        "is_final": True,
        "start_ms": 0,
        "end_ms": 1000,
        "language": "en",
        "text": "hello",
        "speakers": [
            {
                "cluster_id": cluster_id,
                "identity": identity,
                "identity_confidence": 0.0 if identity is None else 1.0,
                "source": "enrolled" if identity else "cluster_only",
            }
        ],
        "translation": None,
    }


class TestUpdateJournalSpeakerIdentity:
    def test_empty_journal(self, storage):
        _make_meeting_dir(storage, "m1")
        (storage._meetings_dir / "m1" / "journal.jsonl").write_text("")
        updated = storage.update_journal_speaker_identity("m1", 0, "Alice")
        assert updated == 0

    def test_missing_journal(self, storage):
        _make_meeting_dir(storage, "m1")
        updated = storage.update_journal_speaker_identity("m1", 0, "Alice")
        assert updated == 0

    def test_updates_matching_cluster(self, storage):
        _make_meeting_dir(storage, "m1")
        journal = storage._meetings_dir / "m1" / "journal.jsonl"
        events = [
            _journal_event("s1", cluster_id=0),
            _journal_event("s2", cluster_id=1),
            _journal_event("s3", cluster_id=0),
            _journal_event("s4", cluster_id=2),
        ]
        journal.write_text("\n".join(json.dumps(e) for e in events) + "\n")

        updated = storage.update_journal_speaker_identity("m1", cluster_id=0, display_name="Alice")
        assert updated == 2

        # Verify journal state: s1 and s3 have identity=Alice, others unchanged
        lines = journal.read_text().strip().split("\n")
        parsed = [json.loads(line) for line in lines]
        assert parsed[0]["speakers"][0]["identity"] == "Alice"
        assert parsed[0]["speakers"][0]["source"] == "enrolled"
        assert parsed[0]["speakers"][0]["identity_confidence"] == 1.0
        assert parsed[1]["speakers"][0]["identity"] is None
        assert parsed[2]["speakers"][0]["identity"] == "Alice"
        assert parsed[3]["speakers"][0]["identity"] is None

    def test_skips_events_with_no_speakers(self, storage):
        _make_meeting_dir(storage, "m1")
        journal = storage._meetings_dir / "m1" / "journal.jsonl"
        event = _journal_event("s1", cluster_id=0)
        event_no_speakers = {**event, "segment_id": "s2", "speakers": []}
        journal.write_text(json.dumps(event) + "\n" + json.dumps(event_no_speakers) + "\n")

        updated = storage.update_journal_speaker_identity("m1", 0, "Alice")
        assert updated == 1

        lines = journal.read_text().strip().split("\n")
        parsed = [json.loads(line) for line in lines]
        assert parsed[0]["speakers"][0]["identity"] == "Alice"
        assert parsed[1]["speakers"] == []

    def test_overwrites_existing_identity(self, storage):
        """Reassigning a cluster to a new name should overwrite the old identity."""
        _make_meeting_dir(storage, "m1")
        journal = storage._meetings_dir / "m1" / "journal.jsonl"
        event = _journal_event("s1", cluster_id=0, identity="Old Name")
        journal.write_text(json.dumps(event) + "\n")

        updated = storage.update_journal_speaker_identity("m1", 0, "New Name")
        assert updated == 1

        parsed = json.loads(journal.read_text().strip())
        assert parsed["speakers"][0]["identity"] == "New Name"

    def test_preserves_corrupt_lines(self, storage):
        """Malformed lines should be written back unchanged, not dropped."""
        _make_meeting_dir(storage, "m1")
        journal = storage._meetings_dir / "m1" / "journal.jsonl"
        journal.write_text(
            json.dumps(_journal_event("s1", cluster_id=0))
            + "\n"
            + "garbage not json\n"
            + json.dumps(_journal_event("s2", cluster_id=0))
            + "\n"
        )

        updated = storage.update_journal_speaker_identity("m1", 0, "Alice")
        assert updated == 2

        lines = journal.read_text().strip().split("\n")
        assert len(lines) == 3
        assert "garbage" in lines[1]  # corrupt line preserved

    def test_cluster_not_found(self, storage):
        _make_meeting_dir(storage, "m1")
        journal = storage._meetings_dir / "m1" / "journal.jsonl"
        event = _journal_event("s1", cluster_id=5)
        journal.write_text(json.dumps(event) + "\n")

        updated = storage.update_journal_speaker_identity("m1", cluster_id=99, display_name="Ghost")
        assert updated == 0

        parsed = json.loads(journal.read_text().strip())
        assert parsed["speakers"][0]["identity"] is None  # unchanged


class TestRoomLayoutModel:
    """Sanity checks on the Pydantic model we serialize to room.json."""

    def test_default_construction(self):
        layout = RoomLayout()
        assert layout.preset == "rectangle"
        assert layout.tables == []
        assert layout.seats == []

    def test_bounds_validation(self):
        # Positions clamped to [0, 100]
        with pytest.raises(Exception):
            SeatPosition(x=-5, y=50)
        with pytest.raises(Exception):
            SeatPosition(x=50, y=150)

    def test_round_trip_json(self):
        layout = RoomLayout(
            preset="round",
            tables=[TableObject(x=50, y=50, width=30, height=30, border_radius=50)],
            seats=[
                SeatPosition(x=20, y=20, speaker_name="A"),
                SeatPosition(x=80, y=80, speaker_name="B"),
            ],
        )
        raw = layout.model_dump_json()
        restored = RoomLayout.model_validate_json(raw)
        assert restored.preset == "round"
        assert len(restored.seats) == 2
        assert restored.tables[0].border_radius == 50


# ── Integration tests against the isolated test server (:9080) ──


import contextlib as _contextlib

import httpx


def _get_test_meetings_dir():
    """Resolve the test server's meetings dir (set via SCRIBE_MEETINGS_DIR env in subprocess)."""
    from pathlib import Path

    from tests import test_server as ts_mod

    if ts_mod._test_meetings_dir:
        return Path(ts_mod._test_meetings_dir)
    # Fallback: use config
    from meeting_scribe.config import ServerConfig

    return Path(ServerConfig.from_env().meetings_dir)


@pytest.fixture
def live_meeting_id(test_server):
    """Create a real meeting on the test server, append synthetic events, return id.

    Writes journal.jsonl + detected_speakers.json directly into the test server's
    meetings dir (SCRIBE_MEETINGS_DIR) so the endpoints under test have data.
    """
    import json as _json

    base_url = test_server["base_url"]
    meetings_root = _get_test_meetings_dir()

    with httpx.Client(verify=False, timeout=30) as c:
        r = c.post(f"{base_url}/api/meeting/start")
        assert r.status_code == 200
        mid = r.json()["meeting_id"]

        meeting_dir = meetings_root / mid
        meeting_dir.mkdir(parents=True, exist_ok=True)
        journal = meeting_dir / "journal.jsonl"
        events = [
            {
                "segment_id": "seg-1",
                "revision": 0,
                "is_final": True,
                "start_ms": 0,
                "end_ms": 1000,
                "language": "en",
                "text": "hello",
                "speakers": [
                    {
                        "cluster_id": 0,
                        "identity": None,
                        "identity_confidence": 0.0,
                        "source": "cluster_only",
                    }
                ],
                "translation": None,
            },
            {
                "segment_id": "seg-2",
                "revision": 0,
                "is_final": True,
                "start_ms": 1000,
                "end_ms": 2000,
                "language": "en",
                "text": "world",
                "speakers": [
                    {
                        "cluster_id": 1,
                        "identity": None,
                        "identity_confidence": 0.0,
                        "source": "cluster_only",
                    }
                ],
                "translation": None,
            },
        ]
        with open(journal, "a") as f:
            for e in events:
                f.write(_json.dumps(e) + "\n")

        # Seed detected_speakers.json
        speakers_path = meeting_dir / "detected_speakers.json"
        speakers_path.write_text(
            _json.dumps(
                [
                    {
                        "cluster_id": 0,
                        "speaker_id": 0,
                        "display_name": "",
                        "segment_count": 1,
                        "first_seen_ms": 0,
                        "last_seen_ms": 1000,
                        "total_speaking_ms": 1000,
                    },
                    {
                        "cluster_id": 1,
                        "speaker_id": 1,
                        "display_name": "",
                        "segment_count": 1,
                        "first_seen_ms": 1000,
                        "last_seen_ms": 2000,
                        "total_speaking_ms": 1000,
                    },
                ]
            )
        )

        yield mid

        with _contextlib.suppress(Exception):
            c.post(f"{base_url}/api/meeting/stop", timeout=30)
        with _contextlib.suppress(Exception):
            c.delete(f"{base_url}/api/meetings/{mid}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_put_meeting_room_layout(live_meeting_id, test_server):
    """PUT /api/meetings/{id}/room/layout persists and returns 200."""
    base_url = test_server["base_url"]
    mid = live_meeting_id
    layout = {
        "preset": "rectangle",
        "tables": [
            {
                "table_id": "t1",
                "x": 50.0,
                "y": 50.0,
                "width": 40.0,
                "height": 20.0,
                "border_radius": 5.0,
                "label": "",
            }
        ],
        "seats": [
            {"seat_id": "s1", "x": 20.0, "y": 50.0, "enrollment_id": None, "speaker_name": "Alice"},
            {"seat_id": "s2", "x": 80.0, "y": 50.0, "enrollment_id": None, "speaker_name": ""},
        ],
    }
    async with httpx.AsyncClient(verify=False, timeout=15) as c:
        r = await c.put(f"{base_url}/api/meetings/{mid}/room/layout", json=layout)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

        # GET it back
        r2 = await c.get(f"{base_url}/api/meetings/{mid}/room")
        assert r2.status_code == 200
        data = r2.json()
        assert len(data["seats"]) == 2
        assert data["seats"][0]["speaker_name"] == "Alice"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_meeting_room_404(test_server):
    """GET /api/meetings/{id}/room returns 404 for unknown meeting."""
    base_url = test_server["base_url"]
    async with httpx.AsyncClient(verify=False, timeout=10) as c:
        r = await c.get(f"{base_url}/api/meetings/nonexistent-12345/room")
        assert r.status_code in (400, 404)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_put_room_layout_validates(live_meeting_id, test_server):
    """PUT with invalid layout returns 422."""
    base_url = test_server["base_url"]
    async with httpx.AsyncClient(verify=False, timeout=10) as c:
        # Invalid: seat.x out of bounds
        r = await c.put(
            f"{base_url}/api/meetings/{live_meeting_id}/room/layout",
            json={
                "preset": "rectangle",
                "tables": [],
                "seats": [{"x": -500, "y": 50, "speaker_name": "bad"}],
            },
        )
        assert r.status_code == 422


@pytest.mark.integration
@pytest.mark.asyncio
async def test_assign_speaker_to_seat(live_meeting_id, test_server):
    """POST /speakers/assign binds cluster to seat and rewrites journal."""
    base_url = test_server["base_url"]
    mid = live_meeting_id

    layout = {
        "preset": "rectangle",
        "tables": [],
        "seats": [
            {"seat_id": "s1", "x": 20, "y": 50, "enrollment_id": None, "speaker_name": ""},
            {"seat_id": "s2", "x": 80, "y": 50, "enrollment_id": None, "speaker_name": ""},
        ],
    }
    async with httpx.AsyncClient(verify=False, timeout=15) as c:
        # Seed the layout
        r = await c.put(f"{base_url}/api/meetings/{mid}/room/layout", json=layout)
        assert r.status_code == 200

        # Assign cluster 0 → seat s1 as "Alice"
        r = await c.post(
            f"{base_url}/api/meetings/{mid}/speakers/assign",
            json={"cluster_id": 0, "seat_id": "s1", "display_name": "Alice"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["updated_events"] == 1  # seg-1 matched cluster 0
        assert data["display_name"] == "Alice"

        # Verify seat name updated
        r2 = await c.get(f"{base_url}/api/meetings/{mid}/room")
        seats = r2.json()["seats"]
        s1 = next(s for s in seats if s["seat_id"] == "s1")
        assert s1["speaker_name"] == "Alice"

        # Verify journal rewrite via direct file read
        import json as _json

        journal = _get_test_meetings_dir() / mid / "journal.jsonl"
        lines = [
            _json.loads(line) for line in journal.read_text().strip().split("\n") if line.strip()
        ]
        seg1 = next(e for e in lines if e["segment_id"] == "seg-1")
        assert seg1["speakers"][0]["identity"] == "Alice"
        assert seg1["speakers"][0]["source"] == "enrolled"
        # seg-2 unaffected
        seg2 = next(e for e in lines if e["segment_id"] == "seg-2")
        assert seg2["speakers"][0]["identity"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_assign_speaker_rename_only(live_meeting_id, test_server):
    """POST /speakers/assign with seat_id=null still updates detected_speakers + journal."""
    base_url = test_server["base_url"]
    mid = live_meeting_id
    async with httpx.AsyncClient(verify=False, timeout=15) as c:
        r = await c.post(
            f"{base_url}/api/meetings/{mid}/speakers/assign",
            json={"cluster_id": 1, "seat_id": None, "display_name": "Bob"},
        )
        assert r.status_code == 200
        assert r.json()["updated_events"] == 1  # seg-2 matched cluster 1

        # Verify journal
        import json as _json

        journal = _get_test_meetings_dir() / mid / "journal.jsonl"
        lines = [
            _json.loads(line) for line in journal.read_text().strip().split("\n") if line.strip()
        ]
        seg2 = next(e for e in lines if e["segment_id"] == "seg-2")
        assert seg2["speakers"][0]["identity"] == "Bob"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_assign_speaker_validation(live_meeting_id, test_server):
    """POST /speakers/assign rejects empty display_name."""
    base_url = test_server["base_url"]
    async with httpx.AsyncClient(verify=False, timeout=10) as c:
        r = await c.post(
            f"{base_url}/api/meetings/{live_meeting_id}/speakers/assign",
            json={"cluster_id": 0, "seat_id": None, "display_name": ""},
        )
        assert r.status_code == 422
