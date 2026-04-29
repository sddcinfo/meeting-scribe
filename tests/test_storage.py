"""Tests for meeting storage — journal replay, metadata persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meeting_scribe.models import MeetingMeta, MeetingState
from meeting_scribe.storage import MeetingStorage


@pytest.fixture
def storage(tmp_path):
    """Create a MeetingStorage with a temp directory."""
    from meeting_scribe.config import ServerConfig

    meetings_dir = tmp_path / "meetings"
    meetings_dir.mkdir()
    cfg = ServerConfig()
    # MeetingStorage expects meetings_dir as a Path-compatible value
    s = MeetingStorage(cfg)
    s._meetings_dir = meetings_dir
    return s


class TestJournalReplay:
    """read_journal_raw for multi-browser sync."""

    def test_read_empty_journal(self, storage):
        meta = storage.create_meeting(MeetingMeta())
        lines = storage.read_journal_raw(meta.meeting_id)
        assert lines == []

    def test_read_journal_with_events(self, storage):
        meta = storage.create_meeting(MeetingMeta())
        mid = meta.meeting_id
        # Write some events
        journal_path = Path(storage._meetings_dir) / mid / "journal.jsonl"
        events = [
            json.dumps({"segment_id": f"s{i}", "text": f"Event {i}", "is_final": True})
            for i in range(5)
        ]
        journal_path.write_text("\n".join(events) + "\n")

        lines = storage.read_journal_raw(mid)
        assert len(lines) == 5
        assert json.loads(lines[0])["segment_id"] == "s0"

    def test_read_journal_max_lines(self, storage):
        meta = storage.create_meeting(MeetingMeta())
        mid = meta.meeting_id
        journal_path = Path(storage._meetings_dir) / mid / "journal.jsonl"
        events = [json.dumps({"segment_id": f"s{i}", "text": f"Event {i}"}) for i in range(100)]
        journal_path.write_text("\n".join(events) + "\n")

        lines = storage.read_journal_raw(mid, max_lines=10)
        assert len(lines) == 10
        # Should be the LAST 10 events
        assert json.loads(lines[0])["segment_id"] == "s90"

    def test_read_journal_nonexistent_meeting(self, storage):
        lines = storage.read_journal_raw("nonexistent-id")
        assert lines == []


def _bring_meeting_to_state(storage: MeetingStorage, target: MeetingState) -> str:
    """Drive a fresh meeting through the legal transition chain to ``target``.

    Keeps state-machine discipline in the fixture itself so tests
    never bypass transition_state().
    """
    meta = storage.create_meeting(MeetingMeta())
    mid = meta.meeting_id
    chain = {
        MeetingState.CREATED: [],
        MeetingState.RECORDING: [MeetingState.RECORDING],
        MeetingState.FINALIZING: [MeetingState.RECORDING, MeetingState.FINALIZING],
        MeetingState.COMPLETE: [
            MeetingState.RECORDING,
            MeetingState.FINALIZING,
            MeetingState.COMPLETE,
        ],
    }[target]
    for step in chain:
        storage.transition_state(mid, step)
    return mid


class TestReprocessStateTransitions:
    """Reprocess uses the same transition_state() discipline as the
    live recording path.

    The legal transitions are COMPLETE -> REPROCESSING (reprocess
    step 0) and REPROCESSING -> COMPLETE (reprocess step 7, or the
    recovery branch after a crash). Everything else must be rejected
    so a stray call on a recording meeting cannot corrupt meta.state.
    """

    def test_complete_can_enter_reprocessing(self, storage):
        mid = _bring_meeting_to_state(storage, MeetingState.COMPLETE)
        updated = storage.transition_state(mid, MeetingState.REPROCESSING)
        assert updated.state == MeetingState.REPROCESSING
        # Persisted to meta.json
        reread = storage._read_meta(mid)
        assert reread.state == MeetingState.REPROCESSING

    def test_reprocessing_returns_to_complete(self, storage):
        mid = _bring_meeting_to_state(storage, MeetingState.COMPLETE)
        storage.transition_state(mid, MeetingState.REPROCESSING)
        updated = storage.transition_state(mid, MeetingState.COMPLETE)
        assert updated.state == MeetingState.COMPLETE

    def test_recording_cannot_enter_reprocessing(self, storage):
        """A still-recording meeting must never be hijacked into
        reprocessing — that would stomp the open journal and the
        transcript would be irretrievable. transition_state() is
        the only code path that flips state and it must refuse."""
        mid = _bring_meeting_to_state(storage, MeetingState.RECORDING)
        with pytest.raises(ValueError, match="Invalid state transition"):
            storage.transition_state(mid, MeetingState.REPROCESSING)

    def test_reprocessing_cannot_go_interrupted(self, storage):
        mid = _bring_meeting_to_state(storage, MeetingState.COMPLETE)
        storage.transition_state(mid, MeetingState.REPROCESSING)
        with pytest.raises(ValueError, match="Invalid state transition"):
            storage.transition_state(mid, MeetingState.INTERRUPTED)


class TestRecoverInterruptedReprocessing:
    """recover_interrupted flips REPROCESSING -> COMPLETE on startup.

    This is the crash-recovery path: if scribe dies while reprocess
    is running (step 0 -> step 7), meta.state persists as
    REPROCESSING on disk forever unless recover_interrupted un-
    sticks it. The branch must route through transition_state() so
    the recovery uses the same state-machine discipline as every
    other transition in the system.
    """

    @staticmethod
    def _seed_journal(storage: MeetingStorage, mid: str) -> None:
        """recover_interrupted deletes zero-event meetings as a
        cleanup step, so seed a single journal event to keep the
        meeting around for the recovery assertion."""
        journal = Path(storage._meetings_dir) / mid / "journal.jsonl"
        journal.write_text(json.dumps({"segment_id": "s0", "text": "hi", "is_final": True}) + "\n")

    def test_stuck_reprocessing_recovers_to_complete(self, storage):
        mid = _bring_meeting_to_state(storage, MeetingState.COMPLETE)
        self._seed_journal(storage, mid)
        # Simulate the "killed mid-reprocess" snapshot.
        storage.transition_state(mid, MeetingState.REPROCESSING)
        count = storage.recover_interrupted()
        assert count >= 1
        reread = storage._read_meta(mid)
        assert reread.state == MeetingState.COMPLETE

    def test_recover_is_idempotent(self, storage):
        mid = _bring_meeting_to_state(storage, MeetingState.COMPLETE)
        self._seed_journal(storage, mid)
        storage.transition_state(mid, MeetingState.REPROCESSING)
        storage.recover_interrupted()
        # Second call sees state=COMPLETE and does not re-trigger.
        second = storage.recover_interrupted()
        assert second == 0
        reread = storage._read_meta(mid)
        assert reread.state == MeetingState.COMPLETE


class TestReprocessLock:
    """``recover_interrupted`` must not touch a REPROCESSING meeting
    whose ``.reprocess.lock`` names a live process.

    The lock is what tells the recovery branch "this REPROCESSING
    state is NOT a crash artefact — a live process is driving it
    right now." Without this check, a server restart mid-way through
    a CLI ``full-reprocess`` flips the state back to COMPLETE, and
    the CLI's step-7 COMPLETE -> COMPLETE transition then crashes
    the whole run. 2026-04-21 mass-reprocess incident.
    """

    @staticmethod
    def _seed(storage: MeetingStorage, mid: str) -> None:
        journal = Path(storage._meetings_dir) / mid / "journal.jsonl"
        journal.write_text(json.dumps({"segment_id": "s0", "text": "hi", "is_final": True}) + "\n")

    def test_recover_skips_live_lock(self, storage):
        import os as _os

        mid = _bring_meeting_to_state(storage, MeetingState.COMPLETE)
        self._seed(storage, mid)
        storage.transition_state(mid, MeetingState.REPROCESSING)
        # Simulate "reprocess actively running in this process".
        lock_path = Path(storage._meetings_dir) / mid / ".reprocess.lock"
        lock_path.write_text(json.dumps({"pid": _os.getpid(), "started_epoch": 1}))

        count = storage.recover_interrupted()
        assert count == 0, "recover must skip REPROCESSING meetings with a live-PID lock"
        reread = storage._read_meta(mid)
        assert reread.state == MeetingState.REPROCESSING
        assert lock_path.exists(), "live lock must survive the recovery pass"

    def test_recover_unsticks_on_dead_pid(self, storage):
        mid = _bring_meeting_to_state(storage, MeetingState.COMPLETE)
        self._seed(storage, mid)
        storage.transition_state(mid, MeetingState.REPROCESSING)
        # Stale lock: PID that cannot exist on a POSIX system (1 is
        # init; use something guaranteed absent instead).
        lock_path = Path(storage._meetings_dir) / mid / ".reprocess.lock"
        lock_path.write_text(json.dumps({"pid": 2_000_000_000, "started_epoch": 1}))

        count = storage.recover_interrupted()
        assert count >= 1
        reread = storage._read_meta(mid)
        assert reread.state == MeetingState.COMPLETE
        assert not lock_path.exists(), (
            "recovery must remove the stale lock so the next recovery "
            "pass doesn't re-probe the dead PID"
        )

    def test_recover_handles_corrupt_lock(self, storage):
        """A lock with unparseable JSON is treated as stale (not
        active) so a single bad write can't permanently strand a
        meeting."""
        mid = _bring_meeting_to_state(storage, MeetingState.COMPLETE)
        self._seed(storage, mid)
        storage.transition_state(mid, MeetingState.REPROCESSING)
        lock_path = Path(storage._meetings_dir) / mid / ".reprocess.lock"
        lock_path.write_text("not json")

        count = storage.recover_interrupted()
        assert count >= 1
        reread = storage._read_meta(mid)
        assert reread.state == MeetingState.COMPLETE
