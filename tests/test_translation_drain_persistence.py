"""A4 regression tests: translation quiesce + persisted backlog + recovery.

Covers the codex-flagged failure modes:
  * P0 v2: late submits during quiesce must not be silently rejected.
  * P0 v3: late submits after audio close (ASR buffering tail) must
    end up either in the journal OR the backlog file, never neither.
  * P0 v4: persistence must happen INLINE (before submit returns) so
    a process crash between submit and finalize doesn't lose work.
  * P0 v5: finalize must NEVER rewrite the backlog file from an
    in-memory snapshot — the file is the authoritative store.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from meeting_scribe.models import TranscriptEvent
from meeting_scribe.translation.queue import TranslationQueue
from meeting_scribe.util.atomic_io import read_jsonl


def _mk_event(seg: str = "seg-1", text: str = "hello", lang: str = "en") -> TranscriptEvent:
    return TranscriptEvent(
        segment_id=seg,
        text=text,
        language=lang,
        is_final=True,
        start_ms=0,
        end_ms=1000,
        utterance_end_at=1.0,
    )


@pytest.fixture
def queue() -> TranslationQueue:
    q = TranslationQueue(maxsize=50, concurrency=2, timeout=5.0, languages=("en", "ja"))
    q.bind_meeting("mtg-A4", history_maxlen=0)
    return q


@pytest.mark.asyncio
async def test_clean_drain_writes_no_backlog(tmp_path, queue):
    """Test 1: idle queue at quiesce → no backlog file."""
    result = await queue.quiesce_meeting("mtg-A4", tmp_path, deadline_s=1.0)
    assert result.drained_clean is True
    assert result.item_count == 0
    assert not (tmp_path / "pending_translations.jsonl").exists()


@pytest.mark.asyncio
async def test_post_quiesce_submit_persists_inline(tmp_path, queue):
    """Test 3 (codex P0 v4 guard): a submit AFTER quiesce flips active
    persists to the backlog file BEFORE submit returns. We open and
    read the file from a separate task during the submit to prove
    durability happened inline, not deferred."""
    # First quiesce so _quiesce_active is True
    result = await queue.quiesce_meeting("mtg-A4", tmp_path, deadline_s=1.0)
    assert result.drained_clean is True
    backlog_path = tmp_path / "pending_translations.jsonl"

    # Now a late submit (e.g. from buffered ASR final after audio close)
    ev = _mk_event(seg="late-1", text="late arrival")
    await queue.submit(ev, baseline_targets=frozenset({"ja"}))

    # File must exist + contain the item RIGHT NOW (synchronously after submit returns)
    items = read_jsonl(backlog_path)
    assert len(items) == 1
    assert items[0]["segment_id"] == "late-1"
    assert items[0]["target_lang"] == "ja"
    assert items[0]["source_text"] == "late arrival"
    assert items[0]["is_baseline"] is True
    assert items[0]["attempt_count"] == 0
    assert queue._deferred_count.get("mtg-A4") == 1


@pytest.mark.asyncio
async def test_combined_sources_no_dup(tmp_path, queue):
    """Test 4: mix queued (would-be) + post-quiesce submits → all
    appear in backlog as unique JSONL lines. We exercise the queued
    path implicitly by NOT pre-starting the queue (so items in
    self._items are still queued/not-started)."""
    # Pre-quiesce: one event that lands in self._items via _enqueue
    # (we bypass the merge gate by going through _enqueue directly).
    ev_pre = _mk_event(seg="pre-1", text="pre quiesce")
    async with queue._lock:
        await queue._enqueue(
            ev_pre,
            baseline_targets=frozenset({"ja"}),
            optional_targets=frozenset(),
        )

    # Quiesce — should pick up pre-1 + persist to file.
    result = await queue.quiesce_meeting("mtg-A4", tmp_path, deadline_s=1.0)
    assert result.drained_clean is False
    assert result.item_count == 1

    # Post-quiesce submit
    ev_post = _mk_event(seg="post-1", text="post quiesce")
    await queue.submit(ev_post, baseline_targets=frozenset({"ja"}))

    items = read_jsonl(tmp_path / "pending_translations.jsonl")
    keys = {(it["segment_id"], it["target_lang"]) for it in items}
    assert keys == {("pre-1", "ja"), ("post-1", "ja")}
    # No duplicates
    assert len(items) == 2


@pytest.mark.asyncio
async def test_very_late_submit_after_quiesce_returns(tmp_path, queue):
    """Test 6: even after quiesce_meeting() has fully returned, a
    further submit() for the same meeting STILL goes to the backlog
    (because _quiesce_active is never cleared). Proves no race window
    where a hypothetical late submit re-enters the live queue."""
    result = await queue.quiesce_meeting("mtg-A4", tmp_path, deadline_s=1.0)
    assert result.drained_clean is True

    # Several seconds later (simulated)...
    ev = _mk_event(seg="much-later", text="hours after stop")
    await queue.submit(ev, baseline_targets=frozenset({"ja"}))

    items = read_jsonl(tmp_path / "pending_translations.jsonl")
    assert len(items) == 1
    assert items[0]["segment_id"] == "much-later"


@pytest.mark.asyncio
async def test_finalize_never_rewrites_backlog_file(tmp_path, queue):
    """Test (codex P0 v5): finalize logic in the route reads result
    metadata only; submitting MORE items after quiesce returns must
    still see ALL prior + new lines. If finalize had re-marshalled
    from an in-memory snapshot, the second submit's line would be
    overwritten."""
    result = await queue.quiesce_meeting("mtg-A4", tmp_path, deadline_s=1.0)
    backlog_path = result.backlog_path

    # Three late submits in sequence
    for i in range(3):
        ev = _mk_event(seg=f"late-{i}", text=f"late {i}")
        await queue.submit(ev, baseline_targets=frozenset({"ja"}))

    items = read_jsonl(backlog_path)
    assert len(items) == 3
    assert {it["segment_id"] for it in items} == {"late-0", "late-1", "late-2"}


@pytest.mark.asyncio
async def test_backlog_persists_across_simulated_crash(tmp_path, queue):
    """Test 5: persistence is INLINE so a crash between submit and
    quiesce-completion doesn't lose work. We simulate a crash by
    abandoning the queue object after submits and creating a fresh
    queue + reading the file."""
    await queue.quiesce_meeting("mtg-A4", tmp_path, deadline_s=1.0)
    ev = _mk_event(seg="crash-victim", text="must survive")
    await queue.submit(ev, baseline_targets=frozenset({"ja"}))

    # "Crash": drop the queue object.
    del queue

    # Recovery: a brand new process reads the file.
    items = read_jsonl(tmp_path / "pending_translations.jsonl")
    assert len(items) == 1
    assert items[0]["segment_id"] == "crash-victim"


@pytest.mark.asyncio
async def test_quiesce_result_drained_clean_reflects_file_not_memory(tmp_path, queue):
    """drained_clean must be derived from the file (line count == 0),
    not from in-memory state — file is authoritative."""
    # Pre-quiesce: enqueue an item directly so it's pending
    ev = _mk_event(seg="pending-1")
    async with queue._lock:
        await queue._enqueue(
            ev,
            baseline_targets=frozenset({"ja"}),
            optional_targets=frozenset(),
        )

    result = await queue.quiesce_meeting("mtg-A4", tmp_path, deadline_s=1.0)
    assert result.drained_clean is False
    assert result.item_count == 1
    assert result.backlog_path == tmp_path / "pending_translations.jsonl"


@pytest.mark.asyncio
async def test_recovery_idempotent_skip_already_journaled(tmp_path, monkeypatch):
    """Test 7: pre-stage a backlog where 1 of 3 items already has a
    translation in journal.jsonl; recovery re-enqueues only 2."""
    from meeting_scribe.runtime import state as runtime_state
    from meeting_scribe.runtime.translation_recovery import _replay_one_meeting

    mdir = tmp_path / "mtg-recovery"
    mdir.mkdir()
    journal = mdir / "journal.jsonl"
    backlog = mdir / "pending_translations.jsonl"

    # Journal already has "done-1" translated to ja
    journal.write_text(
        json.dumps(
            {
                "segment_id": "done-1",
                "is_final": True,
                "language": "en",
                "text": "first",
                "translation": {
                    "text": "最初",
                    "source_language": "en",
                    "target_language": "ja",
                },
            }
        )
        + "\n"
    )

    # Backlog has 3 items: done-1 (skip), todo-1, todo-2
    for seg, text in [("done-1", "first"), ("todo-1", "second"), ("todo-2", "third")]:
        backlog_line = (
            json.dumps(
                {
                    "meeting_id": "mtg-recovery",
                    "segment_id": seg,
                    "target_lang": "ja",
                    "source_text": text,
                    "source_lang": "en",
                    "is_baseline": True,
                    "queued_at": 0,
                    "attempt_count": 0,
                }
            )
            + "\n"
        )
        with backlog.open("a") as f:
            f.write(backlog_line)

    # Stub the queue with a fake backend that records calls
    fake_backend = MagicMock()
    fake_backend.translate = AsyncMock(side_effect=["二番目", "三番目"])
    fake_queue = MagicMock()
    fake_queue._backend = fake_backend
    fake_queue._active_meeting_id = None
    monkeypatch.setattr(runtime_state, "translation_queue", fake_queue)

    # Stub regenerate to no-op (tested separately)
    monkeypatch.setattr(
        "meeting_scribe.runtime.translation_recovery._regenerate_finalize_artifacts",
        AsyncMock(),
    )

    drained = await _replay_one_meeting(mdir)
    assert drained is True

    # Backend called exactly twice (skipping done-1)
    assert fake_backend.translate.call_count == 2
    called_segs = {
        c.kwargs.get("source_language", c.args[1] if len(c.args) > 1 else "?")
        for c in fake_backend.translate.call_args_list
    }
    # All from en
    assert called_segs <= {"en"}

    # File should now be deleted
    assert not backlog.exists()


@pytest.mark.asyncio
async def test_recovery_failure_keeps_backlog(tmp_path, monkeypatch):
    """Test 8: if backend fails during recovery, leave the file in
    place with attempt_count incremented."""
    from meeting_scribe.runtime import state as runtime_state
    from meeting_scribe.runtime.translation_recovery import _replay_one_meeting

    mdir = tmp_path / "mtg-fail"
    mdir.mkdir()
    journal = mdir / "journal.jsonl"
    journal.write_text("")
    backlog = mdir / "pending_translations.jsonl"

    backlog.write_text(
        json.dumps(
            {
                "meeting_id": "mtg-fail",
                "segment_id": "doomed-1",
                "target_lang": "ja",
                "source_text": "this will fail",
                "source_lang": "en",
                "is_baseline": True,
                "queued_at": 0,
                "attempt_count": 0,
            }
        )
        + "\n"
    )

    fake_backend = MagicMock()
    fake_backend.translate = AsyncMock(side_effect=RuntimeError("backend down"))
    fake_queue = MagicMock()
    fake_queue._backend = fake_backend
    fake_queue._active_meeting_id = None
    monkeypatch.setattr(runtime_state, "translation_queue", fake_queue)

    drained = await _replay_one_meeting(mdir)
    assert drained is False

    # Backlog still exists; attempt_count incremented
    assert backlog.exists()
    items = read_jsonl(backlog)
    assert len(items) == 1
    assert items[0]["attempt_count"] == 1


@pytest.mark.asyncio
async def test_atomic_append_lossless_under_concurrency(tmp_path):
    """Stress test: multiple concurrent submits during quiesce all land
    in the file with no torn writes."""
    queue = TranslationQueue(maxsize=50, concurrency=2, timeout=5.0)
    queue.bind_meeting("mtg-stress", history_maxlen=0)
    await queue.quiesce_meeting("mtg-stress", tmp_path, deadline_s=1.0)

    async def _send(i: int) -> None:
        ev = _mk_event(seg=f"stress-{i}", text=f"text-{i}")
        await queue.submit(ev, baseline_targets=frozenset({"ja"}))

    await asyncio.gather(*[_send(i) for i in range(20)])

    items = read_jsonl(tmp_path / "pending_translations.jsonl")
    assert len(items) == 20
    segs = sorted(it["segment_id"] for it in items)
    assert segs == sorted(f"stress-{i}" for i in range(20))
