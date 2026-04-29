"""Tests for the meeting start/stop lifecycle mutex.

The server wraps /api/meeting/start and /api/meeting/stop in a single
asyncio.Lock so concurrent UI clicks, browser retries, and auto-recovery
can't race through "create new meeting + open audio writer" twice.
"""

from __future__ import annotations

import asyncio

import pytest

from meeting_scribe.server_support import lifecycle_lock as srv


class TestLifecycleLockSingleton:
    def test_lock_is_lazily_created(self):
        srv._meeting_lifecycle_lock = None
        lock = srv._get_meeting_lifecycle_lock()
        assert isinstance(lock, asyncio.Lock)

    def test_lock_is_shared_across_calls(self):
        srv._meeting_lifecycle_lock = None
        a = srv._get_meeting_lifecycle_lock()
        b = srv._get_meeting_lifecycle_lock()
        assert a is b


class TestLockSerializesConcurrentCriticalSections:
    async def test_serialized_execution(self):
        """Two tasks grabbing the same lock must run one-at-a-time."""
        srv._meeting_lifecycle_lock = None
        lock = srv._get_meeting_lifecycle_lock()
        order: list[str] = []

        async def worker(tag: str):
            async with lock:
                order.append(f"{tag}-enter")
                await asyncio.sleep(0.02)
                order.append(f"{tag}-exit")

        await asyncio.gather(worker("A"), worker("B"), worker("C"))
        # Each worker's enter must be followed immediately by its own exit
        # (i.e. no interleaving). Check pairs are adjacent.
        for i in range(0, len(order), 2):
            tag = order[i].split("-")[0]
            assert order[i] == f"{tag}-enter"
            assert order[i + 1] == f"{tag}-exit"

    async def test_concurrent_start_calls_see_serialized_state(self):
        """Mimics the race the lock was added to fix: two start handlers
        enter the critical section, but only the first gets `created=True`;
        the second sees current_meeting already set and takes the
        idempotent fast-path."""
        srv._meeting_lifecycle_lock = None
        lock = srv._get_meeting_lifecycle_lock()
        shared_state = {"meeting": None, "created_count": 0}

        async def fake_start_locked(tag: str):
            async with lock:
                if shared_state["meeting"] is None:
                    # Simulate "create new meeting" — must run exactly once
                    await asyncio.sleep(0.01)
                    shared_state["meeting"] = f"meeting-by-{tag}"
                    shared_state["created_count"] += 1
                    return {"state": "created", "id": shared_state["meeting"]}
                return {"state": "resumed", "id": shared_state["meeting"]}

        results = await asyncio.gather(
            fake_start_locked("A"),
            fake_start_locked("B"),
            fake_start_locked("C"),
            fake_start_locked("D"),
        )
        created = [r for r in results if r["state"] == "created"]
        resumed = [r for r in results if r["state"] == "resumed"]
        assert len(created) == 1
        assert len(resumed) == 3
        assert shared_state["created_count"] == 1
        # All callers see the same meeting id
        assert {r["id"] for r in results} == {shared_state["meeting"]}

    async def test_start_stop_cannot_interleave(self):
        """A stop must wait for an in-flight start to finish, and vice
        versa. Otherwise the audio writer from the new meeting would
        race a cleanup pass from the old stop."""
        srv._meeting_lifecycle_lock = None
        lock = srv._get_meeting_lifecycle_lock()
        events: list[str] = []

        async def start():
            async with lock:
                events.append("start-begin")
                await asyncio.sleep(0.03)
                events.append("start-end")

        async def stop():
            # Small head-start for start
            await asyncio.sleep(0.005)
            async with lock:
                events.append("stop-begin")
                events.append("stop-end")

        await asyncio.gather(start(), stop())
        assert events == ["start-begin", "start-end", "stop-begin", "stop-end"]


class TestLockReentryAfterCancellation:
    async def test_cancelled_holder_releases_lock(self):
        """If the task holding the lock is cancelled mid-critical-section,
        the lock must release so the next caller can proceed. This is
        real-life behaviour when the client disconnects during start."""
        srv._meeting_lifecycle_lock = None
        lock = srv._get_meeting_lifecycle_lock()
        second_entered = asyncio.Event()

        async def holder():
            async with lock:
                await asyncio.sleep(5)  # will be cancelled

        async def waiter():
            async with lock:
                second_entered.set()

        h = asyncio.create_task(holder())
        await asyncio.sleep(0.01)  # let holder acquire
        w = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)  # waiter now blocked on the lock
        h.cancel()
        with pytest.raises(asyncio.CancelledError):
            await h
        await asyncio.wait_for(second_entered.wait(), timeout=1.0)
        await w
