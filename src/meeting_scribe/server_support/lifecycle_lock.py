"""Singleton ``asyncio.Lock`` that serializes meeting start/stop/cancel.

The UI can fire ``/api/meeting/start`` twice (user double-click,
browser retry, auto-recovery) and without this lock two concurrent
handlers race through the "create new meeting + open audio writer"
path, leaving one meeting orphaned with a dangling audio writer
and a confused WS client. The lock makes start/stop atomic — the
second caller waits for the first to finish and gets the current
meeting state back.

Lazily constructed so the lock binds to the active event loop on
first use; this matters in tests where the event loop is recreated
per test (constructing at module load time would bind to the import
loop and silently misbehave).
"""

from __future__ import annotations

import asyncio

_meeting_lifecycle_lock: asyncio.Lock | None = None


def _get_meeting_lifecycle_lock() -> asyncio.Lock:
    global _meeting_lifecycle_lock
    if _meeting_lifecycle_lock is None:
        _meeting_lifecycle_lock = asyncio.Lock()
    return _meeting_lifecycle_lock
