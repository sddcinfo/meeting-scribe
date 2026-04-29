"""Refinement-drain registry for post-meeting transcript polish.

When a meeting ends the refinement worker keeps running for a short
window (the "drain") to finish any in-flight ASR re-checks and
language-correction passes. Each drain has a unique ``drain_id`` so
the admin API can poll it for status; the registry is bounded to the
last 32 drains so long-running processes don't leak memory.

The registry is a *list*, not a dict, because ``meeting_id`` can repeat
— a single meeting that's stopped, re-started mid-session (future path),
and stopped again produces two independent drain entries. Keying by
meeting_id alone would overwrite the older entry and lose its
authoritative counter snapshot.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _DrainEntry:
    """One meeting's refinement drain state + counter snapshot.

    Counters are snapshotted twice:
      * at drain kickoff, while the worker is still reachable, so the
        endpoint can serve something even if ``_drain_refinement``
        hasn't run yet;
      * after ``worker.stop()`` returns, so the final values include
        any calls that landed during ``_process_remaining``.
    """

    drain_id: int
    meeting_id: str
    task: asyncio.Task
    state: str  # "draining" | "complete" | "partial" | "failed"
    started_at: float
    finished_at: float | None = None
    error: str | None = None
    translate_calls: int = 0
    asr_calls: int = 0
    errors_at_stop: int = 0


_drain_seq: int = 0
_refinement_drains: list[_DrainEntry] = []
_REFINEMENT_DRAINS_CAP = 32


def _next_drain_id() -> int:
    """Atomically allocate the next drain id and return it.

    Callers used to do ``global _drain_seq; _drain_seq += 1`` from
    inside server.py. Now that the seq lives here, we expose this
    helper so callers don't have to reach into the module's internals
    via attribute mutation.
    """
    global _drain_seq
    _drain_seq += 1
    return _drain_seq


async def _drain_refinement(worker: Any, meeting_id: str, drain_id: int) -> None:
    """Background driver for one refinement drain.

    Looks up its entry by ``drain_id`` (unambiguous — meeting_id can
    repeat), awaits ``worker.stop()`` under a 60 s budget that matches
    the user-facing "post-meeting delay < 60 s" claim in
    ``refinement.py`` module doc, and re-snapshots the worker's
    counters into the entry so post-drain validation reads the final
    numbers (including anything ``_process_remaining`` added).
    """
    entry = next((e for e in _refinement_drains if e.drain_id == drain_id), None)
    if entry is None:
        logger.warning("Refinement drain %d has no registry entry", drain_id)
        return
    # If the worker was backgrounded at meeting start and stop fired
    # before start() finished, the worker's internal _task / _asr_client
    # are still None and worker.stop() would crash on _process_remaining.
    # Cancel the pending start task first; if it hadn't completed yet,
    # the worker never ran a loop and there's nothing to drain.
    start_task = getattr(worker, "_start_task", None)
    if start_task is not None and not start_task.done():
        start_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await start_task
        # Start was cancelled mid-flight — nothing to stop/drain.
        entry.state = "complete"
        entry.finished_at = time.time()
        return
    try:
        await asyncio.wait_for(worker.stop(), timeout=60.0)
        entry.state = "complete"
    except TimeoutError:
        entry.state = "partial"
        entry.error = "drain exceeded 60s budget"
        logger.warning(
            "Refinement drain for meeting %s exceeded 60s budget (partial)",
            meeting_id,
        )
    except Exception as exc:
        entry.state = "failed"
        entry.error = f"{type(exc).__name__}: {exc}"
        logger.exception("Refinement drain for meeting %s failed", meeting_id)
    finally:
        entry.finished_at = time.time()
        entry.translate_calls = getattr(worker, "translate_call_count", 0)
        entry.asr_calls = getattr(worker, "asr_call_count", 0)
        entry.errors_at_stop = getattr(worker, "last_error_count", 0)


def _evict_completed_drains(limit: int = _REFINEMENT_DRAINS_CAP) -> None:
    """Drop oldest completed entries so the list stays bounded."""
    while len(_refinement_drains) > limit:
        for i, e in enumerate(_refinement_drains):
            if e.state != "draining":
                _refinement_drains.pop(i)
                break
        else:
            # All entries still draining — leave them; we can't evict
            # in-flight drains without losing their task handle.
            break


def _find_drains_by_meeting(meeting_id: str) -> list[_DrainEntry]:
    return [e for e in _refinement_drains if e.meeting_id == meeting_id]


def _drain_entry_to_dict(entry: _DrainEntry) -> dict:
    return {
        "drain_id": entry.drain_id,
        "meeting_id": entry.meeting_id,
        "state": entry.state,
        "started_at": entry.started_at,
        "finished_at": entry.finished_at,
        "error": entry.error,
        "translate_calls": entry.translate_calls,
        "asr_calls": entry.asr_calls,
        "errors_at_stop": entry.errors_at_stop,
    }
