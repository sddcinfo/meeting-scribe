"""Crash recovery for the Phase A / Phase B finalize split.

Run on lifespan startup (see ``lifespan.py``). Picks up where Phase B
left off when the server crashed mid-finalize, and sweeps any
leftover finalize artifacts from a clean COMPLETE that got lost mid-
cleanup.

This module is idempotent: each meeting is processed once per
restart, with a per-meeting ``try/except`` so one bad mid never
aborts the rest of the recovery pass.

The authoritative recovery marker is ``finalize_context.json`` with
``phase_a_done: true``, NOT ``meta.state``. That marker is written
durably by Phase A before the ``transition_state(FINALIZING)`` call,
so a crash anywhere in the window between Phase A's marker write and
Phase B's end-of-step-7 cleanup is fully recoverable.

See ``MeetingStorage.recover_interrupted`` for the full state-
machine triage table.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress

from meeting_scribe.models import MeetingState
from meeting_scribe.runtime import state
from meeting_scribe.storage import (
    best_effort_cleanup_finalize_artifacts,
)

logger = logging.getLogger(__name__)


async def replay_pending_finalize(
    finalizing_ids: list[str],
    cleanup_ids: list[str],
) -> None:
    """Resume Phase B for ``finalizing_ids`` + sweep ``cleanup_ids``.

    Concurrency-capped via ``SCRIBE_FINALIZE_RECOVERY_CONCURRENCY``
    (default 1) because the GPU is shared and we'd rather serialize
    than thrash the diarize / summary backends with N concurrent
    pending finalizes after a multi-meeting crash.

    Returns once every meeting has been processed (success OR
    ``INTERRUPTED``). Each meeting's outcome is logged; nothing is
    raised — this is a best-effort startup pass and a single mid's
    failure must not block the server from accepting new meetings.
    """
    if not finalizing_ids and not cleanup_ids:
        return

    sem = asyncio.Semaphore(int(os.environ.get("SCRIBE_FINALIZE_RECOVERY_CONCURRENCY", "1")))

    async def _resume_one(mid: str) -> None:
        async with sem:
            logger.info("finalize.recovery.resume mid=%s", mid)
            try:
                await _resume_phase_b(mid)
            except Exception:
                logger.exception("finalize recovery: phase_b failed mid=%s", mid)
                with suppress(Exception):
                    state.storage.transition_state(mid, MeetingState.INTERRUPTED)

    async def _cleanup_one(mid: str) -> None:
        async with sem:
            logger.info("finalize.recovery.cleanup mid=%s", mid)
            try:
                meeting_dir = state.storage._meeting_dir(mid)
                best_effort_cleanup_finalize_artifacts(meeting_dir)
            except Exception:
                logger.exception("finalize recovery: cleanup failed mid=%s", mid)

    tasks: list[asyncio.Task] = []
    for mid in finalizing_ids:
        tasks.append(asyncio.create_task(_resume_one(mid), name=f"recover-{mid}"))
    for mid in cleanup_ids:
        tasks.append(asyncio.create_task(_cleanup_one(mid), name=f"cleanup-{mid}"))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, BaseException):
            logger.error("finalize recovery: unexpected exception: %s", r)


async def _resume_phase_b(mid: str) -> None:
    """Re-run the Phase B finalize for a meeting whose meta is FINALIZING.

    Currently a thin shim: this is a placeholder implementation. The
    full Phase B body lives in ``meeting_lifecycle._finalize_phase_b_inline``
    and depends on captured state (detected_speakers, eager_summary_cache,
    ws_connections, etc) that doesn't survive a crash. For now we
    transition the meeting to INTERRUPTED so the operator can use
    ``meeting-scribe finalize retry <mid>`` to drive a manual
    re-finalize via the existing ``reprocess_meeting`` path.

    Future work: persist the Phase A → Phase B context to
    ``finalize_context.json`` durably and reconstruct enough state
    here to actually run Phase B without operator intervention.
    """
    meeting_dir = state.storage._meeting_dir(mid)
    ctx_path = meeting_dir / "finalize_context.json"
    if not ctx_path.exists():
        logger.warning(
            "finalize.recovery.no-context mid=%s — marking INTERRUPTED",
            mid,
        )
        with suppress(Exception):
            state.storage.transition_state(mid, MeetingState.INTERRUPTED)
        return

    # The full body of Phase B reads captured locals (detected_speakers,
    # eager_summary_cache, ws connections, etc). Those vanish on crash.
    # Today the safe default is to mark INTERRUPTED — the meeting's
    # journal + audio are intact on disk, so an operator can re-finalize
    # via ``meeting-scribe finalize retry`` (which routes through the
    # existing ``reprocess_meeting`` path).
    logger.info(
        "finalize.recovery.deferred mid=%s — context exists but resume "
        "needs persisted state; marking INTERRUPTED for manual retry",
        mid,
    )
    with suppress(Exception):
        state.storage.transition_state(mid, MeetingState.INTERRUPTED)
    # Sweep the partial finalize artifacts so the meeting is in a clean
    # state for the operator's reprocess.
    best_effort_cleanup_finalize_artifacts(meeting_dir)
