"""Async translation queue with merge gate and backpressure.

Flow: is_final=True events -> Merge Gate -> AsyncQueue -> Workers -> Callback

The merge gate combines adjacent single-speaker segments before translation
when all conditions are satisfied. The queue enforces bounded size, per-segment
cancellation on revision updates, and configurable concurrency.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from meeting_scribe.models import TranscriptEvent, TranslationStatus

if TYPE_CHECKING:
    from meeting_scribe.backends.base import TranslateBackend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Work item
# ---------------------------------------------------------------------------


@dataclass
class _WorkItem:
    """Internal work item tracked by the queue."""

    event: TranscriptEvent
    segment_id: str
    submitted_at: float = field(default_factory=time.monotonic)
    started: bool = False
    cancelled: bool = False


# ---------------------------------------------------------------------------
# Merge gate
# ---------------------------------------------------------------------------

_MERGE_MAX_GAP_MS = 2000  # Adjacent segments must be within 2 seconds


def _can_merge(a: TranscriptEvent, b: TranscriptEvent) -> bool:
    """Determine whether two final segments should be merged before translation.

    Conditions:
    - Both segments are final.
    - Same language.
    - Adjacent in time (gap < 2 seconds).
    - Same speaker (if diarization is available): matching cluster_id or identity.
    """
    if not (a.is_final and b.is_final):
        return False

    # Must be same language (don't merge JA and EN segments).
    if a.language != b.language:
        return False

    # Adjacent in time.
    first, second = (a, b) if a.start_ms <= b.start_ms else (b, a)
    gap_ms = second.start_ms - first.end_ms
    if gap_ms < 0 or gap_ms >= _MERGE_MAX_GAP_MS:
        return False

    # If both have speakers, they must match (by cluster_id or identity).
    if a.speakers and b.speakers:
        sa, sb = a.speakers[0], b.speakers[0]
        if sa.identity and sb.identity:
            if sa.identity != sb.identity:
                return False
        elif sa.cluster_id != sb.cluster_id:
            return False

    return True


def _merge_events(a: TranscriptEvent, b: TranscriptEvent) -> TranscriptEvent:
    """Merge two adjacent segments into one for translation.

    The result keeps the first segment's id/revision/speakers and
    concatenates the text.
    """
    first, second = (a, b) if a.start_ms <= b.start_ms else (b, a)
    return first.model_copy(
        update={
            "text": f"{first.text} {second.text}".strip(),
            "end_ms": second.end_ms,
        },
    )


# ---------------------------------------------------------------------------
# Translation queue
# ---------------------------------------------------------------------------


class TranslationQueue:
    """Bounded async queue that feeds a pool of translation workers.

    Usage::

        queue = TranslationQueue(
            maxsize=50,
            concurrency=2,
            timeout=10.0,
            on_result=my_callback,
        )
        await queue.start(backend)
        ...
        await queue.submit(event)       # from ASR pipeline
        queue.cancel("segment-xyz")     # segment got a new revision
        ...
        await queue.stop()

    Output is callback-based: each completed (or failed/skipped) translation
    invokes *on_result* with the updated ``TranscriptEvent``.
    """

    def __init__(
        self,
        *,
        maxsize: int = 50,
        concurrency: int = 2,
        timeout: float = 10.0,
        on_result: Callable[[TranscriptEvent], Awaitable[None]] | None = None,
    ) -> None:
        self._maxsize = maxsize
        self._concurrency = concurrency
        self._timeout = timeout
        self._on_result = on_result

        self._queue: asyncio.Queue[_WorkItem] = asyncio.Queue(maxsize=0)
        # We manage logical capacity ourselves so we can drop oldest non-started.
        self._items: list[_WorkItem] = []
        self._lock = asyncio.Lock()

        self._workers: list[asyncio.Task[None]] = []
        self._backend: TranslateBackend | None = None
        self._running = False

        # Merge gate: hold the last final event to check for adjacent merge.
        self._pending_merge: TranscriptEvent | None = None
        self._fire_and_forget: set[asyncio.Task] = set()  # prevent GC

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, backend: TranslateBackend) -> None:
        """Start worker tasks consuming from the queue."""
        if self._running:
            return
        self._backend = backend
        self._running = True
        for i in range(self._concurrency):
            task = asyncio.create_task(self._worker(i), name=f"translate-worker-{i}")
            self._workers.append(task)
        logger.info(
            "Translation queue started: maxsize=%d concurrency=%d timeout=%.1fs",
            self._maxsize,
            self._concurrency,
            self._timeout,
        )

    async def stop(self) -> None:
        """Flush the pending merge event, drain the queue, and stop workers."""
        if not self._running:
            return
        self._running = False

        # Flush any held merge candidate.
        await self._flush_pending_merge()

        # Signal workers to exit via sentinel None values.
        for _ in self._workers:
            await self._queue.put(None)  # type: ignore[arg-type]

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("Translation queue stopped")

    # ------------------------------------------------------------------
    # Submit / cancel
    # ------------------------------------------------------------------

    async def submit(self, event: TranscriptEvent) -> None:
        """Submit a final transcript event for translation.

        The merge gate holds each final event briefly: if the next
        submitted event satisfies merge conditions, they are combined
        before entering the queue. Otherwise the held event is queued
        individually.

        Non-final events are ignored.
        """
        if not event.is_final:
            return

        async with self._lock:
            if self._pending_merge is not None:
                prev = self._pending_merge
                if _can_merge(prev, event):
                    merged = _merge_events(prev, event)
                    self._pending_merge = None
                    await self._enqueue(merged)
                    return
                else:
                    # Queue the held event individually.
                    self._pending_merge = None
                    await self._enqueue(prev)

            # Hold the new event as the next merge candidate.
            self._pending_merge = event

    async def flush_merge_gate(self) -> None:
        """Force-flush any event held by the merge gate.

        Useful at end-of-stream or when latency matters more than merging.
        """
        async with self._lock:
            await self._flush_pending_merge()

    def cancel(self, segment_id: str) -> None:
        """Cancel any pending (not yet started) translation for *segment_id*.

        Cancelled items emit a ``status="skipped"`` terminal event when
        a worker picks them up.
        """
        cancelled_count = 0
        for item in self._items:
            if item.segment_id == segment_id and not item.started:
                item.cancelled = True
                cancelled_count += 1
        if cancelled_count:
            logger.debug(
                "Cancelled %d pending item(s) for segment_id=%s",
                cancelled_count,
                segment_id,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _flush_pending_merge(self) -> None:
        """Enqueue any held merge-candidate event (caller must hold _lock)."""
        if self._pending_merge is not None:
            await self._enqueue(self._pending_merge)
            self._pending_merge = None

    async def _enqueue(self, event: TranscriptEvent) -> None:
        """Add a work item, enforcing maxsize by dropping oldest non-started."""
        # Cancel any existing pending items for the same segment_id (revision update).
        for item in self._items:
            if item.segment_id == event.segment_id and not item.started:
                item.cancelled = True

        item = _WorkItem(event=event, segment_id=event.segment_id)

        # Enforce logical capacity.
        while self._logical_size() >= self._maxsize:
            dropped = self._drop_oldest_non_started()
            if dropped is None:
                # All items are started; we cannot drop anything.
                logger.warning(
                    "Queue full (%d items, all started) — forced to exceed maxsize",
                    len(self._items),
                )
                break

        self._items.append(item)
        await self._queue.put(item)

    def _logical_size(self) -> int:
        """Count non-cancelled items still in the queue."""
        return sum(1 for it in self._items if not it.cancelled)

    def is_idle(self) -> bool:
        """Return True if no items are pending or in progress."""
        return self._logical_size() == 0 and self._pending_merge is None

    def _drop_oldest_non_started(self) -> _WorkItem | None:
        """Mark the oldest non-started, non-cancelled item as cancelled and emit skip."""
        for item in self._items:
            if not item.started and not item.cancelled:
                item.cancelled = True
                logger.debug(
                    "Dropped oldest non-started item segment_id=%s",
                    item.segment_id,
                )
                # Emit skip asynchronously (best-effort).
                task = asyncio.ensure_future(self._emit_skip(item))
                self._fire_and_forget.add(task)
                task.add_done_callback(self._fire_and_forget.discard)
                return item
        return None

    async def _emit_skip(self, item: _WorkItem) -> None:
        """Emit a skipped result for a dropped/cancelled work item."""
        if self._on_result is not None:
            skipped = item.event.with_translation(TranslationStatus.SKIPPED)
            try:
                await self._on_result(skipped)
            except Exception:
                logger.exception(
                    "on_result callback failed for skipped segment_id=%s",
                    item.segment_id,
                )

    async def _worker(self, worker_id: int) -> None:
        """Worker loop: pull items from the asyncio queue and translate."""
        logger.debug("Worker %d started", worker_id)
        while True:
            item: _WorkItem | None = await self._queue.get()

            # Sentinel: shut down.
            if item is None:
                self._queue.task_done()
                break

            try:
                await self._process_item(item, worker_id)
            except Exception:
                logger.exception(
                    "Worker %d unhandled error on segment_id=%s",
                    worker_id,
                    item.segment_id,
                )
            finally:
                self._queue.task_done()
                # Remove from tracked items.
                with contextlib.suppress(ValueError):
                    self._items.remove(item)

        logger.debug("Worker %d stopped", worker_id)

    async def _process_item(self, item: _WorkItem, worker_id: int) -> None:
        """Translate a single work item, respecting cancellation and timeout."""
        # Skip if cancelled while waiting in the queue.
        if item.cancelled:
            logger.debug(
                "Worker %d skipping cancelled segment_id=%s",
                worker_id,
                item.segment_id,
            )
            await self._emit_skip(item)
            return

        item.started = True
        event = item.event
        assert self._backend is not None

        # Determine languages — only translate ja↔en
        if event.language not in ("ja", "en"):
            logger.debug("Skipping translation for unsupported language: %s", event.language)
            await self._emit_skip(item)
            return
        target_lang = "en" if event.language == "ja" else "ja"

        # Emit in_progress.
        if self._on_result is not None:
            in_progress = event.with_translation(TranslationStatus.IN_PROGRESS)
            try:
                await self._on_result(in_progress)
            except Exception:
                logger.exception(
                    "on_result callback failed for in_progress segment_id=%s",
                    item.segment_id,
                )

        # Translate with timeout.
        translate_start = time.monotonic()
        try:
            translated_text = await asyncio.wait_for(
                self._backend.translate(
                    text=event.text,
                    source_language=event.language,
                    target_language=target_lang,
                ),
                timeout=self._timeout,
            )
        except TimeoutError:
            logger.warning(
                "Worker %d timeout (%.1fs) on segment_id=%s",
                worker_id,
                self._timeout,
                item.segment_id,
            )
            if self._on_result is not None:
                failed = event.with_translation(TranslationStatus.FAILED)
                await self._on_result(failed)
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Worker %d translation error on segment_id=%s",
                worker_id,
                item.segment_id,
            )
            if self._on_result is not None:
                failed = event.with_translation(TranslationStatus.FAILED)
                await self._on_result(failed)
            return

        # Check cancellation after translation completes (race window).
        if item.cancelled:
            logger.debug(
                "Worker %d: segment_id=%s cancelled during translation, emitting skip",
                worker_id,
                item.segment_id,
            )
            await self._emit_skip(item)
            return

        # Emit done.
        translate_ms = (time.monotonic() - translate_start) * 1000
        logger.info(
            "Worker %d translated segment_id=%s in %.0fms (%d→%d chars)",
            worker_id,
            item.segment_id,
            translate_ms,
            len(event.text),
            len(translated_text),
        )
        if self._on_result is not None:
            done = event.with_translation(TranslationStatus.DONE, text=translated_text)
            try:
                await self._on_result(done)
            except Exception:
                logger.exception(
                    "on_result callback failed for done segment_id=%s",
                    item.segment_id,
                )
