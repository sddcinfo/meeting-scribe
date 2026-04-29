"""Async translation queue with merge gate and backpressure.

Flow: is_final=True events -> Merge Gate -> AsyncQueue -> Workers -> Callback

The merge gate combines adjacent single-speaker segments before translation
when all conditions are satisfied. The queue enforces bounded size, per-segment
cancellation on revision updates, and configurable concurrency.

Monolingual meetings (``len(languages) == 1``): no translation requests are
ever enqueued and no translation events are ever emitted. Client code must
treat ``translation`` as an optional field on ``TranscriptEvent`` / WebSocket
payloads, not a guaranteed one — bilingual meetings also have the absent case
during the pre-translation window.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from meeting_scribe.models import TranscriptEvent, TranslationStatus
from meeting_scribe.util.atomic_io import atomic_append_jsonl, read_jsonl

if TYPE_CHECKING:
    from meeting_scribe.backends.base import TranslateBackend

logger = logging.getLogger(__name__)

# Demand-driven multi-target fan-out. Default ON as of 2026-04 after
# the TranscriptEvent consumer audit + fixes landed:
#   - storage.read_journal_raw now dedups by (segment_id, target_language)
#     so replay streams one line per target.
#   - export._load_events_with_corrections merges per-target translations
#     into a `translations` dict and deduplicates source text.
#   - static/js/segment-store.js accumulates translations[target_lang]
#     while keeping the flat .translation slot populated for legacy
#     render paths.
# Set MEETING_SCRIBE_MULTI_TARGET=0 to temporarily disable (single-
# target legacy path still compiles and tests pass). See also
# ~/.claude/plans/toasty-imagining-badger.md.
_MULTI_TARGET_ENV = os.environ.get("MEETING_SCRIBE_MULTI_TARGET", "").lower()
MULTI_TARGET_ENABLED: bool = _MULTI_TARGET_ENV not in ("0", "false", "no", "off")

# ---------------------------------------------------------------------------
# Work item
# ---------------------------------------------------------------------------


@dataclass
class _WorkItem:
    """Internal work item tracked by the queue.

    Under multi-target mode, ``baseline_targets`` and ``optional_targets``
    are a snapshot of demand taken at enqueue time. Mid-meeting listener
    changes only affect FUTURE segments — a segment already in flight
    keeps its frozen target set so the translation result is reproducible
    and backpressure decisions are stable.

    ``revision_id`` matches ``event.revision`` so cancellation by a new
    ASR revision skips the stale in-flight targets cleanly.
    """

    event: TranscriptEvent
    segment_id: str
    revision_id: int = 0
    # Targets that MUST be translated (meeting language_pair — journal,
    # captions, exports). Dropping a baseline target is a loud failure.
    baseline_targets: frozenset[str] = frozenset()
    # Targets added from live listener demand. Droppable under load.
    optional_targets: frozenset[str] = frozenset()
    submitted_at: float = field(default_factory=time.monotonic)
    started: bool = False
    cancelled: bool = False
    # Generation token captured at enqueue time.  ``bind_epoch``
    # increments on every ``bind_meeting`` / ``clear_meeting`` call —
    # including the dev-reset rebind where ``meeting_id`` stays the
    # same — so a stale item can be detected even when its
    # ``meeting_id`` still matches the currently-active one.  Set at
    # enqueue time inside ``_submit_or_merge`` from the queue's current
    # ``_bind_epoch`` + ``_active_meeting_id``.
    meeting_id: str = ""
    bind_epoch: int = 0


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
# Quiesce result (A4)
# ---------------------------------------------------------------------------


@dataclass
class QuiesceResult:
    """Outcome of ``TranslationQueue.quiesce_meeting()``.

    Carries metadata only — the actual backlog items live in
    ``backlog_path`` on disk. Caller MUST read the file if it needs
    item content; never trust the in-memory list (which doesn't exist).
    """

    drained_clean: bool
    backlog_path: Path
    item_count: int
    deferred_post_quiesce: int = 0


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
        languages: list[str] | tuple[str, ...] = ("en", "ja"),
    ) -> None:
        self._maxsize = maxsize
        self._concurrency = concurrency
        self._timeout = timeout
        self._on_result = on_result
        self._languages: list[str] = list(languages)

        # Active meeting id — set by ``bind_meeting`` (or its A0 alias
        # ``set_active_meeting_id``) from the server's meeting lifecycle
        # hooks.  Used to (a) attribute log rows to the right meeting
        # for the validation harness, and (b) anchor meeting-scoped
        # context history (Phase B1).  ``None`` when no meeting is
        # bound (startup, between meetings, stopped).
        self._active_meeting_id: str | None = None

        # Monotonic generation token.  Incremented on EVERY
        # ``bind_meeting`` / ``clear_meeting`` call — including
        # ``dev_reset``'s bind-to-None-first-then-rebind-same-id flow
        # where ``_active_meeting_id`` does not actually change.
        # Every ``_WorkItem`` is stamped with the current epoch at
        # enqueue time; ``_translate_one`` drops items whose
        # ``bind_epoch`` does not match the current value.  This is
        # the correctness invariant that prevents cross-meeting or
        # cross-reset history leakage.
        self._bind_epoch: int = 0

        # Meeting-scoped rolling history for the live JA→EN translate
        # path (Phase B1).  Keyed by meeting_id first, then by
        # ``(source_lang, target_lang)`` direction.  Keyed-by-meeting
        # structure makes accidental cross-meeting reads structurally
        # impossible — there is no shared dict to leak across.  Each
        # inner deque is bounded by the B1 knob
        # ``ServerConfig.live_translate_context_window_ja_en``.
        from collections import deque as _deque  # local import; dataclass doesn't need it

        self._live_history: dict[str, dict[tuple[str, str], _deque]] = {}
        self._live_history_maxlen: int = 0  # Set by ``bind_meeting`` caller
        # Phase B2 — when True, ``prior_context`` only attaches to
        # utterances that look fragmentary.  Default False for
        # backwards-compat with B1 code paths; gets flipped on per
        # meeting via ``bind_meeting(fragment_gated=True)``.
        self._fragment_gated: bool = False

        self._queue: asyncio.Queue[_WorkItem] = asyncio.Queue(maxsize=0)
        # We manage logical capacity ourselves so we can drop oldest non-started.
        self._items: list[_WorkItem] = []
        self._lock = asyncio.Lock()

        self._workers: list[asyncio.Task[None]] = []
        self._backend: TranslateBackend | None = None
        self._running = False

        # Merge gate: hold the last final event + its demand snapshot to
        # check for adjacent merge. None when empty; otherwise
        # ``(event, baseline_targets, optional_targets)``.
        self._pending_merge: tuple[TranscriptEvent, frozenset[str], frozenset[str]] | None = None
        self._fire_and_forget: set[asyncio.Task] = set()  # prevent GC

        # Intake gate: when paused, ``submit()`` drops incoming final events
        # without enqueuing.  Operator-toggled by ``meeting-scribe
        # pause-translation`` / ``resume-translation`` around model-swap
        # windows (a drain alone doesn't prevent NEW work arriving while
        # the old model is still unloading).  Does NOT affect already-queued
        # work — those drain normally.
        self._paused: bool = False

        # ── Quiesce state (A4) ────────────────────────────────────
        # Once ``_quiesce_active[meeting_id]`` is True, every
        # ``submit()`` for that meeting persists directly to the
        # on-disk ``pending_translations.jsonl`` BEFORE returning,
        # then drops out without entering the live queue. The file
        # is the authoritative store for any not-yet-translated
        # work; in-memory state is never the source of truth once
        # quiesce starts.
        #
        # Quiesce stays active for a meeting forever — by the time
        # ``quiesce_meeting()`` returns, finalize is in progress and
        # any further hypothetical late submits should still go
        # durable. New meetings get fresh entries via the existing
        # ``bind_meeting()`` flow.
        self._quiesce_active: dict[str, bool] = {}
        self._backlog_path: dict[str, Path] = {}
        self._deferred_count: dict[str, int] = {}

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

    async def submit(
        self,
        event: TranscriptEvent,
        *,
        baseline_targets: frozenset[str] | None = None,
        optional_targets: frozenset[str] | None = None,
    ) -> None:
        """Submit a final transcript event for translation.

        The merge gate holds each final event briefly: if the next
        submitted event satisfies merge conditions, they are combined
        before entering the queue. Otherwise the held event is queued
        individually.

        Non-final events are ignored.

        Under multi-target fan-out (``MULTI_TARGET_ENABLED``), callers
        pass a demand snapshot. ``baseline_targets`` are mandatory
        (journal/captions); ``optional_targets`` come from live listener
        prefs and are droppable under backpressure. When omitted the
        queue derives a single baseline target from ``languages`` for
        backwards-compat with the legacy single-target path.
        """
        if not event.is_final:
            return

        if self._paused:
            # Operator-gated intake — used during model-swap windows so
            # new ASR output doesn't pile up while the translation
            # backend is unloading.  Drop silently; the transcript
            # still lands in the journal via its own path.
            logger.debug(
                "Translation queue paused; dropping segment_id=%s",
                event.segment_id,
            )
            return

        if baseline_targets is None and optional_targets is None:
            baseline_targets, optional_targets = self._derive_targets(event)
        baseline_targets = baseline_targets or frozenset()
        optional_targets = (optional_targets or frozenset()) - baseline_targets

        # ── Quiesce hot-path (A4): persist inline before returning ──
        # If this meeting has started finalize, the queue is no longer
        # the source of truth — pending_translations.jsonl is. Persist
        # the work directly under the lock so a process crash between
        # this return and finalize completion still preserves the
        # backlog. The file IS the durable store; we never enter the
        # live queue for quiesced work.
        active_mid = self._active_meeting_id or ""
        if active_mid and self._quiesce_active.get(active_mid):
            async with self._lock:
                if self._quiesce_active.get(active_mid):
                    self._persist_to_backlog(
                        meeting_id=active_mid,
                        event=event,
                        baseline_targets=baseline_targets,
                        optional_targets=optional_targets,
                    )
                    self._deferred_count[active_mid] = self._deferred_count.get(active_mid, 0) + 1
                    logger.info(
                        "translation.persisted_to_backlog meeting=%s seg=%s targets=%d",
                        active_mid,
                        event.segment_id,
                        len(baseline_targets) + len(optional_targets),
                    )
                    return

        async with self._lock:
            if self._pending_merge is not None:
                prev_event, prev_base, prev_opt = self._pending_merge
                if _can_merge(prev_event, event):
                    merged = _merge_events(prev_event, event)
                    self._pending_merge = None
                    # Union demand across merged segments — whichever
                    # listener set was live at either submit still wants
                    # the merged segment's translation.
                    await self._enqueue(
                        merged,
                        baseline_targets=prev_base | baseline_targets,
                        optional_targets=(prev_opt | optional_targets)
                        - (prev_base | baseline_targets),
                    )
                    return
                else:
                    # Queue the held event individually.
                    self._pending_merge = None
                    await self._enqueue(
                        prev_event,
                        baseline_targets=prev_base,
                        optional_targets=prev_opt,
                    )

            # Hold the new event as the next merge candidate.
            self._pending_merge = (event, baseline_targets, optional_targets)

    def _derive_targets(self, event: TranscriptEvent) -> tuple[frozenset[str], frozenset[str]]:
        """Derive a baseline target set from the meeting's languages.

        Used as the default when the caller did not pass an explicit
        demand snapshot. Preserves legacy single-target behavior.
        Monolingual meetings short-circuit here: no request is ever
        enqueued.
        """
        if len(self._languages) == 1:
            return frozenset(), frozenset()
        from meeting_scribe.languages import get_translation_target

        target = get_translation_target(event.language, self._languages)
        if target is None:
            return frozenset(), frozenset()
        return frozenset({target}), frozenset()

    async def flush_merge_gate(self) -> None:
        """Force-flush any event held by the merge gate.

        Useful at end-of-stream or when latency matters more than merging.
        """
        async with self._lock:
            await self._flush_pending_merge()

    def set_languages(self, languages: list[str] | tuple[str, ...]) -> None:
        """Update the meeting's languages (1 = monolingual, 2 = bilingual pair).

        Called at meeting start so live translations honor the meeting's
        configuration instead of the process-wide default loaded at
        server startup. For a monolingual meeting, no translation
        requests will ever be enqueued after this is set.
        """
        self._languages = list(languages)
        logger.info("Translation queue languages set to %s", self._languages)

    def bind_meeting(
        self,
        meeting_id: str | None,
        *,
        history_maxlen: int = 0,
        fragment_gated: bool = False,
    ) -> None:
        """Bind (or unbind) the queue to a specific meeting.

        Side effects:
          * Increments ``_bind_epoch`` — always, even when rebinding to
            the same ``meeting_id``.  This is the generation token
            ``_translate_one`` uses to drop stale items (pre-flight
            and post-await); relying on ``meeting_id`` equality alone
            is insufficient because ``dev_reset_meeting`` keeps the
            same id across the reset.
          * When ``meeting_id`` is not ``None`` and no history dict
            exists yet, creates one with a bounded deque per direction
            sized by ``history_maxlen``.  When ``meeting_id`` is
            ``None``, the dict is left in place (``clear_meeting`` is
            the explicit cleanup API); unbinding is cheap, clearing
            takes a deliberate call.

        Called from ``start_meeting``, ``_stop_meeting_locked`` (unbind
        after flush), and ``dev_reset_meeting`` (bind → clear → rebind
        same id to cycle the epoch).
        """
        from collections import deque

        self._active_meeting_id = meeting_id
        self._bind_epoch += 1
        if history_maxlen and history_maxlen > 0:
            self._live_history_maxlen = history_maxlen
        # Fragment-gating is a per-meeting preference so operators can
        # flip it without a process restart by rebinding (dev-reset,
        # meeting start).  Default stays off — B2 turns it on via
        # config.
        self._fragment_gated = bool(fragment_gated)

        if meeting_id is not None and meeting_id not in self._live_history:
            # Lazily create the meeting's direction map so both JA→EN
            # and EN→JA can accumulate independent histories.  Each
            # inner deque uses the maxlen captured above.
            self._live_history[meeting_id] = {}

        if meeting_id:
            logger.info(
                "Translation queue bound to meeting %s (epoch=%d, history_maxlen=%d)",
                meeting_id,
                self._bind_epoch,
                self._live_history_maxlen,
            )
        else:
            logger.info(
                "Translation queue unbound (no active meeting, epoch=%d)",
                self._bind_epoch,
            )
        # ``deque`` is imported lazily inside the function to keep the
        # top-level import surface small; exactly as inside __init__.
        _ = deque  # silence unused-import lints if any linter cares

    def clear_meeting(self, meeting_id: str) -> None:
        """Drop the per-meeting history dict for ``meeting_id``.

        Also increments ``_bind_epoch`` so any in-flight item stamped
        with the pre-clear epoch is dropped by the post-await check
        inside ``_translate_one`` — if we bumped the epoch only on
        ``bind_meeting`` the clear-then-rebind-same-id flow in
        ``dev_reset_meeting`` would have a one-increment window where
        leaked items could still match.
        """
        self._live_history.pop(meeting_id, None)
        self._bind_epoch += 1
        logger.info(
            "Translation queue cleared meeting %s history (epoch=%d)",
            meeting_id,
            self._bind_epoch,
        )

    # Backwards-compat alias for the A0-era callers.  A0 wired
    # ``set_active_meeting_id`` into start/stop/dev-reset; B1 renames
    # to ``bind_meeting`` + adds the history/epoch side effects.  Keep
    # the old name pointing at the new behaviour so call sites can
    # migrate incrementally.
    def set_active_meeting_id(self, meeting_id: str | None) -> None:
        self.bind_meeting(meeting_id)

    # Phase B2 — fragment gating for the live context window.
    # Short affirmatives ("はい", "そうですね", "OK") don't benefit
    # from context and they are the LRU cache's best customers.
    # Routing context through them just bypasses the cache for no
    # quality win.  When ``live_translate_context_fragment_gated`` is
    # on, we only attach ``prior_context`` to utterances that look
    # fragmentary — short OR ending in a Japanese particle where the
    # ASR likely trailed off mid-sentence.  Keeps the 15-20% cache
    # hit rate on short affirmatives (see
    # ``backends/translate_vllm.py:99-107``).
    _JA_FRAGMENT_PARTICLES: tuple[str, ...] = ("の", "が", "で", "に", "って", "を")

    @classmethod
    def _looks_like_fragment(cls, text: str) -> bool:
        """Cheap heuristic: short OR trails off with a bare particle."""
        t = (text or "").strip()
        if len(t) < 12:
            return True
        return t.endswith(cls._JA_FRAGMENT_PARTICLES)

    def _resolve_prior_context(
        self, source_lang: str, target_lang: str, text: str = ""
    ) -> list[tuple[str, str]] | None:
        """Direction-gated rolling context for the live translate path.

        Returns ``None`` unless ALL of:
          * the queue is bound to an active meeting;
          * direction is JA → EN (the direction with the measured win
            per the 2026-04-19 sweep);
          * the live context-window knob
            ``self._live_history_maxlen`` is > 0;
          * the per-meeting deque for this direction is non-empty
            (OR the refinement pool has same-meeting same-direction
            entries — see B5 below);
          * fragment-gating is either off (``self._fragment_gated`` =
            False) or the utterance looks fragmentary.

        Phase B5: when the refinement worker is running AND it is
        bound to the SAME ``meeting_id`` as the queue, prefer the
        refinement pool's tail over the live-path history dict.
        Refinement's ``_results`` carries higher-quality ASR +
        translations anchored to the same meeting context, and the
        refinement worker trails 45 s behind live so it has a
        materially better view of the meeting than anything the live
        path has produced itself.  The match on ``meeting_id`` is the
        non-negotiable isolation check — identical to B1's bind-epoch
        invariant: never serve a different meeting's history as live
        context.
        """
        if self._active_meeting_id is None:
            return None
        if self._live_history_maxlen <= 0:
            return None
        if (source_lang, target_lang) != ("ja", "en"):
            return None
        if self._fragment_gated and not self._looks_like_fragment(text):
            return None

        refinement_tail = self._refinement_pool_tail(source_lang, target_lang)
        if refinement_tail is not None:
            return refinement_tail

        per_meeting = self._live_history.get(self._active_meeting_id)
        if not per_meeting:
            return None
        deque_ = per_meeting.get((source_lang, target_lang))
        if not deque_:
            return None
        return list(deque_)

    def _refinement_pool_tail(
        self, source_lang: str, target_lang: str
    ) -> list[tuple[str, str]] | None:
        """Return the tail of the refinement worker's pool if — and ONLY if —
        it is bound to the same meeting the queue is bound to.

        Never returns cross-meeting data; never accesses a stale pool.
        ``refinement_worker`` lives on
        ``meeting_scribe.runtime.state``; we import lazily so tests
        that don't stand up the full server can still exercise the
        queue.
        """
        try:
            from meeting_scribe.runtime import state as _state
        except Exception:
            return None
        worker = getattr(_state, "refinement_worker", None)
        if worker is None:
            return None
        if getattr(worker, "_meeting_id", None) != self._active_meeting_id:
            return None
        collected: list[tuple[str, str]] = []
        try:
            results = list(getattr(worker, "_results", []))
        except Exception:
            return None
        for prior in reversed(results):
            if prior.get("language") != source_lang:
                continue
            translation = prior.get("translation")
            if not isinstance(translation, dict):
                continue
            if translation.get("target_language") != target_lang:
                continue
            src = prior.get("text") or ""
            tgt = translation.get("text") or ""
            if src and tgt:
                collected.append((src, tgt))
            if len(collected) >= self._live_history_maxlen:
                break
        if not collected:
            return None
        collected.reverse()
        return collected

    def _append_live_history(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str,
        translated_text: str,
    ) -> None:
        """Record one successful (source, translation) pair into the
        active meeting's direction deque.

        No-op when the live context window is off or the direction is
        not JA → EN — the history is only exercised by the direction
        that reads it, so leaving other directions empty is the cheap
        default.  No-op when the queue is not bound to a meeting.
        """
        from collections import deque

        if self._active_meeting_id is None:
            return
        if self._live_history_maxlen <= 0:
            return
        if (source_lang, target_lang) != ("ja", "en"):
            return
        if not source_text or not translated_text:
            return
        per_meeting = self._live_history.setdefault(self._active_meeting_id, {})
        direction_deque = per_meeting.get((source_lang, target_lang))
        if direction_deque is None:
            direction_deque = deque(maxlen=self._live_history_maxlen)
            per_meeting[(source_lang, target_lang)] = direction_deque
        direction_deque.append((source_text, translated_text))

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
            ev, base, opt = self._pending_merge
            self._pending_merge = None
            await self._enqueue(ev, baseline_targets=base, optional_targets=opt)

    async def _enqueue(
        self,
        event: TranscriptEvent,
        *,
        baseline_targets: frozenset[str] = frozenset(),
        optional_targets: frozenset[str] = frozenset(),
    ) -> None:
        """Add a work item, enforcing maxsize by dropping oldest non-started."""
        # Cancel any existing pending items for the same segment_id (revision update).
        for item in self._items:
            if item.segment_id == event.segment_id and not item.started:
                item.cancelled = True

        item = _WorkItem(
            event=event,
            segment_id=event.segment_id,
            revision_id=event.revision,
            baseline_targets=baseline_targets,
            optional_targets=optional_targets,
            # Stamp with the current bind generation so _translate_one
            # can detect stale items after stop/start or dev_reset.
            meeting_id=self._active_meeting_id or "",
            bind_epoch=self._bind_epoch,
        )

        # Enforce logical capacity. Under multi-target mode, first
        # shed optional targets from pending items (baseline-safe);
        # only drop whole items if still over capacity.
        if MULTI_TARGET_ENABLED and self._logical_size() >= self._maxsize:
            while self._logical_size() >= self._maxsize and self._trim_optional_targets():
                pass
        while self._logical_size() >= self._maxsize:
            dropped = self._drop_oldest_non_started()
            if dropped is None:
                logger.warning(
                    "Queue full (%d items, all started) — forced to exceed maxsize",
                    len(self._items),
                )
                break
            if MULTI_TARGET_ENABLED and dropped.baseline_targets:
                # Loud-error: baseline translation (journal/captions)
                # was sacrificed to backpressure. This is a code-smell
                # signal that producer rate > translator throughput.
                logger.error(
                    "Translation backpressure DROPPED BASELINE segment_id=%s "
                    "baseline=%s optional=%s — journal/captions will miss this segment",
                    dropped.segment_id,
                    sorted(dropped.baseline_targets),
                    sorted(dropped.optional_targets),
                )

        self._items.append(item)
        await self._queue.put(item)

    def _logical_size(self) -> int:
        """Count non-cancelled items still in the queue."""
        return sum(1 for it in self._items if not it.cancelled)

    def is_idle(self) -> bool:
        """Return True if no items are pending or in progress."""
        return self._logical_size() == 0 and self._pending_merge is None

    def active_count(self) -> int:
        """Return the number of items currently being translated (workers busy).

        Used by ``meeting-scribe drain`` to gate the model-unload step:
        the drain must block until every in-flight translation has either
        completed, failed, or been cancelled.
        """
        return sum(1 for it in self._items if it.started and not it.cancelled)

    def pending_count(self) -> int:
        """Return the number of queued items that have not yet started.

        Counts cancelled items too — a cancelled item still has to be
        picked up by a worker to emit its skip callback.  Drain waits
        on that callback before considering the queue empty.
        """
        return sum(1 for it in self._items if not it.started)

    def merge_gate_held(self) -> bool:
        """Return True iff the merge gate is holding a final event.

        The held event hasn't reached the queue yet; drain has to either
        flush it (preferred — see ``flush_merge_gate()``) or wait for the
        next submit to push it through.
        """
        return self._pending_merge is not None

    def is_paused(self) -> bool:
        """Return True iff intake is currently paused."""
        return self._paused

    def pause(self) -> None:
        """Gate new intake.  Idempotent.

        Called by ``meeting-scribe pause-translation`` before a model
        swap so new ASR output doesn't queue behind the about-to-unload
        backend.  Already-queued + in-flight items continue normally
        — use ``drain`` to wait them out.
        """
        if self._paused:
            return
        self._paused = True
        logger.info("Translation queue intake paused")

    def resume(self) -> None:
        """Re-open intake.  Idempotent."""
        if not self._paused:
            return
        self._paused = False
        logger.info("Translation queue intake resumed")

    # ------------------------------------------------------------------
    # Quiesce + persisted backlog (A4)
    # ------------------------------------------------------------------

    def _persist_to_backlog(
        self,
        *,
        meeting_id: str,
        event: TranscriptEvent,
        baseline_targets: frozenset[str],
        optional_targets: frozenset[str],
    ) -> None:
        """Append one record per (segment_id, target_lang) to the backlog file.

        Caller MUST hold ``self._lock``. The file is written via
        ``atomic_append_jsonl`` so each line is durable on disk before
        this returns (fsync per record).

        Idempotency replay key is ``(meeting_id, segment_id, target_lang)``;
        the recovery task will skip items that the journal already shows
        as translated.
        """
        path = self._backlog_path.get(meeting_id)
        if path is None:
            logger.error(
                "Quiesce backlog path missing for meeting=%s — refusing to persist",
                meeting_id,
            )
            return
        all_targets = baseline_targets | optional_targets
        if not all_targets:
            return
        for target in sorted(all_targets):
            atomic_append_jsonl(
                path,
                {
                    "meeting_id": meeting_id,
                    "segment_id": event.segment_id,
                    "target_lang": target,
                    "source_text": event.text,
                    "source_lang": event.language,
                    "is_baseline": target in baseline_targets,
                    "queued_at": time.time(),
                    "attempt_count": 0,
                },
            )

    async def quiesce_meeting(
        self,
        meeting_id: str,
        meeting_dir: Path,
        *,
        deadline_s: float = 30.0,
    ) -> QuiesceResult:
        """Quiesce a meeting's translation work to a durable backlog file.

        The contract:
          1. Flips ``_quiesce_active[meeting_id] = True`` under the lock.
             From this instant, every ``submit()`` for this meeting
             persists inline to ``meeting_dir/pending_translations.jsonl``
             before returning — the file is the authoritative store.
          2. Releases the lock so workers can drain in-flight items
             into the journal normally.
          3. Polls (under the lock) until either no item with
             ``meeting_id == meeting_id and started and not cancelled``
             remains, or ``deadline_s`` elapses.
          4. Re-acquires the lock and persists every still-queued item
             (and any cancelled-but-incomplete in-flight) to the same
             backlog file. Marks them cancelled so workers skip them.
          5. Quiesce stays ACTIVE for this meeting forever — by the
             time we return, finalize is in progress; any further late
             submits should still go durable. New meetings get fresh
             state via ``bind_meeting()``.

        Returns ``QuiesceResult`` with file path + line count. The file
        IS the truth; never trust an in-memory snapshot.
        """
        backlog_path = meeting_dir / "pending_translations.jsonl"

        async with self._lock:
            self._backlog_path[meeting_id] = backlog_path
            self._quiesce_active[meeting_id] = True
            self._deferred_count.setdefault(meeting_id, 0)

            # Flush merge gate while we have the lock so its held event
            # doesn't escape into the live queue post-quiesce.
            if self._pending_merge is not None:
                ev, base, opt = self._pending_merge
                self._pending_merge = None
                if (self._active_meeting_id or "") == meeting_id:
                    self._persist_to_backlog(
                        meeting_id=meeting_id,
                        event=ev,
                        baseline_targets=base,
                        optional_targets=opt,
                    )
                else:
                    # Different meeting was bound; re-enqueue normally.
                    await self._enqueue(
                        ev,
                        baseline_targets=base,
                        optional_targets=opt,
                    )

        # Wait for in-flight workers to drain (they may complete and
        # write to the journal — which we want — or still be running).
        deadline = time.monotonic() + deadline_s
        while True:
            async with self._lock:
                in_flight = sum(
                    1
                    for it in self._items
                    if it.meeting_id == meeting_id and it.started and not it.cancelled
                )
            if in_flight == 0:
                break
            if time.monotonic() >= deadline:
                logger.warning(
                    "Quiesce deadline reached meeting=%s with %d in-flight — "
                    "items will be persisted to backlog",
                    meeting_id,
                    in_flight,
                )
                break
            await asyncio.sleep(0.1)

        # Final pass: persist remaining queued + cancel any in-flight
        # that didn't complete in time. They will be replayed on
        # recovery via the (segment_id, target_lang) idempotency key.
        async with self._lock:
            remaining = [it for it in self._items if it.meeting_id == meeting_id]
            for item in remaining:
                if item.cancelled:
                    continue
                # Persist regardless of started: a started-but-not-done
                # item is racing the deadline; recovery will skip it
                # if the journal already has the result.
                self._persist_to_backlog(
                    meeting_id=meeting_id,
                    event=item.event,
                    baseline_targets=item.baseline_targets,
                    optional_targets=item.optional_targets,
                )
                item.cancelled = True

        item_count = len(read_jsonl(backlog_path))
        deferred = self._deferred_count.get(meeting_id, 0)
        return QuiesceResult(
            drained_clean=item_count == 0,
            backlog_path=backlog_path,
            item_count=item_count,
            deferred_post_quiesce=deferred,
        )

    def cancel_all(self) -> int:
        """Force-cancel every not-yet-started item in the queue.

        Used by ``meeting-scribe drain --force`` when the operator chooses
        to drop in-flight work rather than wait the full timeout.
        Returns the number of items cancelled.  Items already being
        translated by a worker keep running to their natural end — the
        timeout on the HTTP request to the backend is the outer bound.
        """
        cancelled = 0
        for item in self._items:
            if not item.started and not item.cancelled:
                item.cancelled = True
                cancelled += 1
        if cancelled:
            logger.warning(
                "Force-cancelled %d pending translation item(s) (drain --force)",
                cancelled,
            )
        return cancelled

    def _trim_optional_targets(self) -> bool:
        """Trim one optional target from the oldest non-started item.

        Returns True iff a target was actually trimmed. Multi-target
        backpressure prefers this over dropping whole items — baseline
        targets stay, optional targets bleed off first.
        """
        for item in self._items:
            if item.started or item.cancelled:
                continue
            if not item.optional_targets:
                continue
            trimmed = next(iter(sorted(item.optional_targets)))
            item.optional_targets = item.optional_targets - {trimmed}
            logger.info(
                "Translation backpressure trimmed optional target=%s from segment_id=%s",
                trimmed,
                item.segment_id,
            )
            return True
        return False

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
            skipped = item.event.with_translation(TranslationStatus.SKIPPED, target_language="")
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
        """Translate a single work item, respecting cancellation and timeout.

        Under multi-target mode, translates once per target language in
        ``item.baseline_targets ∪ item.optional_targets`` and emits a
        callback per target. Cancellation is rechecked between targets
        so a mid-flight revision bump can short-circuit the rest of the
        fan-out. Under legacy mode, translates a single target derived
        from the meeting language pair.
        """
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

        if MULTI_TARGET_ENABLED:
            targets = sorted(item.baseline_targets | item.optional_targets)
            if not targets:
                logger.debug(
                    "Skipping translation — no targets for segment_id=%s lang=%s",
                    item.segment_id,
                    event.language,
                )
                await self._emit_skip(item)
                return
        else:
            from meeting_scribe.languages import get_translation_target

            legacy_target = get_translation_target(event.language, self._languages)
            if legacy_target is None:
                logger.debug(
                    "Skipping translation — language '%s' not in languages %s",
                    event.language,
                    self._languages,
                )
                await self._emit_skip(item)
                return
            targets = [legacy_target]

        for target_lang in targets:
            if target_lang == event.language:
                # Source == target: nothing to translate.
                continue
            if item.cancelled:
                logger.debug(
                    "Worker %d mid-fanout cancel segment_id=%s at target=%s",
                    worker_id,
                    item.segment_id,
                    target_lang,
                )
                await self._emit_skip(item)
                return
            await self._translate_one(item, target_lang, worker_id)

    async def _translate_one(self, item: _WorkItem, target_lang: str, worker_id: int) -> None:
        """Translate a single (segment, target_lang) pair and fire the callback.

        Pre-flight and post-await epoch checks drop items whose bind
        generation no longer matches the queue's current one — stale
        because of meeting stop/start or dev_reset.  The pair of
        checks closes the gap between "pulled from the queue but not
        started" (pre-flight) and "already awaiting backend when the
        reset fires" (post-await).
        """
        # Pre-flight generation check — drop cleanly via _emit_skip.
        # Covers items enqueued before a stop/start or dev_reset that
        # reach the worker after the bind boundary.
        if item.bind_epoch != self._bind_epoch:
            logger.debug(
                "Worker %d dropping stale translation for segment_id=%s "
                "(item epoch %d != active %d)",
                worker_id,
                item.segment_id,
                item.bind_epoch,
                self._bind_epoch,
            )
            await self._emit_skip(item)
            return

        event = item.event
        assert self._backend is not None

        if self._on_result is not None:
            in_progress = event.with_translation(
                TranslationStatus.IN_PROGRESS, target_language=target_lang
            )
            try:
                await self._on_result(in_progress)
            except Exception:
                logger.exception(
                    "on_result callback failed for in_progress segment_id=%s target=%s",
                    item.segment_id,
                    target_lang,
                )

        # Compute prior_context for the live path.  Only fires for
        # JA→EN when the B1 knob is > 0 AND the per-meeting deque for
        # this direction is non-empty.  Under B2 fragment-gating, the
        # utterance text must also look fragmentary.  Anchor is the
        # ACTIVE meeting's history dict, never a stale one
        # (stop/dev_reset pops it).
        prior_context = self._resolve_prior_context(event.language, target_lang, event.text)

        # Capture the entry epoch so the post-await check can detect a
        # reset that fired while the backend call was in flight.
        entry_epoch = item.bind_epoch

        translate_start = time.monotonic()
        try:
            translated_text = await asyncio.wait_for(
                self._backend.translate(
                    text=event.text,
                    source_language=event.language,
                    target_language=target_lang,
                    prior_context=prior_context,
                    meeting_id=self._active_meeting_id,
                ),
                timeout=self._timeout,
            )
        except TimeoutError:
            logger.warning(
                "Worker %d timeout (%.1fs) on segment_id=%s target=%s",
                worker_id,
                self._timeout,
                item.segment_id,
                target_lang,
            )
            if self._on_result is not None:
                failed = event.with_translation(
                    TranslationStatus.FAILED, target_language=target_lang
                )
                await self._on_result(failed)
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Worker %d translation error on segment_id=%s target=%s",
                worker_id,
                item.segment_id,
                target_lang,
            )
            if self._on_result is not None:
                failed = event.with_translation(
                    TranslationStatus.FAILED, target_language=target_lang
                )
                await self._on_result(failed)
            return

        if item.cancelled:
            logger.debug(
                "Worker %d: segment_id=%s target=%s cancelled post-translate, skip",
                worker_id,
                item.segment_id,
                target_lang,
            )
            await self._emit_skip(item)
            return

        # Post-await epoch re-check.  Item was valid when the backend
        # call started but could have raced with a dev_reset mid-call
        # (same meeting_id, new epoch); skip the result rather than
        # deliver it or pollute the fresh history.
        if entry_epoch != self._bind_epoch:
            logger.debug(
                "Worker %d dropping completed translation for segment_id=%s "
                "(entry epoch %d → current %d)",
                worker_id,
                item.segment_id,
                entry_epoch,
                self._bind_epoch,
            )
            await self._emit_skip(item)
            return

        translate_ms = (time.monotonic() - translate_start) * 1000
        logger.info(
            "Worker %d translated segment_id=%s target=%s in %.0fms (%d→%d chars)",
            worker_id,
            item.segment_id,
            target_lang,
            translate_ms,
            len(event.text),
            len(translated_text),
        )

        # Append to per-meeting history only after a successful,
        # still-in-generation call — never for timeouts, cancellations,
        # or epoch-stale results.
        self._append_live_history(event.language, target_lang, event.text, translated_text)

        if self._on_result is not None:
            done = event.with_translation(
                TranslationStatus.DONE, text=translated_text, target_language=target_lang
            )
            if done.translation is not None:
                done.translation.completed_at = time.monotonic()
            try:
                await self._on_result(done)
            except Exception:
                logger.exception(
                    "on_result callback failed for done segment_id=%s target=%s",
                    item.segment_id,
                    target_lang,
                )
