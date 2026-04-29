"""Async job runner for the slide translation pipeline.

Orchestrates the full flow:
  1. Upload -> sandboxed extract (validate + render originals + extract text)
  2. Activate deck (original PNGs ready -> viewers can see slides)
  3. Translate text via TranslationQueue (slide items, optional-only priority)
  4. Sandboxed reinsert (reinsert translations + render translated slides)

Supports cancel-and-replace: a new upload cancels any in-progress job.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from meeting_scribe.slides.worker import run_partial_translated_render, run_reinsert
from meeting_scribe.util.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)

# Per-deck slide-translation stats written as JSONL.  Consumed by the
# Phase 4a A/B analysis script to compare parse-failure rates between
# the current and prior production translation models (+ the JSON-schema
# variant in 4b).  Path is user-level rather than repo-local so stats
# survive container rebuilds of the scribe service.  Eval runs override
# via the ``slide_stats_dir`` runtime-config knob so artifacts land
# under the privacy-gated shadow root instead of leaking into the
# production path.
_DEFAULT_STATS_DIR = Path.home() / ".local" / "share" / "meeting-scribe"
_STATS_FILENAME = "slide-translation-stats.jsonl"

# Size-adaptive express-batch schedule.  The first few batches are tiny
# so the user sees a translated slide within ~3-5s of starting a deck;
# once the user is past the first visible batches we amortize LO
# startup by grouping 6 slides per call.  Every entry is the number of
# pending slides required to flush one batch — the Nth batch flushed
# uses EXPRESS_BATCH_SCHEDULE[N] if N < len, else EXPRESS_BATCH_STEADY.
EXPRESS_BATCH_SCHEDULE: tuple[int, ...] = (1, 2, 3)
EXPRESS_BATCH_STEADY: int = 6


def express_batch_threshold(batches_fired: int) -> int:
    """Slides required to flush a batch given how many batches have
    already fired. Pure function — safe to unit-test without spinning
    up a SlideJob.
    """
    if batches_fired < len(EXPRESS_BATCH_SCHEDULE):
        return EXPRESS_BATCH_SCHEDULE[batches_fired]
    return EXPRESS_BATCH_STEADY


def _resolve_stats_path() -> Path:
    """Return the current stats JSONL path, honoring runtime_config overrides.

    Read fresh every emission so a SIGHUP mid-eval-run flips the
    destination without a process restart.
    """
    from meeting_scribe import runtime_config as _rc

    override = _rc.get("slide_stats_dir", None)
    base = Path(override).expanduser() if override else _DEFAULT_STATS_DIR
    return base / _STATS_FILENAME


def _emit_deck_stats(*, meeting_id: str, deck_id: str, stats: _DeckStats) -> None:
    """Append one JSON line of deck stats to the slide-translation-stats
    JSONL sink.  Exceptions are swallowed — translation instrumentation
    must never break the slide pipeline.

    The schema here is the contract the Phase 4a A/B analysis reads, so
    field names and units are load-bearing:
        runs_requested  — total LLM input items across the deck
        runs_returned   — items that survived regex + dedup + ID-in-range
        parse_failures  — regex misses, out-of-range IDs, duplicate IDs,
                          empty-string returns
        id_coverage_failures — requested IDs that never appeared + dupes
        slides_total / slides_with_text — denominators for rate metrics
    """
    try:
        stats_path = _resolve_stats_path()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": stats.finished_at or time.time(),
            "started_at": stats.started_at,
            "meeting_id": meeting_id,
            "deck_id": deck_id,
            "model_id": stats.model_id or os.environ.get("SCRIBE_TRANSLATE_VLLM_MODEL", ""),
            "source_lang": stats.source_lang,
            "target_lang": stats.target_lang,
            "slides_total": stats.slides_total,
            "slides_with_text": stats.slides_with_text,
            "runs_requested": stats.runs_requested,
            "runs_returned": stats.runs_returned,
            "parse_failures": stats.parse_failures,
            "id_coverage_failures": stats.id_coverage_failures,
            "used_json_schema": stats.used_json_schema,
            "wall_seconds": ((stats.finished_at or time.time()) - stats.started_at),
        }
        with stats_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        logger.exception("Failed to emit slide-translation-stats for deck %s", deck_id)


class _DeckStats:
    """Mutable per-deck translation stats tracked during a slide job.

    Fields:
        runs_requested: total text runs sent to the LLM across all slides.
        runs_returned: total text runs the response parser reconstructed.
        parse_failures: lines in the LLM response that failed the numbered-
            list regex (``N. translated text``).  Today's parser at
            ``_translate_slide_text`` silently skips these lines, which is
            the #1 source of "slide translation stopped halfway" UX bugs.
        id_coverage_failures: slides where the set of run IDs returned
            didn't exactly match the set of runs requested (dupes, missing,
            hallucinated IDs).  Bumped by the ``slide_use_json_schema``
            path's post-parse validation (Phase 4b).
        slides_total / slides_with_text: denominators for the rate metrics.
        started_at / finished_at: wall-clock bracketing for the deck.
        model_id / source_lang / target_lang: copied from the job context
            so the A/B analysis can key on them.
    """

    __slots__ = (
        "finished_at",
        "id_coverage_failures",
        "model_id",
        "parse_failures",
        "runs_requested",
        "runs_returned",
        "slides_total",
        "slides_with_text",
        "source_lang",
        "started_at",
        "target_lang",
        "used_json_schema",
    )

    def __init__(self) -> None:
        self.runs_requested: int = 0
        self.runs_returned: int = 0
        self.parse_failures: int = 0
        self.id_coverage_failures: int = 0
        self.slides_total: int = 0
        self.slides_with_text: int = 0
        self.started_at: float = time.time()
        self.finished_at: float | None = None
        self.model_id: str = ""
        self.source_lang: str = ""
        self.target_lang: str = ""
        self.used_json_schema: bool = False


class SlideJobRunner:
    """Manages the lifecycle of slide processing jobs for a meeting."""

    def __init__(
        self,
        meetings_dir: Path,
        translate_fn: Callable[..., Awaitable[str | None]],
        broadcast_fn: Callable[[dict], Awaitable[None]],
    ) -> None:
        """
        Args:
            meetings_dir: Base meetings directory (contains {meeting_id}/).
            translate_fn: Async function to translate text. Signature:
                translate_fn(text: str, source_lang: str, target_lang: str,
                             system_prompt: str, max_tokens: int) -> str | None
            broadcast_fn: Async function to broadcast WS events to viewers.
        """
        self._meetings_dir = meetings_dir
        self._translate_fn = translate_fn
        self._broadcast_fn = broadcast_fn

        self._current_task: asyncio.Task | None = None
        self._cancelled = False

        # Per-meeting slide state
        self.active_deck_id: str | None = None
        self.current_slide_index: int = 0
        self._active_meta: dict | None = None

        # Per-deck translation stats.  Keyed by deck_id while a job is
        # in-flight; emitted to the JSONL sink on job completion (or on
        # cancel/failure, so aborted jobs still produce a diagnostic row).
        # See ``_emit_deck_stats()``.
        self._deck_stats: dict[str, _DeckStats] = {}

    def is_running(self) -> bool:
        """Return True iff a slide-processing task is currently in flight.

        Used by ``meeting-scribe drain`` to block the model-unload step
        until the current deck has either finished reinsertion, failed,
        or been cancelled.
        """
        return self._current_task is not None and not self._current_task.done()

    async def cancel_current_job(self) -> bool:
        """Force-cancel the currently-running slide pipeline, if any.

        Returns True iff a job was actually cancelled.  Used by
        ``meeting-scribe drain --force`` when the operator chooses to
        drop in-flight work (e.g., mid-reinsert of a 50-slide deck)
        rather than wait for it to finish.  The partially-produced deck
        stays on disk marked as failed; the user can re-upload later.
        """
        if not (self._current_task and not self._current_task.done()):
            return False
        self._cancelled = True
        self._current_task.cancel()
        try:
            await self._current_task
        except (asyncio.CancelledError, Exception):
            pass
        logger.warning("Force-cancelled in-flight slide pipeline (drain --force)")
        return True

    def in_flight(self) -> dict[str, int | bool | str]:
        """Return a coarse snapshot of in-flight slide work.

        Lightweight enough to call in a polling loop (no locks, no I/O).
        Includes the current deck_id if a job is running, and the per-deck
        translation counters from live stats so drain can show meaningful
        progress.
        """
        if not self.is_running():
            return {"running": False}

        deck_id = self.active_deck_id or ""
        stats = self._deck_stats.get(deck_id)
        return {
            "running": True,
            "deck_id": deck_id,
            "runs_requested": stats.runs_requested if stats else 0,
            "runs_returned": stats.runs_returned if stats else 0,
            "parse_failures": stats.parse_failures if stats else 0,
        }

    @property
    def total_slides(self) -> int:
        if self._active_meta:
            return self._active_meta.get("total_slides", 0)
        return 0

    def _slides_dir(self, meeting_id: str) -> Path:
        return self._meetings_dir / meeting_id / "slides"

    def _deck_dir(self, meeting_id: str, deck_id: str) -> Path:
        return self._slides_dir(meeting_id) / deck_id

    def _active_deck_id_path(self, meeting_id: str) -> Path:
        return self._slides_dir(meeting_id) / "active_deck_id"

    def _read_active_deck_id(self, meeting_id: str) -> str | None:
        p = self._active_deck_id_path(meeting_id)
        if p.exists():
            return p.read_text().strip()
        return None

    def _write_active_deck_id(self, meeting_id: str, deck_id: str) -> None:
        """Atomic write of active_deck_id file."""
        p = self._active_deck_id_path(meeting_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(deck_id)
        tmp.rename(p)

    async def start_job(
        self,
        meeting_id: str,
        pptx_bytes: bytes,
        source_lang: str,
        target_lang: str,
        skip_language_detection: bool = False,
        upload_filename: str = "",
        monolingual: bool = False,
    ) -> str:
        """Start a new slide processing job. Returns the deck_id.

        Multi-deck behavior:
          * Each upload is content-hashed (sha256). If a previous deck for
            this same meeting has the SAME content hash AND is fully
            processed, we reactivate it instead of re-running the
            translation pipeline (cached translation reuse).
          * Old decks are no longer deleted on a new upload — they stay
            on disk so the user can switch back via the decks API.

        ``skip_language_detection`` pins the source/target pair as given
        and bypasses the auto-detect-and-swap logic.

        ``monolingual`` skips the translate / reinsert / render-translated
        stages end-to-end. The pipeline renders the original deck and
        jumps straight to ``complete`` — no translator is ever invoked.
        The flag is persisted on the deck's meta.json so the job still
        skips translation when resumed after a server restart.
        """
        import hashlib

        content_hash = hashlib.sha256(pptx_bytes).hexdigest()

        # Cache hit: a previously-completed deck on this meeting matches
        # the upload byte-for-byte → reactivate it without spending more
        # GPU on translation.
        for existing in self.list_decks(meeting_id):
            if (
                existing.get("content_hash") == content_hash
                and existing.get("stage") == "complete"
                and existing.get("deck_id")
            ):
                cached_id = existing["deck_id"]
                logger.info(
                    "Reusing cached deck %s for meeting %s (content hash match)",
                    cached_id,
                    meeting_id,
                )
                await self._activate_existing_deck(meeting_id, cached_id, existing)
                return cached_id

        deck_id = uuid.uuid4().hex

        # Cancel any in-flight job. We DON'T clear the active deck —
        # the previous deck stays on disk + remains pickable; the new
        # one will become active once it finishes activation.
        if self._current_task and not self._current_task.done():
            self._cancelled = True
            self._current_task.cancel()
            try:
                await self._current_task
            except (asyncio.CancelledError, Exception):
                pass

        self._cancelled = False
        self._current_task = asyncio.create_task(
            self._run_pipeline(
                meeting_id,
                deck_id,
                pptx_bytes,
                source_lang,
                target_lang,
                skip_language_detection=skip_language_detection,
                content_hash=content_hash,
                upload_filename=upload_filename,
                monolingual=monolingual,
            ),
            name=f"slide-job-{deck_id[:8]}",
        )

        return deck_id

    async def _activate_existing_deck(
        self,
        meeting_id: str,
        deck_id: str,
        meta: dict,
    ) -> None:
        """Make a previously-processed deck the active one for viewers.

        No re-translation. Just flips the active_deck_id pointer + tells
        connected viewers to switch to this deck.
        """
        self._write_active_deck_id(meeting_id, deck_id)
        self.active_deck_id = deck_id
        self.current_slide_index = 0
        self._active_meta = meta
        await self._broadcast_fn(
            {
                "type": "slide_deck_changed",
                "deck_id": deck_id,
                "total_slides": meta.get("total_slides", 0),
            }
        )
        await self._broadcast_fn(
            {
                "type": "slide_job_progress",
                "deck_id": deck_id,
                "stage": "complete",
                "progress": None,
                "from_cache": True,
            }
        )

    async def _run_pipeline(
        self,
        meeting_id: str,
        deck_id: str,
        pptx_bytes: bytes,
        source_lang: str,
        target_lang: str,
        skip_language_detection: bool = False,
        content_hash: str | None = None,
        upload_filename: str = "",
        monolingual: bool = False,
    ) -> None:
        """Full pipeline: validate (fast) -> activate -> render + translate.

        For monolingual meetings the translate / reinsert / render-translated
        stages are short-circuited and marked ``skipped`` in the persisted
        meta. The flag is stamped on meta.json at validation so a resumed
        job still skips those stages after a server restart.
        """
        deck_dir = self._deck_dir(meeting_id, deck_id)

        stats = _DeckStats()
        stats.source_lang = source_lang
        stats.target_lang = target_lang
        self._deck_stats[deck_id] = stats

        try:
            # ── Stage 1: Fast validation (< 1 second) ───────────
            await self._broadcast_fn(
                {
                    "type": "slide_job_progress",
                    "deck_id": deck_id,
                    "stage": "validating",
                    "progress": None,
                }
            )

            from meeting_scribe.slides.worker import (
                run_extract_text,
                run_render_originals,
                run_validate,
            )

            meta = await run_validate(pptx_bytes, deck_dir)
            meta["deck_id"] = deck_id

            if self._cancelled:
                self._cleanup_deck(meeting_id, deck_id, was_active=False)
                return

            # Save source PPTX for reinsertion and OnlyOffice
            source_path = deck_dir / "source.pptx"
            if not source_path.exists():
                source_path.write_bytes(pptx_bytes)

            # ── Activate deck IMMEDIATELY after validation ───────
            # Viewers see the deck right away. Slides render progressively.
            self._write_active_deck_id(meeting_id, deck_id)
            self.active_deck_id = deck_id
            self.current_slide_index = 0
            self._active_meta = meta

            await self._broadcast_fn(
                {
                    "type": "slide_deck_changed",
                    "deck_id": deck_id,
                    "total_slides": meta.get("total_slides", 0),
                }
            )

            # Old decks are intentionally NOT deleted — they remain
            # selectable from the deck switcher for the rest of the meeting
            # AND for past-meeting review. The active_deck_id pointer is
            # what changes; everything else stays on disk.
            #
            # Stamp the content_hash on the new deck's meta as soon as we
            # have it so the cache-hit path on subsequent uploads can find
            # this deck before it finishes processing.
            if content_hash or upload_filename or monolingual:
                try:
                    meta_path = deck_dir / "meta.json"
                    if meta_path.exists():
                        m = json.loads(meta_path.read_text())
                        if content_hash:
                            m["content_hash"] = content_hash
                        if upload_filename:
                            m["upload_filename"] = upload_filename
                        if monolingual:
                            # Persist the flag so a resumed job (after a
                            # server restart) still skips the translation
                            # stages. Read back at _run_pipeline reentry.
                            m["monolingual"] = True
                        atomic_write_json(meta_path, m)
                except Exception:
                    logger.debug("Failed to stamp deck metadata", exc_info=True)

            if self._cancelled:
                return

            # ── Instant placeholder: embedded thumbnail (sub-50ms) ─
            # Most PowerPoint/Keynote-saved decks ship with a small preview
            # JPEG at docProps/thumbnail.jpeg. Drop it in as slide_001.png
            # and broadcast a partial-ready event so the popout has something
            # to show before LibreOffice even finishes loading.
            from meeting_scribe.slides.convert import (
                extract_embedded_thumbnail,
                render_first_slide_fast,
            )

            original_dir = deck_dir / "original"
            original_dir.mkdir(parents=True, exist_ok=True)
            thumb_dest = original_dir / "slide_001.png"
            try:
                if extract_embedded_thumbnail(source_path, thumb_dest):
                    logger.info(
                        "Embedded thumbnail used as slide-1 placeholder for deck %s",
                        deck_id,
                    )
                    await self._broadcast_fn(
                        {
                            "type": "slide_partial_ready",
                            "deck_id": deck_id,
                            "kind": "original",
                            "index": 0,
                            "total": meta.get("total_slides", 0),
                            "placeholder": True,
                        }
                    )
            except Exception:
                logger.debug("Thumbnail extraction failed", exc_info=True)

            # ── Express render: real 300 DPI slide-1 via minimal PPTX ─
            # Run a 1-slide LibreOffice render in parallel with the bulk
            # pipeline. ~3-5s cold for any deck size, vs ~25-30s for the
            # full-deck render. Overwrites the placeholder when it lands.
            async def _express_first_slide() -> None:
                if self._cancelled:
                    return
                work = deck_dir / "_express_first"
                ok = await asyncio.to_thread(render_first_slide_fast, source_path, thumb_dest, work)
                if ok and not self._cancelled:
                    await self._broadcast_fn(
                        {
                            "type": "slide_partial_ready",
                            "deck_id": deck_id,
                            "kind": "original",
                            "index": 0,
                            "total": meta.get("total_slides", 0),
                        }
                    )
                # Best-effort cleanup
                if work.exists():
                    shutil.rmtree(work, ignore_errors=True)

            express_first_task = asyncio.create_task(
                _express_first_slide(),
                name=f"express-orig-{deck_id[:8]}",
            )

            # ── Stage 2-3: Render originals + extract text (slow) ─
            await self._broadcast_fn(
                {
                    "type": "slide_job_progress",
                    "deck_id": deck_id,
                    "stage": "rendering_original",
                    "progress": None,
                }
            )

            async def _orig_progress(idx: int, total: int) -> None:
                # Per-slide ready event so popout viewers can refresh JUST
                # the affected slide image as soon as it lands on disk.
                await self._broadcast_fn(
                    {
                        "type": "slide_partial_ready",
                        "deck_id": deck_id,
                        "kind": "original",
                        "index": idx,
                        "total": total,
                    }
                )

            # ── Parallelize: text extraction (~1-2s) finishes long before
            # the LibreOffice bulk render (~25-30s for a 50-slide deck).
            # Kick the render off as a background task and wait on the
            # extraction first so translation can start immediately.
            render_task = asyncio.create_task(
                run_render_originals(deck_dir, progress_broadcast=_orig_progress),
                name=f"render-orig-{deck_id[:8]}",
            )
            try:
                await run_extract_text(deck_dir)
            except Exception:
                logger.exception("Text extraction failed — falling back to render-then-extract")

            # Use whatever total_slides we know already (validation reported
            # it; render hasn't necessarily finished yet). Broadcast a
            # deck_changed event so the popout updates the slide count.
            existing_total = self._active_meta.get("total_slides", 0) if self._active_meta else 0
            await self._broadcast_fn(
                {
                    "type": "slide_deck_changed",
                    "deck_id": deck_id,
                    "total_slides": existing_total,
                }
            )

            if self._cancelled:
                if not render_task.done():
                    render_task.cancel()
                return

            # ── Monolingual short-circuit ────────────────────────
            # No translation → skip translate/reinsert/rendering_translated
            # end-to-end. Wait for the bulk render to finish so the deck
            # is fully renderable, mark the skipped stages, and broadcast
            # ``complete`` directly from here.
            if monolingual:
                try:
                    finished_meta = await render_task
                    finished_meta["deck_id"] = deck_id
                    self._active_meta = finished_meta
                except Exception:
                    logger.exception("Bulk render failed")
                    return
                meta_path = deck_dir / "meta.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                meta["monolingual"] = True
                meta["source_lang"] = source_lang
                meta["target_lang"] = ""
                meta["detected_lang"] = source_lang
                meta["stages"]["translating"] = {"status": "skipped"}
                meta["stages"]["reinserting"] = {"status": "skipped"}
                meta["stages"]["rendering_translated"] = {"status": "skipped"}
                meta["stage"] = "complete"
                atomic_write_json(meta_path, meta)
                await self._broadcast_fn(
                    {
                        "type": "slide_job_progress",
                        "deck_id": deck_id,
                        "stage": "complete",
                        "progress": None,
                    }
                )
                _emit_deck_stats(meeting_id=meeting_id, deck_id=deck_id, stats=stats)
                return

            # ── Stage 4: Translate via TranslationQueue ──────────
            text_extract_path = deck_dir / "text_extract.json"
            if not text_extract_path.exists():
                # Wait for the bulk render to finish (it includes extract in
                # the legacy bundled path). If extraction truly failed, bail.
                try:
                    meta = await render_task
                    meta["deck_id"] = deck_id
                    self._active_meta = meta
                except Exception:
                    logger.exception("Bulk render failed")
                    return
                if not text_extract_path.exists():
                    logger.warning("No text_extract.json — skipping translation")
                    return

            slides_data = json.loads(text_extract_path.read_text())
            all_translations: list[dict] = []
            total_slides = len(slides_data)

            from meeting_scribe.slides.convert import SlideText, TextRun, detect_slide_language

            if skip_language_detection:
                # Caller pinned the language pair (e.g. user override from
                # the UI). Trust it without running detection or swap logic.
                slide_source = source_lang
                slide_target = target_lang
                detected_lang = source_lang
                logger.info(
                    "Slide language pinned by caller: %s→%s (auto-detect skipped)",
                    slide_source,
                    slide_target,
                )
            else:
                # Auto-detect presentation language from extracted text
                detected_slides = [
                    SlideText(
                        index=sd["index"],
                        runs=[
                            TextRun(
                                id=r["id"],
                                slide_index=r["slide_index"],
                                shape_id=r["shape_id"],
                                para_index=0,
                                run_index=0,
                                text=r["text"],
                            )
                            for r in sd.get("runs", [])
                        ],
                    )
                    for sd in slides_data
                ]
                detected_lang = detect_slide_language(detected_slides)
                logger.info(
                    "Detected slide language: %s (meeting pair: %s→%s)",
                    detected_lang,
                    source_lang,
                    target_lang,
                )

                # Pane-by-language alignment: the popout shows lang A on
                # the left and lang B on the right (matches the transcript
                # columns). Each pane needs slides in ITS language. So
                # whichever language the deck is in becomes the source,
                # and we translate to the OTHER language in the pair to
                # populate the other pane.
                #
                #   meeting pair (ja, en) → left=JA right=EN
                #   deck JA  → translate JA→EN  (left=original, right=translated)
                #   deck EN  → translate EN→JA  (left=translated, right=original)
                #   deck ZH  → translate ZH→EN  (audience language; the
                #              JA pane will render the EN PNG as a fallback
                #              since we only have one translation direction)
                slide_source = detected_lang or source_lang
                if slide_source == source_lang:
                    slide_target = target_lang
                elif slide_source == target_lang:
                    slide_target = source_lang
                else:
                    # Deck is in a third language entirely. Translate to
                    # the audience's language; the source-language pane
                    # will fall back to showing the same translated PNGs.
                    slide_target = target_lang
                if slide_source != source_lang or slide_target != target_lang:
                    logger.info(
                        "Slide direction: %s→%s (deck=%s, meeting=%s→%s)",
                        slide_source,
                        slide_target,
                        detected_lang,
                        source_lang,
                        target_lang,
                    )

            # Update meta to translating stage
            meta_path = deck_dir / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
            meta["stage"] = "translating"
            meta["source_lang"] = slide_source
            meta["target_lang"] = slide_target
            meta["detected_lang"] = detected_lang
            meta["stages"]["translating"] = {
                "status": "in_progress",
                "progress": "0/" + str(total_slides),
            }
            atomic_write_json(meta_path, meta)

            await self._broadcast_fn(
                {
                    "type": "slide_job_progress",
                    "deck_id": deck_id,
                    "stage": "translating",
                    "progress": f"0/{total_slides}",
                }
            )

            # Translate slides SEQUENTIALLY — one at a time to avoid
            # overloading the shared vLLM instance during a live meeting.
            # Each slide waits 0.3s between requests to let live translation
            # requests get through first (vLLM priority handles the rest).
            #
            # Per-slide express render across the WHOLE deck, batched.
            # Size-adaptive express batching. Pure-size-6 batches force the
            # user to wait for 6 slides + one LO call (~10-15s) before they
            # see anything. We flush tiny batches (1, 2, 3) first so first-
            # paint is fast, then amortize LO startup by grouping 6 per
            # batch in steady state. For a 50-slide deck this adds ~2
            # extra LO calls vs the flat size-6 schedule in exchange for
            # a first translated slide visible in ~3-5s instead of ~15s.
            # Schedule + threshold helper live at module scope so tests
            # can validate the progression without spinning up a SlideJob.
            #
            # The bulk reinsert+render path is now skipped — express batches
            # cover every slide. If a batch fails for any reason we fall
            # back to a single bulk render at the end as a safety net.
            completed = 0
            express_translations: list[dict] = []  # all translations (with _slide tag)
            express_rendered: set[int] = set()  # slide indices already on disk
            express_pending: list[int] = []  # buffered indices awaiting next batch
            express_tasks: list[asyncio.Task] = []
            # Cap concurrent LibreOffice invocations — sized from
            # ``config.slide_render_parallelism`` (default 4). Bench at
            # ``scripts/slide_batch_bench/bench_parallel.py`` showed 2x
            # wall-clock speedup at parallelism=4 on a 50-slide deck;
            # above 4 returns plateau because font cache + disk IO
            # contend. Serializing everything (the old Lock() path)
            # wastes ~19 of the GB10's 20 cores.
            from meeting_scribe.config import ServerConfig as _ServerConfig

            _parallelism = max(1, int(_ServerConfig.from_env().slide_render_parallelism))
            express_sem = asyncio.Semaphore(_parallelism)
            express_batches_fired = 0  # count of batches dispatched so far

            async def translate_slide(slide_data: dict) -> list[dict]:
                nonlocal completed
                runs = slide_data.get("runs", [])
                if not runs:
                    completed += 1
                    return []

                if self._cancelled:
                    return []

                # Brief yield between slides so live translation gets priority
                await asyncio.sleep(0.3)

                result = await self._translate_slide_text(
                    runs,
                    slide_source,
                    slide_target,
                )
                completed += 1

                await self._broadcast_fn(
                    {
                        "type": "slide_job_progress",
                        "deck_id": deck_id,
                        "stage": "translating",
                        "progress": f"{completed}/{total_slides}",
                    }
                )

                # Update meta progress
                if meta_path.exists():
                    m = json.loads(meta_path.read_text())
                    m["stages"]["translating"]["progress"] = f"{completed}/{total_slides}"
                    atomic_write_json(meta_path, m)

                return result

            async def _express_render_batch(slide_indices: list[int]) -> None:
                """Render a BATCH of translated slides via one LO call.

                Builds a minimal PPTX containing only the requested slide
                positions, runs LibreOffice once, splits the resulting
                PDF into per-slide PNGs. Throttled by ``express_sem``
                (default 4 concurrent) to avoid font-cache + disk-IO
                contention while still using the GB10's extra cores —
                serial renders were leaving ~19 of 20 cores idle.
                """
                if self._cancelled or not slide_indices:
                    return
                async with express_sem:
                    fresh = [i for i in slide_indices if i not in express_rendered]
                    if not fresh:
                        return
                    # Snapshot translations for the targeted slides
                    targets = set(fresh)
                    tr_for_batch = [t for t in express_translations if t.get("_slide") in targets]
                    if not tr_for_batch:
                        return
                    clean = [{"id": t["id"], "translated": t["translated"]} for t in tr_for_batch]
                    try:
                        rendered = await run_partial_translated_render(
                            pptx_bytes,
                            clean,
                            sorted(fresh),
                            deck_dir / "translated",
                        )
                    except Exception:
                        logger.exception("Express batch render failed for slides %s", sorted(fresh))
                        return
                    for idx in rendered:
                        express_rendered.add(idx)
                        await self._broadcast_fn(
                            {
                                "type": "slide_partial_ready",
                                "deck_id": deck_id,
                                "kind": "translated",
                                "index": idx,
                                "total": total_slides,
                            }
                        )

            def _flush_express_batch(force: bool = False) -> None:
                """Fire an express batch render if buffer has reached the
                current threshold (or force=True for the final flush).
                Schedules as a background task so the translation loop
                keeps pulling from vLLM."""
                nonlocal express_batches_fired
                if not express_pending:
                    return
                threshold = express_batch_threshold(express_batches_fired)
                if not force and len(express_pending) < threshold:
                    return
                batch = list(express_pending)
                express_pending.clear()
                express_batches_fired += 1
                express_tasks.append(
                    asyncio.create_task(
                        _express_render_batch(batch),
                        name=f"express-{deck_id[:8]}-batch{len(express_tasks)}",
                    )
                )

            # Sequential per-slide translation. Every translated slide gets
            # buffered into the pending express batch; once the batch hits
            # EXPRESS_BATCH_SIZE, fire it as a background render. Slides
            # stream in continuously instead of waiting for one big render
            # at the end.
            for slide_pos, sd in enumerate(slides_data):
                if self._cancelled:
                    break
                try:
                    result = await translate_slide(sd)
                    all_translations.extend(result)
                    if result:
                        for r in result:
                            express_translations.append({**r, "_slide": sd["index"]})
                        express_pending.append(sd["index"])
                        _flush_express_batch(force=False)
                except Exception as exc:
                    logger.warning("Slide translation error: %s", exc)

            # Final flush — anything left in the buffer goes out as one
            # last batch (probably smaller than EXPRESS_BATCH_SIZE).
            _flush_express_batch(force=True)

            # Wait for all batches to finish so the journal+UI reflect the
            # complete render before we mark the job complete.
            if express_tasks:
                await asyncio.gather(*express_tasks, return_exceptions=True)

            if self._cancelled:
                return

            # Mark translating as done
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
            meta["stages"]["translating"] = {
                "status": "done",
                "progress": f"{total_slides}/{total_slides}",
            }
            atomic_write_json(meta_path, meta)

            if not all_translations:
                logger.info("No translations produced — skipping reinsert")
                meta["stage"] = "complete"
                meta["completed_at"] = datetime.now(UTC).isoformat()
                meta["stages"]["reinserting"] = {"status": "skipped"}
                meta["stages"]["rendering_translated"] = {"status": "skipped"}
                atomic_write_json(meta_path, meta)
                # Don't strand the bulk-render task on the early return.
                try:
                    await render_task
                except Exception:
                    pass
                return

            # ── Stage 5-6: Sandboxed reinsert + render (FALLBACK) ─
            # Express batches above already cover every successfully
            # translated slide. The bulk reinsert+render only runs as a
            # safety net when one or more batches failed (some indices
            # missing from express_rendered).
            translated_slide_indices: set[int] = {
                s for t in express_translations if isinstance((s := t.get("_slide")), int)
            }
            missing = sorted(translated_slide_indices - express_rendered)

            # Always make sure the original render task has finished before
            # we exit (its PDF output is what the popout serves). Doesn't
            # block translated slide visibility — those are already on disk.
            try:
                render_meta = await render_task
                if render_meta:
                    render_meta["deck_id"] = deck_id
                    self._active_meta = render_meta
            except Exception:
                logger.exception("Bulk render of originals failed (translation continued)")

            if not missing:
                # Express covered everything. Mark stages done and bail.
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                meta["stage"] = "complete"
                meta["completed_at"] = datetime.now(UTC).isoformat()
                meta["stages"]["reinserting"] = {"status": "skipped_express"}
                meta["stages"]["rendering_translated"] = {
                    "status": "done",
                    "progress": f"{len(express_rendered)}/{total_slides}",
                }
                atomic_write_json(meta_path, meta)
                await self._broadcast_fn(
                    {
                        "type": "slide_job_progress",
                        "deck_id": deck_id,
                        "stage": "complete",
                        "progress": None,
                    }
                )
                logger.info(
                    "Slide job %s completed via express batches (%d/%d slides)",
                    deck_id,
                    len(express_rendered),
                    total_slides,
                )
                return

            logger.warning(
                "Express batches missed %d slide(s) %s — running bulk reinsert as fallback",
                len(missing),
                missing,
            )
            await self._broadcast_fn(
                {
                    "type": "slide_job_progress",
                    "deck_id": deck_id,
                    "stage": "reinserting",
                    "progress": None,
                }
            )

            async def _trans_progress(idx: int, total: int) -> None:
                # Skip slides already broadcast by the express lane to avoid
                # redundant client refreshes (the PNGs are equivalent).
                if idx in express_rendered:
                    return
                await self._broadcast_fn(
                    {
                        "type": "slide_partial_ready",
                        "deck_id": deck_id,
                        "kind": "translated",
                        "index": idx,
                        "total": total,
                    }
                )

            meta = await run_reinsert(
                pptx_bytes,
                all_translations,
                deck_dir,
                progress_broadcast=_trans_progress,
            )
            self._active_meta = meta

            # Make sure no express-render tasks are still pending (they may
            # have been overtaken by the bulk render, which is fine).
            if express_tasks:
                for t in express_tasks:
                    if not t.done():
                        t.cancel()
            if express_first_task and not express_first_task.done():
                express_first_task.cancel()

            await self._broadcast_fn(
                {
                    "type": "slide_job_progress",
                    "deck_id": deck_id,
                    "stage": "complete",
                    "progress": None,
                }
            )

            logger.info("Slide job %s completed successfully", deck_id)

        except asyncio.CancelledError:
            logger.info("Slide job %s cancelled", deck_id)
            # Only clean up if this deck was never activated
            if self.active_deck_id != deck_id:
                self._cleanup_deck(meeting_id, deck_id, was_active=False)
            raise

        except Exception:
            logger.exception("Slide job %s failed", deck_id)
            # Update meta with error
            meta_path = deck_dir / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    meta["error"] = "Pipeline failed (see server logs)"
                    atomic_write_json(meta_path, meta)
                except Exception:
                    pass

        finally:
            # Flush per-deck stats regardless of completion / cancel /
            # failure so drain + Phase 4a analysis always see an exit
            # row.  Pull slides_total from the meta if available.
            final_stats = self._deck_stats.pop(deck_id) if deck_id in self._deck_stats else None
            if final_stats is not None:
                final_stats.finished_at = time.time()
                try:
                    meta_path = deck_dir / "meta.json"
                    if meta_path.exists():
                        meta_doc = json.loads(meta_path.read_text())
                        final_stats.slides_total = int(meta_doc.get("total_slides", 0) or 0)
                except Exception:
                    pass
                _emit_deck_stats(meeting_id=meeting_id, deck_id=deck_id, stats=final_stats)

    async def _translate_slide_text(
        self,
        runs: list[dict],
        source_lang: str,
        target_lang: str,
    ) -> list[dict]:
        """Translate a slide's text in ONE vLLM call.

        Concatenates all text runs with numbered markers, translates as a
        single block, then splits back. This is 1 vLLM call per slide
        instead of N calls per run.
        """
        if not runs:
            return []

        stats = self._deck_stats.get(self.active_deck_id or "")
        if stats is not None:
            stats.runs_requested += len(runs)
            stats.slides_with_text += 1

        # Build numbered text block: "1. First text\n2. Second text\n..."
        lines = [f"{i + 1}. {r['text']}" for i, r in enumerate(runs)]
        combined = "\n".join(lines)

        system_prompt = (
            f"Translate the following numbered text items from {source_lang} to {target_lang}. "
            f"Keep the same numbering format. Return ONLY the translated numbered list, "
            f"one item per line. Keep it concise — this is for presentation slides."
        )

        result_text = await self._translate_fn(
            combined,
            source_lang,
            target_lang,
            system_prompt=system_prompt,
            max_tokens=min(1024, 64 * len(runs)),
        )

        if not result_text:
            # Empty model response — every run was effectively a parse
            # failure.  Log WARNING so operators see it at service-log
            # level (previously silent at line 565-572).
            if stats is not None:
                stats.parse_failures += len(runs)
            logger.warning(
                "Slide translation empty response: %d runs requested, 0 returned (%s→%s)",
                len(runs),
                source_lang,
                target_lang,
            )
            return []

        # Parse numbered response back to individual translations
        results: list[dict] = []
        response_lines = result_text.strip().split("\n")
        returned_ids: set[int] = set()

        import re

        for line in response_lines:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"(\d+)[.)]\s*(.*)", line)
            if match:
                idx = int(match.group(1)) - 1
                translated = match.group(2).strip()
                if 0 <= idx < len(runs) and translated:
                    if idx in returned_ids:
                        # Duplicate ID from the model — count as an
                        # ID-coverage failure.  Keep the first occurrence
                        # so downstream behavior is deterministic.
                        if stats is not None:
                            stats.id_coverage_failures += 1
                        continue
                    returned_ids.add(idx)
                    results.append({"id": runs[idx]["id"], "translated": translated})
                else:
                    # Out-of-range index (hallucinated ID) or empty text.
                    if stats is not None:
                        stats.parse_failures += 1
            else:
                # Regex miss — the line wasn't in numbered-list format.
                # This is the single largest source of silent slide skips
                # on the 3.5 production model.
                if stats is not None:
                    stats.parse_failures += 1

        # Missing-ID check: every requested run should appear in the
        # response.  Anything missing counts against ID coverage.
        missing = set(range(len(runs))) - returned_ids
        if missing and stats is not None:
            stats.id_coverage_failures += len(missing)

        if stats is not None:
            stats.runs_returned += len(results)

        if missing or (stats is not None and stats.parse_failures):
            # Elevate silent failures to the service log so regressions
            # are visible without grepping the translation-stats JSONL.
            logger.warning(
                "Slide translated with parse/coverage issues: %d/%d runs returned, "
                "%d missing IDs (%s→%s) — see slide-translation-stats.jsonl",
                len(results),
                len(runs),
                len(missing),
                source_lang,
                target_lang,
            )
        else:
            logger.info(
                "Slide translated: %d/%d runs in 1 call",
                len(results),
                len(runs),
            )
        return results

    def advance_slide(self, index: int) -> bool:
        """Set current slide index. Returns True if valid."""
        if index < 0 or index >= self.total_slides:
            return False
        self.current_slide_index = index
        return True

    def clear_active_state(self, meeting_id: str) -> None:
        """Forget the in-memory active deck WITHOUT deleting artifacts.

        Use at the end of a meeting (finalize/stop) — slides become part of
        the meeting record, viewable from the past-meeting view, so we
        keep the on-disk artifacts and just drop the runtime pointer.
        """
        self.active_deck_id = None
        self.current_slide_index = 0
        self._active_meta = None

    def cleanup_meeting(self, meeting_id: str) -> None:
        """Delete ALL slide artifacts for a meeting (destructive).

        Use only when the meeting itself is being cancelled/discarded
        (cancel-empty path, manual delete) — not at normal finalize, where
        slides should persist as part of the meeting record.
        """
        slides_dir = self._slides_dir(meeting_id)
        if slides_dir.exists():
            shutil.rmtree(slides_dir, ignore_errors=True)
            logger.info("Cleaned up slides for meeting %s", meeting_id)
        self.clear_active_state(meeting_id)

    def _cleanup_deck(self, meeting_id: str, deck_id: str, *, was_active: bool) -> None:
        """Clean up a single deck directory."""
        if was_active:
            # Don't delete active deck — wait for replacement
            return
        deck_dir = self._deck_dir(meeting_id, deck_id)
        if deck_dir.exists():
            shutil.rmtree(deck_dir, ignore_errors=True)
            logger.info("Cleaned up cancelled deck %s", deck_id)

    def get_metadata(self, meeting_id: str) -> dict | None:
        """Get current active deck metadata."""
        if not self._ensure_active_deck(meeting_id):
            return None
        assert self.active_deck_id is not None  # _ensure_active_deck postcondition

        deck_dir = self._deck_dir(meeting_id, self.active_deck_id)
        meta_path = deck_dir / "meta.json"
        if not meta_path.exists():
            return None

        try:
            meta = json.loads(meta_path.read_text())
            meta["current_slide_index"] = self.current_slide_index
            return meta
        except Exception:
            return None

    def _ensure_active_deck(self, meeting_id: str) -> bool:
        """Recover active_deck_id from disk if memory is cleared."""
        if not self.active_deck_id:
            disk_id = self._read_active_deck_id(meeting_id)
            if disk_id:
                self.active_deck_id = disk_id
        return bool(self.active_deck_id)

    def get_meeting_deck_id(self, meeting_id: str) -> str | None:
        """Read the active_deck_id file for a specific meeting.

        Unlike ``_ensure_active_deck``, this does NOT mutate the singleton
        ``self.active_deck_id`` — so it's safe to call for past meetings
        without disturbing the live recording's slide state.
        """
        p = self._active_deck_id_path(meeting_id)
        if p.exists():
            try:
                return p.read_text().strip() or None
            except Exception:
                return None
        return None

    def get_slide_image_path(
        self,
        meeting_id: str,
        index: int,
        translated: bool = False,
    ) -> Path | None:
        """Get path to a slide PNG. Works for past meetings too — looks up
        the deck for the SPECIFIC meeting, not the singleton active deck."""
        deck_id = self.get_meeting_deck_id(meeting_id)
        if not deck_id:
            return None
        deck_dir = self._deck_dir(meeting_id, deck_id)
        subdir = "translated" if translated else "original"
        path = deck_dir / subdir / f"slide_{index + 1:03d}.png"
        if path.exists():
            return path
        return None

    def get_original_pdf_path(self, meeting_id: str) -> Path | None:
        """Get path to the original PDF for lossless source viewing."""
        deck_id = self.get_meeting_deck_id(meeting_id)
        if not deck_id:
            return None
        deck_dir = self._deck_dir(meeting_id, deck_id)
        path = deck_dir / "original" / "original.pdf"
        if path.exists():
            return path
        return None

    def get_source_pptx_path(self, meeting_id: str) -> Path | None:
        """Get path to the original uploaded PPTX for OnlyOffice viewer."""
        deck_id = self.get_meeting_deck_id(meeting_id)
        if not deck_id:
            return None
        deck_dir = self._deck_dir(meeting_id, deck_id)
        path = deck_dir / "source.pptx"
        if path.exists():
            return path
        return None

    def get_meeting_deck_meta(self, meeting_id: str) -> dict | None:
        """Read the meta.json for a specific meeting's deck WITHOUT touching
        the singleton active_deck_id. Returns None if no slides exist for
        this meeting. Includes the resolved deck_id for caller convenience.
        """
        deck_id = self.get_meeting_deck_id(meeting_id)
        if not deck_id:
            return None
        meta_path = self._deck_dir(meeting_id, deck_id) / "meta.json"
        if not meta_path.exists():
            return None
        try:
            data = json.loads(meta_path.read_text())
            data["deck_id"] = deck_id
            return data
        except Exception:
            return None

    def list_decks(self, meeting_id: str) -> list[dict]:
        """List every deck on disk for this meeting, newest first.

        Each entry is the deck's meta.json with ``deck_id`` injected
        (the dirname is authoritative — the in-meta deck_id field can
        be stale from the validation phase). Used by the deck switcher
        UI and by the cache-hit path on new uploads.
        """
        slides_root = self._slides_dir(meeting_id)
        if not slides_root.exists():
            return []
        active = self.get_meeting_deck_id(meeting_id)
        out: list[dict] = []
        for child in slides_root.iterdir():
            if not child.is_dir():
                continue
            meta_path = child / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                continue
            meta["deck_id"] = child.name  # dirname is the source of truth
            meta["is_active"] = child.name == active
            out.append(meta)

        # Newest first by completed_at if present, else mtime
        def _sort_key(m: dict) -> str:
            return m.get("completed_at") or m.get("started_at") or ""

        out.sort(key=_sort_key, reverse=True)
        return out

    async def set_active_deck(self, meeting_id: str, deck_id: str) -> dict | None:
        """Make ``deck_id`` the active deck for ``meeting_id``.

        Validates the deck exists on disk. Broadcasts a
        ``slide_deck_changed`` event so connected viewers switch over
        without a manual reload. Returns the activated deck's meta or
        None if the deck wasn't found.
        """
        deck_dir = self._deck_dir(meeting_id, deck_id)
        meta_path = deck_dir / "meta.json"
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            return None
        meta["deck_id"] = deck_id
        await self._activate_existing_deck(meeting_id, deck_id, meta)
        return meta
