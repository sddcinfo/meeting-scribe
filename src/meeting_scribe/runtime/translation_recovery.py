"""Replay translation backlogs left by a partial finalize (A4).

When ``_stop_meeting_locked`` calls ``translation_queue.quiesce_meeting``
and the queue had non-empty pending/in-flight work at the deadline,
those items are persisted to ``meeting_dir/pending_translations.jsonl``.

This module's startup task scans for those files and replays each item
that the journal does NOT already have a translation for, then
regenerates the dependent finalize artifacts (summary + status). Once
all items succeed, the file is deleted and the meeting's
``summary.status.json`` is rewritten with ``partial: false``.

Idempotency: the replay key is ``(segment_id, target_lang)``. If
``journal.jsonl`` already contains a translation event matching that
key, we skip the re-enqueue. The journal is the canonical source.

Failure: if any item fails to translate during recovery (backend down,
timeout), the backlog file is LEFT IN PLACE and ``attempt_count`` is
incremented in each surviving line. The next startup will retry. This
is conservative — better to keep retrying than silently lose user
content.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from meeting_scribe.runtime import state
from meeting_scribe.util.atomic_io import atomic_append_jsonl, read_jsonl

logger = logging.getLogger(__name__)

BACKLOG_FILENAME = "pending_translations.jsonl"


def _journal_translations(journal_path: Path) -> set[tuple[str, str]]:
    """Return ``{(segment_id, target_lang)}`` for every translation event
    already present in the journal."""
    if not journal_path.exists():
        return set()
    out: set[tuple[str, str]] = set()
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        seg = ev.get("segment_id")
        translation = ev.get("translation") or {}
        target = translation.get("target_language")
        if seg and target and (translation.get("text") or "").strip():
            out.add((seg, target))
    return out


def _meeting_has_pending_translations(meeting_dir: Path) -> bool:
    """True iff a non-empty backlog file exists for this meeting."""
    p = meeting_dir / BACKLOG_FILENAME
    return p.exists() and p.stat().st_size > 0


async def _replay_one_meeting(meeting_dir: Path) -> bool:
    """Replay backlog for a single meeting. Returns True iff fully
    drained AND artifacts were regenerated, False if any items remain."""
    backlog_path = meeting_dir / BACKLOG_FILENAME
    journal_path = meeting_dir / "journal.jsonl"
    meeting_id = meeting_dir.name

    items = read_jsonl(backlog_path)
    if not items:
        # Empty backlog file is the same as no backlog.
        try:
            backlog_path.unlink(missing_ok=True)
        except OSError:
            pass
        return True

    already = _journal_translations(journal_path)
    to_replay = [it for it in items if (it.get("segment_id"), it.get("target_lang")) not in already]
    skipped = len(items) - len(to_replay)
    logger.info(
        "translation_recovery meeting=%s items=%d skipped_already_done=%d to_replay=%d",
        meeting_id,
        len(items),
        skipped,
        len(to_replay),
    )

    if not to_replay:
        # All items already in journal — nothing to do, just delete file.
        try:
            backlog_path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning(
                "translation_recovery: could not delete drained backlog %s: %s",
                backlog_path,
                e,
            )
        return True

    queue = state.translation_queue
    if queue is None or queue._backend is None:  # type: ignore[union-attr]
        logger.warning(
            "translation_recovery: no live queue/backend — leaving backlog for meeting=%s",
            meeting_id,
        )
        return False

    # Replay each item. We bind the queue to this meeting briefly so
    # results land with the correct meeting attribution. If the queue
    # is currently bound to a different meeting (live recording), we
    # skip recovery for now — the user-facing meeting takes priority.
    if queue._active_meeting_id and queue._active_meeting_id != meeting_id:
        logger.info(
            "translation_recovery: queue is bound to live meeting=%s; deferring recovery for meeting=%s",
            queue._active_meeting_id,
            meeting_id,
        )
        return False

    failed: list[dict] = []
    for it in to_replay:
        seg = it.get("segment_id") or ""
        target = it.get("target_lang") or ""
        source_text = it.get("source_text") or ""
        source_lang = it.get("source_lang") or ""
        if not (seg and target and source_text):
            continue
        try:
            translated = await queue._backend.translate(  # type: ignore[union-attr]
                source_text,
                source_language=source_lang,
                target_language=target,
            )
        except Exception as e:
            it["attempt_count"] = int(it.get("attempt_count", 0)) + 1
            failed.append(it)
            logger.warning(
                "translation_recovery.item_failed meeting=%s seg=%s target=%s err=%s",
                meeting_id,
                seg,
                target,
                type(e).__name__,
            )
            continue
        # Append a translation event to the journal so the next
        # _journal_translations() call sees it as done. Mirrors the
        # shape produced by the live path.
        if translated:
            event = {
                "segment_id": seg,
                "is_final": True,
                "language": source_lang,
                "text": source_text,
                "type": "transcript_revision",
                "translation": {
                    "text": translated,
                    "source_language": source_lang,
                    "target_language": target,
                },
            }
            with journal_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    if failed:
        # Rewrite the backlog with only the failed items so the next
        # restart retries them. Use a fresh write (delete + repopulate)
        # since atomic_append only adds.
        try:
            backlog_path.unlink(missing_ok=True)
        except OSError:
            pass
        for it in failed:
            atomic_append_jsonl(backlog_path, it)
        logger.warning(
            "translation_recovery meeting=%s left %d items in backlog for next retry",
            meeting_id,
            len(failed),
        )
        return False

    # All items drained — regenerate dependent artifacts (summary +
    # timeline) so the meeting reflects the now-complete translations.
    try:
        await _regenerate_finalize_artifacts(meeting_dir)
    except Exception as e:
        logger.warning(
            "translation_recovery.regen_failed meeting=%s err=%s — backlog drained but artifacts may be stale",
            meeting_id,
            type(e).__name__,
        )
        # Keep going — the file is gone, but a follow-up reprocess
        # can refresh artifacts.

    try:
        backlog_path.unlink(missing_ok=True)
    except OSError as e:
        logger.warning(
            "translation_recovery: could not delete drained backlog %s: %s",
            backlog_path,
            e,
        )
    logger.info("translation_recovery meeting=%s drained successfully", meeting_id)
    return True


async def _regenerate_finalize_artifacts(meeting_dir: Path) -> None:
    """Rewrite summary + status after recovery so the meeting reflects
    the now-translated segments.

    Uses the same code paths as the route's regenerate branch but
    without the meeting-lifecycle lock (recovery runs at startup, no
    live meeting can race). Writes ``summary.json`` (success) or
    updates ``summary.status.json`` (failure path) — both via existing
    helpers.
    """
    from meeting_scribe.server_support.summary_status import (
        SummaryStatus,
        classify_summary_error,
        next_attempt_id,
        write_status,
    )
    from meeting_scribe.summary import generate_summary

    journal_path = meeting_dir / "journal.jsonl"
    attempt = next_attempt_id(meeting_dir)
    write_status(
        meeting_dir,
        SummaryStatus.GENERATING,
        attempt_id=attempt,
        journal_path=journal_path,
        extra={"recovery": True},
    )
    try:
        summary = await generate_summary(
            meeting_dir,
            vllm_url=state.config.translate_vllm_url,
        )
        if isinstance(summary, dict) and "error" in summary:
            raise RuntimeError(summary["error"])
        write_status(
            meeting_dir,
            SummaryStatus.COMPLETE,
            attempt_id=attempt,
            journal_path=journal_path,
            extra={"partial": False, "recovery": True},
        )
    except Exception as e:
        code = classify_summary_error(e)
        write_status(
            meeting_dir,
            SummaryStatus.ERROR,
            attempt_id=attempt,
            journal_path=journal_path,
            error_code=code,
            extra={"recovery": True},
        )
        raise


async def replay_pending_translations() -> None:
    """Lifespan-startup entry point: scan for backlogs and replay each.

    Sleeps a few seconds first to let the translation backend finish
    warming up — replaying immediately would race the queue's start.
    """
    await asyncio.sleep(5.0)
    if state.storage is None:
        return
    meetings_root: Path = state.config.meetings_dir
    if not meetings_root.exists():
        return

    backlog_meetings: list[Path] = []
    for child in meetings_root.iterdir():
        if not child.is_dir():
            continue
        if _meeting_has_pending_translations(child):
            backlog_meetings.append(child)

    if not backlog_meetings:
        return

    logger.info(
        "translation_recovery: found %d meeting(s) with pending backlog: %s",
        len(backlog_meetings),
        [m.name for m in backlog_meetings],
    )

    for mdir in backlog_meetings:
        try:
            await _replay_one_meeting(mdir)
        except Exception as e:
            logger.exception(
                "translation_recovery: unexpected error on meeting=%s: %s",
                mdir.name,
                e,
            )
