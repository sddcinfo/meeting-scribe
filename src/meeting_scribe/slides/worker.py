"""PPTX processing worker — runs conversion in background threads.

Trusted environment — no Docker sandboxing. Processing runs in
asyncio.to_thread() to avoid blocking the event loop.

Split into fast (validate) and slow (render + extract) phases so the
deck can be activated immediately after validation, before rendering.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from meeting_scribe.slides.convert import (
    convert_pptx_to_images,
    extract_text_from_pptx,
    reinsert_translated_text,
    render_translated_to_images,
    validate_pptx_contents,
    write_text_extract,
)
from meeting_scribe.slides.models import SlideMeta, StageProgress, StageStatus

logger = logging.getLogger(__name__)

ProgressBroadcast = Callable[[int, int], Awaitable[None]]
"""Async callable invoked from the event loop with (slide_idx_0based, total)."""


def _make_thread_safe_progress(
    loop: asyncio.AbstractEventLoop,
    broadcast: ProgressBroadcast | None,
):
    """Bridge a sync callback (called in worker thread) onto the event loop."""
    if broadcast is None:
        return None

    async def _invoke(idx: int, total: int) -> None:
        # Wrapper coroutine — run_coroutine_threadsafe wants a concrete
        # Coroutine, not a generic Awaitable (which broadcast's protocol
        # advertises).  The extra await is free.
        await broadcast(idx, total)

    def cb(idx: int, total: int) -> None:
        try:
            asyncio.run_coroutine_threadsafe(_invoke(idx, total), loop)
        except Exception:
            logger.debug("progress broadcast schedule failed", exc_info=True)

    return cb


async def check_worker_available() -> bool:
    """Check if LibreOffice is available for slide processing."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "libreoffice",
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=10)
        return proc.returncode == 0
    except Exception:
        return False


def _atomic_write_json(path: Path, data: dict) -> None:
    """Atomic JSON write via tmp + rename.

    The tmp filename carries a uuid suffix so concurrent writers to the
    same target path (e.g. the extract-text and render-originals phases
    both updating meta.json in parallel) can't race on the same tmp
    file. Without the uuid, thread A's rename completes first and
    thread B's rename then fails with FileNotFoundError — which skipped
    text extraction and silently disabled translation on the 6-page
    deck observed 2026-04-20.
    """
    tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.rename(path)


# ── Phase 1: Fast validation (< 1 second) ───────────────────


def _validate_sync(pptx_bytes: bytes, output_dir: Path) -> dict:
    """Validate PPTX and save input file. Returns meta dict with slide count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "_upload.pptx"
    input_path.write_bytes(pptx_bytes)
    logger.info("Worker input: %s (%d bytes)", input_path, len(pptx_bytes))

    meta = SlideMeta(deck_id="worker")
    meta.stage = "validating"
    meta.stages["validating"].status = StageStatus.IN_PROGRESS
    _atomic_write_json(output_dir / "meta.json", meta.to_dict())

    result = validate_pptx_contents(input_path)
    if not result.valid:
        meta.stages["validating"] = StageProgress(
            status=StageStatus.FAILED,
            error=result.error,
        )
        meta.error = result.error
        _atomic_write_json(output_dir / "meta.json", meta.to_dict())
        raise RuntimeError(result.error)

    meta.total_slides = result.slide_count
    meta.stages["validating"].status = StageStatus.DONE
    _atomic_write_json(output_dir / "meta.json", meta.to_dict())
    logger.info("Validation passed: %d slides", result.slide_count)
    return meta.to_dict()


async def run_validate(pptx_bytes: bytes, output_dir: Path) -> dict:
    """Fast validation — returns immediately with slide count."""
    return await asyncio.to_thread(_validate_sync, pptx_bytes, output_dir)


# ── Phase 2: Slow render + extract (10-60 seconds) ──────────


def _extract_text_sync(output_dir: Path) -> dict:
    """Run python-pptx text extraction only. Fast (~1-2s) — splits out
    from the bulk LibreOffice render so translation can start in parallel
    with that ~25-30s render instead of waiting for it to finish.
    """
    input_path = output_dir / "_upload.pptx"
    meta_path = output_dir / "meta.json"
    meta_dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    meta_dict["stage"] = "extracting_text"
    meta_dict.setdefault("stages", {})["extracting_text"] = {"status": "in_progress"}
    _atomic_write_json(meta_path, meta_dict)

    slides = extract_text_from_pptx(input_path)
    write_text_extract(slides, output_dir / "text_extract.json")
    meta_dict["stages"]["extracting_text"] = {"status": "done"}
    _atomic_write_json(meta_path, meta_dict)

    total_runs = sum(len(s.runs) for s in slides)
    logger.info("Extracted %d text runs from %d slides", total_runs, len(slides))
    return meta_dict


async def run_extract_text(output_dir: Path) -> dict:
    """Async wrapper around the python-pptx text extraction step."""
    return await asyncio.to_thread(_extract_text_sync, output_dir)


def _render_originals_sync(output_dir: Path, progress_cb=None) -> dict:
    """Render originals (LibreOffice + pdftoppm). The slow part — typically
    25-30s for a 50-slide deck. Caller is expected to start text extraction
    + translation in parallel with this.
    """
    input_path = output_dir / "_upload.pptx"
    meta_path = output_dir / "meta.json"
    meta_dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    meta_dict["stage"] = "rendering_original"
    meta_dict.setdefault("stages", {})["rendering_original"] = {"status": "in_progress"}
    _atomic_write_json(meta_path, meta_dict)

    original_dir = output_dir / "original"
    original_dir.mkdir(exist_ok=True)
    count = convert_pptx_to_images(input_path, original_dir, progress_cb=progress_cb)
    meta_dict["total_slides"] = count
    meta_dict["stages"]["rendering_original"] = {
        "status": "done",
        "progress": f"{count}/{count}",
    }
    _atomic_write_json(meta_path, meta_dict)
    logger.info("Rendered %d original slides", count)
    return meta_dict


async def run_render_originals(
    output_dir: Path,
    progress_broadcast: ProgressBroadcast | None = None,
) -> dict:
    """Render originals only. Pair with ``run_extract_text`` (fast) so the
    caller can start translation as soon as text is extracted, in parallel
    with this slow render."""
    loop = asyncio.get_running_loop()
    cb = _make_thread_safe_progress(loop, progress_broadcast)
    return await asyncio.to_thread(_render_originals_sync, output_dir, cb)


# Legacy wrapper — keeps the prior render-then-extract behavior for any
# caller that still expects them bundled. New code should call
# run_extract_text() and run_render_originals() concurrently instead.
def _render_and_extract_sync(output_dir: Path, progress_cb=None) -> dict:
    meta = _render_originals_sync(output_dir, progress_cb)
    extract_meta = _extract_text_sync(output_dir)
    # Latest meta on disk wins; return the merged view
    return {**meta, **extract_meta, "total_slides": meta.get("total_slides", 0)}


async def run_render_and_extract(
    output_dir: Path,
    progress_broadcast: ProgressBroadcast | None = None,
) -> dict:
    """Legacy: render originals + extract text sequentially in one thread.

    Prefer running ``run_extract_text`` and ``run_render_originals`` as
    concurrent tasks — extracting text first lets translation start
    ~25-30s earlier on a 50-slide deck.
    """
    loop = asyncio.get_running_loop()
    cb = _make_thread_safe_progress(loop, progress_broadcast)
    return await asyncio.to_thread(_render_and_extract_sync, output_dir, cb)


# ── Phase 3: Reinsert + render translated ────────────────────


def _run_reinsert_sync(
    pptx_bytes: bytes,
    translations: list[dict],
    output_dir: Path,
    progress_cb=None,
) -> dict:
    """Reinsert translated text and render translated slides."""
    input_path = output_dir / "_upload.pptx"
    input_path.write_bytes(pptx_bytes)

    try:
        meta_path = output_dir / "meta.json"
        meta_dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        meta_dict["stage"] = "reinserting"
        meta_dict.setdefault("stages", {})["reinserting"] = {"status": "in_progress"}
        _atomic_write_json(meta_path, meta_dict)

        translated_pptx = output_dir / "translated.pptx"
        reinsert_translated_text(input_path, translations, translated_pptx)
        meta_dict["stages"]["reinserting"] = {"status": "done"}
        _atomic_write_json(meta_path, meta_dict)
        logger.info("Reinserted translations into PPTX")

        meta_dict["stage"] = "rendering_translated"
        meta_dict["stages"]["rendering_translated"] = {"status": "in_progress"}
        _atomic_write_json(meta_path, meta_dict)

        translated_dir = output_dir / "translated"
        translated_dir.mkdir(exist_ok=True)
        count = render_translated_to_images(
            translated_pptx, translated_dir, progress_cb=progress_cb
        )
        meta_dict["stages"]["rendering_translated"] = {
            "status": "done",
            "progress": f"{count}/{count}",
        }

        from datetime import UTC, datetime

        meta_dict["completed_at"] = datetime.now(UTC).isoformat()
        meta_dict["stage"] = "complete"
        _atomic_write_json(meta_path, meta_dict)
        logger.info("Rendered %d translated slides", count)

        return meta_dict

    finally:
        input_path.unlink(missing_ok=True)


async def run_reinsert(
    pptx_bytes: bytes,
    translations: list[dict],
    output_dir: Path,
    max_output_mb: int = 200,
    progress_broadcast: ProgressBroadcast | None = None,
) -> dict:
    """Run text reinsertion + translated slide rendering in a background thread.

    ``progress_broadcast`` is awaited on the event loop after each translated
    PNG is finalized: ``await progress_broadcast(idx, total)``.
    """
    loop = asyncio.get_running_loop()
    cb = _make_thread_safe_progress(loop, progress_broadcast)
    return await asyncio.to_thread(
        _run_reinsert_sync, pptx_bytes, translations, output_dir, cb
    )


# ── Express-lane partial render (first 1-2 translated slides) ─


async def run_partial_translated_render(
    source_pptx_bytes: bytes,
    translations: list[dict],
    slide_indices_0based: list[int],
    output_dir: Path,
) -> list[int]:
    """Render translated PNGs for ONLY the given slides (express lane).

    Used to make the first 1-2 translated slides visible quickly, before the
    bulk reinsert + render of the entire deck completes.
    """
    from meeting_scribe.slides.convert import render_partial_translated

    def _sync() -> list[int]:
        # Write source bytes to a PER-INVOCATION scratch dir inside the
        # deck dir. A shared `_express` dir would race when the caller
        # fans out multiple batches in parallel (which the scribe does
        # under SCRIBE_SLIDE_RENDER_PARALLELISM=4) — every call writes
        # the same source.pptx + partial_translated.pptx + minimal PPTX
        # names, so concurrent invocations would trample each other.
        deck_dir = output_dir.parent  # output_dir is .../{deck_id}/translated
        work_dir = deck_dir / f"_express_{uuid.uuid4().hex}"
        work_dir.mkdir(parents=True, exist_ok=True)
        scratch_pptx = work_dir / "source.pptx"
        scratch_pptx.write_bytes(source_pptx_bytes)
        try:
            return render_partial_translated(
                scratch_pptx,
                translations,
                slide_indices_0based,
                output_dir,
                work_dir,
            )
        finally:
            scratch_pptx.unlink(missing_ok=True)
            # Remove everything the render left behind (LO profile,
            # intermediate PDFs, minimal PPTX). Best-effort; any leftover
            # files are ignored so transient failures don't surface here.
            import shutil as _shutil

            _shutil.rmtree(work_dir, ignore_errors=True)

    return await asyncio.to_thread(_sync)
