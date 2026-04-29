"""Slide deck upload, rendering, and viewer endpoints.

The slides feature pipes a PPTX through render → translate → PNG
on disk, then streams the resulting images to the popout viewer +
admin UI. Nine endpoints in this module:

* ``/slides/upload`` — accepts PPTX, kicks off the SlideJobRunner
  pipeline, returns a deck_id immediately (translation continues
  in the background).
* ``/decks`` (GET) + ``/decks/active`` (PUT) — multi-deck switch
  surface; broadcasts ``slide_deck_changed`` so popout viewers
  flip without manual reload.
* ``/slides`` (GET) — deck metadata, including ``current_slide_index``
  for the singleton active deck.
* ``/slides/original.pdf`` and ``/slides/source.pptx`` — file
  exports of the rendered + original assets.
* ``/slides/{index}/original`` and ``/slides/{index}/translated``
  — per-slide PNG streams.
* ``/slides/current`` (PUT) — set the active slide index and
  broadcast ``slide_change`` to all viewers.

The terminal-log endpoint also lives here because it sits between
the slides routes in the original server.py and shares the same
admin gate; moving it later in its own commit would just make the
diff bigger without making the file more cohesive.

All write/upload routes use ``_require_admin_or_raise`` (HTTPException
fail-fast); the per-meeting terminal log uses
``_require_admin_response`` for explicit-403 semantics.
"""

from __future__ import annotations

import logging

import fastapi
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from meeting_scribe.runtime import state
from meeting_scribe.server_support.admin_guard import (
    _require_admin_or_raise,
    _require_admin_response,
)
from meeting_scribe.server_support.broadcast import _broadcast_json
from meeting_scribe.server_support.safe_paths import _safe_meeting_dir

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/meetings/{meeting_id}/slides/upload")
async def upload_slides(meeting_id: str, request: fastapi.Request):
    """Upload a PPTX file for slide translation.

    Triggers the full pipeline: validate -> render -> extract -> translate
    -> reinsert -> render translated. Original slides are viewable as soon
    as the rendering stage completes.
    """
    _require_admin_or_raise(request)

    if not state.slides_enabled or state.slide_job_runner is None:
        return JSONResponse(
            {"error": "Slide processing unavailable (worker container not found)"},
            status_code=503,
        )

    form = await request.form()
    upload = form.get("file")
    if upload is None or isinstance(upload, str):
        return JSONResponse({"error": "No file uploaded"}, status_code=400)

    contents = await upload.read()
    upload_filename = getattr(upload, "filename", None) or ""
    logger.info(
        "Slide upload: %d bytes filename=%r",
        len(contents),
        upload_filename,
    )

    # Default to THIS MEETING's language pair (so a Dutch meeting doesn't
    # silently get the global ja↔en pair). Fall back to the global default
    # only if the meeting's meta is unreadable. Explicit form overrides win
    # and bypass auto-detection — useful when the detector misclassifies
    # (e.g. a Japanese deck dominated by English brand names).
    meeting_source = ""
    meeting_target = ""
    meeting_monolingual = False
    try:
        _meta = state.storage._read_meta(meeting_id)
        if _meta and _meta.language_pair:
            if _meta.is_monolingual:
                meeting_source = _meta.language_pair[0]
                meeting_target = ""  # no target for monolingual decks
                meeting_monolingual = True
            elif len(_meta.language_pair) == 2:
                meeting_source, meeting_target = _meta.language_pair[0], _meta.language_pair[1]
    except Exception:
        pass
    if not meeting_monolingual and not (meeting_source and meeting_target):
        # Fall back to the process-wide default. Only reached for decks
        # uploaded against a meeting whose meta is unreadable.
        parts = [p.strip() for p in state.config.default_language_pair.split(",")]
        if len(parts) == 1:
            meeting_source = parts[0]
            meeting_target = ""
            meeting_monolingual = True
        elif len(parts) >= 2:
            meeting_source, meeting_target = parts[0], parts[1]
    _src = form.get("source_lang") or ""
    _tgt = form.get("target_lang") or ""
    source_lang = (_src if isinstance(_src, str) else "").strip() or meeting_source
    target_lang = (_tgt if isinstance(_tgt, str) else "").strip() or meeting_target
    explicit_override = bool(form.get("source_lang") and form.get("target_lang"))

    if explicit_override:
        # An explicit target override wins over the meeting's monolingual
        # default — the operator is asking for a specific translation.
        meeting_monolingual = False
        logger.info(
            "Slide upload: explicit language override %s→%s",
            source_lang,
            target_lang,
        )

    deck_id = await state.slide_job_runner.start_job(
        meeting_id,
        contents,
        source_lang,
        target_lang,
        skip_language_detection=explicit_override,
        upload_filename=upload_filename,
        monolingual=meeting_monolingual,
    )

    return JSONResponse({"deck_id": deck_id, "status": "processing"})


@router.get("/api/meetings/{meeting_id}/decks")
async def list_meeting_decks(meeting_id: str):
    """List every slide deck on disk for a meeting (newest first).

    Each deck entry includes its meta + a `is_active` flag. Used by the
    popout's deck switcher so the user can flip between multiple
    uploaded decks during a meeting OR review past decks afterwards.
    """
    if state.slide_job_runner is None:
        return JSONResponse({"error": "Slides not enabled"}, status_code=503)
    return JSONResponse(
        {
            "meeting_id": meeting_id,
            "decks": state.slide_job_runner.list_decks(meeting_id),
        }
    )


@router.put("/api/meetings/{meeting_id}/decks/active")
async def set_active_meeting_deck(meeting_id: str, request: fastapi.Request):
    """Switch the active deck for a meeting. Body: ``{"deck_id": "..."}``.

    Broadcasts ``slide_deck_changed`` so connected viewers swap to the
    chosen deck without a manual reload. Works on both live and past
    meetings — past meeting "switching" just changes which deck the
    UI/endpoints serve next.
    """
    if state.slide_job_runner is None:
        return JSONResponse({"error": "Slides not enabled"}, status_code=503)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    deck_id = (body or {}).get("deck_id", "").strip() if isinstance(body, dict) else ""
    if not deck_id:
        return JSONResponse({"error": "Missing deck_id"}, status_code=400)
    meta = await state.slide_job_runner.set_active_deck(meeting_id, deck_id)
    if meta is None:
        return JSONResponse({"error": "Deck not found"}, status_code=404)
    return JSONResponse({"meeting_id": meeting_id, "deck_id": deck_id, "meta": meta})


@router.get("/api/meetings/{meeting_id}/slides")
async def get_slides_metadata(meeting_id: str):
    """Get deck metadata for any meeting that has slides on disk.

    Works for both the live meeting and past completed meetings (slides
    are part of the meeting record, not just live state). The
    ``current_slide_index`` field is only populated for the singleton
    active deck — past meetings get the persisted final value or 0.
    """
    if state.slide_job_runner is None:
        return JSONResponse({"error": "Slides not enabled"}, status_code=503)

    # Past-meeting path: read deck meta directly from disk for this
    # specific meeting, without disturbing the singleton active state.
    meta = state.slide_job_runner.get_meeting_deck_meta(meeting_id)
    if meta is None:
        return JSONResponse({"error": "No deck for this meeting"}, status_code=404)

    # If THIS meeting is the live one, layer in the in-memory current index
    # so the popout can resume from where the presenter left off.
    if state.slide_job_runner.active_deck_id and state.slide_job_runner.active_deck_id == meta.get(
        "deck_id"
    ):
        meta["current_slide_index"] = state.slide_job_runner.current_slide_index
    else:
        meta.setdefault("current_slide_index", 0)
    return JSONResponse(meta)


@router.get("/api/meetings/{meeting_id}/terminal-log")
async def get_terminal_log(meeting_id: str, request: fastapi.Request):
    """Serve the embedded-terminal output log for a past or current meeting.

    The file is tee'd directly from the PTY, so it contains both inputs
    (echoed by the shell) and outputs, with ANSI escape sequences
    preserved. Gated the same way the live terminal is: admin cookie
    required on the admin LAN; guest scope is rejected outright.
    """
    blocked = _require_admin_response(request)
    if blocked is not None:
        return blocked
    meeting_dir = _safe_meeting_dir(meeting_id)
    if meeting_dir is None:
        return JSONResponse({"error": "invalid meeting_id"}, status_code=400)
    log_path = meeting_dir / "terminal.log"
    if not log_path.exists():
        return JSONResponse({"error": "no terminal log for this meeting"}, status_code=404)
    from starlette.responses import FileResponse

    return FileResponse(log_path, media_type="text/plain; charset=utf-8")


@router.get("/api/meetings/{meeting_id}/slides/original.pdf")
async def get_slides_original_pdf(meeting_id: str):
    """Serve the original slides as a PDF (works for past meetings too)."""
    if state.slide_job_runner is None:
        return JSONResponse({"error": "Slides not enabled"}, status_code=503)

    pdf_path = state.slide_job_runner.get_original_pdf_path(meeting_id)
    if pdf_path is None:
        return JSONResponse({"error": "PDF not yet rendered or no deck"}, status_code=404)

    from starlette.responses import FileResponse

    return FileResponse(pdf_path, media_type="application/pdf")


@router.get("/api/meetings/{meeting_id}/slides/source.pptx")
async def get_slides_source_pptx(meeting_id: str):
    """Serve the original PPTX (works for past meetings too)."""
    if state.slide_job_runner is None:
        return JSONResponse({"error": "Slides not enabled"}, status_code=503)

    pptx_path = state.slide_job_runner.get_source_pptx_path(meeting_id)
    if pptx_path is None:
        return JSONResponse({"error": "Source PPTX not found"}, status_code=404)

    from starlette.responses import FileResponse

    return FileResponse(
        pptx_path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename="presentation.pptx",
    )


@router.get("/api/meetings/{meeting_id}/slides/{index}/original")
async def get_slide_original(meeting_id: str, index: int):
    """Serve an original slide PNG. Works for both the live meeting AND
    past meetings (slides persist as part of the meeting record now)."""
    if state.slide_job_runner is None:
        return JSONResponse({"error": "Slides not enabled"}, status_code=503)

    if index < 0:
        return JSONResponse({"error": f"Invalid slide index {index}"}, status_code=422)

    path = state.slide_job_runner.get_slide_image_path(meeting_id, index, translated=False)
    if path is None:
        return JSONResponse({"error": "Slide not yet rendered or no deck"}, status_code=404)

    from starlette.responses import FileResponse

    return FileResponse(path, media_type="image/png")


@router.get("/api/meetings/{meeting_id}/slides/{index}/translated")
async def get_slide_translated(meeting_id: str, index: int):
    """Serve a translated slide PNG. Works for both the live meeting AND
    past meetings (slides persist as part of the meeting record now)."""
    if state.slide_job_runner is None:
        return JSONResponse({"error": "Slides not enabled"}, status_code=503)

    if index < 0:
        return JSONResponse({"error": f"Invalid slide index {index}"}, status_code=422)

    path = state.slide_job_runner.get_slide_image_path(meeting_id, index, translated=True)
    if path is None:
        return JSONResponse({"error": "Translation not yet ready or no deck"}, status_code=404)

    from starlette.responses import FileResponse

    return FileResponse(path, media_type="image/png")


@router.put("/api/meetings/{meeting_id}/slides/current")
async def set_current_slide(meeting_id: str, request: fastapi.Request):
    """Set the current slide index and broadcast to all viewers."""
    _require_admin_or_raise(request)

    if state.slide_job_runner is None:
        return JSONResponse({"error": "Slides not enabled"}, status_code=503)

    if state.slide_job_runner.active_deck_id is None:
        return JSONResponse({"error": "No active deck"}, status_code=404)

    body = await request.json()
    index = body.get("index")
    if index is None or not isinstance(index, int):
        return JSONResponse({"error": "Missing or invalid 'index'"}, status_code=400)

    if not state.slide_job_runner.advance_slide(index):
        return JSONResponse(
            {
                "error": f"Slide index {index} out of range (0-{state.slide_job_runner.total_slides - 1})"
            },
            status_code=422,
        )

    await _broadcast_json(
        {
            "type": "slide_change",
            "deck_id": state.slide_job_runner.active_deck_id,
            "slide_index": index,
        }
    )

    return JSONResponse({"ok": True, "slide_index": index})
