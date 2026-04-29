"""Meeting CRUD + read endpoints (everything except start/stop/finalize).

Thirteen routes that read or mutate a single meeting's recorded
artifacts:

* ``/api/meetings`` (GET) — list every meeting on disk with a short
  summary preview.
* Per-segment / per-meeting reads: ``/tts/{segment_id}``,
  ``/audio``, ``/timeline``, ``/polished``, ``/export``,
  ``/summary``, ``/versions``, ``/versions/diff``.
* ``/api/meetings/{id}/ask`` — SSE-streaming Q&A over the transcript.
* ``/api/meetings/{id}`` GET / PATCH / DELETE — full read, partial
  update (currently just ``is_favorite``), full removal (refuses
  while the meeting is the active recording).

The two summary-translation helpers (``_translate_text_via_vllm``,
``_translate_summary``) live here too — the only caller is
``GET /summary?lang=…``, no point relocating them to a generic
helper module.
"""

from __future__ import annotations

import json as _json
import logging
import shutil
import struct as _struct
from contextlib import suppress

import fastapi
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from meeting_scribe.runtime import state
from meeting_scribe.server_support.safe_paths import _safe_meeting_dir, _safe_segment_path

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/meetings")
async def list_meetings() -> JSONResponse:
    """List all meetings with metadata + short summary.

    Extracts `executive_summary` and first topic from summary.json so the
    meetings list can show context without a second fetch per meeting.
    """
    meetings_dir = state.storage._meetings_dir
    results = []
    if meetings_dir.exists():
        for d in meetings_dir.iterdir():
            meta_path = d / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = _json.loads(meta_path.read_text())
                journal_path = d / "journal.jsonl"
                event_count = 0
                if journal_path.exists():
                    event_count = sum(1 for _ in journal_path.open())

                # Read summary.json for short context (executive_summary + topics)
                summary_path = d / "summary.json"
                executive_summary = None
                topics_preview: list[str] = []
                if summary_path.exists():
                    try:
                        sdata = _json.loads(summary_path.read_text())
                        es = sdata.get("executive_summary")
                        if isinstance(es, str):
                            executive_summary = es
                        topics_raw = sdata.get("topics", [])
                        if isinstance(topics_raw, list):
                            for t in topics_raw[:5]:
                                if isinstance(t, str):
                                    topics_preview.append(t)
                                elif isinstance(t, dict) and "title" in t:
                                    topics_preview.append(str(t["title"]))
                    except Exception:
                        pass

                results.append(
                    {
                        "meeting_id": meta["meeting_id"],
                        "state": meta["state"],
                        "created_at": meta.get("created_at", ""),
                        "event_count": event_count,
                        "has_room": (d / "room.json").exists(),
                        "has_speakers": (d / "speakers.json").exists(),
                        "has_summary": summary_path.exists(),
                        "has_timeline": (d / "timeline.json").exists(),
                        "has_slides": (d / "slides" / "active_deck_id").exists(),
                        "executive_summary": executive_summary,
                        "topics": topics_preview,
                        "is_favorite": bool(meta.get("is_favorite", False)),
                    }
                )
            except Exception:
                continue

    # Strict chronological order (newest first). Favorites used to rise
    # to the top here, but that reshuffled the timeline whenever a star
    # was toggled. The UI now has a separate "Favorites" tab that filters
    # the list, so favoriting can leave the chronological order intact.
    results.sort(key=lambda m: m["created_at"], reverse=True)
    return JSONResponse({"meetings": results})


@router.get("/api/meetings/{meeting_id}/tts/{segment_id}")
async def get_tts_audio(meeting_id: str, segment_id: str) -> fastapi.responses.Response:
    """Get synthesized TTS audio for a specific segment."""
    from fastapi.responses import FileResponse

    tts_path = _safe_segment_path(meeting_id, "tts", f"{segment_id}.wav")
    if not tts_path or not tts_path.exists():
        return JSONResponse({"error": "No TTS audio for this segment"}, status_code=404)
    return FileResponse(tts_path, media_type="audio/wav")


@router.get("/api/meetings/{meeting_id}/audio")
async def get_meeting_audio(
    meeting_id: str, request: fastapi.Request
) -> fastapi.responses.Response:
    """Stream meeting audio as WAV for a time range.

    Query params: start_ms (default 0), end_ms (default full duration).
    Returns audio/wav with proper headers for browser <audio> element.
    """
    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir:
        return JSONResponse({"error": "Invalid meeting ID"}, status_code=400)
    pcm_path = meeting_dir / "audio" / "recording.pcm"
    if not pcm_path.exists():
        return JSONResponse({"error": "No audio recording"}, status_code=404)

    start_ms = int(request.query_params.get("start_ms", 0))
    end_ms_param = request.query_params.get("end_ms")
    total_duration = state.storage.audio_duration_ms(meeting_id)
    end_ms = int(end_ms_param) if end_ms_param else total_duration

    # Clamp
    start_ms = max(0, min(start_ms, total_duration))
    end_ms = max(start_ms, min(end_ms, total_duration))

    pcm_data = state.storage.read_audio_segment(meeting_id, start_ms, end_ms)
    if not pcm_data:
        return JSONResponse({"error": "No audio in range"}, status_code=404)

    # Build WAV header
    sample_rate = 16000
    channels = 1
    bits_per_sample = 16
    data_size = len(pcm_data)
    header = _struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        sample_rate * channels * bits_per_sample // 8,
        channels * bits_per_sample // 8,
        bits_per_sample,
        b"data",
        data_size,
    )

    def stream():
        yield header
        # Stream PCM in 8KB chunks
        for i in range(0, len(pcm_data), 8192):
            yield pcm_data[i : i + 8192]

    return StreamingResponse(
        stream(),
        media_type="audio/wav",
        headers={
            "Content-Length": str(44 + data_size),
            "Accept-Ranges": "bytes",
        },
    )


@router.get("/api/meetings/{meeting_id}/timeline")
async def get_timeline(meeting_id: str) -> JSONResponse:
    """Get the timeline manifest for podcast player and speaker lane view."""
    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir:
        return JSONResponse({"error": "Invalid meeting ID"}, status_code=400)
    path = meeting_dir / "timeline.json"
    if not path.exists():
        return JSONResponse({"error": "No timeline"}, status_code=404)
    segments = _json.loads(path.read_text())
    if isinstance(segments, list):
        duration_ms = max((s.get("end_ms", 0) for s in segments), default=0)
    else:
        duration_ms = segments.get("duration_ms", 0)
        segments = segments.get("segments", [])

    # Include speaker lanes if available
    lanes_path = meeting_dir / "speaker_lanes.json"
    speaker_lanes = {}
    if lanes_path.exists():
        speaker_lanes = _json.loads(lanes_path.read_text())

    # Include detected speaker info
    speakers_path = meeting_dir / "detected_speakers.json"
    state.detected_speakers = []
    if speakers_path.exists():
        state.detected_speakers = _json.loads(speakers_path.read_text())

    return JSONResponse(
        {
            "duration_ms": duration_ms,
            "segments": segments,
            "speaker_lanes": speaker_lanes,
            "speakers": state.detected_speakers,
        }
    )


@router.get("/api/meetings/{meeting_id}/polished")
async def get_polished(meeting_id: str) -> JSONResponse:
    """Get polished transcript from the refinement worker."""
    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir:
        return JSONResponse({"error": "Invalid meeting ID"}, status_code=400)
    path = meeting_dir / "polished.json"
    if not path.exists():
        return JSONResponse({"error": "No polished transcript"}, status_code=404)
    return JSONResponse(_json.loads(path.read_text()))


@router.get("/api/meetings/{meeting_id}/export")
async def export_meeting(
    meeting_id: str, format: str = "md", lang: str | None = None
) -> fastapi.responses.Response:
    """Export meeting in various formats.

    Query params:
        format: "md" (Markdown), "txt" (plain text), "zip" (full archive)
        lang: "en" or "ja" (for txt format only — filters by language)
    """
    from meeting_scribe.export import meeting_to_markdown, meeting_to_zip, transcript_to_text

    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir or not meeting_dir.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    if format == "md":
        content: str | bytes = meeting_to_markdown(meeting_dir)
        return fastapi.responses.Response(
            content=content,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="meeting-{meeting_id}.md"'},
        )
    elif format == "txt":
        journal = meeting_dir / "journal.jsonl"
        if not journal.exists():
            return JSONResponse({"error": "No transcript"}, status_code=404)
        content = transcript_to_text(journal, lang=lang)
        lang_suffix = f"-{lang}" if lang else ""
        return fastapi.responses.Response(
            content=content,
            media_type="text/plain; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="transcript-{meeting_id}{lang_suffix}.txt"'
            },
        )
    elif format == "zip":
        content = meeting_to_zip(meeting_dir)
        return fastapi.responses.Response(
            content=content,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="meeting-{meeting_id}.zip"'},
        )
    else:
        return JSONResponse({"error": f"Unknown format: {format}"}, status_code=400)


async def _translate_text_via_vllm(text: str, target_lang: str) -> str:
    """Translate a single text string via the live vLLM translation backend.

    Returns the original text on any failure — summary translation is
    best-effort and shouldn't blow up the endpoint if the backend is busy.
    """
    if not text or not text.strip():
        return text
    if state.translate_backend is None:
        return text
    client = getattr(state.translate_backend, "_client", None)
    model = getattr(state.translate_backend, "_model", None)
    if client is None or model is None:
        return text
    from meeting_scribe.languages import LANGUAGE_REGISTRY

    target_name = (
        LANGUAGE_REGISTRY[target_lang].name if target_lang in LANGUAGE_REGISTRY else target_lang
    )
    try:
        resp = await client.post(
            f"{state.config.translate_vllm_url.rstrip('/')}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            f"Translate the user's text into {target_name}. "
                            f"Preserve markdown, line breaks, and any speaker "
                            f"labels (e.g. 'Speaker 1:'). Return only the "
                            f"translation — no preamble, no quotation marks."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                "temperature": 0.2,
                "max_tokens": min(2048, max(64, int(len(text) * 1.5))),
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "priority": 10,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning("Summary text translation failed: %s", exc)
        return text


async def _translate_summary(summary: dict, target_lang: str) -> dict:
    """Translate every user-visible text field in a summary dict.

    Walks the v2 summary schema (executive_summary, key_insights[],
    action_items[], topics[], decisions[], key_quotes[]) and translates
    the textual leaves. Speaker names, cluster_ids, dates, and other
    non-prose fields are left intact.
    """
    out = _json.loads(_json.dumps(summary))  # deep copy via JSON

    if isinstance(out.get("executive_summary"), str):
        out["executive_summary"] = await _translate_text_via_vllm(
            out["executive_summary"], target_lang
        )

    for insight in out.get("key_insights", []) or []:
        if isinstance(insight.get("title"), str):
            insight["title"] = await _translate_text_via_vllm(insight["title"], target_lang)
        if isinstance(insight.get("description"), str):
            insight["description"] = await _translate_text_via_vllm(
                insight["description"], target_lang
            )

    for item in out.get("action_items", []) or []:
        if isinstance(item.get("category"), str):
            item["category"] = await _translate_text_via_vllm(item["category"], target_lang)
        if isinstance(item.get("task"), str):
            item["task"] = await _translate_text_via_vllm(item["task"], target_lang)

    for topic in out.get("topics", []) or []:
        if isinstance(topic.get("title"), str):
            topic["title"] = await _translate_text_via_vllm(topic["title"], target_lang)
        if isinstance(topic.get("description"), str):
            topic["description"] = await _translate_text_via_vllm(topic["description"], target_lang)

    if isinstance(out.get("decisions"), list):
        out["decisions"] = [
            await _translate_text_via_vllm(d, target_lang) if isinstance(d, str) else d
            for d in out["decisions"]
        ]

    for q in out.get("key_quotes", []) or []:
        if isinstance(q, dict) and isinstance(q.get("text"), str):
            q["text"] = await _translate_text_via_vllm(q["text"], target_lang)

    out.setdefault("translation_meta", {})["language"] = target_lang
    return out


@router.get("/api/meetings/{meeting_id}/summary-status")
async def get_summary_status(meeting_id: str) -> JSONResponse:
    """Return the summary-generation status envelope.

    Separate from ``/summary`` so the UI can distinguish:
      * 200 with status=complete → summary.json should exist; fetch it.
      * 200 with status=error → render the user-safe message + retry CTA.
      * 200 with status=generating → spinner; poll again.
      * 404 → no attempt has been made yet.

    Never proxies raw exception text. Marks STALE on the fly when the
    journal hash diverges from the saved one (e.g. after a reprocess).
    """
    from meeting_scribe.server_support.summary_status import (
        SummaryStatus,
        is_stale,
        read_status,
    )

    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir:
        return JSONResponse({"error": "Invalid meeting ID"}, status_code=400)
    envelope = read_status(meeting_dir)
    if envelope is None:
        return JSONResponse({"error": "No summary status available"}, status_code=404)
    journal_path = meeting_dir / "journal.jsonl"
    if is_stale(meeting_dir, journal_path):
        envelope = {**envelope, "status": SummaryStatus.STALE.value}
    return JSONResponse(envelope)


@router.get("/api/meetings/{meeting_id}/summary")
async def get_meeting_summary(
    meeting_id: str,
    lang: str | None = None,
) -> JSONResponse:
    """Get AI-generated meeting summary with speaker name corrections applied.

    ``lang``: optional ISO code from the meeting's language pair. When
    given AND the cached translation doesn't exist yet, every text field
    in the summary is translated via the live translation backend and
    cached as ``summary_{lang}.json``. Subsequent requests are instant.
    """
    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir:
        return JSONResponse({"error": "Invalid meeting ID"}, status_code=400)
    path = meeting_dir / "summary.json"
    if not path.exists():
        return JSONResponse({"error": "No summary available"}, status_code=404)

    # Validate ``lang`` against the meeting's language pair so we don't
    # spawn translations into arbitrary unrelated languages.
    requested_lang = (lang or "").strip().lower() or None
    if requested_lang:
        meta_path = meeting_dir / "meta.json"
        try:
            meeting_meta = _json.loads(meta_path.read_text()) if meta_path.exists() else {}
            valid = set(meeting_meta.get("language_pair", []) or [])
        except Exception:
            valid = set()
        if requested_lang not in valid:
            return JSONResponse(
                {
                    "error": "lang not in this meeting's language pair",
                    "supported": sorted(valid),
                },
                status_code=400,
            )
        # Serve the translated cache if it exists
        cached_path = meeting_dir / f"summary_{requested_lang}.json"
        if cached_path.exists():
            path = cached_path
        else:
            try:
                summary = _json.loads(path.read_text())
                translated = await _translate_summary(summary, requested_lang)
                cached_path.write_text(_json.dumps(translated, ensure_ascii=False, indent=2))
                path = cached_path
            except Exception as exc:
                logger.exception("Summary translation to %s failed", requested_lang)
                return JSONResponse(
                    {"error": f"Summary translation failed: {exc}"},
                    status_code=500,
                )

    summary = _json.loads(path.read_text())

    # Build name mapping from state.detected_speakers.json (updated by rename API)
    # The summary was generated with original "Speaker N" names.
    # state.detected_speakers.json has been updated with renamed display_names.
    name_map: dict[str, str] = {}
    speakers_path = meeting_dir / "detected_speakers.json"
    if speakers_path.exists():
        with suppress(Exception):
            current_speakers = _json.loads(speakers_path.read_text())
            for s in current_speakers:
                cid = s.get("cluster_id")
                current_name = s.get("display_name", "")
                original_name = f"Speaker {cid}"
                if current_name and current_name != original_name:
                    name_map[original_name] = current_name

    # Apply name corrections to summary
    if name_map:
        # Update speaker_stats names
        for stat in summary.get("speaker_stats", []):
            if stat.get("name") in name_map:
                stat["name"] = name_map[stat["name"]]

        # Update action item assignees
        for item in summary.get("action_items", []):
            if item.get("assignee") in name_map:
                item["assignee"] = name_map[item["assignee"]]

        # Update text references in executive_summary, topics, decisions, v2 fields
        for old, new in name_map.items():
            if summary.get("executive_summary"):
                summary["executive_summary"] = summary["executive_summary"].replace(old, new)
            for topic in summary.get("topics", []):
                if topic.get("description"):
                    topic["description"] = topic["description"].replace(old, new)
            summary["decisions"] = [d.replace(old, new) for d in summary.get("decisions", [])]
            # V2 fields: key_insights speaker lists + descriptions
            for insight in summary.get("key_insights", []):
                if insight.get("description"):
                    insight["description"] = insight["description"].replace(old, new)
                insight["speakers"] = [new if s == old else s for s in insight.get("speakers", [])]
            # V2 fields: key_quotes speaker attribution
            for quote in summary.get("key_quotes", []):
                if isinstance(quote, dict):
                    if quote.get("speaker") == old:
                        quote["speaker"] = new
                    if quote.get("text"):
                        quote["text"] = quote["text"].replace(old, new)

    return JSONResponse(summary)


@router.post("/api/meetings/{meeting_id}/ask", response_model=None)
async def ask_meeting(
    meeting_id: str, request: fastapi.Request
) -> fastapi.responses.StreamingResponse | JSONResponse:
    """Ask a question about a meeting transcript. Streams SSE answer chunks."""
    from meeting_scribe.qa import ask_meeting_question

    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir:
        return JSONResponse({"error": "Invalid meeting ID"}, status_code=400)
    if not meeting_dir.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    question = body.get("question", "").strip()
    if not question:
        return JSONResponse({"error": "Missing 'question' field"}, status_code=400)
    if len(question) > 2000:
        return JSONResponse({"error": "Question too long (max 2000 chars)"}, status_code=400)

    return fastapi.responses.StreamingResponse(
        ask_meeting_question(
            meeting_dir=meeting_dir,
            question=question,
            vllm_url=state.config.translate_vllm_url,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/api/meetings/{meeting_id}")
async def delete_meeting(meeting_id: str) -> JSONResponse:
    """Delete a meeting and all its artifacts."""
    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir or not meeting_dir.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    # Don't delete the currently active meeting
    if state.current_meeting and state.current_meeting.meeting_id == meeting_id:
        return JSONResponse({"error": "Cannot delete active meeting"}, status_code=400)

    shutil.rmtree(meeting_dir)
    logger.info("Deleted meeting: %s", meeting_id)
    return JSONResponse({"status": "deleted", "meeting_id": meeting_id})


@router.patch("/api/meetings/{meeting_id}")
async def update_meeting(meeting_id: str, request: fastapi.Request) -> JSONResponse:
    """Update editable fields on a meeting's metadata.

    Currently supports: ``is_favorite`` (bool). Only the listed keys are
    written; everything else in the request body is ignored. Persists via
    the storage layer's atomic write-rename so concurrent reads never see
    a half-written meta.json.
    """
    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir or not meeting_dir.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "Body must be a JSON object"}, status_code=400)

    try:
        meta = state.storage._read_meta(meeting_id)
    except FileNotFoundError:
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    changed: list[str] = []
    if "is_favorite" in body:
        new_val = bool(body["is_favorite"])
        if meta.is_favorite != new_val:
            meta.is_favorite = new_val
            changed.append("is_favorite")

    if not changed:
        return JSONResponse({"meeting_id": meeting_id, "changed": []})

    state.storage._write_meta(meta)
    logger.info("Updated meeting %s: %s", meeting_id, ", ".join(changed))
    return JSONResponse(
        {
            "meeting_id": meeting_id,
            "changed": changed,
            "is_favorite": meta.is_favorite,
        }
    )


@router.get("/api/meetings/{meeting_id}")
async def get_meeting(meeting_id: str) -> JSONResponse:
    """Get a specific meeting's full data (meta + transcript + room layout)."""
    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir or not meeting_dir.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    result = {}

    # Meta
    meta_path = meeting_dir / "meta.json"
    if meta_path.exists():
        result["meta"] = _json.loads(meta_path.read_text())

    # Room layout
    room_path = meeting_dir / "room.json"
    if room_path.exists():
        result["room"] = _json.loads(room_path.read_text())

    # Transcript events — load and apply speaker corrections
    journal_path = meeting_dir / "journal.jsonl"
    events = []
    corrections: dict[str, str] = {}  # segment_id → speaker_name
    if journal_path.exists():
        for line in journal_path.open():
            line = line.strip()
            if not line:
                continue
            with suppress(Exception):
                entry = _json.loads(line)
                if entry.get("type") == "speaker_correction":
                    corrections[entry["segment_id"]] = entry["speaker_name"]
                else:
                    events.append(entry)

    # Deduplicate by segment_id — keep best version per segment
    # Priority: version with translation > highest revision > first seen
    best: dict[str, dict] = {}
    for event in events:
        sid = event.get("segment_id")
        if not sid or not event.get("is_final") or not event.get("text"):
            continue

        has_tr = bool((event.get("translation") or {}).get("text"))
        existing = best.get(sid)

        if not existing:
            best[sid] = event
        elif has_tr and not (existing.get("translation") or {}).get("text"):
            # This version has translation, existing doesn't — always prefer
            best[sid] = event
        elif not has_tr and (existing.get("translation") or {}).get("text"):
            # Existing has translation, this doesn't — keep existing
            pass
        elif event.get("revision", 0) > existing.get("revision", 0):
            # Same translation state, higher revision wins
            # Preserve translation from existing if new one lacks it
            if (existing.get("translation") or {}).get("text") and not has_tr:
                event["translation"] = existing["translation"]
            best[sid] = event

    # Apply corrections to matching segments
    for sid, event in best.items():
        if sid in corrections:
            new_name = corrections[sid]
            speakers = event.get("speakers", [])
            if speakers:
                speakers[0]["identity"] = new_name
                speakers[0]["display_name"] = new_name
            else:
                event["speakers"] = [
                    {"identity": new_name, "display_name": new_name, "cluster_id": 0}
                ]

    # Sort by start_ms for consistent display order
    deduped = sorted(best.values(), key=lambda e: e.get("start_ms", 0))

    result["events"] = deduped
    result["total_events"] = len(deduped)

    return JSONResponse(result)


@router.get("/api/meetings/{meeting_id}/versions")
async def list_meeting_versions(meeting_id: str) -> JSONResponse:
    """List snapshot versions for a meeting (newest first).

    Each reprocess auto-snapshots the prior journal/summary/timeline/
    speakers under ``meetings/{id}/versions/`` so two runs can be
    compared. Returns the manifest for each snapshot.
    """
    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir or not meeting_dir.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)
    from meeting_scribe.versions import list_versions

    return JSONResponse({"meeting_id": meeting_id, "versions": list_versions(meeting_dir)})


@router.get("/api/meetings/{meeting_id}/versions/diff")
async def diff_meeting_versions(
    meeting_id: str,
    baseline: str | None = None,
    compare: str | None = None,
) -> JSONResponse:
    """Diff two versions (or the latest snapshot vs current state).

    ``baseline`` and ``compare`` are version directory names (from
    ``GET /versions``). Either omitted: ``baseline`` defaults to the
    most recent snapshot, ``compare`` defaults to the current state.
    Returns per-dimension verdicts (better/worse/same) so the caller
    can grade whether a code/model change improved transcription quality.
    """
    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir or not meeting_dir.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)
    from meeting_scribe.versions import (
        diff_versions,
        list_versions,
        metrics_for_current,
        metrics_for_version,
    )

    snaps = list_versions(meeting_dir)
    if not snaps and baseline is None:
        return JSONResponse(
            {"error": "No snapshots yet — run reprocess first"},
            status_code=404,
        )

    if baseline is None:
        baseline = snaps[0]["name"]
    base_metrics = metrics_for_version(meeting_dir, baseline)

    if compare is None:
        cmp_metrics = metrics_for_current(meeting_dir)
        cmp_label = "(current)"
    else:
        cmp_metrics = metrics_for_version(meeting_dir, compare)
        cmp_label = compare

    diff = diff_versions(base_metrics, cmp_metrics)
    return JSONResponse(
        {
            "meeting_id": meeting_id,
            "baseline": baseline,
            "compare": cmp_label,
            "diff": diff,
        }
    )
