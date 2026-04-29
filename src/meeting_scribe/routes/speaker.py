"""Per-meeting speaker editing routes.

The four endpoints in this module all mutate the speaker side of a
recorded meeting — segment-level renames, cluster-level renames,
speaker→seat binds, and the read endpoint that the review UI polls
for the current state.

All four are atomic across multiple files (journal.jsonl,
detected_speakers.json, room.json, sometimes summary.json) and
broadcast to live WS clients when the meeting is still active so
the rename appears immediately on every open tab.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
from contextlib import suppress

import fastapi
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from meeting_scribe.runtime import state
from meeting_scribe.server_support.broadcast import _broadcast_json
from meeting_scribe.server_support.safe_paths import _safe_meeting_dir

logger = logging.getLogger(__name__)

router = APIRouter()


@router.put("/api/meetings/{meeting_id}/events/{segment_id}/speaker")
async def update_segment_speaker(
    meeting_id: str, segment_id: str, request: fastapi.Request
) -> JSONResponse:
    """Rename a speaker. Updates journal, state.detected_speakers.json, and room.json."""
    body = await request.json()
    speaker_name = (body.get("speaker_name") or body.get("display_name") or "").strip()
    old_name = (body.get("old_name") or "").strip()
    if not speaker_name:
        return JSONResponse({"error": "speaker_name required"}, status_code=400)

    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir:
        return JSONResponse({"error": "Invalid meeting ID"}, status_code=400)
    journal_path = meeting_dir / "journal.jsonl"
    if not journal_path.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    # 1. Append correction to journal
    correction = {
        "type": "speaker_correction",
        "segment_id": segment_id,
        "speaker_name": speaker_name,
    }
    with journal_path.open("a") as f:
        f.write(_json.dumps(correction) + "\n")

    # 2. Update state.detected_speakers.json (if old_name provided)
    if old_name:
        speakers_path = meeting_dir / "detected_speakers.json"
        if speakers_path.exists():
            with suppress(Exception):
                speakers = _json.loads(speakers_path.read_text())
                changed = False
                for s in speakers:
                    if s.get("display_name") == old_name:
                        s["display_name"] = speaker_name
                        changed = True
                if changed:
                    speakers_path.write_text(_json.dumps(speakers, indent=2))

        # 3. Update room.json seat names
        room_path = meeting_dir / "room.json"
        if room_path.exists():
            with suppress(Exception):
                room = _json.loads(room_path.read_text())
                changed = False
                for seat in room.get("seats", []):
                    if seat.get("speaker_name") == old_name:
                        seat["speaker_name"] = speaker_name
                        changed = True
                if changed:
                    room_path.write_text(_json.dumps(room, indent=2))

        # speaker_lanes.json uses cluster_id keys — names come from state.detected_speakers (already updated above)

    return JSONResponse(
        {"status": "updated", "segment_id": segment_id, "speaker_name": speaker_name}
    )


@router.get("/api/meetings/{meeting_id}/speakers")
async def get_meeting_speakers(meeting_id: str) -> JSONResponse:
    """Get detected speakers for a meeting."""
    speakers = state.storage.load_detected_speakers(meeting_id)
    return JSONResponse({"speakers": speakers})


@router.put("/api/meetings/{meeting_id}/clusters/{cluster_id}/name")
async def rename_cluster(
    meeting_id: str, cluster_id: str, request: fastapi.Request
) -> JSONResponse:
    """Rename all segments in a cluster at once.

    Writes speaker_correction entries for every segment belonging to the
    given cluster_id, updates state.detected_speakers.json, and broadcasts the
    rename to connected WebSocket clients so the live UI updates immediately.

    This is the "click speaker in virtual table → rename" flow.
    """
    body = await request.json()
    new_name = (body.get("speaker_name") or body.get("display_name") or "").strip()
    if not new_name:
        return JSONResponse({"error": "speaker_name required"}, status_code=400)

    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir:
        return JSONResponse({"error": "Invalid meeting ID"}, status_code=400)
    journal_path = meeting_dir / "journal.jsonl"
    if not journal_path.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    # Parse cluster_id (may be int or string-int)
    try:
        target_cid = int(cluster_id)
    except TypeError, ValueError:
        return JSONResponse({"error": "Invalid cluster_id"}, status_code=400)

    # Find all segments belonging to this cluster (apply corrections + dedup)
    affected_segment_ids: list[str] = []
    events_by_sid: dict[str, dict] = {}
    for line in journal_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = _json.loads(line)
        except Exception:
            continue
        if e.get("type") == "speaker_correction":
            continue
        if not e.get("is_final") or not e.get("text"):
            continue
        sid = e.get("segment_id")
        if not sid:
            continue
        rev = e.get("revision", 0)
        if sid not in events_by_sid or rev > events_by_sid[sid].get("revision", 0):
            events_by_sid[sid] = e

    for sid, event in events_by_sid.items():
        sp = (event.get("speakers") or [{}])[0]
        if sp.get("cluster_id") == target_cid:
            affected_segment_ids.append(sid)

    if not affected_segment_ids:
        return JSONResponse(
            {"error": f"No segments found for cluster_id={target_cid}"},
            status_code=404,
        )

    # Append bulk corrections to journal
    with journal_path.open("a") as f:
        for sid in affected_segment_ids:
            f.write(
                _json.dumps(
                    {
                        "type": "speaker_correction",
                        "segment_id": sid,
                        "speaker_name": new_name,
                    }
                )
                + "\n"
            )

    # Update state.detected_speakers.json — find the entry for this cluster_id
    speakers_path = meeting_dir / "detected_speakers.json"
    if speakers_path.exists():
        with suppress(Exception):
            speakers = _json.loads(speakers_path.read_text())
            for s in speakers:
                if s.get("cluster_id") == target_cid:
                    s["display_name"] = new_name
            speakers_path.write_text(_json.dumps(speakers, indent=2))

    # Broadcast to live WS clients so the UI updates immediately
    try:
        await _broadcast_json(
            {
                "type": "speaker_rename",
                "cluster_id": target_cid,
                "display_name": new_name,
                "affected_segments": len(affected_segment_ids),
            }
        )
    except Exception:
        pass

    logger.info(
        "Cluster rename: cluster %d → '%s' (%d segments)",
        target_cid,
        new_name,
        len(affected_segment_ids),
    )

    return JSONResponse(
        {
            "status": "renamed",
            "cluster_id": target_cid,
            "display_name": new_name,
            "affected_segments": len(affected_segment_ids),
        }
    )


@router.post("/api/meetings/{meeting_id}/speakers/assign")
async def assign_speaker_to_seat(
    meeting_id: str,
    request: fastapi.Request,
) -> JSONResponse:
    """Bind a detected voice cluster to a seat and/or name.

    Body: {cluster_id: int, seat_id: str | null, display_name: str}

    Atomic multi-file update:
    - Updates room.json (sets speaker_name on matching seat)
    - Updates state.detected_speakers.json (sets display_name for cluster_id)
    - Rewrites journal.jsonl (sets speakers[0].identity for matching events)
    - Regenerates speaker_lanes.json via _generate_speaker_data

    Returns the updated state.detected_speakers list.
    """
    from meeting_scribe.server_support.meeting_artifacts import (
        _generate_speaker_data,
        _generate_timeline,
    )

    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir:
        return JSONResponse({"error": "Invalid meeting ID"}, status_code=400)

    try:
        body = await request.json()
        cluster_id = int(body.get("cluster_id"))
        seat_id = body.get("seat_id")  # may be None to unbind
        display_name = str(body.get("display_name", "")).strip()
    except Exception as e:
        return JSONResponse({"error": f"Invalid request: {e}"}, status_code=422)

    if not display_name:
        return JSONResponse({"error": "display_name required"}, status_code=422)

    # 1. Update room.json — find seat_id, set speaker_name
    layout = state.storage.load_room_layout(meeting_id)
    if layout and seat_id:
        for seat in layout.seats:
            if seat.seat_id == seat_id:
                seat.speaker_name = display_name
            elif seat.speaker_name == display_name:
                # Unbind any other seat that had this name
                seat.speaker_name = ""
        state.storage.save_room_layout(meeting_id, layout)

    # 2. Update state.detected_speakers.json
    speakers_path = meeting_dir / "detected_speakers.json"
    if speakers_path.exists():
        try:
            speakers_list = _json.loads(speakers_path.read_text())
            for sp in speakers_list:
                # Match by cluster_id stored in speaker_id or as int
                sp_cluster = sp.get("cluster_id", sp.get("speaker_id"))
                try:
                    sp_cluster_int = int(sp_cluster) if sp_cluster is not None else -1
                except ValueError, TypeError:
                    sp_cluster_int = -1
                if sp_cluster_int == cluster_id:
                    sp["display_name"] = display_name
            speakers_path.write_text(_json.dumps(speakers_list, indent=2))
        except (_json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to update state.detected_speakers.json: %s", e)

    # 3. Rewrite journal.jsonl — set speakers[0].identity for matching events
    updated_events = state.storage.update_journal_speaker_identity(
        meeting_id,
        cluster_id,
        display_name,
    )

    # 4. Regenerate speaker_lanes.json + timeline.json from the updated
    # journal so EVERY view that queries the meeting API (post-finalize
    # replay, participant list, timeline scrubber, summary) picks up the
    # rename without a separate refresh.
    try:
        journal_path = meeting_dir / "journal.jsonl"
        if journal_path.exists():
            _generate_speaker_data(meeting_dir, journal_path, _json)
        _generate_timeline(meeting_id)
    except Exception as e:
        logger.warning("Failed to regenerate speaker data: %s", e)

    # 4b. Summary.json bakes the old name into topics / action-items /
    # speaker_stats. Re-run summary in the background so the review view
    # surfaces the new name everywhere without blocking this response.
    # Best-effort: we've already persisted the rename, so summary regen
    # can fail silently.
    async def _regen_summary():
        try:
            summary_path = meeting_dir / "summary.json"
            if summary_path.exists():
                from meeting_scribe.summary import generate_summary

                summary = await generate_summary(
                    meeting_dir, vllm_url=state.config.translate_vllm_url
                )
                if summary and not summary.get("error"):
                    summary_path.write_text(_json.dumps(summary, indent=2, ensure_ascii=False))
                    # Tell any open review UI to reload the summary panel.
                    await _broadcast_json(
                        {
                            "type": "summary_regenerated",
                            "meeting_id": meeting_id,
                        }
                    )
        except Exception as exc:
            logger.warning("Rename: summary regen failed: %s", exc)

    asyncio.create_task(_regen_summary())

    # 5. Broadcast if this is the active meeting
    if state.current_meeting and state.current_meeting.meeting_id == meeting_id:
        await _broadcast_json(
            {
                "type": "speaker_assignment",
                "cluster_id": cluster_id,
                "seat_id": seat_id,
                "display_name": display_name,
            }
        )

    updated_speakers = state.storage.load_detected_speakers(meeting_id)
    return JSONResponse(
        {
            "status": "ok",
            "cluster_id": cluster_id,
            "display_name": display_name,
            "seat_id": seat_id,
            "updated_events": updated_events,
            "speakers": updated_speakers,
        }
    )
