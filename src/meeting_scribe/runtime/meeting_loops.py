"""Long-lived background tasks scoped to a meeting recording.

Three loops, each owned by ``state._eager_summary_task``,
``state._speaker_pulse_task``, ``state._speaker_catchup_task``
respectively. They start in ``/api/meeting/start`` (and on
``/resume``) and tear down in ``/stop`` / ``/cancel``.

* ``eager_summary_loop`` — generates draft summaries during recording
  every 5 min after a 5 min warmup. Uses vLLM priority 5 (well below
  translation's -10) so real-time translation is never starved.
* ``speaker_pulse_loop`` — emits a ``speaker_pulse`` WS broadcast
  every 200 ms with the single most-recently-active speaker so the
  client can animate the active-seat indicator.
* ``speaker_catchup_loop`` — retroactive speaker attribution. Walks
  ``state._pending_speaker_events`` every 400 ms, looks up
  diarization results for each pending event's range, attaches
  cluster_id + enrolled-name (if matched), and re-broadcasts the
  updated event so the frontend updates the block in place.

Pulled out of ``server.py``. ``routes.meeting_lifecycle`` retargets
its lazy imports to this module; tests still reference them via the
docstring/source-grep pattern at ``test_audio_e2e``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import WebSocket

from meeting_scribe.models import SpeakerAttribution
from meeting_scribe.runtime import state
from meeting_scribe.server_support.broadcast import _broadcast, _broadcast_json

logger = logging.getLogger(__name__)


async def _eager_summary_loop(meeting_id: str) -> None:
    """Periodically generate draft summaries during recording.

    Runs every 5 minutes after a 60-second warmup. Uses vLLM priority
    -5 (between translation's -10 and refinement's 5) so real-time
    translation still preempts but the draft has a fighting chance to
    keep up with longer meetings. The cached draft is used at stop
    time if the transcript hasn't grown significantly (>20% growth),
    avoiding the full LLM call during finalization.

    The loop self-terminates when ``state.current_meeting`` changes or
    is cleared.

    Telemetry is updated at every iteration into
    ``state._eager_summary_metrics`` so we can prove the loop is alive
    AND making progress without parsing logs. Surfaces via
    ``GET /api/admin/eager-summary-status``.

    Tuning history (A2 step 3, finalised after telemetry landed):
      * Warmup 300s → 60s — meeting `3db4286e-...` had only 531 draft
        events at stop vs 1565 finals (regen path). 5min warmup meant
        the draft barely had time to materialise; 60s gets a first
        draft before most real-world meetings hit their first cluster
        of finals.
      * Priority 5 → -5 — translation still preempts (it's at -10);
        but TTS / furigana / refinement no longer block draft
        generation entirely under contention.
    """
    import time as _time

    from meeting_scribe.server_support.summary_status import (
        classify_summary_error,
    )

    metrics = state._eager_summary_metrics

    metrics.last_skipped_reason = "warmup"
    await asyncio.sleep(60)
    metrics.last_skipped_reason = None

    while state.current_meeting and state.current_meeting.meeting_id == meeting_id:
        try:
            meeting_dir = state.storage._meeting_dir(meeting_id)
            journal_path = meeting_dir / "journal.jsonl"
            if not journal_path.exists():
                metrics.last_skipped_reason = "no_journal"
                await asyncio.sleep(300)
                continue

            # Count current finals to decide if a draft is worthwhile
            n_finals = 0
            for line in journal_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if e.get("is_final") and e.get("text", "").strip():
                    n_finals += 1

            # Skip if too few events (< 50) or transcript hasn't grown
            # significantly since last draft (< 30% growth)
            if n_finals < 50 or n_finals == state._eager_summary_event_count:
                metrics.last_skipped_reason = "no_growth"
                await asyncio.sleep(300)
                continue
            if (
                state._eager_summary_event_count > 0
                and n_finals < state._eager_summary_event_count * 1.3
            ):
                metrics.last_skipped_reason = "no_growth"
                await asyncio.sleep(300)
                continue

            logger.info(
                "Eager summary: generating draft (%d events, prev=%d)",
                n_finals,
                state._eager_summary_event_count,
            )
            metrics.last_start_at = _time.monotonic()
            metrics.last_skipped_reason = None
            metrics.in_flight = True
            metrics.runs_total += 1

            from meeting_scribe.summary import generate_draft_summary

            draft, event_count = await generate_draft_summary(
                meeting_dir,
                vllm_url=state.config.translate_vllm_url,
            )

            metrics.in_flight = False
            if draft and state.current_meeting and state.current_meeting.meeting_id == meeting_id:
                state._eager_summary_cache = draft
                state._eager_summary_event_count = event_count
                metrics.last_success_at = _time.monotonic()
                metrics.last_error_code = None
                metrics.draft_event_count_at_last_run = event_count
                logger.info(
                    "Eager summary: draft cached (%d events, %.0fms)",
                    event_count,
                    draft.get("metadata", {}).get("generation_ms", 0),
                )

        except asyncio.CancelledError:
            metrics.in_flight = False
            raise
        except Exception as e:
            metrics.in_flight = False
            metrics.errors_total += 1
            metrics.last_error_code = classify_summary_error(e).value
            logger.warning(
                "eager summary loop error meeting=%s error_code=%s",
                meeting_id,
                metrics.last_error_code,
            )
            logger.debug("eager summary loop detail meeting=%s exc=%r", meeting_id, e)

        await asyncio.sleep(300)  # 5 min between drafts to avoid GPU pressure

    logger.info("Eager summary loop exiting (meeting changed or stopped)")


async def _speaker_pulse_loop() -> None:
    """Periodic speaker pulse broadcast (every 200 ms).

    Sends active speaker info to all WebSocket clients for smoother
    speaker indicator animations. "Currently speaking" = the single
    most recently heard speaker, within a short recency window.

    The window must be short enough that natural back-and-forth
    doesn't leave the previous speaker lit up while the next one
    starts — 2 s is wide enough that A/B conversations highlight
    both seats at once. 700 ms still survives the small silence gaps
    inside an utterance but collapses to one seat during a handoff.
    """
    SPEAKING_WINDOW_MS = 700

    while True:
        await asyncio.sleep(0.2)

        if not state.ws_connections or not state.current_meeting:
            continue

        now = time.monotonic()
        meeting_elapsed_ms = int((now - state.metrics.meeting_start) * 1000)
        active_speakers: list[dict] = []

        # Pick the SINGLE most-recently-active detected speaker within
        # the window. Conversations rarely have true overlap, and
        # highlighting only the latest speaker prevents the previous
        # speaker's seat from lingering in a pulse during a handoff.
        most_recent = None
        most_recent_last_seen = -1
        for i, speaker in enumerate(state.detected_speakers):
            if not speaker.last_seen_ms:
                continue
            age_ms = meeting_elapsed_ms - speaker.last_seen_ms
            if 0 <= age_ms < SPEAKING_WINDOW_MS and speaker.last_seen_ms > most_recent_last_seen:
                most_recent = (i, speaker)
                most_recent_last_seen = speaker.last_seen_ms

        if most_recent is not None:
            i, speaker = most_recent
            active_speakers.append(
                {
                    "seat_index": i,
                    "name": speaker.display_name,
                    "confidence": speaker.match_confidence,
                }
            )

        # From diarization backend — only when no enrolled speaker is
        # already claiming the pulse. Also tightened to the same
        # window so diarization tails don't drag previous clusters
        # along.
        if (
            not active_speakers
            and state.diarize_backend
            and hasattr(state.diarize_backend, "get_results_for_range")
        ):
            recent_results = state.diarize_backend.get_results_for_range(
                max(0, meeting_elapsed_ms - SPEAKING_WINDOW_MS),
                meeting_elapsed_ms,
            )
            if recent_results:
                latest = max(recent_results, key=lambda dr: dr.end_ms)
                active_speakers.append(
                    {
                        "cluster_id": latest.cluster_id,
                        "confidence": latest.confidence,
                    }
                )

        pulse_data = json.dumps(
            {
                "type": "speaker_pulse",
                "active_speakers": active_speakers,
                "timestamp_ms": int((now - state.metrics.meeting_start) * 1000),
            }
        )

        dead: list[WebSocket] = []
        for ws in list(state.ws_connections):
            try:
                await ws.send_text(pulse_data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            state.ws_connections.discard(ws)


async def _speaker_catchup_loop() -> None:
    """Background catch-up task for retroactive speaker attribution.

    Fixes the timing mismatch between ASR (finalizes in ~1-2 s) and
    diarization (buffers 4 s before processing). Events that landed
    in ``_process_event`` before diarization had results sit in
    ``state._pending_speaker_events``. Every 400 ms this loop:

    1. Walks the pending queue oldest-first
    2. Asks ``state.diarize_backend.get_results_for_range`` for each
       pending event's range
    3. If results exist, attaches cluster_id, writes a new revision
       to the journal, and re-broadcasts the event — the frontend
       updates the block in-place by segment_id across all views
       (admin, popout, guest)
    4. Also applies enrolled-speaker names retroactively when
       diarization embeddings match a later-enrolled voice
    5. Gives up on events older than 45 s (diarization has processed
       that region long ago; if still no result, it was likely
       silence)
    """
    # WARNING level so it shows up in logs regardless of handler config
    logger.warning("Speaker catch-up loop started (aggressive mode: 400ms poll)")
    MAX_AGE_SECONDS = 45.0
    POLL_INTERVAL = 0.4  # Aggressive polling — GB10 has plenty of headroom
    _tick_count = 0
    _resolved_count = 0

    try:
        while True:
            await asyncio.sleep(POLL_INTERVAL)
            _tick_count += 1

            # Heartbeat every 10 s so we can see the loop is alive +
            # see what state the diarize cache is in while we're
            # debugging.
            if _tick_count % 25 == 0:
                cache_size = (
                    len(getattr(state.diarize_backend, "_result_cache", {}))
                    if state.diarize_backend
                    else 0
                )
                logger.warning(
                    "Catch-up heartbeat: pending=%d, diarize_cache=%d, resolved_total=%d",
                    len(state._pending_speaker_events),
                    cache_size,
                    _resolved_count,
                )

            # Drain any pending centroid-consolidation renames (fix 4)
            # and broadcast them to the UI. The client's speaker
            # registry then collapses "Speaker 41" onto the surviving
            # cluster's label so live view matches the post-finalize
            # view without waiting for the meeting to end.
            if state.diarize_backend is not None:
                pending_renames = getattr(state.diarize_backend, "_cluster_rename", None)
                if pending_renames:
                    renames = dict(pending_renames)
                    pending_renames.clear()
                    if renames:
                        logger.info(
                            "Broadcasting %d speaker_remap(s) to UI",
                            len(renames),
                        )
                        await _broadcast_json(
                            {
                                "type": "speaker_remap",
                                "renames": {str(k): v for k, v in renames.items()},
                            }
                        )

            if not state.current_meeting or not state.diarize_backend:
                continue
            if not state._pending_speaker_events:
                continue

            now = time.monotonic()
            to_resolve: list[str] = []
            to_evict: list[str] = []

            for segment_id, event in list(state._pending_speaker_events.items()):
                age = now - state._pending_speaker_timestamps.get(segment_id, now)
                if age > MAX_AGE_SECONDS:
                    to_evict.append(segment_id)
                    continue

                # Look up diarization results for this event's time range
                try:
                    results = state.diarize_backend.get_results_for_range(
                        event.start_ms,
                        event.end_ms,
                    )
                except Exception:
                    continue

                if not results:
                    continue

                # Overlap-aware speaker attribution: primary =
                # longest-overlapping cluster, secondaries kept when
                # co-speech is meaningful. Same thresholds as
                # ``_attach_speakers_to_events`` in reprocess.py so
                # live + finalize paths agree.
                def _overlap(dr):
                    return max(
                        0,
                        min(dr.end_ms, event.end_ms) - max(dr.start_ms, event.start_ms),
                    )

                ev_dur = max(1, event.end_ms - event.start_ms)
                overlap_by_cluster: dict[int, float] = {}
                from meeting_scribe.backends.diarize_sortformer import (
                    DiarizationResult as _DR,
                )

                results_by_cluster: dict[int, _DR] = {}
                for dr in results:
                    ov = _overlap(dr)
                    if ov <= 0:
                        continue
                    overlap_by_cluster[dr.cluster_id] = (
                        overlap_by_cluster.get(dr.cluster_id, 0) + ov
                    )
                    # Keep the representative result (max-confidence)
                    # per cluster.
                    prev = results_by_cluster.get(dr.cluster_id)
                    if prev is None or (dr.confidence or 0) > (getattr(prev, "confidence", 0) or 0):
                        results_by_cluster[dr.cluster_id] = dr
                if not overlap_by_cluster:
                    continue
                ranked_cids = sorted(overlap_by_cluster.items(), key=lambda kv: -kv[1])
                primary_cid, primary_ov = ranked_cids[0]
                best = results_by_cluster[primary_cid]

                # Resolve identity via enrollment if embedding is available
                identity = None
                identity_conf = 0.0
                source = "diarization"
                if best.embedding is not None and state.enrollment_store.speakers:
                    try:
                        from meeting_scribe.speaker.verification import (
                            cosine_similarity,
                        )

                        best_score = 0.0
                        best_name = None
                        for (
                            eid,
                            name,
                            enrolled_emb,
                        ) in state.enrollment_store.get_all_embeddings():
                            score = cosine_similarity(best.embedding, enrolled_emb)
                            if score > best_score:
                                best_score = score
                                best_name = name
                        if best_score > 0.55 and best_name:
                            identity = best_name
                            identity_conf = best_score
                            source = "enrolled"
                    except Exception:
                        pass

                # Build the updated speakers list: primary first, then
                # any secondary clusters whose overlap is ≥ 30 % of
                # the event AND ≥ 50 % of the primary's overlap.
                updated_speakers = [
                    SpeakerAttribution(
                        cluster_id=best.cluster_id,
                        identity=identity,
                        identity_confidence=identity_conf,
                        source=source,
                    )
                ]
                for cid, ov in ranked_cids[1:]:
                    if ov / ev_dur < 0.30 or ov / primary_ov < 0.50:
                        continue
                    dr2 = results_by_cluster[cid]
                    updated_speakers.append(
                        SpeakerAttribution(
                            cluster_id=dr2.cluster_id,
                            identity=None,
                            identity_confidence=dr2.confidence or 0.0,
                            source="diarization_overlap",
                        )
                    )
                updated = event.model_copy(
                    update={
                        "speakers": updated_speakers,
                        "revision": (event.revision or 0) + 1,
                    }
                )

                # Persist the new revision to the journal (readers
                # dedup by segment_id keeping highest revision).
                if state.current_meeting:
                    try:
                        state.storage.append_event(state.current_meeting.meeting_id, updated)
                    except Exception as e:
                        logger.debug("Catch-up journal append failed: %s", e)

                # Broadcast to all live views (admin, popout, guest) so
                # their rendered blocks update in place by segment_id.
                try:
                    await _broadcast(updated)
                except Exception:
                    pass

                to_resolve.append(segment_id)
                _resolved_count += 1

                logger.warning(
                    "Speaker catch-up: seg=%s age=%.1fs → cluster %d%s",
                    segment_id,
                    age,
                    best.cluster_id,
                    f" ({identity})" if identity else "",
                )

            # Clean up
            for sid in to_resolve + to_evict:
                state._pending_speaker_events.pop(sid, None)
                state._pending_speaker_timestamps.pop(sid, None)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception("Speaker catch-up loop crashed: %s", e)
