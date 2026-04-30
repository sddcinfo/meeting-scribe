"""Meeting lifecycle endpoints — start / stop / cancel / dev-reset / resume / finalize / reprocess.

Seven routes that mutate the active recording state. All
start/stop/cancel paths run under the singleton lifecycle lock
(``server_support.lifecycle_lock._get_meeting_lifecycle_lock``) so
a UI double-click or browser retry can't race two handlers through
the "create new meeting + open audio writer" path.

Several helpers still live in ``server.py`` for now —
``_speaker_pulse_loop``, ``_speaker_catchup_loop``,
``_eager_summary_loop``, ``_start_wifi_ap``, ``_stop_wifi_ap``,
``_generate_speaker_data``, ``_generate_timeline``. The handlers
lazy-import them at call time so this module loads without an
import cycle through ``meeting_scribe.server``. Once those move out
of ``server.py`` the lazy imports get rewritten as top-level imports.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import time
from contextlib import suppress
from typing import Any

import fastapi
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from meeting_scribe.hotspot.ap_control import _start_wifi_ap, _stop_wifi_ap
from meeting_scribe.models import MeetingMeta, MeetingState
from meeting_scribe.runtime import state
from meeting_scribe.server_support.active_meeting import (
    _clear_active_meeting,
    _persist_active_meeting,
)
from meeting_scribe.server_support.broadcast import _broadcast_json
from meeting_scribe.server_support.lifecycle_lock import _get_meeting_lifecycle_lock
from meeting_scribe.server_support.meeting_artifacts import (
    _generate_speaker_data,
    _generate_timeline,
)
from meeting_scribe.server_support.refinement_drains import (
    _drain_refinement,
    _DrainEntry,
    _evict_completed_drains,
    _next_drain_id,
    _refinement_drains,
)
from meeting_scribe.server_support.safe_paths import _safe_meeting_dir
from meeting_scribe.server_support.sessions import (
    _get_draft_layout,
    _get_session_id,
    _set_draft_layout,
)
from meeting_scribe.server_support.summary_status import (
    SummaryStatus,
    classify_summary_error,
    next_attempt_id,
    write_status,
)
from meeting_scribe.server_support.voice_seed import _seed_tts_from_enrollments_async

logger = logging.getLogger(__name__)

router = APIRouter()


def _persist_exclusive_segments(meeting_dir, exclusive_segments: list[dict]) -> None:
    """Write the raw community-1 exclusive timeline as a JSON artifact.

    Lives at ``<meeting_dir>/speaker_lanes_exclusive.json`` parallel to
    the existing ``speaker_lanes.json`` (which is event-aligned).  This
    artifact is the frame-level single-speaker timeline straight from
    pyannote 4.x ``DiarizeOutput.exclusive_speaker_diarization`` —
    cluster ids match the merged-global numbering used in the standard
    diarize pipeline.

    Additive on the artifact set; UI consumers ignore unknown files.
    See ``plans/2026-Q3-followups.md`` Phase C2 for the rationale.
    """
    if not exclusive_segments:
        return
    path = meeting_dir / "speaker_lanes_exclusive.json"
    body = {
        "schema_version": 1,
        "segments": [
            {
                "start_ms": s["start_ms"],
                "end_ms": s["end_ms"],
                "cluster_id": s["cluster_id"],
            }
            for s in exclusive_segments
        ],
    }
    path.write_text(_json.dumps(body, indent=2))
    logger.info(
        "Wrote speaker_lanes_exclusive.json: %d frame-level segments",
        len(exclusive_segments),
    )


async def _meeting_start_preflight() -> JSONResponse | None:
    """Synthetic-inference preflight gate (W4). Runs the same probe
    contract that the W6b recovery supervisor uses, against ASR +
    translate + diarize. Wait-with-deadline admission: re-runs the
    probes on a 2 s cadence up to a total budget (default 30 s,
    override via SCRIBE_PREFLIGHT_BUDGET_S) so a normal cold-start
    warmup completes before the meeting fails.

    Returns ``None`` on success; returns a 503 ``JSONResponse`` on
    REQUIRED-backend failure after the budget is exhausted. Diarize
    is WARNING-ONLY (logged, never blocks).

    Failure response shape MATCHES the existing deep_backend_health
    gate so the frontend's "Backends not ready" modal renders the
    same way (``data.not_ready[].backend / .detail``).

    Probe contract: HTTP 200 + valid response schema. NOT non-empty
    transcribed text — the fixture is a 200 Hz tone, not speech, so
    a healthy backend may legitimately return empty content."""
    from meeting_scribe.runtime.synthetic_probe import (
        asr_synthetic_probe,
        diarize_synthetic_probe,
        translate_synthetic_probe,
    )

    cfg = state.config
    asr_url = cfg.asr_vllm_url
    asr_model = cfg.asr_model
    translate_url = cfg.translate_vllm_url
    translate_model = cfg.translate_vllm_model or "Qwen/Qwen3.6-35B-A3B-FP8"
    diarize_url = cfg.diarize_url

    budget_s = float(os.environ.get("SCRIBE_PREFLIGHT_BUDGET_S", "30"))
    retry_interval_s = float(os.environ.get("SCRIBE_PREFLIGHT_RETRY_INTERVAL_S", "2"))

    deadline = time.monotonic() + budget_s
    attempt = 0
    last_asr = None
    last_translate = None
    last_diarize = None

    while True:
        attempt += 1
        # Run all three probes concurrently — they hit independent
        # backends, so latency is bounded by the slowest, not the sum.
        asr_result, translate_result, diarize_result = await asyncio.gather(
            asr_synthetic_probe(asr_url, asr_model, state.metrics.asr_request_rtt_ms),
            translate_synthetic_probe(
                translate_url, translate_model, state.metrics.translate_request_rtt_ms
            ),
            diarize_synthetic_probe(diarize_url, state.metrics.diarize_request_rtt_ms),
        )
        last_asr, last_translate, last_diarize = asr_result, translate_result, diarize_result

        if asr_result.ok and translate_result.ok:
            # Required backends green — succeed even if diarize is
            # still warming. Diarize is enrichment, not a blocker.
            logger.info(
                "Meeting-start preflight passed on attempt %d "
                "(asr=%.0fms translate=%.0fms diarize=%s/%s/%.0fms)",
                attempt,
                asr_result.latency_ms,
                translate_result.latency_ms,
                diarize_result.status,
                ("ok" if diarize_result.ok else "warn"),
                diarize_result.latency_ms,
            )
            if not diarize_result.ok:
                logger.warning(
                    "Meeting starting with diarize_degraded "
                    "(status=%s latency=%.0fms detail=%s) — speaker "
                    "labels may be delayed until the backend recovers.",
                    diarize_result.status,
                    diarize_result.latency_ms,
                    diarize_result.detail,
                )
            return None

        # Budget check — out of time, return the latest failure shape.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        # Sleep before retrying, but don't oversleep the budget.
        await asyncio.sleep(min(retry_interval_s, max(0.0, remaining)))

    # Budget exhausted. Return the LAST attempt's failures using the
    # ``not_ready`` shape that the frontend modal already understands.
    not_ready: list[dict] = []
    if last_asr is not None and not last_asr.ok:
        not_ready.append({
            "backend": "asr",
            "detail": (
                f"probe {last_asr.status} ({last_asr.latency_ms:.0f}ms): "
                f"{last_asr.detail or 'unknown'}"
            ),
        })
    if last_translate is not None and not last_translate.ok:
        not_ready.append({
            "backend": "translate",
            "detail": (
                f"probe {last_translate.status} ({last_translate.latency_ms:.0f}ms): "
                f"{last_translate.detail or 'unknown'}"
            ),
        })
    logger.warning(
        "Refusing to start meeting after %d preflight attempts (budget %.0fs): %s",
        attempt,
        budget_s,
        not_ready,
    )
    return JSONResponse(
        {
            "error": "Backends not ready",
            "not_ready": not_ready,
            "message": (
                f"Backends did not warm up within {budget_s:.0f}s. "
                "Wait for all backend pills in the header to turn green, "
                "then try again."
            ),
        },
        status_code=503,
    )


@router.post("/api/meeting/start")
async def start_meeting(request: fastapi.Request) -> JSONResponse:
    """Start a new meeting, or return the current one if already recording.

    GATED ON DEEP BACKEND HEALTH: refuses to start if ASR, Translation, or
    Diarization can't actually serve a real request. This prevents the
    "I started the meeting but nothing was translated" failure mode
    where status lied about backend readiness.

    SERIALIZED: all concurrent start/stop calls go through a single
    asyncio.Lock. Without this, a UI double-click or browser retry
    races through the "create new meeting + open audio writer" path
    and leaves one meeting orphaned with a dangling writer.
    """
    async with _get_meeting_lifecycle_lock():
        return await _start_meeting_locked(request)


async def _start_meeting_locked(request: fastapi.Request) -> JSONResponse:
    from meeting_scribe.runtime.meeting_loops import (
        _eager_summary_loop,
        _speaker_catchup_loop,
        _speaker_pulse_loop,
    )
    from meeting_scribe.server_support.backend_health import _deep_backend_health

    # Idempotent fast path: if we're already recording, the caller's
    # intent was "make sure a meeting is running". Return the existing
    # one so duplicate clicks are harmless.
    if state.current_meeting and state.current_meeting.state == MeetingState.RECORDING:
        logger.info(
            "start_meeting called while already recording — returning existing %s",
            state.current_meeting.meeting_id,
        )
        return JSONResponse(
            {
                "meeting_id": state.current_meeting.meeting_id,
                "state": "recording",
                "resumed": True,
                "language_pair": state.current_meeting.language_pair,
            }
        )

    # ── SYNTHETIC INFERENCE PREFLIGHT (W4) ───────────────────────
    # Fail-fast admission gate: confirm that ASR + translate actually
    # respond to a real inference request, not just /v1/models.
    # Today's 2026-04-30 ASR cascade started with a backend whose
    # /health returned 200 + /v1/models returned 200 + inference was
    # wedged. The deep_backend_health gate below catches the first two
    # cases; this preflight catches the third before recording starts.
    # Disabled via SCRIBE_MEETING_PREFLIGHT=0 for emergency operator
    # bypass — see plan W4 + risk-rollback table.
    if os.environ.get("SCRIBE_MEETING_PREFLIGHT", "1") != "0":
        preflight_failure = await _meeting_start_preflight()
        if preflight_failure is not None:
            return preflight_failure

    # ── DEEP HEALTH GATE ──────────────────────────────────────────
    # Force a fresh check (bypass cache) — don't use a stale reading to
    # decide whether to start a meeting. Only ASR + Translate block
    # start; diarize, furigana, and TTS self-heal if still warming.
    health = await _deep_backend_health(force=True)
    REQUIRED = ["asr", "translate"]
    not_ready = [
        (name, health.get(name, {}).get("detail") or "not ready")
        for name in REQUIRED
        if not health.get(name, {}).get("ready")
    ]
    if not_ready:
        logger.warning(
            "Refusing to start meeting — required backends not ready: %s",
            not_ready,
        )
        return JSONResponse(
            {
                "error": "Backends not ready",
                "not_ready": [{"backend": name, "detail": detail} for name, detail in not_ready],
                "message": (
                    "Required backends ("
                    + ", ".join(n.upper() for n in REQUIRED)
                    + ") are not ready yet. Wait for them to finish loading, then retry."
                ),
            },
            status_code=503,
        )

    # Stop any previous meeting
    if state.current_meeting and state.current_meeting.state in (
        MeetingState.CREATED,
        MeetingState.RECORDING,
    ):
        with suppress(Exception):
            state.storage.transition_state(
                state.current_meeting.meeting_id, MeetingState.INTERRUPTED
            )

    # Clear slide state from previous meeting so new meeting starts fresh
    if state.slide_job_runner is not None:
        state.slide_job_runner.active_deck_id = None
        state.slide_job_runner.current_slide_index = 0
        state.slide_job_runner._active_meta = None

    # Close any previous audio writer
    if state.audio_writer:
        state.audio_writer.close()
        state.audio_writer = None

    # Parse optional language_pair from request body. Accepts 1 or 2 distinct
    # codes — length 1 is a monolingual meeting (no translation work). Invalid
    # values are rejected with 400 rather than silently falling back to the
    # default, so a typo can never masquerade as a valid meeting.
    from meeting_scribe.languages import (
        DEFAULT_LANGUAGE_PAIR,
        is_valid_languages,
        parse_languages_strict,
    )

    language_pair = list(DEFAULT_LANGUAGE_PAIR)
    try:
        body = await request.json()
    except Exception:
        body = None
    if isinstance(body, dict) and "language_pair" in body:
        lp = body["language_pair"]
        parts: list[str] | None = None
        if isinstance(lp, str):
            parts = parse_languages_strict(lp)
        elif isinstance(lp, list):
            stripped = [str(p).strip() for p in lp]
            parts = stripped if is_valid_languages(stripped) else None
        if parts is None:
            return JSONResponse(
                {
                    "error": "Invalid language_pair",
                    "message": (
                        "language_pair must be 1 or 2 distinct codes from the "
                        "language registry (see GET /api/languages)."
                    ),
                    "received": lp,
                },
                status_code=400,
            )
        language_pair = parts

    # Create new meeting and persist room setup
    meta = state.storage.create_meeting(MeetingMeta(language_pair=language_pair))

    # Fresh start: clear in-memory voice enrollments AND any seat names /
    # enrollment_ids carried over from a previous meeting.
    state.enrollment_store.clear()
    active_layout = _get_draft_layout(_get_session_id(request))
    for seat in active_layout.seats:
        seat.enrollment_id = None
        seat.speaker_name = ""
    _set_draft_layout(_get_session_id(request), active_layout)

    state.storage.save_room_layout(meta.meeting_id, active_layout)
    meeting_dir = state.storage._meeting_dir(meta.meeting_id)
    speakers_path = meeting_dir / "speakers.json"
    state.enrollment_store._storage_path = speakers_path
    state.enrollment_store._persist()
    logger.info(
        "Persisted room layout (%d seats, all reset) + empty speakers to %s",
        len(active_layout.seats),
        meeting_dir,
    )

    state.storage.transition_state(meta.meeting_id, MeetingState.RECORDING)
    state.current_meeting = meta
    state.current_meeting.state = MeetingState.RECORDING
    state.metrics.reset()
    state.metrics.meeting_start = time.monotonic()

    # Rotate any live embedded-terminal sessions so they reconnect with
    # the new meeting's history log path.
    try:
        await state._terminal_registry.close_all(reason="meeting_rotation")
    except Exception:
        logger.exception("terminal registry rotation on meeting start failed")

    # Reset ASR backend watchdog state for the new meeting
    if state.asr_backend is not None and hasattr(state.asr_backend, "_last_response_time"):
        state.asr_backend._last_response_time = None
        state.asr_backend._buffer = []
        state.asr_backend._buffer_samples = 0
        state.asr_backend._segment_id = None
        state.asr_backend._base_offset = 0
    if state.asr_backend is not None:
        state.asr_backend.audio_wall_at_start = state.metrics.meeting_start

    # Reset diarization global cluster state so each meeting starts fresh
    if state.diarize_backend is not None and hasattr(state.diarize_backend, "_global_centroids"):
        state.diarize_backend._global_centroids.clear()
        state.diarize_backend._global_centroid_counts.clear()
        state.diarize_backend._next_global_id = 1
        state.diarize_backend._last_mapping.clear()
        state.diarize_backend._result_cache.clear()
        state.diarize_backend._rolling_audio.clear()
        state.diarize_backend._rolling_samples = 0
        state.diarize_backend._rolling_start_sample = 0
        state.diarize_backend._samples_since_flush = 0
        state.diarize_backend._last_emitted_end_ms = 0

    # Reset time-proximity pseudo-speaker state so each meeting starts fresh
    state._last_pseudo_cluster_id = 0
    state._last_pseudo_end_ms = 0
    state._next_pseudo_cluster_id = 100

    # Reset TTS voice cache so last meeting's speakers don't leak in.
    if state.tts_backend is not None and hasattr(state.tts_backend, "reset_voice_cache"):
        state.tts_backend.reset_voice_cache()

    # Seed TTS voice cache from enrollment WAVs as a BACKGROUND task.
    if state.tts_backend is not None and hasattr(state.tts_backend, "seed_voice"):
        _speakers_to_seed = list(state.enrollment_store.speakers.items())
        if _speakers_to_seed:
            asyncio.create_task(
                _seed_tts_from_enrollments_async(_speakers_to_seed),
                name=f"tts-seed-{meta.meeting_id}",
            )

    # Open audio writer for recording
    state.audio_writer = state.storage.open_audio_writer(meta.meeting_id)
    state.meeting_start_time = time.monotonic()

    # Initialize speaker tracking
    state.detected_speakers = []
    from meeting_scribe.speaker.verification import SpeakerVerifier

    state.speaker_verifier = SpeakerVerifier(state.enrollment_store)

    logger.info("Audio recording started for %s", meta.meeting_id)
    _persist_active_meeting(meta.meeting_id)

    asyncio.create_task(_start_wifi_ap(meeting_id=meta.meeting_id))

    if state._speaker_pulse_task is None or state._speaker_pulse_task.done():
        state._speaker_pulse_task = asyncio.create_task(_speaker_pulse_loop())
    if state._speaker_catchup_task is None or state._speaker_catchup_task.done():
        state._speaker_catchup_task = asyncio.create_task(_speaker_catchup_loop())
    state._eager_summary_cache = None
    state._eager_summary_event_count = 0
    if state._eager_summary_task and not state._eager_summary_task.done():
        state._eager_summary_task.cancel()
    state._eager_summary_task = asyncio.create_task(_eager_summary_loop(meta.meeting_id))

    meeting_languages = list(meta.language_pair)
    if state.translation_queue is not None:
        state.translation_queue.set_languages(meeting_languages)
        state.translation_queue.bind_meeting(
            meta.meeting_id,
            history_maxlen=state.config.live_translate_context_window_ja_en,
            fragment_gated=state.config.live_translate_context_fragment_gated,
        )
    if state.asr_backend is not None:
        state.asr_backend.set_languages(meeting_languages)

    # Near-realtime refinement worker
    if state.config.refinement_enabled:
        if len(meta.language_pair) != 2:
            logger.info(
                "Refinement worker skipped for meeting %s: %d languages, bilingual only",
                meta.meeting_id,
                len(meta.language_pair),
            )
        else:
            try:
                from meeting_scribe.refinement import RefinementWorker

                force_shared = os.environ.get("SCRIBE_REFINEMENT_FORCE_SHARED_BACKEND", "0") == "1"
                refinement_translate_url = (
                    state.config.translate_vllm_url
                    if force_shared
                    else (
                        state.config.translate_offline_vllm_url or state.config.translate_vllm_url
                    )
                )
                logger.info(
                    "Refinement worker: asr=%s translate=%s (force_shared=%s)",
                    state.config.asr_vllm_url,
                    refinement_translate_url,
                    force_shared,
                )
                state.refinement_worker = RefinementWorker(
                    meeting_id=meta.meeting_id,
                    meeting_dir=state.storage._meeting_dir(meta.meeting_id),
                    asr_url=state.config.asr_vllm_url,
                    translate_url=refinement_translate_url,
                    language_pair=(meta.language_pair[0], meta.language_pair[1]),
                    context_window_segments=state.config.refinement_context_window_segments,
                )
                state.refinement_worker._start_task = asyncio.create_task(
                    state.refinement_worker.start(),
                    name=f"refine-start-{meta.meeting_id}",
                )
            except Exception:
                logger.exception("Refinement worker failed to start")
                state.refinement_worker = None

    logger.info("Meeting started: %s", meta.meeting_id)

    # Warm the translate model so the first real utterance doesn't pay
    # the compile/cold-path tax.
    if state.translate_backend is not None and not meta.is_monolingual:
        _warmup_prior_context: list[tuple[str, str]] | None = None
        if state.config.live_translate_context_window_ja_en > 0 and tuple(
            meta.language_pair[:2]
        ) == ("ja", "en"):
            _warmup_prior_context = [("warmup ja", "warmup en")]

        async def _warm_translate():
            try:
                await state.translate_backend.translate(
                    "warmup",
                    meta.language_pair[0],
                    meta.language_pair[1],
                    prior_context=_warmup_prior_context,
                )
            except Exception:
                pass

        asyncio.create_task(_warm_translate())

    return JSONResponse(
        {
            "meeting_id": meta.meeting_id,
            "state": "recording",
            "resumed": False,
            "language_pair": meta.language_pair,
        }
    )


@router.post("/api/meeting/dev-reset")
async def dev_reset_meeting() -> JSONResponse:
    """Reset the current meeting without stopping it (DEV mode only).

    Clears journal, ASR/diarize/TTS state, speakers — then re-opens
    the audio writer and lets the same meeting keep recording.
    """
    if not state.current_meeting:
        return JSONResponse({"error": "No active meeting"}, status_code=400)

    mid = state.current_meeting.meeting_id
    logger.info("DEV RESET: clearing state for meeting %s", mid)

    # 1. Flush ASR — finalize any in-flight segments then discard
    if state.asr_backend:
        async for _event in state.asr_backend.flush():
            pass

    # 2. Drain translation queue — bind-to-None-first ordering.
    if state.translation_queue:
        state.translation_queue.bind_meeting(None)
        await state.translation_queue.flush_merge_gate()
        for _i in range(50):
            if state.translation_queue.is_idle():
                break
            await asyncio.sleep(0.1)
        state.translation_queue.clear_meeting(mid)
        state.translation_queue.bind_meeting(
            mid,
            history_maxlen=state.config.live_translate_context_window_ja_en,
            fragment_gated=state.config.live_translate_context_fragment_gated,
        )

    # 2b. Tear down refinement worker if running.
    if state.refinement_worker is not None:
        _resetting_worker = state.refinement_worker
        state.refinement_worker = None
        drain_id = _next_drain_id()
        entry = _DrainEntry(
            drain_id=drain_id,
            meeting_id=mid,
            task=asyncio.create_task(
                _drain_refinement(_resetting_worker, mid, drain_id),
                name=f"refinement-drain-{mid}-{drain_id}",
            ),
            state="draining",
            started_at=time.time(),
            translate_calls=_resetting_worker.translate_call_count,
            asr_calls=_resetting_worker.asr_call_count,
            errors_at_stop=_resetting_worker.last_error_count,
        )
        _refinement_drains.append(entry)
        _evict_completed_drains()

    # 3. Flush journal to disk, then truncate it
    state.storage.flush_journal(mid)
    meeting_dir = state.storage._meeting_dir(mid)
    journal_path = meeting_dir / "journal.jsonl"
    if journal_path.exists():
        journal_path.write_text("")
    logger.info("DEV RESET: journal cleared for %s", mid)

    # 4. Reset ASR backend state for fresh segments
    if state.asr_backend is not None:
        if hasattr(state.asr_backend, "_last_response_time"):
            state.asr_backend._last_response_time = None
            state.asr_backend._buffer = []
            state.asr_backend._buffer_samples = 0
            state.asr_backend._segment_id = None
            state.asr_backend._base_offset = 0
        state.asr_backend.audio_wall_at_start = time.monotonic()

    # 5. Reset diarization global cluster state
    if state.diarize_backend is not None and hasattr(state.diarize_backend, "_global_centroids"):
        state.diarize_backend._global_centroids.clear()
        state.diarize_backend._global_centroid_counts.clear()
        state.diarize_backend._next_global_id = 1
        state.diarize_backend._last_mapping.clear()
        state.diarize_backend._result_cache.clear()
        state.diarize_backend._rolling_audio.clear()
        state.diarize_backend._rolling_samples = 0
        state.diarize_backend._rolling_start_sample = 0
        state.diarize_backend._samples_since_flush = 0
        state.diarize_backend._last_emitted_end_ms = 0

    # 6. Reset pseudo-speaker state
    state._last_pseudo_cluster_id = 0
    state._last_pseudo_end_ms = 0
    state._next_pseudo_cluster_id = 100

    # 7. Reset TTS voice cache
    if state.tts_backend is not None and hasattr(state.tts_backend, "reset_voice_cache"):
        state.tts_backend.reset_voice_cache()

    # 8. Reset audio writer — close old, open fresh
    if state.audio_writer:
        state.audio_writer.close()
    state.audio_writer = state.storage.open_audio_writer(mid)
    state.meeting_start_time = time.monotonic()
    state.metrics.reset()
    state.metrics.meeting_start = state.meeting_start_time

    # 8b. Restart refinement worker if it was in use.
    if state.config.refinement_enabled and len(state.current_meeting.language_pair) == 2:
        try:
            from meeting_scribe.refinement import RefinementWorker

            force_shared = os.environ.get("SCRIBE_REFINEMENT_FORCE_SHARED_BACKEND", "0") == "1"
            refinement_translate_url = (
                state.config.translate_vllm_url
                if force_shared
                else (state.config.translate_offline_vllm_url or state.config.translate_vllm_url)
            )
            state.refinement_worker = RefinementWorker(
                meeting_id=mid,
                meeting_dir=state.storage._meeting_dir(mid),
                asr_url=state.config.asr_vllm_url,
                translate_url=refinement_translate_url,
                language_pair=(
                    state.current_meeting.language_pair[0],
                    state.current_meeting.language_pair[1],
                ),
                context_window_segments=state.config.refinement_context_window_segments,
            )
            state.refinement_worker._start_task = asyncio.create_task(
                state.refinement_worker.start(),
                name=f"refine-start-{mid}",
            )
        except Exception:
            logger.exception("Refinement worker restart after dev-reset failed")
            state.refinement_worker = None

    # 9. Reset speaker tracking
    state.detected_speakers = []

    # 10. Broadcast reset to all connected clients so UI clears
    await _broadcast_json(
        {
            "type": "dev_reset",
            "meeting_id": mid,
            "message": "Meeting state reset — ready for next iteration",
        }
    )

    logger.info("DEV RESET complete for %s — meeting still RECORDING", mid)
    return JSONResponse(
        {
            "status": "reset",
            "meeting_id": mid,
            "message": "State cleared, meeting still recording. Start speaking to test.",
        }
    )


@router.post("/api/meetings/{meeting_id}/resume")
async def resume_meeting(meeting_id: str) -> JSONResponse:
    """Resume an interrupted meeting — continue recording into the same meeting."""
    try:
        return await _do_resume_meeting(meeting_id)
    except Exception as e:
        logger.exception("Resume failed for %s", meeting_id)
        return JSONResponse(
            {"error": f"Resume failed: {type(e).__name__}: {e}"},
            status_code=500,
        )


async def _do_resume_meeting(meeting_id: str) -> JSONResponse:
    """Inner resume implementation — wrapped for guaranteed JSON error responses."""
    from meeting_scribe.runtime.meeting_loops import (
        _speaker_catchup_loop,
        _speaker_pulse_loop,
    )

    if state.current_meeting and state.current_meeting.state == MeetingState.RECORDING:
        return JSONResponse({"error": "Another meeting is already recording"}, status_code=409)

    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir or not meeting_dir.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    meta_path = meeting_dir / "meta.json"
    if not meta_path.exists():
        return JSONResponse({"error": "Meeting metadata not found"}, status_code=404)

    meta_data = _json.loads(meta_path.read_text())
    if meta_data.get("state") not in ("interrupted", "complete"):
        return JSONResponse(
            {"error": f"Cannot resume meeting in state: {meta_data.get('state')}"}, status_code=400
        )

    lp = meta_data.get("language_pair", ["en", "ja"])
    try:
        meta = MeetingMeta(
            meeting_id=meeting_id,
            language_pair=lp if isinstance(lp, list) else ["en", "ja"],
        )
    except Exception as e:
        return JSONResponse(
            {"error": f"Corrupt language_pair in persisted meta: {e}"},
            status_code=400,
        )
    meta.state = MeetingState.RECORDING

    meta_data["state"] = "recording"
    meta_path.write_text(_json.dumps(meta_data, indent=2))

    state.current_meeting = meta
    state.metrics.reset()
    state.metrics.meeting_start = time.monotonic()

    if state.asr_backend is not None:
        state.asr_backend.audio_wall_at_start = state.metrics.meeting_start

    from meeting_scribe.storage import AudioWriterProcess

    audio_dir = meeting_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    pcm_path = audio_dir / "recording.pcm"
    writer = AudioWriterProcess(pcm_path, append=True)
    writer.start()
    state.audio_writer = writer
    state.meeting_start_time = time.monotonic()

    speakers_path = meeting_dir / "speakers.json"
    if speakers_path.exists():
        try:
            state.enrollment_store._storage_path = speakers_path
            state.enrollment_store.load()
        except Exception as e:
            logger.warning("Failed to load speaker enrollment for resume: %s", e)

    state.detected_speakers = []
    from meeting_scribe.speaker.verification import SpeakerVerifier

    state.speaker_verifier = SpeakerVerifier(state.enrollment_store)

    meeting_languages = list(meta.language_pair)
    if state.translation_queue is not None:
        state.translation_queue.set_languages(meeting_languages)
    if state.asr_backend is not None:
        state.asr_backend.set_languages(meeting_languages)

    logger.info("Meeting resumed: %s (appending audio)", meeting_id)
    _persist_active_meeting(meeting_id)

    asyncio.create_task(_start_wifi_ap(meeting_id=meeting_id))

    if state._speaker_pulse_task is None or state._speaker_pulse_task.done():
        state._speaker_pulse_task = asyncio.create_task(_speaker_pulse_loop())
    if state._speaker_catchup_task is None or state._speaker_catchup_task.done():
        state._speaker_catchup_task = asyncio.create_task(_speaker_catchup_loop())

    return JSONResponse(
        {
            "meeting_id": meeting_id,
            "state": "recording",
            "resumed": True,
            "language_pair": meta.language_pair,
        }
    )


@router.post("/api/meetings/{meeting_id}/finalize")
async def finalize_meeting(
    meeting_id: str,
    force: bool = False,
    expected_speakers: int | None = None,
) -> JSONResponse:
    """Finalize/reprocess a meeting — generate timeline, speakers, summary.

    Runs full-audio diarization on the recording so speaker separation
    works even when the live streaming path didn't produce results. The
    new diarization cluster_ids are written back to the journal as new
    revisions so subsequent reads dedup to the diarized version.
    """
    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir or not meeting_dir.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    meta_path = meeting_dir / "meta.json"
    if not meta_path.exists():
        return JSONResponse({"error": "Meeting metadata missing"}, status_code=404)
    meta = _json.loads(meta_path.read_text())

    journal_path = meeting_dir / "journal.jsonl"
    pcm_path = meeting_dir / "audio" / "recording.pcm"

    diarize_info = {"ran": False, "segments": 0, "unique_speakers": 0}
    audio_quality = None
    if journal_path.exists() and pcm_path.exists():
        try:
            from meeting_scribe.pipeline.diarize import _diarize_full_audio
            from meeting_scribe.pipeline.quality import _audio_quality_report
            from meeting_scribe.pipeline.speaker_attach import (
                _attach_speakers_to_events,
            )

            events_by_sid: dict[str, dict] = {}
            corrections: dict[str, str] = {}
            for line in journal_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    e = _json.loads(line)
                except Exception:
                    continue
                if e.get("type") == "speaker_correction":
                    corrections[e.get("segment_id", "")] = e.get("speaker_name", "")
                    continue
                if not e.get("is_final") or not e.get("text"):
                    continue
                sid = e.get("segment_id")
                if not sid:
                    continue
                existing = events_by_sid.get(sid)
                if not existing:
                    events_by_sid[sid] = e
                else:
                    e_has_tr = bool((e.get("translation") or {}).get("text"))
                    ex_has_tr = bool((existing.get("translation") or {}).get("text"))
                    e_rev = e.get("revision", 0)
                    ex_rev = existing.get("revision", 0)
                    if e_has_tr and not ex_has_tr:
                        events_by_sid[sid] = e
                    elif not e_has_tr and ex_has_tr:
                        if e_rev > ex_rev:
                            e["translation"] = existing["translation"]
                            events_by_sid[sid] = e
                    elif e_rev > ex_rev:
                        if ex_has_tr and not e_has_tr:
                            e["translation"] = existing["translation"]
                        events_by_sid[sid] = e

            if events_by_sid:
                logger.info(
                    "Finalize: running full-audio diarization on %d events (%.0fMB audio)",
                    len(events_by_sid),
                    pcm_path.stat().st_size / 1024 / 1024,
                )
                pcm_data = pcm_path.read_bytes()
                audio_quality = _audio_quality_report(pcm_data)
                if not audio_quality["usable"]:
                    logger.warning(
                        "Finalize: audio is %0.1f%% zero-filled "
                        "(longest gap %dms) — diarization quality will be poor.",
                        audio_quality["zero_fill_pct"],
                        audio_quality["longest_zero_run_ms"],
                    )
                if expected_speakers is not None and 1 <= expected_speakers <= 12:
                    diar_min = expected_speakers
                    diar_max = expected_speakers
                    expected_for_merge = expected_speakers
                    logger.info(
                        "Finalize: pinning diarization to %d speakers (caller hint)",
                        expected_speakers,
                    )
                else:
                    diar_min = 2
                    diar_max = 8
                    expected_for_merge = None

                diarize_result = await _diarize_full_audio(
                    pcm_data,
                    state.config.diarize_url,
                    max_speakers=diar_max,
                    min_speakers=diar_min,
                    expected_speakers=expected_for_merge,
                )

                if diarize_result:
                    events_list = list(events_by_sid.values())
                    _attach_speakers_to_events(
                        events_list,
                        diarize_result.segments,
                        exclusive_segments=diarize_result.exclusive_segments,
                    )

                    # Persist the raw exclusive timeline alongside the
                    # standard diarization outputs.  Additive on the
                    # artifact set; UI consumers ignore unknown files.
                    # See plans/2026-Q3-followups.md C2 for the rationale.
                    _persist_exclusive_segments(meeting_dir, diarize_result.exclusive_segments)

                    with journal_path.open("a") as f:
                        for event in events_list:
                            if not event.get("speakers"):
                                continue
                            new_revision = (event.get("revision", 0) or 0) + 1
                            event["revision"] = new_revision
                            f.write(_json.dumps(event, ensure_ascii=False) + "\n")

                    unique = len({s["cluster_id"] for s in diarize_result.segments})
                    diarize_info = {
                        "ran": True,
                        "segments": len(diarize_result.segments),
                        "exclusive_segments": len(diarize_result.exclusive_segments),
                        "unique_speakers": unique,
                    }
                    logger.info(
                        "Finalize diarization: %d standard / %d exclusive segments, "
                        "%d unique global speakers",
                        len(diarize_result.segments),
                        len(diarize_result.exclusive_segments),
                        unique,
                    )
                else:
                    logger.warning("Finalize: diarization returned no segments")
        except Exception as e:
            logger.exception("Finalize diarization failed: %s", e)

    if journal_path.exists():
        _generate_speaker_data(
            meeting_dir,
            journal_path,
            _json,
            expected_speakers=expected_speakers,
        )

        if diarize_info.get("ran") and diarize_info.get("unique_speakers", 0) > 0:
            try:
                _ds_raw = (meeting_dir / "detected_speakers.json").read_text()
                _ds = _json.loads(_ds_raw)
                diarize_count = diarize_info["unique_speakers"]
                detected_count = len(_ds)
                if detected_count < diarize_count:
                    lost = diarize_count - detected_count
                    logger.warning(
                        "finalize: diarize found %d unique speakers but "
                        "only %d made it into state.detected_speakers.json "
                        "(%d lost). Lost speakers live in ASR gaps — "
                        "run /api/meetings/%s/reprocess to recover them.",
                        diarize_count,
                        detected_count,
                        lost,
                        meeting_id,
                    )
                    diarize_info["lost_in_asr_gap"] = lost
            except Exception:
                pass

    timeline_path = meeting_dir / "timeline.json"
    _generate_timeline(meeting_id)

    summary_path = meeting_dir / "summary.json"
    summary = {}
    if (not summary_path.exists() or force) and journal_path.exists():
        attempt_id = next_attempt_id(meeting_dir)
        write_status(
            meeting_dir,
            SummaryStatus.GENERATING,
            attempt_id=attempt_id,
            journal_path=journal_path,
        )
        try:
            from meeting_scribe.summary import generate_summary

            summary = await generate_summary(meeting_dir, vllm_url=state.config.translate_vllm_url)
            if isinstance(summary, dict) and "error" in summary:
                raise RuntimeError(summary["error"])
            write_status(
                meeting_dir,
                SummaryStatus.COMPLETE,
                attempt_id=attempt_id,
                journal_path=journal_path,
            )
        except Exception as e:
            code = classify_summary_error(e)
            logger.warning(
                "summary failed meeting=%s error_code=%s attempt=%d",
                meeting_id,
                code.value,
                attempt_id,
            )
            logger.debug("summary failure detail meeting=%s exc=%r", meeting_id, e)
            write_status(
                meeting_dir,
                SummaryStatus.ERROR,
                attempt_id=attempt_id,
                journal_path=journal_path,
                error_code=code,
            )
            summary = {"error_code": code.value}

    meta["state"] = "complete"
    meta_path.write_text(_json.dumps(meta, indent=2))

    return JSONResponse(
        {
            "status": "finalized",
            "meeting_id": meeting_id,
            "has_timeline": timeline_path.exists(),
            "has_summary": "error" not in summary,
            "has_speakers": (meeting_dir / "detected_speakers.json").exists(),
            "diarization": diarize_info,
            "audio_quality": audio_quality,
        }
    )


@router.post("/api/meetings/{meeting_id}/reprocess")
async def reprocess_meeting_endpoint(
    meeting_id: str,
    expected_speakers: int | None = None,
) -> JSONResponse:
    """Full reprocess — re-run ASR + translation + diarization on raw audio."""
    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir or not meeting_dir.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    pcm_path = meeting_dir / "audio" / "recording.pcm"
    if not pcm_path.exists():
        return JSONResponse({"error": "No audio recording found"}, status_code=404)

    meta_path = meeting_dir / "meta.json"
    language_pair = ("en", "ja")
    if meta_path.exists():
        meta = _json.loads(meta_path.read_text())
        lp = meta.get("language_pair", ["en", "ja"])
        language_pair = (lp[0], lp[1]) if isinstance(lp, list) and len(lp) == 2 else ("en", "ja")

    from meeting_scribe.reprocess import reprocess_meeting

    result = await reprocess_meeting(
        meeting_dir,
        state.storage,
        asr_url=state.config.asr_vllm_url,
        translate_url=state.config.translate_vllm_url,
        diarize_url=state.config.diarize_url,
        language_pair=language_pair,
        expected_speakers=expected_speakers,
    )

    if "error" in result:
        return JSONResponse(result, status_code=400)

    return JSONResponse(
        {
            "status": "reprocessed",
            "meeting_id": meeting_id,
            **result,
        }
    )


@router.post("/api/meeting/cancel")
async def cancel_meeting() -> JSONResponse:
    """Cancel the current meeting — discard all artifacts without finalization."""
    async with _get_meeting_lifecycle_lock():
        return await _cancel_meeting_locked()


async def _cancel_meeting_locked() -> JSONResponse:
    """Cancel: stop recording, delete everything, no summary/diarization."""
    if not state.current_meeting:
        return JSONResponse({"error": "No active meeting"}, status_code=400)

    mid = state.current_meeting.meeting_id
    logger.info("Cancelling meeting: %s", mid)

    if state.audio_writer:
        state.audio_writer.close()
        state.audio_writer = None
        state.meeting_start_time = 0.0

    if state.refinement_worker is not None:
        _cancelling_worker = state.refinement_worker
        state.refinement_worker = None
        with suppress(Exception):
            await asyncio.wait_for(_cancelling_worker.stop(), timeout=2.0)

    if state.translation_queue is not None:
        state.translation_queue.bind_meeting(None)
        state.translation_queue.clear_meeting(mid)

    if state._eager_summary_task and not state._eager_summary_task.done():
        state._eager_summary_task.cancel()
        state._eager_summary_task = None
    state._eager_summary_cache = None
    state._eager_summary_event_count = 0

    if state._speaker_pulse_task and not state._speaker_pulse_task.done():
        state._speaker_pulse_task.cancel()
        state._speaker_pulse_task = None
    if state._speaker_catchup_task and not state._speaker_catchup_task.done():
        state._speaker_catchup_task.cancel()
        state._speaker_catchup_task = None
    state._pending_speaker_events.clear()
    state._pending_speaker_timestamps.clear()

    for task in list(state._background_tasks):
        if not task.done():
            task.cancel()
    state._background_tasks.clear()

    if state.translation_queue:
        for item in list(getattr(state.translation_queue, "_items", [])):
            if not getattr(item, "started", False):
                item.cancelled = True

    if state.slide_job_runner is not None:
        state.slide_job_runner.cleanup_meeting(mid)

    import shutil as _cancel_shutil

    meeting_dir = state.storage._meeting_dir(mid)
    if meeting_dir.exists():
        _cancel_shutil.rmtree(meeting_dir)
        logger.info("Cancelled meeting %s — all artifacts deleted", mid)

    await _broadcast_json(
        {
            "type": "meeting_cancelled",
            "meeting_id": mid,
        }
    )

    await asyncio.sleep(0.3)
    for ws in list(state.ws_connections):
        with suppress(Exception):
            await ws.close(1000, "Meeting cancelled")
    state.ws_connections.clear()
    state._client_prefs.clear()
    for ws in list(state._audio_out_clients):
        with suppress(Exception):
            await ws.close(1000, "Meeting cancelled")
    state._audio_out_clients.clear()
    state._audio_out_prefs.clear()

    state.current_meeting = None
    _clear_active_meeting()
    asyncio.create_task(_stop_wifi_ap())

    return JSONResponse({"status": "cancelled", "meeting_id": mid})


@router.post("/api/meeting/stop")
async def stop_meeting(preserve_empty: bool = False) -> JSONResponse:
    """Stop the current meeting (serialized via the lifecycle lock)."""
    async with _get_meeting_lifecycle_lock():
        return await _stop_meeting_locked(preserve_empty)


async def _stop_meeting_locked(preserve_empty: bool = False) -> JSONResponse:
    from meeting_scribe.pipeline.transcript_event import _process_event

    if not state.current_meeting:
        return JSONResponse({"error": "No active meeting"}, status_code=400)

    mid = state.current_meeting.meeting_id
    _finalize_t0 = time.monotonic()
    if os.environ.get("SCRIBE_PRESERVE_EMPTY_ON_STOP", "0") == "1":
        preserve_empty = True
    audio_duration_s = state.audio_writer.duration_ms / 1000 if state.audio_writer else 0
    est_seconds = max(10, int(audio_duration_s * 0.3))

    await _broadcast_json(
        {
            "type": "finalize_progress",
            "step": 1,
            "total_steps": 5,
            "label": "Flushing ASR...",
            "eta_seconds": est_seconds,
        }
    )

    _step_t0 = time.monotonic()
    if state.asr_backend:
        async for event in state.asr_backend.flush():
            await _process_event(event)
    logger.info(
        "finalize.step=asr_flush meeting=%s took=%.1fs",
        mid,
        time.monotonic() - _step_t0,
    )

    state.current_meeting = None

    if state.slide_job_runner is not None:
        state.slide_job_runner.clear_active_state(mid)

    await _broadcast_json(
        {
            "type": "finalize_progress",
            "step": 2,
            "total_steps": 5,
            "label": "Completing translations...",
            "eta_seconds": max(5, est_seconds - 5),
        }
    )

    _step_t0 = time.monotonic()
    _partial_finalize = False
    _pending_translation_count = 0
    if state.translation_queue:
        meeting_dir_for_quiesce = state.storage._meeting_dir(mid)
        # A4: race-safe quiesce — file is the authoritative store. From
        # the moment quiesce flips active, every late submit() persists
        # inline before returning. Recovery on next startup replays
        # idempotently via (meeting_id, segment_id, target_lang).
        result = await state.translation_queue.quiesce_meeting(
            mid,
            meeting_dir_for_quiesce,
            deadline_s=30.0,
        )
        if not result.drained_clean:
            _partial_finalize = True
            _pending_translation_count = result.item_count
            logger.warning(
                "finalize.translation_partial meeting=%s pending=%d backlog=%s",
                mid,
                result.item_count,
                result.backlog_path,
            )
        if result.deferred_post_quiesce:
            logger.info(
                "finalize.translation_late_submits meeting=%s persisted_to_backlog=%d",
                mid,
                result.deferred_post_quiesce,
            )
        state.translation_queue.bind_meeting(None)
        state.translation_queue.clear_meeting(mid)
    logger.info(
        "finalize.step=translation_drain meeting=%s took=%.1fs partial=%s",
        mid,
        time.monotonic() - _step_t0,
        _partial_finalize,
    )

    state.storage.flush_journal(mid)

    if state._eager_summary_task and not state._eager_summary_task.done():
        state._eager_summary_task.cancel()
        state._eager_summary_task = None

    if state.audio_writer:
        duration_ms = state.audio_writer.duration_ms
        state.audio_writer.close()
        state.audio_writer = None
        state.meeting_start_time = 0.0
        logger.info("Audio recording: %dms (%.1fs)", duration_ms, duration_ms / 1000)

    if state.refinement_worker is not None:
        _stopping_worker = state.refinement_worker
        state.refinement_worker = None
        drain_id = _next_drain_id()
        entry = _DrainEntry(
            drain_id=drain_id,
            meeting_id=mid,
            task=asyncio.create_task(
                _drain_refinement(_stopping_worker, mid, drain_id),
                name=f"refinement-drain-{mid}-{drain_id}",
            ),
            state="draining",
            started_at=time.time(),
            translate_calls=_stopping_worker.translate_call_count,
            asr_calls=_stopping_worker.asr_call_count,
            errors_at_stop=_stopping_worker.last_error_count,
        )
        _refinement_drains.append(entry)
        _evict_completed_drains()

    # ── Empty-meeting cleanup ─────────────────────────────────────
    import shutil as _empty_shutil

    _meeting_dir = state.storage._meeting_dir(mid)
    _journal_path = _meeting_dir / "journal.jsonl"
    _has_final = False
    if _journal_path.exists():
        try:
            for _line in _journal_path.read_text().splitlines():
                if not _line.strip():
                    continue
                try:
                    _e = _json.loads(_line)
                except Exception:
                    continue
                if _e.get("is_final") and (_e.get("text") or "").strip():
                    _has_final = True
                    break
        except Exception:
            _has_final = True
    if not _has_final and preserve_empty:
        logger.info("Meeting %s stopped with zero events — preserving (preserve_empty=True)", mid)
    elif not _has_final:
        # SAFETY INVARIANT: never delete a meeting dir that still has real audio.
        _pcm = _meeting_dir / "audio" / "recording.pcm"
        _has_audio = _pcm.exists() and _pcm.stat().st_size > 0
        if _has_audio:
            logger.info(
                "Meeting %s stopped with zero journal events but has "
                "%d bytes of audio — preserving for reprocess",
                mid,
                _pcm.stat().st_size,
            )
        else:
            logger.info("Meeting %s stopped with zero events — deleting", mid)
            try:
                if _meeting_dir.exists():
                    _empty_shutil.rmtree(_meeting_dir)
            except Exception as e:
                logger.warning("Failed to delete empty meeting %s: %s", mid, e)
        await _broadcast_json(
            {
                "type": "finalize_progress",
                "step": 6,
                "label": "Empty meeting discarded",
                "meeting_id": mid,
                "summary": {"discarded": True},
            }
        )
        await asyncio.sleep(0.3)
        for ws in list(state.ws_connections):
            with suppress(Exception):
                await ws.close(1000, "Meeting ended")
        state.ws_connections.clear()
        state._client_prefs.clear()
        for ws in list(state._audio_out_clients):
            with suppress(Exception):
                await ws.close(1000, "Meeting ended")
        state._audio_out_clients.clear()
        state._audio_out_prefs.clear()
        if state._speaker_pulse_task and not state._speaker_pulse_task.done():
            state._speaker_pulse_task.cancel()
            state._speaker_pulse_task = None
        if state._speaker_catchup_task and not state._speaker_catchup_task.done():
            state._speaker_catchup_task.cancel()
            state._speaker_catchup_task = None
        state._pending_speaker_events.clear()
        state._pending_speaker_timestamps.clear()
        state.current_meeting = None
        for task in list(state._background_tasks):
            if not task.done():
                task.cancel()
        state._background_tasks.clear()
        _clear_active_meeting()
        asyncio.create_task(_stop_wifi_ap())
        return JSONResponse({"status": "discarded", "meeting_id": mid, "reason": "no_events"})

    await _broadcast_json(
        {
            "type": "finalize_progress",
            "step": 3,
            "total_steps": 6,
            "label": "Saving speaker data...",
            "eta_seconds": max(3, est_seconds - 10),
        }
    )
    if state.detected_speakers:
        state.storage.save_detected_speakers(mid, state.detected_speakers)
        logger.info("Saved %d detected speakers", len(state.detected_speakers))

    meeting_dir = state.storage._meeting_dir(mid)
    journal_path = meeting_dir / "journal.jsonl"
    pcm_path = meeting_dir / "audio" / "recording.pcm"

    await _broadcast_json(
        {
            "type": "finalize_progress",
            "step": 4,
            "total_steps": 6,
            "label": "Running full-audio diarization...",
            "eta_seconds": max(5, int(audio_duration_s * 0.15)),
        }
    )

    # ── Decide whether the eager draft summary is usable ────────
    _n_final_events = 0
    if journal_path.exists():
        for _line in journal_path.read_text().splitlines():
            if not _line.strip():
                continue
            try:
                _ev = _json.loads(_line)
            except Exception:
                continue
            if _ev.get("is_final") and (_ev.get("text") or "").strip():
                _n_final_events += 1

    _use_draft = (
        state._eager_summary_cache is not None
        and state._eager_summary_event_count > 0
        and _n_final_events > 0
        and (_n_final_events - state._eager_summary_event_count) / state._eager_summary_event_count
        <= 0.20
    )
    if _use_draft:
        logger.info(
            "Eager summary: draft usable (%d draft events, %d final events, %.0f%% growth)",
            state._eager_summary_event_count,
            _n_final_events,
            (_n_final_events - state._eager_summary_event_count)
            / state._eager_summary_event_count
            * 100,
        )
    elif state._eager_summary_cache is not None:
        logger.info(
            "Eager summary: draft stale (%d draft events → %d final events), will regenerate",
            state._eager_summary_event_count,
            _n_final_events,
        )

    # ── Start summary LLM call in parallel with diarization ───
    _parallel_summary_task: asyncio.Task | None = None
    _summary_attempt_id: int = 0
    _summary_dir = state.storage._meeting_dir(mid)
    _partial_extra: dict[str, Any] = (
        {"partial": True, "pending_translation_count": _pending_translation_count}
        if _partial_finalize
        else {}
    )
    if not _use_draft:
        from meeting_scribe.summary import generate_summary as _gen_summary

        _summary_attempt_id = next_attempt_id(_summary_dir)
        write_status(
            _summary_dir,
            SummaryStatus.GENERATING,
            attempt_id=_summary_attempt_id,
            journal_path=journal_path,
            extra=_partial_extra,
        )

        async def _parallel_summary():
            return await _gen_summary(
                _summary_dir,
                vllm_url=state.config.translate_vllm_url,
            )

        _parallel_summary_task = asyncio.create_task(_parallel_summary())

    if journal_path.exists() and pcm_path.exists():
        try:
            from meeting_scribe.pipeline.diarize import _diarize_full_audio
            from meeting_scribe.pipeline.quality import _audio_quality_report
            from meeting_scribe.pipeline.speaker_attach import (
                _attach_speakers_to_events,
            )

            events_by_sid: dict[str, dict] = {}
            for line in journal_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    event = _json.loads(line)
                except Exception:
                    continue
                if event.get("type") == "speaker_correction":
                    continue
                if not event.get("is_final") or not event.get("text"):
                    continue
                sid = event.get("segment_id")
                if not sid:
                    continue
                existing = events_by_sid.get(sid)
                if not existing:
                    events_by_sid[sid] = event
                else:
                    e_has_tr = bool((event.get("translation") or {}).get("text"))
                    ex_has_tr = bool((existing.get("translation") or {}).get("text"))
                    e_rev = event.get("revision", 0)
                    ex_rev = existing.get("revision", 0)
                    if e_has_tr and not ex_has_tr:
                        events_by_sid[sid] = event
                    elif not e_has_tr and ex_has_tr:
                        if e_rev > ex_rev:
                            event["translation"] = existing["translation"]
                            events_by_sid[sid] = event
                    elif e_rev > ex_rev:
                        if ex_has_tr and not e_has_tr:
                            event["translation"] = existing["translation"]
                        events_by_sid[sid] = event

            if events_by_sid:
                # Default `shadow` so every finalize emits a structured
                # comparison line we can grep before promoting to `on`.
                # Behavior is unchanged — finalize artifacts still come
                # from the full-pass — only the comparison logging is
                # added. Set `SCRIBE_FINALIZE_DIARIZE_FAST=off` to
                # silence the comparison entirely; `=on` to skip the
                # full-pass once shadow data justifies it (see plan
                # A5: < 1% segment_disagree_rate on ≥ 5 meetings).
                _fast_mode = os.environ.get("SCRIBE_FINALIZE_DIARIZE_FAST", "shadow").lower()
                if _fast_mode not in ("off", "shadow", "on"):
                    _fast_mode = "off"

                logger.info(
                    "Stop: diarize_fast_mode=%s, %d events (%.0fMB audio)",
                    _fast_mode,
                    len(events_by_sid),
                    pcm_path.stat().st_size / 1024 / 1024,
                )

                # Always build the fast-path consolidation if we'll use
                # it OR if shadow mode is on. Cheap (no model call).
                from meeting_scribe.pipeline.diarize_consolidate import (
                    compare_diarize_results,
                    consolidate_from_events,
                )

                fast_segments: list = []
                if _fast_mode in ("shadow", "on"):
                    fast_segments = consolidate_from_events(events_by_sid)

                # Track standard + exclusive segments separately.  Fast
                # mode (event-based consolidation) only produces standard
                # segments, so exclusive stays empty there and
                # _attach_speakers_to_events falls back to standard-only
                # behaviour gracefully.
                diarize_segments: list = []
                diarize_exclusive: list = []
                if _fast_mode == "on" and fast_segments:
                    # Skip full-pass entirely.
                    diarize_segments = fast_segments
                    logger.info(
                        "finalize.step=fast_diarize meeting=%s segments=%d (skipped full-pass)",
                        mid,
                        len(fast_segments),
                    )
                else:
                    pcm_data = pcm_path.read_bytes()
                    quality = _audio_quality_report(pcm_data)
                    if not quality["usable"]:
                        logger.warning(
                            "Stop finalize: audio is %.1f%% zero-filled — diarization quality limited",
                            quality["zero_fill_pct"],
                        )
                    _diarize_t0 = time.monotonic()
                    diarize_result = await _diarize_full_audio(
                        pcm_data,
                        state.config.diarize_url,
                        max_speakers=8,
                        min_speakers=2,
                    )
                    diarize_segments = diarize_result.segments
                    diarize_exclusive = diarize_result.exclusive_segments
                    logger.info(
                        "finalize.step=full_diarize meeting=%s took=%.1fs segments=%d exclusive=%d",
                        mid,
                        time.monotonic() - _diarize_t0,
                        len(diarize_segments),
                        len(diarize_exclusive),
                    )

                    if _fast_mode == "shadow" and fast_segments:
                        cmp = compare_diarize_results(fast_segments, diarize_segments)
                        logger.info(
                            "finalize.diarize.shadow meeting=%s "
                            "full_speakers=%s fast_speakers=%s "
                            "full_segments=%s fast_segments=%s "
                            "segment_disagree_rate=%s",
                            mid,
                            cmp["full_speakers"],
                            cmp["fast_speakers"],
                            cmp["full_segments"],
                            cmp["fast_segments"],
                            cmp["segment_disagree_rate"],
                        )
                if diarize_segments or diarize_exclusive:
                    events_list = list(events_by_sid.values())
                    _attach_t0 = time.monotonic()
                    _attach_speakers_to_events(
                        events_list,
                        diarize_segments,
                        exclusive_segments=diarize_exclusive,
                    )
                    # See plans/2026-Q3-followups.md C2.
                    _persist_exclusive_segments(meeting_dir, diarize_exclusive)
                    logger.info(
                        "finalize.step=speaker_attach meeting=%s took=%.1fs events=%d",
                        mid,
                        time.monotonic() - _attach_t0,
                        len(events_list),
                    )
                    with journal_path.open("a") as f:
                        for event_dict in events_list:
                            if not event_dict.get("speakers"):
                                continue
                            event_dict["revision"] = (event_dict.get("revision", 0) or 0) + 1
                            f.write(_json.dumps(event_dict, ensure_ascii=False) + "\n")
                    unique = len({s["cluster_id"] for s in (diarize_segments or diarize_exclusive)})
                    logger.info(
                        "Stop diarization: %d standard / %d exclusive segments, "
                        "%d unique global speakers",
                        len(diarize_segments),
                        len(diarize_exclusive),
                        unique,
                    )
                else:
                    logger.warning("Stop: full-audio diarization returned no segments")
        except Exception as e:
            logger.exception("Stop-time diarization failed: %s", e)

    if journal_path.exists():
        _generate_speaker_data(meeting_dir, journal_path, _json)

    await _broadcast_json(
        {
            "type": "finalize_progress",
            "step": 5,
            "total_steps": 6,
            "label": "Generating timeline...",
            "eta_seconds": 5,
        }
    )
    _timeline_t0 = time.monotonic()
    _generate_timeline(mid)
    logger.info(
        "finalize.step=timeline meeting=%s took=%.1fs",
        mid,
        time.monotonic() - _timeline_t0,
    )

    await _broadcast_json(
        {
            "type": "finalize_progress",
            "step": 6,
            "total_steps": 6,
            "label": "Generating meeting summary...",
            "eta_seconds": 3,
        }
    )
    _summary_t0 = time.monotonic()
    summary: dict[str, Any] = {}
    if _use_draft:
        summary = state._eager_summary_cache or {}
        try:
            from meeting_scribe.summary import (
                _calculate_speaker_stats,
                build_transcript_text,
            )

            events, _ = build_transcript_text(meeting_dir)
            speakers_path = meeting_dir / "detected_speakers.json"
            speakers: list[dict] = []
            if speakers_path.exists():
                with suppress(Exception):
                    speakers = _json.loads(speakers_path.read_text())
            speaker_stats = _calculate_speaker_stats(events, speakers)
            duration_ms = max(e.get("end_ms", 0) for e in events) if events else 0
            languages = set(e.get("language", "unknown") for e in events)
            summary["speaker_stats"] = speaker_stats
            summary["metadata"].update(
                {
                    "meeting_id": mid,
                    "duration_min": round(duration_ms / 60000, 1),
                    "num_segments": len(events),
                    "num_speakers": len(speaker_stats),
                    "languages": sorted(languages - {"unknown"}),
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "is_draft": False,
                    "promoted_from_draft": True,
                }
            )
            summary_path = meeting_dir / "summary.json"
            summary_path.write_text(_json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
            logger.info(
                "Eager summary: promoted draft to final (%d events)",
                len(events),
            )
            _promote_attempt = next_attempt_id(meeting_dir)
            write_status(
                meeting_dir,
                SummaryStatus.COMPLETE,
                attempt_id=_promote_attempt,
                journal_path=journal_path,
                extra=_partial_extra,
            )
        except Exception as e:
            logger.warning("Failed to promote draft summary, regenerating: %s", e)
            _use_draft = False

    if not _use_draft:
        if _summary_attempt_id == 0:
            _summary_attempt_id = next_attempt_id(_summary_dir)
            write_status(
                _summary_dir,
                SummaryStatus.GENERATING,
                attempt_id=_summary_attempt_id,
                journal_path=journal_path,
                extra=_partial_extra,
            )
        try:
            if _parallel_summary_task:
                summary = await _parallel_summary_task
            else:
                from meeting_scribe.summary import generate_summary

                meeting_dir = state.storage._meeting_dir(mid)
                summary = await generate_summary(
                    meeting_dir,
                    vllm_url=state.config.translate_vllm_url,
                )
            if isinstance(summary, dict) and "error" in summary:
                raise RuntimeError(summary["error"])
            logger.info(
                "Meeting summary generated: %d topics, %d action items",
                len(summary.get("topics", [])),
                len(summary.get("action_items", [])),
            )
            write_status(
                _summary_dir,
                SummaryStatus.COMPLETE,
                attempt_id=_summary_attempt_id,
                journal_path=journal_path,
                extra=_partial_extra,
            )
        except Exception as e:
            code = classify_summary_error(e)
            logger.warning(
                "summary failed meeting=%s error_code=%s attempt=%d",
                mid,
                code.value,
                _summary_attempt_id,
            )
            logger.debug("summary failure detail meeting=%s exc=%r", mid, e)
            write_status(
                _summary_dir,
                SummaryStatus.ERROR,
                attempt_id=_summary_attempt_id,
                journal_path=journal_path,
                error_code=code,
                extra=_partial_extra,
            )
            summary = {"error_code": code.value}
    logger.info(
        "finalize.step=summary meeting=%s took=%.1fs",
        mid,
        time.monotonic() - _summary_t0,
    )

    state._eager_summary_cache = None
    state._eager_summary_event_count = 0

    try:
        state.storage.transition_state(mid, MeetingState.FINALIZING)
        state.storage.transition_state(mid, MeetingState.COMPLETE)
    except Exception as e:
        logger.warning("Stop transition error: %s", e)
    logger.info(
        "finalize.total meeting=%s took=%.1fs",
        mid,
        time.monotonic() - _finalize_t0,
    )

    await _broadcast_json(
        {
            "type": "finalize_progress",
            "step": 6,
            "label": "Done!",
            "meeting_id": mid,
            "summary": summary,
        }
    )
    await _broadcast_json(
        {
            "type": "meeting_stopped",
            "meeting_id": mid,
            "reason": "user_stop",
        }
    )

    await asyncio.sleep(0.5)
    for ws in list(state.ws_connections):
        with suppress(Exception):
            await ws.close(1000, "Meeting ended")
    state.ws_connections.clear()
    state._client_prefs.clear()
    for ws in list(state._audio_out_clients):
        with suppress(Exception):
            await ws.close(1000, "Meeting ended")
    state._audio_out_clients.clear()
    state._audio_out_prefs.clear()

    if state._speaker_pulse_task and not state._speaker_pulse_task.done():
        state._speaker_pulse_task.cancel()
        state._speaker_pulse_task = None
    if state._speaker_catchup_task and not state._speaker_catchup_task.done():
        state._speaker_catchup_task.cancel()
        state._speaker_catchup_task = None
    state._pending_speaker_events.clear()
    state._pending_speaker_timestamps.clear()

    state.current_meeting = None
    for task in list(state._background_tasks):
        if not task.done():
            task.cancel()
    state._background_tasks.clear()

    _clear_active_meeting()
    logger.info("Meeting stopped: %s", mid)

    asyncio.create_task(_stop_wifi_ap())

    return JSONResponse({"status": "complete", "meeting_id": mid})
