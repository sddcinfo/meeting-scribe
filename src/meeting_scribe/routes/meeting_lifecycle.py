"""Meeting lifecycle endpoints — start / stop / cancel / resume / finalize / reprocess.

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
    prefer_event,
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


# Track in-flight Phase B background tasks, keyed by meeting_id.
# Phase A inserts on spawn; the task's done_callback removes it.
# Used by the CLI ``meeting-scribe finalize status`` to enumerate
# what's running.
_phase_b_tasks: dict[str, asyncio.Task] = {}


def _is_background_finalize_enabled() -> bool:
    """Phase A returns 200 fast and runs Phase B in a background task
    by default (since 2026-05-08 once the GpuLease primitive +
    recovery driver landed).

    Stop's response time becomes bounded by ASR flush + translation
    drain (~25-30 s) instead of the legacy ~446 s, and a new meeting
    can be started immediately while Phase B's diarize + summary
    continue in the background under the GPU lease.

    Operators can fall back with ``SCRIBE_BACKGROUND_FINALIZE=0`` if
    they hit a regression — the legacy synchronous path is still
    intact and tested by the same test suite.
    """
    return os.environ.get("SCRIBE_BACKGROUND_FINALIZE", "1") != "0"


def _persist_exclusive_segments(meeting_dir, exclusive_segments: list[dict]) -> None:
    """Write the raw community-1 exclusive timeline as a JSON artifact.

    Lives at ``<meeting_dir>/speaker_lanes_exclusive.json`` parallel to
    the existing ``speaker_lanes.json`` (which is event-aligned).  This
    artifact is the frame-level single-speaker timeline straight from
    pyannote 4.x ``DiarizeOutput.exclusive_speaker_diarization`` —
    cluster ids match the merged-global numbering used in the standard
    diarize pipeline.

    Additive on the artifact set; UI consumers ignore unknown files.
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
    last_mic = None

    # Phase 5 — mic-readiness probe runs in parallel with the inference
    # probes. The probe-local epoch contract (``audio.mic_liveness``)
    # plugs the two holes codex review caught: stale pre-session values
    # can't satisfy the probe, and ``server_mic_active=True`` alone is
    # never accepted as proof.
    from meeting_scribe.audio.mic_liveness import probe_mic_liveness

    while True:
        attempt += 1
        # Run every probe concurrently — they hit independent backends
        # and the in-process state, so latency is bounded by the
        # slowest, not the sum.
        asr_result, translate_result, diarize_result, mic_result = await asyncio.gather(
            asr_synthetic_probe(asr_url, asr_model, state.metrics.asr_request_rtt_ms),
            translate_synthetic_probe(
                translate_url, translate_model, state.metrics.translate_request_rtt_ms
            ),
            diarize_synthetic_probe(diarize_url, state.metrics.diarize_request_rtt_ms),
            probe_mic_liveness(),
        )
        last_asr, last_translate, last_mic = asr_result, translate_result, mic_result

        if asr_result.ok and translate_result.ok and mic_result.ok:
            # Required backends + mic green — succeed even if diarize is
            # still warming. Diarize is enrichment, not a blocker.
            logger.info(
                "Meeting-start preflight passed on attempt %d "
                "(asr=%.0fms translate=%.0fms diarize=%s/%s/%.0fms mic=ok)",
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
        not_ready.append(
            {
                "backend": "asr",
                "detail": (
                    f"probe {last_asr.status} ({last_asr.latency_ms:.0f}ms): "
                    f"{last_asr.detail or 'unknown'}"
                ),
            }
        )
    if last_translate is not None and not last_translate.ok:
        not_ready.append(
            {
                "backend": "translate",
                "detail": (
                    f"probe {last_translate.status} ({last_translate.latency_ms:.0f}ms): "
                    f"{last_translate.detail or 'unknown'}"
                ),
            }
        )
    if last_mic is not None and not last_mic.ok:
        not_ready.append(
            {
                "backend": "mic",
                "detail": last_mic.detail,
                "reason": last_mic.reason,
            }
        )
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

    # Parse the request body ONCE so the audio_config branch and the
    # downstream language_pair branch can both see it. FastAPI's
    # ``Request.json()`` cannot be called twice (the body stream is
    # consumed on the first read), and the body is small enough that
    # double parsing isn't worth the contortion.
    try:
        body = await request.json()
    except Exception:
        body = None
    if body is not None and not isinstance(body, dict):
        body = None

    # ── ATOMIC AUDIO CONFIG (W4) ─────────────────────────────────
    # When the setup wizard sends ``audio_config`` it is the source of
    # truth for THIS meeting's routing. Validate it; on validation
    # error return 400 before the deep-health gate so the operator
    # gets immediate feedback. Persist the snapshot before reconcile
    # so the live config and the file agree even if reconcile fails.
    has_explicit_audio_config = isinstance(body, dict) and isinstance(
        body.get("audio_config"), dict
    )
    if has_explicit_audio_config:
        from meeting_scribe.audio.audio_routing import validate_routing_payload
        from meeting_scribe.server_support.settings_store import _save_settings_override

        audio_updates, audio_error = validate_routing_payload(body["audio_config"])
        if audio_error is not None:
            return JSONResponse(
                {"error": "audio_config_invalid", "message": audio_error},
                status_code=400,
            )
        if audio_updates:
            _save_settings_override(audio_updates)

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

    # Reset the soft input mute. The mute is meeting-scoped runtime
    # state — a stale toggle from a previous meeting must never quietly
    # suppress a fresh recording. See ``audio.audio_routing`` for the
    # gates that consume this flag.
    state.mic_input_muted = False

    # ── ATOMIC AUDIO RECONCILE (W4) ──────────────────────────────
    # Apply the persisted routing to the live subsystem. If the caller
    # provided an explicit ``audio_config`` and reconcile fails, refuse
    # the start so the operator can fix routing before recording. If
    # no explicit config (operator clicked Start on the legacy
    # browser-mic path) and reconcile fails, proceed with a soft
    # warning on the response — recording from the browser mic is
    # still possible without local audio devices.
    from meeting_scribe.audio.audio_routing import reconcile_audio_routing

    audio_routing_result = await reconcile_audio_routing()
    if not audio_routing_result["ok"] and has_explicit_audio_config:
        return JSONResponse(
            {
                "error": "audio_routing_failed",
                "message": ("Could not apply audio routing — see mic/sink details below."),
                "mic": audio_routing_result["mic"],
                "sink": audio_routing_result["sink"],
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
    # ``body`` was parsed once at the top of this handler — see the
    # atomic-audio-config block above. Reuse it here for the language
    # pair branch; ``Request.json()`` cannot be called twice.
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

    # Broadcast meeting_started so the popout (and the HDMI kiosk mirror)
    # can clear any residual content from a previous meeting AND reset
    # the layout to the default 'translate' preset. Without this signal,
    # stale transcript / slides from the prior meeting linger in the
    # popout DOM until the first new segment arrives — observed during
    # the GB10 HDMI mirror bring-up on 2026-05-14.
    await _broadcast_json(
        {
            "type": "meeting_started",
            "meeting_id": meta.meeting_id,
            "language_pair": language_pair,
        }
    )

    # Recording is sovereign over the GPU. If a prior meeting's Phase B
    # is mid-flight (e.g. running the full diarize HTTP call), this
    # acquire signals preempt and waits up to _PREEMPT_BUDGET_S (1.5 s)
    # for the lease. Bounded — Start's overall latency stays predictable.
    # No-op when SCRIBE_BACKGROUND_FINALIZE=0 (legacy stop blocks until
    # diarize completes; the lease never sees Phase B holding it).
    if _is_background_finalize_enabled():
        from meeting_scribe.runtime.gpu_lease import gpu_lease

        await gpu_lease().acquire_recording()

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
            context_directions=state.config.live_translate_context_directions,
        )
    if state.asr_backend is not None:
        state.asr_backend.set_languages(meeting_languages)

    logger.info("Meeting started: %s", meta.meeting_id)

    # Warm the translate model so the first real utterance doesn't pay
    # the compile/cold-path tax.
    if state.translate_backend is not None and not meta.is_monolingual:
        _warmup_prior_context: list[tuple[str, str]] | None = None
        context_directions = {
            part.strip()
            for part in state.config.live_translate_context_directions.split(",")
            if part.strip()
        }
        if (
            state.config.live_translate_context_window_ja_en > 0
            and f"{meta.language_pair[0]}:{meta.language_pair[1]}" in context_directions
        ):
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
                pass  # warmup is best-effort; first real utterance will retry on the live path

        asyncio.create_task(_warm_translate())

    response_body: dict[str, Any] = {
        "meeting_id": meta.meeting_id,
        "state": "recording",
        "resumed": False,
        "language_pair": meta.language_pair,
    }
    # Soft-warning carriage: when no explicit audio_config was sent but
    # reconcile still failed (e.g. operator launched with a stale BT
    # sink in settings.json), surface the result so the UI can render a
    # non-blocking banner without aborting the start.
    if not audio_routing_result["ok"]:
        response_body["audio_routing"] = audio_routing_result
    return JSONResponse(response_body)


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

    # Resume path: meta restored from disk, transcript file ready,
    # mirror needs the same clear-and-reset hint as a fresh start.
    await _broadcast_json(
        {
            "type": "meeting_started",
            "meeting_id": meta.meeting_id,
            "language_pair": getattr(meta, "language_pair", None),
        }
    )

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
                events_by_sid[sid] = e if existing is None else prefer_event(existing, e)

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

    # Reprocess succeeded — drop any leftover failure-terminal sidecar
    # from the original finalize attempt so the meeting-detail banner
    # stops shouting at the operator.
    with suppress(Exception):
        state.storage.clear_phase_b_progress(meeting_id)

    return JSONResponse(
        {
            "status": "reprocessed",
            "meeting_id": meeting_id,
            **result,
        }
    )


@router.post("/api/meetings/{meeting_id}/finalize/dismiss")
async def dismiss_finalize_failure(meeting_id: str) -> JSONResponse:
    """Operator escape hatch for a stuck failure-terminal sidecar.

    Drops the Phase B failure sidecar without retrying. Broadcasts a
    synthetic terminal event so any open toast / banner clears. Use this
    when an operator has decided the failure is not worth a reprocess
    (e.g. the meeting was a test, or the audio is corrupt).
    """
    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir or not meeting_dir.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    with suppress(Exception):
        state.storage.clear_phase_b_progress(meeting_id)
    with suppress(Exception):
        await _broadcast_json(
            {
                "type": "background_finalize_progress",
                "meeting_id": meeting_id,
                "step": 7,
                "total_steps": 7,
                "label": "Dismissed",
                "terminal": True,
                "error": False,
                "dismissed": True,
            }
        )
    return JSONResponse({"status": "dismissed", "meeting_id": meeting_id})


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

    # ── Phase A → Phase B boundary ───────────────────────────────
    #
    # Everything below up to the final return runs synchronously
    # under the lifecycle lock by default. With ``SCRIBE_BACKGROUND_FINALIZE=1``
    # we capture the local + state context, snapshot WS sets, clear
    # ``state.current_meeting`` so a new meeting can start, then spawn
    # the rest of the body as a background task and return 200 within
    # ~asr_flush + translation_drain (~25–30 s).
    #
    # The closure below reads ONLY from these locals (or singletons
    # storage/config). All mutating state.* writes are confined to
    # post-Phase-A cleanup so a new meeting's start is unaffected.
    if _is_background_finalize_enabled():
        # Snapshot state that the closure must read. Anything cleared
        # here is for the next meeting's benefit.
        _ctx_detected_speakers = list(state.detected_speakers or [])
        _ctx_eager_summary_cache = state._eager_summary_cache
        _ctx_eager_summary_event_count = state._eager_summary_event_count
        # Clear in-state copies so the new meeting starts clean.
        state.detected_speakers = []
        state._eager_summary_cache = None
        state._eager_summary_event_count = 0
        # Clear the speaker-side state so the next meeting's Start
        # sees a clean slate. Speaker tasks were optional —
        # cancel-only-if-running.
        if state._speaker_pulse_task and not state._speaker_pulse_task.done():
            state._speaker_pulse_task.cancel()
            state._speaker_pulse_task = None
        if state._speaker_catchup_task and not state._speaker_catchup_task.done():
            state._speaker_catchup_task.cancel()
            state._speaker_catchup_task = None
        state._pending_speaker_events.clear()
        state._pending_speaker_timestamps.clear()
        # Snapshot WS sets so Phase B's `meeting_stopped` broadcast
        # reaches the OLD listeners; clear state.* so the new meeting
        # binds fresh sockets.
        _ctx_ws_connections = list(state.ws_connections)
        _ctx_audio_out_clients = list(state._audio_out_clients)
        state.ws_connections.clear()
        state._client_prefs.clear()
        state._audio_out_clients.clear()
        state._audio_out_prefs.clear()
        # Clear current_meeting + lifecycle-task tracker so a new
        # /api/meeting/start succeeds while Phase B keeps running.
        state.current_meeting = None
        for task in list(state._background_tasks):
            if not task.done():
                task.cancel()
        state._background_tasks.clear()
        _clear_active_meeting()
        asyncio.create_task(_stop_wifi_ap())
        # Release the recording lease so a new Start can acquire it
        # without blocking on Phase B's preempt round-trip. Phase B
        # acquires its own per-call lease via `_gpu_lease.run_phase_b_call`
        # for each GPU step, so dropping recording here is safe.
        from meeting_scribe.runtime.gpu_lease import gpu_lease as _gpu_lease

        try:
            await _gpu_lease().release_recording()
        except AssertionError:
            # Recording was never acquired (e.g. legacy path racing the
            # flag flip mid-meeting). Tolerate.
            pass

        # Eagerly transition to FINALIZING so the meetings list (and
        # any /api/meetings/{id} consumer) reflects "finalize is in
        # flight" immediately — without this, disk state stays at
        # RECORDING for the whole duration of Phase B and the UI shows
        # stale info. The Phase B body re-issues this transition
        # idempotently inside its `try`-block at the end (see
        # _finalize_phase_b_inline), but landing it here too means a
        # crash mid-Phase-B leaves a recoverable FINALIZING marker.
        try:
            state.storage.transition_state(mid, MeetingState.FINALIZING)
        except Exception as e:
            logger.warning(
                "Phase A: early FINALIZING transition failed mid=%s: %s",
                mid,
                e,
            )

        async def _phase_b_runner() -> None:
            try:
                await _finalize_phase_b_inline(
                    mid=mid,
                    finalize_t0=_finalize_t0,
                    audio_duration_s=audio_duration_s,
                    est_seconds=est_seconds,
                    partial_finalize=_partial_finalize,
                    pending_translation_count=_pending_translation_count,
                    detected_speakers=_ctx_detected_speakers,
                    eager_summary_cache=_ctx_eager_summary_cache,
                    eager_summary_event_count=_ctx_eager_summary_event_count,
                    ws_connections=_ctx_ws_connections,
                    audio_out_clients=_ctx_audio_out_clients,
                )
            except Exception:
                logger.exception("phase_b runner failed mid=%s", mid)
                with suppress(Exception):
                    state.storage.transition_state(mid, MeetingState.INTERRUPTED)
                # Persist a failure-terminal sidecar so the inline
                # banner + corner toast see ground truth across reloads.
                # The last in-flight sidecar (if any) gives us the step
                # that was running when the exception fired — read it
                # back so the UI can say "Failed at step 5" instead of
                # an opaque "Failed".
                _last = None
                with suppress(Exception):
                    _last = state.storage.read_phase_b_progress(mid)
                _last_step = int((_last or {}).get("step") or 3)
                _last_label = str((_last or {}).get("label") or "Finalizing...")
                with suppress(Exception):
                    state.storage.write_phase_b_failure(
                        mid,
                        step=_last_step,
                        total=7,
                        label=_last_label,
                        code="phase_b_failed",
                        message=("Phase B finalize failed — try Reprocess to recover."),
                    )
                with suppress(Exception):
                    await _broadcast_json(
                        {
                            "type": "background_finalize_progress",
                            "meeting_id": mid,
                            "step": _last_step,
                            "total_steps": 7,
                            "label": _last_label,
                            "terminal": True,
                            "error": True,
                            "code": "phase_b_failed",
                            "message": ("Phase B finalize failed — try Reprocess to recover."),
                        }
                    )
            finally:
                _phase_b_tasks.pop(mid, None)

        bg = asyncio.create_task(_phase_b_runner(), name=f"finalize-phase-b-{mid}")
        _phase_b_tasks[mid] = bg
        logger.info(
            "finalize.phase_a.complete meeting=%s took=%.1fs — Phase B running in background",
            mid,
            time.monotonic() - _finalize_t0,
        )
        return JSONResponse({"status": "finalizing", "meeting_id": mid, "phase_b": "running"})

    # ── Legacy path (SCRIBE_BACKGROUND_FINALIZE!=1): everything below
    # runs inline under the lifecycle lock, exactly as before.
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
                events_by_sid[sid] = event if existing is None else prefer_event(existing, event)

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
            # ``_parallel_summary_task`` captured ``events`` at task start —
            # before ``_generate_speaker_data`` rewrote cluster_ids to
            # seq_index. The narrative content is fine (the LLM works from
            # transcript text), but ``speaker_stats`` ends up referencing
            # pre-rewrite raw cluster IDs, often inverted relative to
            # ``detected_speakers.json``. Recompute against the rewritten
            # journal so the on-disk summary's stats line up with the
            # speaker list the rest of the UI joins on. Matches the
            # draft-promote branch above.
            if isinstance(summary, dict) and "error_code" not in summary:
                with suppress(Exception):
                    from meeting_scribe.summary import (
                        _calculate_speaker_stats,
                        build_transcript_text,
                    )

                    _meeting_dir = state.storage._meeting_dir(mid)
                    _events, _ = build_transcript_text(_meeting_dir)
                    _speakers_path = _meeting_dir / "detected_speakers.json"
                    _speakers: list[dict] = []
                    if _speakers_path.exists():
                        with suppress(Exception):
                            _speakers = _json.loads(_speakers_path.read_text())
                    summary["speaker_stats"] = _calculate_speaker_stats(_events, _speakers)
                    if isinstance(summary.get("metadata"), dict):
                        summary["metadata"].update(
                            {
                                "num_segments": len(_events),
                                "num_speakers": len(summary["speaker_stats"]),
                            }
                        )
                    (_meeting_dir / "summary.json").write_text(
                        _json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
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

    # In-session backlog drain: when finalize completed with pending
    # translations persisted to ``pending_translations.jsonl``, the
    # translation backend is now free (no live meeting holding the
    # queue). Kick the same recovery path the lifespan-startup hook
    # uses so the operator doesn't have to restart the server to see
    # the late translations land + summary.status flip from partial→
    # complete. The replay helper short-circuits if a new live meeting
    # has already grabbed the queue, so this is safe to fire-and-forget.
    if _partial_finalize and _pending_translation_count > 0:

        async def _drain_backlog() -> None:
            try:
                from meeting_scribe.runtime.translation_recovery import (
                    _replay_one_meeting,
                )

                meeting_dir = state.storage._meeting_dir(mid)
                await _replay_one_meeting(meeting_dir)
            except Exception:
                logger.exception("post-finalize backlog drain failed meeting=%s", mid)

        _drain_task = asyncio.create_task(_drain_backlog(), name=f"backlog-drain-{mid}")
        state._background_tasks.add(_drain_task)
        _drain_task.add_done_callback(state._background_tasks.discard)

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


async def _finalize_phase_b_inline(
    *,
    mid: str,
    finalize_t0: float,
    audio_duration_s: float,
    est_seconds: int,
    partial_finalize: bool,
    pending_translation_count: int,
    detected_speakers: list,
    eager_summary_cache: dict | None,
    eager_summary_event_count: int,
    ws_connections: list,
    audio_out_clients: list,
) -> None:
    """Phase B: heavy GPU finalize work running outside the lifecycle lock.

    Called as a fire-and-forget asyncio.Task by ``_stop_meeting_locked``
    when ``SCRIBE_BACKGROUND_FINALIZE=1``. Receives a snapshot of all
    state needed to finalize meeting ``mid`` independent of whatever
    is currently running on ``state.current_meeting``.

    Reads / writes only:
      * ``state.storage`` — singleton, per-mid disk paths
      * ``state.config`` — singleton
      * The captured ``ws_connections`` / ``audio_out_clients`` lists
        (as snapshots — NOT ``state.ws_connections``, which by now
        belongs to a possibly-newer meeting)
      * The ``state.storage`` journal + meta files for the OLD ``mid``

    Emits ``background_finalize_progress`` events broadcast to whatever
    is currently in ``state.ws_connections`` so admin clients (whether
    on the old or a new meeting) can render the corner toast. The
    legacy ``finalize_progress`` event is preserved for steps 3-6 so
    operators flipping the flag don't lose the modal until the
    frontend toast lands.
    """
    from meeting_scribe.summary import (
        _calculate_speaker_stats,
        build_transcript_text,
    )

    meeting_dir = state.storage._meeting_dir(mid)
    journal_path = meeting_dir / "journal.jsonl"
    pcm_path = meeting_dir / "audio" / "recording.pcm"

    # ETA decay schedule per step. Matches the legacy sync-finalize
    # budget (lines 1409/1442/1459 of the legacy path) — see the plan's
    # W1 section. Step 7 is the success-terminal broadcast and never
    # writes the sidecar (we clear it instead).
    def _eta_for_step(step: int) -> int:
        decay = {3: 1.0, 4: 0.7, 5: 0.5, 6: 0.25, 7: 0.0}
        return max(0, int(est_seconds * decay.get(step, 0.0))) if step < 7 else 5

    # Track the most recent step + label so the runner wrapper can
    # stamp a failure sidecar without having to thread state through.
    _last_step_label: dict[str, Any] = {"step": 3, "total": 7, "label": ""}
    _phase_b_started_at = time.time()

    async def _bg_progress(step: int, total: int, label: str, **extra: Any) -> None:
        eta_seconds = _eta_for_step(step)
        _last_step_label.update({"step": step, "total": total, "label": label})
        # Persist the sidecar BEFORE broadcasting so a client that
        # receives the WS event and immediately re-fetches
        # /api/meetings/{id} sees the same step/eta value instead of
        # the previous one. Failure of the broadcast or persistence is
        # logged but not fatal — the next tick will overwrite.
        try:
            state.storage.write_phase_b_progress(
                mid,
                step=step,
                total=total,
                label=label,
                eta_seconds=eta_seconds,
                session_uuid=state.session_uuid,
                started_at=_phase_b_started_at,
            )
        except Exception:
            logger.exception("phase_b: progress sidecar write failed mid=%s", mid)
        await _broadcast_json(
            {
                "type": "background_finalize_progress",
                "meeting_id": mid,
                "step": step,
                "total_steps": total,
                "label": label,
                "terminal": False,
                "error": False,
                "eta_seconds": eta_seconds,
                **extra,
            }
        )

    await _bg_progress(3, 7, "Saving speaker data...")
    if detected_speakers:
        state.storage.save_detected_speakers(mid, detected_speakers)
        logger.info("Saved %d detected speakers", len(detected_speakers))

    await _bg_progress(4, 7, "Running full-audio diarization...")

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
        eager_summary_cache is not None
        and eager_summary_event_count > 0
        and _n_final_events > 0
        and (_n_final_events - eager_summary_event_count) / eager_summary_event_count <= 0.20
    )

    _summary_attempt_id: int = 0
    _partial_extra: dict[str, Any] = (
        {"partial": True, "pending_translation_count": pending_translation_count}
        if partial_finalize
        else {}
    )
    _parallel_summary_task: asyncio.Task | None = None
    if not _use_draft:
        from meeting_scribe.summary import generate_summary as _gen_summary

        _summary_attempt_id = next_attempt_id(meeting_dir)
        write_status(
            meeting_dir,
            SummaryStatus.GENERATING,
            attempt_id=_summary_attempt_id,
            journal_path=journal_path,
            extra=_partial_extra,
        )

        async def _parallel_summary():
            return await _gen_summary(
                meeting_dir,
                vllm_url=state.config.translate_vllm_url,
            )

        _parallel_summary_task = asyncio.create_task(_parallel_summary())

    if journal_path.exists() and pcm_path.exists():
        try:
            from meeting_scribe.pipeline.diarize import _diarize_full_audio
            from meeting_scribe.pipeline.quality import _audio_quality_report
            from meeting_scribe.pipeline.speaker_attach import _attach_speakers_to_events
            from meeting_scribe.runtime.gpu_lease import _Preempted, gpu_lease

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
                events_by_sid[sid] = event if existing is None else prefer_event(existing, event)

            if events_by_sid:
                _fast_mode = os.environ.get("SCRIBE_FINALIZE_DIARIZE_FAST", "shadow").lower()
                if _fast_mode not in ("off", "shadow", "on"):
                    _fast_mode = "off"

                logger.info(
                    "Stop: diarize_fast_mode=%s, %d events (%.0fMB audio)",
                    _fast_mode,
                    len(events_by_sid),
                    pcm_path.stat().st_size / 1024 / 1024,
                )

                from meeting_scribe.pipeline.diarize_consolidate import (
                    compare_diarize_results,
                    consolidate_from_events,
                )

                fast_segments: list = []
                if _fast_mode in ("shadow", "on"):
                    fast_segments = consolidate_from_events(events_by_sid)

                diarize_segments: list = []
                diarize_exclusive: list = []
                if _fast_mode == "on" and fast_segments:
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
                    # Wrap the heavy GPU call in the lease so a Start
                    # mid-flight preempts it. On _Preempted we retry —
                    # diarize is restartable from scratch (PCM on disk
                    # is unchanged).
                    diarize_attempt = 0
                    diarize_result = None
                    while diarize_result is None:
                        diarize_attempt += 1

                        async def _do_diarize(_alloc):
                            return await _diarize_full_audio(
                                pcm_data,
                                state.config.diarize_url,
                                max_speakers=8,
                                min_speakers=2,
                            )

                        try:
                            diarize_result = await gpu_lease().run_phase_b_call(
                                _do_diarize, backend="diarize"
                            )
                        except _Preempted:
                            logger.info(
                                "phase_b: diarize preempted by recording (attempt %d) mid=%s",
                                diarize_attempt,
                                mid,
                            )
                            await _bg_progress(
                                4,
                                7,
                                "Paused for active meeting — diarize will resume...",
                            )
                            if diarize_attempt >= 6:
                                raise RuntimeError(
                                    f"diarize preempted {diarize_attempt} times"
                                ) from None
                            await asyncio.sleep(min(2 ** (diarize_attempt - 1), 30))
                        except RuntimeError as e:
                            # No backend registered (lifespan didn't wire
                            # it up — e.g., tests). Fall back to direct
                            # call without preemption.
                            if "no backend registered" not in str(e):
                                raise
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

    await _bg_progress(5, 7, "Generating timeline...")
    _timeline_t0 = time.monotonic()
    _generate_timeline(mid)
    logger.info(
        "finalize.step=timeline meeting=%s took=%.1fs",
        mid,
        time.monotonic() - _timeline_t0,
    )

    await _bg_progress(6, 7, "Generating meeting summary...")
    _summary_t0 = time.monotonic()
    summary: dict[str, Any] = {}
    if _use_draft:
        summary = eager_summary_cache or {}
        try:
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
            logger.info("Eager summary: promoted draft to final (%d events)", len(events))
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
            _summary_attempt_id = next_attempt_id(meeting_dir)
            write_status(
                meeting_dir,
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
                meeting_dir,
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
                meeting_dir,
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

    # Phase A may have already transitioned us to FINALIZING (so the
    # meetings list shows progress immediately). Skip the redundant
    # transition if we're already there — RECORDING → FINALIZING is
    # the only legal precursor to FINALIZING, so an idempotent guard
    # keeps the state machine happy.
    try:
        current_meta = state.storage._read_meta(mid)
        if current_meta and current_meta.state == MeetingState.RECORDING:
            state.storage.transition_state(mid, MeetingState.FINALIZING)
        state.storage.transition_state(mid, MeetingState.COMPLETE)
    except Exception as e:
        logger.warning("Stop transition error: %s", e)
    logger.info(
        "finalize.phase_b.total meeting=%s took=%.1fs",
        mid,
        time.monotonic() - finalize_t0,
    )

    # Success terminal: drop the progress sidecar before broadcasting so
    # any client that re-fetches /api/meetings/{id} after seeing the
    # terminal event finds no in-flight phase_b_progress.
    try:
        state.storage.clear_phase_b_progress(mid)
    except Exception:
        logger.exception("phase_b: success sidecar clear failed mid=%s", mid)

    # Terminal progress event so the frontend toast can flip to "Done".
    await _broadcast_json(
        {
            "type": "background_finalize_progress",
            "meeting_id": mid,
            "step": 7,
            "total_steps": 7,
            "label": "Done!",
            "terminal": True,
            "error": False,
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

    # Close the OLD WS sockets we snapshotted in Phase A. Anything in
    # state.ws_connections by now belongs to the new meeting. Wrap in
    # list() to satisfy the test_preflight_set_iteration_safe regex
    # (defensive even though the parameter is already a list).
    for ws in list(ws_connections):
        with suppress(Exception):
            await ws.close(1000, "Meeting ended")
    for ws in list(audio_out_clients):
        with suppress(Exception):
            await ws.close(1000, "Meeting ended")
    logger.info("Meeting stopped (background): %s", mid)
