"""TTS worker pool — semaphore-first synthesis under deadline budgets.

Six entry points cover the full path from "translation finished" to
"audio fanned out":

* ``_tts_outstanding`` — total in-flight + queued, used by both
  producer gates and the health evaluator.
* ``_enqueue_tts`` — producer gate. Three drops happen here:
  missing-origin, whitelist filler, and stale-on-enqueue.
* ``_start_tts_worker`` — idempotent setup of queue + container
  semaphore + worker pool. Spawns ``state.TTS_WORKER_COUNT`` parallel
  workers under ``state.TTS_CONTAINER_MAX_CONCURRENCY``.
* ``_tts_worker_loop`` — semaphore-first dequeue body. Acquires the
  container semaphore BEFORE pulling an item, so parked workers hold
  nothing.
* ``_record_segment_lag`` — segment-level lag histograms recorded
  exactly once per segment, BEFORE fan-out.
* ``_do_tts_synthesis`` — inner synthesis: text capping, pre-synth
  budget check, deadline-aware ``wait_for``, post-synth re-check,
  one-shot WAV fan-out.

Pulled out of ``server.py`` once the TTS runtime handles + pipeline
config constants moved to ``runtime.state``. Tests still call
``server._enqueue_tts`` / ``server._record_segment_lag`` etc. via a
re-export in server.py, so no test churn from this extraction.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

import numpy as np

from meeting_scribe.audio.output_pipeline import _send_audio_to_listeners
from meeting_scribe.runtime import state
from meeting_scribe.server_support.metrics_helpers import _percentile
from meeting_scribe.server_support.translation_demand import _norm_lang

logger = logging.getLogger(__name__)


# Hard TTS lag cap. Deadline = tts_origin + this value, where
# tts_origin is translation.completed_at (upstream-aware anchor).
# With streaming + two replicas, first-audio latency is sub-second
# and most full-synth calls land inside ~2 s. Tightening from 8 s →
# 5 s drops stale segments that would otherwise play 5+ s behind the
# speaker — freshness beats completeness for live interpretation.
_TTS_MAX_SPEECH_LAG_S = 5.0

# Slack threshold at the producer gate — drop if less than this much
# of the deadline remains when the event arrives AND there is real
# outstanding work.
_TTS_PRODUCER_MIN_SLACK_S = 0.5

# Size-aware wait_for ceiling: base + per-char budget. A 25-char
# segment gets ~8.75 s, a 300-char (MAX_TTS_CHARS) segment gets 23 s.
# The effective timeout used in wait_for is min(size_timeout,
# max(0.2, deadline - now)) so the deadline always wins — see
# [P1-1-i2].
#
# Bumped 2026-04-15 from base=5.0/per_char=0.03 to base=8.0/per_char=0.05
# after observing that faster-qwen3-tts round-robin across two replicas
# can incur ~4 s of container-internal queue wait when requests burst
# in, leaving the old 5 s budget with only ~1 s for actual synthesis.
# Paired with the ASR buffer_seconds bump (1.5 → 3.5) which cuts TTS
# request rate by ~2.5x, the two changes together keep the deadline
# honest while the model-level freshness cap
# (_TTS_MAX_SPEECH_LAG_S = 5 s) remains the real freshness SLA.
_TTS_SYNTH_TIMEOUT_BASE_S = 8.0
_TTS_SYNTH_TIMEOUT_PER_CHAR_S = 0.05

# Whitelist of acknowledgment tokens that should be dropped as
# "filler" when the pipeline has real outstanding work. [P2-2-i1] —
# char-length was too aggressive and would swallow "No", "Stop",
# names. Env-configurable.
_TTS_ACK_WHITELIST: frozenset[str] = frozenset(
    {
        "はい",
        "うん",
        "ええ",
        "そうですね",
        "なるほど",
        "i see",
        "yeah",
        "uh",
        "um",
        "mhm",
        "ok",
        "okay",
        "right",
    }
)
_env_ws = os.environ.get("SCRIBE_TTS_ACK_WHITELIST")
if _env_ws:
    _TTS_ACK_WHITELIST = frozenset(t.strip().lower() for t in _env_ws.split(",") if t.strip())


def _tts_outstanding() -> int:
    """Total outstanding TTS work: in-flight + queued.

    Used by the producer backlog gates AND the health evaluator. With
    the semaphore-first worker loop, workers parked on
    ``state.tts_backend_semaphore`` hold NO queue items, so
    ``state.tts_in_flight + qsize()`` is the complete outstanding count.
    [P1-1-i3 + i4]
    """
    return state.tts_in_flight + (state.tts_queue.qsize() if state.tts_queue else 0)


def _enqueue_tts(event, speaker_id: str) -> None:
    """Push a translation segment onto the FIFO TTS backlog.

    Applies three producer-side gates before enqueue:
      1. Missing-origin drop — event.utterance_end_at is None is a
         code bug (asr_vllm MUST populate it). Refuse the segment.
         [P1-2-i5]
      2. Whitelist filler — drop short acknowledgment tokens
         ("はい", "I see", ...) when there's real outstanding work,
         so one-word interjections don't block real content.
         [P2-2-i1]
      3. Stale-on-enqueue — if the deadline has less than MIN_SLACK
         left AND we have outstanding work, drop rather than compete
         for a worker slot that will miss its budget anyway.
         [P1-2-i1]
    Otherwise the existing drop-oldest policy handles queue saturation.
    """
    if state.tts_queue is None:
        return

    # Gate 1 [P1-2-i5]: missing speech-end origin is a code bug. Refuse.
    if event.utterance_end_at is None:
        logger.warning(
            "TTS refuse seg=%s: utterance_end_at missing — asr_vllm "
            "did not populate it (bug: start_meeting must set "
            "audio_wall_at_start)",
            event.segment_id,
        )
        state.metrics.tts_dropped_missing_origin += 1
        return

    # Deadline anchored to when translation completed (upstream-aware),
    # not when the speaker stopped talking. utterance_end_at is
    # audio-time which doesn't account for ASR + translation latency
    # — those can consume 5-6s of a speech-end-based budget before
    # TTS even sees the segment. translation.completed_at is the
    # wall-clock moment upstream finished, giving TTS a clean budget
    # for synthesis + delivery only.
    tts_origin = (
        event.translation.completed_at
        if event.translation and event.translation.completed_at is not None
        else time.monotonic()
    )
    deadline = tts_origin + _TTS_MAX_SPEECH_LAG_S
    outstanding = _tts_outstanding()

    # Gate 2 [P2-2-i1]: whitelist filler drop.
    text = (event.translation.text or "").strip().lower()
    if text in _TTS_ACK_WHITELIST and outstanding > 0:
        logger.info(
            "TTS skip ack seg=%s text=%r (outstanding=%d)",
            event.segment_id,
            text,
            outstanding,
        )
        state.metrics.tts_dropped_filler += 1
        return

    # Gate 3 [P1-2-i1]: stale-on-enqueue.
    now = time.monotonic()
    remaining = deadline - now
    if remaining < _TTS_PRODUCER_MIN_SLACK_S and outstanding > 0:
        logger.info(
            "TTS skip stale-on-enqueue seg=%s remaining=%.2fs (outstanding=%d)",
            event.segment_id,
            remaining,
            outstanding,
        )
        state.metrics.tts_dropped_stale_producer += 1
        return

    # Drop-oldest-on-full preserves the "play recent content" invariant.
    while state.tts_queue.full():
        try:
            dropped = state.tts_queue.get_nowait()
            state.tts_queue.task_done()
            dropped_event = dropped[0]
            state.metrics.tts_dropped_queue_full += 1
            logger.warning(
                "TTS backlog full (%d) — dropping oldest seg=%s to make room for %s",
                state.TTS_QUEUE_MAXSIZE,
                getattr(dropped_event, "segment_id", "?"),
                event.segment_id,
            )
        except asyncio.QueueEmpty:
            break
    try:
        # Queue tuple shape: (event, speaker_id, deadline, queued_at)
        state.tts_queue.put_nowait((event, speaker_id, deadline, now))
        state.metrics.tts_submitted += 1
    except asyncio.QueueFull:
        logger.warning("TTS enqueue failed despite make-room loop")


def _start_tts_worker() -> None:
    """Create the TTS queue, container semaphore, and worker pool. Idempotent.

    Spawns ``state.TTS_WORKER_COUNT`` parallel worker tasks that drain
    the same shared queue under a semaphore-first flow — each worker
    acquires ``state.tts_backend_semaphore`` BEFORE dequeuing, so at
    most ``state.TTS_CONTAINER_MAX_CONCURRENCY`` workers are ever
    committed to a segment at once. See ``_tts_worker_loop``.
    """
    if state.tts_queue is None:
        state.tts_queue = asyncio.Queue(maxsize=state.TTS_QUEUE_MAXSIZE)
    if state.tts_backend_semaphore is None:
        state.tts_backend_semaphore = asyncio.Semaphore(state.TTS_CONTAINER_MAX_CONCURRENCY)
    # Prune any dead workers so a restart re-spawns them
    state.tts_worker_tasks = [t for t in state.tts_worker_tasks if not t.done()]
    while len(state.tts_worker_tasks) < state.TTS_WORKER_COUNT:
        idx = len(state.tts_worker_tasks)
        state.tts_worker_tasks.append(
            asyncio.create_task(_tts_worker_loop(idx), name=f"tts-worker-{idx}")
        )
    logger.info(
        "TTS worker pool started (workers=%d, container_concurrency=%d, maxsize=%d)",
        len(state.tts_worker_tasks),
        state.TTS_CONTAINER_MAX_CONCURRENCY,
        state.TTS_QUEUE_MAXSIZE,
    )


async def _tts_worker_loop(worker_idx: int = 0) -> None:
    """Semaphore-first TTS worker [P1-1-i4].

    Acquires ``state.tts_backend_semaphore`` BEFORE dequeuing, so
    workers without the semaphore hold nothing — no queue item, no
    in-flight count, no metrics. With
    ``state.TTS_CONTAINER_MAX_CONCURRENCY=1`` only ONE worker is ever
    committed to a segment; the other N-1 are parked on the
    semaphore.
    """
    assert state.tts_queue is not None
    assert state.tts_backend_semaphore is not None
    while True:
        try:
            # 1. ACQUIRE CONTAINER SLOT FIRST — parks here until a slot is free.
            async with state.tts_backend_semaphore:
                # 2. Now dequeue exactly one item. Blocks here if empty.
                event, speaker_id, deadline, _queued_at = await state.tts_queue.get()
                try:
                    # Guard: skip if no active meeting (ghost synthesis after stop).
                    if not state.current_meeting:
                        continue
                    # Guard: no listeners → skip synthesis entirely to save GPU.
                    if not state._audio_out_clients:
                        continue
                    # 3. Dequeue deadline check [P1-1-i1].
                    if time.monotonic() > deadline:
                        logger.warning(
                            "TTS drop stale-at-dequeue seg=%s over=%.2fs",
                            event.segment_id,
                            time.monotonic() - deadline,
                        )
                        state.metrics.tts_dropped_stale_worker += 1
                        continue
                    # 4. Run the synth (pre-synth budget check + synthesis
                    #    + post-synth re-check + segment-level histograms
                    #    all live inside _do_tts_synthesis).
                    await _do_tts_synthesis(event, speaker_id, deadline)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning(
                        "TTS worker error on seg=%s: %s",
                        getattr(event, "segment_id", "?"),
                        e,
                    )
                finally:
                    try:
                        state.tts_queue.task_done()
                    except ValueError:
                        pass
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Defensive: any unexpected error in the semaphore path
            # should not kill the worker task. Log and loop.
            logger.exception("TTS worker %d outer loop error: %s", worker_idx, e)


def _record_segment_lag(event, now: float) -> None:
    """Record segment-level lag histograms ONCE per segment [P1-5-i2].

    Called exactly once per successful synthesis from ``_do_tts_synthesis``,
    BEFORE fan-out to listeners. N listeners cannot inflate these
    histograms to N samples.
    """
    if event.utterance_end_at is not None:
        state.metrics.end_to_end_lag_ms.append((now - event.utterance_end_at) * 1000.0)
        if event.translation and event.translation.completed_at is not None:
            state.metrics.upstream_lag_ms.append(
                (event.translation.completed_at - event.utterance_end_at) * 1000.0
            )
    if event.translation and event.translation.completed_at is not None:
        state.metrics.tts_post_translation_lag_ms.append(
            (now - event.translation.completed_at) * 1000.0
        )


async def _do_tts_synthesis(event, speaker_id, deadline: float) -> None:
    """Inner TTS synthesis — runs under the container semaphore.

    Flow:
      1. Cap text to MAX_TTS_CHARS (sentence-boundary aware).
      2. Filter listeners by language/voice_mode → voice_modes_needed.
      3. Pre-synth budget check: drop if remaining < expected.
         [P1-2-i1]
      4. Size-aware + deadline-aware wait_for. [P1-5-i1 + P1-1-i2]
      5. Post-synth deadline re-check: drop late audio. [P1-1-i2]
      6. Record segment-level histograms ONCE. [P1-5-i2]
      7. Fan out (transport metrics only). [P1-5-i2]

    The caller (``_tts_worker_loop``) already holds
    ``state.tts_backend_semaphore``, so this function does not
    re-acquire it.
    """
    from meeting_scribe.translation.queue import (
        MULTI_TARGET_ENABLED as _MULTI_TARGET_ENABLED,
    )

    # Caller guarantees state.tts_backend is non-None by gating TTS
    # work on the global's presence at submit/worker-spawn time —
    # assert it here so mypy can narrow the optional and every
    # downstream `state.tts_backend.<method>` call typechecks.
    assert state.tts_backend is not None

    text = event.translation.text
    # Cap text to ~30 seconds of audio (~100 words ≈ 200 chars).
    MAX_TTS_CHARS = 300
    if len(text) > MAX_TTS_CHARS:
        cut = text[:MAX_TTS_CHARS].rfind("。")
        if cut < 100:
            cut = text[:MAX_TTS_CHARS].rfind(". ")
        if cut < 100:
            cut = MAX_TTS_CHARS
        text = text[: cut + 1]
        logger.info(
            "TTS text truncated: %d → %d chars",
            len(event.translation.text),
            len(text),
        )

    # Determine which voice modes are needed by listeners.
    target_lang = _norm_lang(event.translation.target_language)
    voice_modes_needed: set[str] = set()
    total_listeners = 0
    skipped_lang = 0
    for ws in state._audio_out_clients:
        pref = state._audio_out_prefs.get(ws)
        if not pref:
            continue
        total_listeners += 1
        pref_lang = _norm_lang(pref.preferred_language)
        if pref_lang and pref_lang != target_lang:
            skipped_lang += 1
            continue
        voice_modes_needed.add(getattr(pref, "voice_mode", "studio"))

    if not voice_modes_needed:
        logger.info(
            "TTS skip seg=%s: no matching voice modes (target_lang=%s, "
            "listeners=%d, skipped_for_lang=%d)",
            event.segment_id,
            target_lang,
            total_listeners,
            skipped_lang,
        )
        return

    # Synthesize each requested voice mode (default voice, no cloning)
    for voice_mode in voice_modes_needed:
        # Pre-synth budget check [P1-2-i1]. The queue + semaphore wait
        # may have eaten budget since the producer admitted this
        # event. Compute expected synth time from rolling P95
        # (fallback to 2 s).
        remaining = deadline - time.monotonic()
        # Cap the P95 estimate — one slow request shouldn't poison
        # all future pre-synth checks and cause a cascade of drops.
        p95_s = min(
            (state.metrics.tts_synth_ms_p95 or 0) / 1000.0,
            _TTS_SYNTH_TIMEOUT_BASE_S,
        )
        expected = max(state.TTS_EXPECTED_SYNTH_DEFAULT_S, p95_s)
        if remaining < expected:
            logger.warning(
                "TTS drop pre-synth seg=%s mode=%s remaining=%.2fs expected=%.2fs",
                event.segment_id,
                voice_mode,
                remaining,
                expected,
            )
            state.metrics.tts_dropped_pre_synth += 1
            return

        used_fallback = False
        logger.info(
            "TTS synthesize: seg=%s mode=%s speaker_id=%s fallback=%s target_lang=%s chars=%d",
            event.segment_id,
            voice_mode,
            speaker_id,
            used_fallback,
            target_lang,
            len(text),
        )

        # Deadline-aware wait_for ceiling [P1-1-i2 + P1-5-i1].
        size_timeout = _TTS_SYNTH_TIMEOUT_BASE_S + _TTS_SYNTH_TIMEOUT_PER_CHAR_S * len(text)
        effective_timeout = min(size_timeout, max(0.2, remaining))

        state.tts_in_flight += 1
        state.tts_inflight_started[(event.segment_id, target_lang)] = time.monotonic()
        synth_start = time.monotonic()
        streamed_chunks: list[np.ndarray] = []
        first_chunk_ms: float | None = None
        try:
            from meeting_scribe.backends.tts_voices import studio_voice_for

            studio_voice = studio_voice_for(target_lang)
            voice_reference = None
            if voice_mode == "cloned":
                voice_reference = state.tts_backend.get_voice(speaker_id)
                # If no cached voice, fall back deterministically to
                # the studio voice for the target language (logged as
                # cloned_fallback.studio).
                if voice_reference is None:
                    logger.debug(
                        "tts.cloned_fallback.studio seg=%s speaker=%s reason=no_cached_voice",
                        event.segment_id,
                        speaker_id,
                    )

            try:

                async def _run_stream():
                    nonlocal first_chunk_ms
                    # Stream FROM the backend (keeps the container's
                    # generator busy and gives us a real TTFA metric),
                    # but do NOT fan out to listeners mid-stream.
                    # Every chunk boundary was producing an audible
                    # click because each chunk was wrapped in its own
                    # WAV/RIFF header and played as a separate
                    # BufferSource on the client; consecutive buffers
                    # rarely join at a zero-crossing. Accumulate here
                    # and send ONE coherent WAV per segment below.
                    # Latency cost: the listener hears each segment
                    # ~synth_time later than the old behaviour, but
                    # given the 3.5 s ASR buffer that's ~1 s additional
                    # delay, which listeners tolerate far better than
                    # constant clicking.
                    async for chunk in state.tts_backend.synthesize_stream(
                        text=text,
                        language=target_lang,
                        voice_reference=voice_reference,
                        studio_voice=studio_voice,
                    ):
                        if len(chunk) == 0:
                            continue
                        if first_chunk_ms is None:
                            first_chunk_ms = (time.monotonic() - synth_start) * 1000.0
                        streamed_chunks.append(chunk)

                await asyncio.wait_for(_run_stream(), timeout=effective_timeout)
            except TimeoutError:
                logger.warning(
                    "TTS stream TIMEOUT seg=%s chars=%d timeout=%.2fs waited=%.2fs",
                    event.segment_id,
                    len(text),
                    effective_timeout,
                    time.monotonic() - synth_start,
                )
                state.metrics.tts_synth_timeouts += 1
                return
        finally:
            state.tts_in_flight -= 1
            state.tts_inflight_started.pop((event.segment_id, target_lang), None)

        audio = (
            np.concatenate(streamed_chunks) if streamed_chunks else np.zeros(0, dtype=np.float32)
        )

        # Post-synth deadline re-check [P1-1-i2]. Budget blown → drop.
        if time.monotonic() > deadline:
            logger.warning(
                "TTS post-synth deadline blown seg=%s over=%.2fs",
                event.segment_id,
                time.monotonic() - deadline,
            )
            state.metrics.tts_dropped_post_synth += 1
            return

        # Double-check meeting still active after synthesis.
        if not state.current_meeting:
            logger.info(
                "TTS post-synthesize: meeting stopped, dropping seg=%s",
                event.segment_id,
            )
            return

        if len(audio) == 0:
            logger.warning(
                "TTS synthesize returned empty audio: seg=%s mode=%s",
                event.segment_id,
                voice_mode,
            )
            continue

        # === Segment-level histograms (ONCE per segment, BEFORE fan-out) ===
        # [P1-5-i2]. This is the ONLY place lag metrics are recorded.
        synth_ms = (time.monotonic() - synth_start) * 1000.0
        state.metrics.tts_synth_ms.append(synth_ms)
        state.metrics.tts_synth_ms_p95 = _percentile(sorted(state.metrics.tts_synth_ms), 0.95)
        _record_segment_lag(event, time.monotonic())
        state.metrics.tts_delivered += 1
        state.metrics.last_delivery_at = time.monotonic()

        # Save TTS output to disk (for replay/download).
        if voice_mode in ("studio", "cloned") and state.current_meeting:
            tts_dir = state.storage._meeting_dir(state.current_meeting.meeting_id) / "tts"
            tts_dir.mkdir(exist_ok=True)
            import soundfile as _sf

            # Multi-target fan-out writes
            # `{segment_id}.{target_lang}.wav` so two targets for the
            # same segment don't collide. Legacy path stays
            # `{segment_id}.wav` so replay/exports keep working.
            wav_name = (
                f"{event.segment_id}.{target_lang}.wav"
                if _MULTI_TARGET_ENABLED
                else f"{event.segment_id}.wav"
            )
            _sf.write(str(tts_dir / wav_name), audio, 24000)

        # Single-shot fan-out: one coherent WAV per segment, no chunk
        # joins on the client side. The per-chunk call that used to
        # live inside _run_stream was removed 2026-04-15 to kill click
        # artifacts — if this call ever goes missing the listener gets
        # nothing, so guard it with an explicit comment.
        if state._audio_out_clients:
            await _send_audio_to_listeners(audio, target_lang, voice_mode)

        logger.info(
            "TTS streamed: seg=%s mode=%s chunks=%d audio_samples=%d ttfa_ms=%.0f total_ms=%.0f",
            event.segment_id,
            voice_mode,
            len(streamed_chunks),
            len(audio),
            first_chunk_ms or 0.0,
            synth_ms,
        )
