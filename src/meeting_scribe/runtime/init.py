"""Backend initialization helpers + the post-startup retry loop.

Five entry points:

* ``init_asr`` — Qwen3-ASR vLLM backend (the only ASR path on GB10).
* ``init_diarization`` — pyannote sortformer backend (optional).
* ``init_translation`` — vLLM translation backend with a warm-up
  request that uses an extended client timeout so model
  compilation doesn't time out during cold start.
* ``init_tts`` — Qwen3-TTS backend; tolerates up to 30 s of "not
  ready" while the container loads.
* ``retry_failed_backends`` — long-lived loop that retries each
  failed backend every 10 s and watches healthy backends for the
  ``degraded`` flag (CUDA crash recovery via container restart).

These all live module-scope (not nested inside ``lifespan``) so the
retry loop can call them after lifespan startup has returned. A 2026
incident was caused by a nested-vs-module name shadow that left
TTS / diarization permanently dead if they missed the initial 30 s
window — the retry loop only saw the broken ``NameError`` debug log.

The transcript-event callback (``_process_event``) and translation
broadcast (``_broadcast_translation``) are imported lazily inside
``init_asr`` / ``init_translation`` so this module doesn't pull in
the live transcript pipeline at import time.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from meeting_scribe.runtime import state
from meeting_scribe.server_support.backend_health import _restart_container
from meeting_scribe.tts.worker import _start_tts_worker

logger = logging.getLogger(__name__)


async def init_tts() -> None:
    """Initialize the TTS backend.

    Module-scope so ``retry_failed_backends`` can call it after
    lifespan startup has returned. The retry loop used to reference
    a nested version of this function that did not exist at module
    scope, which silently raised NameError and left TTS permanently
    dead if it missed the initial 30 s startup window.

    Idempotent: a successful call sets ``state.tts_backend`` /
    ``state.tts_semaphore`` and starts the FIFO TTS worker. If
    ``state.tts_backend`` is already set we are a no-op.
    """
    if state.tts_backend is not None:
        return
    try:
        from meeting_scribe.backends.tts_qwen3 import Qwen3TTSBackend

        tts_url = state.config.tts_vllm_url or (
            state.config.translate_vllm_url if state.config.translate_backend == "vllm" else None
        )
        tts = Qwen3TTSBackend(vllm_url=tts_url or None)
        # Retry: TTS container may still be starting during parallel init.
        for attempt in range(15):
            await tts.start()
            if tts.available:
                state.tts_backend = tts
                state.tts_semaphore = asyncio.Semaphore(1)
                _start_tts_worker()
                logger.info("TTS backend ready (url=%s)", tts_url)
                return
            last_err = getattr(tts, "_last_error", None)
            if attempt < 14:
                logger.info(
                    "TTS not ready (attempt %d/15, last_error=%r), retrying in 2s...",
                    attempt + 1,
                    last_err,
                )
                await asyncio.sleep(2)
        logger.warning(
            "TTS disabled: not available after 30s of retries (last_error=%r)",
            getattr(tts, "_last_error", None),
        )
    except Exception as e:
        logger.warning("TTS disabled: %r", e, exc_info=True)


async def init_diarization() -> None:
    """Initialize the diarization backend. Module-scope for the retry loop.

    See ``init_tts`` for why this was extracted from its previous
    nested definition inside ``lifespan``.
    """
    if state.diarize_backend is not None:
        return
    if not state.config.diarize_enabled:
        return
    try:
        from meeting_scribe.backends.diarize_sortformer import SortformerBackend
        from meeting_scribe.speaker.verification import SpeakerVerifier

        verifier = SpeakerVerifier(state.enrollment_store)
        # Rolling-window diarization config:
        # window_seconds=16 — pyannote needs ~15 s of multi-speaker audio
        # to separate clusters; shorter chunks collapse everything into
        # one. flush_interval_seconds=4 — re-diarize every 4 s of new
        # audio; each flush re-runs the FULL window so pyannote always
        # has context. max_speakers=6 — typical 4-6 person meetings.
        diarize_be = SortformerBackend(
            url=state.config.diarize_url,
            verifier=verifier,
            flush_interval_seconds=4.0,
            window_seconds=16.0,
        )
        await diarize_be.start(max_speakers=6)
        state.diarize_backend = diarize_be
        logger.info("Diarization backend ready")
    except Exception as e:
        logger.warning("Diarization disabled: %r", e, exc_info=True)


async def init_translation(
    default_pair: list[str] | tuple[str, ...] = ("en", "ja"),
) -> None:
    """Initialize the translation backend based on ``state.config``."""
    if not state.config.translate_enabled:
        return

    t0 = time.monotonic()
    try:
        from meeting_scribe.backends.translate_vllm import VllmTranslateBackend

        # Realtime URL (smaller live-path model, optional) takes precedence
        # over the main translate URL.
        realtime_url = state.config.translate_realtime_vllm_url or state.config.translate_vllm_url
        be = VllmTranslateBackend(
            base_url=realtime_url,
            model=state.config.translate_vllm_model or None,
        )
        await be.start()
        # First request may be slow (model compilation/warmup) — use
        # extended timeout.
        old_client = be._client
        be._client = httpx.AsyncClient(timeout=120.0)
        if old_client:
            await old_client.aclose()
        warmup = await be.translate("Hello", default_pair[1], default_pair[0])
        await be._client.aclose()
        be._client = httpx.AsyncClient(timeout=be._timeout)
        state.metrics.translate_warmup_ms = (time.monotonic() - t0) * 1000
        state.translate_backend = be
        logger.info(
            "Translation: vLLM (%.0fms, test: 'Hello' → '%s')",
            state.metrics.translate_warmup_ms,
            warmup[:30],
        )
    except Exception as e:
        logger.warning("Translation vLLM backend failed: %s", e)


async def init_asr(
    default_pair: list[str] | tuple[str, ...] = ("en", "ja"),
) -> None:
    """Initialize the Qwen3-ASR vLLM backend.

    The only ASR path on GB10 — the WhisperLiveKit fallback was
    removed 2026-04-13.
    """
    from meeting_scribe.backends.asr_vllm import VllmASRBackend
    from meeting_scribe.pipeline.transcript_event import _process_event

    be = VllmASRBackend(
        base_url=state.config.asr_vllm_url,
        languages=default_pair,
    )
    be.set_event_callback(_process_event)
    t0 = time.monotonic()
    try:
        await be.start()
    except Exception as e:
        logger.warning("vLLM ASR failed to start: %s", e)
        return
    state.metrics.asr_load_ms = (time.monotonic() - t0) * 1000
    state.asr_backend = be
    logger.info("ASR: vLLM Qwen3-ASR loaded in %.0fms", state.metrics.asr_load_ms)


async def retry_failed_backends(
    default_pair: list[str] | tuple[str, ...],
) -> None:
    """Background task: periodically retry init for backends that failed at startup.

    Checks every 10 s. Stops retrying each backend once it succeeds.
    Handles the case where containers are still loading when the
    server starts AND the case where a healthy backend goes
    ``degraded`` mid-meeting (CUDA crash) — the latter triggers a
    container restart through ``_restart_container``.
    """
    from meeting_scribe.pipeline.transcript_event import _broadcast_translation
    from meeting_scribe.translation.queue import TranslationQueue

    while True:
        await asyncio.sleep(10)
        try:
            if state.translate_backend is None and state.config.translate_enabled:
                logger.info("Retrying translation backend init...")
                await init_translation(default_pair)
                if state.translate_backend:
                    if state.translation_queue is None:
                        state.translation_queue = TranslationQueue(
                            maxsize=state.config.translate_queue_maxsize,
                            concurrency=state.config.translate_queue_concurrency,
                            timeout=state.config.translate_timeout_seconds,
                            on_result=_broadcast_translation,
                            languages=default_pair,
                        )
                        await state.translation_queue.start(state.translate_backend)
                    logger.info("Translation backend recovered")

            if state.diarize_backend is None:
                logger.info("Retrying diarization backend init...")
                await init_diarization()
                if state.diarize_backend:
                    logger.info("Diarization backend recovered")
            elif getattr(state.diarize_backend, "degraded", False):
                # Backend exists but CUDA crashed — check if container restarted
                try:
                    healthy = await state.diarize_backend.check_health()
                    if healthy:
                        logger.info("Diarization recovered from degraded state")
                    else:
                        await _restart_container("scribe-diarization")
                except Exception:
                    await _restart_container("scribe-diarization")

            if state.tts_backend is None:
                logger.info("Retrying TTS backend init...")
                await init_tts()
                if state.tts_backend:
                    logger.info("TTS backend recovered")
            elif state.tts_backend.degraded:
                # TTS exists but CUDA crashed — check if container restarted
                healthy = await state.tts_backend.check_health()
                if healthy:
                    logger.info("TTS recovered from degraded state (container restarted)")
                else:
                    await _restart_container("scribe-tts")

            if state.asr_backend is None:
                logger.info("Retrying ASR backend init...")
                await init_asr(default_pair)
                if state.asr_backend:
                    logger.info("ASR backend recovered")

            # Continuous background health polling [Phase 2]: keep
            # looping for the whole process lifetime so we detect
            # late failures (e.g. TTS container CUDA dispatch failure
            # mid-meeting) and auto-recover via the existing
            # degraded-check paths above. The old "return on
            # all-ready" branch was removed — it silenced every
            # post-startup regression.
        except Exception as e:
            # Upgraded from debug → warning: this handler previously
            # hid real failures (including NameError from the dead
            # `_init_tts`/`_init_diarization` references that silently
            # broke TTS/diarization recovery forever). If you land
            # here, the retry loop is not actually retrying — fix the
            # caller, don't just accept the log line.
            logger.warning("Backend retry error: %r", e, exc_info=True)
