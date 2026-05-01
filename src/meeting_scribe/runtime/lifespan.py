"""FastAPI lifespan — boots backends + background tasks for the whole process.

Single entry point ``lifespan`` (an ``asynccontextmanager`` consumed
by ``FastAPI(..., lifespan=lifespan)``). On startup, in order:

  1. Construct ``state.config`` / ``state.storage`` / ``state.resampler``
     and recover any interrupted meeting.
  2. Wire diagnostics sinks (ring buffer + rotating server.log).
  3. Validate + persist the regulatory domain.
  4. Auto-bring-up the WiFi AP if the persisted ``wifi_mode != "off"``.
  5. Run all backend init helpers concurrently
     (``runtime.init.init_asr/init_tts/init_diarization``,
     translation + queue, name extractor, furigana).
  6. Install crash hooks; spawn the retry loop, loop-lag monitor, TTS
     health evaluator, and silence watchdog.
  7. Dev-mode auto-resume of a recently-interrupted meeting.
  8. Slide translation worker (if LibreOffice is available).
  9. Post-ready bookkeeping (preflight) + systemd READY notification.
 10. Tmux orphan-client cleanup + terminal ticket store sweep task.

On shutdown: cancel background tasks, stop every backend handle, and
close all active terminal PTYs.

Pulled out of ``server.py`` once every dependency was on a stable
extracted surface.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI

from meeting_scribe import diagnostics as _diag
from meeting_scribe.audio.resample import Resampler
from meeting_scribe.config import ServerConfig
from meeting_scribe.runtime import state
from meeting_scribe.runtime.net import _notify_systemd
from meeting_scribe.server_support.active_meeting import _get_interrupted_meeting
from meeting_scribe.server_support.broadcast import _broadcast_json
from meeting_scribe.server_support.crash_tracking import _install_crash_hooks
from meeting_scribe.server_support.regdomain import (
    _current_regdomain,
    _ensure_regdomain,
    _ensure_regdomain_persistent,
)
from meeting_scribe.server_support.settings_store import (
    _effective_regdomain,
    _is_dev_mode,
)
from meeting_scribe.storage import MeetingStorage
from meeting_scribe.translation.queue import TranslationQueue

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    state.config = ServerConfig.from_env()
    state.storage = MeetingStorage(state.config)
    state.resampler = Resampler()

    state.storage.recover_interrupted()
    state.storage.cleanup_retention()

    # Wire diagnostics sinks (ring buffer + rotating server.log) so
    # the web Diagnostics view can show issues + tail logs without a
    # terminal.
    _diag.setup_diagnostics_logging(state.config.meetings_dir.parent)

    # Source-drift sentinel: emits a WARNING per recipe ↔ compose
    # mismatch so a host that booted from a stale image / hand-edited
    # compose surfaces the divergence in logs. CI test
    # (tests/test_recipes.py::TestComposeRecipeDriftGuard) is the
    # strict gate; this is the runtime safety net. See
    # `meeting_scribe.infra.compose.assert_recipe_source_parity`.
    from meeting_scribe.infra.compose import warn_on_recipe_source_drift

    warn_on_recipe_source_drift()

    # Regulatory domain must be JP before any WiFi AP work happens.
    # We enforce + verify at startup so a cold boot doesn't produce an
    # AP that phones can't connect to (country 00 caps 5 GHz TX power).
    # We ALSO install the persistent modprobe file so cfg80211 starts
    # in JP on the next reboot without any runtime sudo.
    try:
        loop = asyncio.get_event_loop()
        persistent_ok = await loop.run_in_executor(None, _ensure_regdomain_persistent)
        if not persistent_ok:
            logger.warning(
                "could not install persistent /etc/modprobe.d/cfg80211-jp.conf — "
                "regdomain will still be set at runtime but may revert on reboot",
            )
        runtime_ok = await loop.run_in_executor(None, _ensure_regdomain)
        target_regdomain = _effective_regdomain()
        if runtime_ok:
            logger.info(
                "regulatory domain validated at startup: country=%s",
                target_regdomain,
            )
        else:
            logger.error(
                "STARTUP CHECK FAILED: regulatory domain is %r (expected %s) — "
                "WiFi hotspot will not work until this is fixed. Try: "
                "sudo iw reg set %s && sudo modprobe -r cfg80211 && sudo modprobe cfg80211",
                _current_regdomain(),
                target_regdomain,
                target_regdomain,
            )
    except Exception as exc:
        logger.error("regdomain startup check raised: %s", exc)

    # Auto-bring-up WiFi AP if wifi_mode != "off" in settings. This
    # replaces the old `sddc gb10 hotspot up` step that used to run
    # before meeting-scribe. wifi_up() handles regdomain, captive
    # portal, firewall, AP activation, and state file write.
    try:
        from meeting_scribe.server_support.settings_store import (
            _load_settings_override as _wifi_settings,
        )
        from meeting_scribe.wifi import build_config as _build_cfg
        from meeting_scribe.wifi import wifi_up as _wifi_up

        wifi_mode = _wifi_settings().get("wifi_mode", "off")
        if wifi_mode != "off":
            logger.info("WiFi auto-bring-up: mode=%s", wifi_mode)
            cfg = _build_cfg(wifi_mode, None, None, "a", 36)
            await _wifi_up(cfg)
            logger.info("WiFi AP started in %s mode", wifi_mode)
        else:
            logger.info("WiFi mode is 'off' — skipping AP bring-up")
    except Exception as exc:
        logger.error("WiFi auto-bring-up failed: %s", exc)

    # Backend init helpers + the post-startup retry loop live in
    # ``meeting_scribe.runtime.init``.
    from meeting_scribe.pipeline.transcript_event import _broadcast_translation
    from meeting_scribe.runtime import init as _init

    async def _init_name_extractor():
        if (
            state.config.name_extraction_backend not in ("llm", "auto")
            or state.config.translate_backend != "vllm"
        ):
            return
        try:
            from meeting_scribe.speaker.name_llm import LLMNameExtractor

            extractor = LLMNameExtractor(base_url=state.config.translate_vllm_url)
            await extractor.start()
            if extractor.available:
                state.name_extractor = extractor
        except Exception as e:
            logger.info("LLM name extraction disabled: %s", e)

    async def _init_translation_and_queue():
        await _init.init_translation(default_pair)
        if state.translate_backend and state.config.translate_enabled:
            state.translation_queue = TranslationQueue(
                maxsize=state.config.translate_queue_maxsize,
                concurrency=state.config.translate_queue_concurrency,
                timeout=state.config.translate_timeout_seconds,
                on_result=_broadcast_translation,
                languages=default_pair,
            )
            await state.translation_queue.start(state.translate_backend)

    async def _init_furigana():
        """Start the furigana backend. pykakasi is a hard dep — if it
        fails to load we raise so lifespan aborts and the operator
        sees the problem immediately instead of discovering mid-meeting
        that Japanese segments have no ruby text."""
        from meeting_scribe.backends.furigana import FuriganaBackend

        state.furigana_backend = FuriganaBackend()
        await state.furigana_backend.start()
        logger.info("Furigana backend ready")

    from meeting_scribe.languages import parse_languages

    default_pair = parse_languages(state.config.default_language_pair)

    await asyncio.gather(
        _init.init_asr(default_pair),
        _init_translation_and_queue(),
        _init.init_tts(),
        _init.init_diarization(),
        _init_name_extractor(),
        _init_furigana(),
    )

    # Install crash hooks now that the loop is running.
    _install_crash_hooks()

    # Start background retry for any backends that failed to init
    _retry_task = asyncio.create_task(
        _init.retry_failed_backends(default_pair),
        name="retry-failed-backends",
    )

    # Phase 2: event-loop lag monitor + TTS health evaluator run for
    # the whole process lifetime (not just until backends are ready).
    # The lag monitor + silence watchdog live in
    # ``runtime.health_monitors`` and the TTS health evaluator in
    # ``runtime.metrics``.
    from meeting_scribe.runtime import health_monitors as _health_monitors
    from meeting_scribe.runtime import metrics as _metrics

    _loop_lag_task = asyncio.create_task(
        _health_monitors.loop_lag_monitor(), name="loop-lag-monitor"
    )
    _health_eval_task = asyncio.create_task(
        _metrics.tts_health_evaluator(), name="tts-health-evaluator"
    )
    _silence_watchdog_task = asyncio.create_task(
        _health_monitors.silence_watchdog_loop(), name="silence-watchdog"
    )
    # Memory-pressure canary — added 2026-05-01 after the 2026-04-30
    # OOM that killed scribe with no in-process warning. PSI samples
    # surface host swap thrash before the kernel global-OOMs.
    _mem_pressure_task = asyncio.create_task(
        _health_monitors.mem_pressure_monitor(), name="mem-pressure-monitor"
    )

    # W6b: ASR recovery supervisor. Awaits backend's
    # `_recovery_requested` event (set by the watchdog escalation
    # at consecutive_fires>=3) and drives the state machine through
    # probe-poll → optional compose_restart → replay. See
    # `runtime/recovery_supervisor.py` for the full design.
    from meeting_scribe.runtime.recovery_supervisor import asr_recovery_loop

    _asr_recovery_task = asyncio.create_task(asr_recovery_loop(), name="asr-recovery-supervisor")

    # A4: replay any pending_translations.jsonl backlogs left by a
    # prior crash / partial finalize. Runs as a background task so the
    # main lifespan completes immediately; the recovery work blocks
    # only itself.
    from meeting_scribe.runtime.translation_recovery import (
        replay_pending_translations,
    )

    _translation_recovery_task = asyncio.create_task(
        replay_pending_translations(), name="translation-recovery"
    )

    logger.info("Meeting Scribe ready on port %d", state.config.port)

    # Dev mode: auto-resume interrupted meeting on server restart, but
    # ONLY if it was interrupted recently. An old interrupted meeting
    # would otherwise pull the entire audio through the reprocess
    # pipeline (5-chunk chunked diarization for a 33-min meeting took
    # 90s of event-loop starvation on 2026-04-13) in parallel with any
    # new live meeting, which manifests as audio drift, WS
    # disconnects, and what feels like "meetings randomly crashing".
    if _is_dev_mode():
        interrupted = _get_interrupted_meeting()
        if interrupted:
            stale_s: float | None = None
            try:
                pcm_path = state.storage._meeting_dir(interrupted) / "audio" / "recording.pcm"
                if pcm_path.exists():
                    stale_s = time.time() - pcm_path.stat().st_mtime
            except Exception:
                stale_s = None

            _AUTO_RESUME_MAX_AGE_S = 120.0
            if stale_s is None or stale_s > _AUTO_RESUME_MAX_AGE_S:
                logger.info(
                    "Dev mode: NOT auto-resuming interrupted meeting %s "
                    "(audio age=%s — over %.0fs threshold; it will appear "
                    "in the meetings list for manual re-open or reprocess).",
                    interrupted,
                    f"{stale_s:.0f}s" if stale_s is not None else "unknown",
                    _AUTO_RESUME_MAX_AGE_S,
                )
            else:
                logger.info(
                    "Dev mode: auto-resuming interrupted meeting %s (audio age=%.0fs)",
                    interrupted,
                    stale_s,
                )
                try:
                    from meeting_scribe.routes.meeting_lifecycle import (
                        _do_resume_meeting,
                    )

                    result = await _do_resume_meeting(interrupted)
                    if result.status_code == 200:
                        logger.info(
                            "Dev mode: meeting %s resumed successfully",
                            interrupted,
                        )
                    else:
                        raw_body = getattr(result, "body", b"") or b""
                        body = (
                            raw_body.decode()
                            if isinstance(raw_body, bytes)
                            else bytes(raw_body).decode()
                        )
                        logger.warning(
                            "Dev mode: meeting resume returned %d: %s",
                            result.status_code,
                            body[:200],
                        )
                except Exception as e:
                    logger.warning("Dev mode: auto-resume failed: %s", e)

    # ── Slide translation worker ──────────────────────────────
    try:
        from meeting_scribe.slides.job import SlideJobRunner
        from meeting_scribe.slides.worker import check_worker_available

        worker_ok = await check_worker_available()
        if worker_ok:

            async def _slide_translate_fn(
                text: str,
                source_lang: str,
                target_lang: str,
                system_prompt: str = "",
                max_tokens: int = 128,
            ) -> str | None:
                """Translate slide text via shared vLLM with lower priority.

                Reads ``slide_translate_url`` from runtime-config per
                call so a hot-reload (Phase 6/7 of the 3.6 plan) flips
                the slide pipeline's endpoint alongside the
                live-translation one. Falls back to
                ``translate_vllm_url`` so production behavior is
                unchanged when the runtime knob is unset.
                """
                if state.translate_backend is None:
                    return None

                from meeting_scribe import runtime_config as _rc

                client = getattr(state.translate_backend, "_client", None)
                model = getattr(state.translate_backend, "_model", None)
                if client is None or model is None:
                    return None
                base_url = _rc.get("slide_translate_url", state.config.translate_vllm_url)
                try:
                    resp = await client.post(
                        f"{base_url.rstrip('/')}/v1/chat/completions",
                        json={
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": text},
                            ],
                            "temperature": 0.3,
                            "max_tokens": max_tokens,
                            "stream": False,
                            "chat_template_kwargs": {"enable_thinking": False},
                            # Same priority as coding agent (10) —
                            # lower than live transcript (-10) under
                            # vLLM's priority scheduler
                            "priority": 10,
                        },
                    )
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"].strip()
                except Exception as exc:
                    logger.warning("Slide translation failed: %s", exc)
                    return None

            state.slide_job_runner = SlideJobRunner(
                meetings_dir=state.storage._meetings_dir,
                translate_fn=_slide_translate_fn,
                broadcast_fn=_broadcast_json,
            )
            state.slides_enabled = True
            logger.info("Slide translation enabled (worker image found)")
        else:
            logger.info("Slide translation disabled (LibreOffice not found)")
    except Exception as exc:
        logger.warning("Slide translation init failed: %s", exc)

    # Inline post-ready bookkeeping: write `last-good-boot`, clear any
    # stale `BOOT_BLOCKED` marker, and tell systemd we're ready. Doing
    # this in-process (rather than via a separate ExecStartPost hook)
    # guarantees we only record "boot is healthy" *after* every backend
    # has actually initialised, because if the lifespan coroutine
    # hadn't run to this point we wouldn't reach these calls.
    try:
        from meeting_scribe import preflight as _pf

        _pf.write_post_ready()
    except Exception as e:
        logger.warning("post-ready bookkeeping failed (non-fatal): %r", e)

    _notify_systemd("READY=1")

    # Clean up tmux clients that were orphaned by a previous
    # meeting-scribe process. Only runs ONCE per startup — after this
    # the socket belongs to us and any future clients are ours.
    try:
        from meeting_scribe.terminal.tmux_helper import detach_orphan_clients

        orphans = await detach_orphan_clients()
        if orphans:
            logger.info("tmux: detached %d orphan client(s) at startup", orphans)
    except Exception:
        logger.exception("tmux orphan-client cleanup failed (non-fatal)")

    _terminal_sweep_task = asyncio.create_task(state._terminal_ticket_store.sweep())

    yield

    _notify_systemd("STOPPING=1")

    _terminal_sweep_task.cancel()
    with suppress(asyncio.CancelledError):
        await _terminal_sweep_task
    # Close all active terminal PTYs on shutdown. We intentionally do
    # NOT kill the tmux server — its sessions persist across restarts,
    # which is exactly the feature.
    try:
        await state._terminal_registry.close_all(reason="shutdown")
    except Exception:
        logger.exception("terminal_registry.close_all failed")

    for _t in (
        _retry_task,
        _loop_lag_task,
        _health_eval_task,
        _silence_watchdog_task,
        _mem_pressure_task,
    ):
        _t.cancel()
    for _t in (
        _retry_task,
        _loop_lag_task,
        _health_eval_task,
        _silence_watchdog_task,
        _mem_pressure_task,
    ):
        try:
            await _t
        except asyncio.CancelledError:
            pass

    # Shutdown
    if state.name_extractor:
        await state.name_extractor.stop()
    if state.diarize_backend:
        await state.diarize_backend.stop()
    if state.translation_queue:
        await state.translation_queue.stop()
    if state.tts_backend:
        await state.tts_backend.stop()
    if state.asr_backend:
        await state.asr_backend.stop()
    if state.translate_backend:
        await state.translate_backend.stop()
