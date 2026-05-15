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
import os
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
)
from meeting_scribe.storage import MeetingStorage
from meeting_scribe.translation.queue import TranslationQueue

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    state.config = ServerConfig.from_env()
    state.storage = MeetingStorage(state.config)
    state.resampler = Resampler()

    # Allocate the per-process identity stamped into Phase B progress
    # sidecars. The boot sweep below rewrites any sidecar from a
    # different session_uuid (i.e. a crashed prior process) as a
    # failure-terminal "interrupted" so the API never serves ambiguous
    # in-flight state from a process that no longer exists.
    import uuid as _uuid

    state.session_uuid = _uuid.uuid4().hex

    _recover_result = state.storage.recover_interrupted()
    _orphan_phase_b = state.storage.sweep_orphan_phase_b_sidecars(state.session_uuid)
    if _orphan_phase_b:
        logger.info(
            "phase_b: marked %d orphan sidecar(s) as interrupted at boot: %s",
            len(_orphan_phase_b),
            ", ".join(_orphan_phase_b),
        )
    state.storage.cleanup_retention()

    # Wire the GPU lease's per-backend abort URL builders so Phase B's
    # preempt path can hit /abort/{request_id} on the right service.
    # These are best-effort — backends that don't yet expose /abort
    # return 404/500 and the lease records `_phase_b_backend_dirty`
    # so the operator knows GPU contention is possible.
    try:
        from meeting_scribe.backends.protocol import abort_url
        from meeting_scribe.runtime.gpu_lease import gpu_lease

        _lease = gpu_lease()
        _lease.register_backend(
            "asr",
            lambda rid: abort_url(state.config.asr_vllm_url, rid),
        )
        _lease.register_backend(
            "translate",
            lambda rid: abort_url(state.config.translate_vllm_url, rid),
        )
        _lease.register_backend(
            "diarize",
            lambda rid: abort_url(state.config.diarize_url, rid),
        )
        _lease.register_backend(
            "summary",
            lambda rid: abort_url(state.config.translate_vllm_url, rid),
        )
    except Exception:
        logger.exception("gpu_lease backend registration failed")

    # Resume Phase B for any meetings caught mid-finalize on the prior
    # crash, and sweep leftover finalize artifacts on COMPLETE meetings.
    # Spawned as a fire-and-forget task so the server starts fast even
    # with a backlog of pending finalizes.
    if _recover_result.finalizing_ids or _recover_result.cleanup_ids:
        from meeting_scribe.runtime.finalize_recovery import replay_pending_finalize

        asyncio.create_task(
            replay_pending_finalize(
                _recover_result.finalizing_ids,
                _recover_result.cleanup_ids,
            ),
            name="finalize-recovery",
        )

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

    # Sidecar opt-out (dev only). When ``SCRIBE_DEV_NO_DEVICE_TOUCH=1``
    # is set, skip every block below that mutates shared hardware state —
    # the WiFi AP, BT card profile, and the PipeWire mic capture. A
    # dev instance running alongside production needs this so the two
    # processes don't fight over the AP / mic; production keeps owning
    # those, and the dev sidecar serves UI only. Production never sets
    # the var.
    _no_device_touch = os.environ.get("SCRIBE_DEV_NO_DEVICE_TOUCH") == "1"
    if _no_device_touch:
        logger.info("SCRIBE_DEV_NO_DEVICE_TOUCH=1 — skipping WiFi/BT/mic boot reconciliation")

    # Boot WiFi reconciliation: honor the operator's persisted
    # ``wifi_mode``. Default when no setting is present is ``admin`` —
    # the unit ships with the admin UI exposed over WiFi rather than
    # silently OWE-open, so a fresh GB10 boots into a known
    # password-protected state. ``meeting`` and ``setup`` (legacy OWE)
    # are honored explicitly; ``off`` skips bring-up. Calling
    # ``wifi_up_setup()`` previously clobbered ``wifi_mode`` to
    # ``"setup"`` on every boot, defeating cross-restart persistence.
    # Sidecar mode skips this block entirely (see top-of-function gate).
    if not _no_device_touch:
        try:
            from meeting_scribe.server_support.settings_store import (
                _load_settings_override as _wifi_settings,
            )
            from meeting_scribe.wifi import (
                build_config as _wifi_build_config,
            )
            from meeting_scribe.wifi import (
                wifi_up as _wifi_up,
            )
            from meeting_scribe.wifi import (
                wifi_up_setup as _wifi_up_setup,
            )

            desired_wifi_mode = _wifi_settings().get("wifi_mode", "admin")
            if desired_wifi_mode == "off":
                logger.info("wifi auto-bring-up skipped: wifi_mode=off persisted")
            elif desired_wifi_mode in ("admin", "meeting"):
                await _wifi_up(_wifi_build_config(desired_wifi_mode))
            elif desired_wifi_mode == "setup":
                # Explicit OWE first-touch — only triggered by an operator
                # who set ``wifi_mode=setup`` on purpose.
                await _wifi_up_setup()
            else:
                # Unrecognized value (typo / stale schema). Fall back to
                # ``admin`` rather than OWE-open so a misconfigured settings
                # file never silently broadcasts an unprotected SSID at boot.
                logger.warning(
                    "wifi_mode=%r is not a recognized value — defaulting to admin",
                    desired_wifi_mode,
                )
                await _wifi_up(_wifi_build_config("admin"))
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

    try:
        from meeting_scribe.bt import bt_status_monitor_loop

        _bt_status_monitor_task = asyncio.create_task(
            bt_status_monitor_loop(),
            name="bt-status-monitor",
        )
    except Exception:
        logger.exception("bt status monitor failed to start")
        _bt_status_monitor_task = None

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

    # Audio routing — register the local-sink listener (TTS → local
    # PipeWire sink) and the server-side mic capture (USB / ALSA / BT
    # source → ASR pipeline) according to the operator's persisted
    # selections in ``audio_meeting_*_node`` /
    # ``audio_meeting_mic_active``. Both are no-ops on a freshly
    # provisioned box that hasn't been configured yet. Sidecar mode
    # (SCRIBE_DEV_NO_DEVICE_TOUCH=1) keeps the read-only routing-setting
    # query but skips the three calls inside that mutate hardware state
    # (BT card profile, PipeWire local-sink listener, pw-record mic).
    _local_sink_listeners = []
    try:
        from meeting_scribe.audio.audio_routing import (
            get_routing_settings as _get_routing_settings,
        )
        from meeting_scribe.audio.audio_routing import (
            reconcile_audio_routing as _reconcile_audio_routing,
        )
        from meeting_scribe.audio.interpretation_buffer import InterpretationBuffer
        from meeting_scribe.audio.local_sink import (
            LocalSinkListener,
            ensure_local_sink_listener_registered,
            should_enable_local_sink,
        )
        from meeting_scribe.server_support import admin_notifications as _admin_notifications
        from meeting_scribe.server_support.settings_store import (
            _effective_interpretation_enabled,
            _effective_interpretation_idle_drain_ms,
            _effective_interpretation_pause_flush_ms,
        )
        from meeting_scribe.server_support.settings_store import (
            _load_settings_override as _routing_settings,
        )

        # Prime the in-memory mirror of persisted admin notifications so
        # any banners that survived the prior process are visible from
        # the first /api/status poll.
        _admin_notifications.load_into_state()

        _routing = _get_routing_settings(_routing_settings())
        if _effective_interpretation_enabled():
            state.interpretation_buffer = InterpretationBuffer(
                pause_flush_ms=_effective_interpretation_pause_flush_ms(),
                idle_drain_ms=_effective_interpretation_idle_drain_ms(),
            )

        if should_enable_local_sink() and not _no_device_touch:
            # If the operator's still using the legacy BT-input toggle
            # (no audio_meeting_mic_node configured yet), reconcile the
            # BT card profile to HFP-mSBC so the BT headset's mic +
            # speaker are both live. Falls through silently when no BT
            # device is connected.
            if not _routing["mic_node"]:
                from meeting_scribe.bt import apply_bt_input_state

                try:
                    bt_reconcile = await apply_bt_input_state(active=True)
                    if bt_reconcile["updated"]:
                        logger.info("bt-input boot reconcile: %s", bt_reconcile["updated"])
                    if bt_reconcile["skipped"]:
                        logger.info("bt-input boot skipped: %s", bt_reconcile["skipped"])
                except Exception:
                    logger.exception("bt-input boot reconciliation failed (non-fatal)")
            ensure_local_sink_listener_registered()
            _local_sink_listeners = [
                c for c in state._audio_out_clients if isinstance(c, LocalSinkListener)
            ]

        # Server-side mic capture takes precedence over the
        # browser-mic WS once started — the ws.audio_input handler
        # drops binary frames while ``state.server_mic_active`` is
        # True. Boot reconciliation matches the persisted selection;
        # the admin /api/admin/audio/route endpoint reapplies
        # mid-process. Sidecar mode skips this so production keeps
        # ownership of the mic.
        #
        # Going through ``reconcile_audio_routing()`` (rather than
        # calling ``reconcile_server_mic`` directly) wires up the
        # Phase-1 auto-rebind: a stale persisted ``mic_node`` will
        # be looked up by its persisted ``stable_id`` against the
        # current PipeWire enumeration and rebound if it resolves to
        # exactly one source. Failure paths (ambiguous / unresolved /
        # capture_failed) set ``state.audio_route_status`` so Phase 5
        # meeting-start preflight can refuse to start meetings on a
        # broken mic.
        if not _no_device_touch:
            try:
                await _reconcile_audio_routing()
            except Exception:
                logger.exception("audio-routing boot reconciliation failed (non-fatal)")
    except Exception:
        logger.exception("audio routing boot setup failed")

    logger.info("Meeting Scribe ready on port %d", state.config.port)

    # Interrupted-meeting handling: any recovered meeting from the prior
    # process stays in the meetings list for the operator to manually
    # resume or finalize. The previous dev-mode auto-resume path was
    # removed alongside the rest of the dev_mode plumbing — its only
    # caller was development workflows that no longer exist.
    interrupted = _get_interrupted_meeting()
    if interrupted:
        logger.info(
            "interrupted meeting %s left for manual resume/finalize",
            interrupted,
        )

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
                call so a hot-reload of the translate endpoint flips
                the slide pipeline alongside live translation. Falls
                back to ``translate_vllm_url`` when the runtime knob
                is unset.
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

    # ── mDNS alias publication ─────────────────────────────────
    # The HTTPS leaf cert carries SAN entries for ``meeting-scribe-<id4>
    # .local`` and ``meeting-<pin>.local`` (the operator-friendly form
    # that matches the 4-digit pin printed in the SSID). Avahi-daemon
    # only advertises the system hostname by default, so we publish
    # the per-device aliases as a side process bound to the lifespan.
    _mdns_publishers: list[asyncio.subprocess.Process] = []
    try:
        from meeting_scribe.cli._common import _required_leaf_dns_sans
        from meeting_scribe.server_support.mdns import publish_aliases

        _mdns_publishers = await publish_aliases(_required_leaf_dns_sans())
        if _mdns_publishers:
            logger.info(
                "mdns: published %d alias(es) via avahi-publish-address",
                len(_mdns_publishers),
            )
    except Exception:
        logger.exception("mdns alias publication failed (non-fatal)")

    # ── Captive-gateway idle GC ────────────────────────────────
    # Every 5 minutes, prune ipset entries whose AP-side IP no longer
    # holds a dnsmasq lease. Trusted clients (admin / guest) that
    # disconnect without explicit logout get cleaned up automatically;
    # explicit logout already removes the entry synchronously. The
    # task swallows per-tick failures so a transient lease-file read
    # error doesn't kill the GC.
    _captive_gc_task: asyncio.Task[None] | None = None
    try:
        from meeting_scribe.server_support import firewall_allowlist

        _captive_gc_task = asyncio.create_task(
            firewall_allowlist.gc_loop(),
            name="captive-allowlist-gc",
        )
    except Exception:
        logger.exception("captive gc startup failed (non-fatal)")

    # ── UDS internal speakerphone listener ─────────────────────
    # Separate ASGI app bound to ${XDG_RUNTIME_DIR}/meeting-scribe.sock
    # (0600, user-owned). Holds *only* the internal speakerphone router,
    # so the daemon-facing namespace is unreachable from the public TCP
    # listener by construction. Off-switch:
    # MEETING_SCRIBE_DISABLE_SPEAKERPHONE_UDS=1 (skip entirely; used by
    # the test harness when binding a tmpdir socket inline).
    _speakerphone_uds_stop: asyncio.Event | None = None
    _speakerphone_uds_task: asyncio.Task[None] | None = None
    if os.environ.get("MEETING_SCRIBE_DISABLE_SPEAKERPHONE_UDS") != "1":
        try:
            from meeting_scribe.speakerphone.uds import serve as _serve_uds

            _speakerphone_uds_stop = asyncio.Event()
            _speakerphone_uds_task = asyncio.create_task(
                _serve_uds(stop_event=_speakerphone_uds_stop),
                name="speakerphone-uds",
            )
        except Exception:
            logger.exception(
                "speakerphone UDS startup failed (non-fatal — daemon will retry)",
            )

    yield

    _notify_systemd("STOPPING=1")

    if _speakerphone_uds_stop is not None:
        _speakerphone_uds_stop.set()
    if _speakerphone_uds_task is not None:
        try:
            await asyncio.wait_for(_speakerphone_uds_task, timeout=15)
        except TimeoutError:
            logger.warning("speakerphone UDS shutdown timed out; cancelling")
            _speakerphone_uds_task.cancel()
            with suppress(asyncio.CancelledError):
                await _speakerphone_uds_task

    if _captive_gc_task is not None and not _captive_gc_task.done():
        _captive_gc_task.cancel()
        with suppress(asyncio.CancelledError):
            await _captive_gc_task

    if _mdns_publishers:
        try:
            from meeting_scribe.server_support.mdns import stop_aliases

            await stop_aliases(_mdns_publishers)
        except Exception:
            logger.exception("mdns alias teardown failed (non-fatal)")

    if state.server_mic is not None:
        try:
            await state.server_mic.stop()
        except Exception:
            logger.exception("server-mic teardown failed")
        state.server_mic = None

    if _local_sink_listeners:
        try:
            if state.interpretation_buffer is not None:
                await state.interpretation_buffer.drain_for_stop()
            from meeting_scribe.audio.local_sink import unregister_local_sink_listener

            for listener in list(_local_sink_listeners):
                unregister_local_sink_listener(listener)
        except Exception:
            logger.exception("local-sink listener teardown failed")
        finally:
            state.interpretation_buffer = None

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
        _bt_status_monitor_task,
    ):
        if _t is None:
            continue
        _t.cancel()
    for _t in (
        _retry_task,
        _loop_lag_task,
        _health_eval_task,
        _silence_watchdog_task,
        _mem_pressure_task,
        _bt_status_monitor_task,
    ):
        if _t is None:
            continue
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
