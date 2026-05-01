"""ASR recovery supervisor — background task that drives the W6a
state machine through to a usable backend after a watchdog
escalation.

Why a separate task: the audio ingest path (`process_audio_bytes`,
called inline from `ws/audio_input.py`) MUST stay non-blocking. A
watchdog escalation triggers `VllmASRBackend._begin_recovery_pending`
synchronously on the audio thread (cheap — just bumps generation +
sets a flag), and this supervisor handles the heavy lifting (probe
polling, optional `compose_restart`, replay) without holding the
WS receive loop.

Why poll the W4 synthetic-speech probe (NOT `/v1/models`): control-
plane liveness does not prove inference works. The 2026-04-30
incident's `cudaErrorNotPermitted` cascade left ASR responding to
`/v1/models` while every transcribe request hung. Reusing the same
probe shape that meeting-start preflight uses guarantees a backend
healthy enough to admit a meeting is also healthy enough to exit
recovery — no false-failure asymmetry.

Recovery exit handles all three paths uniformly:
- Spontaneous self-heal (probe succeeds without supervisor doing
  anything destructive).
- Docker `restart: unless-stopped` auto-restart (when
  AUTO_RECREATE=0).
- Explicit `compose_restart('vllm-asr', recreate=True)` (when
  AUTO_RECREATE=1 + 30s threshold + circuit breaker open).

In all three cases the same path runs after probe success: replay
recording.pcm[recovery_start_offset:current_offset] through the
recovered backend so transcript ordering preserves the missing
window.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable

from meeting_scribe.runtime import state
from meeting_scribe.runtime.synthetic_probe import asr_synthetic_probe

logger = logging.getLogger(__name__)


# Recovery-supervisor knobs. Module-level so tests can patch and the
# integration drill gets a deterministic small value.
SUPERVISOR_POLL_INTERVAL_S = 3.0
RECREATE_AFTER_PENDING_S = 30.0  # only consider recreate after this much
                                  # time in PENDING with failing probes
CIRCUIT_BREAKER_WINDOW_S = 600.0  # max 1 recreate per 10 min


def _auto_recreate_enabled() -> bool:
    """Read the env-var gate. Default OFF for the first week of
    production — counter + probe + decision-logging all run, only
    the destructive `compose_restart` is gated."""
    return os.environ.get("SCRIBE_RELIABILITY_AUTO_RECREATE", "0") == "1"


async def asr_recovery_loop() -> None:
    """Long-lived supervisor task. Started in `runtime.lifespan` and
    cancelled at shutdown. Awaits the backend's `_recovery_requested`
    event in a loop; on each signal, drives the state machine through
    to NORMAL via probe-poll → optional recreate → replay."""
    backend = state.asr_backend
    if backend is None:
        logger.info("recovery_supervisor: no asr_backend on state, exiting")
        return

    # 1-element list cell so _drive_one_recovery's nested closures can
    # mutate the value across iterations.
    last_recreate_ts: list[float] = [0.0]

    while True:
        try:
            await backend._recovery_requested.wait()
        except asyncio.CancelledError:
            raise

        try:
            await _drive_one_recovery(
                backend,
                last_recreate_ts_getter=lambda: last_recreate_ts[0],
                last_recreate_ts_setter=lambda ts: last_recreate_ts.__setitem__(0, ts),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "recovery_supervisor: unhandled exception during recovery; "
                "clearing flag and continuing"
            )
        finally:
            # Whatever happened, clear the event so the next watchdog
            # escalation can re-trigger us. State transition itself is
            # owned by replay_until_caught_up + _drive_one_recovery.
            backend._recovery_requested.clear()


async def _drive_one_recovery(
    backend,
    *,
    last_recreate_ts_getter: Callable[[], float],
    last_recreate_ts_setter: Callable[[float], None],
) -> None:
    """One iteration of the supervisor's recovery cycle. Polls the
    W4 ASR probe; optionally triggers `compose_restart` after the
    grace window with the circuit breaker; on probe success,
    transitions to REPLAYING and calls `replay_until_caught_up`.

    State transitions (driven from the backend):
        NORMAL ──[watchdog escalation]──▶ RECOVERY_PENDING
        RECOVERY_PENDING ──[probe ok]──▶ REPLAYING
        REPLAYING ──[replay done]──▶ NORMAL
    """
    started_at = time.monotonic()
    cfg = state.config
    asr_url = cfg.asr_vllm_url
    asr_model = cfg.asr_model
    histogram = state.metrics.asr_request_rtt_ms

    logger.warning(
        "recovery_supervisor: ASR recovery requested (start_offset=%s, "
        "generation=%d, AUTO_RECREATE=%s)",
        backend._recovery_start_offset,
        backend._recovery_generation,
        _auto_recreate_enabled(),
    )

    while True:
        result = await asr_synthetic_probe(asr_url, asr_model, histogram)
        if result.ok:
            break

        # Probe failed. Decide whether to trigger an explicit recreate.
        time_in_pending_s = time.monotonic() - started_at
        if (
            _auto_recreate_enabled()
            and time_in_pending_s >= RECREATE_AFTER_PENDING_S
        ):
            now = time.monotonic()
            last_ts = last_recreate_ts_getter()
            if (now - last_ts) >= CIRCUIT_BREAKER_WINDOW_S:
                logger.warning(
                    "recovery_supervisor: probe failed for %.0fs, "
                    "triggering compose_restart vllm-asr (recreate)",
                    time_in_pending_s,
                )
                from meeting_scribe.infra.compose import compose_restart
                try:
                    await asyncio.to_thread(
                        compose_restart, "vllm-asr", recreate=True
                    )
                except Exception:
                    logger.exception(
                        "recovery_supervisor: compose_restart vllm-asr failed"
                    )
                last_recreate_ts_setter(now)
            else:
                # Circuit breaker tripped — log + (future) emit
                # structured event. Keep polling: Docker auto-restart
                # may still bring the backend back even if our explicit
                # recreate is suppressed.
                remaining = CIRCUIT_BREAKER_WINDOW_S - (now - last_ts)
                logger.error(
                    "recovery_supervisor: ASR recovery circuit breaker "
                    "tripped (%.0fs remaining in 10-min window); "
                    "skipping compose_restart, polling probe only",
                    remaining,
                )
        elif (
            not _auto_recreate_enabled()
            and time_in_pending_s >= RECREATE_AFTER_PENDING_S
        ):
            # AUTO_RECREATE=0 path. Log the would-have-recreated
            # decision so operators can validate the supervisor's
            # judgment over the first week of production.
            logger.warning(
                "recovery_supervisor: would_recreate (probe failed for "
                "%.0fs, AUTO_RECREATE=0 — relying on Docker "
                "restart:unless-stopped)",
                time_in_pending_s,
            )

        await asyncio.sleep(SUPERVISOR_POLL_INTERVAL_S)

    # Probe succeeded. Transition to REPLAYING and drain the audio
    # window through the recovered backend.
    if backend._recovery_start_offset is None:
        logger.warning(
            "recovery_supervisor: probe succeeded but no "
            "_recovery_start_offset captured; skipping replay"
        )
        backend._recovery_state = "NORMAL"
        return

    logger.warning(
        "recovery_supervisor: ASR probe succeeded after %.1fs; "
        "transitioning to REPLAYING (start_offset=%d)",
        time.monotonic() - started_at,
        backend._recovery_start_offset,
    )
    backend._recovery_state = "REPLAYING"
    try:
        replay_end = await backend.replay_until_caught_up(
            backend._recovery_start_offset
        )
    except Exception:
        logger.exception(
            "recovery_supervisor: replay_until_caught_up raised; "
            "returning to NORMAL with possible transcript hole"
        )
        replay_end = backend._recovery_start_offset
    finally:
        backend._recovery_state = "NORMAL"
        backend._recovery_start_offset = None
        backend._watchdog_consecutive_fires = 0
        logger.info(
            "recovery_supervisor: returned to NORMAL (replay_end=%d)",
            replay_end,
        )
