"""Long-lived health monitors that run for the whole process lifetime.

Three background tasks owned by ``server.lifespan`` (alongside the
TTS-coupled ``_tts_health_evaluator`` and the meeting-scoped speaker
loops, which stay in ``server.py`` for now):

* ``loop_lag_monitor`` — measures async event-loop starvation every
  500 ms and dumps a stack sample on any tick > 1500 ms. Records
  every sample into ``state.metrics.loop_lag_ms``.

* ``silence_watchdog_loop`` — emits a ``meeting_warning`` WS broadcast
  when audio ingestion stalls during a ``recording``-state meeting,
  and a ``meeting_warning_cleared`` event when audio resumes. Reads
  the timestamp of the last audio chunk from
  ``state.last_audio_chunk_ts`` (mutated by ``ws.audio_input``).

* ``mem_pressure_monitor`` — samples ``/proc/pressure/memory`` (PSI)
  and emits WARN/CRIT logs the moment the host starts stalling on
  memory. Added 2026-05-01 after the 2026-04-30 17:35 incident where
  the system swapped itself dry while we had no in-process signal —
  the kernel chose meeting-scribe as its OOM victim before any
  user-space code knew the host was even unhappy.

Pulled out of ``server.py`` so the lifespan orchestrator stays a thin
shell.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from pathlib import Path

from meeting_scribe.runtime import state
from meeting_scribe.server_support.broadcast import _broadcast_json

logger = logging.getLogger(__name__)

# ── Loop lag monitor ──────────────────────────────────────────

_LOOP_LAG_TICK_S = 0.5
_LOOP_LAG_WARN_MS = 250.0


async def loop_lag_monitor() -> None:
    """Measure async event-loop starvation every 500 ms.

    Records lag = (actual wake delta) - (scheduled sleep) in
    ``state.metrics.loop_lag_ms``. Warns when a single tick > 250 ms
    — means something is blocking the loop and TTS / WS fan-out will
    suffer regardless of queue depth.

    On a big spike (>1500 ms) also dumps a stack sample of every
    running task so we can see WHICH coroutine was blocking the loop.
    """
    while True:
        try:
            before = time.monotonic()
            await asyncio.sleep(_LOOP_LAG_TICK_S)
            lag_ms = (time.monotonic() - before - _LOOP_LAG_TICK_S) * 1000.0
            state.metrics.loop_lag_ms.append(max(0.0, lag_ms))
            if lag_ms > _LOOP_LAG_WARN_MS:
                logger.warning("Event loop lag: %.0fms", lag_ms)
            # Big spike → capture what was actually running so we can
            # pinpoint the culprit periodic loop. Cheap — only fires on
            # pathological lags, not routine 300 ms hiccups.
            if lag_ms > 1500:
                try:
                    tasks = asyncio.all_tasks()
                    interesting = []
                    for t in tasks:
                        if t.done():
                            continue
                        name = t.get_name()
                        # Grab a single top frame from each task
                        stack = t.get_stack(limit=1)
                        loc = "?"
                        if stack:
                            f = stack[0]
                            loc = (
                                f"{f.f_code.co_filename.split('/')[-1]}:"
                                f"{f.f_lineno} in {f.f_code.co_name}"
                            )
                        interesting.append(f"{name}@{loc}")
                    sample = interesting[:5]
                    suffix = f" [+{len(interesting) - 5} more]" if len(interesting) > 5 else ""
                    logger.warning(
                        "Loop-lag %dms stack sample: %s%s",
                        int(lag_ms),
                        " | ".join(sample),
                        suffix,
                    )
                except Exception:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("loop lag monitor crashed: %s", e)


# ── Silence watchdog ──────────────────────────────────────────

_SILENCE_WARN_THRESHOLD_S = 10.0


async def silence_watchdog_loop() -> None:
    """Emit a ``meeting_warning`` WS broadcast when audio ingestion stalls.

    The browser's audio WebSocket occasionally drops (NAT, tab throttle,
    CPU starvation) and never reconnects. Server-side state stays
    ``recording`` so the UI looks hung. This loop flips a one-shot
    warning once no audio chunk has arrived for
    ``_SILENCE_WARN_THRESHOLD_S`` seconds and resets it on the next
    chunk. Clients pick the event up off the regular ``/api/ws``
    channel and can render a banner / prompt a refresh.
    """
    POLL = 1.0
    while True:
        await asyncio.sleep(POLL)
        try:
            if not state.current_meeting or not state.last_audio_chunk_ts:
                state.silence_warn_sent = False
                continue
            # Stale-timestamp guard: ``last_audio_chunk_ts`` is bumped by
            # ws.audio_input on every inbound frame, including frames
            # that arrived BEFORE the current meeting started (e.g. a
            # prior meeting, or a stray admin-tab WS). Without this
            # check the watchdog would fire ``no_audio (Hs)`` the
            # instant a meeting starts, where H is the gap since the
            # last non-meeting frame. Anchor against the current
            # meeting's monotonic start time so we only count silence
            # accumulated WITHIN this meeting.
            meeting_start = state.metrics.meeting_start
            anchor = max(state.last_audio_chunk_ts, meeting_start)
            if not anchor:
                state.silence_warn_sent = False
                continue
            age = time.monotonic() - anchor
            if age >= _SILENCE_WARN_THRESHOLD_S and not state.silence_warn_sent:
                logger.warning(
                    "Silence watchdog: no audio in %.0fs for meeting %s — notifying clients",
                    age,
                    state.current_meeting.meeting_id,
                )
                await _broadcast_json(
                    {
                        "type": "meeting_warning",
                        "reason": "no_audio",
                        "age_s": round(age, 1),
                        "meeting_id": state.current_meeting.meeting_id,
                    }
                )
                state.silence_warn_sent = True
            elif age < _SILENCE_WARN_THRESHOLD_S and state.silence_warn_sent:
                logger.info(
                    "Silence watchdog: audio resumed after %.1fs for meeting %s",
                    age,
                    state.current_meeting.meeting_id,
                )
                await _broadcast_json(
                    {
                        "type": "meeting_warning_cleared",
                        "meeting_id": state.current_meeting.meeting_id,
                    }
                )
                state.silence_warn_sent = False
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("silence watchdog error: %s", e)


# ── Memory pressure canary (PSI) ──────────────────────────────
#
# ``/proc/pressure/memory`` exposes Pressure Stall Information for
# memory: the percentage of wall-clock time tasks were stalled
# waiting for memory in the last 10 / 60 / 300 seconds, plus a
# monotonic ``total`` microsecond counter.  Two lines:
#   ``some``: at least one task was stalled (early-warning signal).
#   ``full``: every runnable task was stalled (the kernel is one bad
#   allocation away from OOM-killing something).
#
# Thresholds chosen so a normal idle box stays silent (some.avg60
# floats around 0.4 today on the dev GB10) but a real
# vLLM-cold-load + smoke-test concurrency reliably trips them
# before swap exhausts.

PRESSURE_PATH = Path("/proc/pressure/memory")
_PRESSURE_TICK_S = 5.0
_PRESSURE_SOME_AVG10_WARN = 10.0  # %  >10% of last 10s stalled = warning
_PRESSURE_SOME_AVG10_CRIT = 25.0  # %  >25% = critical (host is hurting)
_PRESSURE_FULL_AVG10_CRIT = 5.0  # %  any sustained "full" stall = critical
# Hysteresis: don't re-spam at every tick. Re-warn only after the
# pressure level visibly worsened or after a 60s cooldown.
_PRESSURE_REWARN_S = 60.0


@dataclasses.dataclass
class MemPressureSnapshot:
    """One sample of /proc/pressure/memory."""

    some_avg10: float
    some_avg60: float
    some_avg300: float
    some_total_us: int
    full_avg10: float
    full_avg60: float
    full_avg300: float
    full_total_us: int

    def severity(self) -> str:
        """Map current sample to ``ok`` / ``warn`` / ``crit``."""
        if (
            self.some_avg10 >= _PRESSURE_SOME_AVG10_CRIT
            or self.full_avg10 >= _PRESSURE_FULL_AVG10_CRIT
        ):
            return "crit"
        if self.some_avg10 >= _PRESSURE_SOME_AVG10_WARN:
            return "warn"
        return "ok"


# Module-level cache of the latest snapshot. The /metrics endpoint
# reads this to expose Prometheus gauges without owning its own
# polling loop.
_LATEST_PRESSURE: MemPressureSnapshot | None = None


def parse_pressure_memory(text: str) -> MemPressureSnapshot | None:
    """Parse the two-line ``/proc/pressure/memory`` format.

    Returns ``None`` for malformed input rather than raising — pressure
    sampling is a soft signal and should never crash the monitor loop.
    """
    fields: dict[str, dict[str, float]] = {"some": {}, "full": {}}
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0] not in fields:
            continue
        head = parts[0]
        for kv in parts[1:]:
            if "=" not in kv:
                continue
            key, val = kv.split("=", 1)
            try:
                fields[head][key] = float(val)
            except ValueError:
                continue
    try:
        return MemPressureSnapshot(
            some_avg10=fields["some"]["avg10"],
            some_avg60=fields["some"]["avg60"],
            some_avg300=fields["some"]["avg300"],
            some_total_us=int(fields["some"]["total"]),
            full_avg10=fields["full"]["avg10"],
            full_avg60=fields["full"]["avg60"],
            full_avg300=fields["full"]["avg300"],
            full_total_us=int(fields["full"]["total"]),
        )
    except KeyError:
        return None


def read_pressure_snapshot() -> MemPressureSnapshot | None:
    """Read /proc/pressure/memory once. ``None`` if PSI unavailable."""
    try:
        text = PRESSURE_PATH.read_text()
    except OSError:
        return None
    return parse_pressure_memory(text)


def latest_pressure() -> MemPressureSnapshot | None:
    """Most recent successful sample, or ``None`` before the first
    poll completes / on hosts without PSI."""
    return _LATEST_PRESSURE


async def mem_pressure_monitor() -> None:
    """Sample /proc/pressure/memory every 5s, log + cache the result.

    Logs at ``warning`` when the ``some`` line crosses
    ``_PRESSURE_SOME_AVG10_WARN`` and at ``error`` for either
    ``some.avg10`` ≥ ``_PRESSURE_SOME_AVG10_CRIT`` or any sustained
    ``full`` stall. Hysteresis avoids re-spamming the same level — only
    a worsening sample or a 60s cooldown re-emits.

    On hosts without PSI (older kernels, restricted containers) the
    loop sleeps quietly forever rather than failing — the absence of
    /proc/pressure/memory is not itself an error.
    """
    global _LATEST_PRESSURE
    if not PRESSURE_PATH.exists():
        logger.info(
            "mem-pressure monitor: %s not available — skipping (PSI not enabled?)",
            PRESSURE_PATH,
        )
        return

    last_severity = "ok"
    last_log_ts = 0.0
    last_logged_some = 0.0
    while True:
        try:
            snap = read_pressure_snapshot()
            if snap is None:
                await asyncio.sleep(_PRESSURE_TICK_S)
                continue
            _LATEST_PRESSURE = snap

            sev = snap.severity()
            now = time.monotonic()
            # Emit on: severity escalation, return to ok from non-ok,
            # or a worse some.avg10 reading after the cooldown elapsed.
            should_log = sev != last_severity or (
                sev != "ok"
                and (
                    snap.some_avg10 > last_logged_some + 5.0
                    or now - last_log_ts >= _PRESSURE_REWARN_S
                )
            )

            if should_log:
                msg = (
                    f"mem-pressure {sev.upper()}: "
                    f"some avg10={snap.some_avg10:.1f}% avg60={snap.some_avg60:.1f}% "
                    f"full avg10={snap.full_avg10:.1f}% avg60={snap.full_avg60:.1f}%"
                )
                if sev == "crit":
                    logger.error(msg)
                elif sev == "warn":
                    logger.warning(msg)
                else:
                    logger.info("mem-pressure recovered: %s", msg)
                last_log_ts = now
                last_logged_some = snap.some_avg10
            last_severity = sev
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("mem-pressure monitor error: %s", e)
        await asyncio.sleep(_PRESSURE_TICK_S)
