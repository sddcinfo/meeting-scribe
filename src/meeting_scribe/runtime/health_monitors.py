"""Long-lived health monitors that run for the whole process lifetime.

Two background tasks owned by ``server.lifespan`` (alongside the
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

Pulled out of ``server.py`` so the lifespan orchestrator stays a thin
shell.
"""

from __future__ import annotations

import asyncio
import logging
import time

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
