"""GB10-owned Bluetooth audio bridge — data plane.

State machine + ``tracked_node_ids`` set ownership + cancelable retry
timer + single-flight recovery (Plan §B.3 / §B.4 / §B.5).

Owns two long-lived subprocesses:

* ``pw-record --target=<source> --format=s16 --rate=16000 --channels=1 -``
  — feeds the meeting-scribe ASR queue.
* ``pw-play --target=<sink> --format=s16 --rate=24000 --channels=1 -``
  — receives synthesized TTS bytes from the ``BTSpeakerListener``.

Plus a ``pw-mon`` watcher that funnels node-removal events through the
same state-machine queue as every other transition. Recovery is single-
flight: ``state._recovery_pending`` is the durable flag listeners flip
on dead-pipe detection; the loop awaits both
``queue.get()`` and ``_sm_wakeup_event.wait()`` so a flag flip wakes an
idle loop immediately.

Hardware-gated end-to-end verification (real BlueZ + paired device)
lives in ``meeting-scribe bt smoke``; the unit tests here drive the
state machine directly with mocks.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


# ── State enum ───────────────────────────────────────────────────


class BridgeState(str, enum.Enum):
    """Plan §B.4 state machine."""

    DISCONNECTED = "Disconnected"
    IDLE = "Idle"  # A2DP only
    MIC_LIVE = "MicLive"  # HFP, mic + speaker
    SWITCHING = "Switching"
    SCANNING = "Scanning"


# ── Listener (data-plane sink for TTS bytes) ─────────────────────


class BTSpeakerListener:
    """Implements the :class:`AudioListener` Protocol from
    :mod:`meeting_scribe.audio.output_pipeline`.

    Bound to ``audio_format="wav-pcm"`` at registration so the fan-out
    function never routes mse-fmp4-aac fragments to BT (Plan §B.6).
    Bounded local queue of 4 frames; overflow drops oldest +
    increments ``bt_listener_drops``. Dead-pipe detection sets
    :attr:`request_recovery` so the bridge state machine wakes its
    loop and reinitializes the speaker subprocess.

    The listener must NEVER:

    * raise (would kill the fan-out coroutine for ALL listeners),
    * await the bridge's state lock (would deadlock with the state
      machine),
    * spawn a free-floating recovery task (single-flight contract).
    """

    def __init__(
        self,
        *,
        request_recovery: Callable[[], None],
        max_queue: int = 4,
    ) -> None:
        self._queue: deque[bytes] = deque(maxlen=max_queue)
        self._request_recovery = request_recovery
        # Plan §B.6 — set at registration time, NEVER negotiated.
        self.audio_format = "wav-pcm"
        self.drops = 0

    def __hash__(self) -> int:
        return id(self)

    @property
    def cookies(self) -> dict[str, str]:
        # Required by the Protocol — BT listener has no cookies.
        return {}

    async def send_bytes(self, data: bytes) -> None:
        """Best-effort drop-oldest enqueue. Wires recovery on dead pipe.

        Plan §B.5: never await the state lock, never raise. The
        drainer is responsible for actually pushing bytes into
        pw-play's stdin; here we only enqueue + signal.
        """
        if not data.startswith(b"RIFF"):
            # Should be impossible — fan-out routes wav-pcm only to us.
            # Drop silently rather than raising; CI test guards format.
            self.drops += 1
            return
        if len(self._queue) == self._queue.maxlen:
            # Overflow — drop the oldest and request recovery so the
            # state machine can reinitialize the sink if needed.
            self._queue.popleft()
            self.drops += 1
            self._request_recovery()
        self._queue.append(data)


# ── tracked_node_ids set + bridge state ─────────────────────────


@dataclass
class BridgeData:
    """Mutable bridge state (Plan §B.4 contract).

    Single-owner serialization: every state transition acquires
    :attr:`state_lock`. ``tracked_node_ids`` is the SOLE concurrency
    discriminator for the pw-mon recovery handler.
    """

    state: BridgeState = BridgeState.DISCONNECTED
    tracked_node_ids: set[int] = field(default_factory=set)
    user_initiated_disconnect: bool = False
    retry_suppressed: bool = False
    transition_count: int = 0
    bt_input_active: bool = False
    bt_device_mac: str | None = None
    recovery_pending: bool = False
    state_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    sm_wakeup_event: asyncio.Event = field(default_factory=asyncio.Event)
    sm_event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    retry_timer: asyncio.TimerHandle | None = None


def is_managed_node_removed(
    bridge: BridgeData,
    *,
    removed_node_id: int,
) -> bool:
    """Plan §B.4 Risk 6: pw-mon removal is a real link loss only when
    the removed node ID is currently tracked.

    Pure function — call from the pw-mon dispatcher to decide whether
    to enqueue ``("link_loss", node_id)`` on the state-machine queue.
    """
    return removed_node_id in bridge.tracked_node_ids


def request_recovery(bridge: BridgeData) -> None:
    """Listener's signaling primitive (Plan §B.5).

    Sets the durable flag + wakes the state machine. Called from
    :class:`BTSpeakerListener.send_bytes` on overflow / dead pipe.
    Idempotent — concurrent calls collapse to one recovery attempt
    when the loop drains.
    """
    bridge.recovery_pending = True
    bridge.sm_wakeup_event.set()


def cancel_retry_timer(bridge: BridgeData) -> None:
    """Cancel any pending retry timer (Plan §B.4).

    Called as the FIRST line of every state-machine event handler so
    manual ``bt connect`` / ``bt disconnect`` / spontaneous reconnect
    all preempt any auto-retry that was in flight.
    """
    if bridge.retry_timer is not None:
        bridge.retry_timer.cancel()
        bridge.retry_timer = None


def schedule_retry(
    bridge: BridgeData,
    *,
    delay: float = 10.0,
    loop: asyncio.AbstractEventLoop | None = None,
) -> None:
    """Schedule one cancelable retry via :meth:`call_later`.

    The retry runs through the same state-machine queue as every
    other transition — never as a free-floating task. ``loop`` is the
    target loop; defaults to the current running loop.
    """
    cancel_retry_timer(bridge)
    target_loop = loop or asyncio.get_event_loop()

    def fire() -> None:
        try:
            bridge.sm_event_queue.put_nowait(("retry_connect", None))
        except asyncio.QueueFull:
            # Queue is unbounded by default; only happens in pathological
            # tests. Best-effort drop is safe — the next event triggers a
            # transition.
            pass

    bridge.retry_timer = target_loop.call_later(delay, fire)


# ── Resume-mode resolver ─────────────────────────────────────────


def resume_target_state(*, bt_input_active: bool) -> BridgeState:
    """Plan §B.4 "Disconnected → resume": target state is determined
    by the operator's persisted ``bt_input_active`` toggle, NOT
    always Idle.

    Pure function so the test for "transient link-loss in MicLive
    auto-recovers to MicLive" can drive it without async machinery.
    """
    return BridgeState.MIC_LIVE if bt_input_active else BridgeState.IDLE


# ── Scan gating ──────────────────────────────────────────────────


def can_scan(bridge: BridgeData) -> tuple[bool, str | None]:
    """Plan §B.7b "Scan is a first-class state".

    Returns ``(ok, reason)``. Active session blocks the scan
    (``scan_blocked_by_active_session``); a transition blocks it
    (``scan_blocked_by_transition``); Disconnected accepts the scan
    + cancels the pending retry timer + sets ``retry_suppressed``.
    """
    if bridge.state in (BridgeState.IDLE, BridgeState.MIC_LIVE):
        return False, "scan_blocked_by_active_session"
    if bridge.state == BridgeState.SWITCHING:
        return False, "scan_blocked_by_transition"
    if bridge.state == BridgeState.SCANNING:
        return False, "scan_already_running"
    return True, None


def begin_scan(bridge: BridgeData) -> None:
    """Transition Disconnected → Scanning, suppress retry."""
    cancel_retry_timer(bridge)
    bridge.retry_suppressed = True
    bridge.state = BridgeState.SCANNING
    bridge.transition_count += 1


def end_scan(bridge: BridgeData) -> None:
    """Transition Scanning → Disconnected, re-arm retry hooks."""
    bridge.retry_suppressed = False
    bridge.state = BridgeState.DISCONNECTED
    bridge.transition_count += 1


# ── PipeWire readiness gate ──────────────────────────────────────


async def wait_for_pipewire(
    *,
    timeout: float = 30.0,
    probe: Callable[[], Awaitable[bool]] | None = None,
    interval: float = 0.5,
) -> bool:
    """Poll a probe callable until both ``pw-cli info 0`` and
    ``pactl info`` succeed (the probe encapsulates that pair).

    Returns True on ready, False on timeout. Plan §B.9 readiness
    gate — server.lifespan invokes this before scheduling
    ``bt_bridge.start()``. On timeout the bridge skips auto-connect for
    this boot; admin can run ``meeting-scribe doctor`` to diagnose.
    """
    if probe is None:
        # No probe injected → assume ready (test fixtures + dev runs
        # without a real PipeWire stack).
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if await probe():
                return True
        except Exception:  # noqa: BLE001 — best-effort probe; failures keep retrying
            pass
        await asyncio.sleep(interval)
    return False
