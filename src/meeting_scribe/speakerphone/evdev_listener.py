"""Async evdev listener with short/long-press classification.

A single :class:`ButtonStateMachine` walks evdev key-down / key-up events
for one button and emits a ``"short"`` or ``"long"`` press depending
on how long the key was held. The threshold is read at construction
time from the mapping config so the daemon, UI copy, and tests all
share one value.

The daemon's :class:`evdev.InputDevice` reader loop calls
:meth:`feed_key_event` for every event on the telephony input device.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PressEvent:
    """Decoded short/long press emission from the state machine."""

    button: str
    kind: str  # "short" | "long"


class ButtonStateMachine:
    """Track one logical button's down→up timing and dispatch short/long.

    Two phases per press cycle:

    1. ``key_down`` arrives → record monotonic timestamp, schedule a
       long-press timer.
    2. Either ``key_up`` arrives before the timer fires (→ ``short``)
       or the timer fires while the key is still held (→ ``long``,
       and the eventual ``key_up`` is suppressed).

    Spurious double-downs (key-down while already held) are ignored —
    the kernel may emit auto-repeat or coalesced events for some
    drivers and we want exactly one emission per physical press.
    """

    def __init__(
        self,
        button: str,
        *,
        long_press_ms: int,
        emit: Callable[[PressEvent], Awaitable[None]],
    ) -> None:
        self._button = button
        self._threshold_s = long_press_ms / 1000.0
        self._emit = emit
        self._held = False
        self._long_timer: asyncio.TimerHandle | None = None
        self._long_fired = False

    async def feed_key_event(self, *, is_down: bool) -> None:
        """Process one key-down (``is_down=True``) or key-up event."""
        if is_down:
            await self._on_down()
        else:
            await self._on_up()

    async def _on_down(self) -> None:
        if self._held:
            # Auto-repeat or coalesced second down — ignore.
            return
        self._held = True
        self._long_fired = False
        loop = asyncio.get_event_loop()
        self._long_timer = loop.call_later(
            self._threshold_s,
            self._schedule_long_emit,
        )

    async def _on_up(self) -> None:
        if not self._held:
            # Up without a corresponding down (e.g. we missed the down
            # at startup). Defensive: ignore.
            return
        self._held = False
        if self._long_timer is not None:
            self._long_timer.cancel()
            self._long_timer = None
        if self._long_fired:
            # The timer already emitted "long"; suppress this up.
            return
        await self._emit(PressEvent(button=self._button, kind="short"))

    def _schedule_long_emit(self) -> None:
        """TimerHandle callback. Synchronous → wraps emit in a task.

        ``call_later`` is sync; we cannot ``await`` here, so schedule
        the async emit on the running loop.
        """
        if not self._held:
            # Released between timer expiry and callback dispatch.
            return
        self._long_fired = True
        loop = asyncio.get_event_loop()

        async def _emit_long() -> None:
            await self._emit(PressEvent(button=self._button, kind="long"))

        loop.create_task(_emit_long())


# ── Per-device evdev reader loop ────────────────────────────────────────


class TelephonyEvdevReader:
    """Read evdev events from one telephony input node into press emissions.

    Construction takes an evdev device object (real or stub) plus a
    ``button_for_keycode`` map that translates the kernel's KEY_*
    code into one of our abstract button names (``"phone"``, etc.).
    """

    def __init__(
        self,
        *,
        device: Any,
        button_for_keycode: dict[int, str],
        long_press_ms: int,
        emit: Callable[[PressEvent], Awaitable[None]],
    ) -> None:
        self._device = device
        self._key_to_button = button_for_keycode
        self._emit = emit
        self._machines: dict[str, ButtonStateMachine] = {}
        self._long_press_ms = long_press_ms

    def _machine(self, button: str) -> ButtonStateMachine:
        m = self._machines.get(button)
        if m is None:
            m = ButtonStateMachine(
                button,
                long_press_ms=self._long_press_ms,
                emit=self._emit,
            )
            self._machines[button] = m
        return m

    async def run(self) -> None:
        """Loop reading evdev events until the device is closed.

        evdev's ``async_read_loop`` yields one event at a time. We only
        care about ``EV_KEY`` (type 1) events; everything else is
        ignored. Cancellation is the normal exit path (the daemon
        cancels this task on unplug).
        """
        async for event in self._device.async_read_loop():
            if getattr(event, "type", None) != 1:  # EV_KEY
                continue
            button = self._key_to_button.get(event.code)
            if button is None:
                continue
            machine = self._machine(button)
            # evdev event values: 0=up, 1=down, 2=auto-repeat
            if event.value == 1:
                await machine.feed_key_event(is_down=True)
            elif event.value == 0:
                await machine.feed_key_event(is_down=False)
            # value=2 (auto-repeat) is ignored — we already detected
            # the down event and the long-press timer is running.


# ── Consumer-page observer (Vol+/Vol-/Mute) ─────────────────────────────

# Static map from kernel KEY_* names to the feedback label_id the daemon
# emits on key-down. Numeric codes are resolved at runtime via
# ``evdev.ecodes`` so we don't hardcode the values (they differ between
# kernels). KEY_MUTE is a TOGGLE — the daemon's callback queries
# PipeWire's default-sink mute state ~30 ms after the key event and
# emits ``system_muted`` or ``system_unmuted`` instead of this sentinel.
# Vol+/Vol- are directional, so the static label is correct as-is.
KERNEL_KEY_TO_CONSUMER_LABEL: dict[str, str] = {
    "KEY_VOLUMEUP": "volume_up",
    "KEY_VOLUMEDOWN": "volume_down",
    "KEY_MUTE": "system_mute_toggled",
}


class ConsumerObserver:
    """Non-grab evdev reader for /dev/input/event3 (consumer page).

    Crucially does NOT call ``EVIOCGRAB`` — observation is additive.
    The kernel's media-key agent still gets the same events and keeps
    handling system volume changes, so this observer is purely a
    notification source for the button-feedback path.

    Fires on **key-down only** (``event.value == 1``). Key-up
    (``value=0``) and key-repeat (``value=2``) are ignored — repeating
    the spoken feedback while a user holds a volume key would be
    intolerable; one announcement per physical press is the right
    cadence.

    KEY_MUTE emits a sentinel label_id (``system_mute_toggled``); the
    daemon's callback is responsible for resolving it to a state-aware
    label by querying PipeWire after the kernel has finished flipping
    the bit.
    """

    def __init__(
        self,
        *,
        device: Any,
        key_to_label: dict[int, str],
        emit: Callable[[str], Awaitable[None]],
    ) -> None:
        self._device = device
        self._key_to_label = key_to_label
        self._emit = emit

    async def run(self) -> None:
        # Visible startup log so operators can confirm the observer is
        # alive on the right device. Added 2026-05-13 after a session
        # where Vol+ presses landed on the kernel (event3) but the
        # observer produced zero log output and we couldn't tell
        # whether it was running, hung, or crashed.
        dev_path = getattr(self._device, "path", "<unknown>")
        logger.info(
            "ConsumerObserver: watching %s for keys=%s",
            dev_path,
            sorted(self._key_to_label.values()),
        )
        async for event in self._device.async_read_loop():
            if getattr(event, "type", None) != 1:  # EV_KEY only
                continue
            if event.value != 1:  # down only — skip up + repeat
                continue
            label = self._key_to_label.get(event.code)
            if label is None:
                # Log unmapped keys at DEBUG so the journal isn't noisy
                # for every "Mic Mute" / "Phone hook" press that lands
                # on this same node — those are handled elsewhere.
                logger.debug(
                    "ConsumerObserver: unmapped key code=%s on %s",
                    event.code,
                    dev_path,
                )
                continue
            # Visible per-press log — operators need this to confirm
            # button → daemon round-trip. Drop to DEBUG if it becomes
            # noisy in practice, but the cost is one INFO line per
            # physical press which is negligible.
            logger.info("ConsumerObserver: %s → emit %s", event.code, label)
            try:
                await self._emit(label)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "ConsumerObserver: feedback emit for %s raised",
                    label,
                )


def resolve_consumer_key_to_label_map() -> dict[int, str]:
    """Translate KERNEL_KEY_TO_CONSUMER_LABEL into numeric kernel codes.

    Returns an empty dict if evdev isn't importable so a CI environment
    without kernel headers can still load the module.
    """
    try:
        import evdev.ecodes as ec
    except Exception:
        return {}

    out: dict[int, str] = {}
    for name, label in KERNEL_KEY_TO_CONSUMER_LABEL.items():
        code = getattr(ec, name, None)
        if code is not None:
            out[int(code)] = label
    return out


__all__ = [
    "KERNEL_KEY_TO_CONSUMER_LABEL",
    "ButtonStateMachine",
    "ConsumerObserver",
    "PressEvent",
    "TelephonyEvdevReader",
    "resolve_consumer_key_to_label_map",
]
