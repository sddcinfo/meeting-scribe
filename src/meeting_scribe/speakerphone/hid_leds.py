"""LED output-report assembly + a pure-Python pattern player.

The Dell SP325 carries its Mute LED on Report ID 5 (LED page 0x08, bit 1
in the output byte). To drive the ring we write a 2-byte report to
``/dev/hidraw*``: ``[0x05, led_byte]``.

The pattern player (``PatternRunner``) walks a list of ``(on_ms, off_ms)``
tuples and asks the caller-supplied writer to flip the bit. It is sync
and side-effect-free except for the writer callback; the daemon wraps
it in an asyncio task that re-reads the chosen pattern every state-machine
tick.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from meeting_scribe.speakerphone.constants import LED_PATTERNS
from meeting_scribe.speakerphone.descriptor import (
    LedBit,
    encode_led_output_report,
)

# Report ID 5 is the telephony+LED report on the SP325.
REPORT_ID_TELEPHONY_LED = 0x05


def build_led_report(states: dict[str, bool]) -> bytes:
    """Render the 2-byte hidraw write payload for the given LED state.

    Format: ``[report_id, led_byte]``. ``led_byte`` is built by
    :func:`descriptor.encode_led_output_report`.
    """
    return bytes((REPORT_ID_TELEPHONY_LED,)) + encode_led_output_report(states)


def mute_ring(on: bool) -> bytes:
    """Convenience: the report bytes for "Mute LED on" or "off"."""
    return build_led_report({"mute_led": on})


# ── Pattern player ──────────────────────────────────────────────────────


PatternSchedule = Sequence[tuple[int, int]]
"""Sequence of (on_ms, off_ms) pairs that loops."""


@dataclass(frozen=True)
class PatternStep:
    """One step in the LED-pattern timeline."""

    on: bool
    duration_ms: int


def expand_pattern(schedule: PatternSchedule) -> tuple[PatternStep, ...]:
    """Flatten an (on, off, on, off, …) schedule into discrete steps.

    A pair ``(on_ms, off_ms)`` becomes one ``on`` step then one ``off``
    step. A zero-length step is dropped (so ``(0, 1000)`` is just an
    off step, and ``(1000, 0)`` is just an on step — solid).
    """
    steps: list[PatternStep] = []
    for pair in schedule:
        if len(pair) % 2 != 0:
            raise ValueError(
                f"pattern step {pair!r} must have an even number of slots",
            )
        on = True
        for slot in pair:
            if slot > 0:
                steps.append(PatternStep(on=on, duration_ms=int(slot)))
            on = not on
    if not steps:
        # Empty schedule → permanently off.
        steps = [PatternStep(on=False, duration_ms=1000)]
    return tuple(steps)


def iter_pattern_forever(schedule: PatternSchedule) -> Iterator[PatternStep]:
    """Loop through the schedule's steps indefinitely.

    Caller is responsible for breaking out (e.g. when the active LED
    state changes). Used by the daemon as the inner loop of its LED
    task.
    """
    steps = expand_pattern(schedule)
    while True:
        yield from steps


def resolve_pattern(name: str) -> PatternSchedule:
    """Look up a canonical pattern name. Falls back to ``off`` if unknown.

    Unknown names log warnings via the caller — this function stays
    pure so it's safe to use from validation and tests.
    """
    return LED_PATTERNS.get(name, LED_PATTERNS["off"])


class PatternRunner:
    """Drive an LED through a pattern using a synchronous writer.

    The runner is single-step: call :meth:`tick` and it returns the
    delay (in seconds) until the next tick along with whether the LED
    should be on or off right now. The daemon's asyncio loop is
    responsible for sleeping that long and writing the appropriate
    report bytes via ``write_cb``.

    Why pull-style instead of running its own thread/loop: the daemon
    re-reads the active LED state every state-machine tick, so the
    pattern needs to be cancel-and-restart cheap. A simple iterator
    keeps that contract obvious.
    """

    def __init__(
        self,
        schedule: PatternSchedule,
        write_cb: Callable[[bool], None],
    ) -> None:
        self._iter = iter_pattern_forever(schedule)
        self._write_cb = write_cb
        self._last_state: bool | None = None

    def tick(self) -> float:
        """Advance one step. Returns seconds to sleep before the next tick.

        Calls the writer only when the desired on/off state actually
        changes; deduplicates writes against the previous tick to keep
        hidraw traffic low.
        """
        step = next(self._iter)
        if step.on != self._last_state:
            self._write_cb(step.on)
            self._last_state = step.on
        return step.duration_ms / 1000.0

    def write_off(self) -> None:
        """Force the LED off and reset the dedup-cache.

        Used when the daemon is shutting down or switching devices.
        """
        self._write_cb(False)
        self._last_state = False


def sleep_ms(ms: int) -> None:
    """Block for ``ms`` milliseconds. Helper for tests + manual driving."""
    time.sleep(ms / 1000.0)


# Re-export so callers don't need to dig into descriptor.py to get the
# LED-bit enum.
__all__ = [
    "REPORT_ID_TELEPHONY_LED",
    "LedBit",
    "PatternRunner",
    "PatternSchedule",
    "PatternStep",
    "build_led_report",
    "expand_pattern",
    "iter_pattern_forever",
    "mute_ring",
    "resolve_pattern",
    "sleep_ms",
]
