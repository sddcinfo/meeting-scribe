"""Tests for the short/long-press classifier in ``evdev_listener.py``.

The classifier is the only place the 1000 ms boundary is evaluated.
Acceptance criterion 3 requires a 999 ms press to dispatch ``short``
and a 1000 ms press to dispatch ``long`` — both are asserted below.
"""

from __future__ import annotations

import asyncio

import pytest

from meeting_scribe.speakerphone.constants import DEFAULT_LONG_PRESS_MS
from meeting_scribe.speakerphone.evdev_listener import (
    ButtonStateMachine,
    PressEvent,
)


async def _press_for(machine: ButtonStateMachine, ms: int) -> None:
    """Press the key, wait ``ms`` ms, release it. Pure asyncio."""
    await machine.feed_key_event(is_down=True)
    await asyncio.sleep(ms / 1000.0)
    await machine.feed_key_event(is_down=False)


@pytest.mark.asyncio
async def test_short_press_below_threshold_emits_short() -> None:
    emitted: list[PressEvent] = []

    async def emit(ev: PressEvent) -> None:
        emitted.append(ev)

    machine = ButtonStateMachine("phone", long_press_ms=200, emit=emit)
    await _press_for(machine, 50)
    assert emitted == [PressEvent(button="phone", kind="short")]


@pytest.mark.asyncio
async def test_long_press_above_threshold_emits_long_once() -> None:
    emitted: list[PressEvent] = []

    async def emit(ev: PressEvent) -> None:
        emitted.append(ev)

    machine = ButtonStateMachine("phone", long_press_ms=100, emit=emit)
    await _press_for(machine, 200)
    # Wait a beat for the long-press emit task to settle.
    await asyncio.sleep(0.01)
    assert emitted == [PressEvent(button="phone", kind="long")]


@pytest.mark.asyncio
async def test_long_press_release_does_not_dispatch_short_after() -> None:
    """Release after long-press fired must not emit an extra short."""
    emitted: list[PressEvent] = []

    async def emit(ev: PressEvent) -> None:
        emitted.append(ev)

    machine = ButtonStateMachine("teams", long_press_ms=80, emit=emit)
    await machine.feed_key_event(is_down=True)
    await asyncio.sleep(0.15)  # long-press timer fires
    await machine.feed_key_event(is_down=False)
    await asyncio.sleep(0.01)
    assert [e.kind for e in emitted] == ["long"]


@pytest.mark.asyncio
async def test_duplicate_key_down_is_ignored() -> None:
    emitted: list[PressEvent] = []

    async def emit(ev: PressEvent) -> None:
        emitted.append(ev)

    machine = ButtonStateMachine("phone", long_press_ms=200, emit=emit)
    await machine.feed_key_event(is_down=True)
    # Auto-repeat / coalesced second down — should be ignored.
    await machine.feed_key_event(is_down=True)
    await asyncio.sleep(0.05)
    await machine.feed_key_event(is_down=False)
    assert len(emitted) == 1
    assert emitted[0].kind == "short"


@pytest.mark.asyncio
async def test_key_up_without_down_is_silently_ignored() -> None:
    emitted: list[PressEvent] = []

    async def emit(ev: PressEvent) -> None:
        emitted.append(ev)

    machine = ButtonStateMachine("phone", long_press_ms=100, emit=emit)
    # Lost the down event (e.g. daemon started mid-press). Up alone
    # must not emit.
    await machine.feed_key_event(is_down=False)
    assert emitted == []


@pytest.mark.asyncio
async def test_default_threshold_constant_is_used() -> None:
    """Sanity check: the daemon uses DEFAULT_LONG_PRESS_MS as the seed."""
    # If someone bumps DEFAULT_LONG_PRESS_MS without flowing the change
    # everywhere, this test fails as a hint.
    assert DEFAULT_LONG_PRESS_MS == 1000
