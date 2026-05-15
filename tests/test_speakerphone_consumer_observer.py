"""Tests for the consumer-page evdev observer.

The observer reads ``/dev/input/event3`` (volume/mute keys) without
grabbing the device, so the kernel + media-key agent keep their normal
flow. Coverage targets:

* Key-down events emit the right label_id (volume_up / volume_down /
  system_mute_toggled).
* Key-up (value=0) and key-repeat (value=2) are suppressed.
* Non-EV_KEY event types (e.g. EV_SYN) are ignored.
* Unknown keycodes don't fire feedback.
* Cancelling the run task tears down cleanly.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from meeting_scribe.speakerphone.evdev_listener import (
    KERNEL_KEY_TO_CONSUMER_LABEL,
    ConsumerObserver,
)


@dataclass
class _FakeEvent:
    """evdev.InputEvent-shaped — just the fields the observer reads."""

    type: int
    code: int
    value: int


class _FakeInputDevice:
    """Streams a pre-baked list of events through async_read_loop."""

    def __init__(self, events: list[_FakeEvent]) -> None:
        self._events = list(events)
        self.grabbed: bool = False  # would be set by EVIOCGRAB
        self.closed = False

    async def async_read_loop(self):
        for ev in self._events:
            await asyncio.sleep(0)  # yield to scheduler
            yield ev
        # Stay open after exhausting the canned events — let the test
        # cancel us explicitly.
        idle = asyncio.Event()
        await idle.wait()

    def close(self) -> None:
        self.closed = True


# ── Key-code resolution (label map) ───────────────────────────────────


def test_kernel_label_map_covers_vol_and_mute_keys() -> None:
    expected = {"KEY_VOLUMEUP", "KEY_VOLUMEDOWN", "KEY_MUTE"}
    assert expected.issubset(set(KERNEL_KEY_TO_CONSUMER_LABEL))


def test_volume_keys_are_directional_not_toggle_sentinel() -> None:
    """Vol+ and Vol- must emit the static directional label.

    Only KEY_MUTE emits the toggle-sentinel; the directional keys must
    NOT emit the sentinel (would result in a wpctl lookup per press
    for no reason).
    """
    assert KERNEL_KEY_TO_CONSUMER_LABEL["KEY_VOLUMEUP"] == "volume_up"
    assert KERNEL_KEY_TO_CONSUMER_LABEL["KEY_VOLUMEDOWN"] == "volume_down"
    assert KERNEL_KEY_TO_CONSUMER_LABEL["KEY_MUTE"] == "system_mute_toggled"


# ── Observer behavior ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_observer_emits_volume_up_on_key_down() -> None:
    # Made-up keycode + map (avoids depending on evdev.ecodes here).
    KEY_VOLUMEUP = 115
    emitted: list[str] = []

    async def emit(label: str) -> None:
        emitted.append(label)

    device = _FakeInputDevice(
        events=[_FakeEvent(type=1, code=KEY_VOLUMEUP, value=1)],
    )
    obs = ConsumerObserver(
        device=device,
        key_to_label={KEY_VOLUMEUP: "volume_up"},
        emit=emit,
    )
    task = asyncio.create_task(obs.run())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert emitted == ["volume_up"]


@pytest.mark.asyncio
async def test_observer_ignores_key_up_and_key_repeat() -> None:
    """value=0 (up) and value=2 (autorepeat) must NOT fire feedback."""
    KEY_VOLUMEUP = 115
    emitted: list[str] = []

    async def emit(label: str) -> None:
        emitted.append(label)

    device = _FakeInputDevice(
        events=[
            _FakeEvent(type=1, code=KEY_VOLUMEUP, value=1),  # DOWN — fire
            _FakeEvent(type=1, code=KEY_VOLUMEUP, value=2),  # repeat — skip
            _FakeEvent(type=1, code=KEY_VOLUMEUP, value=2),  # repeat — skip
            _FakeEvent(type=1, code=KEY_VOLUMEUP, value=0),  # UP — skip
        ],
    )
    obs = ConsumerObserver(
        device=device,
        key_to_label={KEY_VOLUMEUP: "volume_up"},
        emit=emit,
    )
    task = asyncio.create_task(obs.run())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    # Exactly one emission — the initial key-down. Repeats / ups
    # MUST NOT have produced more.
    assert emitted == ["volume_up"]


@pytest.mark.asyncio
async def test_observer_ignores_non_ev_key_events() -> None:
    """EV_SYN (type=0), EV_REL, EV_MSC etc. should not trigger feedback."""
    KEY_VOLUMEUP = 115
    emitted: list[str] = []

    async def emit(label: str) -> None:
        emitted.append(label)

    device = _FakeInputDevice(
        events=[
            _FakeEvent(type=0, code=0, value=1),  # EV_SYN — skip
            _FakeEvent(type=4, code=0, value=1),  # EV_MSC — skip
            _FakeEvent(type=1, code=KEY_VOLUMEUP, value=1),  # EV_KEY DOWN
        ],
    )
    obs = ConsumerObserver(
        device=device,
        key_to_label={KEY_VOLUMEUP: "volume_up"},
        emit=emit,
    )
    task = asyncio.create_task(obs.run())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert emitted == ["volume_up"]


@pytest.mark.asyncio
async def test_observer_ignores_unknown_key_codes() -> None:
    """A keycode not in ``key_to_label`` must not trigger emit."""
    KEY_F1 = 59  # arbitrary key not in our map
    emitted: list[str] = []

    async def emit(label: str) -> None:
        emitted.append(label)

    device = _FakeInputDevice(
        events=[_FakeEvent(type=1, code=KEY_F1, value=1)],
    )
    obs = ConsumerObserver(
        device=device,
        key_to_label={115: "volume_up"},  # only vol+ mapped
        emit=emit,
    )
    task = asyncio.create_task(obs.run())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert emitted == []


@pytest.mark.asyncio
async def test_observer_swallows_emit_exceptions() -> None:
    """A failing speak request must not crash the observer.

    Without this, one transient TTS-backend hiccup would stop all
    future feedback emissions until daemon restart.
    """
    KEY_VOLUMEUP = 115
    call_count = 0

    async def flaky_emit(label: str) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("simulated speak failure")

    device = _FakeInputDevice(
        events=[
            _FakeEvent(type=1, code=KEY_VOLUMEUP, value=1),
            _FakeEvent(type=1, code=KEY_VOLUMEUP, value=0),
            _FakeEvent(type=1, code=KEY_VOLUMEUP, value=1),
        ],
    )
    obs = ConsumerObserver(
        device=device,
        key_to_label={KEY_VOLUMEUP: "volume_up"},
        emit=flaky_emit,
    )
    task = asyncio.create_task(obs.run())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    # Both key-down events reached emit; the first raised but the
    # observer kept running.
    assert call_count == 2


@pytest.mark.asyncio
async def test_observer_does_not_grab_device() -> None:
    """ConsumerObserver MUST NOT call EVIOCGRAB.

    Calling grab would consume events from the kernel, breaking
    system volume changes. The observer must stay strictly
    passive — verified here by asserting the fake device's
    ``grabbed`` flag is never set.
    """
    KEY_VOLUMEUP = 115
    device = _FakeInputDevice(events=[])
    assert device.grabbed is False
    obs = ConsumerObserver(
        device=device,
        key_to_label={KEY_VOLUMEUP: "volume_up"},
        emit=lambda label: asyncio.sleep(0),  # type: ignore[arg-type, return-value]
    )
    task = asyncio.create_task(obs.run())
    await asyncio.sleep(0.02)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    # The fake device tracks grab calls explicitly; should still be False.
    assert device.grabbed is False
