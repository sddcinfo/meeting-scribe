"""Tests for audio/bt_bridge.py — state machine primitives, listener
contract, tracked-node-ids semantics, scan gating, single-flight
recovery, retry-timer cancellation, resume target state.
"""

from __future__ import annotations

import asyncio

import pytest

from meeting_scribe.audio.bt_bridge import (
    BTSpeakerListener,
    BridgeData,
    BridgeState,
    begin_scan,
    can_scan,
    cancel_retry_timer,
    end_scan,
    is_managed_node_removed,
    request_recovery,
    resume_target_state,
    schedule_retry,
    wait_for_pipewire,
)


# ── BTSpeakerListener ────────────────────────────────────────────


def test_listener_format_bound_at_construction() -> None:
    """Plan §B.6: listener format is wav-pcm at registration —
    NEVER negotiated."""
    listener = BTSpeakerListener(request_recovery=lambda: None)
    assert listener.audio_format == "wav-pcm"


def test_listener_drops_non_riff_payload() -> None:
    """Defensive: if fan-out ever routed an fmp4 fragment here, the
    listener drops it silently rather than corrupting downstream
    state."""
    listener = BTSpeakerListener(request_recovery=lambda: None)

    async def scenario() -> None:
        await listener.send_bytes(b"NOT-A-RIFF-CHUNK")

    asyncio.run(scenario())
    assert listener.drops == 1


def test_listener_overflow_requests_recovery() -> None:
    """Filling the queue past max_queue triggers request_recovery."""
    calls = [0]

    def signal() -> None:
        calls[0] += 1

    listener = BTSpeakerListener(request_recovery=signal, max_queue=2)

    async def scenario() -> None:
        for _ in range(5):
            await listener.send_bytes(b"RIFF" + b"\x00" * 16)

    asyncio.run(scenario())
    assert calls[0] >= 1
    assert listener.drops >= 3


# ── tracked_node_ids semantics ───────────────────────────────────


def test_is_managed_node_removed_only_for_tracked_ids() -> None:
    bridge = BridgeData()
    bridge.tracked_node_ids = {41, 42}
    assert is_managed_node_removed(bridge, removed_node_id=41) is True
    assert is_managed_node_removed(bridge, removed_node_id=99) is False


def test_request_recovery_sets_flag_and_wakes_event() -> None:
    bridge = BridgeData()

    async def scenario() -> None:
        request_recovery(bridge)
        assert bridge.recovery_pending is True
        # Event is set → a loop awaiting it returns immediately.
        await asyncio.wait_for(bridge.sm_wakeup_event.wait(), timeout=0.1)

    asyncio.run(scenario())


def test_request_recovery_idempotent() -> None:
    """Plan §B.5: concurrent listener writes collapse to one recovery
    attempt — set-True is the only op."""
    bridge = BridgeData()
    request_recovery(bridge)
    request_recovery(bridge)
    request_recovery(bridge)
    assert bridge.recovery_pending is True


# ── Retry timer cancellation ─────────────────────────────────────


def test_schedule_retry_then_cancel_clears_handle() -> None:
    bridge = BridgeData()

    async def scenario() -> None:
        schedule_retry(bridge, delay=10.0)
        assert bridge.retry_timer is not None
        cancel_retry_timer(bridge)
        assert bridge.retry_timer is None

    asyncio.run(scenario())


def test_schedule_retry_replaces_prior_timer() -> None:
    """Plan §B.4: every state transition cancels any pending retry
    first. Calling schedule_retry while one is already pending
    replaces it."""
    bridge = BridgeData()

    async def scenario() -> None:
        schedule_retry(bridge, delay=20.0)
        first = bridge.retry_timer
        schedule_retry(bridge, delay=20.0)
        second = bridge.retry_timer
        assert first is not second
        cancel_retry_timer(bridge)

    asyncio.run(scenario())


def test_retry_fires_through_state_machine_queue() -> None:
    """Plan §B.4: retry runs through the same queue as every other
    transition — never as a free-floating task."""
    bridge = BridgeData()

    async def scenario() -> None:
        schedule_retry(bridge, delay=0.01)
        await asyncio.sleep(0.05)
        # The fire callback enqueued ``("retry_connect", None)``.
        item = await asyncio.wait_for(bridge.sm_event_queue.get(), timeout=0.5)
        assert item == ("retry_connect", None)

    asyncio.run(scenario())


# ── Resume target state ──────────────────────────────────────────


def test_resume_target_when_mic_was_active() -> None:
    """Plan §B.4 / R29: link-loss during MicLive auto-recovers to
    MicLive on reconnect, not Idle."""
    assert resume_target_state(bt_input_active=True) is BridgeState.MIC_LIVE


def test_resume_target_when_mic_was_inactive() -> None:
    assert resume_target_state(bt_input_active=False) is BridgeState.IDLE


# ── Scan gating ─────────────────────────────────────────────────


def test_can_scan_blocked_during_active_session() -> None:
    bridge = BridgeData(state=BridgeState.IDLE)
    ok, reason = can_scan(bridge)
    assert not ok
    assert reason == "scan_blocked_by_active_session"

    bridge.state = BridgeState.MIC_LIVE
    ok, reason = can_scan(bridge)
    assert not ok
    assert reason == "scan_blocked_by_active_session"


def test_can_scan_blocked_during_transition() -> None:
    bridge = BridgeData(state=BridgeState.SWITCHING)
    ok, reason = can_scan(bridge)
    assert not ok
    assert reason == "scan_blocked_by_transition"


def test_can_scan_allowed_when_disconnected() -> None:
    bridge = BridgeData(state=BridgeState.DISCONNECTED)
    ok, reason = can_scan(bridge)
    assert ok
    assert reason is None


def test_begin_scan_cancels_retry_and_sets_suppression() -> None:
    bridge = BridgeData(state=BridgeState.DISCONNECTED)

    async def scenario() -> None:
        schedule_retry(bridge, delay=10.0)
        assert bridge.retry_timer is not None
        begin_scan(bridge)
        assert bridge.retry_timer is None
        assert bridge.retry_suppressed is True
        assert bridge.state is BridgeState.SCANNING

    asyncio.run(scenario())


def test_end_scan_clears_suppression_and_returns_to_disconnected() -> None:
    bridge = BridgeData(state=BridgeState.DISCONNECTED)

    async def scenario() -> None:
        begin_scan(bridge)
        end_scan(bridge)
        assert bridge.retry_suppressed is False
        assert bridge.state is BridgeState.DISCONNECTED

    asyncio.run(scenario())


# ── PipeWire readiness gate ──────────────────────────────────────


def test_wait_for_pipewire_returns_immediately_without_probe() -> None:
    """Tests / dev runs leave the probe unset and the gate no-ops."""

    async def scenario() -> bool:
        return await wait_for_pipewire(timeout=0.5)

    assert asyncio.run(scenario()) is True


def test_wait_for_pipewire_times_out_when_probe_never_succeeds() -> None:
    async def probe() -> bool:
        return False

    async def scenario() -> bool:
        return await wait_for_pipewire(timeout=0.1, interval=0.02, probe=probe)

    assert asyncio.run(scenario()) is False


def test_wait_for_pipewire_returns_true_on_first_success() -> None:
    counter = [0]

    async def probe() -> bool:
        counter[0] += 1
        return counter[0] >= 3

    async def scenario() -> bool:
        return await wait_for_pipewire(timeout=2.0, interval=0.01, probe=probe)

    assert asyncio.run(scenario()) is True
    assert counter[0] >= 3


def test_wait_for_pipewire_swallows_probe_exceptions() -> None:
    """Transient probe failures (e.g. pactl crash) keep retrying
    until timeout rather than raising."""

    state = {"calls": 0}

    async def probe() -> bool:
        state["calls"] += 1
        if state["calls"] < 3:
            raise RuntimeError("transient")
        return True

    async def scenario() -> bool:
        return await wait_for_pipewire(timeout=2.0, interval=0.01, probe=probe)

    assert asyncio.run(scenario()) is True
