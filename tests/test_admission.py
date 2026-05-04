"""Tests for server_support/admission.py — guest WS caps, drop-tracker,
auth rate limiter, gated_auth helper.
"""

from __future__ import annotations

import asyncio

import pytest

from meeting_scribe.server_support.admission import (
    AuthInFlight,
    AuthRateLimiter,
    GuestCaps,
    TranscriptDropTracker,
    WSAdmission,
    constant_time_wait,
    gated_auth,
)


# ── WS admission caps ────────────────────────────────────────────


def test_ws_admission_global_cap_view() -> None:
    """64-th view-channel admission succeeds; 65-th fails with global_cap."""

    async def scenario() -> None:
        caps = GuestCaps(max_view_ws=3, max_audio_out_ws=3, max_per_ip_ws=10)
        adm = WSAdmission(caps=caps)
        for _ in range(3):
            ok, _ = await adm.try_admit("view", "10.42.0.5", is_admin=False)
            assert ok
        ok, reason = await adm.try_admit("view", "10.42.0.6", is_admin=False)
        assert not ok
        assert reason == "global_cap"

    asyncio.run(scenario())


def test_ws_admission_per_ip_cap() -> None:
    """One IP cannot exceed max_per_ip_ws even when global cap has room."""

    async def scenario() -> None:
        caps = GuestCaps(max_view_ws=64, max_audio_out_ws=32, max_per_ip_ws=2)
        adm = WSAdmission(caps=caps)
        for _ in range(2):
            ok, _ = await adm.try_admit("view", "10.42.0.5", is_admin=False)
            assert ok
        ok, reason = await adm.try_admit("view", "10.42.0.5", is_admin=False)
        assert not ok
        assert reason == "per_ip_cap"
        # Different IP still admits.
        ok, _ = await adm.try_admit("view", "10.42.0.6", is_admin=False)
        assert ok

    asyncio.run(scenario())


def test_ws_admission_admin_bypasses_caps() -> None:
    """Admin-cookie callers bypass caps regardless of channel/IP."""

    async def scenario() -> None:
        caps = GuestCaps(max_view_ws=1, max_audio_out_ws=1, max_per_ip_ws=1)
        adm = WSAdmission(caps=caps)
        for _ in range(50):
            ok, _ = await adm.try_admit("view", "10.42.0.5", is_admin=True)
            assert ok

    asyncio.run(scenario())


def test_ws_admission_release_frees_slot() -> None:
    """Releasing one slot lets the next admission succeed."""

    async def scenario() -> None:
        caps = GuestCaps(max_view_ws=2, max_audio_out_ws=2, max_per_ip_ws=2)
        adm = WSAdmission(caps=caps)
        await adm.try_admit("view", "ip1", is_admin=False)
        await adm.try_admit("view", "ip2", is_admin=False)
        ok, _ = await adm.try_admit("view", "ip3", is_admin=False)
        assert not ok
        await adm.release("view", "ip1")
        ok, _ = await adm.try_admit("view", "ip3", is_admin=False)
        assert ok

    asyncio.run(scenario())


# ── Transcript drop tracker ──────────────────────────────────────


def test_drop_tracker_below_threshold() -> None:
    tracker = TranscriptDropTracker(window_s=30.0, threshold=4)
    for _ in range(4):
        assert tracker.record_drop("k1") is False


def test_drop_tracker_above_threshold_signals_close() -> None:
    tracker = TranscriptDropTracker(window_s=30.0, threshold=4)
    for _ in range(4):
        assert tracker.record_drop("k1") is False
    assert tracker.record_drop("k1") is True


def test_drop_tracker_window_eviction() -> None:
    """Drops outside the sliding window are dropped from the count."""
    tracker = TranscriptDropTracker(window_s=30.0, threshold=2)
    tracker.record_drop("k1", now=0.0)
    tracker.record_drop("k1", now=10.0)
    # 31s after the second drop both prior drops are out of window.
    assert tracker.record_drop("k1", now=41.0) is False


# ── Auth rate limiter ────────────────────────────────────────────


def test_rate_limiter_per_ip_exhaustion() -> None:
    rl = AuthRateLimiter(capacity=2)

    async def scenario() -> None:
        for _ in range(2):
            ok, _, _ = await rl.consume(client_ip="1.1.1.1", client_mac=None, now=0.0)
            assert ok
        ok, err, retry = await rl.consume(client_ip="1.1.1.1", client_mac=None, now=0.0)
        assert not ok
        assert err == "per_ip"
        assert retry > 0

    asyncio.run(scenario())


def test_rate_limiter_per_mac_exhaustion() -> None:
    """Same MAC, rotating IP — per-MAC bucket still binds (Plan R58)."""
    rl = AuthRateLimiter(capacity=2)

    async def scenario() -> None:
        for i in range(2):
            ok, _, _ = await rl.consume(
                client_ip=f"10.0.0.{i}",
                client_mac="aa:bb:cc:dd:ee:ff",
                now=0.0,
            )
            assert ok
        ok, err, _ = await rl.consume(
            client_ip="10.0.0.99",
            client_mac="aa:bb:cc:dd:ee:ff",
            now=0.0,
        )
        assert not ok
        assert err == "per_mac"

    asyncio.run(scenario())


def test_rate_limiter_progressive_backoff_resets_on_success() -> None:
    # Use a fast refill so we can advance synthetic time and replenish
    # the bucket after the reset.
    rl = AuthRateLimiter(capacity=1, refill_per_s=1.0)

    async def scenario() -> None:
        # Burn the bucket → backoff bumps to 60 s.
        ok, _, _ = await rl.consume(client_ip="x", client_mac=None, now=0.0)
        assert ok
        ok, err, _ = await rl.consume(client_ip="x", client_mac=None, now=0.5)
        assert not ok and err == "per_ip"
        # Advance 5 s → still in backoff window.
        ok, err, _ = await rl.consume(client_ip="x", client_mac=None, now=5.0)
        assert err == "backoff"
        # Reset on success — backoff window cleared.
        rl.reset_backoff("x")
        # Advance to t=70 so the bucket has refilled past 1 token AND
        # the backoff entry has been removed.
        ok, err, _ = await rl.consume(client_ip="x", client_mac=None, now=70.0)
        assert ok and err is None

    asyncio.run(scenario())


def test_rate_limiter_in_flight_semaphore_caps_concurrency() -> None:
    """Only ``in_flight_capacity`` auth_evals run at once; further
    callers queue. The third caller cannot enter until one of the first
    two releases its slot."""

    async def scenario() -> None:
        rl = AuthRateLimiter(capacity=10, in_flight_capacity=2)
        a = await rl.acquire_in_flight()
        b = await rl.acquire_in_flight()
        # Third acquisition must NOT complete while a + b are held.
        third = asyncio.create_task(rl.acquire_in_flight())
        await asyncio.sleep(0.02)
        assert not third.done()
        async with a:
            pass
        # a released → third can take its slot.
        slot = await asyncio.wait_for(third, timeout=0.5)
        async with slot:
            pass
        async with b:
            pass

    asyncio.run(scenario())


def test_constant_time_wait_pads_to_target() -> None:
    import time as _time

    async def scenario() -> None:
        start = _time.monotonic()
        await constant_time_wait(target_seconds=0.05, started_at=start)
        elapsed = _time.monotonic() - start
        assert elapsed >= 0.05

    asyncio.run(scenario())


def test_gated_auth_returns_invalid_secret_on_failed_handler() -> None:
    """Bucket has tokens, handler returns False (wrong secret) → result
    is ``(False, "invalid_secret", 0.0)`` AND the constant-time pad
    elapsed.
    """
    rl = AuthRateLimiter(capacity=5)

    async def handler() -> bool:
        return False

    async def scenario() -> tuple[bool, str | None, float]:
        return await gated_auth(
            client_ip="1.2.3.4",
            client_mac=None,
            rate_limiter=rl,
            success_handler=handler,
            constant_time_target=0.01,
        )

    ok, err, _ = asyncio.run(scenario())
    assert ok is False
    assert err == "invalid_secret"


def test_gated_auth_resets_backoff_on_success() -> None:
    rl = AuthRateLimiter(capacity=5)
    rl._backoff["1.2.3.4"] = (3, 0.0)  # mock prior backoff

    async def handler() -> bool:
        return True

    async def scenario() -> bool:
        ok, _, _ = await gated_auth(
            client_ip="1.2.3.4",
            client_mac=None,
            rate_limiter=rl,
            success_handler=handler,
            constant_time_target=0.0,
        )
        return ok

    assert asyncio.run(scenario()) is True
    assert "1.2.3.4" not in rl._backoff


def test_gated_auth_short_circuits_on_exhausted_bucket() -> None:
    """Bucket exhausted → handler is NEVER called (no auth_eval cost
    for the attacker's source)."""
    rl = AuthRateLimiter(capacity=0)
    handler_called = [0]

    async def handler() -> bool:
        handler_called[0] += 1
        return True

    async def scenario() -> tuple[bool, str | None, float]:
        return await gated_auth(
            client_ip="x",
            client_mac=None,
            rate_limiter=rl,
            success_handler=handler,
            constant_time_target=0.0,
        )

    ok, err, _ = asyncio.run(scenario())
    assert not ok
    assert err == "per_ip"
    assert handler_called[0] == 0
