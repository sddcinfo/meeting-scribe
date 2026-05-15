"""Microtests for :class:`meeting_scribe.runtime.gpu_lease.GpuLease`.

Covers every race-condition regression caught by the plan-review iterations:

* iter-4 — recording must preempt an in-flight Phase B before its work
  coroutine completes.
* iter-11/12 — recording priority gate (`_recording_waiters > 0`) prevents
  a queued Phase B retry from stealing the idle transition created by
  preempt. Test runs against the **backend-aware** ``run_phase_b_call``
  to prevent regressions where the priority predicate is fixed in one
  sketch but missed in another.
* iter-13 — single shared monotonic deadline drives both the abort POST
  and the local task settle wait. Total wall-clock from preempt to
  recording acquire stays bounded by ``_PREEMPT_BUDGET_S``.
* iter-13 — fan-out abort: a factory minting multiple rids has every
  rid posted to ``/abort/{rid}`` on preempt, not just the primary.
* iter-14 — mint participates in the same cond lock as the preempt
  snapshot, so an HTTP request issued AFTER preempt cannot escape the
  abort fan-out.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from meeting_scribe.backends.protocol import abort_url
from meeting_scribe.runtime.gpu_lease import (
    GpuLease,
    _Preempted,
    _RidAllocator,
)

# ─── Synthetic GPU backend ──────────────────────────────────────────────


@dataclass
class _RequestRecord:
    started_at: float | None = None
    aborted_at: float | None = None
    completed_at: float | None = None


@dataclass
class _SyntheticBackendState:
    requests: dict[str, _RequestRecord] = field(default_factory=dict)
    cancellation: dict[str, asyncio.Event] = field(default_factory=dict)
    pre_aborted: set[str] = field(default_factory=set)
    cooperative: bool = True
    work_seconds: float = 30.0
    abort_delay_s: float = 0.0


def _make_backend_app(state: _SyntheticBackendState) -> FastAPI:
    app = FastAPI()

    @app.post("/work")
    async def work(request_id: str) -> JSONResponse:
        state.requests[request_id] = _RequestRecord(started_at=time.monotonic())
        cancel_event = asyncio.Event()
        state.cancellation[request_id] = cancel_event
        # Race fix: if /abort fired before /work even registered, the
        # rid is in pre_aborted; treat it as already cancelled.
        if request_id in state.pre_aborted:
            cancel_event.set()
        try:
            try:
                await asyncio.wait_for(cancel_event.wait(), timeout=state.work_seconds)
                state.requests[request_id].aborted_at = time.monotonic()
                return JSONResponse({"status": "aborted"}, status_code=499)
            except TimeoutError:
                state.requests[request_id].completed_at = time.monotonic()
                return JSONResponse({"status": "ok"})
        except asyncio.CancelledError:
            # Client disconnected mid-flight.
            state.requests[request_id].aborted_at = time.monotonic()
            raise

    @app.post("/abort/{request_id}")
    async def abort(request_id: str) -> JSONResponse:
        if state.abort_delay_s > 0:
            await asyncio.sleep(state.abort_delay_s)
        if not state.cooperative:
            return JSONResponse({"status": "ignored"}, status_code=500)
        ev = state.cancellation.get(request_id)
        if ev is None:
            # /abort raced ahead of /work. Remember the intent so /work
            # picks it up when it arrives.
            state.pre_aborted.add(request_id)
            return JSONResponse({"status": "deferred"}, status_code=200)
        ev.set()
        return JSONResponse({"status": "aborted"})

    return app


@pytest.fixture
def backend() -> Iterator[dict[str, Any]]:
    state = _SyntheticBackendState()
    app = _make_backend_app(state)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5.0
    while (
        not server.started or not getattr(server, "servers", None)
    ) and time.monotonic() < deadline:
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield {
        "base_url": f"http://127.0.0.1:{port}",
        "state": state,
    }
    server.should_exit = True
    thread.join(timeout=3.0)


@pytest.fixture
def lease(backend: dict[str, Any]) -> GpuLease:
    lease = GpuLease(preempt_budget_s=2.0)
    lease.register_backend(
        "diarize",
        lambda rid: abort_url(backend["base_url"], rid),
    )
    return lease


async def _do_work(client_base: str, rid: str) -> str:
    import httpx

    async with httpx.AsyncClient(timeout=60.0) as c:
        r = await c.post(f"{client_base}/work", params={"request_id": rid})
        return r.text


# ─── Tests ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_recording_preempts_in_flight_phase_b(
    lease: GpuLease, backend: dict[str, Any]
) -> None:
    """iter-4 regression: recording must preempt mid-flight Phase B."""
    state = backend["state"]
    state.work_seconds = 30.0

    async def factory(alloc: _RidAllocator) -> str:
        return await _do_work(backend["base_url"], alloc.primary)

    phase_b = asyncio.create_task(lease.run_phase_b_call(factory, backend="diarize"))
    # Let Phase B start its HTTP request.
    for _ in range(50):
        if state.requests:
            break
        await asyncio.sleep(0.05)
    assert state.requests, "Phase B HTTP did not start"

    t0 = time.monotonic()
    await lease.acquire_recording()
    elapsed = time.monotonic() - t0
    assert elapsed <= 2.0, f"recording acquire took {elapsed:.2f}s, > budget"
    assert lease.holder == "recording"

    with pytest.raises(_Preempted):
        await phase_b

    rid = next(iter(state.requests))
    assert state.requests[rid].aborted_at is not None
    assert state.requests[rid].completed_at is None
    await lease.release_recording()


@pytest.mark.asyncio
async def test_recording_priority_over_queued_phase_b(
    lease: GpuLease, backend: dict[str, Any]
) -> None:
    """iter-11/12 regression: queued Phase B retries cannot steal the
    idle transition created by recording's preempt request."""
    state = backend["state"]
    state.work_seconds = 30.0

    async def factory(alloc: _RidAllocator) -> str:
        return await _do_work(backend["base_url"], alloc.primary)

    in_flight = asyncio.create_task(lease.run_phase_b_call(factory, backend="diarize"))
    for _ in range(50):
        if state.requests:
            break
        await asyncio.sleep(0.05)

    queued = [
        asyncio.create_task(lease.run_phase_b_call(factory, backend="diarize")) for _ in range(5)
    ]

    holder_when_recording_acquires: list[str] = []

    async def take_recording() -> None:
        await lease.acquire_recording()
        holder_when_recording_acquires.append(lease.holder)

    rec_task = asyncio.create_task(take_recording())
    await rec_task
    assert holder_when_recording_acquires == ["recording"]
    assert lease.holder == "recording"

    # The in-flight Phase B was preempted.
    with pytest.raises(_Preempted):
        await in_flight

    # Queued retries are still pending — they MUST NOT have advanced past
    # the priority gate while recording was pending or while it holds.
    for q in queued:
        assert not q.done(), "queued phase_b ran while recording held the lease"

    await lease.release_recording()

    # Now queued retries can run; clean them up.
    for q in queued:
        q.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await q


@pytest.mark.asyncio
async def test_preempt_total_wallclock_under_budget(
    backend: dict[str, Any],
) -> None:
    """iter-13 regression: ONE shared deadline drives both the abort POST
    and the local task settle wait. Even with a slow abort endpoint, the
    end-to-end preempt path stays within ``_PREEMPT_BUDGET_S``."""
    backend["state"].work_seconds = 30.0
    backend["state"].abort_delay_s = 0.8  # half the budget

    lease = GpuLease(preempt_budget_s=1.5)
    lease.register_backend(
        "diarize",
        lambda rid: abort_url(backend["base_url"], rid),
    )

    async def factory(alloc: _RidAllocator) -> str:
        return await _do_work(backend["base_url"], alloc.primary)

    phase_b = asyncio.create_task(lease.run_phase_b_call(factory, backend="diarize"))
    for _ in range(50):
        if backend["state"].requests:
            break
        await asyncio.sleep(0.05)

    t0 = time.monotonic()
    await lease.acquire_recording()
    elapsed = time.monotonic() - t0
    # Slack for asyncio scheduling / process-startup variance.
    assert elapsed <= 1.5 + 0.5, f"recording acquire took {elapsed:.2f}s, exceeds budget+slack"

    with pytest.raises(_Preempted):
        await phase_b
    await lease.release_recording()


@pytest.mark.asyncio
async def test_fan_out_abort_aborts_every_minted_rid(
    lease: GpuLease, backend: dict[str, Any]
) -> None:
    """iter-13 regression: every rid minted by the factory is in the
    abort fan-out, not just the primary. Translate replay relies on this
    so a preempt aborts the entire batch."""
    state = backend["state"]
    state.work_seconds = 30.0
    minted: list[str] = []

    async def batch_factory(alloc: _RidAllocator) -> list[str]:
        async def _one() -> str:
            rid = await alloc.mint()
            minted.append(rid)
            return await _do_work(backend["base_url"], rid)

        return await asyncio.gather(*(_one() for _ in range(5)))

    phase_b = asyncio.create_task(lease.run_phase_b_call(batch_factory, backend="diarize"))
    # Wait for all 5 backend requests to start.
    for _ in range(100):
        if len(state.requests) >= 5:
            break
        await asyncio.sleep(0.05)
    assert len(state.requests) >= 5, f"only {len(state.requests)} requests started"

    await lease.acquire_recording()
    with pytest.raises(_Preempted):
        await phase_b

    for rid in minted:
        assert rid in state.requests
        assert state.requests[rid].aborted_at is not None, f"rid {rid} was never aborted"
        assert state.requests[rid].completed_at is None
    await lease.release_recording()


@pytest.mark.asyncio
async def test_mint_blocks_after_preempt_request_unit() -> None:
    """iter-14 regression: ``_RidAllocator.mint`` raises CancelledError
    once ``_preempt_request`` is set. The unit-level test bypasses the
    factory-cancellation race so we can drive the exact predicate the
    plan promises.

    The lease runs the factory under ``asyncio.create_task`` and cancels
    it when preempt fires; that cancellation propagates up the factory's
    own ``await`` points before the factory ever gets to call mint a
    second time. The end-to-end fan-out abort coverage is asserted in
    :func:`test_fan_out_abort_aborts_every_minted_rid`. This test is the
    targeted predicate check for the mint-vs-preempt race.
    """
    lease = GpuLease(preempt_budget_s=0.5)
    lease.register_backend("diarize", lambda rid: f"http://x/{rid}")
    allocator = _RidAllocator(lease, primary="primary-rid")

    # Without preempt: mint succeeds and the rid is registered.
    rid = await allocator.mint()
    assert rid in lease._inflight_rids

    # Simulate preempt being requested.
    async with lease._cond:
        lease._preempt_request = True
        lease._cond.notify_all()

    with pytest.raises(asyncio.CancelledError, match="preempt in progress"):
        await allocator.mint()


@pytest.mark.asyncio
async def test_uncooperative_backend_marks_dirty_but_releases_lease(
    backend: dict[str, Any],
) -> None:
    """iter-7 regression: if the backend ignores ``/abort``, recording
    still acquires within budget and the lease records a dirty flag."""
    backend["state"].work_seconds = 30.0
    backend["state"].cooperative = False  # /abort returns 500

    lease = GpuLease(preempt_budget_s=1.5)
    lease.register_backend(
        "diarize",
        lambda rid: abort_url(backend["base_url"], rid),
    )

    async def factory(alloc: _RidAllocator) -> str:
        return await _do_work(backend["base_url"], alloc.primary)

    phase_b = asyncio.create_task(lease.run_phase_b_call(factory, backend="diarize"))
    for _ in range(50):
        if backend["state"].requests:
            break
        await asyncio.sleep(0.05)

    t0 = time.monotonic()
    await lease.acquire_recording()
    elapsed = time.monotonic() - t0
    assert elapsed <= 2.0
    assert lease.is_backend_dirty("diarize") is True

    with pytest.raises(_Preempted):
        await phase_b
    await lease.release_recording()


@pytest.mark.asyncio
async def test_unregistered_backend_raises() -> None:
    lease = GpuLease(preempt_budget_s=0.5)

    async def factory(alloc: _RidAllocator) -> None:
        await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="no backend registered"):
        await lease.run_phase_b_call(factory, backend="diarize")


@pytest.mark.asyncio
async def test_release_recording_asserts_holder() -> None:
    lease = GpuLease(preempt_budget_s=0.5)
    with pytest.raises(AssertionError):
        await lease.release_recording()


@pytest.mark.asyncio
async def test_acquire_recording_is_idempotent_when_holder_already_recording() -> None:
    lease = GpuLease(preempt_budget_s=0.5)
    await lease.acquire_recording()
    await lease.acquire_recording()
    assert lease.holder == "recording"
    await lease.release_recording()
