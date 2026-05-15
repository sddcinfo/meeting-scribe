"""Single-holder GPU lease with priority-based preemption.

Recording must always win the GPU when a meeting is RECORDING. Phase B
finalize work runs in the background but only when the GPU is free.
This module is the synchronization primitive that makes that contract
true end-to-end:

* :meth:`GpuLease.acquire_recording` is sovereign — if Phase B holds the
  lease, it signals preempt, waits for the in-flight Phase B call to
  cancel + the backend to ack the abort, then takes the lease.
* :meth:`GpuLease.run_phase_b_call` runs ONE GPU call (single HTTP for
  ASR/diarize/summary, or a fan-out batch for translation). The factory
  receives an :class:`_RidAllocator`; it MUST mint a fresh request id
  for every outbound HTTP via ``await alloc.mint()``. Every minted id
  is registered with the lease so a preempt aborts the entire fan-out.
* :class:`_RidAllocator.mint` participates in the same condition lock
  as the preempt snapshot, so once preemption begins no new HTTP request
  can be issued unless its rid is also guaranteed to be in the abort set.

Holder transitions are atomic under ``self._cond``. Recording priority
is enforced via ``_recording_waiters`` so a queued Phase B retry can
never steal the idle transition that recording's preempt request created.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Final, Literal, TypeVar

import httpx

logger = logging.getLogger(__name__)


T = TypeVar("T")
BackendTag = Literal["asr", "translate", "diarize", "summary"]
HolderState = Literal["idle", "recording", "phase_b"]


class _Preempted(Exception):
    """Raised inside Phase B when recording acquires the lease.

    The caller (:func:`_run_phase_b_gpu_step`) catches this and retries
    after the lease is free again.
    """


#: How long the lease will spend on a single backend abort POST. The
#: per-call ceiling — but the overall preempt path is bounded by
#: :data:`_PREEMPT_BUDGET_S` via a shared monotonic deadline.
_BACKEND_ABORT_TIMEOUT_S: Final[float] = 1.0

#: Total budget from preempt-request to ``acquire_recording()`` returning.
#: Drives a single deadline shared across the abort POST and the local
#: task settle wait.
_PREEMPT_BUDGET_S: Final[float] = 1.5


class _RidAllocator:
    """Issued by :meth:`GpuLease.run_phase_b_call` to its factory.

    Single-call factories use ``allocator.primary`` for their one HTTP
    request. Batch factories (e.g. translation backlog replay) call
    ``await allocator.mint()`` per outbound HTTP request — every minted
    rid is registered with the lease so a preempt aborts the entire
    fan-out, not just the primary.

    Mint participates in the SAME cond-lock as the preempt snapshot,
    which makes the two outcomes mutually exclusive:

    1. mint completes before preempt: rid is in ``_inflight_rids`` when
       :meth:`GpuLease.run_phase_b_call` snapshots; the call is aborted.
    2. mint runs after preempt is requested: it sees
       ``_preempt_request == True`` under the lock and raises
       ``CancelledError`` BEFORE allocating a rid or issuing an HTTP
       request. The factory's coroutine fails fast; gather() propagates;
       the lease's ``run_phase_b_call`` observes ``_Preempted``.
    """

    def __init__(self, lease: GpuLease, primary: str) -> None:
        self._lease = lease
        self._primary = primary

    @property
    def primary(self) -> str:
        return self._primary

    async def mint(self) -> str:
        async with self._lease._cond:
            if self._lease._preempt_request:
                raise asyncio.CancelledError("rid allocation blocked: preempt in progress")
            rid = uuid.uuid4().hex
            self._lease._inflight_rids.add(rid)
            return rid


class GpuLease:
    """The runtime's single GPU-ownership primitive.

    Parameters
    ----------
    preempt_budget_s :
        Override the default :data:`_PREEMPT_BUDGET_S`. Tests pass a
        smaller value so they don't have to wait the full default budget
        when verifying the preempt-and-recover round-trip.
    """

    def __init__(self, *, preempt_budget_s: float | None = None) -> None:
        self._cond = asyncio.Condition()
        self._holder: HolderState = "idle"
        self._preempt_request: bool = False
        self._recording_waiters: int = 0
        self._inflight_backend: BackendTag | None = None
        self._inflight_rids: set[str] = set()
        self._backends: dict[BackendTag, Callable[[str], str]] = {}
        self._backend_dirty: dict[BackendTag, bool] = {}
        self._preempt_budget_s = preempt_budget_s or _PREEMPT_BUDGET_S

    # ─── Backend registration ─────────────────────────────────────────

    def register_backend(
        self,
        tag: BackendTag,
        abort_url_builder: Callable[[str], str],
    ) -> None:
        """Register the abort-URL builder for a backend.

        Called once at lifespan startup for each of the four GPU
        backends. ``abort_url_builder(rid)`` returns the absolute URL
        the lease will POST to when preempting an in-flight call.
        """
        self._backends[tag] = abort_url_builder

    def is_backend_dirty(self, tag: BackendTag) -> bool:
        """True if the most recent abort for ``tag`` failed.

        Recording acquire logs a warning when the next leased call
        targets a dirty backend so the operator knows GPU contention is
        possible.
        """
        return self._backend_dirty.get(tag, False)

    @property
    def holder(self) -> HolderState:
        # Snapshot read for tests/observability — callers that need a
        # consistent transition should still go through the cond lock.
        return self._holder

    # ─── Recording (sovereign) ────────────────────────────────────────

    async def acquire_recording(self) -> None:
        """Block until the lease is held by recording.

        Marks ourselves as a pending recording waiter BEFORE entering
        the wait loop so any racing Phase B acquire sees
        ``_recording_waiters > 0`` and yields. If Phase B holds, we
        signal preempt and wait for its release.
        """
        async with self._cond:
            self._recording_waiters += 1
            try:
                while self._holder == "phase_b":
                    self._preempt_request = True
                    self._cond.notify_all()  # wake the preempt watcher
                    await self._cond.wait()
                if self._holder == "recording":
                    # Idempotent recovery path: start_meeting's fast
                    # path handles the normal "already recording"
                    # case before it calls us. If we still see
                    # holder=recording here, runtime state has lost
                    # the current meeting while the lease retained the
                    # sovereign holder. Treat it as already acquired
                    # instead of crashing the next Start request.
                    self._preempt_request = False
                    return
                # Phase B woke + released; we hold the lock now.
                # No queued Phase B waiter can have stolen the idle
                # transition because run_phase_b_call's predicate gates
                # on _recording_waiters > 0.
                assert self._holder == "idle", f"unexpected holder={self._holder!r}"
                self._holder = "recording"
                self._preempt_request = False
            finally:
                self._recording_waiters -= 1
            self._cond.notify_all()

    async def release_recording(self) -> None:
        async with self._cond:
            assert self._holder == "recording", (
                f"release_recording called with holder={self._holder!r}"
            )
            self._holder = "idle"
            self._cond.notify_all()

    # ─── Phase B (preemptable) ────────────────────────────────────────

    async def run_phase_b_call(
        self,
        factory: Callable[[_RidAllocator], Awaitable[T]],
        *,
        backend: BackendTag,
    ) -> T:
        """Run a single GPU-bound Phase B call under the lease.

        The factory receives a :class:`_RidAllocator`; it MUST mint a
        rid for every outbound HTTP request. On preempt, every minted
        rid is aborted in parallel under one shared deadline.

        Raises
        ------
        _Preempted
            Recording acquired the lease while this call was in flight.
            Caller (typically :func:`_run_phase_b_gpu_step`) retries
            after the lease is free again.
        """
        primary_rid = uuid.uuid4().hex
        abort_url_builder = self._backends.get(backend)
        if abort_url_builder is None:
            raise RuntimeError(f"GpuLease has no backend registered for tag={backend!r}")

        async with self._cond:
            # Recording-priority gate: yield to any pending recording
            # waiter so a queued Phase B retry can't steal the idle
            # transition created by preempt.
            while self._holder != "idle" or self._recording_waiters > 0:
                await self._cond.wait()
            self._holder = "phase_b"
            self._preempt_request = False
            self._inflight_backend = backend
            self._inflight_rids = {primary_rid}
            self._cond.notify_all()
        try:
            allocator = _RidAllocator(self, primary_rid)
            work: asyncio.Task[T] = asyncio.create_task(factory(allocator))
            preempt_watch: asyncio.Task[None] = asyncio.create_task(self._wait_for_preempt())
            done, _pending = await asyncio.wait(
                {work, preempt_watch}, return_when=asyncio.FIRST_COMPLETED
            )
            if preempt_watch in done and not work.done():
                deadline = time.monotonic() + self._preempt_budget_s
                work.cancel()
                # Snapshot rids under the cond lock so a racing mint()
                # either completes BEFORE the snapshot (rid in set, will
                # be aborted) or sees _preempt_request=True and refuses.
                async with self._cond:
                    rids_to_abort = list(self._inflight_rids)
                # Abort all rids in parallel; per-call timeout = remaining budget.
                abort_results = await asyncio.gather(
                    *(self._abort_one(abort_url_builder, rid, deadline) for rid in rids_to_abort)
                )
                self._backend_dirty[backend] = not all(abort_results)
                # Settle local task within the REMAINING budget.
                remaining = max(0.0, deadline - time.monotonic())
                with contextlib.suppress(asyncio.CancelledError, Exception, asyncio.TimeoutError):
                    await asyncio.wait_for(work, timeout=remaining)
                raise _Preempted()
            preempt_watch.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await preempt_watch
            return work.result()
        finally:
            async with self._cond:
                self._holder = "idle"
                self._inflight_backend = None
                self._inflight_rids = set()
                self._preempt_request = False
                self._cond.notify_all()

    async def _abort_one(
        self,
        abort_url_builder: Callable[[str], str],
        rid: str,
        deadline: float,
    ) -> bool:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return False
        timeout_s = min(_BACKEND_ABORT_TIMEOUT_S, remaining)
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                r = await client.post(abort_url_builder(rid))
                ok = r.status_code in (200, 404)
                if not ok:
                    logger.warning(
                        "backend abort returned %d for rid=%s",
                        r.status_code,
                        rid,
                    )
                return ok
        except Exception:
            logger.exception("backend abort failed rid=%s", rid)
            return False

    async def _wait_for_preempt(self) -> None:
        async with self._cond:
            while not self._preempt_request:
                await self._cond.wait()


_gpu_lease_singleton: GpuLease | None = None


def gpu_lease() -> GpuLease:
    """Process-wide GpuLease singleton.

    Created lazily so test fixtures can construct their own instances
    without importing this module triggering the global. In production
    the singleton is the one registered with backends in
    :mod:`meeting_scribe.runtime.lifespan`.
    """
    global _gpu_lease_singleton
    if _gpu_lease_singleton is None:
        budget_env = os.environ.get("SCRIBE_GPU_LEASE_PREEMPT_BUDGET_S")
        budget = float(budget_env) if budget_env else None
        _gpu_lease_singleton = GpuLease(preempt_budget_s=budget)
    return _gpu_lease_singleton


def _reset_gpu_lease_for_tests() -> None:
    """Test hook: drop the singleton so the next call constructs fresh."""
    global _gpu_lease_singleton
    _gpu_lease_singleton = None
