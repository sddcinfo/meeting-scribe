"""Guest admission control + per-IP/MAC auth rate limit (Plan §Guest
Admission Control).

Two responsibilities:

1. **WS connection caps** for guest fan-out. Per-listener bounded queues
   (drop-oldest for transcript JSON; immediate-close for stateful
   MSE/fMP4 audio). Per-IP fairness so one phone with many tabs cannot
   monopolize. Auth-aware exemption — a request carrying a valid
   ``scribe_admin`` cookie bypasses guest caps regardless of path.

2. **Multi-layer rate limit** on ``/api/admin/authorize``: per-IP
   token bucket, per-DHCP-MAC token bucket, global in-flight semaphore.
   No global progressive backoff (Plan R62) — per-source backoff only,
   resets on first success.

Pure async-safe code with no FastAPI / Starlette imports — the server
wires the helpers up via middleware.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


# ── Cap configuration ─────────────────────────────────────────────


@dataclass(frozen=True)
class GuestCaps:
    """Static caps consulted at every WS handshake.

    Defaults match Plan §Guest Admission Control. Production reads
    from ``config.py`` (env-overridable); tests construct fresh
    instances per case.
    """

    max_view_ws: int = 64
    max_audio_out_ws: int = 32
    max_per_ip_ws: int = 4
    transcript_drop_threshold: int = 32  # drops in 30s → close listener
    transcript_drop_window_s: float = 30.0


DEFAULT_CAPS = GuestCaps()


# ── WS connection counter ─────────────────────────────────────────


@dataclass
class WSAdmission:
    """Tracks live WS counts per channel + per IP. Async-safe via the
    embedded lock.

    Channels are named strings (``"view"``, ``"audio_out"``) so adding a
    new fan-out doesn't require changing the data structure.
    """

    caps: GuestCaps = DEFAULT_CAPS
    _per_channel: dict[str, int] = field(default_factory=dict)
    _per_ip_per_channel: dict[tuple[str, str], int] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def _channel_cap(self, channel: str) -> int:
        if channel == "view":
            return self.caps.max_view_ws
        if channel == "audio_out":
            return self.caps.max_audio_out_ws
        # Unknown channels share the smallest cap as a safety floor.
        return min(self.caps.max_view_ws, self.caps.max_audio_out_ws)

    async def try_admit(
        self,
        channel: str,
        client_ip: str,
        *,
        is_admin: bool,
    ) -> tuple[bool, str | None]:
        """Try to reserve one slot on ``channel`` for ``client_ip``.

        Returns ``(ok, reason)``. ``reason`` carries one of
        ``"global_cap"`` / ``"per_ip_cap"`` on failure so the caller can
        emit the right HTTP/WS code.

        Admin-cookie callers (``is_admin=True``) bypass every cap.
        """
        if is_admin:
            return True, None
        async with self._lock:
            channel_count = self._per_channel.get(channel, 0)
            if channel_count >= self._channel_cap(channel):
                return False, "global_cap"
            ip_count = self._per_ip_per_channel.get((channel, client_ip), 0)
            if ip_count >= self.caps.max_per_ip_ws:
                return False, "per_ip_cap"
            self._per_channel[channel] = channel_count + 1
            self._per_ip_per_channel[(channel, client_ip)] = ip_count + 1
            return True, None

    async def release(self, channel: str, client_ip: str) -> None:
        """Release a slot — call from the WS handler's ``finally``."""
        async with self._lock:
            count = self._per_channel.get(channel, 0)
            if count > 0:
                self._per_channel[channel] = count - 1
            ip_count = self._per_ip_per_channel.get((channel, client_ip), 0)
            if ip_count > 0:
                self._per_ip_per_channel[(channel, client_ip)] = ip_count - 1
            if self._per_ip_per_channel.get((channel, client_ip)) == 0:
                self._per_ip_per_channel.pop((channel, client_ip), None)


# ── Drop-oldest deque (transcript JSON) ──────────────────────────


@dataclass
class TranscriptDropTracker:
    """Tracks per-listener transcript-frame drop counts in a sliding
    window. Listeners that drop too many frames in the window are
    closed with WS code 1013 ``slow_consumer``.
    """

    window_s: float
    threshold: int
    _drops: dict[Any, deque[float]] = field(default_factory=dict)

    def record_drop(self, listener_key: Any, *, now: float | None = None) -> bool:
        """Record one drop; return True if the listener should be closed."""
        current = now if now is not None else time.monotonic()
        bucket = self._drops.setdefault(listener_key, deque())
        bucket.append(current)
        cutoff = current - self.window_s
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        return len(bucket) > self.threshold

    def discard(self, listener_key: Any) -> None:
        self._drops.pop(listener_key, None)


# ── Auth rate limit ──────────────────────────────────────────────


@dataclass
class _Bucket:
    """Token-bucket primitive — one capacity, refills at ``rate`` per
    second. ``last`` carries the float timestamp of the last refill.
    """

    capacity: int
    rate: float
    tokens: float
    last: float


@dataclass
class AuthRateLimiter:
    """Per-IP + per-MAC + global in-flight limiter for
    ``/api/admin/authorize``.

    Plan R50/R54/R58/R62: drop the global-progressive-backoff —
    a persistent attacker on one source slows down their own
    source only; legitimate operators on a different IP+MAC are
    unaffected.
    """

    capacity: int = 5
    refill_per_s: float = 5.0 / 60.0  # 5 per 60 s
    in_flight_capacity: int = 2
    _per_ip: dict[str, _Bucket] = field(default_factory=dict)
    _per_mac: dict[str, _Bucket] = field(default_factory=dict)
    _backoff: dict[str, tuple[int, float]] = field(default_factory=dict)
    # in_flight semaphore is None at construction time so the asyncio
    # loop is owned at first acquire (matches the rest of the project).
    _in_flight: asyncio.Semaphore | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def _bucket(self, store: dict[str, _Bucket], key: str, *, now: float) -> _Bucket:
        bucket = store.get(key)
        if bucket is None:
            bucket = _Bucket(
                capacity=self.capacity,
                rate=self.refill_per_s,
                tokens=float(self.capacity),
                last=now,
            )
            store[key] = bucket
        return bucket

    @staticmethod
    def _refill(bucket: _Bucket, now: float) -> None:
        elapsed = max(0.0, now - bucket.last)
        bucket.tokens = min(float(bucket.capacity), bucket.tokens + elapsed * bucket.rate)
        bucket.last = now

    async def consume(
        self,
        *,
        client_ip: str,
        client_mac: str | None,
        now: float | None = None,
    ) -> tuple[bool, str | None, float]:
        """Attempt one auth attempt for ``(client_ip, client_mac)``.

        Returns ``(ok, error_code, retry_after_seconds)``.

        Failure paths:
          * ``"per_ip"`` — IP bucket exhausted.
          * ``"per_mac"`` — MAC bucket exhausted (IP-rotating attacker
            with a stable DHCP MAC).
          * ``"backoff"`` — progressive-backoff still active for this
            source.

        Successful paths: the caller still has to acquire the
        in-flight semaphore via :meth:`acquire_in_flight`.
        """
        current = now if now is not None else time.monotonic()
        async with self._lock:
            # Progressive backoff per source — checked first.
            backoff_until = self._backoff.get(client_ip, (0, 0.0))[1]
            if backoff_until and current < backoff_until:
                return False, "backoff", backoff_until - current

            ip_bucket = self._bucket(self._per_ip, client_ip, now=current)
            self._refill(ip_bucket, current)
            if ip_bucket.tokens < 1:
                self._bump_backoff(client_ip, current)
                _, until = self._backoff[client_ip]
                return False, "per_ip", until - current

            mac_bucket: _Bucket | None = None
            if client_mac:
                mac_bucket = self._bucket(self._per_mac, client_mac, now=current)
                self._refill(mac_bucket, current)
                if mac_bucket.tokens < 1:
                    return False, "per_mac", 60.0

            ip_bucket.tokens -= 1
            if mac_bucket is not None:
                mac_bucket.tokens -= 1
            return True, None, 0.0

    def _bump_backoff(self, client_ip: str, now: float) -> None:
        steps = (60.0, 120.0, 240.0, 600.0, 600.0)
        attempts, _ = self._backoff.get(client_ip, (0, 0.0))
        idx = min(attempts, len(steps) - 1)
        self._backoff[client_ip] = (attempts + 1, now + steps[idx])

    def reset_backoff(self, client_ip: str) -> None:
        """First successful auth on a source resets its progressive
        backoff. Called by the route handler on the success path."""
        self._backoff.pop(client_ip, None)

    async def acquire_in_flight(self) -> "AuthInFlight":
        """Take one in-flight slot. Up to ``in_flight_capacity`` auth
        evaluations may run concurrently; further attempts queue
        briefly. This is an in-flight semaphore (NOT a fixed-window
        bucket) so persistent failed attempts don't lock out
        legitimate operators."""
        if self._in_flight is None:
            self._in_flight = asyncio.Semaphore(self.in_flight_capacity)
        await self._in_flight.acquire()
        return AuthInFlight(self._in_flight)


@dataclass
class AuthInFlight:
    """Async context manager — release the in-flight slot on exit."""

    sem: asyncio.Semaphore

    async def __aenter__(self) -> "AuthInFlight":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.sem.release()


# ── Constant-time wait helper ────────────────────────────────────


async def constant_time_wait(*, target_seconds: float, started_at: float) -> None:
    """Pad the elapsed handler time so failure responses always take
    ``target_seconds`` end-to-end. Mitigates timing-oracle leaks on
    ``hmac.compare_digest`` — Plan §Multi-layer rate limit.
    """
    elapsed = time.monotonic() - started_at
    remaining = target_seconds - elapsed
    if remaining > 0:
        await asyncio.sleep(remaining)


# ── In-process helper for the route handler ──────────────────────


async def gated_auth(
    *,
    client_ip: str,
    client_mac: str | None,
    rate_limiter: AuthRateLimiter,
    success_handler: Callable[[], Awaitable[bool]],
    constant_time_target: float = 0.05,
) -> tuple[bool, str | None, float]:
    """Run one ``/api/admin/authorize`` evaluation under the limiter.

    * Per-IP / per-MAC bucket consumption via ``rate_limiter.consume``.
    * If the bucket is exhausted: return immediately with
      ``(False, error_code, retry_after)``; no auth_eval runs.
    * Otherwise acquire one in-flight slot, run ``success_handler()``,
      pad to ``constant_time_target`` so the failure path takes a
      fixed wall-clock time.
    * On a successful auth, the per-source progressive backoff is
      reset.

    Returns ``(ok, error_code, retry_after)``. The caller turns this
    tuple into the final HTTP response.
    """
    ok, err, retry = await rate_limiter.consume(client_ip=client_ip, client_mac=client_mac)
    if not ok:
        return False, err, retry
    started = time.monotonic()
    in_flight = await rate_limiter.acquire_in_flight()
    try:
        async with in_flight:
            authed = await success_handler()
    finally:
        await constant_time_wait(target_seconds=constant_time_target, started_at=started)
    if authed:
        rate_limiter.reset_backoff(client_ip)
        return True, None, 0.0
    return False, "invalid_secret", 0.0
