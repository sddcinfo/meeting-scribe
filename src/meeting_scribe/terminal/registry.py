"""Capacity-bounded registry of active :class:`TerminalSession` objects.

Capacity is enforced by a bounded :class:`asyncio.Queue` of sentinel
tokens: :meth:`reserve` does a non-blocking ``get_nowait``, returning
``False`` if the queue is empty. Both ``get_nowait`` and ``put_nowait``
are event-loop-atomic — no await point, no race — which is the property
a Semaphore-based design lacks when callers want a strictly
non-blocking reservation.

Lifecycle:

    registry.reserve(ws)  # True iff slot grabbed; False if full
    try:
        session = await TerminalSession.spawn(...)
    except Exception:
        registry.rollback(ws)
        raise
    else:
        registry.fulfill(ws, session)
    # ... WS lifetime ...
    await registry.unregister(ws)   # idempotent; releases token
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import WebSocket

    from meeting_scribe.terminal.pty_session import TerminalSession

logger = logging.getLogger(__name__)


# Sentinel identity doesn't matter; only the count of tokens in the queue.
_TOKEN = object()


@dataclass
class ActiveTerminals:
    max_concurrent: int = 4
    items: dict[Any, TerminalSession] = field(default_factory=dict)
    _tokens: asyncio.Queue[object] = field(init=False)
    _reserved_tokens: dict[Any, object] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._tokens = asyncio.Queue(maxsize=self.max_concurrent)
        for _ in range(self.max_concurrent):
            self._tokens.put_nowait(_TOKEN)

    def reserve(self, ws: WebSocket) -> bool:
        """Atomically grab a slot for *ws*. Returns False if at capacity.

        Must be followed by either :meth:`fulfill` (on success) or
        :meth:`rollback` (on spawn failure).
        """
        if ws in self._reserved_tokens:
            # Already reserved — treat as no-op success so idempotent callers don't double-book.
            return True
        try:
            token = self._tokens.get_nowait()
        except asyncio.QueueEmpty:
            return False
        self._reserved_tokens[ws] = token
        return True

    def fulfill(self, ws: WebSocket, session: TerminalSession) -> None:
        """Attach *session* to the reservation held by *ws*."""
        if ws not in self._reserved_tokens:
            raise RuntimeError("fulfill() without preceding reserve()")
        self.items[ws] = session

    def rollback(self, ws: WebSocket) -> None:
        """Release the reservation when spawn failed before fulfill."""
        token = self._reserved_tokens.pop(ws, None)
        if token is not None:
            self._tokens.put_nowait(token)

    def unregister(self, ws: WebSocket) -> None:
        """Release *ws*'s slot. Idempotent, synchronous, cancellation-safe.

        Kept sync (not async) so the WS handler can call it from a finally
        block that may itself be racing with coroutine cancellation — we
        never want a stale entry to linger just because close() was
        interrupted mid-await.
        """
        self.items.pop(ws, None)
        token = self._reserved_tokens.pop(ws, None)
        if token is not None:
            self._tokens.put_nowait(token)

    async def close_sessions_by_name(self, name: str, *, reason: str) -> int:
        """Close every active session whose tmux_session matches *name*.

        Used by the reset endpoint so clicking the panel's X cleanly
        drops the WS AND signals the tmux-level session kill. Returns
        the number of sessions closed.
        """
        victims = [(ws, sess) for ws, sess in list(self.items.items()) if sess.tmux_session == name]
        for ws, session in victims:
            self.unregister(ws)
            try:
                await session.close(reason=reason)
            except Exception:
                logger.exception(
                    "terminal close_sessions_by_name: %s reason=%s failed",
                    session.tmux_session,
                    reason,
                )
        return len(victims)

    async def close_all(self, *, reason: str = "shutdown") -> None:
        """Close every session and drain the registry. Tokens re-fill."""
        sessions = list(self.items.values())
        self.items.clear()
        self._reserved_tokens.clear()
        # Refill tokens to max (queue started full, tokens were checked out as
        # sessions attached; after clearing items/reservations we want the
        # invariant `qsize() == max_concurrent` to hold for post-shutdown state).
        while self._tokens.qsize() < self.max_concurrent:
            self._tokens.put_nowait(_TOKEN)
        for session in sessions:
            try:
                await session.close(reason=reason)
            except Exception:
                logger.exception(
                    "terminal close_all: failed to close session %s",
                    getattr(session, "tmux_session", "<unknown>"),
                )

    def summary(self) -> dict[str, Any]:
        return {
            "count": len(self.items),
            "available": self._tokens.qsize(),
            "max": self.max_concurrent,
            "sessions": [session.summary() for session in self.items.values()],
        }
