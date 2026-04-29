"""Registry capacity + reserve/fulfill/rollback/unregister semantics."""

from __future__ import annotations

import pytest

from meeting_scribe.terminal.registry import ActiveTerminals


class FakeWS:
    pass


class FakeSession:
    def __init__(self, name: str = "fake"):
        self.tmux_session = name
        self.closed_reason: str | None = None

    async def close(self, *, reason: str) -> None:
        self.closed_reason = reason

    def summary(self) -> dict:
        return {"tmux_session": self.tmux_session}


async def test_reserve_then_fulfill_happy_path():
    reg = ActiveTerminals(max_concurrent=2)
    ws = FakeWS()
    assert reg.reserve(ws)  # type: ignore[arg-type]
    session = FakeSession("a")
    reg.fulfill(ws, session)  # type: ignore[arg-type]
    assert reg.summary()["count"] == 1
    assert reg.summary()["available"] == 1


async def test_reserve_returns_false_at_cap():
    reg = ActiveTerminals(max_concurrent=2)
    ws1, ws2, ws3 = FakeWS(), FakeWS(), FakeWS()
    assert reg.reserve(ws1)  # type: ignore[arg-type]
    assert reg.reserve(ws2)  # type: ignore[arg-type]
    # Third attempt — capacity exhausted
    assert not reg.reserve(ws3)  # type: ignore[arg-type]
    # ws3 was NOT added to reserved set
    assert ws3 not in reg._reserved_tokens  # type: ignore[operator]


async def test_rollback_restores_token():
    reg = ActiveTerminals(max_concurrent=2)
    ws = FakeWS()
    assert reg.reserve(ws)  # type: ignore[arg-type]
    assert reg._tokens.qsize() == 1
    reg.rollback(ws)  # type: ignore[arg-type]
    assert reg._tokens.qsize() == 2
    # Rollback is idempotent — second call is a no-op
    reg.rollback(ws)  # type: ignore[arg-type]
    assert reg._tokens.qsize() == 2


async def test_rollback_noop_without_reserve():
    reg = ActiveTerminals(max_concurrent=2)
    reg.rollback(FakeWS())  # type: ignore[arg-type]
    assert reg._tokens.qsize() == 2


async def test_unregister_is_idempotent():
    reg = ActiveTerminals(max_concurrent=1)
    ws = FakeWS()
    assert reg.reserve(ws)  # type: ignore[arg-type]
    reg.fulfill(ws, FakeSession())  # type: ignore[arg-type]
    assert reg._tokens.qsize() == 0
    reg.unregister(ws)  # type: ignore[arg-type]
    assert reg._tokens.qsize() == 1
    # Second unregister does nothing
    reg.unregister(ws)  # type: ignore[arg-type]
    assert reg._tokens.qsize() == 1


async def test_fulfill_without_reserve_raises():
    reg = ActiveTerminals(max_concurrent=1)
    with pytest.raises(RuntimeError, match="without"):
        reg.fulfill(FakeWS(), FakeSession())  # type: ignore[arg-type]


async def test_reserve_idempotent_for_same_ws():
    reg = ActiveTerminals(max_concurrent=2)
    ws = FakeWS()
    assert reg.reserve(ws)  # type: ignore[arg-type]
    # Re-reserving for the same ws should not consume a second token
    assert reg.reserve(ws)  # type: ignore[arg-type]
    assert reg._tokens.qsize() == 1


async def test_close_all_closes_every_session_and_refills():
    reg = ActiveTerminals(max_concurrent=3)
    ws1, ws2 = FakeWS(), FakeWS()
    s1, s2 = FakeSession("a"), FakeSession("b")
    assert reg.reserve(ws1)  # type: ignore[arg-type]
    assert reg.reserve(ws2)  # type: ignore[arg-type]
    reg.fulfill(ws1, s1)  # type: ignore[arg-type]
    reg.fulfill(ws2, s2)  # type: ignore[arg-type]

    await reg.close_all(reason="shutdown")

    assert s1.closed_reason == "shutdown"
    assert s2.closed_reason == "shutdown"
    assert reg.summary()["count"] == 0
    assert reg._tokens.qsize() == 3


async def test_capacity_restored_after_unregister_cycle():
    reg = ActiveTerminals(max_concurrent=1)
    for _ in range(5):
        ws = FakeWS()
        assert reg.reserve(ws)  # type: ignore[arg-type]
        reg.fulfill(ws, FakeSession())  # type: ignore[arg-type]
        reg.unregister(ws)  # type: ignore[arg-type]
        assert reg._tokens.qsize() == 1


async def test_summary_fields():
    reg = ActiveTerminals(max_concurrent=4)
    ws = FakeWS()
    assert reg.reserve(ws)  # type: ignore[arg-type]
    reg.fulfill(ws, FakeSession("scribe"))  # type: ignore[arg-type]
    s = reg.summary()
    assert s == {
        "count": 1,
        "available": 3,
        "max": 4,
        "sessions": [{"tmux_session": "scribe"}],
    }
