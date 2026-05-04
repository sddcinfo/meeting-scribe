"""Tests for the v36 cookie-session machinery: per-boot signing subkey,
session_id revocation, decode_verified_cookie helper, and the
register_admin_ws context manager.

Covers Plan A.4 of the unified-hotspot trust model.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from meeting_scribe.terminal.auth import (
    CookieSigner,
    decode_verified_cookie,
    derive_cookie_subkey,
    prune_revoked,
    register_admin_ws,
    revoke_session,
)


def test_per_boot_subkey_rotation_invalidates_old_cookies() -> None:
    """A cookie issued under one boot_session_id fails verify under a fresh one.

    This is the strong logout-all-on-restart guarantee from Plan §A.4 and
    risk R5: server restart rotates ``boot_session_id``; every previously
    issued cookie's HMAC no longer matches the new subkey.
    """
    master = b"master-admin-secret"
    boot_a = b"a" * 32
    boot_b = b"b" * 32
    subkey_a = derive_cookie_subkey(master, boot_a)
    subkey_b = derive_cookie_subkey(master, boot_b)
    assert subkey_a != subkey_b

    signer_a = CookieSigner(secret=subkey_a)
    signer_b = CookieSigner(secret=subkey_b)
    cookie = signer_a.issue()
    assert signer_a.verify(cookie)
    assert not signer_b.verify(cookie)


def test_decode_verified_cookie_returns_session_id() -> None:
    """The decode helper exposes session_id + issued_at to callers that
    need them (logout, WS registration). Bool path stays equivalent."""
    signer = CookieSigner(secret=b"deadbeef" * 4)
    now = 1_700_000_000.0
    cookie = signer.issue(now=now)

    ok, session_id, issued = decode_verified_cookie(signer, cookie, now=now)
    assert ok is True
    assert session_id is not None
    assert len(session_id) == 32  # 16 hex bytes
    assert issued == int(now)
    # Bool helper agrees.
    assert signer.verify(cookie, now=now) is True


def test_decode_verified_cookie_failure_returns_none_tuple() -> None:
    signer = CookieSigner(secret=b"x" * 16)
    ok, sid, issued = decode_verified_cookie(signer, "", now=1.0)
    assert ok is False
    assert sid is None
    assert issued is None
    ok, sid, issued = decode_verified_cookie(signer, "garbage", now=1.0)
    assert (ok, sid, issued) == (False, None, None)


def test_old_format_cookie_returns_false_no_raise() -> None:
    """Pre-cutover two-part cookies (``<issued>.<sig>``) verify as False
    without raising. Self-healing migration: user re-auths and gets a
    fresh new-format cookie.
    """
    signer = CookieSigner(secret=b"y" * 16)
    # An old-format cookie crafted by hand: just two parts.
    old_format = "1700000000.abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
    assert signer.verify(old_format) is False
    ok, sid, issued = decode_verified_cookie(signer, old_format)
    assert (ok, sid, issued) == (False, None, None)


def test_revoked_session_rejects_verify() -> None:
    """Adding the cookie's session_id to the revocation set causes
    subsequent verifies to fail even though the HMAC is still valid."""
    revoked: dict[str, float] = {}
    signer = CookieSigner(secret=b"k" * 16, is_revoked=lambda sid: sid in revoked)
    now = 1_700_000_000.0
    cookie = signer.issue(now=now)
    assert signer.verify(cookie, now=now) is True

    _, sid, issued = decode_verified_cookie(signer, cookie, now=now)
    assert sid is not None
    revoke_session(sid, expiry_epoch=issued + signer.max_age_seconds, revoked_sessions=revoked)
    assert signer.verify(cookie, now=now) is False


def test_revocation_only_targets_named_session() -> None:
    """Two independent admin sessions with distinct session_ids: revoking
    A leaves B verifying. This is the multi-admin correct-behavior bit
    from Plan §Tests test_logout_revokes_session.
    """
    revoked: dict[str, float] = {}
    signer = CookieSigner(secret=b"k" * 16, is_revoked=lambda sid: sid in revoked)
    now = 1_700_000_000.0
    cookie_x = signer.issue(now=now)
    cookie_y = signer.issue(now=now)
    _, sid_x, _ = decode_verified_cookie(signer, cookie_x, now=now)
    _, sid_y, _ = decode_verified_cookie(signer, cookie_y, now=now)
    assert sid_x != sid_y
    revoke_session(sid_x, expiry_epoch=now + 999, revoked_sessions=revoked)
    assert signer.verify(cookie_x, now=now) is False
    assert signer.verify(cookie_y, now=now) is True


def test_prune_revoked_drops_expired() -> None:
    """Entries whose expiry has passed are dropped on prune; non-expired
    stay. Lets the revocation dict stay bounded across the boot."""
    revoked: dict[str, float] = {
        "live_session": time.time() + 3600,
        "stale_session_a": time.time() - 10,
        "stale_session_b": time.time() - 1000,
    }
    pruned = prune_revoked(revoked)
    assert pruned == 2
    assert "live_session" in revoked
    assert "stale_session_a" not in revoked
    assert "stale_session_b" not in revoked


def test_explicit_session_id_passthrough() -> None:
    """Callers can supply a specific session_id (for tests, or for
    re-auth flows that want to bind a known nonce)."""
    signer = CookieSigner(secret=b"z" * 16)
    cookie = signer.issue(session_id="cafebabecafebabecafebabecafebabe")
    _, sid, _ = decode_verified_cookie(signer, cookie)
    assert sid == "cafebabecafebabecafebabecafebabe"


def test_register_admin_ws_context_manager() -> None:
    """The async ctx manager adds the WS to by_session on entry and
    removes it on exit. Empty buckets are popped so the dict stays sized
    to active sessions."""

    class FakeWS:
        def __init__(self, name: str) -> None:
            self.name = name

        def __hash__(self) -> int:
            return hash(self.name)

        def __eq__(self, other: object) -> bool:
            return isinstance(other, FakeWS) and self.name == other.name

    async def scenario() -> None:
        by_session: dict[str, set] = {}
        ws_a = FakeWS("a")
        ws_b = FakeWS("b")

        async with register_admin_ws("sid-1", ws_a, by_session):
            assert ws_a in by_session["sid-1"]
            async with register_admin_ws("sid-1", ws_b, by_session):
                assert {ws_a, ws_b} == by_session["sid-1"]
            # ws_b deregistered, ws_a still active under sid-1
            assert by_session["sid-1"] == {ws_a}
        # ws_a deregistered → bucket empty → key popped.
        assert "sid-1" not in by_session

    asyncio.run(scenario())


def test_register_admin_ws_cleans_up_on_exception() -> None:
    """Even on an exception inside the with-block, the WS is removed from
    the session map (no leaks across abnormal exit)."""

    class FakeWS:
        def __hash__(self) -> int:
            return id(self)

    async def scenario() -> None:
        by_session: dict[str, set] = {}
        ws = FakeWS()
        with pytest.raises(RuntimeError, match="boom"):
            async with register_admin_ws("sid-x", ws, by_session):
                raise RuntimeError("boom")
        assert "sid-x" not in by_session

    asyncio.run(scenario())


def test_cookie_session_id_is_unique_per_issue() -> None:
    """Calling issue() twice produces distinct session_ids — re-auth
    on the same browser doesn't recycle the prior session_id."""
    signer = CookieSigner(secret=b"u" * 16)
    seen = {decode_verified_cookie(signer, signer.issue())[1] for _ in range(50)}
    assert len(seen) == 50, seen
