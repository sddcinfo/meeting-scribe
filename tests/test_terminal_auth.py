"""Tests for admin secret, signed cookies, and single-use tickets."""

from __future__ import annotations

import asyncio
import time

import pytest

from meeting_scribe.terminal.auth import AdminSecretStore, CookieSigner, TicketStore

# ── AdminSecretStore ─────────────────────────────────────────────


def test_secret_auto_created_with_mode_600(tmp_path):
    path = tmp_path / "nested" / "admin-secret"
    store = AdminSecretStore.load_or_create(path)
    assert path.exists()
    assert len(store.secret) >= 32
    mode = path.stat().st_mode & 0o777
    assert mode == 0o600, f"expected 0o600 got {oct(mode)}"


def test_secret_is_persisted(tmp_path):
    path = tmp_path / "admin-secret"
    a = AdminSecretStore.load_or_create(path)
    b = AdminSecretStore.load_or_create(path)
    assert a.secret == b.secret


def test_secret_rejects_world_readable(tmp_path):
    path = tmp_path / "admin-secret"
    path.write_bytes(b"deadbeef")
    path.chmod(0o644)
    with pytest.raises(RuntimeError, match="mode"):
        AdminSecretStore.load_or_create(path)


def test_secret_rejects_group_readable(tmp_path):
    path = tmp_path / "admin-secret"
    path.write_bytes(b"deadbeef")
    path.chmod(0o640)
    with pytest.raises(RuntimeError, match="mode"):
        AdminSecretStore.load_or_create(path)


def test_secret_allows_owner_read_only(tmp_path):
    path = tmp_path / "admin-secret"
    path.write_bytes(b"deadbeef")
    path.chmod(0o400)
    store = AdminSecretStore.load_or_create(path)
    assert store.secret == b"deadbeef"


def test_secret_empty_refused(tmp_path):
    path = tmp_path / "admin-secret"
    path.touch(mode=0o600)
    with pytest.raises(RuntimeError, match="empty"):
        AdminSecretStore.load_or_create(path)


def test_verify_rejects_wrong_secret(tmp_path):
    store = AdminSecretStore.load_or_create(tmp_path / "admin-secret")
    assert store.verify(store.secret.decode())
    assert not store.verify("")
    assert not store.verify("wrong")
    # Near-miss: one byte changed
    wrong = store.secret.decode()
    wrong = wrong[:-1] + ("0" if wrong[-1] != "0" else "1")
    assert not store.verify(wrong)


def test_verify_is_constant_time(tmp_path):
    # Not a timing-attack test — just validates we're using hmac.compare_digest.
    store = AdminSecretStore.load_or_create(tmp_path / "admin-secret")
    # A completely different-length string must still return a bool, not raise.
    assert store.verify("x") is False
    assert store.verify(store.secret.decode() + "extra") is False


def test_env_override_is_respected(tmp_path, monkeypatch):
    env_path = tmp_path / "env-secret"
    monkeypatch.setenv("SCRIBE_ADMIN_SECRET_FILE", str(env_path))
    store = AdminSecretStore.load_or_create()
    assert store.path == env_path
    assert env_path.exists()


# ── CookieSigner ─────────────────────────────────────────────────


def test_cookie_roundtrip():
    s = CookieSigner(secret=b"test-secret")
    token = s.issue()
    assert s.verify(token)


def test_cookie_tamper_detected():
    s = CookieSigner(secret=b"test-secret")
    token = s.issue()
    issued, sig = token.split(".", 1)
    # Flip one char of the sig
    bad = issued + "." + ("0" if sig[0] != "0" else "1") + sig[1:]
    assert not s.verify(bad)


def test_cookie_wrong_secret_rejected():
    a = CookieSigner(secret=b"secret-a")
    b = CookieSigner(secret=b"secret-b")
    token = a.issue()
    assert not b.verify(token)


def test_cookie_expiry_enforced():
    s = CookieSigner(secret=b"test-secret", max_age_seconds=60)
    past = time.time() - 120
    token = s.issue(now=past)
    assert not s.verify(token)


def test_cookie_future_issued_rejected():
    s = CookieSigner(secret=b"test-secret", max_age_seconds=3600)
    future = time.time() + 3600
    token = s.issue(now=future)
    # Far-future 'issued_at' must be rejected (mild clock-skew allowance only)
    assert not s.verify(token)


def test_cookie_malformed_rejected():
    s = CookieSigner(secret=b"test-secret")
    assert not s.verify(None)
    assert not s.verify("")
    assert not s.verify("no-dot-token")
    assert not s.verify("notanumber.deadbeef")


def test_cookie_bounded_by_max_age():
    s = CookieSigner(secret=b"test-secret", max_age_seconds=10)
    now = time.time()
    token = s.issue(now=now)
    # Within max-age
    assert s.verify(token, now=now + 5)
    # Past max-age
    assert not s.verify(token, now=now + 20)


# ── TicketStore ──────────────────────────────────────────────────


async def test_ticket_single_use():
    store = TicketStore(secret=b"shh")
    t = await store.mint()
    assert await store.consume(t) is True
    assert await store.consume(t) is False


async def test_ticket_expiry():
    store = TicketStore(secret=b"shh", ttl_seconds=0.05)
    t = await store.mint()
    await asyncio.sleep(0.1)
    assert await store.consume(t) is False


async def test_ticket_tampered_hmac_rejected():
    store = TicketStore(secret=b"shh")
    t = await store.mint()
    random_half, _, sig = t.partition(".")
    flipped = ("0" if sig[0] != "0" else "1") + sig[1:]
    bad = f"{random_half}.{flipped}"
    assert await store.consume(bad) is False
    # Original is still valid — tampered consumption didn't remove it.
    assert await store.consume(t) is True


async def test_ticket_unknown_rejected():
    store = TicketStore(secret=b"shh")
    fake = "a" * 64 + "." + "b" * 64
    assert await store.consume(fake) is False


async def test_ticket_malformed_rejected():
    store = TicketStore(secret=b"shh")
    for bad in ["", "no-dot", ".only-dot", "only-dot.", None]:  # type: ignore[list-item]
        assert await store.consume(bad) is False  # type: ignore[arg-type]


async def test_ticket_sweep_drops_expired():
    store = TicketStore(secret=b"shh", ttl_seconds=0.02)
    store._sweep_interval_seconds = 0.05  # type: ignore[attr-defined]
    for _ in range(10):
        await store.mint()
    assert store._size() == 10
    task = asyncio.create_task(store.sweep())
    try:
        # Wait for TTL + one sweep cycle
        await asyncio.sleep(0.2)
        assert store._size() == 0
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def test_ticket_mint_uniqueness():
    store = TicketStore(secret=b"shh")
    tickets = [await store.mint() for _ in range(100)]
    assert len(set(tickets)) == 100
