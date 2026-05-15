"""Authorize/logout hooks add/remove client IPs from the captive ipsets.

Admin authorize → ms-allowed-admins add.
Guest PIN     → ms-allowed-guests add.
Logout        → corresponding remove.
Failure paths must NOT mutate the ipset.
Allowlist failures must NOT break the HTTP handler.
"""

from __future__ import annotations

import pytest


class _IpsetSpy:
    """Records every call but doesn't actually run ipset."""

    def __init__(self) -> None:
        self.add_admin: list[str] = []
        self.remove_admin: list[str] = []
        self.add_guest: list[str] = []
        self.remove_guest: list[str] = []

    def install(self, monkeypatch) -> None:
        from meeting_scribe.server_support import firewall_allowlist

        async def _add_a(ip: str) -> bool:
            self.add_admin.append(ip)
            return True

        async def _rm_a(ip: str) -> bool:
            self.remove_admin.append(ip)
            return True

        async def _add_g(ip: str) -> bool:
            self.add_guest.append(ip)
            return True

        async def _rm_g(ip: str) -> bool:
            self.remove_guest.append(ip)
            return True

        monkeypatch.setattr(firewall_allowlist, "add_admin", _add_a)
        monkeypatch.setattr(firewall_allowlist, "remove_admin", _rm_a)
        monkeypatch.setattr(firewall_allowlist, "add_guest", _add_g)
        monkeypatch.setattr(firewall_allowlist, "remove_guest", _rm_g)


@pytest.fixture
def ipset_spy(monkeypatch) -> _IpsetSpy:
    spy = _IpsetSpy()
    spy.install(monkeypatch)
    return spy


# ─── Admin authorize ──────────────────────────────────────────


def test_admin_authorize_adds_caller_ip_to_admins(ipset_spy) -> None:
    """The successful authorize path mirrors the client IP into ipset."""
    from meeting_scribe.terminal.bootstrap import _captive_add_admin

    class _FakeReq:
        class client:
            host = "10.42.0.42"

    import asyncio

    asyncio.run(_captive_add_admin(_FakeReq()))
    assert ipset_spy.add_admin == ["10.42.0.42"]


def test_admin_logout_removes_caller_ip_from_admins(ipset_spy) -> None:
    from meeting_scribe.terminal.bootstrap import _captive_remove_admin

    class _FakeReq:
        class client:
            host = "10.42.0.42"

    import asyncio

    asyncio.run(_captive_remove_admin(_FakeReq()))
    assert ipset_spy.remove_admin == ["10.42.0.42"]


def test_admin_hook_swallows_allowlist_failure(monkeypatch, caplog) -> None:
    """A firewall_allowlist exception MUST NOT fail the authorize path —
    the cookie should still be issued, the GC tick reconciles later."""
    from meeting_scribe.server_support import firewall_allowlist
    from meeting_scribe.terminal.bootstrap import _captive_add_admin

    async def _explode(ip: str) -> bool:
        raise RuntimeError("ipset is on fire")

    monkeypatch.setattr(firewall_allowlist, "add_admin", _explode)

    class _FakeReq:
        class client:
            host = "10.42.0.42"

    import asyncio

    # Must not raise.
    with caplog.at_level("ERROR"):
        asyncio.run(_captive_add_admin(_FakeReq()))
    assert any("add_admin hook failed" in m for m in caplog.messages)


def test_admin_hook_skips_when_request_has_no_client(ipset_spy) -> None:
    """Some test/internal request objects have no .client. Don't crash."""
    from meeting_scribe.terminal.bootstrap import _captive_add_admin

    class _FakeReq:
        client = None

    import asyncio

    asyncio.run(_captive_add_admin(_FakeReq()))
    assert ipset_spy.add_admin == []


# ─── Guest PIN ────────────────────────────────────────────────


def test_guest_pin_adds_caller_ip_to_guests(ipset_spy) -> None:
    from meeting_scribe.routes.guest_auth import _captive_add_guest

    class _FakeReq:
        class client:
            host = "10.42.0.55"

    import asyncio

    asyncio.run(_captive_add_guest(_FakeReq()))
    assert ipset_spy.add_guest == ["10.42.0.55"]


def test_guest_logout_removes_caller_ip_from_guests(ipset_spy) -> None:
    from meeting_scribe.routes.guest_auth import _captive_remove_guest

    class _FakeReq:
        class client:
            host = "10.42.0.55"

    import asyncio

    asyncio.run(_captive_remove_guest(_FakeReq()))
    assert ipset_spy.remove_guest == ["10.42.0.55"]


def test_guest_hook_swallows_allowlist_failure(monkeypatch, caplog) -> None:
    from meeting_scribe.routes.guest_auth import _captive_add_guest
    from meeting_scribe.server_support import firewall_allowlist

    async def _explode(ip: str) -> bool:
        raise RuntimeError("ipset is on fire")

    monkeypatch.setattr(firewall_allowlist, "add_guest", _explode)

    class _FakeReq:
        class client:
            host = "10.42.0.55"

    import asyncio

    with caplog.at_level("ERROR"):
        asyncio.run(_captive_add_guest(_FakeReq()))
    assert any("add_guest hook failed" in m for m in caplog.messages)
