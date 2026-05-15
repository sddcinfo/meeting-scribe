"""``_is_captive_acked`` returns True for admin/guest ipset members
even when the legacy ``_captive_acked`` set is empty (Phase H).

OS CNAs poll captive probes until they see Success — after Phase H,
ipset membership IS Success. Without that the CNA keeps popping the
portal even though the operator is fully authenticated.
"""

from __future__ import annotations

from meeting_scribe.server_support import captive_ack


class _Req:
    def __init__(self, host: str = "10.42.0.42") -> None:
        class _Client:
            pass

        c = _Client()
        c.host = host
        self.client = c


def test_unauthorized_ip_not_acked(monkeypatch) -> None:
    monkeypatch.setattr(
        "meeting_scribe.server_support.firewall_allowlist.is_admin", lambda ip: False
    )
    monkeypatch.setattr(
        "meeting_scribe.server_support.firewall_allowlist.is_guest", lambda ip: False
    )
    assert captive_ack._is_captive_acked(_Req("10.42.0.99")) is False


def test_admin_ipset_member_is_acked(monkeypatch) -> None:
    """Admin in the ipset → CNA dismisses (probes return Success)."""
    monkeypatch.setattr(
        "meeting_scribe.server_support.firewall_allowlist.is_admin", lambda ip: ip == "10.42.0.42"
    )
    monkeypatch.setattr(
        "meeting_scribe.server_support.firewall_allowlist.is_guest", lambda ip: False
    )
    assert captive_ack._is_captive_acked(_Req("10.42.0.42")) is True


def test_guest_ipset_member_is_acked(monkeypatch) -> None:
    """Guests also get Success — they don't reach WAN, but the CNA
    doesn't pop the portal at every connection check."""
    monkeypatch.setattr(
        "meeting_scribe.server_support.firewall_allowlist.is_admin", lambda ip: False
    )
    monkeypatch.setattr(
        "meeting_scribe.server_support.firewall_allowlist.is_guest", lambda ip: ip == "10.42.0.55"
    )
    assert captive_ack._is_captive_acked(_Req("10.42.0.55")) is True


def test_legacy_acked_set_still_honored(monkeypatch) -> None:
    """The pre-Phase-H ack set keeps working for clients that came in
    via the portal-page-visit path (no ipset entry yet)."""
    monkeypatch.setattr(
        "meeting_scribe.server_support.firewall_allowlist.is_admin", lambda ip: False
    )
    monkeypatch.setattr(
        "meeting_scribe.server_support.firewall_allowlist.is_guest", lambda ip: False
    )
    captive_ack._captive_acked.add("10.42.0.77")
    try:
        assert captive_ack._is_captive_acked(_Req("10.42.0.77")) is True
    finally:
        captive_ack._captive_acked.discard("10.42.0.77")


def test_no_client_returns_false(monkeypatch) -> None:
    """Defensive: a request with no client attribute (test/internal)
    must return False without crashing."""
    monkeypatch.setattr(
        "meeting_scribe.server_support.firewall_allowlist.is_admin", lambda ip: False
    )
    monkeypatch.setattr(
        "meeting_scribe.server_support.firewall_allowlist.is_guest", lambda ip: False
    )

    class _NoClientReq:
        client = None

    assert captive_ack._is_captive_acked(_NoClientReq()) is False


def test_firewall_allowlist_exception_falls_through(monkeypatch) -> None:
    """An ipset error MUST NOT crash the probe handler — it just falls
    through to the legacy ack-set check."""

    def _boom(ip):
        raise RuntimeError("ipset is on fire")

    monkeypatch.setattr("meeting_scribe.server_support.firewall_allowlist.is_admin", _boom)
    monkeypatch.setattr("meeting_scribe.server_support.firewall_allowlist.is_guest", _boom)
    # No exception, falls back to the ack-set; that's empty → False.
    captive_ack._captive_acked.discard("10.42.0.99")
    assert captive_ack._is_captive_acked(_Req("10.42.0.99")) is False
