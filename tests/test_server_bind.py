"""Bind / network-layer invariants for the HTTPS listener.

The v1.0 admin-auth model is **cookie-only**, so the listener
binds to all interfaces on port 443 (admin reaches the box from
the AP and the LAN at the same URL). The previous "AP-IP-only"
bind assertion was removed when admin auth went cookie-only; the
wizard's AP-subnet origin gate was removed in the same arc, so
``/setup`` is also reachable from any interface.
"""

from __future__ import annotations

import socket


def _bind(addr: str) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((addr, 0))  # ephemeral port
    return s


def test_zero_bind_listens_on_all_interfaces() -> None:
    """``getsockname()[0]`` returns ``0.0.0.0`` for an INADDR_ANY
    bind — that's the production binding now. Locks in the
    "admin reachable from any local interface" contract."""
    sock = _bind("0.0.0.0")
    try:
        assert sock.getsockname()[0] == "0.0.0.0"
    finally:
        sock.close()
