"""Tests for runtime/net.py:_serve_two_apps — the v36 unified-hotspot
listener orchestration.

Hardware-side validation (real IP_FREEBIND bind on 10.42.0.1) is
covered separately. Here we drive the orchestration with mock
``_NoSignalServer`` instances + locally-bound sockets to prove the
ordering contract: TLS server starts first, HTTP captive sub-app
starts only after TLS startup completes, both shut down together.
"""

from __future__ import annotations

import asyncio
import socket

import pytest

from meeting_scribe.runtime.net import _make_tcp_socket, _serve_two_apps


class _StubServer:
    """Mocks _NoSignalServer's interface enough for the orchestrator.

    Tracks call ordering so the test can assert TLS started before HTTP
    even attempted to start. Records both ``should_exit`` flips so the
    shutdown contract can be verified.
    """

    def __init__(self, name: str, *, startup_delay: float = 0.0) -> None:
        self.name = name
        self.started = False
        self.should_exit = False
        self.serve_called_at: float | None = None
        self._startup_delay = startup_delay

    async def serve(self, sockets) -> None:
        loop = asyncio.get_running_loop()
        self.serve_called_at = loop.time()
        if self._startup_delay > 0:
            await asyncio.sleep(self._startup_delay)
        self.started = True
        # Wait for shutdown signal.
        while not self.should_exit:
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_serve_two_apps_starts_tls_then_http() -> None:
    """HTTP captive sub-app does NOT start until TLS server reports
    ``started == True``."""
    tls = _StubServer("tls", startup_delay=0.05)
    http = _StubServer("http")
    tls_sock = _make_tcp_socket("127.0.0.1", 0)
    http_sock = _make_tcp_socket("127.0.0.1", 0)

    async def shutdown_after_a_moment() -> None:
        await asyncio.sleep(0.2)
        tls.should_exit = True
        http.should_exit = True

    shutdown = asyncio.create_task(shutdown_after_a_moment())
    try:
        await _serve_two_apps(
            tls_app_server=tls,
            tls_sockets=[tls_sock],
            http_app_server=http,
            http_sockets=[http_sock],
        )
    finally:
        shutdown.cancel()
        try:
            await shutdown
        except asyncio.CancelledError:
            pass
        tls_sock.close()
        http_sock.close()

    assert tls.serve_called_at is not None
    assert http.serve_called_at is not None
    # HTTP sub-app starts strictly after TLS server's startup completes.
    assert http.serve_called_at >= tls.serve_called_at + 0.04


@pytest.mark.asyncio
async def test_serve_two_apps_short_circuits_when_tls_dies_in_startup() -> None:
    """If TLS startup raises, HTTP sub-app is NOT started."""

    class _DyingTLS(_StubServer):
        async def serve(self, sockets) -> None:
            raise RuntimeError("tls boot failed")

    tls = _DyingTLS("tls")
    http = _StubServer("http")
    tls_sock = _make_tcp_socket("127.0.0.1", 0)
    http_sock = _make_tcp_socket("127.0.0.1", 0)

    try:
        with pytest.raises(RuntimeError, match="tls boot failed"):
            await _serve_two_apps(
                tls_app_server=tls,
                tls_sockets=[tls_sock],
                http_app_server=http,
                http_sockets=[http_sock],
            )
    finally:
        tls_sock.close()
        http_sock.close()
    # HTTP server never had its serve() invoked.
    assert http.serve_called_at is None
