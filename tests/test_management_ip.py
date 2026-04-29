"""Tests for management IP detection cascade and LAN recovery task."""

from __future__ import annotations

import asyncio
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from meeting_scribe.runtime.net import (
    _detect_management_ip,
    _detect_management_ip_via_nm,
    _wait_for_management_ip,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _route_success(ip: str = "192.168.8.153") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["ip", "-4", "route", "get", "1.1.1.1"],
        returncode=0,
        stdout=f"1.1.1.1 via 192.168.8.1 dev enP7s7 src {ip} uid 1000\n    cache",
    )


def _route_failure() -> subprocess.CalledProcessError:
    return subprocess.CalledProcessError(
        returncode=2,
        cmd=["ip", "-4", "route", "get", "1.1.1.1"],
    )


def _nm_connections(lines: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["nmcli", "-t", "-f", "TYPE,DEVICE", "connection", "show", "--active"],
        returncode=0,
        stdout=lines,
    )


def _nm_device_ip(ip: str, prefix: int = 24) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["nmcli", "-t", "-f", "IP4.ADDRESS", "device", "show", "enP7s7"],
        returncode=0,
        stdout=f"IP4.ADDRESS[1]:{ip}/{prefix}\n",
    )


# ---------------------------------------------------------------------------
# Detection cascade tests
# ---------------------------------------------------------------------------


class TestDetectManagementIp:
    """Tests for the _detect_management_ip() three-tier cascade."""

    def test_normal_route_success(self):
        """Tier 2: ip route get succeeds on first attempt."""
        with patch(
            "meeting_scribe.runtime.net.subprocess.run", return_value=_route_success("10.0.0.5")
        ):
            assert _detect_management_ip() == "10.0.0.5"

    def test_route_fails_nm_single_ethernet(self, monkeypatch):
        """Tier 3: route fails, NM finds single ethernet connection."""
        monkeypatch.setenv("SCRIBE_MGMT_IP_WAIT", "1")
        monkeypatch.delenv("SCRIBE_MANAGEMENT_IP", raising=False)

        def fake_run(cmd, **kw):
            if cmd[0] == "ip":
                raise _route_failure()
            if cmd[0] == "nmcli":
                if "connection" in cmd:
                    return _nm_connections(
                        "802-3-ethernet:enP7s7\nbridge:br-af71855f7b53\nloopback:lo\n"
                    )
                if "device" in cmd:
                    return _nm_device_ip("192.168.1.100")
            raise ValueError(f"unexpected command: {cmd}")

        with patch("meeting_scribe.runtime.net.subprocess.run", side_effect=fake_run):
            assert _detect_management_ip() == "192.168.1.100"

    def test_route_fails_nm_no_ethernet(self, monkeypatch):
        """Tier 3 → Tier 4: only bridge/loopback in NM → localhost fallback."""
        monkeypatch.setenv("SCRIBE_MGMT_IP_WAIT", "1")
        monkeypatch.delenv("SCRIBE_MANAGEMENT_IP", raising=False)

        def fake_run(cmd, **kw):
            if cmd[0] == "ip":
                raise _route_failure()
            if cmd[0] == "nmcli":
                return _nm_connections("bridge:docker0\nloopback:lo\n")
            raise ValueError(f"unexpected command: {cmd}")

        with patch("meeting_scribe.runtime.net.subprocess.run", side_effect=fake_run):
            assert _detect_management_ip() == "127.0.0.1"

    def test_route_fails_nm_multiple_ethernet(self, monkeypatch):
        """Tier 3 → Tier 4: multiple ethernet connections → refuse to guess."""
        monkeypatch.setenv("SCRIBE_MGMT_IP_WAIT", "1")
        monkeypatch.delenv("SCRIBE_MANAGEMENT_IP", raising=False)

        def fake_run(cmd, **kw):
            if cmd[0] == "ip":
                raise _route_failure()
            if cmd[0] == "nmcli":
                return _nm_connections("802-3-ethernet:enP7s7\n802-3-ethernet:enP8s8\n")
            raise ValueError(f"unexpected command: {cmd}")

        with patch("meeting_scribe.runtime.net.subprocess.run", side_effect=fake_run):
            assert _detect_management_ip() == "127.0.0.1"

    def test_route_fails_nm_not_available(self, monkeypatch):
        """Tier 3 → Tier 4: nmcli not installed → localhost fallback."""
        monkeypatch.setenv("SCRIBE_MGMT_IP_WAIT", "1")
        monkeypatch.delenv("SCRIBE_MANAGEMENT_IP", raising=False)

        def fake_run(cmd, **kw):
            if cmd[0] == "ip":
                raise _route_failure()
            if cmd[0] == "nmcli":
                raise FileNotFoundError("nmcli not found")
            raise ValueError(f"unexpected command: {cmd}")

        with patch("meeting_scribe.runtime.net.subprocess.run", side_effect=fake_run):
            assert _detect_management_ip() == "127.0.0.1"

    def test_ip_command_not_found(self, monkeypatch):
        """Both ip and nmcli missing → localhost fallback."""
        monkeypatch.setenv("SCRIBE_MGMT_IP_WAIT", "1")
        monkeypatch.delenv("SCRIBE_MANAGEMENT_IP", raising=False)

        with patch(
            "meeting_scribe.runtime.net.subprocess.run",
            side_effect=FileNotFoundError("not found"),
        ):
            assert _detect_management_ip() == "127.0.0.1"

    def test_env_override(self, monkeypatch):
        """Tier 1: SCRIBE_MANAGEMENT_IP override takes precedence."""
        monkeypatch.setenv("SCRIBE_MANAGEMENT_IP", "10.99.99.1")
        assert _detect_management_ip() == "10.99.99.1"

    def test_budget_exhaustion(self, monkeypatch):
        """SCRIBE_MGMT_IP_WAIT=1 causes single attempt then fallthrough."""
        monkeypatch.setenv("SCRIBE_MGMT_IP_WAIT", "1")
        monkeypatch.delenv("SCRIBE_MANAGEMENT_IP", raising=False)
        call_count = 0

        def fake_run(cmd, **kw):
            nonlocal call_count
            if cmd[0] == "ip":
                call_count += 1
                raise _route_failure()
            # NM also fails.
            raise FileNotFoundError("not found")

        with patch("meeting_scribe.runtime.net.subprocess.run", side_effect=fake_run):
            result = _detect_management_ip()

        assert result == "127.0.0.1"
        assert call_count == 1


# ---------------------------------------------------------------------------
# NM helper tests
# ---------------------------------------------------------------------------


class TestDetectManagementIpViaNm:
    """Tests for _detect_management_ip_via_nm() in isolation."""

    def test_single_ethernet_with_ip(self):
        def fake_run(cmd, **kw):
            if "connection" in cmd:
                return _nm_connections("802-3-ethernet:eth0\n")
            if "device" in cmd:
                return _nm_device_ip("10.0.0.42")
            raise ValueError(f"unexpected: {cmd}")

        with patch("meeting_scribe.runtime.net.subprocess.run", side_effect=fake_run):
            assert _detect_management_ip_via_nm() == "10.0.0.42"

    def test_no_ethernet(self):
        with patch(
            "meeting_scribe.runtime.net.subprocess.run",
            return_value=_nm_connections("bridge:docker0\nloopback:lo\n"),
        ):
            assert _detect_management_ip_via_nm() is None

    def test_ethernet_no_ip(self):
        def fake_run(cmd, **kw):
            if "connection" in cmd:
                return _nm_connections("802-3-ethernet:eth0\n")
            if "device" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=0,
                    stdout="",
                )
            raise ValueError(f"unexpected: {cmd}")

        with patch("meeting_scribe.runtime.net.subprocess.run", side_effect=fake_run):
            assert _detect_management_ip_via_nm() is None


# ---------------------------------------------------------------------------
# Recovery task tests
# ---------------------------------------------------------------------------


def _mock_admin_server() -> MagicMock:
    """Create a mock admin_server with the attributes _wait_for_management_ip needs."""
    srv = MagicMock()
    srv.servers = []
    srv.config = MagicMock()
    srv.config.http_protocol_class = MagicMock
    srv.config.ssl = None
    srv.config.backlog = 128
    srv.server_state = MagicMock()
    srv.lifespan = MagicMock()
    srv.lifespan.state = {}
    return srv


class TestWaitForManagementIp:
    """Tests for the async _wait_for_management_ip() recovery task.

    The recovery task calls ``_detect_quick`` via ``run_in_executor`` and
    then ``loop.create_server`` to hot-add the socket. We mock both the
    detection function and ``loop.create_server`` while letting the real
    event loop handle ``run_in_executor`` (which just runs the callable
    in the default thread pool).
    """

    @pytest.mark.asyncio
    async def test_recovery_succeeds(self, monkeypatch):
        """Task detects IP on second poll and hot-adds a server."""
        monkeypatch.delenv("SCRIBE_MANAGEMENT_IP", raising=False)

        poll_count = 0

        def fake_detect():
            nonlocal poll_count
            poll_count += 1
            if poll_count >= 2:
                return "192.168.8.200"
            return "127.0.0.1"

        admin_server = _mock_admin_server()
        mock_tcp_sock = MagicMock()
        mock_async_server = MagicMock()

        # Patch the real loop's create_server to return our mock.
        real_loop = asyncio.get_running_loop()
        orig_create_server = real_loop.create_server

        async def fake_create_server(*args, **kwargs):
            return mock_async_server

        with (
            patch("meeting_scribe.runtime.net._detect_management_ip", side_effect=fake_detect),
            patch("meeting_scribe.runtime.net._make_tcp_socket", return_value=mock_tcp_sock),
            patch.object(real_loop, "create_server", side_effect=fake_create_server),
        ):
            await _wait_for_management_ip(admin_server, port=8080, poll_interval=0)

        assert len(admin_server.servers) == 1
        assert admin_server.servers[0] is mock_async_server
        assert poll_count == 2

    @pytest.mark.asyncio
    async def test_shutdown_before_recovery(self, monkeypatch):
        """Task is cancelled cleanly before IP appears."""
        monkeypatch.delenv("SCRIBE_MANAGEMENT_IP", raising=False)

        admin_server = _mock_admin_server()

        with patch(
            "meeting_scribe.runtime.net._detect_management_ip",
            return_value="127.0.0.1",
        ):
            task = asyncio.create_task(
                _wait_for_management_ip(admin_server, port=8080, poll_interval=0)
            )
            # Let it poll once.
            await asyncio.sleep(0.05)
            task.cancel()
            await task  # Should not raise.

        assert len(admin_server.servers) == 0

    @pytest.mark.asyncio
    async def test_one_shot_behavior(self, monkeypatch):
        """Task exits after first successful recovery (doesn't keep polling)."""
        monkeypatch.delenv("SCRIBE_MANAGEMENT_IP", raising=False)

        detect_count = 0

        def fake_detect():
            nonlocal detect_count
            detect_count += 1
            return "192.168.1.1"

        admin_server = _mock_admin_server()

        real_loop = asyncio.get_running_loop()

        async def fake_create_server(*args, **kwargs):
            return MagicMock()

        with (
            patch("meeting_scribe.runtime.net._detect_management_ip", side_effect=fake_detect),
            patch("meeting_scribe.runtime.net._make_tcp_socket", return_value=MagicMock()),
            patch.object(real_loop, "create_server", side_effect=fake_create_server),
        ):
            await _wait_for_management_ip(admin_server, port=8080, poll_interval=0)

        assert detect_count == 1
        assert len(admin_server.servers) == 1

    @pytest.mark.asyncio
    async def test_recovery_bind_failure_retries(self, monkeypatch):
        """Socket bind failure logs warning and retries on next cycle."""
        monkeypatch.delenv("SCRIBE_MANAGEMENT_IP", raising=False)

        detect_count = 0

        def fake_detect():
            nonlocal detect_count
            detect_count += 1
            return "192.168.1.1"

        bind_count = 0

        def fake_make_socket(host, port):
            nonlocal bind_count
            bind_count += 1
            if bind_count == 1:
                raise OSError("Address already in use")
            return MagicMock()

        admin_server = _mock_admin_server()

        real_loop = asyncio.get_running_loop()

        async def fake_create_server(*args, **kwargs):
            return MagicMock()

        with (
            patch("meeting_scribe.runtime.net._detect_management_ip", side_effect=fake_detect),
            patch("meeting_scribe.runtime.net._make_tcp_socket", side_effect=fake_make_socket),
            patch.object(real_loop, "create_server", side_effect=fake_create_server),
        ):
            await _wait_for_management_ip(admin_server, port=8080, poll_interval=0)

        # First attempt: detect OK, bind fails → retry.
        # Second attempt: detect OK, bind OK → success.
        assert detect_count == 2
        assert bind_count == 2
        assert len(admin_server.servers) == 1
