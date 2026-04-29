"""Tests for LocalRunner, runner factory, port-80 automation, and
the unified captive-portal-80.py handler.

These verify the GB10-local path (no SSH required) and that port 80
is wired end-to-end so fresh deploys come up without manual commands.
"""

from __future__ import annotations

import http.client
import socket
import subprocess
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from meeting_scribe.infra.local import LocalRunner
from meeting_scribe.infra.runner import get_runner, is_local
from meeting_scribe.infra.ssh import SSHRunner

REPO_ROOT = Path(__file__).resolve().parent.parent
PORT80_SCRIPT = REPO_ROOT / "scripts" / "captive-portal-80.py"


class TestIsLocal:
    @pytest.mark.parametrize(
        "host",
        ["", "local", "localhost", "127.0.0.1", "::1", "0.0.0.0", None],
    )
    def test_obvious_local_hosts(self, host):
        assert is_local(host) is True

    def test_current_hostname(self):
        assert is_local(socket.gethostname()) is True

    def test_remote_ip(self):
        assert is_local("10.0.0.42") is False

    def test_env_var_force_local(self, monkeypatch):
        monkeypatch.setenv("SCRIBE_GB10_LOCAL", "1")
        assert is_local("10.0.0.42") is True


class TestGetRunner:
    def test_local_returns_local_runner(self):
        assert isinstance(get_runner("localhost"), LocalRunner)
        assert isinstance(get_runner("127.0.0.1"), LocalRunner)
        assert isinstance(get_runner(None), LocalRunner)

    def test_remote_returns_ssh_runner(self):
        assert isinstance(get_runner("10.0.0.42"), SSHRunner)


class TestLocalRunner:
    def test_run_captures_output(self):
        runner = LocalRunner()
        result = runner.run(["echo", "hello-local"], check=True)
        assert result.returncode == 0
        assert "hello-local" in result.stdout

    def test_is_reachable_always_true(self):
        assert LocalRunner().is_reachable() is True

    def test_docker_ps_no_ssh(self):
        runner = LocalRunner()
        with patch("meeting_scribe.infra.local.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="abc123\tscribe-asr\tUp 1 minute", stderr=""
            )
            out = runner.docker_ps(name_filter="scribe")

        assert "scribe-asr" in out
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "docker"
        assert "ssh" not in cmd

    def test_docker_stop_returns_bool(self):
        runner = LocalRunner()
        with patch("meeting_scribe.infra.local.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )
            assert runner.docker_stop("abc123") is True

            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="no such container"
            )
            assert runner.docker_stop("missing") is False


class TestGb10StatusNoHost:
    """`gb10 status` and `containers` must work without any arguments."""

    def test_gb10_status_accepts_no_host(self):
        from click.testing import CliRunner as ClickCliRunner

        from meeting_scribe.cli import cli

        with (
            patch("meeting_scribe.infra.containers.list_containers") as mock_list,
            patch("meeting_scribe.infra.health.check_all_services") as mock_health,
        ):
            mock_list.return_value = [
                {"id": "abc", "name": "scribe-asr", "status": "Up"},
            ]

            async def _fake_health(*args, **kwargs):
                return {}

            mock_health.side_effect = _fake_health

            result = ClickCliRunner().invoke(cli, ["gb10", "status"])

        assert result.exit_code == 0, result.output
        assert "scribe-asr" in result.output


class TestEnsurePort80Bind:
    """_ensure_port80_bind — the one helper that makes deploys automatic."""

    def test_already_granted_fast_path(self, monkeypatch):
        from meeting_scribe.cli import _common as cli_module

        fake_py = Path("/fake/python")

        def _fake_venv_python():
            return fake_py

        monkeypatch.setattr(cli_module, "_venv_python", _fake_venv_python)
        monkeypatch.setattr(Path, "exists", lambda self: True)
        monkeypatch.setattr(Path, "resolve", lambda self, strict=False: self)
        monkeypatch.setattr(cli_module.shutil, "which", lambda name: f"/usr/sbin/{name}")

        def _fake_subprocess_run(cmd, **kw):
            if cmd[0] == "getcap":
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout=f"{fake_py} cap_net_bind_service=ep\n", stderr=""
                )
            pytest.fail(f"unexpected command in fast path: {cmd}")

        monkeypatch.setattr(cli_module.subprocess, "run", _fake_subprocess_run)

        ok, detail = cli_module._ensure_port80_bind()
        assert ok is True
        assert str(fake_py) in detail

    def test_sudo_grant_succeeds(self, monkeypatch):
        from meeting_scribe.cli import _common as cli_module

        fake_py = Path("/fake/python")
        monkeypatch.setattr(cli_module, "_venv_python", lambda: fake_py)
        monkeypatch.setattr(Path, "exists", lambda self: True)
        monkeypatch.setattr(Path, "resolve", lambda self, strict=False: self)
        monkeypatch.setattr(cli_module.shutil, "which", lambda name: f"/usr/sbin/{name}")

        calls: list[list[str]] = []

        def _fake_subprocess_run(cmd, **kw):
            calls.append(list(cmd))
            if cmd[0] == "getcap":
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            if cmd[:3] == ["sudo", "-n", "setcap"]:
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            pytest.fail(f"unexpected command: {cmd}")

        monkeypatch.setattr(cli_module.subprocess, "run", _fake_subprocess_run)

        ok, _detail = cli_module._ensure_port80_bind()
        assert ok is True
        assert any(c[:3] == ["sudo", "-n", "setcap"] for c in calls)
        assert "cap_net_bind_service=+ep" in [a for c in calls for a in c]

    def test_sudo_grant_fails_returns_instructions(self, monkeypatch):
        from meeting_scribe.cli import _common as cli_module

        fake_py = Path("/fake/python")
        monkeypatch.setattr(cli_module, "_venv_python", lambda: fake_py)
        monkeypatch.setattr(Path, "exists", lambda self: True)
        monkeypatch.setattr(Path, "resolve", lambda self, strict=False: self)
        monkeypatch.setattr(cli_module.shutil, "which", lambda name: f"/usr/sbin/{name}")

        def _fake_subprocess_run(cmd, **kw):
            if cmd[0] == "getcap":
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            if cmd[:3] == ["sudo", "-n", "setcap"]:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=1, stdout="", stderr="password required"
                )
            pytest.fail(f"unexpected command: {cmd}")

        monkeypatch.setattr(cli_module.subprocess, "run", _fake_subprocess_run)

        ok, detail = cli_module._ensure_port80_bind()
        assert ok is False
        assert "sudo setcap" in detail
        assert "cap_net_bind_service=+ep" in detail


class TestCaptivePortal80:
    """The port-80 script: captive probes + Host-header-based redirect."""

    def _free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    @pytest.fixture
    def live_server(self, monkeypatch):
        """Spawn captive-portal-80.py on a free high port, yield that port."""
        port = self._free_port()

        # Run the script via the current interpreter on a user port, with the
        # module's hard-coded bind port monkey-patched via an inline shim.
        # We re-import the handler and run it ourselves to sidestep port 80.
        # Execute the script in a fresh module namespace
        import importlib.util
        from http.server import ThreadingHTTPServer

        spec = importlib.util.spec_from_file_location("_scribe_port80", PORT80_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        # Avoid the __main__ bind at import time
        source = PORT80_SCRIPT.read_text()
        source = source.replace(
            'if __name__ == "__main__":',
            "if False:",
        )
        code = compile(source, str(PORT80_SCRIPT), "exec")
        exec(code, module.__dict__)

        httpd = ThreadingHTTPServer(("127.0.0.1", port), module.Handler)
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            yield port
        finally:
            httpd.shutdown()
            httpd.server_close()

    def _get(self, port: int, path: str, host: str = "example.local"):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request("GET", path, headers={"Host": host})
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        return resp, body

    def test_generic_host_redirects_with_host_header(self, live_server):
        resp, _ = self._get(live_server, "/ui/thing?x=1", host="box42.local")
        assert resp.status == 302  # transient — clients don't cache once off hotspot
        assert resp.getheader("Location") == "https://box42.local:8080/ui/thing?x=1"

    def test_ios_hotspot_probe_returns_success(self, live_server):
        resp, body = self._get(live_server, "/hotspot-detect.html", host="captive.apple.com")
        assert resp.status == 200
        assert b"Success" in body

    def test_android_generate_204(self, live_server):
        resp, _ = self._get(live_server, "/generate_204", host="connectivitycheck.gstatic.com")
        assert resp.status == 204

    def test_rfc8910_captive_api(self, live_server):
        resp, body = self._get(live_server, "/api/captive")
        assert resp.status == 200
        import json as _json

        assert _json.loads(body) == {"captive": False}

    def test_missing_host_header_falls_back_to_ap_ip(self, live_server, monkeypatch):
        # Default AP_IP fallback is 10.42.0.1 in the script
        conn = http.client.HTTPConnection("127.0.0.1", live_server, timeout=2)
        conn.putrequest("GET", "/anything", skip_host=True, skip_accept_encoding=True)
        conn.endheaders()
        resp = conn.getresponse()
        loc = resp.getheader("Location")
        conn.close()
        assert resp.status == 302
        assert loc == "https://10.42.0.1:8080/anything"
