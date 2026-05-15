"""Unit tests for ``meeting-scribe validate --customer-flow``.

Each phase is tested in isolation with the host-state probe stubbed
(systemctl, rfkill, nmcli, /etc/sudoers.d) and the HTTP probes
mocked at the ``httpx.AsyncClient`` boundary. The point is to verify
the *gates* — what each phase considers a pass or fail — not to
exercise a real install (the live device is the only place that
actually matters for that, and is what the validator is designed to
be run on).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meeting_scribe import validate_customer


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(
        args=["dummy"], returncode=returncode, stdout=stdout, stderr=stderr
    )


# ── _phase_systemd_unit ────────────────────────────────────────────


class TestSystemdUnit:
    @patch("meeting_scribe.validate_customer.shutil.which", return_value=None)
    def test_skip_when_systemctl_missing(self, _which: MagicMock):
        result = validate_customer._phase_systemd_unit()
        assert result.status == "skip"
        assert "systemctl" in result.detail

    @patch("meeting_scribe.validate_customer.subprocess.run")
    @patch("meeting_scribe.validate_customer.shutil.which", return_value="/bin/systemctl")
    def test_pass_when_active(self, _which: MagicMock, mock_run: MagicMock):
        mock_run.return_value = _completed(stdout="active\n")
        result = validate_customer._phase_systemd_unit()
        assert result.status == "pass"
        assert result.metrics["state"] == "active"

    @patch("meeting_scribe.validate_customer.subprocess.run")
    @patch("meeting_scribe.validate_customer.shutil.which", return_value="/bin/systemctl")
    def test_fail_when_inactive(self, _which: MagicMock, mock_run: MagicMock):
        mock_run.return_value = _completed(stdout="inactive\n", returncode=3)
        result = validate_customer._phase_systemd_unit()
        assert result.status == "fail"
        assert "install-service" in result.detail


# ── _phase_wifi_radio ──────────────────────────────────────────────


class TestWifiRadio:
    @patch("meeting_scribe.validate_customer.subprocess.run")
    @patch("meeting_scribe.validate_customer.shutil.which", side_effect=lambda b: f"/bin/{b}")
    def test_pass_when_both_clear(self, _which: MagicMock, mock_run: MagicMock):
        def fake(cmd, **_kw):
            if cmd[0] == "rfkill":
                return _completed(stdout="0: phy0: Wireless LAN\n\tSoft blocked: no\n")
            if cmd[0] == "nmcli":
                return _completed(stdout="enabled\n")
            return _completed()

        mock_run.side_effect = fake
        result = validate_customer._phase_wifi_radio()
        assert result.status == "pass"

    @patch("meeting_scribe.validate_customer.subprocess.run")
    @patch("meeting_scribe.validate_customer.shutil.which", side_effect=lambda b: f"/bin/{b}")
    def test_fail_when_rfkill_blocked(self, _which: MagicMock, mock_run: MagicMock):
        def fake(cmd, **_kw):
            if cmd[0] == "rfkill":
                return _completed(stdout="0: phy0: Wireless LAN\n\tSoft blocked: yes\n")
            if cmd[0] == "nmcli":
                return _completed(stdout="enabled\n")
            return _completed()

        mock_run.side_effect = fake
        result = validate_customer._phase_wifi_radio()
        assert result.status == "fail"
        assert "soft-blocked" in result.detail

    @patch("meeting_scribe.validate_customer.subprocess.run")
    @patch("meeting_scribe.validate_customer.shutil.which", side_effect=lambda b: f"/bin/{b}")
    def test_fail_when_nmcli_disabled(self, _which: MagicMock, mock_run: MagicMock):
        def fake(cmd, **_kw):
            if cmd[0] == "rfkill":
                return _completed(stdout="0: phy0: Wireless LAN\n\tSoft blocked: no\n")
            if cmd[0] == "nmcli":
                return _completed(stdout="disabled\n")
            return _completed()

        mock_run.side_effect = fake
        result = validate_customer._phase_wifi_radio()
        assert result.status == "fail"
        assert "nmcli radio wifi=disabled" in result.detail


# ── _phase_sudoers_d ──────────────────────────────────────────────


class TestSudoersD:
    def test_fail_when_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        # Redirect the path constant: the function reads a hard-coded
        # /etc/sudoers.d/meeting-scribe. We use a local path that
        # doesn't exist by overriding Path() resolution via monkeypatch.
        monkeypatch.setattr(validate_customer, "Path", lambda *_a, **_kw: tmp_path / "missing")
        result = validate_customer._phase_sudoers_d()
        assert result.status == "fail"
        assert "missing" in result.detail.lower()

    def test_pass_when_present(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        present = tmp_path / "sudoers"
        present.write_text("dummy")
        present.chmod(0o440)
        monkeypatch.setattr(validate_customer, "Path", lambda *_a, **_kw: present)
        result = validate_customer._phase_sudoers_d()
        assert result.status == "pass"


# ── _phase_multipart_dep ───────────────────────────────────────────


class TestMultipartDep:
    def test_pass_when_importable(self):
        # python-multipart is in the runtime deps so this should
        # always succeed in the dev tree. The test exists as an
        # explicit invariant — if it ever fails locally, the dep
        # has drifted out of pyproject.
        result = validate_customer._phase_multipart_dep()
        assert result.status == "pass"


# ── _phase_status_latency ──────────────────────────────────────────


class _FakeAsyncResponse:
    def __init__(self, status_code: int = 200, text: str = ""):
        self.status_code = status_code
        self.text = text


class _FakeAsyncClient:
    """Minimal async-context-manager wrapper that records calls and
    returns canned responses."""

    def __init__(self, get_responses: list[_FakeAsyncResponse] | None = None):
        self._gets = list(get_responses or [])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return None

    async def get(self, *_a, **_kw):
        if self._gets:
            return self._gets.pop(0)
        return _FakeAsyncResponse()

    async def post(self, *_a, **_kw):
        return _FakeAsyncResponse(200, '{"meeting_id":"fake"}')


class TestStatusLatency:
    def test_pass_under_budget(self, monkeypatch: pytest.MonkeyPatch):
        from meeting_scribe.config import ServerConfig

        config = ServerConfig.from_env()
        client = _FakeAsyncClient(get_responses=[_FakeAsyncResponse(200, "{}") for _ in range(3)])

        def fake_async_client(**_kw):
            return client

        monkeypatch.setattr(validate_customer.httpx, "AsyncClient", fake_async_client)

        import asyncio

        result = asyncio.run(validate_customer._phase_status_latency(config))
        assert result.status == "pass"
        assert "p95_ms" in result.metrics

    def test_fail_when_status_returns_500(self, monkeypatch: pytest.MonkeyPatch):
        from meeting_scribe.config import ServerConfig

        config = ServerConfig.from_env()
        client = _FakeAsyncClient(get_responses=[_FakeAsyncResponse(500, "internal server error")])

        def fake_async_client(**_kw):
            return client

        monkeypatch.setattr(validate_customer.httpx, "AsyncClient", fake_async_client)

        import asyncio

        result = asyncio.run(validate_customer._phase_status_latency(config))
        assert result.status == "fail"
        assert "HTTP 500" in result.detail


# ── _phase_slides_upload ───────────────────────────────────────────


class TestSlidesUpload:
    def test_skip_when_admin_secret_unreadable(self, monkeypatch: pytest.MonkeyPatch):
        from meeting_scribe.config import ServerConfig

        monkeypatch.setattr(validate_customer, "_signed_admin_cookie", lambda: None)

        import asyncio

        result = asyncio.run(validate_customer._phase_slides_upload(ServerConfig.from_env()))
        assert result.status == "skip"

    def test_pass_when_upload_returns_non_500(self, monkeypatch: pytest.MonkeyPatch):
        """Whatever the route does after multipart parsing is fine —
        we only want to fail on the python-multipart assertion."""
        from meeting_scribe.config import ServerConfig

        monkeypatch.setattr(validate_customer, "_signed_admin_cookie", lambda: ("k", "v"))

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_a):
                return None

            async def post(self, url: str, **_kw):
                if url.endswith("/api/meeting/start"):
                    resp = _FakeAsyncResponse(200, '{"meeting_id":"abc"}')
                    resp.json = lambda: {"meeting_id": "abc"}  # type: ignore[attr-defined]
                    return resp
                if "slides/upload" in url:
                    return _FakeAsyncResponse(503, "slide processing unavailable")
                if url.endswith("/api/meeting/cancel"):
                    return _FakeAsyncResponse(200, "{}")
                return _FakeAsyncResponse(404)

        monkeypatch.setattr(validate_customer.httpx, "AsyncClient", lambda **_kw: _Client())

        import asyncio

        result = asyncio.run(validate_customer._phase_slides_upload(ServerConfig.from_env()))
        assert result.status == "pass"
        assert result.metrics["upload_status"] == 503

    def test_fail_when_upload_hits_multipart_assertion(self, monkeypatch: pytest.MonkeyPatch):
        from meeting_scribe.config import ServerConfig

        monkeypatch.setattr(validate_customer, "_signed_admin_cookie", lambda: ("k", "v"))

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_a):
                return None

            async def post(self, url: str, **_kw):
                if url.endswith("/api/meeting/start"):
                    resp = _FakeAsyncResponse(200, '{"meeting_id":"abc"}')
                    resp.json = lambda: {"meeting_id": "abc"}  # type: ignore[attr-defined]
                    return resp
                if "slides/upload" in url:
                    return _FakeAsyncResponse(
                        500,
                        "AssertionError: The python-multipart library must be installed",
                    )
                return _FakeAsyncResponse(200, "{}")

        monkeypatch.setattr(validate_customer.httpx, "AsyncClient", lambda **_kw: _Client())

        import asyncio

        result = asyncio.run(validate_customer._phase_slides_upload(ServerConfig.from_env()))
        assert result.status == "fail"
        assert "python-multipart" in result.detail


# ── _phase_meeting_qr ──────────────────────────────────────────────


class TestMeetingQR:
    def test_skip_when_admin_secret_unreadable(self, monkeypatch: pytest.MonkeyPatch):
        from meeting_scribe.config import ServerConfig

        monkeypatch.setattr(validate_customer, "_signed_admin_cookie", lambda: None)
        import asyncio

        result = asyncio.run(validate_customer._phase_meeting_qr(ServerConfig.from_env()))
        assert result.status == "skip"

    def test_pass_when_qr_returns(self, monkeypatch: pytest.MonkeyPatch):
        from meeting_scribe.config import ServerConfig

        monkeypatch.setattr(validate_customer, "_signed_admin_cookie", lambda: ("k", "v"))
        # Tighten timeout so the test doesn't actually wait 60s.
        monkeypatch.setattr(validate_customer, "_AP_WAIT_S", 1.0)

        class _Client:
            def __init__(self):
                self._wifi_calls = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_a):
                return None

            async def post(self, *_a, **_kw):
                resp = _FakeAsyncResponse(200, '{"meeting_id":"x"}')
                resp.json = lambda: {"meeting_id": "x"}  # type: ignore[attr-defined]
                return resp

            async def get(self, url: str, **_kw):
                self._wifi_calls += 1
                # First call: hotspot still warming up. Second call:
                # QR ready.
                if self._wifi_calls == 1:
                    return _FakeAsyncResponse(503, '{"error":"Hotspot not active"}')
                resp = _FakeAsyncResponse(
                    200,
                    '{"ssid":"Dell Demo XXXX","password":"PASS","wifi_qr_svg":"<svg ..."}',
                )
                resp.json = lambda: {  # type: ignore[attr-defined]
                    "ssid": "Dell Demo XXXX",
                    "password": "PASS",
                    "wifi_qr_svg": "<svg ...",
                }
                return resp

        monkeypatch.setattr(validate_customer.httpx, "AsyncClient", lambda **_kw: _Client())
        # Speed up the polling sleep.
        monkeypatch.setattr(validate_customer.asyncio, "sleep", _async_noop)

        import asyncio

        result = asyncio.run(validate_customer._phase_meeting_qr(ServerConfig.from_env()))
        assert result.status == "pass"
        assert result.metrics["ssid"] == "Dell Demo XXXX"

    def test_fail_when_hotspot_never_comes_up(self, monkeypatch: pytest.MonkeyPatch):
        from meeting_scribe.config import ServerConfig

        monkeypatch.setattr(validate_customer, "_signed_admin_cookie", lambda: ("k", "v"))
        monkeypatch.setattr(validate_customer, "_AP_WAIT_S", 1.0)

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_a):
                return None

            async def post(self, *_a, **_kw):
                resp = _FakeAsyncResponse(200, '{"meeting_id":"x"}')
                resp.json = lambda: {"meeting_id": "x"}  # type: ignore[attr-defined]
                return resp

            async def get(self, *_a, **_kw):
                # Always 503. Hotspot never comes up.
                return _FakeAsyncResponse(503, '{"error":"Hotspot not active"}')

        monkeypatch.setattr(validate_customer.httpx, "AsyncClient", lambda **_kw: _Client())
        monkeypatch.setattr(validate_customer.asyncio, "sleep", _async_noop)

        import asyncio

        result = asyncio.run(validate_customer._phase_meeting_qr(ServerConfig.from_env()))
        assert result.status == "fail"
        assert "did not come up" in result.detail


async def _async_noop(*_a, **_kw):
    return None
