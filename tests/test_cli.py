"""Tests for CLI commands — verifies command registration and basic execution."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from meeting_scribe.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestCLIRegistration:
    """All CLI commands are registered and have help text."""

    def test_main_group_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Meeting Scribe" in result.output

    def test_start_registered(self, runner):
        result = runner.invoke(cli, ["start", "--help"])
        assert result.exit_code == 0
        assert "Start" in result.output

    def test_stop_registered(self, runner):
        result = runner.invoke(cli, ["stop", "--help"])
        assert result.exit_code == 0

    def test_status_registered(self, runner):
        result = runner.invoke(cli, ["status", "--help"])
        assert result.exit_code == 0

    def test_logs_registered(self, runner):
        result = runner.invoke(cli, ["logs", "--help"])
        assert result.exit_code == 0

    def test_health_registered(self, runner):
        result = runner.invoke(cli, ["health", "--help"])
        assert result.exit_code == 0

    def test_diagnose_registered(self, runner):
        result = runner.invoke(cli, ["diagnose", "--help"])
        assert result.exit_code == 0

    def test_setup_registered(self, runner):
        result = runner.invoke(cli, ["setup", "--help"])
        assert result.exit_code == 0

    def test_finalize_registered(self, runner):
        result = runner.invoke(cli, ["finalize", "--help"])
        assert result.exit_code == 0

    def test_reprocess_registered(self, runner):
        result = runner.invoke(cli, ["reprocess", "--help"])
        assert result.exit_code == 0

    def test_audio_registered(self, runner):
        result = runner.invoke(cli, ["audio", "--help"])
        assert result.exit_code == 0
        assert "audio routing" in result.output.lower()

    def test_audio_route_registered(self, runner):
        result = runner.invoke(cli, ["audio", "route", "--help"])
        assert result.exit_code == 0
        assert "--room-sink-node" in result.output


class TestGB10Subcommands:
    """GB10 infrastructure subcommands."""

    def test_gb10_group(self, runner):
        result = runner.invoke(cli, ["gb10", "--help"])
        assert result.exit_code == 0
        assert "GB10" in result.output

    def test_gb10_up_registered(self, runner):
        result = runner.invoke(cli, ["gb10", "up", "--help"])
        assert result.exit_code == 0

    def test_gb10_down_registered(self, runner):
        result = runner.invoke(cli, ["gb10", "down", "--help"])
        assert result.exit_code == 0

    def test_gb10_status_registered(self, runner):
        result = runner.invoke(cli, ["gb10", "status", "--help"])
        assert result.exit_code == 0

    def test_gb10_setup_registered(self, runner):
        result = runner.invoke(cli, ["gb10", "setup", "--help"])
        assert result.exit_code == 0

    def test_gb10_pull_models_registered(self, runner):
        result = runner.invoke(cli, ["gb10", "pull-models", "--help"])
        assert result.exit_code == 0

    def test_gb10_start_registered(self, runner):
        result = runner.invoke(cli, ["gb10", "start", "--help"])
        assert result.exit_code == 0


class TestCLIExecution:
    """Basic CLI execution — works whether server is running or not."""

    def test_status_runs(self, runner):
        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        # Output contains either "running" or "not running"
        assert "running" in result.output.lower()

    def test_health_runs(self, runner):
        result = runner.invoke(cli, ["health"])
        # May fail if server not running, but shouldn't crash
        assert result.exit_code == 0 or "not running" in result.output.lower()


class TestAdminCertSelfHeal:
    """The admin TLS cert pair self-heals when missing.

    Regression guard for the 2026-04-20 cafa11c incident where the
    release workflow wiped ``certs/`` from the live working tree.
    The scribe service then refused to start until manual
    ``meeting-scribe setup``. ``_ensure_admin_tls_certs`` has been
    wired into the ``start`` path so a wiped cert pair regenerates
    silently and the service comes back up without intervention.
    """

    def test_ensure_tls_certs_idempotent(self, tmp_path, monkeypatch):
        import meeting_scribe.cli._common as cli_mod

        # Redirect PROJECT_ROOT at the helper's namespace so the
        # test doesn't touch the real certs/ directory.
        monkeypatch.setattr(cli_mod, "PROJECT_ROOT", tmp_path)

        ok, detail = cli_mod._ensure_admin_tls_certs()
        assert ok, detail
        key = tmp_path / "certs" / "key.pem"
        cert = tmp_path / "certs" / "cert.pem"
        assert key.exists()
        assert cert.exists()
        initial_key = key.read_bytes()

        # Second call is a no-op — must not regenerate.
        ok2, detail2 = cli_mod._ensure_admin_tls_certs()
        assert ok2, detail2
        assert "exist" in detail2.lower()
        assert key.read_bytes() == initial_key, "idempotent call must not rewrite existing certs"

    def test_ensure_tls_certs_regenerates_after_wipe(self, tmp_path, monkeypatch):
        import meeting_scribe.cli._common as cli_mod

        monkeypatch.setattr(cli_mod, "PROJECT_ROOT", tmp_path)

        ok, _ = cli_mod._ensure_admin_tls_certs()
        assert ok
        certs_dir = tmp_path / "certs"
        # Simulate `git clean -fdx` or the cafa11c-class wipe.
        for p in certs_dir.iterdir():
            p.unlink()

        ok2, detail = cli_mod._ensure_admin_tls_certs()
        assert ok2, detail
        assert (certs_dir / "key.pem").exists()
        assert (certs_dir / "cert.pem").exists()


# ──────────────────────────────────────────────────────────────
# Phase 2 — UnauthenticatedRedirect contract for _api_request
# ──────────────────────────────────────────────────────────────


class TestUnauthenticatedRedirectContract:
    """Regression guard for the 2026-05-14 demo debug: every CLI
    command must handle a benign auth redirect (302 → /signin / /setup
    / /auth) gracefully instead of reporting "not responding" or
    silently treating it as empty success."""

    def _patch_api_to_raise(self, monkeypatch, suffix: str) -> None:
        """Make every ``_api_request`` call raise ``UnauthenticatedRedirect``."""
        from meeting_scribe.cli._common import UnauthenticatedRedirect

        def _boom(*_args, **_kwargs):
            raise UnauthenticatedRedirect(suffix)

        # Patch BOTH the source module and every importer's bound name
        # so subprocess-style callers don't accidentally get the original.
        monkeypatch.setattr("meeting_scribe.cli._common._api_request", _boom)
        monkeypatch.setattr("meeting_scribe.cli.lifecycle._api_request", _boom)
        monkeypatch.setattr("meeting_scribe.cli.finalize._api_request", _boom)

    def test_status_treats_setup_redirect_as_running_awaiting_auth(self, runner, monkeypatch):
        """``meeting-scribe status`` against a server that 302s to
        /setup must print "awaiting auth" and exit 0 (server IS healthy,
        just unauthenticated). The previous behaviour was "not
        responding on /api/status" which misled operators."""
        # Simulate a foreground-started server so ``_server_state``
        # returns a PID and the status command reaches the API call.
        from meeting_scribe.cli import lifecycle as _lifecycle

        monkeypatch.setattr(_lifecycle, "_server_state", lambda: ("foreground", 12345, None))
        self._patch_api_to_raise(monkeypatch, "/setup")

        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0, result.output
        assert "awaiting auth" in result.output.lower()
        assert "/setup" in result.output

    def test_status_handles_signin_suffix(self, runner, monkeypatch):
        from meeting_scribe.cli import lifecycle as _lifecycle

        monkeypatch.setattr(_lifecycle, "_server_state", lambda: ("foreground", 99, None))
        self._patch_api_to_raise(monkeypatch, "/signin")

        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0, result.output
        assert "/signin" in result.output

    def test_finalize_status_module_handles_auth_redirect_clearly(self, runner, monkeypatch):
        """Direct unit-test of the ``finalize.status`` Click command so
        the auth-redirect handling is exercised without relying on the
        finalize-group's CLI binding order (``cli/meetings.py`` registers
        a top-level ``finalize`` command that shadows the group)."""
        from meeting_scribe.cli import finalize as _finalize_mod
        from meeting_scribe.cli._common import UnauthenticatedRedirect

        def _boom(*_args, **_kwargs):
            raise UnauthenticatedRedirect("/setup")

        monkeypatch.setattr(_finalize_mod, "_api_request", _boom)
        # The group command's callback is reachable directly through
        # the registered Click command object.
        result = runner.invoke(_finalize_mod.finalize_group, ["status"], standalone_mode=False)
        # The command sys.exit(2)s on this path; capture as either a
        # SystemExit exception or exit_code in the result.
        assert isinstance(result.exception, SystemExit) or result.exit_code != 0
        # The output should mention the redirect target and "auth".
        combined_output = (result.output or "") + (
            str(result.exception) if result.exception else ""
        )
        assert "/setup" in combined_output
        assert "auth" in combined_output.lower()


# ──────────────────────────────────────────────────────────────
# Phase 2 — `meeting-scribe logs` resolves the right log path
# ──────────────────────────────────────────────────────────────


class TestLogsCommandPathResolution:
    """``meeting-scribe logs`` must read from the diagnostics path the
    systemd unit actually writes to, not /tmp/meeting-scribe.log which
    only exists during a foreground start."""

    def test_diagnostics_log_path_uses_meetings_dir_parent(self, tmp_path, monkeypatch):
        """Mirrors ``ServerConfig.meetings_dir.parent`` resolution."""
        from meeting_scribe.cli import lifecycle as _lifecycle

        meetings_dir = tmp_path / "data" / "meetings"
        meetings_dir.mkdir(parents=True)
        monkeypatch.setenv("SCRIBE_MEETINGS_DIR", str(meetings_dir))
        path = _lifecycle._diagnostics_log_path()
        assert path == tmp_path / "data" / "diagnostics" / "server.log"

    def test_diagnostics_log_path_falls_back_to_project_root(self, tmp_path, monkeypatch):
        """Without ``SCRIBE_MEETINGS_DIR`` we anchor on ``PROJECT_ROOT``."""
        from meeting_scribe.cli import _common as _cli_common
        from meeting_scribe.cli import lifecycle as _lifecycle

        monkeypatch.delenv("SCRIBE_MEETINGS_DIR", raising=False)
        monkeypatch.setattr(_cli_common, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(_lifecycle, "PROJECT_ROOT", tmp_path)
        path = _lifecycle._diagnostics_log_path()
        assert path == tmp_path / "diagnostics" / "server.log"

    def test_logs_reads_diagnostics_when_present(self, runner, tmp_path, monkeypatch):
        """When the diagnostics log exists, ``logs`` reads from it
        (not from the legacy /tmp path)."""
        diag_dir = tmp_path / "diagnostics"
        diag_dir.mkdir()
        diag_file = diag_dir / "server.log"
        diag_file.write_text("line-1\nline-2\nFINGERPRINT_DIAG_LOG_PATH\nline-5\n")
        meetings_dir = tmp_path / "meetings"
        meetings_dir.mkdir()
        monkeypatch.setenv("SCRIBE_MEETINGS_DIR", str(meetings_dir))

        captured: dict[str, list[str]] = {}

        def _fake_subprocess_run(cmd, **_kwargs):
            captured["cmd"] = cmd
            return None

        monkeypatch.setattr("meeting_scribe.cli.lifecycle.subprocess.run", _fake_subprocess_run)
        result = runner.invoke(cli, ["logs", "-n", "3"])
        assert result.exit_code == 0, result.output
        # The diagnostics path must be the last positional arg of the
        # tail invocation.
        assert captured["cmd"][-1] == str(diag_file)
        assert "-n3" in captured["cmd"]

    def test_logs_journal_source_invokes_journalctl(self, runner, monkeypatch):
        """``--source=journal`` forces ``journalctl --user -u meeting-scribe``
        regardless of file presence."""
        captured: dict[str, list[str]] = {}

        def _fake_subprocess_run(cmd, **_kwargs):
            captured["cmd"] = cmd
            return None

        monkeypatch.setattr("meeting_scribe.cli.lifecycle.subprocess.run", _fake_subprocess_run)
        result = runner.invoke(cli, ["logs", "--source", "journal", "-n", "5"])
        assert result.exit_code == 0, result.output
        cmd = captured["cmd"]
        assert cmd[0] == "journalctl"
        assert "--user" in cmd
        assert "-u" in cmd
        assert "5" in cmd
