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
