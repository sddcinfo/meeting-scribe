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
