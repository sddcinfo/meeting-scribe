"""Unit tests for the meeting-scribe bench CLI subcommand.

Avoids the network-touching scripts entirely by monkey-patching the
state-file paths so each test gets its own scratch dir, and by
substituting the slo_probe.py spawn with a stub that just sleeps.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from meeting_scribe.cli import bench as bench_mod
from meeting_scribe.cli import cli as cli_root


@pytest.fixture
def temp_state(tmp_path: Path, monkeypatch):
    state = tmp_path / "scribe-bench-state.json"
    reason_file = tmp_path / "scribe-bench-window.txt"
    log_file = tmp_path / "scribe-slo-probe.log"
    monkeypatch.setattr(bench_mod, "STATE_FILE", state)
    monkeypatch.setattr(bench_mod, "WINDOW_REASON_FILE", reason_file)
    monkeypatch.setattr(bench_mod, "DEFAULT_LOG", log_file)
    return {"state": state, "reason_file": reason_file, "log": log_file}


def test_bench_status_reports_no_window_when_state_absent(temp_state) -> None:
    runner = CliRunner()
    result = runner.invoke(cli_root, ["bench", "status"])
    assert result.exit_code == 1, result.output
    assert "No bench window declared." in result.output


def test_bench_stop_is_noop_when_no_state(temp_state) -> None:
    runner = CliRunner()
    result = runner.invoke(cli_root, ["bench", "stop"])
    assert result.exit_code == 0, result.output
    assert "No bench window declared." in result.output


def test_bench_start_writes_state_and_stop_cleans_up(temp_state, monkeypatch) -> None:
    """End-to-end smoke: start writes state file, stop SIGTERMs + removes it."""
    runner = CliRunner()

    # Stub preflight to always succeed.
    class _FakeCompleted:
        returncode = 0

    monkeypatch.setattr(bench_mod.subprocess, "run", lambda *a, **k: _FakeCompleted())

    # Stub slo_probe spawn with a sleep loop the test can SIGTERM.
    class _FakeProc:
        def __init__(self):
            import sys

            self._proc = bench_mod.subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
            )
            self.pid = self._proc.pid

    fake = _FakeProc()
    monkeypatch.setattr(bench_mod.subprocess, "Popen", lambda *a, **k: fake)

    result = runner.invoke(
        cli_root,
        ["bench", "start", "--reason", "test window", "--api-url", "http://stub"],
    )
    assert result.exit_code == 0, result.output
    assert "Bench window OPEN" in result.output
    assert temp_state["state"].exists()
    assert "test window" in temp_state["reason_file"].read_text()

    # status reports alive + reason
    result = runner.invoke(cli_root, ["bench", "status"])
    assert "test window" in result.output

    # stop cleans up
    result = runner.invoke(cli_root, ["bench", "stop", "--force"])
    assert result.exit_code == 0, result.output
    assert "Bench window CLOSED" in result.output
    assert not temp_state["state"].exists()


def test_bench_start_refuses_double_open(temp_state) -> None:
    """If a state file already exists, `start` errors out cleanly."""
    temp_state["state"].write_text('{"reason": "earlier", "started_at": "yesterday"}\n')
    runner = CliRunner()
    result = runner.invoke(
        cli_root, ["bench", "start", "--reason", "should fail", "--api-url", "http://stub"]
    )
    assert result.exit_code == 2, result.output
    assert "already open" in result.output
