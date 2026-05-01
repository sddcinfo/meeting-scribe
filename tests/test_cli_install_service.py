"""Tests for ``meeting-scribe install-service`` / ``uninstall-service``.

The actual systemctl/loginctl calls are mocked so the test runs on any
host (including CI containers without a user-bus). What we exercise:

- Unit-file rendering (``ExecStart=`` paths point at the venv).
- Idempotent writes (running twice doesn't rewrite an unchanged file).
- The CLI surface routes through the right ``systemctl --user`` calls.
- ``uninstall-service`` tolerates a not-installed unit gracefully.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from meeting_scribe.cli import cli
from meeting_scribe.cli.install_service import _render_unit, _user_unit_dir


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Redirect XDG + venv-entry checks so the real user's $HOME is untouched."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("USER", "tester")
    # Pretend the entry point exists so the install command doesn't bail.
    venv_bin = Path(__file__).resolve().parents[1] / ".venv" / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
    entry = venv_bin / "meeting-scribe"
    if not entry.exists():
        entry.write_text("#!/bin/sh\n")
        entry.chmod(0o755)
    yield tmp_path


# ── Unit rendering ──────────────────────────────────────────────────


def test_render_unit_includes_notify_protocol():
    body = _render_unit()
    assert "Type=notify" in body
    assert "NotifyAccess=main" in body


def test_render_unit_uses_foreground_flag():
    """systemd's Type=notify needs the unforked child."""
    body = _render_unit()
    assert "meeting-scribe start --foreground" in body


def test_render_unit_targets_venv_entry():
    """ExecStart= must resolve to the venv's binary so PATH-shadowing
    can't redirect the unit to a different install."""
    body = _render_unit()
    assert ".venv/bin/meeting-scribe" in body


def test_render_unit_install_section_for_default_target():
    body = _render_unit()
    assert "WantedBy=default.target" in body


def test_user_unit_dir_respects_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "x"))
    assert _user_unit_dir() == tmp_path / "x" / "systemd" / "user"


def test_user_unit_dir_falls_back_to_dot_config(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert _user_unit_dir() == tmp_path / ".config" / "systemd" / "user"


# ── install-service ─────────────────────────────────────────────────


class _MockProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_install_service_writes_unit_and_enables(runner, fake_home):
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return _MockProc(returncode=0)

    with (
        patch("meeting_scribe.cli.install_service.shutil.which", return_value="/bin/systemctl"),
        patch("meeting_scribe.cli.install_service.subprocess.run", side_effect=fake_run),
    ):
        result = runner.invoke(cli, ["install-service"])

    assert result.exit_code == 0, result.output
    unit = fake_home / ".config" / "systemd" / "user" / "meeting-scribe.service"
    assert unit.exists()
    body = unit.read_text()
    assert "Type=notify" in body

    # daemon-reload + enable --now should be in the call sequence.
    cmds = [" ".join(c) for c in calls]
    assert any("systemctl --user daemon-reload" in c for c in cmds)
    assert any("systemctl --user enable --now meeting-scribe.service" in c for c in cmds)


def test_install_service_no_start_only_enables(runner, fake_home):
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return _MockProc(returncode=0)

    with (
        patch("meeting_scribe.cli.install_service.shutil.which", return_value="/bin/systemctl"),
        patch("meeting_scribe.cli.install_service.subprocess.run", side_effect=fake_run),
    ):
        result = runner.invoke(cli, ["install-service", "--no-start"])

    assert result.exit_code == 0, result.output
    cmds = [" ".join(c) for c in calls]
    # enable WITHOUT --now
    assert any("systemctl --user enable meeting-scribe.service" in c for c in cmds)
    assert not any("enable --now" in c for c in cmds)


def test_install_service_no_enable_skips_systemctl_enable(runner, fake_home):
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return _MockProc(returncode=0)

    with (
        patch("meeting_scribe.cli.install_service.shutil.which", return_value="/bin/systemctl"),
        patch("meeting_scribe.cli.install_service.subprocess.run", side_effect=fake_run),
    ):
        result = runner.invoke(cli, ["install-service", "--no-enable"])

    assert result.exit_code == 0, result.output
    cmds = [" ".join(c) for c in calls]
    # daemon-reload still runs so the file is recognized; enable does not
    assert any("daemon-reload" in c for c in cmds)
    assert not any("systemctl --user enable" in c for c in cmds)
    assert not any("loginctl" in c for c in cmds)


def test_install_service_idempotent_when_unchanged(runner, fake_home):
    """Running twice in a row writes the unit once and the second pass
    detects no drift."""
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return _MockProc(returncode=0)

    with (
        patch("meeting_scribe.cli.install_service.shutil.which", return_value="/bin/systemctl"),
        patch("meeting_scribe.cli.install_service.subprocess.run", side_effect=fake_run),
    ):
        first = runner.invoke(cli, ["install-service", "--no-enable"])
        second = runner.invoke(cli, ["install-service", "--no-enable"])

    assert first.exit_code == 0 and second.exit_code == 0
    assert "wrote" in first.output
    assert "already up-to-date" in second.output


def test_install_service_fails_loudly_without_systemctl(runner, fake_home):
    with patch("meeting_scribe.cli.install_service.shutil.which", return_value=None):
        result = runner.invoke(cli, ["install-service"])
    assert result.exit_code == 1
    assert "systemctl not found" in result.output


def test_install_service_bails_when_venv_entry_missing(runner, fake_home, monkeypatch):
    """The venv must contain the ``meeting-scribe`` entry point or the
    unit's ExecStart= would point at nothing."""
    venv_bin = Path(__file__).resolve().parents[1] / ".venv" / "bin"
    entry = venv_bin / "meeting-scribe"
    if entry.exists():
        entry.unlink()
    with patch("meeting_scribe.cli.install_service.shutil.which", return_value="/bin/systemctl"):
        result = runner.invoke(cli, ["install-service"])
    assert result.exit_code == 1
    assert "entry point missing" in result.output
    # Restore so other tests in the same run don't fail.
    entry.write_text("#!/bin/sh\n")
    entry.chmod(0o755)


# ── uninstall-service ───────────────────────────────────────────────


def test_uninstall_service_removes_unit_file(runner, fake_home):
    unit = fake_home / ".config" / "systemd" / "user" / "meeting-scribe.service"
    unit.parent.mkdir(parents=True)
    unit.write_text("# stale unit content\n")

    def fake_run(cmd, *args, **kwargs):
        return _MockProc(returncode=0)

    with (
        patch("meeting_scribe.cli.install_service.shutil.which", return_value="/bin/systemctl"),
        patch("meeting_scribe.cli.install_service.subprocess.run", side_effect=fake_run),
    ):
        result = runner.invoke(cli, ["uninstall-service"])

    assert result.exit_code == 0, result.output
    assert not unit.exists()


def test_uninstall_service_tolerates_unit_not_loaded(runner, fake_home):
    """When systemctl reports the unit isn't loaded, that's a no-op
    success — we shouldn't fail the user's uninstall."""

    def fake_run(cmd, *args, **kwargs):
        if "stop" in cmd or "disable" in cmd:
            return _MockProc(returncode=1, stderr="Unit meeting-scribe.service not loaded.")
        return _MockProc(returncode=0)

    with (
        patch("meeting_scribe.cli.install_service.shutil.which", return_value="/bin/systemctl"),
        patch("meeting_scribe.cli.install_service.subprocess.run", side_effect=fake_run),
    ):
        result = runner.invoke(cli, ["uninstall-service"])

    assert result.exit_code == 0
    # No "[WARN]" should appear for the expected not-loaded case.
    assert "[WARN]" not in result.output
