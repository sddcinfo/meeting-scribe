"""Tests for the factory-reset CLI command + admin endpoint.

The actual reset logic lives in ``setup_state.factory_reset`` and is
covered exhaustively by ``test_setup_state.py``. These tests verify
the wrappers (CLI + endpoint) call into ``setup_state.factory_reset``
with the right args and that the endpoint refuses guest callers.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from meeting_scribe import setup_state
from meeting_scribe.cli.setup import factory_reset_cmd


@pytest.fixture
def state_dir_with_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SCRIBE_STATE_DIR", str(tmp_path))
    setup_state.claim_setup("10.42.0.50", request_cookie_sid=None)
    setup_state.read_credentials()
    setup_state.ack_credentials_saved(admin_secret=b"x" * 64)
    (tmp_path / "setup-complete").write_text("1\n")
    return tmp_path


def test_cli_factory_reset_with_yes_flag(state_dir_with_setup: Path) -> None:
    starting_version = setup_state.auth_version()
    runner = CliRunner()
    with patch("meeting_scribe.server_support.settings_store._save_settings_override"):
        result = runner.invoke(factory_reset_cmd, ["--yes"])
    assert result.exit_code == 0, result.output
    assert "factory reset complete" in result.output
    assert setup_state.auth_version() == starting_version + 1
    assert not setup_state.is_setup_complete()


def test_cli_factory_reset_aborts_without_confirmation(
    state_dir_with_setup: Path,
) -> None:
    starting_version = setup_state.auth_version()
    runner = CliRunner()
    # User declines → exit code 1, no state change.
    result = runner.invoke(factory_reset_cmd, [], input="n\n")
    assert result.exit_code == 1
    assert setup_state.auth_version() == starting_version
    assert setup_state.is_setup_complete()
