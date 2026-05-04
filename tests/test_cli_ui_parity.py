"""Tests for the CLI ↔ UI parity matrix lint.

The matrix lives at ``docs/cli-ui-parity.md``; the lint at
``scripts/check_cli_ui_parity.py``. This test re-runs the lint in
isolation so a PR that adds an unmatched operator-facing CLI command
fails CI.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def parity_module():
    """Load scripts/check_cli_ui_parity.py as a module without
    executing main()."""
    path = PROJECT_ROOT / "scripts" / "check_cli_ui_parity.py"
    spec = importlib.util.spec_from_file_location("check_cli_ui_parity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parity_lint_passes(parity_module) -> None:
    """The matrix and the live CLI agree — no missing operator-facing
    entries. (Stale info messages are non-blocking.)"""
    rc = parity_module.main()
    assert rc == 0


def test_parity_matrix_lists_bt_subcommands(parity_module) -> None:
    """Sanity: the BT group's command-set is represented."""
    matrix = parity_module._matrix_command_names()
    expected_tokens = {"bt", "pair", "connect", "disconnect", "forget", "status"}
    matrix_tokens: set[str] = set()
    for name in matrix:
        for token in name.split():
            matrix_tokens.add(token)
    missing = expected_tokens - matrix_tokens
    assert not missing, missing


def test_parity_matrix_lists_trust_subcommands(parity_module) -> None:
    matrix = parity_module._matrix_command_names()
    matrix_tokens: set[str] = set()
    for name in matrix:
        for token in name.split():
            matrix_tokens.add(token)
    for required in ("trust-install", "trust-uninstall", "cert-fingerprint", "export-cert"):
        assert required in matrix_tokens, required


def test_parity_lint_finds_cli_entries(parity_module) -> None:
    """The lint walks the cli/ tree — sanity check it actually finds
    something. Empty result ⇒ the regex got broken."""
    cli_names = parity_module._cli_command_names()
    assert "bt" in cli_names
    assert "wifi" in cli_names
    # `setup` is a top-level command, registered without a string arg,
    # so it lands in the no-arg regex path.
    assert "setup" in cli_names
