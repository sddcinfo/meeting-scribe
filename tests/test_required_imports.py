"""Tests for the required-imports startup self-check.

Plan §1.6b: ``_assert_required_imports()`` must fail BEFORE any socket
is created, exiting 78 (EX_CONFIG) when a required import is missing.
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest


def test_passes_when_all_imports_succeed() -> None:
    """The current venv has every required import installed (test
    runner couldn't have started otherwise), so the check is a no-op."""
    from meeting_scribe.cli._common import _assert_required_imports

    _assert_required_imports()  # must not raise / not exit


def test_exits_78_on_missing_import() -> None:
    """When importlib.import_module raises ImportError for any name in
    pyproject.toml's tool.meeting-scribe.required-imports, the function
    must SystemExit(78). Asserting on SystemExit, not on side effects,
    because the call site is `start --foreground` BEFORE any socket
    bind, so we just need exit-78."""
    from meeting_scribe.cli._common import _assert_required_imports

    real_import = importlib.import_module

    def _selective_fail(name: str, *args, **kwargs):
        if name == "python_multipart":
            raise ImportError("simulated missing dep")
        return real_import(name, *args, **kwargs)

    with (
        patch.object(importlib, "import_module", side_effect=_selective_fail),
        pytest.raises(SystemExit) as excinfo,
    ):
        _assert_required_imports()
    assert excinfo.value.code == 78


def test_no_socket_bind_attempted_when_check_fires() -> None:
    """The whole point of moving the check to the CLI entrypoint
    (rather than the FastAPI lifespan) is that no socket is ever
    bound when the check fails. This test patches socket.socket and
    asserts it's never instantiated during the check."""
    from meeting_scribe.cli._common import _assert_required_imports

    real_import = importlib.import_module

    def _selective_fail(name: str, *args, **kwargs):
        if name == "python_multipart":
            raise ImportError("simulated missing dep")
        return real_import(name, *args, **kwargs)

    with (
        patch("socket.socket") as mock_sock,
        patch.object(importlib, "import_module", side_effect=_selective_fail),
    ):
        with pytest.raises(SystemExit):
            _assert_required_imports()
        mock_sock.assert_not_called()


def test_tolerates_missing_pyproject(monkeypatch, tmp_path) -> None:
    """A stripped runtime layout shouldn't crash the server — the
    function silently no-ops when pyproject.toml is absent. The
    lockfile is the authoritative install gate; this is a tripwire,
    not a substitute."""
    from meeting_scribe.cli import _common

    monkeypatch.setattr(_common, "PROJECT_ROOT", tmp_path)
    _common._assert_required_imports()  # no exception


def test_tolerates_missing_required_imports_key(monkeypatch, tmp_path) -> None:
    """A pyproject.toml without the [tool.meeting-scribe] table must
    not crash — older checkouts that pre-date this plan should still
    boot."""
    from meeting_scribe.cli import _common

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "x"\nversion = "0.0.0"\n')
    monkeypatch.setattr(_common, "PROJECT_ROOT", tmp_path)
    _common._assert_required_imports()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
