"""Plan §1.6a — assert that ``requirements.lock`` is byte-equal to a
fresh ``pip-compile`` against the committed ``pyproject.toml``.

Marked with ``lockfile_sync`` so the default ``pytest`` run skips it
(it depends on ``pip-tools`` being installed in the venv); CI / the
pre-commit hook gate it explicitly with ``pytest -m lockfile_sync``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_lockfile_in_sync.py"


@pytest.mark.lockfile_sync
def test_pip_compile_matches_committed_lock() -> None:
    """Run the helper script and assert exit 0. The script's own
    diff output goes to stderr on failure, so pytest's captured-output
    will surface it on a real drift."""
    if shutil.which("pip-compile") is None:
        pytest.skip("pip-compile not on PATH (install dev deps)")
    proc = subprocess.run(
        ["python", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"requirements.lock drifted from pyproject.toml.\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}\n"
        f"Regenerate with:\n"
        f"    pip-compile --resolver=backtracking --upgrade "
        f"--output-file=requirements.lock pyproject.toml"
    )
