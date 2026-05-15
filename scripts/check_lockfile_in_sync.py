#!/usr/bin/env python3
"""Verify ``requirements.lock`` is byte-equal to a fresh ``pip-compile``.

Plan §1.6a: the lockfile is the customer's authoritative dependency
state — `bootstrap.sh` runs `pip-sync requirements.lock` and the
customer venv ends up bit-for-bit a function of it. If the lockfile
drifts from `pyproject.toml`, the customer install silently picks up
versions that nobody tested.

This script:
  1. Runs `pip-compile --resolver=backtracking --output-file <tmp>`
     against the committed `pyproject.toml`.
  2. Diffs <tmp> against `requirements.lock` (mode-0644 in the repo).
  3. Exits 0 on match, 1 on drift, 2 on tooling failure.

Wired into pytest as a `lockfile_sync` marker (`pytest -m lockfile_sync`)
AND used as a pre-commit hook so PRs that touch `pyproject.toml` without
also regenerating the lock are rejected.

Regenerate the lockfile with:
    pip-compile --resolver=backtracking --upgrade --output-file=requirements.lock pyproject.toml
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
LOCKFILE = REPO_ROOT / "requirements.lock"


def main() -> int:
    if shutil.which("pip-compile") is None:
        print(
            "pip-compile not on PATH — install dev deps with "
            "`python -m pip install -e .[dev]` and re-run.",
            file=sys.stderr,
        )
        return 2

    if not LOCKFILE.is_file():
        print(
            f"missing committed lockfile: {LOCKFILE.relative_to(REPO_ROOT)}\n"
            "Generate it with:\n"
            "    pip-compile --resolver=backtracking "
            "--output-file=requirements.lock pyproject.toml",
            file=sys.stderr,
        )
        return 1

    # pip-compile bakes the literal CLI args into the lockfile's
    # auto-generated comment block. To get a comparable result, run
    # from REPO_ROOT with both paths exactly as the docs print
    # (`pip-compile --output-file=requirements.lock pyproject.toml`)
    # and capture into a temp file IN REPO_ROOT named identically to
    # the committed lock — only the directory differs (a hidden subdir).
    with tempfile.TemporaryDirectory(prefix=".lockfile-sync-", dir=REPO_ROOT) as td:
        td_path = Path(td)
        candidate = td_path / "requirements.lock"
        # pip-compile -o requirements.lock writes the literal arg into
        # the comment. To match, run pip-compile with cwd=td_path and
        # symlink pyproject.toml into td_path so it can see it.
        symlinked_pyproject = td_path / "pyproject.toml"
        symlinked_pyproject.symlink_to(PYPROJECT)
        proc = subprocess.run(
            [
                "pip-compile",
                "--resolver=backtracking",
                "--quiet",
                "--output-file",
                "requirements.lock",
                "pyproject.toml",
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=td_path,
        )
        if proc.returncode != 0:
            print(f"pip-compile failed (rc={proc.returncode}):", file=sys.stderr)
            sys.stderr.write(proc.stderr)
            return 2

        actual = LOCKFILE.read_bytes()
        expected = candidate.read_bytes()
        if actual == expected:
            print(f"OK — {LOCKFILE.name} matches a fresh pip-compile.")
            return 0

        print(
            f"DRIFT: {LOCKFILE.name} does not match a fresh pip-compile run "
            f"against pyproject.toml.\n"
            f"Regenerate with:\n"
            f"    pip-compile --resolver=backtracking --upgrade "
            f"--output-file={LOCKFILE.name} pyproject.toml",
            file=sys.stderr,
        )
        # Show a diff hint.
        diff = subprocess.run(
            ["diff", "-u", str(LOCKFILE), str(candidate)],
            capture_output=True,
            text=True,
            check=False,
        )
        sys.stderr.write(diff.stdout[:4000])  # truncate, full diff is in tmp
        return 1


if __name__ == "__main__":
    sys.exit(main())
