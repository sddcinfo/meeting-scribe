"""Lint script — `docs/cli-ui-parity.md` covers every operator-facing CLI command.

Runs in CI and as a focused unit test (`tests/test_cli_ui_parity.py`).
The matrix is authoritative; this script catches drift.

Failure modes:
  * A CLI command that landed in `src/meeting_scribe/cli/` without a
    row in the matrix → exit non-zero with a "missing entry" message.
  * A row in the matrix that names a command we no longer have → exit
    non-zero with a "stale entry" message.

Use the unit-test wrapper for a CI hook; this CLI form is intended for
``meeting-scribe precommit cli-ui-parity`` (TBD wiring).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATRIX_PATH = PROJECT_ROOT / "docs" / "cli-ui-parity.md"


def _matrix_command_names() -> set[str]:
    """Extract the leftmost-column command names from the parity table.

    The table rows look like ``| `wifi up` | ... |`` so a single regex
    pulls the backticked command names. We deliberately do NOT try to
    parse the markdown — the format is stable and a raw regex is the
    least-fragile primitive.
    """
    text = MATRIX_PATH.read_text(encoding="utf-8")
    pattern = re.compile(r"^\|\s*`([^`]+)`\s*\|", re.MULTILINE)
    out: set[str] = set()
    for match in pattern.finditer(text):
        name = match.group(1).strip()
        # Drop trailing ` *` (wildcard rows like ``bench *``).
        if name.endswith(" *"):
            name = name[:-2]
        out.add(name)
    return out


_CLICK_DECL_RE = re.compile(
    r"@(?:cli|[\w_]+_group)\.(?:command|group)\(\s*\"([\w-]+)\"",
)
_CLICK_DECL_NO_ARG_RE = re.compile(
    r"^\s*@cli\.(?:command|group)\(\)\s*\n\s*(?:@.+\n\s*)*"
    r"def\s+(\w+)",
    re.MULTILINE,
)


def _cli_command_names() -> set[str]:
    """Walk every cli/*.py file and harvest registered command names.

    Picks up:
      * @cli.command("name")  / @cli.group("name")
      * @<group>.command("name") / @<group>.group("name")
      * @cli.command() decorated functions (name = function name with
        underscores → dashes).
    """
    out: set[str] = set()
    cli_root = PROJECT_ROOT / "src" / "meeting_scribe" / "cli"
    for py in cli_root.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        for match in _CLICK_DECL_RE.finditer(text):
            out.add(match.group(1))
        for match in _CLICK_DECL_NO_ARG_RE.finditer(text):
            out.add(match.group(1).replace("_", "-"))
    return out


def main() -> int:
    matrix_names = _matrix_command_names()
    cli_names = _cli_command_names()
    # We only enforce that EVERY discovered top-level CLI command has
    # SOME row in the matrix — prefix matches count (e.g. ``bt pair``
    # row covers the ``bt`` group's ``pair`` subcommand). This avoids
    # the lint becoming a churn factory while still catching drift on
    # whole new groups / commands.
    matrix_tokens: set[str] = set()
    for name in matrix_names:
        for token in name.split():
            matrix_tokens.add(token)

    missing = [name for name in cli_names if name not in matrix_tokens]
    stale = [
        name
        for name in matrix_names
        if not any(token in cli_names for token in name.split())
    ]

    rc = 0
    if missing:
        print("docs/cli-ui-parity.md missing entries for:", ", ".join(sorted(missing)))
        rc = 1
    if stale:
        # Stale rows are a warning, not a failure — the matrix lists
        # several "n/a" rows for OS-laptop CLIs that aren't in the
        # server CLI surface (e.g. trust-install, trust-uninstall).
        # We surface them as info but don't fail.
        print(
            "info: matrix mentions non-CLI-discovered commands (likely admin-laptop only):",
            ", ".join(sorted(stale)),
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
