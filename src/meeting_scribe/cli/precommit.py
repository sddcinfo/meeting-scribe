"""precommit subcommand: scan working tree for sensitive data + style regressions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

from meeting_scribe.cli import cli


@cli.command("precommit")
@click.option("--verbose", "-v", is_flag=True, default=False)
@click.option(
    "--warn-only",
    is_flag=True,
    default=False,
    help="Treat all hits as warnings (exit 0 even on block-level findings).",
)
@click.option(
    "--include-all-tracked",
    is_flag=True,
    default=False,
    help="Scan every tracked file (deep scan) instead of only the working tree.",
)
@click.pass_context
def precommit(
    ctx: click.Context,
    verbose: bool,
    warn_only: bool,
    include_all_tracked: bool,
) -> None:
    """Scan meeting-scribe working tree for sensitive data + style regressions.

    Runs two validators:
      - scripts/check_ui_style.py (STYLING.md rule set: no native popups,
        no em-dashes in user-facing HTML, modal text selectable + wrapping,
        glossary coverage)
      - precommit_scanner (embedded journal entries, recording paths,
        audio references, credentials, LAN identity)

    Also advises the developer to enable .githooks/pre-commit once per
    clone (see STYLING.md) so style regressions block commits locally.
    """
    from meeting_scribe import precommit_scanner

    repo_root = Path(__file__).resolve().parents[3]

    style_rc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "check_ui_style.py")],
        check=False,
    ).returncode
    if style_rc != 0:
        click.secho(
            "UI style regressions detected. Fix them (see STYLING.md) and rerun.",
            fg="red",
        )
        if not warn_only:
            sys.exit(style_rc)

    ctx.invoke(
        precommit_scanner.precommit,
        repo=repo_root,
        include_staged=True,
        include_all_tracked=include_all_tracked,
        verbose=verbose,
        warn_only=warn_only,
    )
