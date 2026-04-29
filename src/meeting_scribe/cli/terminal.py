"""terminal sub-group: status + kill-tmux for the in-browser terminal panel."""

from __future__ import annotations

import os

import click

from meeting_scribe.cli import cli

# ── Embedded terminal panel ──────────────────────────────────


@cli.group("terminal")
def terminal_group() -> None:
    """Manage the in-browser terminal panel (tmux sessions on -L scribe)."""


@terminal_group.command("status")
def terminal_status() -> None:
    """List tmux sessions on the meeting-scribe socket + live registry usage."""
    import asyncio as _asyncio

    from meeting_scribe.terminal import tmux_helper as _tmux

    sessions = _asyncio.run(_tmux.list_sessions())
    if not sessions:
        click.echo("No tmux sessions on socket 'scribe' (server not running or empty).")
        click.echo(f"Config: {_tmux.config_path()}")
        return
    click.echo(f"Socket: /tmp/tmux-{os.geteuid()}/scribe")
    click.secho(
        f"{'Session':<24} {'Attached':>9} {'Windows':>8} {'Age':>8}",
        fg="cyan",
        bold=True,
    )
    import time as _time

    now = int(_time.time())
    for s in sessions:
        age_s = now - s.created
        if age_s < 60:
            age = f"{age_s}s"
        elif age_s < 3600:
            age = f"{age_s // 60}m"
        else:
            age = f"{age_s // 3600}h"
        click.echo(f"{s.name:<24} {s.attached:>9} {s.windows:>8} {age:>8}")


@terminal_group.command("kill-tmux")
@click.confirmation_option(prompt="Kill the meeting-scribe tmux server and ALL its sessions?")
def terminal_kill_tmux() -> None:
    """Explicit admin cleanup — tears down the -L scribe tmux server.

    meeting-scribe never runs this automatically. Use it if a tmux session
    is wedged or you want a fresh start.
    """
    import asyncio as _asyncio

    from meeting_scribe.terminal import tmux_helper as _tmux

    killed = _asyncio.run(_tmux.kill_server())
    if killed:
        click.secho("tmux -L scribe: killed.", fg="green")
    else:
        click.echo("No tmux server was running on socket 'scribe'.")
