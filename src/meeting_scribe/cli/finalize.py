"""finalize sub-group: status + retry for the Phase A / Phase B finalize split.

When ``SCRIBE_BACKGROUND_FINALIZE=1`` the heavy finalize work runs as
a background task after Stop returns. These commands let the operator
inspect that work and recover from a meeting that landed in
INTERRUPTED via Phase B failure or restart-mid-finalize.
"""

from __future__ import annotations

import json
import sys

import click

from meeting_scribe.cli import cli
from meeting_scribe.cli._common import DEFAULT_PORT, UnauthenticatedRedirect, _api_request


@cli.group("finalize")
def finalize_group():
    """Background finalize management (status / retry).

    \b
    Examples:
      meeting-scribe finalize status                   # list in-flight Phase B
      meeting-scribe finalize status <mid>             # one meeting's Phase B
      meeting-scribe finalize retry <mid>              # re-run finalize
    """


@finalize_group.command("status")
@click.argument("meeting_id", required=False)
@click.option("--port", "-p", default=DEFAULT_PORT, help="Admin HTTPS port")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def status(meeting_id: str | None, port: int, as_json: bool) -> None:
    """Show in-flight Phase B tasks + GPU lease holder.

    Without a meeting_id, lists every in-flight Phase B (one line per
    meeting). With a meeting_id, prints just that meeting's row, or a
    "not finalizing" line if it isn't running. Exits 0 either way.
    """
    try:
        payload = _api_request("/api/admin/finalize/status", port=port)
    except UnauthenticatedRedirect as exc:
        click.secho(
            f"Server returned an auth redirect (→ {exc.suffix}). "
            "Visit the setup page or re-run admin setup to authenticate, then retry.",
            fg="red",
        )
        sys.exit(2)
    if not payload:
        click.secho(
            "Could not reach /api/admin/finalize/status — is the server running?",
            fg="red",
        )
        sys.exit(2)

    if as_json:
        click.echo(json.dumps(payload, indent=2))
        return

    holder = payload.get("gpu_lease_holder", "idle")
    click.secho(f"GPU lease holder: {holder}", fg="green" if holder == "idle" else "yellow")

    tasks = payload.get("phase_b_tasks", [])
    if meeting_id:
        match = next((t for t in tasks if t.get("meeting_id") == meeting_id), None)
        if not match:
            click.echo(f"Meeting {meeting_id}: no Phase B in flight.")
            return
        click.echo(_format_task_line(match))
        return

    if not tasks:
        click.echo("No Phase B tasks in flight.")
        return

    for task in tasks:
        click.echo(_format_task_line(task))


def _format_task_line(task: dict) -> str:
    mid = task.get("meeting_id", "?")
    state = task.get("state", "?")
    name = task.get("name", "?")
    return f"  {mid}  state={state}  task={name}"


@finalize_group.command("retry")
@click.argument("meeting_id")
@click.option("--port", "-p", default=DEFAULT_PORT, help="Admin HTTPS port")
def retry(meeting_id: str, port: int) -> None:
    """Re-run finalize for a meeting in INTERRUPTED state.

    Routes through the existing /api/meetings/{id}/reprocess endpoint
    which re-runs the diarize + speaker_attach + timeline + summary
    pipeline against the durable journal + PCM. Use after a Phase B
    crash or a server restart that left a meeting INTERRUPTED.
    """
    try:
        resp = _api_request(
            f"/api/meetings/{meeting_id}/reprocess",
            port=port,
            method="POST",
        )
    except UnauthenticatedRedirect as exc:
        click.secho(
            f"Server returned an auth redirect (→ {exc.suffix}). "
            "Authenticate via the setup page, then retry.",
            fg="red",
        )
        sys.exit(2)
    if not resp:
        click.secho(f"Reprocess request failed for {meeting_id}", fg="red")
        sys.exit(2)
    click.secho(f"Reprocess started: {resp.get('status', '?')}", fg="green")
    if resp.get("attempt_id"):
        click.echo(f"  attempt_id: {resp['attempt_id']}")
