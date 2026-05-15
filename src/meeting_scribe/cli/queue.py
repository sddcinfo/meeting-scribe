"""drain / pause-translation / resume-translation subcommands.

All three poke the running server's ``/api/admin/*`` endpoints —
they're grouped here because they share the ``_post_admin`` helper
and the "fail loud if no server is running" semantics.
"""

from __future__ import annotations

import time

import click

from meeting_scribe.cli import cli
from meeting_scribe.cli._common import (
    DEFAULT_PORT,
    _post_admin,
    _server_state,
)


@cli.command("drain")
@click.option(
    "--timeout",
    default=60.0,
    show_default=True,
    type=float,
    help="Seconds to wait for outstanding translation + slide work to clear.",
)
@click.option(
    "--poll-interval",
    default=1.0,
    show_default=True,
    type=float,
    help="Seconds between status polls (lower = tighter wait but more HTTP traffic).",
)
@click.option(
    "--force",
    is_flag=True,
    help=(
        "Cancel pending translation items + abort the in-flight slide "
        "pipeline instead of waiting.  In-flight work is dropped — use "
        "only for incidents, never to skip a wait."
    ),
)
def drain(timeout: float, poll_interval: float, force: bool) -> None:
    """Block until outstanding translation + slide work has cleared.

    \b
    Exits 0 when everything is idle, non-zero on timeout.  This is the
    load-bearing gate before any model unload — log silence in
    scribe-translations.jsonl is NOT a drain condition (buffered writes,
    in-flight HTTP, slide jobs mid-reinsert all hide).

    Hits /api/admin/drain on the running server.  If the server isn't
    reachable, drain exits 0 (nothing to drain) with a warning.

    \b
    --force drops in-flight work. Reserved for incidents (regression
    rollback, crash loops).  Dropped slide jobs are unrecoverable.
    """
    import urllib.error
    import urllib.request

    _state, pid, _origin = _server_state()
    if pid is None:
        click.echo(click.style("No server running; nothing to drain.", fg="yellow"))
        return

    host, admin_port = "127.0.0.1", DEFAULT_PORT
    url = f"http://{host}:{admin_port}/api/admin/drain"
    deadline = time.monotonic() + timeout

    import json as _json

    first_poll = True
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise click.ClickException(
                f"drain timed out after {timeout:.1f}s; outstanding work did not clear"
            )
        # Force flag is only honored on the first request — subsequent
        # polls just wait for the cancellations to propagate through.
        query = f"?timeout={poll_interval}"
        if force and first_poll:
            query += "&force=true"
        first_poll = False
        try:
            req = urllib.request.Request(f"{url}{query}", method="POST")
            with urllib.request.urlopen(req, timeout=poll_interval + 2) as resp:
                body = _json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise click.ClickException(f"drain endpoint error {e.code}: {e.reason}") from e
        except urllib.error.URLError as e:
            raise click.ClickException(f"drain endpoint unreachable: {e}") from e

        if "force_cancelled" in body:
            fc = body["force_cancelled"]
            click.echo(
                click.style(
                    f"  force-cancelled: translation={fc.get('translation', 0)} "
                    f"slide={'yes' if fc.get('slide') else 'no'}",
                    fg="yellow",
                )
            )

        if body.get("idle"):
            click.echo(click.style("drained: queue empty, no slide job in flight", fg="green"))
            return

        active = body.get("translation_active", 0)
        pending = body.get("translation_pending", 0)
        slide = body.get("slide_in_flight", {})
        click.echo(
            f"  draining: translate active={active} pending={pending}  "
            f"slide={'yes' if slide.get('running') else 'no'}"
        )
        time.sleep(poll_interval)


@cli.command("pause-translation")
def pause_translation() -> None:
    """Gate new translation intake; already-queued items continue.

    Run before ``drain`` during a model-swap window so new ASR output
    doesn't pile up against the about-to-unload backend.  Idempotent.
    """
    body = _post_admin("pause-translation")
    if body.get("paused"):
        click.echo(click.style("translation intake paused", fg="green"))
    else:
        click.echo(click.style("translation intake still live (server rejected pause)", fg="red"))


@cli.command("resume-translation")
def resume_translation() -> None:
    """Re-open translation intake after a pause.  Idempotent."""
    body = _post_admin("resume-translation")
    if not body.get("paused"):
        click.echo(click.style("translation intake resumed", fg="green"))
    else:
        click.echo(
            click.style("translation intake still paused (server rejected resume)", fg="red")
        )
