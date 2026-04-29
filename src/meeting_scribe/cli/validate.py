"""``meeting-scribe validate`` — unified component validation (Workstream B).

Three modes; see ``src/meeting_scribe/validate.py`` for what each phase
actually probes:

  meeting-scribe validate           # default --quick
  meeting-scribe validate --quick   # liveness + furigana, ≤ 5 s
  meeting-scribe validate --full    # + ASR/Translate/Diarize/TTS quality, ≤ 5 min
  meeting-scribe validate --e2e     # + live meeting end-to-end lag, ≤ 10 min

Exits 0 on success, 1 if any phase failed.
"""

from __future__ import annotations

import asyncio
import json
import sys

import click

from meeting_scribe.cli import cli


@cli.command()
@click.option(
    "--quick", "mode", flag_value="quick", default=True, help="Liveness + furigana only (≤5s)."
)
@click.option(
    "--full", "mode", flag_value="full", help="Quick + per-backend quality probes (≤5min)."
)
@click.option(
    "--e2e", "mode", flag_value="e2e", help="Full + live-meeting end-to-end lag (≤10min)."
)
@click.option(
    "--hardware-class",
    default="gb10",
    show_default=True,
    help="Baseline class to gate against (currently only `gb10`).",
)
@click.option("--json", "json_only", is_flag=True, help="Emit machine-readable JSON to stdout.")
def validate(mode: str, hardware_class: str, json_only: bool) -> None:
    """Validate every backend reaches AND produces sane output."""
    from meeting_scribe.validate import run_validate

    if not json_only:
        click.echo(f"meeting-scribe validate --{mode} (baseline={hardware_class})")
        click.echo("")

    try:
        report = asyncio.run(
            run_validate(mode=mode, hardware_class=hardware_class, json_only=json_only)
        )
    except KeyboardInterrupt:
        click.secho("interrupted", fg="yellow", err=True)
        sys.exit(130)

    if json_only:
        click.echo(json.dumps(report.to_json(), indent=2))
    else:
        click.echo("")
        if report.passed:
            click.secho("VALIDATE GREEN", fg="green", bold=True)
        else:
            failing = [p.name for p in report.phases if p.status == "fail"]
            click.secho(f"VALIDATE FAILED: {', '.join(failing)}", fg="red", bold=True)

    sys.exit(0 if report.passed else 1)
