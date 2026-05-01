"""``meeting-scribe doctor`` — operator diagnostics.

Currently houses ``watch-pressure``: a foreground tail of
``/proc/pressure/memory`` for ad-hoc inspection during a known-heavy
workload. The same sampling runs as a background task inside the
server lifespan (see ``runtime.health_monitors.mem_pressure_monitor``);
this CLI exists so an operator can watch the host without bringing
the whole stack up.

Added 2026-05-01 alongside the other guardrails for the 2026-04-30 OOM.
"""

from __future__ import annotations

import time

import click

from meeting_scribe.cli import cli
from meeting_scribe.runtime.health_monitors import (
    PRESSURE_PATH,
    read_pressure_snapshot,
)


@cli.group()
def doctor() -> None:
    """Diagnostic helpers (memory pressure, host health)."""


@doctor.command("watch-pressure")
@click.option("--interval", default=2.0, show_default=True,
              help="Seconds between samples.")
@click.option("--once", is_flag=True,
              help="Print one sample and exit. Useful for `watch -n 1`.")
def watch_pressure(interval: float, once: bool) -> None:
    """Tail /proc/pressure/memory with severity classification.

    PSI ``some`` is the early-warning channel (at least one task
    waiting on memory); ``full`` is the alarm channel (every runnable
    task stalled — kernel is one bad allocation away from OOM-killing
    something). Severity buckets here match the logged thresholds in
    ``mem_pressure_monitor`` so the human-readable view and the
    machine-watchable log agree.
    """
    if not PRESSURE_PATH.exists():
        raise click.ClickException(
            f"{PRESSURE_PATH} not available — kernel built without "
            "PSI support, or running in a container without it exposed."
        )

    def _line(snap) -> str:
        sev = snap.severity()
        color = {"ok": "green", "warn": "yellow", "crit": "red"}[sev]
        text = (
            f"[{time.strftime('%H:%M:%S')}] {sev.upper():4s}  "
            f"some avg10={snap.some_avg10:5.1f}% avg60={snap.some_avg60:5.1f}% "
            f"avg300={snap.some_avg300:5.1f}%  |  "
            f"full avg10={snap.full_avg10:5.1f}% avg60={snap.full_avg60:5.1f}%"
        )
        return click.style(text, fg=color)

    if once:
        snap = read_pressure_snapshot()
        if snap is None:
            raise click.ClickException("could not parse /proc/pressure/memory")
        click.echo(_line(snap))
        return

    click.echo(
        "Watching /proc/pressure/memory — Ctrl-C to stop. "
        "WARN ≥ some.avg10=10%; CRIT ≥ some.avg10=25% or full.avg10=5%."
    )
    try:
        while True:
            snap = read_pressure_snapshot()
            if snap is not None:
                click.echo(_line(snap))
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
