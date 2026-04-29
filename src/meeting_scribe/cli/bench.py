"""bench sub-group: open / close model-challenger bench windows safely.

Production safety scaffolding for the 2026-Q3 bench follow-ups
(plans/2026-Q3-followups.md).  Replaces the manual "edit shell env,
edit /tmp/scribe-bench-window.txt, start slo_probe.py in a side
terminal" ritual with three subcommands::

    meeting-scribe bench start --reason "<reason>"
        Runs scripts/bench/preflight.py, spawns scripts/bench/slo_probe.py
        as a daemon, writes /tmp/scribe-bench-state.json, exports the
        MEETING_SCRIBE_BENCH_WINDOW=1 marker via /tmp/scribe-bench-window.txt.

    meeting-scribe bench status
        Reads /tmp/scribe-bench-state.json + the SLO probe log; reports
        OK / ABORT (reason) / no window declared.

    meeting-scribe bench stop
        SIGTERMs the SLO probe daemon, removes /tmp/scribe-bench-state.json,
        leaves the log file in place for forensics.

The canonical implementations of preflight + slo_probe stay under
``scripts/bench/``; this CLI is purely an orchestration layer that
subprocess-runs them, mirroring the pattern in ``cli/benchmark.py``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import click

from meeting_scribe.cli import cli

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_BENCH = _REPO_ROOT / "scripts" / "bench"

STATE_FILE = Path("/tmp/scribe-bench-state.json")
WINDOW_REASON_FILE = Path("/tmp/scribe-bench-window.txt")
DEFAULT_LOG = Path("/tmp/scribe-slo-probe.log")


@cli.group("bench")
def bench_group() -> None:
    """Model-challenger bench window lifecycle.

    \b
    Open a window before any bench harness runs:
      meeting-scribe bench start --reason "Track A real-corpus re-run"
      meeting-scribe bench status
      meeting-scribe bench stop

    See plans/2026-Q3-followups.md for the per-bench checklist.
    """


def _read_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return None


def _write_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


def _slo_probe_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but isn't ours.  Treat as alive for "status" purposes.
        return True
    return True


@bench_group.command("start")
@click.option("--reason", required=True, help="Free-text reason for the bench window.")
@click.option(
    "--api-url",
    default="https://localhost:8080",
    show_default=True,
    help="meeting-scribe API base URL for the preflight check.",
)
@click.option(
    "--log",
    type=Path,
    default=DEFAULT_LOG,
    show_default=True,
    help="SLO probe log path.",
)
@click.option(
    "--interval",
    type=int,
    default=10,
    show_default=True,
    help="SLO probe sampling interval (seconds).",
)
def bench_start(reason: str, api_url: str, log: Path, interval: int) -> None:
    """Open a bench window: preflight + start the SLO probe daemon."""
    if STATE_FILE.exists():
        existing = _read_state() or {}
        click.secho(
            f"Bench window already open (started {existing.get('started_at', '?')}, "
            f"reason: {existing.get('reason', '?')!r}). Run `meeting-scribe bench stop` first.",
            fg="yellow",
        )
        sys.exit(2)

    WINDOW_REASON_FILE.write_text(reason.strip() + "\n")

    # Run preflight with the env var set.  Hard-fail if it doesn't pass.
    env = {**os.environ, "MEETING_SCRIBE_BENCH_WINDOW": "1"}
    preflight_cmd = [
        sys.executable,
        str(_SCRIPTS_BENCH / "preflight.py"),
        "--api-url",
        api_url,
    ]
    click.secho(f"==> Running preflight: {' '.join(preflight_cmd)}", fg="cyan")
    rc = subprocess.run(preflight_cmd, env=env).returncode
    if rc != 0:
        click.secho(
            "Preflight FAILED.  Bench window NOT opened.  Resolve the issue and retry.",
            fg="red",
        )
        # Leave the reason file in place for the next attempt.
        sys.exit(rc)

    # Spawn the SLO probe daemon.
    log.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log.open("ab")
    probe_proc = subprocess.Popen(
        [
            sys.executable,
            str(_SCRIPTS_BENCH / "slo_probe.py"),
            "--log",
            str(log),
            "--interval",
            str(interval),
        ],
        stdout=log_fh,
        stderr=log_fh,
        env=env,
        start_new_session=True,
    )

    state = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "reason": reason.strip(),
        "api_url": api_url,
        "slo_probe_pid": probe_proc.pid,
        "slo_probe_log": str(log),
        "interval_s": interval,
        "window_env_var": "MEETING_SCRIBE_BENCH_WINDOW=1",
        "reason_file": str(WINDOW_REASON_FILE),
    }
    _write_state(state)

    click.echo()
    click.secho("Bench window OPEN.", fg="green", bold=True)
    click.echo(f"  reason:    {reason}")
    click.echo(f"  state:     {STATE_FILE}")
    click.echo(f"  probe pid: {probe_proc.pid}")
    click.echo(f"  probe log: {log}")
    click.echo()
    click.secho(
        "Remember: every bench harness must run with MEETING_SCRIBE_BENCH_WINDOW=1 in its env.",
        fg="yellow",
    )
    click.echo("  e.g.  MEETING_SCRIBE_BENCH_WINDOW=1 python3 scripts/bench/asr_ja_wer_run.py ...")


@bench_group.command("status")
def bench_status() -> None:
    """Report on the current bench window: open / aborted / closed."""
    state = _read_state()
    if state is None:
        click.secho("No bench window declared.", fg="yellow")
        sys.exit(1)

    pid = state.get("slo_probe_pid")
    alive = _slo_probe_alive(pid)
    log_path = Path(state.get("slo_probe_log") or DEFAULT_LOG)

    click.secho("Bench window status", bold=True)
    click.echo(f"  started:    {state.get('started_at', '?')}")
    click.echo(f"  reason:     {state.get('reason', '?')!r}")
    click.echo(f"  api_url:    {state.get('api_url', '?')}")
    click.echo(f"  probe pid:  {pid}  ({'alive' if alive else 'DEAD'})")
    click.echo(f"  probe log:  {log_path}")

    if log_path.exists():
        # Tail the last few lines and look for ABORT.
        try:
            tail_lines = log_path.read_text().splitlines()[-20:]
        except Exception as e:
            click.secho(f"  log read failed: {e}", fg="red")
            return
        abort_lines = [ln for ln in tail_lines if " ABORT " in ln]
        if abort_lines:
            click.secho("\n  SLO ABORT detected:", fg="red", bold=True)
            for ln in abort_lines:
                click.echo(f"    {ln}")
            sys.exit(3)
        click.echo()
        click.secho("  recent probe samples (tail):", fg="cyan")
        for ln in tail_lines[-5:]:
            click.echo(f"    {ln}")

    if not alive:
        click.secho(
            "\nProbe is DEAD.  Run `meeting-scribe bench stop` to clean up the state file.",
            fg="red",
        )
        sys.exit(2)


@bench_group.command("stop")
@click.option("--force", is_flag=True, help="SIGKILL if SIGTERM doesn't terminate the probe.")
def bench_stop(force: bool) -> None:
    """Close the bench window: SIGTERM the SLO probe + remove state file."""
    state = _read_state()
    if state is None:
        click.secho("No bench window declared.", fg="yellow")
        sys.exit(0)

    pid = state.get("slo_probe_pid")
    if pid and _slo_probe_alive(pid):
        click.echo(f"==> SIGTERM SLO probe pid {pid}")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        # Wait up to 5 s for the probe to exit cleanly.
        for _ in range(10):
            if not _slo_probe_alive(pid):
                break
            time.sleep(0.5)

        if _slo_probe_alive(pid):
            if force:
                click.secho(f"==> SIGKILL SLO probe pid {pid}", fg="yellow")
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                click.secho(
                    f"Probe pid {pid} still alive after SIGTERM.  Run with --force to SIGKILL.",
                    fg="red",
                )
                sys.exit(2)

    STATE_FILE.unlink(missing_ok=True)
    click.secho("Bench window CLOSED.", fg="green", bold=True)
    click.echo(f"  log preserved at: {state.get('slo_probe_log', DEFAULT_LOG)}")
