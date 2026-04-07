"""Meeting Scribe CLI — manage server lifecycle and development tools.

Usage:
    meeting-scribe start     # Start the server
    meeting-scribe stop      # Stop gracefully
    meeting-scribe restart   # Stop + start
    meeting-scribe status    # Check server + backend status
    meeting-scribe test      # Run E2E pipeline test
    meeting-scribe logs      # Tail server logs
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import click

PROJECT_ROOT = Path(__file__).parent.parent.parent
_TMPDIR = Path(tempfile.gettempdir())
LOG_FILE = _TMPDIR / "meeting-scribe.log"
PID_FILE = _TMPDIR / "meeting-scribe.pid"
DEFAULT_PORT = 8080


def _get_pid() -> int | None:
    """Read the server PID from the PID file."""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            # Check if process is alive
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            PID_FILE.unlink(missing_ok=True)
    return None


def _server_url(port: int = DEFAULT_PORT) -> str:
    return f"http://127.0.0.1:{port}"


def _api_request(path: str, method: str = "GET", port: int = DEFAULT_PORT) -> dict | None:
    """Make an API request to the running server."""
    try:
        req = urllib.request.Request(f"{_server_url(port)}{path}", method=method)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """Meeting Scribe — real-time bilingual transcription."""


@cli.command()
@click.option("--port", "-p", default=DEFAULT_PORT, help="Server port")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--foreground", "-f", is_flag=True, help="Run in foreground (not daemonized)")
def start(port: int, debug: bool, foreground: bool) -> None:
    """Start the meeting-scribe server."""
    existing = _get_pid()
    if existing:
        click.secho(f"Server already running (PID {existing})", fg="yellow")
        click.echo(f"  URL: {_server_url(port)}")
        return

    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python3"
    if not venv_python.exists():
        click.secho("No .venv found. Run: python3 -m venv .venv && pip install -e .", fg="red")
        sys.exit(1)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["SCRIBE_PORT"] = str(port)

    # Load .env file if present (for HF_TOKEN etc.)
    dotenv = PROJECT_ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                env.setdefault(key.strip(), val.strip())

    log_level = "debug" if debug else "info"

    cmd = [
        str(venv_python),
        "-m",
        "uvicorn",
        "meeting_scribe.server:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        log_level,
    ]

    if foreground:
        click.secho(f"Starting server on {_server_url(port)} (foreground)...", fg="cyan")
        try:
            subprocess.run(cmd, env=env, check=True)
        except KeyboardInterrupt:
            click.echo("\nStopped.")
        return

    # Daemonize
    click.echo(f"Starting server on {_server_url(port)}...")

    with open(LOG_FILE, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(PROJECT_ROOT),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    PID_FILE.write_text(str(proc.pid))

    # Wait for startup (Ollama warmup can take 60s+ on cold start)
    for _i in range(120):
        time.sleep(1)
        data = _api_request("/api/status", port=port)
        if data:
            click.secho(f"Server ready (PID {proc.pid})", fg="green")
            click.echo(f"  URL:       {_server_url(port)}")
            click.echo(f"  ASR:       {'✓' if data['backends']['asr'] else '✗'}")
            click.echo(f"  Translate: {'✓' if data['backends']['translate'] else '✗'}")
            click.echo(f"  Diarize:   {'✓' if data['backends']['diarize'] else '✗'}")
            click.echo(f"  Logs:      {LOG_FILE}")
            return

        # Check if process died
        if proc.poll() is not None:
            click.secho("Server failed to start. Check logs:", fg="red")
            click.echo(f"  {LOG_FILE}")
            PID_FILE.unlink(missing_ok=True)
            sys.exit(1)

    click.secho("Server startup timed out (120s). Check logs.", fg="red")


@cli.command()
def stop() -> None:
    """Stop the meeting-scribe server."""
    pid = _get_pid()
    if not pid:
        click.echo("Server not running.")
        return

    click.echo(f"Stopping server (PID {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait for graceful shutdown
        for _ in range(10):
            time.sleep(0.5)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
        else:
            # Force kill
            os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    PID_FILE.unlink(missing_ok=True)
    click.secho("Server stopped.", fg="green")


@cli.command()
@click.option("--port", "-p", default=DEFAULT_PORT)
@click.option("--debug", is_flag=True)
def restart(port: int, debug: bool) -> None:
    """Restart the server."""
    from click.testing import CliRunner

    runner = CliRunner()
    runner.invoke(stop)
    time.sleep(1)
    runner.invoke(start, [f"--port={port}"] + (["--debug"] if debug else []))


@cli.command()
@click.option("--port", "-p", default=DEFAULT_PORT)
def status(port: int) -> None:
    """Check server and backend status."""
    pid = _get_pid()
    if not pid:
        click.secho("Server: not running", fg="red")
        return

    data = _api_request("/api/status", port=port)
    if not data:
        click.secho(f"Server: running (PID {pid}) but not responding", fg="yellow")
        return

    click.secho(f"Server: running (PID {pid})", fg="green")
    click.echo(f"  URL:       {_server_url(port)}")
    click.echo(f"  ASR:       {'✓ active' if data['backends']['asr'] else '✗ disabled'}")
    click.echo(f"  Translate: {'✓ active' if data['backends']['translate'] else '✗ disabled'}")
    click.echo(f"  Diarize:   {'✓ active' if data['backends']['diarize'] else '✗ disabled'}")

    meeting = data.get("meeting")
    if meeting and meeting.get("state"):
        state_color = "green" if meeting["state"] == "recording" else "yellow"
        click.secho(f"  Meeting:   {meeting['state']} ({meeting['id'][:8]}...)", fg=state_color)
    else:
        click.echo("  Meeting:   none")

    click.echo(f"  WSS:       {data.get('connections', 0)} connections")


@cli.command()
@click.option("--lines", "-n", default=50, help="Number of lines to show")
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
def logs(lines: int, follow: bool) -> None:
    """View server logs."""
    if not LOG_FILE.exists():
        click.echo("No log file found. Start the server first.")
        return

    cmd = ["tail", f"-n{lines}"]
    if follow:
        cmd.append("-f")
    cmd.append(str(LOG_FILE))

    with contextlib.suppress(KeyboardInterrupt):
        subprocess.run(cmd, check=True)


@cli.command()
@click.option(
    "--text", "-t", default="こんにちは、今日の会議を始めましょう。", help="Japanese text for TTS"
)
@click.option("--port", "-p", default=DEFAULT_PORT)
def test(text: str, port: int) -> None:
    """Run an end-to-end pipeline test using macOS TTS."""
    data = _api_request("/api/status", port=port)
    if not data:
        click.secho("Server not running. Start it first: meeting-scribe start", fg="red")
        sys.exit(1)

    click.echo("Running E2E pipeline test...")
    test_script = PROJECT_ROOT / "scripts" / "test_e2e.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    subprocess.run(
        [str(PROJECT_ROOT / ".venv" / "bin" / "python3"), str(test_script), "--tts", text],
        env=env,
        check=False,
    )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
