"""Meeting Scribe CLI — manage server lifecycle and development tools.

Usage:
    meeting-scribe start     # Start the server
    meeting-scribe stop      # Stop gracefully
    meeting-scribe restart   # Stop + start
    meeting-scribe status    # Check server + backend status
    meeting-scribe logs      # Tail server logs
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import click

PROJECT_ROOT = Path(__file__).parent.parent.parent
_TMPDIR = Path(tempfile.gettempdir())
LOG_FILE = _TMPDIR / "meeting-scribe.log"
PID_FILE = _TMPDIR / "meeting-scribe.pid"
DEFAULT_PORT = 8080
GUEST_PORT = 80
# Hotspot AP IP — mirrors server.AP_IP, duplicated here so the CLI can
# print the guest URL without importing the heavy server module.
AP_IP = "10.42.0.1"


def _read_pid(pid_file: Path) -> int | None:
    """Return the PID from `pid_file` if the process is alive, else None."""
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        pid_file.unlink(missing_ok=True)
        return None


def _get_pid() -> int | None:
    """Read the server PID from the PID file (legacy path, foreground/dev)."""
    return _read_pid(PID_FILE)


_SYSTEMD_UNIT = "meeting-scribe.service"


def _server_state() -> tuple[str, int | None, str | None]:
    """Return ``(mode, pid, active_state)`` for the meeting-scribe server.

    Systemd is authoritative when present: the mode is ``"systemd"`` and
    ``pid`` / ``active_state`` come straight from ``systemctl --user show``.
    If the user manager does not know the unit — e.g. someone is running
    ``meeting-scribe start`` by hand in dev — we fall back to the legacy
    ``/tmp/meeting-scribe.pid`` file and report mode ``"foreground"``.

    Returning a tuple instead of just a pid lets ``stop`` / ``restart``
    dispatch through ``systemctl --user stop|restart`` (so ``ExecStopPost``
    fires and ``Restart=on-failure`` does not treat a clean stop as a
    crash) when the systemd path is live, while keeping the old signal-
    based path for foreground runs.
    """
    try:
        r = subprocess.run(
            [
                "systemctl", "--user", "show", _SYSTEMD_UNIT,
                "-p", "MainPID", "-p", "ActiveState", "-p", "LoadState",
            ],
            capture_output=True, text=True, check=False, timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ("foreground", _get_pid(), None)

    if r.returncode != 0 or not r.stdout.strip():
        return ("foreground", _get_pid(), None)

    # `systemctl show` emits `Key=value` pairs. Without `--value` the order
    # is stable (alphabetical), but we parse by key to be robust either way.
    kv: dict[str, str] = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            kv[k.strip()] = v.strip()

    # LoadState=not-found means the unit isn't installed — treat like foreground.
    if kv.get("LoadState") != "loaded":
        return ("foreground", _get_pid(), None)

    try:
        pid = int(kv.get("MainPID", "0")) or None
    except ValueError:
        pid = None
    if pid is not None:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            pid = None
    return ("systemd", pid, kv.get("ActiveState") or None)


def _venv_python() -> Path:
    return PROJECT_ROOT / ".venv" / "bin" / "python3"


def _ensure_port80_bind() -> tuple[bool, str]:
    """Ensure the venv python has CAP_NET_BIND_SERVICE (port < 1024).

    The in-process guest listener binds port 80 directly from the server
    process, so the venv python needs the capability before ``meeting-scribe
    start`` runs. Idempotent: fast path if the cap is already set;
    otherwise attempts ``sudo -n setcap`` so fresh deploys succeed without
    interactive prompts.

    Returns (granted, detail).
    """
    venv_py = _venv_python()
    if not venv_py.exists():
        return False, f"venv python not found at {venv_py}"

    real_py = str(venv_py.resolve())

    if not shutil.which("getcap"):
        return False, "getcap not installed (install libcap2-bin)"

    existing = subprocess.run(["getcap", real_py], capture_output=True, text=True, check=False)
    if "cap_net_bind_service" in existing.stdout:
        return True, real_py

    if not shutil.which("setcap"):
        return False, "setcap not installed (install libcap2-bin)"

    # Non-interactive sudo: works on deploy boxes with passwordless sudo;
    # fails quickly elsewhere so we can print a clear hint.
    grant = subprocess.run(
        ["sudo", "-n", "setcap", "cap_net_bind_service=+ep", real_py],
        capture_output=True,
        text=True,
        check=False,
    )
    if grant.returncode == 0:
        return True, real_py

    hint = f"run one-time:\n    sudo setcap 'cap_net_bind_service=+ep' {real_py}"
    return False, hint


def _management_ip() -> str:
    """Return the LAN/management IP, matching server.main()'s detection.

    Parses ``ip route get 1.1.1.1`` and extracts ``src``. Honours the
    ``SCRIBE_MANAGEMENT_IP`` override so tests and unusual multi-homed
    boxes can pin the address. Falls back to ``127.0.0.1`` only if
    detection outright fails — that's a lousy fallback for a real admin
    listener but keeps CLI helpers usable in containers and CI.
    """
    override = os.environ.get("SCRIBE_MANAGEMENT_IP", "").strip()
    if override:
        return override
    try:
        out = subprocess.run(
            ["ip", "-4", "route", "get", "1.1.1.1"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        ).stdout
        tokens = out.split()
        if "src" in tokens:
            return tokens[tokens.index("src") + 1]
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "127.0.0.1"


def _server_url(port: int = DEFAULT_PORT) -> str:
    return f"https://{_management_ip()}:{port}"


def _api_request(path: str, method: str = "GET", port: int = DEFAULT_PORT) -> dict | None:
    """Make an API request to the admin listener on the management IP.

    The admin socket binds to the management IP only (not 0.0.0.0), so
    we must hit that address — 127.0.0.1 no longer routes to it.
    """
    import ssl

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    host = _management_ip()
    url = f"https://{host}:{port}{path}"
    try:
        req = urllib.request.Request(url, method=method)
        with urllib.request.urlopen(req, timeout=5, context=ctx) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


_COMPOSE_FILE = PROJECT_ROOT / "docker-compose.gb10.yml"
_REQUIRED_CONTAINERS = ["scribe-diarization", "scribe-tts", "scribe-asr"]


def _ensure_containers_running() -> None:
    """Auto-launch model containers via docker compose if not already running.

    Checks each required container's status and brings up the compose stack
    if any are missing. This makes ``meeting-scribe start`` self-contained —
    users don't need to remember a separate ``docker compose up`` step.
    """
    if not _COMPOSE_FILE.exists():
        return  # Not on GB10 or compose file missing — skip silently

    if not shutil.which("docker"):
        return  # Docker not installed — server will degrade gracefully

    # Check which required containers are running
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        running = set(result.stdout.strip().splitlines())
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return  # Can't check — let the server handle it

    missing = [c for c in _REQUIRED_CONTAINERS if c not in running]
    if not missing:
        return  # All containers running

    click.echo(f"Starting model containers ({', '.join(missing)})...")
    try:
        subprocess.run(
            ["docker", "compose", "-f", str(_COMPOSE_FILE), "up", "-d", "--pull", "never"],
            cwd=str(PROJECT_ROOT),
            timeout=60,
            check=True,
        )
        click.secho("Containers launched", fg="green")
    except subprocess.CalledProcessError as e:
        click.secho(f"Container launch failed (exit {e.returncode}) — server will retry backends", fg="yellow")
    except subprocess.TimeoutExpired:
        click.secho("Container launch timed out — containers may still be starting", fg="yellow")


@click.group()
@click.version_option(version="1.3.0")
def cli() -> None:
    """Meeting Scribe — real-time bilingual transcription."""


@cli.command("setup")
def first_run_setup():
    """First-time setup — install dependencies, configure credentials, pull models."""
    import shutil

    click.secho("=== Meeting Scribe Setup ===", fg="cyan", bold=True)
    click.echo()

    issues: list[str] = []

    # 1. Check / create .venv
    venv_dir = PROJECT_ROOT / ".venv"
    if venv_dir.exists():
        click.secho("[OK] .venv exists", fg="green")
    else:
        click.echo("Creating virtual environment...")
        result = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            click.secho("[OK] .venv created", fg="green")
        else:
            click.secho("[FAIL] Could not create .venv", fg="red")
            issues.append("venv creation failed")

    # 2. Check pip install -e .
    pip_bin = venv_dir / "bin" / "pip"
    if pip_bin.exists():
        result = subprocess.run(
            [str(pip_bin), "show", "meeting-scribe"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            click.secho("[OK] meeting-scribe installed (editable)", fg="green")
        else:
            click.echo("Installing meeting-scribe in editable mode...")
            result = subprocess.run(
                [str(pip_bin), "install", "-e", str(PROJECT_ROOT)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                click.secho("[OK] meeting-scribe installed", fg="green")
            else:
                click.secho("[FAIL] pip install failed", fg="red")
                issues.append("pip install -e . failed")
    else:
        issues.append(".venv/bin/pip not found")

    # 3. Check HF_TOKEN
    dotenv = PROJECT_ROOT / ".env"
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token and dotenv.exists():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                hf_token = line.partition("=")[2].strip()
                break

    if hf_token:
        click.secho(f"[OK] HF_TOKEN set ({hf_token[:4]}...)", fg="green")
    else:
        click.secho("[MISSING] HF_TOKEN not found", fg="yellow")
        token = click.prompt(
            "Enter your HuggingFace token (or press Enter to skip)", default="", show_default=False
        )
        if token.strip():
            # Append to .env
            with open(dotenv, "a") as f:
                f.write(f"\nHF_TOKEN={token.strip()}\n")
            click.secho("[OK] HF_TOKEN saved to .env", fg="green")
        else:
            issues.append("HF_TOKEN not configured (needed for model downloads)")

    # 4. TLS certs
    certs_dir = PROJECT_ROOT / "certs"
    cert_pem = certs_dir / "cert.pem"
    key_pem = certs_dir / "key.pem"
    if cert_pem.exists() and key_pem.exists():
        click.secho("[OK] TLS certs exist in certs/", fg="green")
    else:
        click.echo("Generating self-signed TLS certificates...")
        certs_dir.mkdir(exist_ok=True)
        result = subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-keyout",
                str(key_pem),
                "-out",
                str(cert_pem),
                "-days",
                "365",
                "-nodes",
                "-subj",
                "/CN=meeting-scribe",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            click.secho("[OK] Self-signed certs generated", fg="green")
        else:
            click.secho("[FAIL] openssl not available or cert generation failed", fg="red")
            issues.append("TLS cert generation failed")

    # 5. Port 80 bind capability for the in-process guest HTTP listener
    granted, detail = _ensure_port80_bind()
    if granted:
        click.secho(f"[OK] Python can bind port 80 ({detail})", fg="green")
    else:
        click.secho("[WARN] Port 80 bind capability not granted", fg="yellow")
        for line in detail.splitlines():
            click.echo(f"  {line}")
        issues.append("port 80 bind capability not granted (guest listener will fail)")

    # 6. Docker
    if shutil.which("docker"):
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            click.secho("[OK] Docker is available", fg="green")
        else:
            click.secho("[WARN] Docker installed but not running or no permission", fg="yellow")
            issues.append("Docker not accessible (check daemon / group membership)")
    else:
        click.secho("[MISSING] Docker not found", fg="yellow")
        issues.append("Docker not installed")

    # Summary
    click.echo()
    if issues:
        click.secho(f"Setup complete with {len(issues)} issue(s):", fg="yellow")
        for issue in issues:
            click.echo(f"  - {issue}")
    else:
        click.secho("Setup complete! All checks passed.", fg="green")
    click.echo()
    click.echo("Next steps:")
    click.echo("  meeting-scribe gb10 up      # Start model containers")
    click.echo("  meeting-scribe start         # Start the server")


@cli.command()
@click.option("--host", envvar="SCRIBE_GB10_HOST", default="localhost")
def up(host):
    """Start all model containers (alias for gb10 up)."""
    gb10_up.callback(host=host)


@cli.command()
@click.option("--host", envvar="SCRIBE_GB10_HOST", default="localhost")
def down(host):
    """Stop all model containers (alias for gb10 down)."""
    gb10_down.callback(host=host)


@cli.command()
@click.option("--host", envvar="SCRIBE_GB10_HOST", default="localhost")
def containers(host):
    """Check container status (alias for gb10 status)."""
    gb10_status.callback(host=host)


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
    """Scan meeting-scribe working tree for sensitive data before commit.

    Uses the vendored ``precommit_scanner`` module — no external tool
    required. Flags embedded journal entries, recording paths, audio
    references, credentials, and LAN identity.
    """
    from meeting_scribe import precommit_scanner

    repo_root = Path(__file__).resolve().parents[2]
    ctx.invoke(
        precommit_scanner.precommit,
        repo=repo_root,
        include_staged=True,
        include_all_tracked=include_all_tracked,
        verbose=verbose,
        warn_only=warn_only,
    )


@cli.command()
@click.option("--port", "-p", default=DEFAULT_PORT, help="Admin HTTPS port")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--foreground", "-f", is_flag=True, help="Run in foreground (not daemonized)")
def start(port: int, debug: bool, foreground: bool) -> None:
    """Start the meeting-scribe server (admin HTTPS + guest HTTP)."""
    # When systemd owns the service, always delegate instead of spawning
    # a parallel daemon process. Running `meeting-scribe start` directly
    # used to create a Popen-daemonized child with PPID=1 that was
    # completely invisible to the systemd unit — when systemd later tried
    # to start its own instance, it would race with the stray and land
    # the unit in a restart loop. The only exception is --foreground,
    # which is the entry point systemd itself uses via ExecStart.
    if not foreground:
        mode, sd_pid, active = _server_state()
        if mode == "systemd":
            if active == "active":
                click.secho(f"Server already running via systemd (PID {sd_pid})", fg="yellow")
                click.echo(f"  URL: {_server_url(port)}")
                return
            click.echo("Starting meeting-scribe.service via systemctl --user...")
            rc = subprocess.run(
                ["systemctl", "--user", "start", _SYSTEMD_UNIT],
                check=False,
            ).returncode
            if rc != 0:
                click.secho(f"systemctl --user start failed (rc={rc})", fg="red")
                sys.exit(rc)
            # Re-query to surface the new PID + URL to the user.
            _, sd_pid, _ = _server_state()
            click.echo(f"Server started via systemd (PID {sd_pid})")
            click.echo(f"  URL: {_server_url(port)}")
            return

    existing = _get_pid()
    if existing:
        # In foreground mode this is almost always systemd taking over from an
        # out-of-band user launch. Silently returning 0 here is catastrophic:
        # Type=notify never gets READY=1, systemd marks the unit
        # `Result: protocol` and restarts — a tight loop that trips
        # StartLimitBurst in seconds while the stray process keeps running
        # unsupervised. Reclaim the PID instead.
        if foreground:
            click.secho(
                f"Reclaiming conflicting server (PID {existing}) for foreground takeover",
                fg="yellow",
            )
            try:
                os.kill(existing, signal.SIGTERM)
                for _ in range(30):
                    time.sleep(0.5)
                    try:
                        os.kill(existing, 0)
                    except ProcessLookupError:
                        break
                else:
                    os.kill(existing, signal.SIGKILL)
                    time.sleep(0.5)
            except ProcessLookupError:
                pass
            PID_FILE.unlink(missing_ok=True)
        else:
            click.secho(f"Server already running (PID {existing})", fg="yellow")
            click.echo(f"  URL: {_server_url(port)}")
            return

    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python3"
    if not venv_python.exists():
        click.secho("No .venv found. Run: python3 -m venv .venv && pip install -e .", fg="red")
        sys.exit(1)

    # Guest HTTP listener binds port 80 in-process — requires
    # CAP_NET_BIND_SERVICE on the venv interpreter. Grant it before spawn.
    granted, detail = _ensure_port80_bind()
    if not granted:
        click.secho("Cannot bind port 80 (guest listener):", fg="red")
        for line in detail.splitlines():
            click.echo(f"  {line}")
        sys.exit(1)

    # Auto-launch model containers if not already running
    _ensure_containers_running()

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
    env.setdefault("SCRIBE_LOG_LEVEL", log_level)

    # Dual-listener entry point lives in meeting_scribe.server:main(). It
    # binds admin HTTPS to the detected management IP and guest HTTP to
    # the hotspot AP IP (via IP_FREEBIND), sharing one FastAPI app so
    # in-process globals (TTS queue, audio-out clients, current meeting)
    # are shared between the two listeners.
    #
    # Invoked via ``-c`` rather than ``-m`` so the module loads under its
    # canonical name (``meeting_scribe.server``) and nothing that does
    # ``from meeting_scribe.server import X`` later ends up with a second
    # copy of module-level globals.
    cmd = [
        str(venv_python),
        "-c",
        "from meeting_scribe.server import main; main()",
    ]

    # Verify admin TLS certs exist before we spawn — server.main() will
    # raise RuntimeError otherwise and we'd rather surface it here.
    ssl_key = PROJECT_ROOT / "certs" / "key.pem"
    ssl_cert = PROJECT_ROOT / "certs" / "cert.pem"
    if not (ssl_key.exists() and ssl_cert.exists()):
        click.secho(
            f"Admin TLS certs missing: expected {ssl_key} and {ssl_cert}.",
            fg="red",
        )
        click.echo("  Run: meeting-scribe setup")
        sys.exit(1)

    management_ip = _management_ip()

    if foreground:
        click.secho(
            f"Starting server (admin=https://{management_ip}:{port}, "
            f"guest=http://{AP_IP}:{GUEST_PORT}) foreground...",
            fg="cyan",
        )
        # exec-replace the click wrapper with the server process so the
        # server *is* MainPID under systemd. Previously we ran it as a
        # subprocess.run() child; with Type=notify + NotifyAccess=main
        # that meant sd_notify("READY=1") came from the child PID and
        # systemd silently dropped it ("reception only permitted for
        # main PID"), leaving the unit stuck in `activating (start)`
        # until TimeoutStartSec. execv-ing also means Ctrl-C, SIGTERM,
        # and the systemd cgroup walk all see a single process instead
        # of a wrapper+child pair.
        os.execvpe(cmd[0], cmd, env)

    # Daemonize
    click.echo(
        f"Starting server: admin=https://{management_ip}:{port}, guest=http://{AP_IP}:{GUEST_PORT}"
    )

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

    # Wait for startup — model loading can take several minutes on first run
    timeout_s = 300
    click.echo(f"Waiting for server (up to {timeout_s}s)...")
    for _i in range(timeout_s):
        time.sleep(1)
        # Check process health first — if the child died (e.g. port already
        # in use), don't be fooled by an /api/status 200 from an unrelated
        # stale server already holding the port.
        if proc.poll() is not None:
            click.secho("Server failed to start. Check logs:", fg="red")
            click.echo(f"  {LOG_FILE}")
            PID_FILE.unlink(missing_ok=True)
            sys.exit(1)

        data = _api_request("/api/status", port=port)
        if data:
            click.secho(f"Server ready (PID {proc.pid})", fg="green")
            click.echo(f"  Admin:     https://{management_ip}:{port}")
            click.echo(f"  Guest:     http://{AP_IP}:{GUEST_PORT} (hotspot only)")
            click.echo(f"  ASR:       {'✓' if data['backends']['asr'] else '✗'}")
            click.echo(f"  Translate: {'✓' if data['backends']['translate'] else '✗'}")
            click.echo(f"  Diarize:   {'✓' if data['backends']['diarize'] else '✗'}")
            click.echo(f"  TTS:       {'✓' if data['backends']['tts'] else '✗'}")
            click.echo(f"  Logs:      {LOG_FILE}")
            return

        # Progress indicator every 30s
        if (_i + 1) % 30 == 0:
            click.echo(f"  Still waiting... ({_i + 1}s)")

    click.secho(f"Server startup timed out ({timeout_s}s). Check logs:", fg="red")
    click.echo(f"  {LOG_FILE}")


@cli.command()
def stop() -> None:
    """Stop the meeting-scribe server.

    Delegates to ``systemctl --user stop`` when the unit is managed by
    systemd so that ``ExecStopPost`` fires and ``Restart=on-failure`` does
    not interpret the stop as a crash. Falls back to SIGTERM on the PID
    file for foreground/dev runs.
    """
    mode, pid, active = _server_state()

    if mode == "systemd":
        if active in (None, "inactive", "failed"):
            # Systemd isn't running the service, but there may still be a
            # stray PID-file daemon left behind by an older `meeting-scribe
            # start` (pre-systemd-delegation) or a manual launch. Clean it
            # up so subsequent systemctl starts don't race with it.
            stray = _get_pid()
            if stray:
                click.echo(f"Systemd inactive but found stray PID {stray} — terminating.")
                try:
                    os.kill(stray, signal.SIGTERM)
                    for _ in range(20):
                        time.sleep(0.5)
                        try:
                            os.kill(stray, 0)
                        except ProcessLookupError:
                            break
                    else:
                        os.kill(stray, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                PID_FILE.unlink(missing_ok=True)
                click.secho("Stray server stopped.", fg="green")
                return
            click.echo(f"Server not running (systemd ActiveState={active or 'unknown'}).")
            return
        click.echo(f"Stopping server via systemctl --user (PID {pid}, ActiveState={active})...")
        r = subprocess.run(
            ["systemctl", "--user", "stop", _SYSTEMD_UNIT],
            capture_output=True, text=True, check=False,
        )
        if r.returncode == 0:
            click.secho("Server stopped.", fg="green")
        else:
            click.secho(f"systemctl --user stop failed: {r.stderr.strip()}", fg="red")
            sys.exit(r.returncode or 1)
        return

    # Foreground / dev fallback.
    if not pid:
        click.echo("Server not running.")
        return

    click.echo(f"Stopping server (PID {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            time.sleep(0.5)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
        else:
            os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    PID_FILE.unlink(missing_ok=True)
    click.secho("Server stopped.", fg="green")


@cli.command()
@click.option("--port", "-p", default=DEFAULT_PORT)
@click.option("--debug", is_flag=True)
def restart(port: int, debug: bool) -> None:
    """Restart the server + run a smoke test so silent partial failures don't slip through.

    Delegates to ``systemctl --user restart`` when the unit is systemd-
    managed so preflight / notify semantics are preserved. Falls back to
    the in-process click stop+start dance for foreground/dev runs.
    """
    from pathlib import Path

    from click.testing import CliRunner

    mode, _pid, _active = _server_state()
    if mode == "systemd":
        click.echo(f"Restarting {_SYSTEMD_UNIT} via systemctl --user...")
        r = subprocess.run(
            ["systemctl", "--user", "restart", _SYSTEMD_UNIT],
            capture_output=True, text=True, check=False,
        )
        if r.returncode != 0:
            click.secho(f"systemctl --user restart failed: {r.stderr.strip()}", fg="red")
            sys.exit(r.returncode or 1)
    else:
        runner = CliRunner()
        runner.invoke(stop)
        time.sleep(1)
        runner.invoke(start, [f"--port={port}"] + (["--debug"] if debug else []))

    # Smoke test — surfaces broken startup (empty-reply-from-server,
    # wrong-port binds, crashed background loops). Runs best-effort:
    # a green start + green smoke is the contract; a red smoke still
    # leaves the server running so the user can inspect it.
    smoke = Path(__file__).resolve().parent.parent.parent / "scripts" / "smoke-test.sh"
    if smoke.exists():
        try:
            # Give backends a beat to settle before probing.
            time.sleep(2)
            result = subprocess.run(
                [str(smoke)],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            click.echo(result.stdout)
            if result.stderr:
                click.echo(result.stderr)
            if result.returncode == 0:
                pass  # smoke_test already printed GREEN
            elif result.returncode == 2:
                click.secho("Smoke test: optional components degraded (ok to proceed)", fg="yellow")
            else:
                click.secho("Smoke test FAILED — inspect logs before using the UI", fg="red")
        except subprocess.TimeoutExpired:
            click.secho("Smoke test timed out", fg="yellow")
        except Exception as e:
            click.secho(f"Smoke test crashed: {e}", fg="yellow")


@cli.command()
@click.option("--port", "-p", default=DEFAULT_PORT)
def status(port: int) -> None:
    """Check server and backend status."""
    mode, pid, active = _server_state()

    if mode == "systemd":
        if active not in ("active", "activating", "reloading"):
            click.secho(
                f"Server: not running (systemd ActiveState={active or 'unknown'})",
                fg="red",
            )
            return
        pid_display = f"PID {pid}" if pid else "no MainPID"
        running_label = f"running via systemd ({active}, {pid_display})"
    else:
        if not pid:
            click.secho("Server: not running", fg="red")
            return
        running_label = f"running (PID {pid}, foreground)"

    data = _api_request("/api/status", port=port)
    if not data:
        click.secho(f"Server: {running_label} but not responding on /api/status", fg="yellow")
        return

    click.secho(f"Server: {running_label}", fg="green")
    click.echo(f"  URL:       {_server_url(port)}")
    click.echo(f"  ASR:       {'✓ active' if data['backends']['asr'] else '✗ disabled'}")
    click.echo(f"  Translate: {'✓ active' if data['backends']['translate'] else '✗ disabled'}")
    click.echo(f"  Diarize:   {'✓ active' if data['backends']['diarize'] else '✗ disabled'}")

    meeting = data.get("meeting")
    if meeting and meeting.get("state"):
        state_color = "green" if meeting["state"] == "recording" else "yellow"
        click.secho(f"  Meeting:   {meeting['state']} ({meeting['id']})", fg=state_color)
    else:
        click.echo("  Meeting:   none")

    click.echo(f"  WSS:       {data.get('connections', 0)} connections")


@cli.command()
@click.option(
    "--mode",
    type=click.Choice(["manual", "precondition", "boot", "shutdown"]),
    default="manual",
    help=(
        "Which phases to run. 'manual' (default) runs all phases and prints a "
        "human report; 'precondition' runs Phase 0 only (used by systemd "
        "ExecCondition); 'boot' runs Phase 1+2 (used by ExecStartPre); "
        "'shutdown' runs the static subset (used by ExecStopPost)."
    ),
)
@click.option(
    "--wait",
    default=300.0,
    type=float,
    help="Total wall-clock budget in seconds for live checks (default 300).",
)
def preflight(mode: str, wait: float) -> None:
    """Run meeting-scribe preflight checks.

    This is the operator-facing entry point for the check runner in
    ``meeting_scribe.preflight``. In ``--mode=manual`` it runs every phase
    live, prints a human-readable report, and — on all-green — clears any
    stale ``BOOT_BLOCKED`` marker. It is the documented recovery path from
    a stuck boot. The other modes (``precondition``, ``boot``, ``shutdown``)
    are for systemd hooks and use reserved exit codes the unit file cares
    about.
    """
    from meeting_scribe import preflight as _pf

    if mode == "manual":
        sys.exit(_pf.cmd_manual(wait_seconds=wait))
    elif mode == "precondition":
        sys.exit(_pf.cmd_precondition())
    elif mode == "boot":
        sys.exit(_pf.cmd_boot(wait_seconds=wait))
    elif mode == "shutdown":
        sys.exit(_pf.cmd_shutdown())


@cli.command(name="shutdown")
@click.option(
    "--reboot", "action", flag_value="reboot",
    help="Reboot the machine after the service stops (only on green preflight).",
)
@click.option(
    "--poweroff", "action", flag_value="poweroff",
    help="Power off the machine after the service stops (only on green preflight).",
)
@click.option(
    "--force", is_flag=True,
    help="Proceed with stop/reboot/poweroff even if preflight fails. BOOT_BLOCKED is still written.",
)
@click.option(
    "--wait", default=300.0, type=float,
    help="Total wall-clock budget in seconds for live preflight (default 300).",
)
@click.option(
    "--reason", default="", help="Audit reason recorded in last-good-shutdown.",
)
def shutdown_cmd(action: str | None, force: bool, wait: float, reason: str) -> None:
    """Validate, then stop (and optionally reboot/poweroff) meeting-scribe.

    Runs the full ``preflight --mode=manual`` check set **before** doing
    anything destructive. On red without ``--force`` we print the report,
    write ``BOOT_BLOCKED`` with the failure reasons, and exit non-zero —
    the service stays up so the operator can fix the underlying problem.
    On green we stop the user unit (which fires ``ExecStopPost``) and, if
    ``--reboot`` or ``--poweroff`` was passed, hand control to ``systemctl
    reboot``/``poweroff``.

    This is the operator-facing shutdown gate: it is the *only* command
    we should use to take the machine down, because it is the only place
    that actually proves the system will come back up cleanly before we
    pull the plug.
    """
    from meeting_scribe import preflight as _pf

    ctx = _pf.make_context(wait)
    # The shutdown wrapper is a live-validation gate — an existing stale
    # BOOT_BLOCKED marker shouldn't prevent an operator from validating
    # and proceeding with a clean reboot.
    results = asyncio.run(_pf.run_all(ctx, skip_blocker=True))
    report = _pf.format_report(results)
    click.echo("meeting-scribe shutdown preflight")
    click.echo(report)

    exit_code = _pf.classify_exit(results)
    green = exit_code == _pf.EXIT_OK

    if not green:
        failure_reasons = "; ".join(
            f"{r.name}: {r.detail}" for r in results if not r.passed and not r.warn_only
        )
        _pf.write_boot_blocked(
            f"shutdown wrapper (mode={action or 'stop-only'}, force={force}, "
            f"reason={reason!r}): {failure_reasons}"
        )
        if not force:
            click.secho(
                "Preflight FAILED — refusing to stop the service. Fix the issues "
                "above and re-run, or pass --force to override (BOOT_BLOCKED will "
                "still be written so the next boot is gated).",
                fg="red",
            )
            sys.exit(exit_code)
        click.secho("Preflight FAILED but --force given; continuing.", fg="yellow")
    else:
        click.secho("Preflight GREEN", fg="green")

    # Write a small audit record of the shutdown intent before stopping.
    state_dir = _pf._state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "last-good-shutdown.json").write_text(
        json.dumps(
            {
                "ts": time.time(),
                "action": action or "stop-only",
                "forced": force,
                "green": green,
                "reason": reason,
            },
            indent=2,
        )
    )

    # Stop the user unit. Delegates to systemctl so ExecStopPost fires.
    mode_, _pid, active = _server_state()
    if mode_ == "systemd" and active not in (None, "inactive", "failed"):
        click.echo("Stopping meeting-scribe.service via systemctl --user...")
        r = subprocess.run(
            ["systemctl", "--user", "stop", _SYSTEMD_UNIT],
            capture_output=True, text=True, check=False,
        )
        if r.returncode != 0:
            click.secho(f"systemctl --user stop failed: {r.stderr.strip()}", fg="red")
            sys.exit(r.returncode or 1)
    else:
        click.echo(f"meeting-scribe not running via systemd (mode={mode_}); skipping stop.")

    if action == "reboot":
        click.echo("Executing systemctl reboot...")
        r = subprocess.run(["systemctl", "reboot"], capture_output=True, text=True, check=False)
        if r.returncode != 0:
            click.secho(f"systemctl reboot failed: {r.stderr.strip()}", fg="red")
            sys.exit(r.returncode or 1)
    elif action == "poweroff":
        click.echo("Executing systemctl poweroff...")
        r = subprocess.run(["systemctl", "poweroff"], capture_output=True, text=True, check=False)
        if r.returncode != 0:
            click.secho(f"systemctl poweroff failed: {r.stderr.strip()}", fg="red")
            sys.exit(r.returncode or 1)

    click.secho("Done.", fg="green")


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
@click.option("--port", "-p", default=DEFAULT_PORT)
def health(port: int) -> None:
    """Check all backends: ASR, Translation, TTS, Diarization, GPU."""
    import ssl
    import urllib.request

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    # Check server status
    data = _api_request("/api/status", port=port)
    if not data:
        click.secho("Server: not running or not responding", fg="red")
        return

    click.secho("Server: running", fg="green")

    # Backends
    backends = data.get("backends", {})
    for name, active in backends.items():
        color = "green" if active else "red"
        click.secho(f"  {name:12s} {'active' if active else 'inactive'}", fg=color)

    # GPU
    gpu = data.get("gpu")
    if gpu:
        pct = gpu.get("vram_pct", 0)
        color = "green" if pct < 80 else ("yellow" if pct < 90 else "red")
        click.secho(
            f"  GPU VRAM:    {gpu['vram_used_mb']}MB / {gpu['vram_total_mb']}MB ({pct}%)", fg=color
        )
    else:
        click.echo("  GPU VRAM:    not available")

    # Direct container health checks.  URLs come from ServerConfig so a
    # translate-endpoint move (e.g. Qwen3.5-INT4 on 8000 → Qwen3.6-FP8 on
    # 8010) stops reporting spurious "not responding".  We still hit each
    # container directly — this panel is the "bypass scribe, hit the
    # model server" diagnostic, so it must not piggyback on scribe's own
    # active/inactive flag.
    click.echo()
    click.echo("Container health:")
    from meeting_scribe.config import ServerConfig
    cfg: ServerConfig = ServerConfig.from_env()
    # Read the hot-reload translate URL if the operator flipped it for
    # a rollback; otherwise the static ServerConfig default.
    from meeting_scribe import runtime_config
    translate_url = (
        runtime_config.get("translate_url") or cfg.translate_vllm_url
    ).rstrip("/")
    asr_url = cfg.asr_vllm_url.rstrip("/")
    diarize_url = cfg.diarize_url.rstrip("/")
    # TTS is a comma-separated replica pool — ping the first replica for
    # the health indicator (all replicas share the same model).
    tts_primary = cfg.tts_vllm_url.split(",")[0].strip().rstrip("/")
    services = [
        ("ASR (vLLM)", asr_url),
        ("Translation (vLLM)", translate_url),
        ("Diarization", diarize_url),
        ("TTS (vLLM)", tts_primary),
    ]
    for name, url in services:
        try:
            req = urllib.request.Request(f"{url}/health")
            with urllib.request.urlopen(req, timeout=3) as resp:
                color = "green" if resp.status == 200 else "yellow"
                click.secho(f"  {name:24s} {url}: healthy", fg=color)
        except Exception:
            click.secho(f"  {name:24s} {url}: not responding", fg="red")

    # Meeting state
    meeting = data.get("meeting", {})
    if meeting.get("state"):
        click.echo(f"\n  Meeting: {meeting['state']} ({meeting['id']})")
    else:
        click.echo("\n  Meeting: none")


@cli.command()
@click.option("--port", "-p", default=DEFAULT_PORT)
def diagnose(port: int) -> None:
    """Full system diagnostic report."""

    click.secho("=== Meeting Scribe Diagnostic Report ===", fg="cyan", bold=True)
    click.echo()

    # Server
    pid = _get_pid()
    if pid:
        click.secho(f"Server PID: {pid}", fg="green")
    else:
        click.secho("Server: NOT RUNNING", fg="red")

    # API check
    data = _api_request("/api/status", port=port)
    if data:
        click.secho(f"API: responding on port {port}", fg="green")
        b = data.get("backends", {})
        for name, active in b.items():
            click.echo(f"  Backend {name}: {'active' if active else 'INACTIVE'}")
        m = data.get("metrics", {})
        click.echo(f"  Audio: {m.get('audio_s', 0):.0f}s, ASR events: {m.get('asr_finals', 0)}")
    else:
        click.secho("API: NOT RESPONDING", fg="red")

    # Docker containers
    click.echo()
    click.echo("Docker containers:")
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.strip().splitlines():
            parts = line.split("\t")
            name = parts[0]
            status = parts[1] if len(parts) > 1 else "unknown"
            color = "green" if "Up" in status else "red"
            click.secho(f"  {name:30s} {status}", fg=color)
    except Exception as e:
        click.secho(f"  Docker not available: {e}", fg="red")

    # Disk usage
    click.echo()
    meetings_dir = PROJECT_ROOT / "meetings"
    if meetings_dir.exists():
        total_size = sum(f.stat().st_size for f in meetings_dir.rglob("*") if f.is_file())
        num_meetings = len([d for d in meetings_dir.iterdir() if d.is_dir()])
        click.echo(f"Meetings: {num_meetings} meetings, {total_size / 1024 / 1024:.0f}MB disk")
    else:
        click.echo("Meetings: directory not found")

    # Python/package versions
    click.echo()
    click.echo(f"Python: {sys.version.split()[0]}")
    try:
        result = subprocess.run(
            [str(PROJECT_ROOT / ".venv" / "bin" / "pip"), "show", "meeting-scribe"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                click.echo(f"meeting-scribe: {line.split(': ')[1]}")
    except Exception:
        pass


@cli.command()
@click.option("--meeting-id", "-m", default=None, help="Specific meeting ID")
def finalize(meeting_id: str | None) -> None:
    """Finalize interrupted meetings — generate timeline and summary.

    Finds meetings stuck in 'interrupted' or 'recording' state and
    completes their post-processing (timeline, summary, state → complete).
    """
    import asyncio

    from meeting_scribe.summary import generate_summary

    meetings_dir = PROJECT_ROOT / "meetings"
    if not meetings_dir.exists():
        click.secho("No meetings directory found.", fg="red")
        return

    dirs = [meetings_dir / meeting_id] if meeting_id else sorted(meetings_dir.iterdir())

    finalized = 0
    for meeting_dir in dirs:
        if not meeting_dir.is_dir():
            continue
        meta_path = meeting_dir / "meta.json"
        if not meta_path.exists():
            continue

        import json as _json

        meta = _json.loads(meta_path.read_text())
        state = meta.get("state", "")
        mid = meeting_dir.name

        # `stopped` is a legacy state from an older schema (pre-MeetingState
        # enum cleanup) that still shows up on meetings recorded before the
        # refactor. Semantically it's the same as `interrupted` — a meeting
        # whose recording ended without a clean finalize — so fold it in
        # here instead of silently skipping and leaving the user stuck.
        if state not in ("interrupted", "recording", "stopped"):
            if meeting_id:
                click.echo(f"  {mid}: state is '{state}', not interrupted")
            continue

        journal = meeting_dir / "journal.jsonl"
        if not journal.exists():
            click.echo(f"  {mid}: no journal, skipping")
            continue

        click.echo(f"  {mid}: finalizing (was '{state}')...")

        # Generate timeline if missing
        timeline_path = meeting_dir / "timeline.json"
        if not timeline_path.exists():
            events = []
            for line in journal.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    e = _json.loads(line)
                except Exception:
                    continue
                if e.get("is_final") and e.get("text"):
                    sp = e.get("speakers", [])
                    events.append(
                        {
                            "segment_id": e.get("segment_id", ""),
                            "start_ms": e.get("start_ms", 0),
                            "end_ms": e.get("end_ms", 0),
                            "language": e.get("language", "unknown"),
                            "speaker_id": sp[0].get("cluster_id", 0) if sp else 0,
                            "text": e.get("text", "")[:100],
                        }
                    )
            pcm = meeting_dir / "audio" / "recording.pcm"
            duration_ms = int(pcm.stat().st_size / 32000 * 1000) if pcm.exists() else 0
            timeline_path.write_text(
                _json.dumps({"duration_ms": duration_ms, "segments": events}, indent=2)
            )
            click.echo(f"    timeline: {len(events)} segments")

        # Generate summary if missing
        summary_path = meeting_dir / "summary.json"
        if not summary_path.exists():
            try:
                summary = asyncio.run(
                    generate_summary(meeting_dir, vllm_url="http://localhost:8010")
                )
                if "error" not in summary:
                    click.secho(f"    summary: {len(summary.get('topics', []))} topics", fg="green")
                else:
                    click.secho(f"    summary: {summary['error']}", fg="yellow")
            except Exception as e:
                click.secho(f"    summary failed: {e}", fg="yellow")

        # Update state to complete
        meta["state"] = "complete"
        meta_path.write_text(_json.dumps(meta, indent=2))
        click.secho("    state → complete", fg="green")
        finalized += 1

    if finalized:
        click.secho(f"Finalized {finalized} meeting(s).", fg="green")
    elif not meeting_id:
        click.echo("No interrupted meetings found.")


def _resolve_meeting_ids(
    values: tuple[str, ...],
    meetings_dir: Path,
    require_explicit: bool,
) -> list[Path]:
    """Resolve ``-m`` option values to a list of meeting directories.

    Supports three forms, mixable in a single invocation:
      - literal meeting id (full UUID or 8-char prefix): ``-m 415bfa55``
      - stdin marker: ``-m -`` reads one id per line from stdin
      - file reference: ``-m @file.txt`` reads one id per line from file

    Blank lines and lines starting with ``#`` are ignored in both stdin
    and file forms so that the output of something like
    ``meeting-scribe library ls --ids-only`` pipes cleanly, and so users
    can annotate their id lists without breaking parsing.

    When ``values`` is empty and ``require_explicit`` is False, returns
    every meeting dir that has a ``journal.jsonl``. When empty and
    ``require_explicit`` is True, raises ``click.UsageError`` — used by
    destructive commands like ``full-reprocess`` that must never
    silently operate on every meeting.
    """
    ids: list[str] = []
    for v in values:
        if v == "-":
            for line in sys.stdin.read().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    ids.append(line)
            continue
        if v.startswith("@"):
            f = Path(v[1:]).expanduser()
            if not f.exists():
                raise click.BadParameter(f"id file not found: {f}")
            for line in f.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    ids.append(line)
            continue
        ids.append(v)

    if not ids:
        if require_explicit:
            raise click.UsageError(
                "no meeting ids given; use -m <id>, -m -, or -m @file.txt"
            )
        return sorted(
            [
                d
                for d in meetings_dir.iterdir()
                if d.is_dir() and (d / "journal.jsonl").exists()
            ],
            key=lambda d: d.stat().st_mtime,
        )

    resolved: list[Path] = []
    missing: list[str] = []
    for mid in ids:
        exact = meetings_dir / mid
        if exact.exists() and exact.is_dir():
            resolved.append(exact)
            continue
        # Prefix match (8-char or similar).
        candidates = [
            d for d in meetings_dir.iterdir() if d.is_dir() and d.name.startswith(mid)
        ]
        if len(candidates) == 1:
            resolved.append(candidates[0])
        elif len(candidates) == 0:
            missing.append(mid)
        else:
            raise click.BadParameter(
                f"ambiguous id {mid!r}: matches {len(candidates)} meetings"
            )
    if missing:
        raise click.BadParameter(f"meeting(s) not found: {', '.join(missing)}")
    return resolved


@cli.command()
@click.option(
    "--meeting-id",
    "-m",
    "meeting_ids",
    multiple=True,
    help=(
        "Specific meeting id (repeatable). Also accepts `-m -` to read ids "
        "from stdin and `-m @file.txt` to read from a file. Default: all meetings."
    ),
)
@click.option(
    "--vllm-url", default="http://localhost:8010", help="vLLM endpoint for summary generation"
)
def reprocess(meeting_ids: tuple[str, ...], vllm_url: str) -> None:
    """Re-process meetings: generate summaries and polished transcripts.

    Re-runs AI summary generation and refinement on completed meetings.
    Useful after model upgrades or when processing was interrupted.

    Pipe from `library ls --ids-only` for batch runs:
        meeting-scribe library ls --state complete --max-events 300 --ids-only \\
          | meeting-scribe reprocess -m -
    """
    import asyncio

    from meeting_scribe.summary import generate_summary

    meetings_dir = PROJECT_ROOT / "meetings"
    if not meetings_dir.exists():
        click.secho("No meetings directory found.", fg="red")
        return

    dirs = _resolve_meeting_ids(meeting_ids, meetings_dir, require_explicit=False)
    if not dirs:
        click.echo("No meetings matched.")
        return

    click.echo(f"Re-processing {len(dirs)} meeting(s)...")

    for meeting_dir in dirs:
        mid = meeting_dir.name
        journal = meeting_dir / "journal.jsonl"
        if not journal.exists():
            click.echo(f"  {mid}: no journal, skipping")
            continue

        existing_summary = meeting_dir / "summary.json"
        if existing_summary.exists():
            click.echo(f"  {mid}: summary already exists, regenerating...")
        else:
            click.echo(f"  {mid}: generating summary...")

        try:
            summary = asyncio.run(generate_summary(meeting_dir, vllm_url=vllm_url))
            if "error" in summary:
                click.secho(f"  {mid}: {summary['error']}", fg="yellow")
            else:
                topics = len(summary.get("topics", []))
                actions = len(summary.get("action_items", []))
                click.secho(f"  {mid}: {topics} topics, {actions} action items", fg="green")
        except Exception as e:
            click.secho(f"  {mid}: failed ({e})", fg="red")

    click.secho("Re-processing complete!", fg="green")


def _parse_duration_spec(spec: str | None) -> float | None:
    """Parse a duration like ``30m``, ``48h``, ``7d``, ``2w`` into hours.

    Returns None for empty/None input. Raises click.BadParameter on
    malformed input so the CLI layer surfaces a clean error instead of a
    stack trace. Kept here (rather than a util module) because everything
    that calls it is CLI-layer.
    """
    if not spec:
        return None
    import re

    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([mhdw])", spec.strip().lower())
    if not m:
        raise click.BadParameter(
            f"invalid duration {spec!r}; use e.g. '30m', '48h', '7d', '2w'"
        )
    value = float(m.group(1))
    unit = m.group(2)
    per_hour = {"m": 1 / 60, "h": 1.0, "d": 24.0, "w": 24.0 * 7}[unit]
    return value * per_hour


@cli.command()
@click.option("--dry-run", is_flag=True, help="Show what would happen without changing anything")
@click.option(
    "--yes", "-y", is_flag=True, help="Apply all actions without interactive confirmation"
)
@click.option(
    "--vllm-url", default="http://localhost:8010", help="vLLM endpoint for summary generation"
)
@click.option(
    "--min-events",
    type=int,
    default=0,
    help="Delete meetings with fewer than N journal events (default 0 = conservative). "
    "Applies to all meetings regardless of state. Use e.g. --min-events 200 to prune low-signal meetings.",
)
@click.option(
    "--older-than",
    default=None,
    help="Also delete meetings older than SPEC (e.g. 7d, 48h, 2w). Stacks with --min-events "
    "— a meeting must satisfy BOTH thresholds to be deleted when both are given.",
)
@click.option(
    "--exclude",
    default=None,
    help="Comma-separated list of meeting IDs (or 8-char prefixes) to protect from any action.",
)
@click.option(
    "--archive",
    "archive_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=(
        "Instead of permanently deleting, move meetings to this directory. "
        "Created on demand. Any existing entry with the same id is overwritten. "
        "Must not be inside the meetings dir (would cause re-audit loops). "
        "Use this for reversible cleanup runs on a cron."
    ),
)
def cleanup(
    dry_run: bool,
    yes: bool,
    vllm_url: str,
    min_events: int,
    older_than: str | None,
    exclude: str | None,
    archive_dir: Path | None,
) -> None:
    """Audit and clean up the meeting library.

    Actions performed (in order):
      1. Finalize interrupted/stopped meetings with substantial content (>60s audio, >5 events)
      2. Regenerate missing summaries for completed meetings with content
      3. Delete meetings below the event threshold and/or older than age threshold
         (default: only 0-event empties >1h old)
      4. Delete corrupt meeting directories (missing/unparseable meta.json)

    Flags:
      --min-events N      prune meetings with < N journal events
      --older-than SPEC   prune meetings older than SPEC (7d, 48h, 2w, 30m)
      --exclude ID[,ID]   protect specific meetings (full UUID or 8-char prefix)

    When both --min-events and --older-than are given, a meeting must satisfy
    BOTH thresholds to be deleted (AND semantics — conservative default).

    Legacy `stopped` state (pre-enum-cleanup schema) is treated as
    `interrupted` for finalization. High-audio / low-event meetings are
    flagged as suspected ASR failures before deletion.

    Use --dry-run to preview. Use --yes to skip the confirmation prompt.
    """
    import asyncio
    import json as _json
    import shutil as _shutil

    from meeting_scribe.config import ServerConfig
    from meeting_scribe.storage import MeetingStorage
    from meeting_scribe.summary import generate_summary

    older_than_hours = _parse_duration_spec(older_than)
    excluded: set[str] = set()
    if exclude:
        excluded = {x.strip() for x in exclude.split(",") if x.strip()}

    def _is_excluded(meeting_id: str) -> bool:
        return any(meeting_id == e or meeting_id.startswith(e) for e in excluded)

    config = ServerConfig.from_env()
    storage = MeetingStorage(config)
    audit = storage.audit_meetings()

    # Validate archive destination up front so we fail fast, before any
    # mutation. Refuse dirs inside the meetings root because the next
    # audit pass would scan them as bogus meetings.
    archive_resolved: Path | None = None
    if archive_dir is not None:
        archive_resolved = archive_dir.expanduser().resolve()
        meetings_root_resolved = Path(config.meetings_dir).resolve()
        try:
            archive_resolved.relative_to(meetings_root_resolved)
            raise click.BadParameter(
                f"--archive path {archive_resolved} is inside the meetings dir "
                f"{meetings_root_resolved}; pick a location outside it"
            )
        except ValueError:
            pass  # not inside meetings dir — good

    # Skip the currently-recording meeting
    active_file = Path("/tmp/meeting-scribe-active.json")
    active_id: str | None = None
    if active_file.exists():
        try:
            active_id = _json.loads(active_file.read_text()).get("meeting_id")
        except Exception:
            pass

    # Classify meetings. Note: `audit_meetings()` silently skips dirs with
    # missing/corrupt meta.json, so we enumerate the meetings root directly
    # to catch those as a separate "corrupt" bucket.
    to_finalize: list[dict] = []
    to_regen_summary: list[dict] = []
    to_delete: list[dict] = []
    to_delete_corrupt: list[Path] = []
    # Meetings that would be deleted but have >=60s of audio — almost
    # always an ASR failure, not a garbage recording. Surfaced as a
    # warning in the report so the user can rescue them with
    # `meeting-scribe full-reprocess` before deciding to delete.
    suspected_asr_failures: list[dict] = []

    audited_ids = {Path(m["meeting_dir"]).name for m in audit}
    meetings_root = Path(config.meetings_dir)
    if meetings_root.exists():
        for d in sorted(meetings_root.iterdir()):
            if (
                d.is_dir()
                and d.name not in audited_ids
                and d.name != "__pycache__"
                and not _is_excluded(d.name)
            ):
                to_delete_corrupt.append(d)

    for m in audit:
        if m["meeting_id"] == active_id:
            continue  # Never touch the active meeting
        if m["state"] == "recording":
            continue
        if _is_excluded(m["meeting_id"]):
            continue

        # Build the delete predicate. When both --min-events and --older-than
        # are set, require BOTH (AND semantics) — that matches user intent
        # of "trim small old meetings" rather than "trim anything small OR
        # anything old", which would be surprising and hard to undo.
        events_below = min_events > 0 and m["journal_lines"] < min_events
        age_above = older_than_hours is not None and m["age_hours"] > older_than_hours

        if min_events > 0 and older_than_hours is not None:
            delete_me = events_below and age_above
        elif min_events > 0:
            delete_me = events_below
        elif older_than_hours is not None:
            delete_me = age_above
        else:
            # Legacy conservative fallback: 0 events, 0 audio, >1h old.
            delete_me = (
                m["audio_duration_s"] < 5
                and m["journal_lines"] == 0
                and m["age_hours"] > 1
            )

        if delete_me:
            to_delete.append(m)
            # Suspected ASR failure: ≥60s of audio but a transcription rate
            # below ~5 events/min. Normal Japanese/English speech produces
            # 10–60 finalized events per minute; anything under 5 almost
            # always means the ASR backend was offline or misconfigured
            # when the meeting was recorded. We deliberately DO NOT key
            # this off the user's `--min-events` threshold — that value
            # expresses "noise level I don't care about", not "ASR was
            # broken", and conflating them would cry wolf on every prune.
            if m["audio_duration_s"] >= 60:
                events_per_min = m["journal_lines"] / (m["audio_duration_s"] / 60)
                if events_per_min < 5:
                    suspected_asr_failures.append(m)
            continue

        # Legacy `stopped` state is folded in with `interrupted` here (same
        # semantics — meeting ended without clean finalize). finalize command
        # also accepts it for standalone reruns.
        if (
            m["state"] in ("interrupted", "stopped")
            and m["audio_duration_s"] >= 60
            and m["journal_lines"] >= 5
        ):
            to_finalize.append(m)
            continue

        if m["state"] == "complete" and not m["has_summary"] and m["journal_lines"] >= 5:
            to_regen_summary.append(m)

    # Print audit
    click.secho("─" * 72, fg="cyan")
    click.secho(f"Meeting Library Audit — {len(audit)} total meetings", bold=True)
    click.secho("─" * 72, fg="cyan")
    action_word = "archive" if archive_resolved else "delete"
    click.echo(f"  Active (skipped):          {'1' if active_id else '0'}")
    click.echo(f"  To finalize (interrupted): {len(to_finalize)}")
    click.echo(f"  To regenerate summary:     {len(to_regen_summary)}")
    if min_events > 0:
        click.echo(f"  To {action_word} (<{min_events} events):    {len(to_delete)}")
    else:
        click.echo(f"  To {action_word} (empty):         {len(to_delete)}")
    click.echo(f"  To {action_word} (corrupt):       {len(to_delete_corrupt)}")
    if archive_resolved:
        click.secho(f"  Archive destination:       {archive_resolved}", fg="cyan")

    if to_finalize:
        click.echo()
        click.secho("Interrupted meetings to finalize:", fg="yellow")
        for m in to_finalize:
            click.echo(
                f"  {m['meeting_id']}  {m['audio_duration_s']:.0f}s audio  "
                f"{m['journal_lines']} events  age={m['age_hours']:.0f}h"
            )

    if to_regen_summary:
        click.echo()
        click.secho("Complete meetings missing summary:", fg="yellow")
        for m in to_regen_summary:
            click.echo(
                f"  {m['meeting_id']}  {m['audio_duration_s']:.0f}s audio  "
                f"{m['journal_lines']} events"
            )

    if to_delete:
        click.echo()
        verb = "archive" if archive_resolved else "delete"
        label = (
            f"Meetings to {verb} (<{min_events} events):"
            if min_events > 0
            else f"Empty meetings to {verb}:"
        )
        click.secho(label, fg="red")
        for m in to_delete:
            click.echo(
                f"  {m['meeting_id']}  state={m['state']}  "
                f"audio={m['audio_duration_s']:.0f}s  events={m['journal_lines']}  "
                f"age={m['age_hours']:.0f}h"
            )

    if to_delete_corrupt:
        click.echo()
        click.secho("Corrupt meeting dirs to delete (missing/unparseable meta.json):", fg="red")
        for d in to_delete_corrupt:
            try:
                size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            except OSError:
                size = 0
            click.echo(f"  {d.name}  size={size // 1024}KB")

    if suspected_asr_failures:
        click.echo()
        click.secho(
            "⚠  Suspected ASR failures — would be deleted but have substantial audio:",
            fg="yellow",
            bold=True,
        )
        click.echo(
            "   These meetings have ≥60s of recorded audio but very few journal"
        )
        click.echo(
            "   events. Likely ASR backend was offline/misconfigured when they"
        )
        click.echo(
            "   were recorded. Consider `meeting-scribe full-reprocess -m <id>`"
        )
        click.echo(
            "   before deletion, or `--exclude <id>` to protect them."
        )
        for m in suspected_asr_failures:
            click.echo(
                f"     {m['meeting_id']}  audio={m['audio_duration_s']:.0f}s  "
                f"events={m['journal_lines']}"
            )

    if excluded:
        click.echo()
        click.secho(
            f"Protected by --exclude (skipped from all actions): {len(excluded)} id(s)",
            fg="cyan",
        )

    if dry_run:
        click.echo()
        click.secho("(dry-run — no changes made)", fg="cyan")
        return

    if not (to_finalize or to_regen_summary or to_delete or to_delete_corrupt):
        click.echo()
        click.secho("Nothing to do — meeting library is clean!", fg="green")
        return

    if not yes:
        click.echo()
        if not click.confirm("Apply these changes?", default=False):
            click.echo("Aborted.")
            return

    # Apply: finalize + regen summary (both go through generate_summary)
    finalized = 0
    for m in to_finalize + to_regen_summary:
        meeting_dir = m["meeting_dir"]
        mid = m["meeting_id"]
        click.echo(f"  {mid}: generating summary...")

        # Generate summary (LLM call)
        try:
            summary = asyncio.run(generate_summary(meeting_dir, vllm_url=vllm_url))
            if "error" in summary:
                click.secho(f"  {mid}: summary failed: {summary['error']}", fg="yellow")
                continue

            # Transition state to complete if it was interrupted or
            # the legacy `stopped`. Leaving a finalized summary attached
            # to an "interrupted" meeting is a UI lie — the meeting's
            # output is done, the state should reflect that.
            meta_path = meeting_dir / "meta.json"
            if meta_path.exists():
                meta = _json.loads(meta_path.read_text())
                if meta.get("state") in ("interrupted", "stopped"):
                    meta["state"] = "complete"
                    meta_path.write_text(_json.dumps(meta, indent=2))

            topics = len(summary.get("topics", []))
            actions = len(summary.get("action_items", []))
            click.secho(
                f"  {mid}: ✓ {topics} topics, {actions} action items",
                fg="green",
            )
            finalized += 1
        except Exception as e:
            click.secho(f"  {mid}: failed ({e})", fg="red")

    # Helper: archive-or-delete. Archive is a reversible move to an
    # external directory; if an entry with the same name already exists
    # in the archive (re-running cleanup hit the same meeting twice)
    # we overwrite it — the newer copy is always the one the user cares
    # about, and refusing would leave the user with a surprise error
    # mid-run.
    def _remove_dir(src: Path) -> None:
        if archive_resolved is None:
            _shutil.rmtree(src)
            return
        archive_resolved.mkdir(parents=True, exist_ok=True)
        dest = archive_resolved / src.name
        if dest.exists():
            _shutil.rmtree(dest)
        _shutil.move(str(src), str(dest))

    verb_past = "archived" if archive_resolved else "deleted"

    # Apply: remove below-threshold meetings
    removed = 0
    for m in to_delete:
        meeting_dir = m["meeting_dir"]
        mid = m["meeting_id"]
        try:
            _remove_dir(meeting_dir)
            reason = f"<{min_events} events" if min_events > 0 else "empty"
            click.secho(f"  {mid}: {verb_past} ({reason})", fg="red")
            removed += 1
        except Exception as e:
            click.secho(f"  {mid}: {verb_past[:-1]} failed ({e})", fg="red")

    # Apply: remove corrupt directories
    removed_corrupt = 0
    for d in to_delete_corrupt:
        try:
            _remove_dir(d)
            click.secho(f"  {d.name}: {verb_past} (corrupt)", fg="red")
            removed_corrupt += 1
        except Exception as e:
            click.secho(f"  {d.name}: {verb_past[:-1]} failed ({e})", fg="red")

    click.echo()
    click.secho("─" * 72, fg="cyan")
    click.secho(
        f"Cleanup complete: {finalized} finalized, {removed} {verb_past}, "
        f"{removed_corrupt} corrupt {verb_past}",
        fg="green",
        bold=True,
    )
    if archive_resolved:
        click.secho(f"Archive: {archive_resolved}", fg="cyan")


@cli.command("full-reprocess")
@click.option(
    "--meeting-id",
    "-m",
    "meeting_ids",
    multiple=True,
    required=True,
    help=(
        "Meeting id to fully reprocess (repeatable). Also accepts `-m -` "
        "to read from stdin and `-m @file.txt` to read from a file. "
        "Required — full-reprocess never operates on the whole library silently."
    ),
)
@click.option("--asr-url", default="http://localhost:8003", help="ASR endpoint")
@click.option("--translate-url", default="http://localhost:8010", help="Translation endpoint")
@click.option(
    "--expected-speakers",
    type=click.IntRange(1, 12),
    default=None,
    help=(
        "Pin the speaker count when known. Constrains pyannote per-chunk "
        "and forces the cluster collapse to exactly N speakers."
    ),
)
def full_reprocess(
    meeting_ids: tuple[str, ...],
    asr_url: str,
    translate_url: str,
    expected_speakers: int | None,
) -> None:
    """Fully reprocess one or more meetings — re-run ASR + translation on raw audio.

    Reads the raw PCM recording, re-transcribes with Qwen3-ASR,
    translates all segments, and regenerates timeline + summary for
    each meeting. Original journal is backed up as journal.jsonl.bak.

    Examples:
      meeting-scribe full-reprocess -m 415bfa55
      meeting-scribe full-reprocess -m 415bfa55 -m dba10719
      meeting-scribe library ls --state complete --max-events 50 --ids-only \\
        | meeting-scribe full-reprocess -m -     # rescue suspected ASR failures
    """
    import asyncio
    import json as _json

    from meeting_scribe.reprocess import reprocess_meeting

    meetings_dir = PROJECT_ROOT / "meetings"
    dirs = _resolve_meeting_ids(meeting_ids, meetings_dir, require_explicit=True)

    successes = 0
    failures: list[tuple[str, str]] = []

    for meeting_dir in dirs:
        meeting_id = meeting_dir.name
        pcm = meeting_dir / "audio" / "recording.pcm"
        if not pcm.exists():
            click.secho(
                f"{meeting_id}: no recording.pcm found — skipping.", fg="yellow"
            )
            failures.append((meeting_id, "no recording.pcm"))
            continue

        duration_s = pcm.stat().st_size / (16000 * 2)

        # Read languages per meeting — mixing e.g. ja/en and it/en in
        # one batch must still respect each meeting's original config.
        # Length-1 (monolingual) meetings are respected too — reprocess
        # skips the translation pass end-to-end.
        meta_path = meeting_dir / "meta.json"
        language_pair: list[str] = ["en", "ja"]
        if meta_path.exists():
            try:
                meta_data = _json.loads(meta_path.read_text())
                lp = meta_data.get("language_pair", ["en", "ja"])
                from meeting_scribe.languages import is_valid_languages
                if isinstance(lp, list) and is_valid_languages(lp):
                    language_pair = list(lp)
            except Exception:
                pass

        click.secho(f"Full reprocess: {meeting_id}", fg="cyan", bold=True)
        click.echo(
            f"  Audio: {duration_s:.0f}s ({pcm.stat().st_size / 1024 / 1024:.0f}MB)"
        )
        click.echo(f"  ASR: {asr_url}")
        click.echo(f"  Translation: {translate_url}")
        click.echo(f"  Languages: {'/'.join(language_pair)}")

        def on_progress(step: int, total: int, msg: str) -> None:
            click.echo(f"  [{step}/{total}] {msg}")

        try:
            result = asyncio.run(
                reprocess_meeting(
                    meeting_dir,
                    asr_url=asr_url,
                    translate_url=translate_url,
                    language_pair=language_pair,
                    on_progress=on_progress,
                    expected_speakers=expected_speakers,
                )
            )
        except Exception as e:
            click.secho(f"  Crashed: {e}", fg="red")
            failures.append((meeting_id, str(e)))
            continue

        if "error" in result:
            click.secho(f"  Error: {result['error']}", fg="red")
            failures.append((meeting_id, result["error"]))
            continue

        click.secho(
            f"  Done: {result['segments']} segments, {result['translated']} translated, "
            f"{result.get('speakers', 0)} speakers, "
            f"summary={'yes' if result.get('has_summary') else 'no'}",
            fg="green",
        )
        successes += 1
        click.echo()

    if len(dirs) > 1:
        click.echo()
        click.secho("─" * 72, fg="cyan")
        click.secho(
            f"Full reprocess complete: {successes} ok, {len(failures)} failed",
            fg="green" if not failures else "yellow",
            bold=True,
        )
        for mid, reason in failures:
            click.echo(f"  {mid}  {reason}")


# ---------------------------------------------------------------------------
# library — read-only meeting library inspection (ls, stats, verify)
#
# Mutations live in `cleanup` / `finalize` / `reprocess` / `full-reprocess`;
# this group is strictly inspection so it's safe to pipe into other tools
# without side effects.
# ---------------------------------------------------------------------------


@cli.group()
def library() -> None:
    """Read-only meeting library inspection (ls, stats, verify)."""


def _library_audit() -> list[dict]:
    from meeting_scribe.config import ServerConfig
    from meeting_scribe.storage import MeetingStorage

    cfg = ServerConfig.from_env()
    return MeetingStorage(cfg).audit_meetings()


def _apply_library_filters(
    rows: list[dict],
    state: str | None,
    min_events: int | None,
    max_events: int | None,
    older_than: str | None,
    newer_than: str | None,
) -> list[dict]:
    older_h = _parse_duration_spec(older_than)
    newer_h = _parse_duration_spec(newer_than)
    out = []
    for m in rows:
        if state and m["state"] != state:
            continue
        if min_events is not None and m["journal_lines"] < min_events:
            continue
        if max_events is not None and m["journal_lines"] >= max_events:
            continue
        if older_h is not None and m["age_hours"] < older_h:
            continue
        if newer_h is not None and m["age_hours"] > newer_h:
            continue
        out.append(m)
    return out


@library.command("ls")
@click.option(
    "--state",
    type=click.Choice(["recording", "interrupted", "complete", "stopped", "finalizing"]),
    default=None,
    help="Filter by state",
)
@click.option("--min-events", type=int, default=None, help="Only meetings with ≥ N events")
@click.option("--max-events", type=int, default=None, help="Only meetings with < N events")
@click.option("--older-than", default=None, help="Only meetings older than SPEC (e.g. 7d, 48h)")
@click.option("--newer-than", default=None, help="Only meetings newer than SPEC")
@click.option(
    "--ids-only",
    is_flag=True,
    help="Emit only meeting IDs, one per line (for piping into xargs).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON")
def library_ls(
    state: str | None,
    min_events: int | None,
    max_events: int | None,
    older_than: str | None,
    newer_than: str | None,
    ids_only: bool,
    as_json: bool,
) -> None:
    """List meetings with filters — pipeable into reprocess/finalize/etc.

    Examples:
      meeting-scribe library ls --state interrupted --min-events 200
      meeting-scribe library ls --older-than 30d --max-events 50 --ids-only \
        | xargs -I{} meeting-scribe cleanup --exclude {}   # dry inspection
    """
    import json as _json

    rows = _apply_library_filters(
        _library_audit(), state, min_events, max_events, older_than, newer_than
    )
    if ids_only:
        for m in rows:
            click.echo(m["meeting_id"])
        return
    if as_json:
        serial = [
            {k: (str(v) if hasattr(v, "__fspath__") else v) for k, v in m.items()}
            for m in rows
        ]
        click.echo(_json.dumps(serial, indent=2))
        return
    click.echo(
        f"{'ID':<38} {'STATE':<12} {'EVENTS':>7} {'AUDIO':>8} {'AGE':>8}  SUMMARY"
    )
    for m in rows:
        click.echo(
            f"{m['meeting_id']:<38} {m['state']:<12} "
            f"{m['journal_lines']:>7} {m['audio_duration_s']:>7.0f}s "
            f"{m['age_hours']:>7.0f}h  {'yes' if m['has_summary'] else 'no'}"
        )
    click.echo(f"\n{len(rows)} meeting(s)")


@library.command("stats")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON")
def library_stats(as_json: bool) -> None:
    """Aggregate library stats: state counts, disk usage, event histogram."""
    import json as _json
    from collections import Counter

    rows = _library_audit()
    if not rows:
        click.echo("Library is empty.")
        return

    by_state = Counter(m["state"] for m in rows)
    events = [m["journal_lines"] for m in rows]
    audio_s = [m["audio_duration_s"] for m in rows]

    total_bytes = 0
    audio_bytes = 0
    journal_bytes = 0
    summary_bytes = 0
    other_bytes = 0
    for m in rows:
        d = Path(m["meeting_dir"])
        for f in d.rglob("*"):
            if not f.is_file():
                continue
            try:
                sz = f.stat().st_size
            except OSError:
                continue
            total_bytes += sz
            name = f.name
            if name == "summary.json":
                summary_bytes += sz
            elif name == "journal.jsonl":
                journal_bytes += sz
            elif f.suffix in (".pcm", ".wav", ".mp3", ".opus"):
                audio_bytes += sz
            else:
                other_bytes += sz

    buckets = [
        (0, 50),
        (50, 100),
        (100, 200),
        (200, 500),
        (500, 1000),
        (1000, None),
    ]
    histogram: list[dict[str, Any]] = []
    for lo, hi in buckets:
        if hi is None:
            n = sum(1 for e in events if e >= lo)
            label = f"{lo}+"
        else:
            n = sum(1 for e in events if lo <= e < hi)
            label = f"{lo}-{hi}"
        histogram.append({"range": label, "count": n})

    sorted_events = sorted(events)
    stats: dict[str, Any] = {
        "total_meetings": len(rows),
        "by_state": dict(by_state),
        "events": {
            "min": min(events),
            "max": max(events),
            "mean": round(sum(events) / len(events), 1),
            "median": sorted_events[len(events) // 2],
        },
        "audio_seconds": {
            "min": round(min(audio_s), 1),
            "max": round(max(audio_s), 1),
            "mean": round(sum(audio_s) / len(audio_s), 1),
            "total_hours": round(sum(audio_s) / 3600, 2),
        },
        "disk_bytes": {
            "total": total_bytes,
            "audio": audio_bytes,
            "journal": journal_bytes,
            "summary": summary_bytes,
            "other": other_bytes,
        },
        "event_histogram": histogram,
    }

    if as_json:
        click.echo(_json.dumps(stats, indent=2))
        return

    def _mb(b: int) -> str:
        return f"{b / 1024 / 1024:7.1f} MB"

    click.secho(f"Meeting Library Stats — {len(rows)} meetings", bold=True)
    click.echo()
    click.echo("By state:")
    for s, n in sorted(by_state.items(), key=lambda x: -x[1]):
        click.echo(f"  {s:<14} {n}")
    click.echo()
    click.echo(
        f"Events:  min={stats['events']['min']}  "
        f"median={stats['events']['median']}  "
        f"mean={stats['events']['mean']}  "
        f"max={stats['events']['max']}"
    )
    click.echo(
        f"Audio:   total={stats['audio_seconds']['total_hours']}h  "
        f"mean={stats['audio_seconds']['mean']}s  "
        f"max={stats['audio_seconds']['max']}s"
    )
    click.echo()
    click.echo("Disk:")
    click.echo(f"  total    {_mb(total_bytes)}")
    click.echo(f"  audio    {_mb(audio_bytes)}")
    click.echo(f"  journal  {_mb(journal_bytes)}")
    click.echo(f"  summary  {_mb(summary_bytes)}")
    click.echo(f"  other    {_mb(other_bytes)}")
    click.echo()
    click.echo("Event-count histogram:")
    max_count = max((h["count"] for h in histogram), default=0)
    for h in histogram:
        bar_len = int(40 * h["count"] / max_count) if max_count else 0
        bar = "█" * bar_len
        click.echo(f"  {h['range']:>12}  {h['count']:>4}  {bar}")


@library.command("verify")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON")
def library_verify(as_json: bool) -> None:
    """Integrity check — parseable meta, valid JSONL, audio sanity, state consistency.

    Read-only. Reports findings; does not mutate. Run `meeting-scribe cleanup`
    or `meeting-scribe finalize` to act on the findings.
    """
    import json as _json

    from meeting_scribe.config import ServerConfig

    cfg = ServerConfig.from_env()
    rows = _library_audit()
    audited = {Path(m["meeting_dir"]).name for m in rows}
    issues: list[dict] = []

    meetings_root = Path(cfg.meetings_dir)
    if meetings_root.exists():
        for d in sorted(meetings_root.iterdir()):
            if not d.is_dir() or d.name == "__pycache__":
                continue
            if d.name not in audited:
                issues.append(
                    {
                        "meeting_id": d.name,
                        "severity": "error",
                        "issue": "missing or unparseable meta.json",
                    }
                )

    for m in rows:
        mid = m["meeting_id"]
        d = Path(m["meeting_dir"])

        jp = d / "journal.jsonl"
        if jp.exists():
            bad = 0
            with jp.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        _json.loads(line)
                    except _json.JSONDecodeError:
                        bad += 1
            if bad:
                issues.append(
                    {
                        "meeting_id": mid,
                        "severity": "error",
                        "issue": f"{bad} malformed journal line(s)",
                    }
                )

        if m["journal_lines"] >= 50 and m["audio_duration_s"] < 1:
            issues.append(
                {
                    "meeting_id": mid,
                    "severity": "warning",
                    "issue": f"{m['journal_lines']} events but 0s audio",
                }
            )

        if m["state"] in ("interrupted", "stopped") and m["has_summary"] and m["has_timeline"]:
            issues.append(
                {
                    "meeting_id": mid,
                    "severity": "warning",
                    "issue": (
                        f"state={m['state']} but already has summary+timeline "
                        "(run `meeting-scribe finalize` to transition to complete)"
                    ),
                }
            )

        if m["state"] == "stopped":
            issues.append(
                {
                    "meeting_id": mid,
                    "severity": "warning",
                    "issue": "legacy 'stopped' state (run `meeting-scribe finalize`)",
                }
            )

    if as_json:
        click.echo(
            _json.dumps(
                {
                    "checked": len(rows),
                    "issue_count": len(issues),
                    "issues": issues,
                },
                indent=2,
            )
        )
        return

    click.secho(
        f"Verified {len(rows)} meetings — {len(issues)} issue(s)", bold=True
    )
    if not issues:
        click.secho("  all clean", fg="green")
        return
    for iss in issues:
        color = "red" if iss["severity"] == "error" else "yellow"
        click.secho(
            f"  [{iss['severity']:<7}] {iss['meeting_id']}  {iss['issue']}",
            fg=color,
        )


@cli.group()
def versions() -> None:
    """Reprocess version snapshots — list + diff past runs against current.

    Every reprocess auto-snapshots the prior journal/summary/timeline/
    speakers into ``meetings/{id}/versions/{ts}__pre-reprocess/``. Use
    these commands to compare runs and judge whether a code/model change
    actually improved transcription quality.
    """


@versions.command("list")
@click.option("-m", "--meeting-id", required=True, help="Meeting id to list versions for")
def versions_list(meeting_id: str) -> None:
    """List snapshots for a meeting (newest first)."""
    from meeting_scribe.versions import list_versions

    meetings_dir = PROJECT_ROOT / "meetings"
    mdir = meetings_dir / meeting_id
    if not mdir.is_dir():
        click.secho(f"Meeting not found: {meeting_id}", fg="red")
        return
    rows = list_versions(mdir)
    if not rows:
        click.echo("(no snapshots — this meeting hasn't been reprocessed yet)")
        return
    click.secho(f"{len(rows)} snapshot(s) for {meeting_id}:", fg="cyan", bold=True)
    for r in rows:
        m = r["manifest"]
        label = m.get("label") or "(no label)"
        ts = m.get("snapshot_at_utc", "")
        commit = (m.get("git_commit") or "")[:8] or "-"
        inputs = m.get("inputs") or {}
        lp = ",".join(inputs.get("language_pair", []) or [])
        es = inputs.get("expected_speakers")
        click.echo(
            f"  {r['name']}"
            f"\n    label={label}  ts={ts}  git={commit}"
            f"  language_pair={lp or '-'}  expected_speakers={es if es else '-'}"
        )


@versions.command("diff")
@click.option("-m", "--meeting-id", required=True, help="Meeting id")
@click.option(
    "--baseline",
    default=None,
    help="Snapshot dir name to use as baseline (default: most recent snapshot)",
)
@click.option(
    "--compare",
    default=None,
    help="Snapshot dir name to compare to baseline (default: current state)",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON")
def versions_diff(
    meeting_id: str,
    baseline: str | None,
    compare: str | None,
    as_json: bool,
) -> None:
    """Compare two versions of a meeting (or the latest snapshot vs current).

    By default: ``--baseline`` is the most recent snapshot and ``--compare``
    is the current top-level state. So after running reprocess, this
    command tells you "did the new run improve things vs the prior run".
    """
    import json as _json

    from meeting_scribe.versions import (
        diff_versions,
        list_versions,
        metrics_for_current,
        metrics_for_version,
    )

    meetings_dir = PROJECT_ROOT / "meetings"
    mdir = meetings_dir / meeting_id
    if not mdir.is_dir():
        click.secho(f"Meeting not found: {meeting_id}", fg="red")
        return

    snaps = list_versions(mdir)
    if not snaps and baseline is None:
        click.secho("(no snapshots yet — run reprocess once to create one)", fg="yellow")
        return

    if baseline is None:
        baseline = snaps[0]["name"]
    base_metrics = metrics_for_version(mdir, baseline)
    base_label = baseline

    if compare is None:
        cmp_metrics = metrics_for_current(mdir)
        cmp_label = "(current)"
    else:
        cmp_metrics = metrics_for_version(mdir, compare)
        cmp_label = compare

    diff = diff_versions(base_metrics, cmp_metrics)

    if as_json:
        click.echo(_json.dumps(
            {"baseline": base_label, "compare": cmp_label, "diff": diff},
            ensure_ascii=False, indent=2,
        ))
        return

    click.secho(f"Diff: {base_label}  →  {cmp_label}", fg="cyan", bold=True)
    click.echo()
    color_for = {"better": "green", "worse": "red", "same": "white"}
    sign_for = {"better": "▲", "worse": "▼", "same": "·"}
    for key, info in diff["dimensions"].items():
        verdict = info["verdict"]
        delta_pct = info["delta_rel"] * 100.0
        click.secho(
            f"  {sign_for[verdict]} {key:<40} {info['baseline']!s:>10}  →  {info['compare']!s:<10}"
            f"  ({delta_pct:+.1f}%)  [{verdict}]",
            fg=color_for[verdict],
        )
    t = diff["totals"]
    click.echo()
    click.secho(
        f"Totals: {t['better']} better · {t['worse']} worse · {t['same']} same",
        bold=True,
    )
    if diff.get("language_distribution"):
        click.echo()
        click.secho("Language tag distribution:", fg="cyan")
        click.echo(f"  baseline: {diff['language_distribution']['baseline']}")
        click.echo(f"  compare:  {diff['language_distribution']['compare']}")


@cli.group()
def gb10() -> None:
    """GB10 infrastructure management — setup, start, stop model containers."""


@gb10.command()
@click.option(
    "--host",
    envvar="SCRIBE_GB10_HOST",
    default="localhost",
    help="GB10 IP address (default: localhost)",
)
@click.option("--turboquant", is_flag=True, help="Build TurboQuant image (bjk110 branch)")
def setup(host: str, turboquant: bool) -> None:
    """Provision a GB10 node: build vLLM image and pull models."""
    from meeting_scribe.infra.containers import build_vllm_image, pull_models
    from meeting_scribe.infra.runner import get_runner
    from meeting_scribe.recipes import all_model_ids

    ssh = get_runner(host)

    if not ssh.is_reachable():
        click.secho(f"Cannot reach GB10 at {host}", fg="red")
        sys.exit(1)

    click.secho(f"Connected to GB10 at {host}", fg="green")

    # Build spark-vllm-docker image
    click.echo("Building spark-vllm-docker image...")
    tag = build_vllm_image(ssh, turboquant=turboquant)
    click.secho(f"Image built: {tag}", fg="green")

    # Pull all required models
    model_ids = all_model_ids()
    click.echo(f"Pulling {len(model_ids)} models...")
    pull_models(ssh, model_ids)
    click.secho("Models downloaded", fg="green")


@gb10.command("up")
@click.option(
    "--host",
    envvar="SCRIBE_GB10_HOST",
    default="localhost",
    help="GB10 IP address (default: localhost)",
)
@click.option(
    "--offline",
    is_flag=True,
    help="Never pull images — for offline/boot use. Systemd passes this flag.",
)
def gb10_up(host: str, offline: bool) -> None:
    """Launch the model container stack via docker compose.

    docker-compose.gb10.yml is the single source of truth for what
    runs. This CLI is a thin wrapper — we let compose decide ports,
    volumes, container names, restart policies, and label-based
    autoheal. The recipes/ directory stays useful for pull-models
    (it carries the HF model IDs and URL mapping) but is NO LONGER
    used to decide what to launch.

    The 35B translate model is NOT listed in compose — it's owned by
    autosre and scribe reads from it as a tenant on port 8010. That's
    the whole point of keeping the two systems separate.

    Health waiting is deliberately NOT done here. Compose returns as
    soon as containers are CREATED (not when models are loaded). The
    systemd unit's Phase 2 preflight (``preflight --mode=boot --wait
    720``) does the concurrent health polling with a proper shared
    deadline. Duplicating that wait here was causing 5+ minute boot
    delays because the old sequential poll burned its entire budget
    on the translation backend before even checking TTS/ASR.
    """
    from meeting_scribe.infra.compose import compose_up

    click.echo("Starting scribe container stack (docker compose up -d)...")
    try:
        compose_up(pull="never" if offline else None)
    except Exception as e:
        click.secho(f"compose up failed: {e}", fg="red")
        sys.exit(1)
    click.secho("compose up: done", fg="green")


@gb10.command("down")
@click.option(
    "--host",
    envvar="SCRIBE_GB10_HOST",
    default="localhost",
    help="GB10 IP address (default: localhost)",
)
def gb10_down(host: str) -> None:
    """Stop the scribe container stack via docker compose."""
    from meeting_scribe.infra.compose import compose_down

    click.echo("Stopping scribe container stack (docker compose down)...")
    try:
        compose_down()
    except Exception as e:
        click.secho(f"compose down failed: {e}", fg="red")
        sys.exit(1)
    click.secho("compose down: done", fg="green")


@gb10.command("restart-container")
@click.argument("service")
@click.option("--timeout", default=30, help="Seconds to wait for graceful stop before kill")
@click.option(
    "--recreate",
    is_flag=True,
    help=(
        "Force-recreate the container so it picks up a changed compose env / "
        "image / volume / ports spec. Default `restart` keeps the OLD env."
    ),
)
def gb10_restart_container(service: str, timeout: int, recreate: bool) -> None:
    """Restart a single compose service.

    Default mode (``restart``) is for **CUDA recovery** — a container whose
    process is alive but whose CUDA context is wedged (``/health`` returns 200
    but inference calls return 500 with ``torch.AcceleratorError: CUDA error:
    unknown error``). Fast (~5 s) and preserves the running env / image /
    volumes.

    Pass ``--recreate`` when you've **edited the compose file** (env vars,
    image tag, volume mounts, ports). A plain restart silently keeps the OLD
    env, so a config edit appears to do nothing. Recreate does
    ``docker compose up -d --force-recreate --no-deps`` so the new spec
    actually lands. Slower (~15-30 s for small images, much longer for vLLM).

    SERVICE is the compose service name (e.g. ``pyannote-diarize``,
    ``vllm-asr``, ``qwen3-tts``, ``vllm-tts``). Run
    ``docker compose -f docker-compose.gb10.yml config --services``
    to see the full list.
    """
    from meeting_scribe.infra.compose import compose_restart

    mode = "recreate" if recreate else "restart"
    click.echo(f"{mode.capitalize()}-ing compose service {service}...")
    try:
        compose_restart(service, timeout=timeout, recreate=recreate)
    except Exception as e:
        click.secho(f"compose {mode} failed: {e}", fg="red")
        raise SystemExit(1) from e
    click.secho(f"{service} {mode}d", fg="green")


@gb10.command("status")
@click.option(
    "--host",
    envvar="SCRIBE_GB10_HOST",
    default="localhost",
    help="GB10 IP address (default: localhost)",
)
def gb10_status(host: str) -> None:
    """Check health of all services on the GB10."""
    import asyncio

    from meeting_scribe.infra.containers import list_containers
    from meeting_scribe.infra.health import check_all_services
    from meeting_scribe.infra.runner import get_runner, is_local

    ssh = get_runner(host)

    # List running containers
    containers = list_containers(ssh)
    click.echo(f"Containers: {len(containers)} running")
    for c in containers:
        click.echo(f"  {c['name']}: {c['status']}")

    # Health check services (hit localhost when running on the GB10 itself)
    click.echo("\nService health:")
    health_host = "localhost" if is_local(host) else host
    results = asyncio.run(check_all_services(health_host))
    for name, svc_status in results.items():
        color = "green" if svc_status.healthy else "red"
        model_info = f" ({svc_status.model})" if svc_status.model else ""
        click.secho(
            f"  {name}: {'healthy' if svc_status.healthy else 'unhealthy'}{model_info}", fg=color
        )


@gb10.command("pull-models")
@click.option(
    "--host",
    envvar="SCRIBE_GB10_HOST",
    default="localhost",
    help="GB10 IP address (default: localhost)",
)
def pull_models_cmd(host: str) -> None:
    """Download all required models to the GB10's HuggingFace cache."""
    from meeting_scribe.infra.containers import pull_models as _pull
    from meeting_scribe.infra.runner import get_runner
    from meeting_scribe.recipes import all_model_ids

    ssh = get_runner(host)

    model_ids = all_model_ids()
    click.echo(f"Pulling {len(model_ids)} models to {host}:/data/huggingface...")
    for mid in model_ids:
        click.echo(f"  {mid}")
    _pull(ssh, model_ids)
    click.secho("All models downloaded", fg="green")


@gb10.command("start")
@click.option("--port", "-p", default=DEFAULT_PORT, help="Server port")
@click.option("--debug", is_flag=True, help="Enable debug logging")
def gb10_start(port: int, debug: bool) -> None:
    """Start meeting-scribe server with GB10 profile.

    Binds to 0.0.0.0 so the server is accessible from the WiFi hotspot.
    """
    os.environ["SCRIBE_PROFILE"] = "gb10"
    os.environ.setdefault("SCRIBE_HOST", "0.0.0.0")
    click.echo("Starting with GB10 profile (Qwen3-ASR, vLLM, pyannote)...")
    click.echo(f"  Bind: 0.0.0.0:{port} (accessible from hotspot)")

    from click.testing import CliRunner

    runner = CliRunner()
    args = [f"--port={port}", "--foreground"]
    if debug:
        args.append("--debug")
    runner.invoke(start, args)


@cli.command("reprocess-summaries")
@click.option("--dry-run", is_flag=True, help="List meetings that would be processed without calling the LLM")
@click.option("--resume", is_flag=True, help="Skip meetings already processed (from checkpoint state)")
@click.option("--vllm-url", default="http://localhost:8010", help="vLLM endpoint URL")
def reprocess_summaries(dry_run: bool, resume: bool, vllm_url: str) -> None:
    """Regenerate all meeting summaries with the v2 schema.

    Processes each completed meeting sequentially. Backs up existing summaries,
    validates v2 output before replacing, and checkpoints progress for resumability.
    """
    import asyncio
    import json as _json
    import shutil

    from meeting_scribe.config import ServerConfig
    from meeting_scribe.summary import generate_summary

    config = ServerConfig.from_env()
    meetings_dir = config.meetings_dir
    state_path = meetings_dir / "reprocess-state.json"

    if not meetings_dir.exists():
        click.secho("No meetings directory found", fg="red")
        raise SystemExit(1)

    # Load checkpoint state
    done_ids: set[str] = set()
    if resume and state_path.exists():
        with open(state_path) as f:
            state = _json.load(f)
            done_ids = set(state.get("completed", []))
        click.echo(f"Resuming: {len(done_ids)} meetings already processed")

    # Find all completed meetings with transcripts
    candidates = []
    for d in sorted(meetings_dir.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        journal_path = d / "journal.jsonl"
        if not meta_path.exists() or not journal_path.exists():
            continue
        try:
            meta = _json.loads(meta_path.read_text())
        except _json.JSONDecodeError:
            continue
        if meta.get("state") != "complete":
            continue
        if resume and d.name in done_ids:
            continue
        candidates.append(d)

    if not candidates:
        click.secho("No meetings to process", fg="yellow")
        return

    click.echo(f"Found {len(candidates)} meeting(s) to reprocess")

    if dry_run:
        for d in candidates:
            summary_path = d / "summary.json"
            status = "has summary" if summary_path.exists() else "no summary"
            click.echo(f"  {d.name} ({status})")
        click.echo(f"\nDry run: {len(candidates)} meetings would be processed")
        return

    # Process sequentially
    succeeded = 0
    failed = 0
    failed_ids: list[str] = []

    for i, meeting_dir in enumerate(candidates, 1):
        meeting_id = meeting_dir.name
        summary_path = meeting_dir / "summary.json"
        backup_path = meeting_dir / "summary.v1.bak.json"

        click.echo(f"[{i}/{len(candidates)}] {meeting_id}: ", nl=False)

        try:
            # Backup existing summary
            if summary_path.exists() and not backup_path.exists():
                shutil.copy2(summary_path, backup_path)

            # Generate new summary (writes to summary.json internally)
            # We need to intercept and validate, so we generate to tmp first
            result = asyncio.run(generate_summary(
                meeting_dir=meeting_dir,
                vllm_url=vllm_url,
            ))

            if result.get("error"):
                click.secho(f"FAILED ({result['error']})", fg="red")
                failed += 1
                failed_ids.append(meeting_id)
                # Restore backup if summary.json was overwritten
                if backup_path.exists() and not summary_path.exists():
                    shutil.copy2(backup_path, summary_path)
                continue

            # Validate v2 fields are present
            if not result.get("key_insights") or not result.get("named_entities"):
                click.secho("FAILED (missing v2 fields)", fg="red")
                failed += 1
                failed_ids.append(meeting_id)
                # Restore backup
                if backup_path.exists():
                    shutil.copy2(backup_path, summary_path)
                continue

            click.secho("OK", fg="green")
            succeeded += 1

            # Update checkpoint
            done_ids.add(meeting_id)
            state_path.write_text(_json.dumps({
                "completed": sorted(done_ids),
                "last_updated": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
            }, indent=2) + "\n")

        except Exception as e:
            click.secho(f"FAILED ({e})", fg="red")
            failed += 1
            failed_ids.append(meeting_id)
            # Restore backup
            if backup_path.exists() and summary_path.exists():
                # generate_summary already wrote; restore backup
                shutil.copy2(backup_path, summary_path)

    click.echo()
    click.secho(f"Done: {succeeded} succeeded, {failed} failed", fg="green" if failed == 0 else "yellow")
    if failed_ids:
        click.echo("Failed meetings:")
        for mid in failed_ids:
            click.echo(f"  {mid}")


# ── WiFi hotspot management ──────────────────────────────────


@cli.group("wifi")
def wifi_group():
    """WiFi hotspot management (off / meeting / admin mode).

    \b
    The GB10 MT7925 radio supports one AP at a time. Two modes:
      meeting  — rotating SSID, captive portal, guest isolation
      admin    — fixed SSID, no portal, admin UI reachable over WiFi
    """


@wifi_group.command("up")
@click.option(
    "--mode",
    type=click.Choice(["meeting", "admin"]),
    default="meeting",
    show_default=True,
    help="meeting = rotating guest SSID + captive portal; "
    "admin = fixed SSID, no portal, admin UI over WiFi",
)
@click.option("--ssid", default=None, help="Override SSID")
@click.option("--password", default=None, help="Override password")
@click.option(
    "--band",
    default="a",
    show_default=True,
    type=click.Choice(["a", "bg"]),
    help="WiFi band (a=5GHz, bg=2.4GHz)",
)
@click.option(
    "--channel", default=36, show_default=True, type=int, help="WiFi channel"
)
def wifi_up(mode: str, ssid: str | None, password: str | None, band: str, channel: int) -> None:
    """Bring up the WiFi AP in meeting or admin mode."""
    import asyncio as _asyncio

    from meeting_scribe.wifi import build_config
    from meeting_scribe.wifi import wifi_up as _wifi_up

    cfg = build_config(mode, ssid, password, band, channel)
    click.echo(click.style(f"==> Bringing up WiFi AP ({mode} mode)...", fg="cyan"))
    _asyncio.run(_wifi_up(cfg))
    click.echo()
    click.echo(click.style(f"WiFi AP is live! ({mode} mode)", fg="green", bold=True))
    click.echo(f"  Mode:      {mode}")
    click.echo(f"  SSID:      {cfg.ssid}")
    click.echo(f"  Password:  {cfg.password}")
    click.echo("  Security:  WPA3-SAE (PMF required)")
    click.echo(f"  Band:      {'5 GHz' if band == 'a' else '2.4 GHz'} (channel {channel})")
    if mode == "admin":
        click.echo(f"  Admin UI:  https://{cfg.ap_ip}:8080/")
        click.echo("  Portal:    disabled (admin mode)")
    else:
        click.echo(f"  Guest:     http://{cfg.ap_ip}/ (phones)")
        click.echo("  Admin:     https://<mgmt-ip>:8080/ (LAN only)")
        click.echo(f"  Captive:   DNS wildcard → {cfg.ap_ip}")


@wifi_group.command("down")
def wifi_down_cmd() -> None:
    """Tear down the WiFi AP. Persists wifi_mode=off."""
    import asyncio as _asyncio

    from meeting_scribe.wifi import wifi_down as _wifi_down

    click.echo(click.style("==> Tearing down WiFi AP...", fg="cyan"))
    _asyncio.run(_wifi_down())
    click.secho("WiFi AP stopped. wifi_mode=off persisted.", fg="green")


@wifi_group.command("status")
def wifi_status_cmd() -> None:
    """Show live WiFi status (reads from nmcli/wpa_cli, not just state file)."""
    from meeting_scribe.wifi import wifi_status_sync

    info = wifi_status_sync()
    desired = info.get("desired_mode", "off")
    live = info.get("live_mode", "off")

    click.echo(click.style("WiFi Status", fg="cyan", bold=True))
    click.echo()

    dm_color = "green" if desired != "off" else "yellow"
    click.echo(f"  Desired:     {click.style(desired.upper(), fg=dm_color)}")

    lm_color = "green" if live != "off" else ("yellow" if live == "unknown" else "red")
    click.echo(f"  Live:        {click.style(live.upper(), fg=lm_color)}")

    if info.get("ssid"):
        click.echo(f"  SSID:        {info['ssid']}")
    if info.get("security"):
        sec = info["security"]
        km = sec.get("key_mgmt", "?")
        label = click.style("WPA3-SAE", fg="green") if km == "SAE" else click.style(km, fg="yellow")
        click.echo(f"  Security:    {label} pairwise={sec.get('pairwise_cipher', '?')}")
    if info.get("client_count") is not None:
        click.echo(f"  Clients:     {info['client_count']}")

    rd_configured = info.get("regdomain")
    rd_live = info.get("regdomain_live")
    if rd_live and rd_configured:
        if rd_live == rd_configured:
            click.echo(f"  Regdomain:   {click.style(rd_live, fg='green')}")
        else:
            click.echo(
                f"  Regdomain:   {click.style('DRIFT', fg='red')} "
                f"(configured={rd_configured} live={rd_live})"
            )

    cap = info.get("captive_active")
    if cap is True:
        click.echo(f"  Captive:     {click.style('active', fg='green')}")
    elif cap is False:
        click.echo(f"  Captive:     {click.style('disabled', fg='yellow')}")


@cli.group("config")
def config_group() -> None:
    """Hot-reloadable runtime-config knobs (translate_url, slide_translate_url, ...).

    \b
    These three settings can be flipped without restarting the server:
      translate_url          — live-translation vLLM endpoint (Phase 7 rollback)
      slide_translate_url    — slide-pipeline vLLM endpoint
      slide_use_json_schema  — Phase 4b response_format flag

    `config set` persists to $XDG_DATA_HOME/meeting-scribe/runtime-config.json
    and sends SIGHUP to the running server so it re-reads on the next
    translation request.  Every other setting still lives on the static
    ServerConfig dataclass loaded from env.
    """


@config_group.command("get")
@click.argument("key", required=False)
def config_get(key: str | None) -> None:
    """Show one knob or all of them."""
    from meeting_scribe import runtime_config

    snapshot = runtime_config.instance().as_dict()
    if key is None:
        if not snapshot:
            click.echo("(no runtime overrides; all knobs fall back to ServerConfig)")
            return
        for k, v in snapshot.items():
            click.echo(f"{k} = {v!r}")
        return
    if key not in snapshot:
        click.echo(f"{key}: (unset)")
        return
    click.echo(f"{key} = {snapshot[key]!r}")


@config_group.command("set")
@click.argument("key")
@click.argument("value")
@click.option(
    "--no-reload",
    is_flag=True,
    help="Persist to disk but do NOT send SIGHUP; the server will pick up on next restart.",
)
def config_set(key: str, value: str, no_reload: bool) -> None:
    """Persist KEY=VALUE and (by default) SIGHUP the running server."""
    from meeting_scribe import runtime_config

    # Coerce booleans for the known boolean knobs so CLI users don't
    # have to quote "true"/"false" and then worry about string vs bool.
    coerced: object = value
    if key == "slide_use_json_schema":
        if value.lower() in {"true", "1", "yes", "on"}:
            coerced = True
        elif value.lower() in {"false", "0", "no", "off"}:
            coerced = False
        else:
            raise click.BadParameter(
                f"slide_use_json_schema expects a boolean (true/false), got {value!r}"
            )

    try:
        runtime_config.instance().set(key, coerced)
    except KeyError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"{key} = {coerced!r} (persisted to {runtime_config.instance().path})")

    if no_reload:
        return
    _sighup_running_server()


@config_group.command("unset")
@click.argument("key")
@click.option("--no-reload", is_flag=True, help="Don't SIGHUP.")
def config_unset(key: str, no_reload: bool) -> None:
    """Clear a knob so the next read falls back to ServerConfig."""
    from meeting_scribe import runtime_config

    try:
        runtime_config.instance().unset(key)
    except KeyError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"{key} cleared")
    if not no_reload:
        _sighup_running_server()


@config_group.command("reload")
def config_reload() -> None:
    """SIGHUP the running server to re-read runtime-config.json."""
    _sighup_running_server()


def _sighup_running_server() -> None:
    """Best-effort SIGHUP to the running meeting-scribe server."""
    _state, pid, origin = _server_state()
    if pid is None:
        click.echo(
            click.style("(no server running; changes will apply on next start)", fg="yellow")
        )
        return
    try:
        os.kill(pid, signal.SIGHUP)
        click.echo(
            click.style(f"SIGHUP sent to PID {pid} ({origin})", fg="green")
        )
    except ProcessLookupError:
        click.echo(click.style(f"(PID {pid} not found; stale state?)", fg="yellow"))
    except PermissionError as err:
        raise click.ClickException(
            f"Cannot signal PID {pid}; systemd-managed services may require "
            f"`sudo systemctl reload {_SYSTEMD_UNIT}` instead."
        ) from err


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


def _post_admin(endpoint: str) -> dict:
    """POST /api/admin/<endpoint> on the running server; return JSON body.

    Shared by ``pause-translation`` and ``resume-translation`` so both
    commands handle a not-running server identically.
    """
    import json as _json
    import urllib.error

    _state, pid, _origin = _server_state()
    if pid is None:
        click.echo(click.style("No server running.", fg="yellow"))
        raise click.exceptions.Exit(0)

    host, admin_port = "127.0.0.1", DEFAULT_PORT
    url = f"http://{host}:{admin_port}/api/admin/{endpoint}"
    req = urllib.request.Request(url, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return _json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise click.ClickException(f"{endpoint} endpoint error {e.code}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise click.ClickException(f"{endpoint} endpoint unreachable: {e}") from e


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
        click.echo(click.style("translation intake still paused (server rejected resume)", fg="red"))


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
