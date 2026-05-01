"""Shared helpers + constants for the meeting-scribe CLI.

Every topic module under ``meeting_scribe.cli`` imports from here for
PID handling, systemd state, TLS-cert self-heal, management-IP
detection, admin-API requests, and docker-compose helpers. Keeping
these in one module avoids duplicating the 200+ lines of bookkeeping
across every command file.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import click

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_TMPDIR = Path(tempfile.gettempdir())
LOG_FILE = _TMPDIR / "meeting-scribe.log"
PID_FILE = _TMPDIR / "meeting-scribe.pid"
DEFAULT_PORT = 8080
GUEST_PORT = 80
# Hotspot AP IP — mirrors server.AP_IP, duplicated here so the CLI can
# print the guest URL without importing the heavy server module.
AP_IP = "10.42.0.1"

_SYSTEMD_UNIT = "meeting-scribe.service"

_COMPOSE_FILE = PROJECT_ROOT / "docker-compose.gb10.yml"
_REQUIRED_CONTAINERS = ["scribe-diarization", "scribe-tts", "scribe-asr"]

# sysexits — EX_CONFIG. Used by the required-imports self-check below.
_EX_CONFIG = 78


def _assert_required_imports() -> None:
    """Pre-bind dependency probe (plan §1.6b).

    Reads `tool.meeting-scribe.required-imports` from `pyproject.toml`
    and `importlib.import_module()`s each one. Exits 78 (EX_CONFIG)
    on the first ImportError. Called by `cli/lifecycle.py:start()`
    BEFORE any subprocess spawn or `os.execvpe`, so the failure
    happens before any socket is created — never reaches the
    uvicorn-binds-then-lifespan-runs window.

    Tolerates a missing pyproject.toml or a missing list (returns
    silently): a stripped runtime layout shouldn't crash the server,
    only a missing actual import should. The lockfile is the
    authoritative install gate; this is a startup tripwire that
    catches the case where the lockfile was wired up but
    `pip install -e .` skipped a dep.
    """
    import importlib
    import sys
    import tomllib

    pyproject = PROJECT_ROOT / "pyproject.toml"
    if not pyproject.is_file():
        return
    try:
        cfg = tomllib.loads(pyproject.read_text())
    except OSError, tomllib.TOMLDecodeError:
        return
    required = cfg.get("tool", {}).get("meeting-scribe", {}).get("required-imports", [])
    if not isinstance(required, list):
        return
    missing: list[str] = []
    for name in required:
        if not isinstance(name, str):
            continue
        try:
            importlib.import_module(name)
        except ImportError as e:
            missing.append(f"{name} ({e})")
    if missing:
        click.secho(
            "MissingDependencyError: pyproject.toml lists imports that are "
            "not installed in the active venv:\n  - "
            + "\n  - ".join(missing)
            + "\n\nThis usually means `pip install -e .` skipped a dependency "
            "(e.g. an `editable + lockfile` race during bootstrap). Re-run:\n"
            "    pip install -r requirements.lock && pip install --no-deps -e .\n"
            "or re-run `bootstrap.sh` if the venv was hand-edited.",
            fg="red",
            err=True,
        )
        sys.exit(_EX_CONFIG)


def _read_pid(pid_file: Path) -> int | None:
    """Return the PID from `pid_file` if the process is alive, else None."""
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return pid
    except ValueError, ProcessLookupError, PermissionError:
        pid_file.unlink(missing_ok=True)
        return None


def _get_pid() -> int | None:
    """Read the server PID from the PID file (legacy path, foreground/dev)."""
    return _read_pid(PID_FILE)


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
                "systemctl",
                "--user",
                "show",
                _SYSTEMD_UNIT,
                "-p",
                "MainPID",
                "-p",
                "ActiveState",
                "-p",
                "LoadState",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except FileNotFoundError, subprocess.SubprocessError:
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
        except ProcessLookupError, PermissionError:
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


def _ensure_admin_tls_certs() -> tuple[bool, str]:
    """Generate a self-signed cert pair under ``certs/`` if missing.

    Idempotent: returns immediately if both files already exist.
    Used by both ``meeting-scribe setup`` and the ``start`` path so
    that a wiped ``certs/`` directory (e.g. a release workflow that
    ran ``git clean -fdx`` on the live tree) self-heals rather than
    crashing the service. The cert is a CN=meeting-scribe self-
    signed RSA-2048 with a 365-day validity — same parameters
    ``setup`` has always used.

    Returns (ok, detail):
        ok=True  — certs present (either pre-existing or just generated)
        ok=False — generation failed (openssl missing or error); caller
                   surfaces the detail message to the user.
    """
    certs_dir = PROJECT_ROOT / "certs"
    cert_pem = certs_dir / "cert.pem"
    key_pem = certs_dir / "key.pem"
    if cert_pem.exists() and key_pem.exists():
        return True, "TLS certs exist in certs/"

    if not shutil.which("openssl"):
        return (
            False,
            "openssl not on PATH — install via `apt install openssl` and re-run.",
        )

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
        check=False,
    )
    if result.returncode == 0:
        return True, f"self-signed certs generated at {certs_dir}"
    return (
        False,
        f"openssl cert generation failed (rc={result.returncode}): {result.stderr.strip()[:200]}",
    )


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
    except subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired:
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
    except subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired:
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
        click.secho(
            f"Container launch failed (exit {e.returncode}) — server will retry backends",
            fg="yellow",
        )
    except subprocess.TimeoutExpired:
        click.secho("Container launch timed out — containers may still be starting", fg="yellow")


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
        click.echo(click.style(f"SIGHUP sent to PID {pid} ({origin})", fg="green"))
    except ProcessLookupError:
        click.echo(click.style(f"(PID {pid} not found; stale state?)", fg="yellow"))
    except PermissionError as err:
        raise click.ClickException(
            f"Cannot signal PID {pid}; systemd-managed services may require "
            f"`sudo systemctl reload {_SYSTEMD_UNIT}` instead."
        ) from err


def _parse_duration_spec(spec: str | None) -> float | None:
    """Parse a duration like ``30m``, ``48h``, ``7d``, ``2w`` into hours.

    Returns None for empty/None input. Raises click.BadParameter on
    malformed input so the CLI layer surfaces a clean error instead of
    a stack trace. Kept here (rather than a util module) because
    everything that calls it is CLI-layer.
    """
    if not spec:
        return None
    import re

    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([mhdw])", spec.strip().lower())
    if not m:
        raise click.BadParameter(f"invalid duration {spec!r}; use e.g. '30m', '48h', '7d', '2w'")
    value = float(m.group(1))
    unit = m.group(2)
    per_hour = {"m": 1 / 60, "h": 1.0, "d": 24.0, "w": 24.0 * 7}[unit]
    return value * per_hour


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
