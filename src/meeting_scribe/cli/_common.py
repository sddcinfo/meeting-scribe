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
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

import click

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_TMPDIR = Path(tempfile.gettempdir())
LOG_FILE = _TMPDIR / "meeting-scribe.log"
PID_FILE = _TMPDIR / "meeting-scribe.pid"
DEFAULT_PORT = int(os.environ.get("SCRIBE_PORT") or "443")
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


# Subject Alternative Names enforced on every meeting-scribe leaf cert. The
# appliance is reachable only on the hotspot AP — the IP SAN covers direct
# ``https://10.42.0.1`` access AND the per-device mDNS name covers
# ``https://meeting-scribe-${id4}.local`` access (Phase C publishes that
# name only on the AP iface). The leaf-only trust anchor (no CA) means
# broader SANs would expand the spoofable surface beyond the v1.0 boundary.
# ``runtime/cert_check.py`` enforces the same set on every server start.
_REQUIRED_LEAF_IP_SANS: frozenset[str] = frozenset({AP_IP})


def _appliance_id_short() -> str:
    """First 4 hex of the appliance ID — used as the unique-per-device
    mDNS suffix so a multi-appliance LAN can't collide. Stable across
    reboots; only changes if the operator wipes ``certs/appliance-id``.
    """
    return _read_or_mint_appliance_id()[:4]


def appliance_pin() -> str:
    """Return the 4-digit decimal PIN derived from the appliance ID.

    Stable per-device, public (printed in the SSID), used as BOTH
    the SSID's last 4 chars (so attendees see ``Dell Meeting 1234``)
    AND the guest-auth PIN. One number for both means the operator
    only needs to remember / display one thing.

    The PIN is NOT a security boundary on its own — it's derivable
    from the SSID anyone in Wi-Fi range can see. Application-layer
    auth (admin password) covers sensitive operations; the PIN is
    a UX convenience that gates casual guest access the same way a
    posted meeting-room Wi-Fi password does.
    """
    import hashlib

    digest = hashlib.sha256(_read_or_mint_appliance_id().encode("utf-8")).digest()
    n = int.from_bytes(digest[:4], "big") % 10000
    return f"{n:04d}"


def _required_leaf_dns_sans() -> frozenset[str]:
    """Per-device DNS SANs — re-evaluated each call because the
    appliance-id is minted lazily and may not exist when the module is
    imported (fresh checkout, before ``meeting-scribe setup``).

    Two names per device:
      * ``meeting-scribe-<id4>.local`` — engineering / fleet-unique form
      * ``meeting-<pin>.local`` — operator-friendly short form that
        matches the 4-digit pin shown in the SSID and admin UI, e.g.
        ``meeting-1618.local``. Both are published over mDNS by the
        lifespan task (see ``server_support.mdns``).
    """
    return frozenset(
        {
            f"meeting-scribe-{_appliance_id_short()}.local",
            f"meeting-{appliance_pin()}.local",
        }
    )


def _appliance_id_path() -> Path:
    """Path that holds the per-device appliance ID.

    Production deployments persist this at ``/etc/meeting-scribe/appliance-id``
    (root-writable, service-readable). Dev / single-tree workflows use
    ``certs/appliance-id`` next to the cert pair so a fresh checkout self-heals
    without root. The prod path wins when present so a single test box can
    transition from dev-checkout-style to prod-installed-style without losing
    its ID.
    """
    prod = Path("/etc/meeting-scribe/appliance-id")
    if prod.exists():
        return prod
    return PROJECT_ROOT / "certs" / "appliance-id"


def _read_or_mint_appliance_id() -> str:
    """Return the appliance ID, generating one on first call.

    The ID is a 16-char hex token (``secrets.token_hex(8)``) — hostname-safe,
    long enough that two random GB10s cannot collide in a fleet of millions.
    Generated once, then frozen on disk; subsequent setup runs reuse it so
    rotated leaf certs preserve the same Subject CN suffix.
    """
    import secrets

    path = _appliance_id_path()
    if path.exists():
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    new_id = secrets.token_hex(8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_id + "\n", encoding="utf-8")
    return new_id


def _ensure_admin_tls_certs() -> tuple[bool, str]:
    """Generate a per-device self-signed leaf under ``certs/`` if missing.

    Idempotent: returns immediately if both files already exist AND already
    carry the required SANs. Stale certs (no SAN, or SAN missing the AP IP)
    are regenerated in place — that path triggers when an older
    pre-cutover-format leaf is still on disk; the new run produces the v1.0
    leaf-only-with-SAN format.

    The generated leaf carries:
      * Subject ``CN=meeting-scribe/<appliance_id>`` — stable per device,
        looked up via :func:`_read_or_mint_appliance_id`.
      * Subject Alternative Name ``IP:10.42.0.1`` only — the appliance is
        reachable nowhere else.
      * RSA-2048, 365-day validity, no encryption on the key.

    Used by both ``meeting-scribe setup`` and the ``start`` path so a wiped
    ``certs/`` directory (e.g. a release workflow that ran ``git clean -fdx``)
    self-heals rather than crashing the service.

    Returns (ok, detail):
        ok=True  — certs present (either pre-existing AND valid, or just
                   generated); detail names the appliance ID.
        ok=False — generation failed (openssl missing or error); caller
                   surfaces the detail message to the user.
    """
    from meeting_scribe.runtime.cert_check import (
        CertConfigError,
        assert_cert_sans,
    )

    certs_dir = PROJECT_ROOT / "certs"
    cert_pem = certs_dir / "cert.pem"
    key_pem = certs_dir / "key.pem"

    if cert_pem.exists() and key_pem.exists():
        try:
            assert_cert_sans(
                cert_pem,
                required_ips=set(_REQUIRED_LEAF_IP_SANS),
                required_dns=set(_required_leaf_dns_sans()),
            )
            return True, f"TLS certs exist in {certs_dir} (SANs validated)"
        except CertConfigError:
            # Stale cert with old SANs (pre-cutover or pre-DNS-SAN
            # rev) — regenerate. The new run produces the v1.0 leaf
            # with both IP:10.42.0.1 AND DNS:meeting-scribe-${id4}.local.
            pass

    if not shutil.which("openssl"):
        return (
            False,
            "openssl not on PATH — install via `apt install openssl` and re-run.",
        )

    certs_dir.mkdir(exist_ok=True)
    appliance_id = _read_or_mint_appliance_id()
    san_pieces = [f"IP:{ip}" for ip in sorted(_REQUIRED_LEAF_IP_SANS)]
    san_pieces += [f"DNS:{name}" for name in sorted(_required_leaf_dns_sans())]
    san_arg = ",".join(san_pieces)

    # Atomic install via temp paths + rename — avoids a half-written cert
    # surviving an interrupted setup run.
    tmp_cert = cert_pem.with_suffix(".pem.tmp")
    tmp_key = key_pem.with_suffix(".pem.tmp")
    result = subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(tmp_key),
            "-out",
            str(tmp_cert),
            "-days",
            "365",
            "-nodes",
            "-subj",
            # OpenSSL's -subj parser splits RDN on `/`, so the literal slash
            # in the appliance-id-bearing CN must be backslash-escaped.
            rf"/CN=meeting-scribe\/{appliance_id}",
            "-addext",
            f"subjectAltName = {san_arg}",
            "-addext",
            "keyUsage = digitalSignature, keyEncipherment",
            "-addext",
            "extendedKeyUsage = serverAuth",
            "-addext",
            "basicConstraints = critical, CA:FALSE",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        for stale in (tmp_cert, tmp_key):
            stale.unlink(missing_ok=True)
        return (
            False,
            f"openssl cert generation failed (rc={result.returncode}): "
            f"{result.stderr.strip()[:200]}",
        )

    # Move both files into place; if either rename fails we leave the prior
    # cert untouched so the service keeps serving.
    os.replace(tmp_cert, cert_pem)
    os.replace(tmp_key, key_pem)
    try:
        os.chmod(key_pem, 0o600)
    except OSError:
        pass
    return True, (
        f"self-signed leaf generated at {certs_dir} (appliance_id={appliance_id}, SAN={san_arg})"
    )


def _server_url(port: int = DEFAULT_PORT) -> str:
    return f"https://{AP_IP}:{port}"


# Auth-redirect suffixes the admin middleware uses when an
# unauthenticated request hits an admin-gated route. ``/auth`` is the
# legacy entrypoint; ``/signin`` and ``/setup`` were added with the
# multi-step setup wizard. ``meeting-scribe status`` used to interpret
# every benign 302 as "server not responding" because only ``/auth``
# was matched — the 2026-05-14 demo debug session burned an hour on this.
_BENIGN_AUTH_REDIRECT_SUFFIXES: tuple[str, ...] = ("/auth", "/signin", "/setup")


class UnauthenticatedRedirect(Exception):
    """Raised by :func:`_api_request` when the server is healthy but the
    request landed on a sign-in page after the auto-authorize retry.

    Callers that should treat this as "running, awaiting auth" (e.g.
    ``meeting-scribe status``) catch it explicitly. Callers that don't
    catch it get a non-zero CLI exit via Click's default handler with
    a clear message — preferable to the previous silent ``None``
    behavior which made the CLI report empty success for an
    unauthenticated request.

    The ``suffix`` attribute carries the redirect target (``/setup``,
    ``/signin``, ``/auth``) so callers can render useful diagnostics.
    """

    def __init__(self, suffix: str) -> None:
        super().__init__(f"server redirected to {suffix} (not authenticated)")
        self.suffix = suffix


def _api_request(path: str, method: str = "GET", port: int = DEFAULT_PORT) -> dict | None:
    """Make an API request to the canonical AP-IP listener.

    The HTTPS listener binds to ``10.42.0.1`` only (not 0.0.0.0), so
    we hit that address directly. ``127.0.0.1`` no longer routes to
    it. If the admin middleware redirects to a benign auth/signin/setup
    page, authenticate with the deterministic local admin password and
    retry once so CLI diagnostics stay useful on hardened deployments.

    Return values:
      * ``dict`` — successful JSON response.
      * ``None`` — true "no response" (timeout, refused, non-JSON,
        unexpected redirect target).
      * raises :class:`UnauthenticatedRedirect` — the server responded
        but the authorize retry could not pass the auth gate
        (deterministic password mismatch, hardened deployment, etc.).
    """
    import ssl

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    url = f"https://{AP_IP}:{port}{path}"

    jar = CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ctx),
        urllib.request.HTTPCookieProcessor(jar),
    )

    # Sentinel returned by ``_read_json`` to distinguish "benign auth
    # redirect" (try authorize then re-read) from "no response at all"
    # (None — return as-is). We can't raise the typed exception from
    # the inner reader directly because the outer dispatcher needs a
    # chance to authorize first.
    _AUTH_REDIRECT = object()

    def _read_json() -> dict | object | None:
        try:
            req = urllib.request.Request(url, method=method)
            with opener.open(req, timeout=5) as resp:
                final_url = resp.geturl()
                if any(final_url.endswith(s) for s in _BENIGN_AUTH_REDIRECT_SUFFIXES):
                    return _AUTH_REDIRECT
                content_type = resp.headers.get("content-type", "")
                if "json" not in content_type:
                    return None
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            # 401/403 - probably hotspot_guard rejecting a POST that
            # arrived without an admin cookie. Returning the redirect
            # sentinel signals the outer dispatcher to authorize +
            # retry; without this branch the HTTPError was caught
            # silently by the outer ``except Exception`` and the
            # authorize-retry path never ran. The same fix applies to
            # every POST/PUT admin endpoint, surfaced most visibly on
            # ``/api/admin/kiosk/mint-nonce`` (2026-05-14).
            if exc.code in {401, 403}:
                return _AUTH_REDIRECT
            raise

    def _authorize() -> bool:
        try:
            from meeting_scribe.setup_state import _mint_admin_password

            password = os.environ.get("SCRIBE_ADMIN_PASSWORD") or _mint_admin_password()
            body = urllib.parse.urlencode({"password": password}).encode()
            req = urllib.request.Request(
                f"https://{AP_IP}:{port}/api/admin/authorize",
                data=body,
                method="POST",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            with opener.open(req, timeout=5) as resp:
                final_url = resp.geturl()
                return resp.status in {200, 303} or not any(
                    final_url.endswith(s) for s in _BENIGN_AUTH_REDIRECT_SUFFIXES
                )
        except Exception:
            return False

    def _redirect_suffix_from(url_str: str) -> str:
        for suffix in _BENIGN_AUTH_REDIRECT_SUFFIXES:
            if url_str.endswith(suffix):
                return suffix
        return "/auth"  # fall-through default; shouldn't be reachable

    try:
        data = _read_json()
    except Exception:
        return None
    if isinstance(data, dict):
        return data
    if data is None:
        return None
    # data is the auth-redirect sentinel — try to authorize and re-read.
    if _authorize():
        try:
            second = _read_json()
        except Exception:
            return None
        if isinstance(second, dict):
            return second
        if second is None:
            return None
        # Still hitting the auth page after a successful-looking authorize —
        # raise so the caller knows the server is up but unauthenticated.
        raise UnauthenticatedRedirect(_redirect_suffix_from(url))
    # Authorize itself failed; surface the underlying redirect.
    raise UnauthenticatedRedirect(_redirect_suffix_from(url))


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
