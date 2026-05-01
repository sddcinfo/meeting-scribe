"""start / stop / restart / status / preflight / shutdown / logs / health / diagnose.

Server lifecycle commands. Every command in this module is a top-level
``meeting-scribe <name>`` command (no sub-group), so each is decorated
directly on the root ``cli`` group.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import time

import click

from meeting_scribe.cli import cli
from meeting_scribe.cli._common import (
    _SYSTEMD_UNIT,
    AP_IP,
    DEFAULT_PORT,
    GUEST_PORT,
    LOG_FILE,
    PID_FILE,
    PROJECT_ROOT,
    _api_request,
    _assert_required_imports,
    _ensure_admin_tls_certs,
    _ensure_containers_running,
    _ensure_port80_bind,
    _get_pid,
    _management_ip,
    _server_state,
    _server_url,
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

    # Pre-bind dependency self-check (plan §1.6b). Reads
    # tool.meeting-scribe.required-imports from pyproject.toml and
    # importlib.import_module()s each one. Exits 78 (EX_CONFIG) on
    # ImportError BEFORE any socket creation, so the unit never
    # reaches "Type=notify activating" with a stale/broken process
    # listening on a half-bound port. Catches the 2026-05-01 PPTX
    # upload regression where python-multipart was missing from the
    # customer venv.
    _assert_required_imports()

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

    # Self-heal admin TLS certs: if certs/ has been wiped (e.g. a
    # gitignored-dir cleanup that missed the fact that certs/ is
    # load-bearing for the admin HTTPS listener), regenerate them
    # silently rather than crashing. server.main() would raise
    # RuntimeError on missing certs; surfacing it here lets us
    # recover without manual `meeting-scribe setup`.
    ok, detail = _ensure_admin_tls_certs()
    if not ok:
        click.secho(f"Admin TLS certs unavailable: {detail}", fg="red")
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
            capture_output=True,
            text=True,
            check=False,
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
@click.pass_context
def restart(ctx: click.Context, port: int, debug: bool) -> None:
    """Restart the server + run a smoke test so silent partial failures don't slip through.

    Delegates to ``systemctl --user restart`` when the unit is systemd-
    managed so preflight / notify semantics are preserved. Falls back to
    the in-process click stop+start dance for foreground/dev runs.
    """
    mode, _pid, _active = _server_state()
    if mode == "systemd":
        click.echo(f"Restarting {_SYSTEMD_UNIT} via systemctl --user...")
        r = subprocess.run(
            ["systemctl", "--user", "restart", _SYSTEMD_UNIT],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            click.secho(f"systemctl --user restart failed: {r.stderr.strip()}", fg="red")
            sys.exit(r.returncode or 1)
    else:
        # ctx.invoke runs the target command in the live click.Context
        # so its click.echo output reaches the terminal. The previous
        # CliRunner().invoke(...) was Click's *test* harness — it
        # captured stdout into a Result object, so a failed restart
        # exited with no visible diagnostics.
        ctx.invoke(stop)
        time.sleep(1)
        ctx.invoke(start, port=port, debug=debug, foreground=False)

    # Smoke test — surfaces broken startup (empty-reply-from-server,
    # wrong-port binds, crashed background loops). Runs best-effort:
    # a green start + green smoke is the contract; a red smoke still
    # leaves the server running so the user can inspect it.
    smoke = PROJECT_ROOT / "scripts" / "smoke-test.sh"
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
    "--reboot",
    "action",
    flag_value="reboot",
    help="Reboot the machine after the service stops (only on green preflight).",
)
@click.option(
    "--poweroff",
    "action",
    flag_value="poweroff",
    help="Power off the machine after the service stops (only on green preflight).",
)
@click.option(
    "--force",
    is_flag=True,
    help="Proceed with stop/reboot/poweroff even if preflight fails. BOOT_BLOCKED is still written.",
)
@click.option(
    "--wait",
    default=300.0,
    type=float,
    help="Total wall-clock budget in seconds for live preflight (default 300).",
)
@click.option(
    "--reason",
    default="",
    help="Audit reason recorded in last-good-shutdown.",
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
            capture_output=True,
            text=True,
            check=False,
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
    # translate-endpoint move (e.g. swapping ports or model variants)
    # stops reporting spurious "not responding".  We still hit each
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

    translate_url = (runtime_config.get("translate_url") or cfg.translate_vllm_url).rstrip("/")
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
