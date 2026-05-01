"""``meeting-scribe install-service`` / ``uninstall-service``.

Writes a user-level systemd unit for the scribe server so it starts at
boot. Why user-level instead of system-wide:

* No sudo on every ``start`` / ``stop`` / ``restart``. The customer
  owns their checkout and their venv; matching ownership at the
  service layer means ``systemctl --user`` works without elevation.
* ``loginctl enable-linger <user>`` makes user services start at boot
  before login — which is what a fresh customer GB10 needs after a
  power-cycle.

The unit uses ``Type=notify`` because the FastAPI lifespan calls
``sd_notify(READY=1)`` once all backends are reachable; that's the
right signal for ``RequiredBy`` consumers later (e.g. an external
reverse proxy waiting on the scribe API).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import click

from meeting_scribe.cli import cli
from meeting_scribe.cli._common import _SYSTEMD_UNIT, PROJECT_ROOT


def _user_unit_dir() -> Path:
    """Resolve ``$XDG_CONFIG_HOME/systemd/user`` (or the ``~/.config`` default)."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "systemd" / "user"


def _venv_bin() -> Path:
    """The venv's ``bin/`` directory containing ``meeting-scribe``."""
    return PROJECT_ROOT / ".venv" / "bin"


_OOM_DROP_IN_NAME = "oom-priority.conf"


def _render_oom_drop_in() -> str:
    """Drop-in that ships the OOM-protection settings as part of
    `meeting-scribe install-service`. Lives at
    ``service.d/oom-priority.conf`` so the operator-tunable bits stay
    decoupled from the rendered base unit. Without this drop-in,
    user-mode systemd inherits OOMScoreAdjust=+100 from
    user@<uid>.service and another +100 from app.slice, leaving the
    transcription server as the kernel's preferred OOM victim — the
    2026-04-30 17:35 incident root cause. Paired with the +500 the
    sddc-cli QEMU smoke-test sets on its own subprocess
    (sddc.iso_qemu.run_qemu) so a runaway smoke-test guest is the
    obvious target instead.
    """
    return """[Service]
# Operator intent: protect the live transcription server from being
# the OOM-killer's first choice. The kernel clamps -100 to +100 for
# unprivileged user services (user@<uid>.service has no
# CAP_SYS_RESOURCE, so a child cannot descend below the user
# manager's baseline), but the declaration is honoured the day
# CAP_SYS_RESOURCE is granted. The actual differential protection
# comes from the matching +500 set on the QEMU smoke-test child in
# sddc-cli/src/sddc/iso_qemu.py:run_qemu.
OOMScoreAdjust=-100
# Soft 2 GiB memory reservation for scribe — under host pressure the
# kernel reclaim machinery skips anonymous pages below this floor
# until every other unprotected cgroup has been touched. cgroup
# memory.low works for user services without elevated capabilities,
# unlike OOMScoreAdjust.
MemoryLow=2G
"""


def _render_unit() -> str:
    venv_bin = _venv_bin()
    return f"""[Unit]
Description=Meeting Scribe — real-time multilingual meeting transcription
Documentation=https://github.com/sddcinfo/meeting-scribe
# docker.service is a system-level unit and not directly visible to a
# user manager; the After= still drains the right event for us thanks
# to the user manager's late-boot order.
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=notify
NotifyAccess=main
WorkingDirectory={PROJECT_ROOT}
Environment=HOME={Path.home()}
Environment=PATH={venv_bin}:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# ``--foreground`` skips the daemonize fork — systemd needs the
# top-level PID to receive READY=1 over NotifyAccess=main.
ExecStart={venv_bin}/meeting-scribe start --foreground
Restart=on-failure
RestartSec=10
# Cold-start budget includes the pyannote/vLLM-ASR/TTS containers
# coming up healthy; allow a generous window before systemd marks
# the unit failed.
TimeoutStartSec=300
TimeoutStopSec=60

[Install]
WantedBy=default.target
"""


def _systemctl_user(*args: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["systemctl", "--user", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _enable_linger(quiet: bool) -> None:
    """``loginctl enable-linger`` so the user manager runs at boot.

    Without lingering, ``--user`` services only start when the user has
    an active login session. A fresh GB10 with no logged-in console
    session won't autostart anything — exactly the boot scenario this
    command exists to handle. ``loginctl`` requires root to set the
    flag for another user; for the invoking user (which is what we
    want) most distros allow it without sudo via polkit, but we still
    fall back to sudo if the unprivileged path fails.
    """
    user = os.environ.get("USER") or Path.home().name
    proc = subprocess.run(
        ["loginctl", "enable-linger", user],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        if not quiet:
            click.secho(f"[OK] linger enabled for user '{user}'", fg="green")
        return
    # Polkit-rejected — escalate. Non-fatal: services still install,
    # they just won't autostart on a cold boot until the user
    # manually enables linger.
    sudo_proc = subprocess.run(
        ["sudo", "-n", "loginctl", "enable-linger", user],
        capture_output=True,
        text=True,
        check=False,
    )
    if sudo_proc.returncode == 0:
        if not quiet:
            click.secho(f"[OK] linger enabled (via sudo) for user '{user}'", fg="green")
        return
    click.secho(
        f"[WARN] could not enable linger for '{user}'. The service will install "
        "but won't autostart at boot until you run:",
        fg="yellow",
    )
    click.echo(f"  sudo loginctl enable-linger {user}")


@cli.command("install-service")
@click.option(
    "--no-enable",
    is_flag=True,
    help="Write the unit file but do not enable/start it.",
)
@click.option(
    "--no-start",
    is_flag=True,
    help="Enable for boot but skip starting now.",
)
@click.option("-q", "--quiet", is_flag=True, help="Reduce output to errors only.")
def install_service(no_enable: bool, no_start: bool, quiet: bool) -> None:
    """Install the user systemd unit so meeting-scribe starts on boot.

    Writes ``~/.config/systemd/user/meeting-scribe.service`` (or
    ``$XDG_CONFIG_HOME/systemd/user/...``), runs ``daemon-reload``,
    enables the unit for boot, and optionally starts it now. Also
    runs ``loginctl enable-linger`` so user services run before
    login — required for a customer GB10 that boots without a
    console session.
    """
    if shutil.which("systemctl") is None:
        click.secho("systemctl not found — this OS doesn't use systemd.", fg="red")
        sys.exit(1)

    # Sanity: the venv must actually contain the meeting-scribe entry
    # point, otherwise the ExecStart= line points at nothing and the
    # unit will fail in a confusing way at boot.
    venv_entry = _venv_bin() / "meeting-scribe"
    if not venv_entry.exists():
        click.secho(
            f"meeting-scribe entry point missing: {venv_entry}\n"
            "Run ./bootstrap.sh (or `pip install -e .` inside .venv) first.",
            fg="red",
        )
        sys.exit(1)

    unit_dir = _user_unit_dir()
    unit_path = unit_dir / _SYSTEMD_UNIT
    unit_dir.mkdir(parents=True, exist_ok=True, mode=0o755)

    body = _render_unit()
    existing = unit_path.read_text() if unit_path.exists() else None
    if existing != body:
        # Atomic write so a concurrent ``daemon-reload`` doesn't see
        # half the file.
        tmp = unit_path.with_suffix(unit_path.suffix + f".tmp.{os.getpid()}")
        tmp.write_text(body)
        os.replace(tmp, unit_path)
        if not quiet:
            click.secho(f"[OK] wrote {unit_path}", fg="green")
    elif not quiet:
        click.secho(f"[OK] {unit_path} already up-to-date", fg="green")

    # OOM-priority drop-in. Kept out of the base unit body so an
    # operator can hand-tune MemoryLow/OOMScoreAdjust without churning
    # the rendered unit hash on every install-service run.
    drop_in_dir = unit_dir / f"{_SYSTEMD_UNIT}.d"
    drop_in_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
    drop_in_path = drop_in_dir / _OOM_DROP_IN_NAME
    drop_in_body = _render_oom_drop_in()
    drop_in_existing = drop_in_path.read_text() if drop_in_path.exists() else None
    if drop_in_existing != drop_in_body:
        tmp = drop_in_path.with_suffix(drop_in_path.suffix + f".tmp.{os.getpid()}")
        tmp.write_text(drop_in_body)
        os.replace(tmp, drop_in_path)
        if not quiet:
            click.secho(f"[OK] wrote {drop_in_path}", fg="green")
    elif not quiet:
        click.secho(f"[OK] {drop_in_path} already up-to-date", fg="green")

    rc, _, err = _systemctl_user("daemon-reload")
    if rc != 0:
        click.secho(f"systemctl --user daemon-reload failed: {err}", fg="red")
        sys.exit(rc or 1)
    if not quiet:
        click.secho("[OK] systemd user manager reloaded", fg="green")

    if no_enable:
        click.secho(
            "Skipping enable/start (--no-enable). To activate later:\n"
            f"  systemctl --user enable --now {_SYSTEMD_UNIT}",
            fg="cyan",
        )
        return

    _enable_linger(quiet)

    enable_args = ["enable"] if no_start else ["enable", "--now"]
    rc, _, err = _systemctl_user(*enable_args, _SYSTEMD_UNIT)
    if rc != 0:
        click.secho(
            f"systemctl --user {' '.join(enable_args)} {_SYSTEMD_UNIT} failed: {err}",
            fg="red",
        )
        sys.exit(rc or 1)

    if no_start:
        click.secho(
            f"[OK] {_SYSTEMD_UNIT} enabled for boot (not started yet — run "
            f"`systemctl --user start {_SYSTEMD_UNIT}` when ready)",
            fg="green",
        )
    else:
        click.secho(
            f"[OK] {_SYSTEMD_UNIT} enabled + started — will auto-start on boot",
            fg="green",
        )
        click.echo(
            "  Verify:  systemctl --user status meeting-scribe.service\n"
            "  Logs:    journalctl --user -u meeting-scribe.service -f"
        )


@cli.command("uninstall-service")
@click.option("-q", "--quiet", is_flag=True, help="Reduce output to errors only.")
def uninstall_service(quiet: bool) -> None:
    """Stop, disable, and remove the user systemd unit.

    Does not undo ``loginctl enable-linger`` — that flag is generic
    and may be wanted by other user services.
    """
    if shutil.which("systemctl") is None:
        click.secho("systemctl not found — nothing to uninstall.", fg="yellow")
        return

    # Stop + disable, but tolerate "unit not loaded" since the user
    # may be running this on a system where install never happened.
    for action in ("stop", "disable"):
        rc, _, err = _systemctl_user(action, _SYSTEMD_UNIT)
        if rc != 0 and "not loaded" not in err.lower() and "no such file" not in err.lower():
            click.secho(
                f"[WARN] systemctl --user {action} {_SYSTEMD_UNIT}: {err}",
                fg="yellow",
            )

    unit_path = _user_unit_dir() / _SYSTEMD_UNIT
    if unit_path.exists():
        unit_path.unlink()
        if not quiet:
            click.secho(f"[OK] removed {unit_path}", fg="green")

    rc, _, err = _systemctl_user("daemon-reload")
    if rc != 0 and not quiet:
        click.secho(f"daemon-reload after removal: {err}", fg="yellow")

    if not quiet:
        click.secho(f"[OK] {_SYSTEMD_UNIT} uninstalled", fg="green")
