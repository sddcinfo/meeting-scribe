"""``meeting-scribe speakerphone`` — USB-speakerphone integration CLI.

Subcommands:

* ``install``   — write udev rule + render user systemd unit (matches
  the ``install-service`` pattern for the main service).
* ``uninstall`` — symmetric teardown.
* ``listen``    — long-running daemon process. Foreground when invoked
  by hand; systemd runs it as the unit ExecStart.
* ``status``    — show connected devices + current mapping + last
  button press, reading the same internal state endpoint the daemon
  uses (so GUI ↔ device consistency is provable from the shell).
* ``test``      — interactive smoke. Prompts the operator to press
  each button and verifies the kernel event arrives.
* ``capture-descriptor`` — dump + annotate the HID report descriptor
  of a connected device (dev aid for identifying vendor-page Teams
  buttons).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import click

from meeting_scribe.cli import cli

logger = logging.getLogger(__name__)


# ── CLI group ───────────────────────────────────────────────────────────


@cli.group("speakerphone")
def speakerphone_group() -> None:
    """USB speakerphone (e.g. Dell SP325) — listener + mapping ops."""


# ── install / uninstall ─────────────────────────────────────────────────

_UDEV_RULE_BODY = """\
# meeting-scribe speakerphone — Dell HID-telephony devices
# Grants plugdev rw on the SP325/SP3022 event* and hidraw* nodes so the
# unprivileged user-mode daemon can read button events + write the LED
# state. TAG+="uaccess" opts the nodes into logind's per-seat ACL so an
# active console session can also access them.
#
# Also grants USB-device access (/dev/bus/usb/*) for libusb-based vendor
# HID config (sp325_hid.Sp325HidClient) — needed to send the wideband-
# enable command sequence at device attach. SUBSYSTEM=="usb",
# ATTR{...}, MODE/GROUP/TAG must be on the matching parent device,
# distinct from the SUBSYSTEMS=="usb", ATTRS{...} child-node form
# above for the event*/hidraw* nodes.

SUBSYSTEMS=="usb", ATTRS{idVendor}=="413c", ATTRS{idProduct}=="8223", \\
  KERNEL=="event*", GROUP="plugdev", MODE="0660", TAG+="uaccess"
SUBSYSTEMS=="usb", ATTRS{idVendor}=="413c", ATTRS{idProduct}=="8223", \\
  KERNEL=="hidraw*", GROUP="plugdev", MODE="0660", TAG+="uaccess"
SUBSYSTEM=="usb", ATTR{idVendor}=="413c", ATTR{idProduct}=="8223", \\
  MODE="0660", GROUP="plugdev", TAG+="uaccess"

SUBSYSTEMS=="usb", ATTRS{idVendor}=="413c", ATTRS{idProduct}=="8222", \\
  KERNEL=="event*", GROUP="plugdev", MODE="0660", TAG+="uaccess"
SUBSYSTEMS=="usb", ATTRS{idVendor}=="413c", ATTRS{idProduct}=="8222", \\
  KERNEL=="hidraw*", GROUP="plugdev", MODE="0660", TAG+="uaccess"
SUBSYSTEM=="usb", ATTR{idVendor}=="413c", ATTR{idProduct}=="8222", \\
  MODE="0660", GROUP="plugdev", TAG+="uaccess"

# SP3022 (newer revision, 8205).
SUBSYSTEMS=="usb", ATTRS{idVendor}=="413c", ATTRS{idProduct}=="8205", \\
  KERNEL=="event*", GROUP="plugdev", MODE="0660", TAG+="uaccess"
SUBSYSTEMS=="usb", ATTRS{idVendor}=="413c", ATTRS{idProduct}=="8205", \\
  KERNEL=="hidraw*", GROUP="plugdev", MODE="0660", TAG+="uaccess"
SUBSYSTEM=="usb", ATTR{idVendor}=="413c", ATTR{idProduct}=="8205", \\
  MODE="0660", GROUP="plugdev", TAG+="uaccess"
"""

_UDEV_RULE_PATH = Path("/etc/udev/rules.d/99-sddc-speakerphone.rules")
_SYSTEMD_UNIT_NAME = "meeting-scribe-speakerphone.service"


# ── WirePlumber pin rule ──────────────────────────────────────────────
# Without this rule WirePlumber's profile heuristic regularly elects
# the SP325 to the "Off" profile (no sink, no source), which makes the
# admin SPA's audio surface silently regress to a Plantronics / HDMI
# fallback whenever the device is unplugged or wireplumber restarts.
# Pinning the card to Pro Audio + bumping node priority makes the
# device deterministic across plug events.
#
# WP 0.4 user-level Lua snippets land in ``~/.config/wireplumber/main.lua.d/``;
# WP 0.5 prefers ``~/.config/wireplumber/wireplumber.conf.d/*.conf``.
# We target 0.4 because that's what Ubuntu 24.04 (and the GB10 image)
# ships; a 0.5-compatible variant can be added later if needed.
_WP_RULE_NAME = "51-sp325-pin.lua"
_WP_RULE_BODY = """\
-- meeting-scribe: pin the Dell SP325 Speakerphone to its Pro Audio
-- profile and elect it as the default audio sink + source.
--
-- WHY THIS FILE EXISTS
--   Out of the box, WirePlumber's profile heuristic picks "Off" for the
--   SP325 surprisingly often, leaving the device with no PipeWire sink
--   at all (only "Dummy Output" remains). Meeting-scribe's
--   ``audio_room_tts_sink_node`` setting then no longer resolves and
--   the admin SPA "Default sink" picker flips to whatever leftover
--   device (Plantronics Poly, HDMI, ...) WP elects. That's the
--   2026-05-13 "settings don't stay configured to the SP325" symptom.
--
-- WHAT THIS FILE DOES
--   1. Forces ``device.profile = pro-audio`` on the SP325 alsa-card.
--      Pro Audio exposes BOTH the input (pro-input-0) and output
--      (pro-output-0) sides of the speakerphone, which is what
--      meeting-scribe's mic-capture (``pw-record``) and room-TTS
--      (``pw-cat``) both expect.
--   2. Boosts ``priority.session`` on the SP325 sink + source so
--      WirePlumber's default-node election picks the SP325 over the
--      built-in HDMI sink or the Plantronics Poly.
--
-- INSTALLED BY
--   ``meeting-scribe speakerphone install`` (writes this file).
--   ``meeting-scribe speakerphone uninstall`` (removes it).
--
-- VERIFY
--   ``wpctl status`` — under "Sinks:" the row marked ``*`` should be
--   the SP325 Pro sink; under "Sources:" likewise.

table.insert(alsa_monitor.rules, {
  matches = {
    {
      -- 413c:8223 is the USB vid:pid for the Dell SP325 family. The
      -- ALSA card-name slug embeds the model string verbatim. SP3022
      -- (8205) and the legacy 8222 share the same product family
      -- branding and use the same card-name prefix.
      { "device.name", "matches", "alsa_card.usb-Dell_Inc._Dell_SP325_Speakerphone*" },
    },
  },
  apply_properties = {
    -- Force the Pro Audio profile on attach. WirePlumber's default
    -- heuristic occasionally picks "Off" which leaves the speakerphone
    -- without any sink/source nodes at all.
    ["device.profile"] = "pro-audio",
    -- Disable WP's auto-profile chooser for this card so the explicit
    -- profile above is not overwritten on subsequent attach events.
    ["api.acp.auto-profile"] = false,
  },
})

-- Bump priority on the SP325 nodes so the default-node election picks
-- them over the built-in HDMI sink and any Bluetooth/Plantronics
-- devices. WirePlumber elects the highest-priority node as default.
-- 2000 sits comfortably above the typical 1000-1500 range for analog
-- onboard devices and the 1009 we observed for the SP325's own
-- analog-stereo profile.
table.insert(alsa_monitor.rules, {
  matches = {
    {
      { "node.name", "matches", "alsa_output.usb-Dell_Inc._Dell_SP325_Speakerphone*" },
    },
    {
      { "node.name", "matches", "alsa_input.usb-Dell_Inc._Dell_SP325_Speakerphone*" },
    },
  },
  apply_properties = {
    ["priority.session"] = 2000,
    ["priority.driver"] = 2000,
  },
})
"""


def _wp_rule_path() -> Path:
    """User-level WP Lua snippet path; respects XDG_CONFIG_HOME."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "wireplumber" / "main.lua.d" / _WP_RULE_NAME


def _install_wireplumber_rule(quiet: bool) -> bool:
    """Drop the WP pin rule + restart wireplumber so it takes effect.

    Idempotent: a no-op if the file is already on disk with identical
    body. Best-effort restart: if wireplumber isn't being managed under
    this user's systemd we just skip the restart and leave the file in
    place for the next session.
    """
    path = _wp_rule_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    if path.exists() and path.read_text() == _WP_RULE_BODY:
        if not quiet:
            click.secho(f"[OK] {path} already up-to-date", fg="green")
        return True

    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(_WP_RULE_BODY)
    os.replace(tmp, path)
    if not quiet:
        click.secho(f"[OK] wrote {path}", fg="green")

    # Restart wireplumber so the new rule is loaded. The user-level
    # service is the typical case on Ubuntu desktops; if the install
    # doesn't run under a user session the restart will fail silently
    # and the rule still takes effect on the next pipewire/wireplumber
    # start.
    rc = subprocess.run(
        ["systemctl", "--user", "restart", "wireplumber"],
        capture_output=True,
        text=True,
        check=False,
    ).returncode
    if rc != 0 and not quiet:
        click.secho(
            "[WARN] could not restart wireplumber automatically; rule "
            "will load on next session start",
            fg="yellow",
        )
    return True


def _uninstall_wireplumber_rule(quiet: bool) -> None:
    path = _wp_rule_path()
    if path.exists():
        path.unlink()
        if not quiet:
            click.secho(f"[OK] removed {path}", fg="green")
        subprocess.run(
            ["systemctl", "--user", "restart", "wireplumber"],
            capture_output=True,
            text=True,
            check=False,
        )


def _user_unit_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "systemd" / "user"


def _venv_python() -> Path:
    from meeting_scribe.cli._common import PROJECT_ROOT

    return PROJECT_ROOT / ".venv" / "bin" / "python3"


def _render_systemd_unit() -> str:
    py = _venv_python()
    return f"""[Unit]
Description=meeting-scribe USB speakerphone listener
Documentation=https://github.com/sddcinfo/meeting-scribe
After=meeting-scribe.service pipewire.service
Wants=meeting-scribe.service

[Service]
Type=simple
ExecStart={py} -m meeting_scribe speakerphone listen
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""


def _install_udev_rule(quiet: bool) -> bool:
    """Write the udev rule via ``sudo tee`` + reload udev.

    Idempotent: a no-op if the rule already exists with identical body.
    Returns True if the write succeeded; False on failure (caller
    decides whether to abort or continue).
    """
    if _UDEV_RULE_PATH.exists() and _UDEV_RULE_PATH.read_text() == _UDEV_RULE_BODY:
        if not quiet:
            click.secho(f"[OK] {_UDEV_RULE_PATH} already up-to-date", fg="green")
        return True

    proc = subprocess.run(
        ["sudo", "tee", str(_UDEV_RULE_PATH)],
        input=_UDEV_RULE_BODY,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        click.secho(
            f"[WARN] could not install udev rule: {proc.stderr.strip()}",
            fg="yellow",
        )
        click.secho(
            f"  hand-install: sudo tee {_UDEV_RULE_PATH} <<EOF\n{_UDEV_RULE_BODY}EOF",
            fg="yellow",
        )
        return False

    if not quiet:
        click.secho(f"[OK] wrote {_UDEV_RULE_PATH}", fg="green")

    for action in (("control", "--reload"), ("trigger",)):
        rc = subprocess.run(
            ["sudo", "udevadm", *action],
            capture_output=True,
            text=True,
            check=False,
        ).returncode
        if rc != 0 and not quiet:
            click.secho(
                f"[WARN] udevadm {' '.join(action)} returned {rc}",
                fg="yellow",
            )
    return True


def _install_systemd_unit(quiet: bool, no_enable: bool, no_start: bool) -> int:
    unit_dir = _user_unit_dir()
    unit_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
    unit_path = unit_dir / _SYSTEMD_UNIT_NAME
    body = _render_systemd_unit()
    if unit_path.exists() and unit_path.read_text() == body:
        if not quiet:
            click.secho(f"[OK] {unit_path} already up-to-date", fg="green")
    else:
        tmp = unit_path.with_suffix(unit_path.suffix + f".tmp.{os.getpid()}")
        tmp.write_text(body)
        os.replace(tmp, unit_path)
        if not quiet:
            click.secho(f"[OK] wrote {unit_path}", fg="green")

    rc = subprocess.run(
        ["systemctl", "--user", "daemon-reload"],
        capture_output=True,
        text=True,
        check=False,
    ).returncode
    if rc != 0:
        click.secho("systemctl --user daemon-reload failed", fg="red")
        return rc or 1

    if no_enable:
        click.secho(
            f"Skipping enable/start (--no-enable). To activate later:\n"
            f"  systemctl --user enable --now {_SYSTEMD_UNIT_NAME}",
            fg="cyan",
        )
        return 0

    enable_args = ["enable"] if no_start else ["enable", "--now"]
    rc = subprocess.run(
        ["systemctl", "--user", *enable_args, _SYSTEMD_UNIT_NAME],
        capture_output=True,
        text=True,
        check=False,
    ).returncode
    if rc != 0:
        click.secho(
            f"systemctl --user {' '.join(enable_args)} {_SYSTEMD_UNIT_NAME} failed",
            fg="red",
        )
        return rc or 1

    click.secho(
        f"[OK] {_SYSTEMD_UNIT_NAME} {'enabled' if no_start else 'enabled + started'}",
        fg="green",
    )
    return 0


@speakerphone_group.command("install")
@click.option("--no-enable", is_flag=True, help="Write the unit but do not enable/start.")
@click.option("--no-start", is_flag=True, help="Enable for boot but skip starting now.")
@click.option("-q", "--quiet", is_flag=True, help="Reduce output to errors only.")
def install_cmd(no_enable: bool, no_start: bool, quiet: bool) -> None:
    """Install the udev rule + WirePlumber pin + user systemd unit."""
    if shutil.which("systemctl") is None:
        click.secho("systemctl not found — this OS doesn't use systemd.", fg="red")
        sys.exit(1)
    _install_udev_rule(quiet)
    _install_wireplumber_rule(quiet)
    rc = _install_systemd_unit(quiet, no_enable, no_start)
    sys.exit(rc)


@speakerphone_group.command("uninstall")
@click.option("-q", "--quiet", is_flag=True, help="Reduce output to errors only.")
def uninstall_cmd(quiet: bool) -> None:
    """Stop, disable, and remove the unit + udev rule."""
    if shutil.which("systemctl"):
        for action in ("stop", "disable"):
            subprocess.run(
                ["systemctl", "--user", action, _SYSTEMD_UNIT_NAME],
                capture_output=True,
                text=True,
                check=False,
            )
        unit_path = _user_unit_dir() / _SYSTEMD_UNIT_NAME
        if unit_path.exists():
            unit_path.unlink()
            if not quiet:
                click.secho(f"[OK] removed {unit_path}", fg="green")
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            capture_output=True,
            text=True,
            check=False,
        )
    if _UDEV_RULE_PATH.exists():
        rc = subprocess.run(
            ["sudo", "rm", "-f", str(_UDEV_RULE_PATH)],
            capture_output=True,
            text=True,
            check=False,
        ).returncode
        if rc == 0 and not quiet:
            click.secho(f"[OK] removed {_UDEV_RULE_PATH}", fg="green")
        subprocess.run(
            ["sudo", "udevadm", "control", "--reload"],
            capture_output=True,
            text=True,
            check=False,
        )
    _uninstall_wireplumber_rule(quiet)


# ── listen / status ─────────────────────────────────────────────────────


@speakerphone_group.command("listen")
def listen_cmd() -> None:
    """Run the long-lived speakerphone listener daemon (foreground)."""
    # Configure logging so daemon INFO lines flow to stderr → systemd
    # journal. The root meeting_scribe logger is set up by server.py but
    # this CLI runs as its own process, so we wire it explicitly.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )

    from meeting_scribe.speakerphone.daemon import SpeakerphoneDaemon

    daemon = SpeakerphoneDaemon()
    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        click.secho("speakerphone listen: stopped", fg="cyan")


@speakerphone_group.command("status")
def status_cmd() -> None:
    """Show connected devices + last press, read from the UDS endpoint."""
    from meeting_scribe.speakerphone.meeting_client import UdsMeetingClient

    async def _go() -> None:
        client = UdsMeetingClient()
        try:
            state = await client.get_state()
        finally:
            await client.aclose()
        click.echo(json.dumps(state, indent=2, sort_keys=True))

    try:
        asyncio.run(_go())
    except Exception as e:
        click.secho(f"status failed: {e!r}", fg="red")
        sys.exit(1)


# ── test / capture-descriptor / detect-rate ─────────────────────────────


@speakerphone_group.command("set-wideband")
@click.option(
    "--settle",
    "settle_seconds",
    type=float,
    default=15.0,
    show_default=True,
    help="Seconds to wait after sending commands for SP325 DSP buffer to flush.",
)
@click.option(
    "--verify/--no-verify",
    "verify",
    default=True,
    show_default=True,
    help="After settle, run compliance and report whether wideband took effect.",
)
def set_wideband_cmd(settle_seconds: float, verify: bool) -> None:
    """Apply the SP325 wideband-mic-capture vendor command sequence.

    Sends the three winning USB Set_Report commands identified by the
    2026-05-13 per-command bisect (now `meeting-scribe speakerphone benchmark`):

      • [0xD0, 0x02, …]  primary wideband-enable, median 41.82% high_band
      • [0xD1, 0x01, …]  secondary, median 31.55%
      • [0xC0, 0x04, …]  EQ-default, median 2.04% (stable, low-magnitude)

    Requires write access to ``/dev/bus/usb/<bus>/<dev>`` — install the
    udev rule via ``meeting-scribe speakerphone install`` (or run this
    command with sudo). The audio interfaces are momentarily detached
    during the apply (so the vendor commands actually take effect at
    the firmware level) and reattached before this command returns.

    Exit codes: 0 ok / verified, 2 applied but verify FAIL, 3 device-not-found.
    """
    import json as _json
    import sys as _sys

    from meeting_scribe.speakerphone.sp325_hid import Sp325Error, Sp325HidClient

    try:
        with Sp325HidClient.open_default() as cli:
            applied = cli.apply_wideband_good(settle_seconds=settle_seconds)
    except Sp325Error as e:
        click.secho(f"SP325 apply failed: {e}", fg="red")
        _sys.exit(3)

    click.echo(_json.dumps({"applied": applied}, indent=2))

    if not verify:
        return

    # Verify via compliance — 3 samples to ride out the noise.
    from meeting_scribe.speakerphone import compliance
    from meeting_scribe.speakerphone.daemon import _guess_pipewire_source_name

    target = _guess_pipewire_source_name("413c:8223")
    if not target:
        click.secho("verify skipped — no PipeWire source node", fg="yellow")
        return

    min_hbp, min_rolloff = compliance.expected_thresholds("413c:8223")
    samples: list[float] = []
    for _ in range(3):
        r = compliance.probe_device(
            target,
            capture_seconds=3.0,
            min_high_band_pct=min_hbp,
            min_rolloff_pct=min_rolloff,
        )
        samples.append(r.high_band_pct or 0.0)

    median = sorted(samples)[len(samples) // 2]
    status = "pass" if median >= min_hbp else "fail"
    click.echo(
        _json.dumps(
            {
                "verify": {
                    "status": status,
                    "samples_high_band_pct": [round(s, 2) for s in samples],
                    "median": round(median, 2),
                    "threshold": min_hbp,
                }
            },
            indent=2,
        )
    )
    if status != "pass":
        _sys.exit(2)


@speakerphone_group.command("compliance")
@click.option(
    "--node",
    "node_name",
    default=None,
    help="PipeWire source node (default: SP325 pro-input-0).",
)
@click.option(
    "--device-key",
    "device_key",
    default="413c:8223",
    show_default=True,
    help="Catalog key for threshold lookup.",
)
@click.option(
    "--seconds",
    type=float,
    default=5.0,
    show_default=True,
)
def compliance_cmd(node_name: str | None, device_key: str, seconds: float) -> None:
    """Empirical SP325 wideband-mode check.

    Captures from the configured source for 5 s and reports whether the
    spectral signature matches the verified-good config (firmware
    1.3.6.0, AI Noise Cancellation OFF on both directions, EQ preset
    Default). A ``fail`` result means *something* about the device
    regressed — the operator should re-check Dell Peripheral Manager
    on a Windows machine. See speakerphone/compliance.py for the
    failure modes covered.

    Exit codes: 0 pass, 2 warn, 3 fail.
    """
    import json as _json
    import sys as _sys

    from meeting_scribe.speakerphone import compliance
    from meeting_scribe.speakerphone.daemon import _guess_pipewire_source_name

    target = node_name or _guess_pipewire_source_name(device_key)
    if not target:
        click.secho(
            f"no PipeWire source name for device_key={device_key}; pass --node explicitly",
            fg="red",
        )
        _sys.exit(2)

    min_hbp, min_rolloff = compliance.expected_thresholds(device_key)
    result = compliance.probe_device(
        target,
        capture_seconds=seconds,
        min_high_band_pct=min_hbp,
        min_rolloff_pct=min_rolloff,
    )
    click.echo(_json.dumps(result.to_dict(), indent=2))
    exit_code = {"pass": 0, "warn": 2, "fail": 3}.get(result.status, 3)
    _sys.exit(exit_code)


@speakerphone_group.command("benchmark")
@click.option(
    "--node",
    "node_name",
    default=None,
    help="PipeWire source node (default: SP325 pro-input-0).",
)
@click.option(
    "--device-key",
    "device_key",
    default="413c:8223",
    show_default=True,
    help="Catalog key for threshold lookup.",
)
@click.option(
    "--baseline-seconds",
    type=float,
    default=30.0,
    show_default=True,
    help="Pre-trial settle so the device drifts back to its natural state.",
)
@click.option(
    "--settle-seconds",
    type=float,
    default=15.0,
    show_default=True,
    help="DSP-flush wait after each SET_REPORT.",
)
@click.option(
    "--samples-per-cell",
    type=int,
    default=5,
    show_default=True,
)
@click.option(
    "--capture-seconds",
    type=float,
    default=3.0,
    show_default=True,
    help="Per-sample audio capture window.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the JSON report to this path "
    "(default: reports/speakerphone-quality/sp325-benchmark-<ts>.json).",
)
@click.option(
    "--quick/--full",
    default=False,
    help=(
        "Quick: 1 sample per cell, 3 s baseline, 5 s settle "
        "(~90 s total) — for smoke / wiring verification, NOT trend data."
    ),
)
def benchmark_cmd(
    node_name: str | None,
    device_key: str,
    baseline_seconds: float,
    settle_seconds: float,
    samples_per_cell: int,
    capture_seconds: float,
    output_path: Path | None,
    quick: bool,
) -> None:
    """Sweep SP325 settings and grade each cell's wideband signature.

    Iterates the ``benchmark.DEFAULT_CELLS`` matrix — the three known
    winners (D0/0x02, D1/0x01, C0/0x04) plus the DPM-derived NR-off
    pair that REGRESSES SP325 (D2/0x01, D5/0x01) plus coverage drift
    on D0/D1/D3/D4/C0 — and writes a JSON + Markdown summary to
    ``reports/speakerphone-quality/``.

    ETA at defaults: ~10 minutes (10 cells × ~60 s).

    Use ``--quick`` for ~90 s smoke; the ``--full`` defaults are the
    trend-data settings.

    Exit codes:
      0 — every known winner (label ∈ wideband_enable_*, eq_preset_default)
          ranked ``num_pass ≥ 4`` AND every known regression
          (mic_ns_*_set_off_dpm) dropped median below threshold.
      1 — winners failed to win or regressions failed to regress.
      2 — hardware unavailable.
    """
    import sys as _sys

    from meeting_scribe.speakerphone import benchmark
    from meeting_scribe.speakerphone.daemon import _guess_pipewire_source_name

    target = node_name or _guess_pipewire_source_name(device_key)
    if not target:
        click.secho(
            f"no PipeWire source name for device_key={device_key}; pass --node explicitly",
            fg="red",
        )
        _sys.exit(2)

    if quick:
        baseline_seconds = min(baseline_seconds, 3.0)
        settle_seconds = min(settle_seconds, 5.0)
        samples_per_cell = 1

    def _progress(i: int, n: int, cell: benchmark.Cell, phase: str) -> None:
        click.echo(
            f"[{i:>2}/{n}] {cell.key:<35} — {phase}",
            err=True,
        )

    click.secho(
        f"sweeping {len(benchmark.DEFAULT_CELLS)} cells on {target} "
        f"(ETA ~{int(len(benchmark.DEFAULT_CELLS) * (baseline_seconds + settle_seconds + samples_per_cell * capture_seconds + (samples_per_cell - 1) * 1.0))}s)",
        fg="cyan",
        err=True,
    )

    try:
        report = benchmark.run_sweep(
            target,
            device_key=device_key,
            baseline_seconds=baseline_seconds,
            settle_seconds=settle_seconds,
            samples_per_cell=samples_per_cell,
            capture_seconds=capture_seconds,
            progress=_progress,
        )
    except Exception as e:
        click.secho(f"sweep aborted: {e}", fg="red", err=True)
        _sys.exit(2)

    repo_root = Path(__file__).resolve().parents[3]
    out_path = output_path or benchmark.default_report_path(repo_root / "reports")
    benchmark.write_report(report, out_path)

    click.echo(benchmark.render_markdown_table(report))
    click.secho(f"\nJSON: {out_path}", fg="green", err=True)
    click.secho(f"MD:   {out_path.with_suffix('.md')}", fg="green", err=True)

    winners_by_label = {
        r.cell.label: r.is_winner
        for r in report.cells
        if r.cell.label.startswith(("wideband_enable_", "eq_preset_default"))
    }
    regressions_by_label = {
        r.cell.label: r.median_high_band_pct < report.min_high_band_pct
        for r in report.cells
        if r.cell.label.startswith("mic_ns_") and r.cell.label.endswith("_dpm")
    }
    winners_ok = all(winners_by_label.values()) if winners_by_label else False
    regressions_ok = all(regressions_by_label.values()) if regressions_by_label else False
    if winners_ok and regressions_ok:
        _sys.exit(0)
    click.secho(
        f"GATE FAILED: winners={winners_by_label} regressions={regressions_by_label}",
        fg="red",
        err=True,
    )
    _sys.exit(1)


@speakerphone_group.command("detect-rate")
@click.argument("node_name")
@click.option(
    "--default",
    type=int,
    default=48000,
    show_default=True,
    help="Fallback rate when detection fails.",
)
def detect_rate(node_name: str, default: int) -> None:
    """Show what capture rate ``server_mic`` will pick for this source.

    Validates the SP325/SP3022 quality fix: a Dell speakerphone source
    should resolve to 16000, while a Poly Sync 20-M source resolves to
    48000. Useful for debugging an unexpected ASR-quality regression.
    """
    from meeting_scribe.audio.native_rate import (
        _wpctl_alsa_card_dev,
        detect_capture_rate,
        supported_capture_rates,
    )

    resolved = _wpctl_alsa_card_dev(node_name)
    if resolved is None:
        click.echo(f"{node_name}: could not resolve to ALSA card", err=True)
    else:
        card, dev = resolved
        rates = supported_capture_rates(card)
        click.echo(f"{node_name} → card{card}/device{dev} rates={rates}")
    picked = detect_capture_rate(node_name, default=default)
    click.echo(f"picked: {picked} Hz")


@speakerphone_group.command("capture-descriptor")
@click.option(
    "--hidraw",
    "hidraw_name",
    type=str,
    default="hidraw0",
    show_default=True,
    help="hidraw node name (basename only, e.g. ``hidraw0``).",
)
def capture_descriptor(hidraw_name: str) -> None:
    """Dump + decode the HID report descriptor for a connected device.

    Reads from ``/sys/class/hidraw/<name>/device/report_descriptor``
    (world-readable), not ``/dev/hidraw*`` (root:root unless the udev
    rule is installed), so this works before ``install`` runs.
    """
    sysfs = Path("/sys/class/hidraw") / hidraw_name / "device" / "report_descriptor"
    if not sysfs.exists():
        click.secho(f"no descriptor at {sysfs}", fg="red")
        sys.exit(2)
    data = sysfs.read_bytes()
    click.echo(f"# report descriptor: {len(data)} bytes")
    click.echo(data.hex())
    click.echo()
    from meeting_scribe.speakerphone.descriptor import describe, parse_descriptor

    entries = parse_descriptor(data)
    click.echo(describe(entries))


@speakerphone_group.command("test")
@click.option(
    "--seconds",
    type=int,
    default=20,
    show_default=True,
    help="Capture window in seconds.",
)
def test_cmd(seconds: int) -> None:
    """Interactive: press buttons; verify each emits the expected event.

    Walks the operator through Phone / Teams / Phone-Mute. Read-only —
    does not touch settings or LEDs. Use ``meeting-scribe speakerphone
    capture-descriptor`` to inspect vendor-page emissions for the
    Teams button if it's not arriving as KEY_PHONE.
    """
    try:
        import evdev
    except Exception:
        click.secho("evdev unavailable; install python-evdev first", fg="red")
        sys.exit(2)

    devices = [evdev.InputDevice(p) for p in evdev.list_devices()]
    target = next(
        (
            d
            for d in devices
            if (
                d.info.vendor == 0x413C
                and d.info.product in (0x8222, 0x8223)
                and "leds" in [c.name for c in d.capabilities(verbose=False).get(0x11, [])]
            )
            or "Speakerphone" in d.name
        ),
        None,
    )
    if target is None:
        click.secho("no SP325/SP3022 detected", fg="red")
        sys.exit(2)
    click.echo(f"Capturing from {target.path} ({target.name}) for {seconds} s")
    click.echo("Press each button (Phone / Teams / Phone Mute). Ctrl-C to stop.")

    import select
    import time

    deadline = time.monotonic() + seconds
    seen: dict[int, int] = {}
    while time.monotonic() < deadline:
        timeout = max(0.0, deadline - time.monotonic())
        r, _, _ = select.select([target.fd], [], [], timeout)
        if not r:
            continue
        for event in target.read():
            if event.type != evdev.ecodes.EV_KEY:
                continue
            seen[event.code] = seen.get(event.code, 0) + 1
            click.echo(f"  KEY code={event.code} value={event.value} ts={event.timestamp():.3f}")
    click.echo()
    click.echo("Distinct key codes seen:")
    for code, count in sorted(seen.items()):
        name = evdev.ecodes.KEY.get(code, "?")
        click.echo(f"  {code:5d}  {name}  x{count}")
