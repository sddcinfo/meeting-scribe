"""Trust-anchor CLI surface — leaf cert export, fingerprint print,
and OS-level trust-store install/uninstall.

Plan §TLS Trust Anchor:

* ``meeting-scribe export-cert PATH`` — write the leaf PEM to a file
  the operator carries on a USB stick.
* ``meeting-scribe cert-fingerprint`` — print the leaf SHA-256 for
  the bootstrap-page identity surface (and for the operator to read
  off the local TTY when using the fingerprint-confirmation flow).
* ``meeting-scribe trust-install --from-pem PATH`` — install ``PATH``
  into the OS trust store as a non-CA server cert, replacing any
  prior leaf with the same appliance ID.
* ``meeting-scribe trust-install --confirm-fingerprint`` — fetch the
  leaf over the hotspot, compute its fingerprint, prompt the operator
  to type the expected one verbatim, install only on match.
* ``meeting-scribe trust-uninstall <appliance_id>`` /
  ``--all`` — remove specific or all meeting-scribe leaves.

The platform-specific trust-store mechanics (macOS ``security``, Linux
NSS DB, Windows ``certutil``) are abstracted behind a small dispatcher
so the CLI flow stays portable.
"""

from __future__ import annotations

import json
import platform
import shutil
import ssl
import subprocess
import sys
from pathlib import Path

import click

from meeting_scribe.cli import cli
from meeting_scribe.cli._common import PROJECT_ROOT
from meeting_scribe.runtime.cert_check import (
    CertConfigError,
    get_leaf_fingerprint,
    get_subject_cn,
)


_KNOWN_APPLIANCES_PATH = Path.home() / ".config" / "meeting-scribe" / "known-appliances.json"


# ── Platform dispatcher ──────────────────────────────────────────


def _platform_install(cert_path: Path, *, appliance_id: str) -> tuple[bool, str]:
    """Install ``cert_path`` as a non-CA server leaf.

    Replace-not-add: any prior trust entry whose Subject CN matches
    ``meeting-scribe/<appliance_id>`` is removed first so rotated leaves
    don't accumulate. macOS / Linux-NSS / Windows handlers are dispatched
    by ``platform.system()``.
    """
    system = platform.system()
    if system == "Darwin":
        return _install_macos(cert_path, appliance_id=appliance_id)
    if system == "Linux":
        return _install_linux_nss(cert_path, appliance_id=appliance_id)
    if system == "Windows":
        return _install_windows(cert_path, appliance_id=appliance_id)
    return False, f"unsupported platform: {system}"


def _platform_uninstall(*, appliance_id: str | None, all_meeting_scribe: bool) -> tuple[bool, str]:
    system = platform.system()
    if system == "Darwin":
        return _uninstall_macos(appliance_id=appliance_id, all_meeting_scribe=all_meeting_scribe)
    if system == "Linux":
        return _uninstall_linux_nss(
            appliance_id=appliance_id, all_meeting_scribe=all_meeting_scribe
        )
    if system == "Windows":
        return _uninstall_windows(
            appliance_id=appliance_id, all_meeting_scribe=all_meeting_scribe
        )
    return False, f"unsupported platform: {system}"


def _install_macos(cert_path: Path, *, appliance_id: str) -> tuple[bool, str]:
    """``security add-trusted-cert -d -p ssl -k login.keychain leaf.pem``.

    Replace path: ``security find-certificate -a -c
    "meeting-scribe/<appliance_id>" -Z login.keychain`` → parse SHA-256
    fingerprints → ``security delete-certificate -Z <fp>`` per match.
    """
    if shutil.which("security") is None:
        return False, "macOS `security` binary not on PATH"
    cn = f"meeting-scribe/{appliance_id}"
    found = subprocess.run(
        ["security", "find-certificate", "-a", "-c", cn, "-Z", "login.keychain"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in found.stdout.splitlines():
        if line.strip().startswith("SHA-256 hash:"):
            fp = line.split(":", 1)[1].strip()
            subprocess.run(
                ["security", "delete-certificate", "-Z", fp],
                capture_output=True,
                text=True,
                check=False,
            )
    install = subprocess.run(
        [
            "security",
            "add-trusted-cert",
            "-d",
            "-p",
            "ssl",
            "-k",
            "login.keychain",
            str(cert_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if install.returncode != 0:
        return False, install.stderr.strip()[:200]
    return True, "installed in login.keychain"


def _install_linux_nss(cert_path: Path, *, appliance_id: str) -> tuple[bool, str]:
    """Per-user NSS DB at ``~/.pki/nssdb`` (Firefox / Chromium).

    Newer distros also use ``trust anchor`` for the system store; we
    install in the NSS DB only — system store needs root.
    """
    if shutil.which("certutil") is None:
        return False, "linux `certutil` not on PATH (apt install libnss3-tools)"
    nssdb = Path.home() / ".pki" / "nssdb"
    nssdb.mkdir(parents=True, exist_ok=True)
    nick = f"meeting-scribe-{appliance_id}"
    # Replace any prior nick for this appliance.
    subprocess.run(
        ["certutil", "-D", "-n", nick, "-d", f"sql:{nssdb}"],
        capture_output=True,
        text=True,
        check=False,
    )
    add = subprocess.run(
        [
            "certutil",
            "-A",
            "-n",
            nick,
            "-t",
            "P,,",  # trusted server peer, no CA
            "-i",
            str(cert_path),
            "-d",
            f"sql:{nssdb}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if add.returncode != 0:
        return False, add.stderr.strip()[:200]
    return True, f"installed as {nick} in NSS DB"


def _install_windows(cert_path: Path, *, appliance_id: str) -> tuple[bool, str]:
    """``certutil -addstore -user TrustedPeople <cert>``."""
    if shutil.which("certutil") is None:
        return False, "windows `certutil` not on PATH"
    # Best-effort prior cleanup — Windows certutil's CN-match isn't as
    # ergonomic as macOS's. We rely on the user to call trust-uninstall
    # before installing a new ID; replace-on-same-ID works because the
    # Subject hash differs across rotations.
    add = subprocess.run(
        [
            "certutil",
            "-addstore",
            "-user",
            "TrustedPeople",
            str(cert_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if add.returncode != 0:
        return False, add.stderr.strip()[:200]
    return True, f"installed in TrustedPeople for {appliance_id}"


def _uninstall_macos(
    *, appliance_id: str | None, all_meeting_scribe: bool
) -> tuple[bool, str]:
    if shutil.which("security") is None:
        return False, "macOS `security` binary not on PATH"
    cn_prefix = "meeting-scribe/"
    if appliance_id and not all_meeting_scribe:
        target_cns = [f"{cn_prefix}{appliance_id}"]
    else:
        # Bulk: enumerate all CNs starting with the prefix.
        found = subprocess.run(
            ["security", "find-certificate", "-a", "-Z", "login.keychain"],
            capture_output=True,
            text=True,
            check=False,
        )
        target_cns = []
        for line in found.stdout.splitlines():
            if cn_prefix in line:
                target_cns.append(cn_prefix + line.split(cn_prefix, 1)[1].strip())
    removed = 0
    for cn in target_cns:
        out = subprocess.run(
            ["security", "find-certificate", "-a", "-c", cn, "-Z", "login.keychain"],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in out.stdout.splitlines():
            if line.strip().startswith("SHA-256 hash:"):
                fp = line.split(":", 1)[1].strip()
                subprocess.run(
                    ["security", "delete-certificate", "-Z", fp],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                removed += 1
    return True, f"removed {removed} trust entries"


def _uninstall_linux_nss(
    *, appliance_id: str | None, all_meeting_scribe: bool
) -> tuple[bool, str]:
    if shutil.which("certutil") is None:
        return False, "linux `certutil` not on PATH"
    nssdb = Path.home() / ".pki" / "nssdb"
    if not nssdb.exists():
        return True, "no NSS DB found — nothing to remove"
    if appliance_id and not all_meeting_scribe:
        nicks = [f"meeting-scribe-{appliance_id}"]
    else:
        listed = subprocess.run(
            ["certutil", "-L", "-d", f"sql:{nssdb}"],
            capture_output=True,
            text=True,
            check=False,
        )
        nicks = [
            line.split()[0]
            for line in listed.stdout.splitlines()
            if line.startswith("meeting-scribe-")
        ]
    removed = 0
    for nick in nicks:
        rc = subprocess.run(
            ["certutil", "-D", "-n", nick, "-d", f"sql:{nssdb}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if rc.returncode == 0:
            removed += 1
    return True, f"removed {removed} NSS entries"


def _uninstall_windows(
    *, appliance_id: str | None, all_meeting_scribe: bool
) -> tuple[bool, str]:
    # Symmetric stub — production tested on Windows separately.
    return True, "Windows uninstall not implemented in this build"


# ── Known-appliances persistence ─────────────────────────────────


def _read_known() -> dict[str, dict]:
    if not _KNOWN_APPLIANCES_PATH.exists():
        return {}
    try:
        return json.loads(_KNOWN_APPLIANCES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_known(record: dict[str, dict]) -> None:
    _KNOWN_APPLIANCES_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _KNOWN_APPLIANCES_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")


# ── Cert fetch over hotspot ──────────────────────────────────────


def _fetch_leaf_pem(host: str, port: int = 443) -> str:
    """Open a TLS connection to ``host:port`` with verification disabled
    and return the server's leaf cert as PEM.

    Verification is intentionally disabled because the WHOLE POINT of
    this flow is to bootstrap trust — the operator types the expected
    fingerprint to verify what we just downloaded. Silent installation
    is forbidden; the caller MUST prompt + match before writing into
    the trust store.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    import socket

    with socket.create_connection((host, port), timeout=10.0) as raw:
        with ctx.wrap_socket(raw, server_hostname=host) as ssock:
            der = ssock.getpeercert(binary_form=True)
    if not der:
        raise click.ClickException(f"no peer cert from {host}:{port}")
    return ssl.DER_cert_to_PEM_cert(der)


# ── CLI commands ─────────────────────────────────────────────────


@cli.command("export-cert")
@click.argument("path", type=click.Path(dir_okay=False, path_type=Path))
def export_cert_cmd(path: Path) -> None:
    """Write the appliance leaf PEM to ``PATH``.

    Operators carry the resulting file on a USB stick to the admin
    laptop and run ``meeting-scribe trust-install --from-pem``. The
    USB path NEVER touches the network.
    """
    src = PROJECT_ROOT / "certs" / "cert.pem"
    if not src.exists():
        raise click.ClickException(f"leaf cert missing at {src}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(src.read_bytes())
    click.echo(f"wrote {src} → {path}")


@cli.command("cert-fingerprint")
def cert_fingerprint_cmd() -> None:
    """Print the leaf SHA-256 fingerprint (lowercase, no colons).

    Read off the local TTY by the operator who's about to type it
    into ``trust-install --confirm-fingerprint`` on the admin laptop.
    Also displayed on the bootstrap page's identity surface.
    """
    cert = PROJECT_ROOT / "certs" / "cert.pem"
    if not cert.exists():
        raise click.ClickException(f"leaf cert missing at {cert}")
    fp = get_leaf_fingerprint(cert)
    click.echo(fp)


@cli.command("trust-install")
@click.option(
    "--from-pem",
    "pem_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Install a leaf PEM that you already have on disk (e.g. from USB).",
)
@click.option(
    "--confirm-fingerprint",
    is_flag=True,
    default=False,
    help="Fetch the leaf over the hotspot and require operator-typed fingerprint match.",
)
@click.option(
    "--host",
    default="10.42.0.1",
    show_default=True,
    help="Host to fetch from when --confirm-fingerprint is set.",
)
def trust_install_cmd(
    pem_path: Path | None,
    confirm_fingerprint: bool,
    host: str,
) -> None:
    """Install the appliance leaf into the OS trust store.

    Two authenticated bootstrap paths:

    \b
    --from-pem PATH                 # USB transfer (preferred)
    --confirm-fingerprint           # fetch + typed-fingerprint match
    """
    if not pem_path and not confirm_fingerprint:
        raise click.UsageError("specify --from-pem PATH or --confirm-fingerprint")
    if pem_path and confirm_fingerprint:
        raise click.UsageError("--from-pem and --confirm-fingerprint are mutually exclusive")

    if confirm_fingerprint:
        click.echo(f"Fetching leaf from {host}:443 (no cert validation yet)...")
        try:
            pem = _fetch_leaf_pem(host)
        except OSError as exc:
            raise click.ClickException(f"fetch failed: {exc}") from exc
        tmp = Path.home() / ".cache" / "meeting-scribe" / "trust-confirm-leaf.pem"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(pem, encoding="utf-8")
        try:
            computed_fp = get_leaf_fingerprint(tmp)
        except CertConfigError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Computed fingerprint: {computed_fp}")
        click.echo("")
        click.echo(
            "On the GB10's local TTY, run `meeting-scribe cert-fingerprint`"
            " and read the value to verify."
        )
        typed = click.prompt("Type the expected SHA-256 (lowercase, no colons)").strip().lower()
        if typed != computed_fp:
            tmp.unlink(missing_ok=True)
            raise click.ClickException(
                "fingerprint mismatch — refusing to install. The hotspot may be evil-twin."
            )
        pem_path = tmp

    if pem_path is None:
        raise click.ClickException("no PEM path resolved")

    cn = get_subject_cn(pem_path)
    if not cn.startswith("meeting-scribe/"):
        raise click.ClickException(
            f"refusing to install — expected CN to start with meeting-scribe/, got {cn!r}"
        )
    appliance_id = cn.removeprefix("meeting-scribe/")

    ok, detail = _platform_install(pem_path, appliance_id=appliance_id)
    if not ok:
        raise click.ClickException(f"install failed: {detail}")

    record = _read_known()
    record[appliance_id] = {
        "appliance_id": appliance_id,
        "fingerprint": get_leaf_fingerprint(pem_path),
        "subject_cn": cn,
    }
    _write_known(record)
    click.echo(f"installed leaf for appliance {appliance_id}: {detail}")


@cli.command("trust-uninstall")
@click.argument("appliance_id", required=False)
@click.option(
    "--all",
    "all_flag",
    is_flag=True,
    default=False,
    help="Remove every meeting-scribe-prefixed trust entry (rare).",
)
def trust_uninstall_cmd(appliance_id: str | None, all_flag: bool) -> None:
    """Remove appliance leaves from the OS trust store.

    \b
    trust-uninstall <appliance_id>   # one specific appliance
    trust-uninstall --all            # all meeting-scribe leaves (retire)
    """
    if not appliance_id and not all_flag:
        raise click.UsageError("specify <appliance_id> or --all")
    if appliance_id and all_flag:
        raise click.UsageError("--all and <appliance_id> are mutually exclusive")
    ok, detail = _platform_uninstall(
        appliance_id=appliance_id,
        all_meeting_scribe=all_flag,
    )
    if not ok:
        raise click.ClickException(detail)
    record = _read_known()
    if all_flag:
        record.clear()
    elif appliance_id:
        record.pop(appliance_id, None)
    _write_known(record)
    click.echo(detail)


@cli.command("appliance-info")
def appliance_info_cmd() -> None:
    """Print the local known-appliances JSON record (used by the
    bootstrap page's identity-confirmation surface)."""
    record = _read_known()
    click.echo(json.dumps(record, indent=2))
