"""First-run setup + thin top-level aliases that proxy gb10 subcommands.

``meeting-scribe up`` / ``down`` / ``containers`` are kept as top-level
aliases of ``meeting-scribe gb10 up|down|status`` so the most common
operator commands stay one word.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import click

from meeting_scribe.cli import cli
from meeting_scribe.cli._common import (
    PROJECT_ROOT,
    _ensure_admin_tls_certs,
    _ensure_port80_bind,
)


@cli.command("setup")
def first_run_setup():
    """First-time setup — install dependencies, configure credentials, pull models."""
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
    ok, detail = _ensure_admin_tls_certs()
    if ok:
        click.secho(f"[OK] {detail}", fg="green")
    else:
        click.secho(f"[FAIL] {detail}", fg="red")
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
    from meeting_scribe.cli.gb10 import gb10_up

    gb10_up.callback(host=host, offline=False)


@cli.command()
@click.option("--host", envvar="SCRIBE_GB10_HOST", default="localhost")
def down(host):
    """Stop all model containers (alias for gb10 down)."""
    from meeting_scribe.cli.gb10 import gb10_down

    gb10_down.callback(host=host)


@cli.command()
@click.option("--host", envvar="SCRIBE_GB10_HOST", default="localhost")
def containers(host):
    """Check container status (alias for gb10 status)."""
    from meeting_scribe.cli.gb10 import gb10_status

    gb10_status.callback(host=host)
