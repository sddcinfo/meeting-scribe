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


def _hf_token_in_dotenv(path) -> bool:
    """True iff ``path`` already contains a non-empty HF_TOKEN= line."""
    try:
        if not path.exists():
            return False
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN=") and line.partition("=")[2].strip():
                return True
    except OSError:
        return False
    return False


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

    # 3. Check HF_TOKEN — and validate it against every gated model
    #    meeting-scribe will download. Plan §1.3 call site #3
    #    (interactive/advisory: network errors warn-only, token/EULA
    #    failures hard-fail, all-OK saves token to .env).
    dotenv = PROJECT_ROOT / ".env"
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token and dotenv.exists():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                hf_token = line.partition("=")[2].strip()
                break

    if not hf_token:
        click.secho("[MISSING] HF_TOKEN not configured.", fg="yellow")
        click.echo(
            "\nMeeting-scribe needs a Hugging Face READ token to download these\n"
            '4 gated models. You MUST click "Agree and access repository" on\n'
            "each one (in a browser, while logged in to the same HF account)\n"
            "BEFORE pasting the token here:\n"
        )
        # Recipe-driven, so adding a new gated model auto-extends the prompt.
        try:
            from meeting_scribe.recipes import all_model_ids

            for mid in all_model_ids(include_shared=True):
                click.echo(f"  https://huggingface.co/{mid}")
        except Exception:
            # Hard-fallback list — only used if recipes import fails.
            for mid in (
                "Qwen/Qwen3.6-35B-A3B-FP8",
                "Qwen/Qwen3-ASR-1.7B",
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                "pyannote/speaker-diarization-community-1",
            ):
                click.echo(f"  https://huggingface.co/{mid}")
        click.echo(
            "\nMint a token at: https://huggingface.co/settings/tokens\n"
            '(role "read" is sufficient; "write" is unnecessary).\n'
        )
        token = click.prompt(
            "Paste the token here, or press Enter to skip",
            default="",
            show_default=False,
        ).strip()
        if not token:
            issues.append("HF_TOKEN not configured (needed for model downloads)")
        else:
            hf_token = token

    if hf_token:
        # Validate against gated-model list before persisting to .env.
        from meeting_scribe.hf_preflight import validate_hf_access
        from meeting_scribe.recipes import all_model_ids

        click.echo(f"\nValidating HF_TOKEN ({hf_token[:6]}…) against gated models…")
        report = validate_hf_access(hf_token, all_model_ids(include_shared=True))
        if report.ok:
            click.secho(f"[OK] HF_TOKEN verified for {len(report.results)} model(s)", fg="green")
            if not _hf_token_in_dotenv(dotenv):
                with open(dotenv, "a") as f:
                    f.write(f"\nHF_TOKEN={hf_token}\n")
                click.secho("[OK] HF_TOKEN saved to .env (mode 0600)", fg="green")
                os.chmod(dotenv, 0o600)
        elif report.has_only_network_failures:
            # Advisory site: log yellow + save token + flag in issues.
            click.secho(report.render(), fg="yellow", err=True)
            click.secho(
                "[WARN] Network unreachable from this host — token saved unverified. "
                "The customer-side probe (during `sddc gb10 onboard` stage 2.5) is "
                "the authoritative gate.",
                fg="yellow",
            )
            if not _hf_token_in_dotenv(dotenv):
                with open(dotenv, "a") as f:
                    f.write(f"\nHF_TOKEN={hf_token}\n")
                os.chmod(dotenv, 0o600)
            issues.append("HF_TOKEN saved but not verified — network unreachable")
        else:
            # Token invalid OR EULA missing OR not-found → don't save, hard-fail.
            click.secho(report.render(), fg="red", err=True)
            issues.append("HF_TOKEN failed validation — see above for the model URL(s) to accept")

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
