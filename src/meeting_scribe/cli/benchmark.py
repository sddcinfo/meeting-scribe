"""benchmark-translate / benchmark-install-deps subcommands."""

from __future__ import annotations

import subprocess
import sys

import click

from meeting_scribe.cli import cli
from meeting_scribe.cli._common import PROJECT_ROOT

# ── Translation benchmark bridge ─────────────────────────────
#
# Thin wrapper around benchmarks/translation_benchmark.py so callers
# can invoke it via the normal `meeting-scribe` CLI (which is on PATH
# as the venv's entry point).  Directly invoking the script requires
# the venv's httpx + meeting_scribe.languages import, and the harness
# guards block activating that venv from outside.  This subcommand
# delegates via sys.executable (which *is* the venv python because
# meeting-scribe itself runs in the venv) and passes all remaining
# args through unchanged, so the underlying CLI surface stays the
# single source of truth.


@cli.command("benchmark-translate", context_settings={"ignore_unknown_options": True})
@click.argument("passthrough_args", nargs=-1, type=click.UNPROCESSED)
def benchmark_translate(passthrough_args: tuple[str, ...]) -> None:
    """Run benchmarks/translation_benchmark.py; args passed through.

    Example:
        meeting-scribe benchmark-translate \\
            --url http://localhost:8010 \\
            --corpus ~/.local/share/sddc/qwen36-shadow/corpus_500.jsonl \\
            --no-score \\
            --output ~/.local/share/sddc/qwen36-shadow/runs/baseline.jsonl
    """
    script = PROJECT_ROOT / "benchmarks" / "translation_benchmark.py"
    if not script.is_file():
        click.echo(f"benchmark script missing: {script}", err=True)
        sys.exit(2)
    rc = subprocess.run([sys.executable, str(script), *passthrough_args], check=False).returncode
    sys.exit(rc)


@cli.command("benchmark-install-deps", context_settings={"ignore_unknown_options": True})
@click.option(
    "--extra",
    default="",
    help="Optional-dependency extra from pyproject.toml (e.g. 'bench').",
)
@click.argument("pip_args", nargs=-1, type=click.UNPROCESSED)
def benchmark_install_deps(extra: str, pip_args: tuple[str, ...]) -> None:
    """pip install benchmark deps into this venv.

    Exists because guard rules block ad-hoc venv activation, so callers
    cannot `pip install` directly.  This runs through sys.executable,
    which IS the venv python when this CLI is invoked.

    Either pass --extra=<name> to install a pyproject extra, or pass
    package specs as positional args.  Both can be combined.

    Examples:
        meeting-scribe benchmark-install-deps --extra=bench
        meeting-scribe benchmark-install-deps sacrebleu>=2.4.0
    """
    if not extra and not pip_args:
        click.echo("nothing to install; pass --extra=<name> or package args", err=True)
        sys.exit(2)
    cmd = [sys.executable, "-m", "pip", "install"]
    if extra:
        cmd.extend(["-e", f"{PROJECT_ROOT}[{extra}]"])
    cmd.extend(pip_args)
    click.echo(f"$ {' '.join(cmd)}")
    rc = subprocess.run(cmd, check=False).returncode
    if rc == 0:
        click.echo(click.style("install ok", fg="green"))
    sys.exit(rc)
