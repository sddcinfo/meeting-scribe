"""gb10 sub-group: provision + lifecycle for the GB10 model containers."""

from __future__ import annotations

import os
import sys

import click

from meeting_scribe.cli import cli
from meeting_scribe.cli._common import DEFAULT_PORT


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
def setup(host: str) -> None:
    """Provision a GB10 node: pull HuggingFace models into the local cache.

    Container images are pulled by docker compose on ``meeting-scribe gb10 up``.
    """
    from meeting_scribe.infra.containers import pull_models
    from meeting_scribe.infra.runner import get_runner
    from meeting_scribe.recipes import all_model_ids

    ssh = get_runner(host)

    if not ssh.is_reachable():
        click.secho(f"Cannot reach GB10 at {host}", fg="red")
        sys.exit(1)

    click.secho(f"Connected to GB10 at {host}", fg="green")

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
    ``vllm-asr``, ``qwen3-tts``). Run
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
    from click.testing import CliRunner

    from meeting_scribe.cli.lifecycle import start

    os.environ["SCRIBE_PROFILE"] = "gb10"
    os.environ.setdefault("SCRIBE_HOST", "0.0.0.0")
    click.echo("Starting with GB10 profile (Qwen3-ASR, vLLM, pyannote)...")
    click.echo(f"  Bind: 0.0.0.0:{port} (accessible from hotspot)")

    runner = CliRunner()
    args = [f"--port={port}", "--foreground"]
    if debug:
        args.append("--debug")
    runner.invoke(start, args)
