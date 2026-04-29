"""Thin wrapper around ``docker compose -f docker-compose.gb10.yml``.

Meeting-scribe used to have two parallel launch systems:

  1. ``docker-compose.gb10.yml`` — compose file with container_names,
     volumes, ports, labels for willfarrell/autoheal, profiles for
     gated services (e.g. ``bench`` for benchmark sidecars), and a
     single source of truth for the stack topology.

  2. ``src/meeting_scribe/recipes/*.yaml`` + ``infra/containers.py`` —
     a parallel path that reimplemented half of the above by reading
     recipes and calling ``docker run`` directly.

The two paths produced containers with DIFFERENT names for the same
model (e.g. ``scribe-asr`` from compose, ``scribe-asr-vllm`` from the
recipe) and neither cleaned up the other's containers. On 2026-04-14
we hit a system-RAM OOM because the recipe path was launching a
duplicate of the 35B translate model every time ``meeting-scribe gb10
up`` ran, while autosre's 35B was already serving the same port.

The refactor (2026-04-14):

  - Compose is the single source of truth. Every container the stack
    runs lives in ``docker-compose.gb10.yml``.
  - This module is a thin wrapper around ``docker compose`` subprocess
    calls, exposing ``compose_up``, ``compose_down``, ``compose_restart``,
    and ``compose_services`` for the CLI to call.
  - ``infra/containers.py`` is now limited to docker-level helpers
    (``list_containers``, ``pull_models``) used for pre-flight tasks
    before compose runs.
  - The recipe files stay useful for:
      - pull-models (iterates recipe model_ids)
      - test assertions (port expectations, model_id stability)
    but are NOT used by the runtime launch path.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILE = _REPO_ROOT / "docker-compose.gb10.yml"


def _compose_cmd(*args: str) -> list[str]:
    return ["docker", "compose", "-f", str(COMPOSE_FILE), *args]


def _run(cmd: list[str], timeout: int) -> None:
    logger.info("compose: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"docker compose failed (rc={result.returncode}): {stderr}")


def compose_up(
    services: list[str] | None = None,
    profiles: list[str] | None = None,
    pull: str | None = None,
) -> None:
    """Bring up the stack (detached). Optional service + profile filters.

    Without filters, starts every service in the compose file EXCEPT
    those gated behind a profile (e.g. ``funcosyvoice`` under the
    ``bench`` profile stays off until explicitly requested).

    Args:
        pull: Docker compose pull policy (``"never"``, ``"always"``,
              ``"missing"``). ``None`` keeps Docker's default (pull if
              missing). Pass ``"never"`` from the systemd boot path to
              avoid hanging on network-unavailable cold boots.
    """
    cmd: list[str] = []
    if profiles:
        for p in profiles:
            cmd.extend(["--profile", p])
    up_args = ["up", "-d"]
    if pull:
        up_args.extend(["--pull", pull])
    cmd = _compose_cmd(*cmd, *up_args)
    if services:
        cmd.extend(services)
    _run(cmd, timeout=600)  # 10 min — create can be slow on cold boot


def compose_down(remove_volumes: bool = False) -> None:
    """Stop + remove the stack containers.

    ``remove_volumes=False`` by default — HF model cache volumes are
    too expensive to rebuild (model re-download).
    """
    args = ["down"]
    if remove_volumes:
        args.append("-v")
    _run(_compose_cmd(*args), timeout=300)


def compose_restart(service: str, timeout: int = 30, recreate: bool = False) -> None:
    """Restart a single compose service.

    Default mode is `docker compose restart`, which preserves the running
    container's environment, image, and volumes — fast (~5s) and right for
    CUDA-context recovery.

    Pass ``recreate=True`` for a full recreate via
    ``docker compose up -d --force-recreate``, which is required when the
    compose file's environment / image / volume / port spec has changed.
    A plain restart silently keeps the OLD env even if the file was edited.
    """
    if recreate:
        # `up -d --force-recreate` alone fails with "container is running:
        # stop the container before removing" on host-network services where
        # the network reuse logic gets confused. The reliable shape is
        # explicit stop → rm → up. Each step is bounded by the same timeout
        # budget so a hung container can't park us indefinitely.
        _run(_compose_cmd("stop", "-t", str(timeout), service), timeout=timeout + 30)
        _run(_compose_cmd("rm", "-f", service), timeout=timeout + 30)
        _run(_compose_cmd("up", "-d", "--no-deps", service), timeout=timeout + 120)
    else:
        _run(_compose_cmd("restart", "-t", str(timeout), service), timeout=timeout + 60)


def compose_services() -> list[str]:
    """Return the list of service names declared in the compose file
    (including profile-gated ones)."""
    result = subprocess.run(
        _compose_cmd("config", "--services"),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        return []
    return [s.strip() for s in result.stdout.splitlines() if s.strip()]
