"""Container lifecycle for meeting-scribe's model stack.

Builds spark-vllm-docker images, starts/stops model containers,
and manages the multi-container GB10 deployment.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meeting_scribe.infra.local import LocalRunner
    from meeting_scribe.infra.ssh import SSHRunner

    # Either runner works — both expose `run`, `docker_ps`, `rsync`, etc.
    # Alias lets call-sites pass a LocalRunner without casting.
    Runner = LocalRunner | SSHRunner
else:
    # Runtime: the names aren't needed; TYPE_CHECKING-only import keeps
    # these modules out of the hot-path imports.
    Runner = object

logger = logging.getLogger(__name__)

# spark-vllm-docker repos
SPARK_VLLM_REPO = "https://github.com/eugr/spark-vllm-docker.git"
TURBOQUANT_REPO = "https://github.com/bjk110/spark_vllm_docker.git"
TURBOQUANT_BRANCH = "feat/turboquant"

# Container naming prefix
CONTAINER_PREFIX = "scribe"

# Default build/storage directory on GB10
DATA_MOUNT = "/data"
BUILD_DIR = f"{DATA_MOUNT}/spark-vllm"


def build_vllm_image(
    ssh: Runner,
    *,
    turboquant: bool = False,
) -> str:
    """Build spark-vllm-docker image on the GB10 node.

    Args:
        ssh: SSH connection to the GB10 node.
        turboquant: If True, build from bjk110's TurboQuant branch.

    Returns:
        Docker image tag that was built.
    """
    if turboquant:
        repo = TURBOQUANT_REPO
        branch = TURBOQUANT_BRANCH
        tag = "bjk110/spark-vllm:turboquant"
    else:
        repo = SPARK_VLLM_REPO
        branch = "main"
        tag = "eugr/spark-vllm:latest"

    # Clone or update repo
    result = ssh.run(["test", "-d", f"{BUILD_DIR}/.git"], check=False)
    if result.returncode == 0:
        logger.info("Updating spark-vllm-docker at %s", BUILD_DIR)
        ssh.run(
            [
                "bash",
                "-c",
                f"cd {BUILD_DIR} && git fetch origin && git checkout {branch} && git pull",
            ],
            timeout=120,
        )
    else:
        logger.info("Cloning spark-vllm-docker from %s", repo)
        ssh.run(
            ["git", "clone", "-b", branch, repo, BUILD_DIR],
            timeout=300,
        )

    # Build Docker image (~20-30 minutes)
    logger.info("Building Docker image: %s (this may take 20-30 minutes)", tag)
    ssh.run(
        ["bash", "-c", f"cd {BUILD_DIR} && docker build -t {tag} ."],
        timeout=3600,
    )

    return tag


#
# NOTE: ``start_container``, ``stop_container``, ``restart_container``,
# and ``stop_all_containers`` were removed on 2026-04-14. The stack
# is now managed exclusively through ``docker-compose.gb10.yml``;
# the CLI wraps compose via ``infra/compose.py``. Recipes remain
# useful for pull-models + test assertions but are no longer the
# runtime launch path.
#
# If you need per-container lifecycle operations, use:
#
#   from meeting_scribe.infra.compose import compose_restart, compose_up, compose_down
#


def list_containers(ssh: Runner) -> list[dict[str, str]]:
    """List running meeting-scribe containers."""
    output = ssh.docker_ps(name_filter=CONTAINER_PREFIX)
    if not output:
        return []

    containers = []
    for line in output.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        containers.append(
            {
                "id": parts[0],
                "name": parts[1] if len(parts) > 1 else "",
                "status": parts[2] if len(parts) > 2 else "",
            }
        )
    return containers


def pull_models(
    ssh: Runner,
    model_ids: list[str],
    hf_cache_dir: str = "/data/huggingface",
) -> None:
    """Download models to the GB10's HuggingFace cache.

    Args:
        ssh: SSH connection to the GB10 node.
        model_ids: List of HuggingFace model IDs to download.
        hf_cache_dir: Path to HuggingFace cache on the GB10.
    """
    for model_id in model_ids:
        logger.info("Downloading model: %s", model_id)
        ssh.run(
            [
                "bash",
                "-c",
                f'HF_HOME="{hf_cache_dir}" huggingface-cli download "{model_id}"',
            ],
            timeout=3600,
            check=False,
        )
