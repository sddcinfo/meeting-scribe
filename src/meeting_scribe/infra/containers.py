"""Container lifecycle for meeting-scribe's model stack.

Docker Compose (`docker-compose.gb10.yml`) is the single source of truth
for which images run. This module provides the helpers that aren't
compose's job: listing scribe-prefixed containers and pulling
HuggingFace models into the local cache.
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

# Container naming prefix
CONTAINER_PREFIX = "scribe"


#
# NOTE: ``build_vllm_image`` (custom spark-vllm-docker / TurboQuant build)
# was removed in 1.5.0. ASR + TTS now run on stock ``vllm/vllm-openai``
# pulled by docker compose. Set ``SCRIBE_VLLM_IMAGE`` if you need to
# override the default.
#
# ``start_container``, ``stop_container``, ``restart_container``, and
# ``stop_all_containers`` were removed on 2026-04-14. The stack is
# managed exclusively through ``docker-compose.gb10.yml``; the CLI
# wraps compose via ``infra/compose.py``.
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
