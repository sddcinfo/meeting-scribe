"""Local command runner — mirrors SSHRunner but executes via subprocess.

Used when meeting-scribe runs on the GB10 itself (the common dev setup).
"""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)


class LocalRunner:
    """Execute commands on the local host with the same surface as SSHRunner."""

    def __init__(self) -> None:
        self.node = None

    @property
    def ssh_target(self) -> str:
        return "local"

    def run(
        self,
        cmd: list[str],
        timeout: int = 30,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        logger.debug("LOCAL: %s", " ".join(cmd))
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
        )

    def run_bg(self, cmd: list[str]) -> str:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return str(proc.pid)

    def is_reachable(self, timeout: int = 5) -> bool:
        return True

    def rsync(
        self,
        src: str,
        dest: str,
        exclude: list[str] | None = None,
        timeout: int = 3600,
    ) -> bool:
        rsync_cmd = ["rsync", "-a", "--progress"]
        if exclude:
            for pattern in exclude:
                rsync_cmd.extend(["--exclude", pattern])
        rsync_cmd.extend([src, dest])
        result = subprocess.run(
            rsync_cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return result.returncode == 0

    def docker_run(
        self,
        image: str,
        cmd: list[str] | None = None,
        *,
        name: str | None = None,
        detach: bool = True,
        remove: bool = True,
        gpus: str = "all",
        network: str = "host",
        volumes: list[str] | None = None,
        env: dict[str, str] | None = None,
        shm_size: str = "16g",
        extra_args: list[str] | None = None,
    ) -> str:
        docker_cmd = ["docker", "run"]
        if detach:
            docker_cmd.append("-d")
        if remove:
            docker_cmd.append("--rm")
        if name:
            docker_cmd.extend(["--name", name])
        if gpus:
            docker_cmd.extend(["--gpus", gpus])
        if network:
            docker_cmd.extend(["--network", network])
        if shm_size:
            docker_cmd.extend(["--shm-size", shm_size])
        docker_cmd.extend(["--ulimit", "memlock=-1"])

        if volumes:
            for vol in volumes:
                docker_cmd.extend(["-v", vol])
        if env:
            for key, value in env.items():
                docker_cmd.extend(["-e", f"{key}={value}"])
        if extra_args:
            docker_cmd.extend(extra_args)

        docker_cmd.append(image)
        if cmd:
            docker_cmd.extend(cmd)

        result = self.run(docker_cmd, timeout=60, check=True)
        return result.stdout.strip()

    def docker_stop(self, container_id: str, timeout: int = 30) -> bool:
        result = self.run(
            ["docker", "stop", "-t", str(timeout), container_id],
            timeout=timeout + 10,
            check=False,
        )
        return result.returncode == 0

    def docker_container_exists(self, name: str) -> tuple[bool, bool]:
        """Return (exists, running). Used by start_container() to pick
        between `docker start` (existing) and `docker run` (fresh)."""
        result = self.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", name],
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return (False, False)
        running = result.stdout.strip().lower() == "true"
        return (True, running)

    def docker_start(self, container_id: str) -> bool:
        result = self.run(
            ["docker", "start", container_id],
            timeout=30,
            check=False,
        )
        return result.returncode == 0

    def docker_remove(self, container_id: str, force: bool = True) -> bool:
        args = ["docker", "rm"]
        if force:
            args.append("-f")
        args.append(container_id)
        result = self.run(args, timeout=30, check=False)
        return result.returncode == 0

    def docker_restart(self, container_id: str, timeout: int = 30) -> bool:
        """Restart a running container in place (single docker restart).

        Used to recover a container whose process is alive but whose
        CUDA context has been corrupted (e.g. pyannote after concurrent
        calls wedged the GPU). Much faster than docker stop + up for
        a single container.
        """
        result = self.run(
            ["docker", "restart", "-t", str(timeout), container_id],
            timeout=timeout + 30,
            check=False,
        )
        return result.returncode == 0

    def docker_ps(self, name_filter: str | None = None) -> str:
        cmd = ["docker", "ps", "--format", "{{.ID}}\t{{.Names}}\t{{.Status}}"]
        if name_filter:
            cmd.extend(["--filter", f"name={name_filter}"])
        result = self.run(cmd, timeout=10, check=False)
        return result.stdout.strip()
