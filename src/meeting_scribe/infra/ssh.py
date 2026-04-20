"""SSH transport via subprocess. No paramiko dependency.

Adapted from auto-sre's infra.ssh for meeting-scribe's
multi-container model stack on GB10.
"""

from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meeting_scribe.infra.types import GB10Node

logger = logging.getLogger(__name__)


class SSHRunner:
    """Execute commands on a remote GB10 node via SSH.

    Uses subprocess.run(["ssh", ...]) for zero compiled dependencies.
    Assumes key-based auth is configured (password auth not supported).
    """

    def __init__(self, node: GB10Node) -> None:
        self.node = node

    def _build_ssh_cmd(self, cmd: list[str]) -> list[str]:
        """Build the full SSH command with options."""
        ssh_cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "BatchMode=yes",
        ]
        if self.node.ssh_key:
            ssh_cmd.extend(["-i", self.node.ssh_key])
        ssh_cmd.append(self.node.ssh_target)
        ssh_cmd.extend(cmd)
        return ssh_cmd

    def run(
        self,
        cmd: list[str],
        timeout: int = 30,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run a command on the remote node.

        Args:
            cmd: Command and arguments to execute remotely.
            timeout: Timeout in seconds.
            check: If True, raise CalledProcessError on non-zero exit.

        Returns:
            CompletedProcess with stdout/stderr captured as text.
        """
        ssh_cmd = self._build_ssh_cmd(cmd)
        logger.debug("SSH: %s", " ".join(ssh_cmd))
        return subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
        )

    def run_bg(self, cmd: list[str]) -> str:
        """Run a command in the background on the remote node.

        Returns:
            The remote PID as a string.
        """
        bg_cmd = ["nohup", *cmd, ">/dev/null", "2>&1", "&", "echo", "$!"]
        result = self.run(bg_cmd, timeout=10, check=False)
        return result.stdout.strip()

    def is_reachable(self, timeout: int = 5) -> bool:
        """Check if the node is reachable via SSH."""
        try:
            result = self.run(["echo", "ok"], timeout=timeout, check=False)
            return result.returncode == 0 and "ok" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def rsync(
        self,
        src: str,
        dest: str,
        exclude: list[str] | None = None,
        timeout: int = 3600,
    ) -> bool:
        """Rsync files to/from the remote node.

        Args:
            src: Source path (local or remote with node prefix).
            dest: Destination path (local or remote with node prefix).
            exclude: Patterns to exclude.
            timeout: Timeout in seconds (default 1 hour for large model transfers).

        Returns:
            True if rsync succeeded.
        """
        rsync_cmd = [
            "rsync",
            "-az",
            "--progress",
            "-e",
            self._build_ssh_arg(),
        ]
        if exclude:
            for pattern in exclude:
                rsync_cmd.extend(["--exclude", pattern])
        rsync_cmd.extend([src, dest])

        result = subprocess.run(
            rsync_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return result.returncode == 0

    def _build_ssh_arg(self) -> str:
        """Build the -e argument for rsync."""
        parts = ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"]
        if self.node.ssh_key:
            parts.extend(["-i", self.node.ssh_key])
        return " ".join(parts)

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
        """Run a Docker container on the remote node.

        Returns:
            Container ID if detached, or stdout if not.
        """
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
        """Stop a Docker container on the remote node."""
        result = self.run(
            ["docker", "stop", "-t", str(timeout), container_id],
            timeout=timeout + 10,
            check=False,
        )
        return result.returncode == 0

    def docker_restart(self, container_id: str, timeout: int = 30) -> bool:
        """Restart a Docker container on the remote node in place."""
        result = self.run(
            ["docker", "restart", "-t", str(timeout), container_id],
            timeout=timeout + 30,
            check=False,
        )
        return result.returncode == 0

    def docker_container_exists(self, name: str) -> tuple[bool, bool]:
        """Return (exists, running) for a container by name."""
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
        """Start an existing (stopped) container."""
        result = self.run(
            ["docker", "start", container_id], timeout=30, check=False,
        )
        return result.returncode == 0

    def docker_remove(self, container_id: str, force: bool = True) -> bool:
        """Remove a container (force-kill if still running)."""
        args = ["docker", "rm"]
        if force:
            args.append("-f")
        args.append(container_id)
        result = self.run(args, timeout=30, check=False)
        return result.returncode == 0

    def docker_ps(self, name_filter: str | None = None) -> str:
        """List running Docker containers on the remote node."""
        cmd = ["docker", "ps", "--format", "{{.ID}}\t{{.Names}}\t{{.Status}}"]
        if name_filter:
            cmd.extend(["--filter", f"name={name_filter}"])
        result = self.run(cmd, timeout=10, check=False)
        return result.stdout.strip()
