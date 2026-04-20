"""Shared types for GB10 node management.

Adapted from auto-sre's infra.types for meeting-scribe's
multi-container model stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class NodeRole(Enum):
    """Role of a GB10 node in the cluster."""

    HEAD = "head"
    WORKER = "worker"


@dataclass
class GB10Node:
    """A Dell Pro Max GB10 (NVIDIA Grace Blackwell) node.

    Attributes:
        hostname: Node hostname (e.g., "gb10-1").
        ip: Node IP address (e.g., "192.168.1.101").
        ssh_user: SSH username for remote operations.
        ssh_key: Path to SSH private key (None = default ~/.ssh/id_ed25519).
        role: Node role in the cluster (head or worker).
    """

    hostname: str
    ip: str
    ssh_user: str = "root"
    ssh_key: str | None = None
    role: NodeRole = NodeRole.WORKER

    def to_dict(self) -> dict[str, str]:
        """Serialize to a dict suitable for YAML."""
        d: dict[str, str] = {
            "hostname": self.hostname,
            "ip": self.ip,
            "ssh_user": self.ssh_user,
            "role": self.role.value,
        }
        if self.ssh_key:
            d["ssh_key"] = self.ssh_key
        return d

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> GB10Node:
        """Deserialize from a dict (e.g., from YAML)."""
        return cls(
            hostname=str(data["hostname"]),
            ip=str(data["ip"]),
            ssh_user=str(data.get("ssh_user", "root")),
            ssh_key=str(data["ssh_key"]) if data.get("ssh_key") else None,
            role=NodeRole(str(data.get("role", "worker"))),
        )

    @property
    def ssh_target(self) -> str:
        """SSH connection string (user@ip)."""
        return f"{self.ssh_user}@{self.ip}"


@dataclass
class GB10Config:
    """Configuration for meeting-scribe's GB10 deployment.

    Persisted at ~/.local/share/meeting-scribe/gb10.yaml.
    """

    nodes: list[GB10Node] = field(default_factory=list)
    docker_image: str = "ghcr.io/bjk110/vllm-spark:turboquant"
    docker_image_turboquant: str = "ghcr.io/bjk110/vllm-spark:turboquant"
    hf_cache_dir: str = "/data/huggingface"
    nccl_socket_ifname: str = "enp1s0f0np0"

    @property
    def head_node(self) -> GB10Node:
        """The head node (first HEAD role, or first node)."""
        for node in self.nodes:
            if node.role == NodeRole.HEAD:
                return node
        if self.nodes:
            return self.nodes[0]
        msg = "No nodes configured"
        raise ValueError(msg)

    @property
    def worker_nodes(self) -> list[GB10Node]:
        """All non-head worker nodes."""
        head = self.head_node
        return [n for n in self.nodes if n is not head]

    @property
    def is_cluster(self) -> bool:
        """True if more than one node is configured (TP=2)."""
        return len(self.nodes) > 1

    def to_dict(self) -> dict:
        """Serialize for YAML persistence."""
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "docker_image": self.docker_image,
            "docker_image_turboquant": self.docker_image_turboquant,
            "hf_cache_dir": self.hf_cache_dir,
            "nccl_socket_ifname": self.nccl_socket_ifname,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GB10Config:
        """Deserialize from YAML dict."""
        nodes = [GB10Node.from_dict(n) for n in data.get("nodes", [])]
        return cls(
            nodes=nodes,
            docker_image=data.get("docker_image", cls.docker_image),
            docker_image_turboquant=data.get(
                "docker_image_turboquant", cls.docker_image_turboquant
            ),
            hf_cache_dir=data.get("hf_cache_dir", cls.hf_cache_dir),
            nccl_socket_ifname=data.get("nccl_socket_ifname", cls.nccl_socket_ifname),
        )
