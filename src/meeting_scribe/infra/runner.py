"""Runner factory — pick LocalRunner or SSHRunner based on target host.

GB10 is often the local machine. SSH'ing to localhost is both pointless and
fragile (requires sshd, root login, key trust). The factory returns a
LocalRunner whenever the target resolves to this machine.
"""

from __future__ import annotations

import os
import socket

from meeting_scribe.infra.local import LocalRunner
from meeting_scribe.infra.ssh import SSHRunner
from meeting_scribe.infra.types import GB10Node, NodeRole

Runner = LocalRunner | SSHRunner

LOCAL_HOSTS = {"", "local", "localhost", "127.0.0.1", "::1", "0.0.0.0"}


def is_local(host: str | None) -> bool:
    """True if `host` refers to this machine."""
    if os.environ.get("SCRIBE_GB10_LOCAL") == "1":
        return True
    if host is None:
        return True
    h = host.strip().lower()
    if h in LOCAL_HOSTS:
        return True
    try:
        if h == socket.gethostname().lower():
            return True
        if h == socket.getfqdn().lower():
            return True
    except OSError:
        pass
    return False


def get_runner(host: str | None = None) -> Runner:
    """Return a LocalRunner for local targets, SSHRunner otherwise."""
    if is_local(host):
        return LocalRunner()
    node = GB10Node(hostname="gb10", ip=host or "localhost", role=NodeRole.HEAD)
    return SSHRunner(node)
