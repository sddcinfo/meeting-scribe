"""tmux argv builder + server-wide config writer + list-sessions wrapper.

The tmux server is pinned to the ``scribe`` socket via ``-L scribe`` so
demo sessions never collide with the user's personal tmux. Server-wide
options (history-limit, true-colour overrides, mouse, status style) come
from a generated config file that tmux only reads at server start — so
per-attach argv never mutates shared state.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)


TMUX_SOCKET_NAME_DEFAULT: Final[str] = "scribe"


def socket_name() -> str:
    """Socket name the meeting-scribe terminal uses.

    ``SCRIBE_TMUX_SOCKET`` env var wins. Defaults to ``scribe``. Tests that
    don't want to pollute the user's live terminal session set their own
    value — see ``tests/browser/conftest.py``.
    """
    return os.environ.get("SCRIBE_TMUX_SOCKET", TMUX_SOCKET_NAME_DEFAULT)


# Back-compat alias for call sites that read the module constant directly.
TMUX_SOCKET_NAME = TMUX_SOCKET_NAME_DEFAULT

# Bumped whenever the config contents change — write_tmux_config() is a
# no-op unless the file on disk has a different marker.
_CONFIG_VERSION: Final[int] = 2

_CONFIG_BODY: Final[str] = """\
# meeting-scribe tmux config — applied only on tmux server start via -f.
# DO NOT EDIT BY HAND: this file is rewritten by tmux_helper.write_tmux_config().
set -g history-limit 10000
set -g default-terminal "tmux-256color"
set -ga terminal-overrides ",xterm-256color:Tc,tmux-256color:Tc,*256col*:Tc"
set -g mouse on
set -g escape-time 10
set -g status-style "bg=#0d1117,fg=#e6edf3"
# Layer the user's personal tmux config on top so their customizations
# (prefix bindings like C-a, plugins, window-name aliases) apply when
# they open the embedded terminal. -q = silent if file absent.
source-file -q ~/.tmux.conf
source-file -q ~/.config/tmux/tmux.conf
"""


def config_path() -> Path:
    """Where scribe.tmux.conf lives.

    ``SCRIBE_TMUX_CONF`` env override wins. Otherwise
    ``$XDG_CONFIG_HOME/meeting-scribe/scribe.tmux.conf`` or
    ``~/.config/meeting-scribe/scribe.tmux.conf``.
    """
    override = os.environ.get("SCRIBE_TMUX_CONF")
    if override:
        return Path(override)
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "meeting-scribe" / "scribe.tmux.conf"


def _marker() -> str:
    """Stable content marker; lets us detect drift across version bumps."""
    body_hash = hashlib.sha256(_CONFIG_BODY.encode()).hexdigest()[:16]
    return f"# scribe-tmux-config v{_CONFIG_VERSION} {body_hash}\n"


def write_tmux_config(path: Path | None = None) -> Path:
    """Idempotently write the tmux config file, returning its path."""
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    body = _marker() + _CONFIG_BODY
    try:
        existing = target.read_text()
        if existing == body:
            return target
    except FileNotFoundError:
        pass
    # Per-PID tmp suffix so concurrent imports (pytest-xdist) don't race
    # on the same path. Same pattern as AdminSecretStore.load_or_create.
    tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(body)
    try:
        os.replace(tmp, target)
    except FileNotFoundError:
        # A racing worker beat us to the rename. Drop our tmp and accept
        # whatever they wrote — the body is deterministic so the value at
        # `target` is interchangeable with what we would have written.
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        if not target.exists():
            raise
    logger.info("wrote %s (tmux config v%d)", target, _CONFIG_VERSION)
    return target


def build_argv(
    tmux_session: str,
    *,
    config: Path | None = None,
    socket: str | None = None,
    shell_override: str | None = None,
) -> list[str]:
    """Return the argv to exec inside the PTY.

    Production path: ``tmux -L <socket> -f <config> new-session -A -s <name>``.

    Test path: if ``shell_override`` is provided (explicitly or via
    ``SCRIBE_TERM_SHELL`` env var), returns ``[shell]`` — lets the test
    suite exercise the PTY path without requiring tmux.

    The socket name comes from ``SCRIBE_TMUX_SOCKET`` env var or the
    ``socket`` kwarg; defaults to ``scribe``. Tests set this to keep
    their sessions off the user's live terminal.
    """
    override = shell_override or os.environ.get("SCRIBE_TERM_SHELL")
    if override:
        return [override]
    cfg = config or config_path()
    sock = socket or socket_name()
    # NOTE: we deliberately do NOT use `-D` here. `-D` detaches every
    # other client attached to the session on each new attach, which
    # makes multi-tab demos (main view + popout + admin) fight — the
    # most-recent tab kicks all others. We instead clean up orphan
    # clients once, on server startup, via
    # :func:`detach_orphan_clients`. Legitimate multi-client attach is
    # supported by tmux and by our architecture.
    return [
        "tmux",
        "-L",
        sock,
        "-f",
        str(cfg),
        "new-session",
        "-A",  # attach if session exists, else create
        "-s",
        tmux_session,
    ]


# ── list-sessions helper ──────────────────────────────────────────


@dataclass(frozen=True)
class TmuxSessionInfo:
    name: str
    attached: int
    windows: int
    created: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "attached": self.attached,
            "windows": self.windows,
            "created": self.created,
        }


async def list_sessions(
    *, socket: str | None = None, timeout_s: float = 2.0
) -> list[TmuxSessionInfo]:
    socket_n = socket or socket_name()
    """Return the live tmux sessions on the given socket.

    "No server" is a normal state (no one has spawned a session yet) and
    is returned as an empty list — only actual errors raise.
    """
    if shutil.which("tmux") is None:
        return []
    proc = await asyncio.create_subprocess_exec(
        "tmux",
        "-L",
        socket_n,
        "list-sessions",
        "-F",
        "#{session_name}|#{session_attached}|#{session_windows}|#{session_created}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        logger.warning("tmux list-sessions timed out on socket %r", socket_n)
        return []

    if proc.returncode != 0:
        # "no server running" is the canonical expected failure
        text = (stderr or b"").decode(errors="replace").strip()
        if "no server running" in text or "no sessions" in text:
            return []
        logger.debug("tmux list-sessions failed: %s", text)
        return []

    sessions: list[TmuxSessionInfo] = []
    for line in (stdout or b"").decode(errors="replace").splitlines():
        parts = line.split("|")
        if len(parts) != 4:
            continue
        name, attached, windows, created = parts
        try:
            sessions.append(
                TmuxSessionInfo(
                    name=name,
                    attached=int(attached),
                    windows=int(windows),
                    created=int(created),
                )
            )
        except ValueError:
            continue
    return sessions


async def kill_session(name: str, *, socket: str | None = None, timeout_s: float = 3.0) -> bool:
    """Kill a specific tmux session by name.

    Returns True if the session existed and was killed, False if it
    didn't exist or tmux isn't installed. The tmux SERVER survives;
    only the named session goes away.
    """
    if shutil.which("tmux") is None:
        return False
    sock = socket or socket_name()
    proc = await asyncio.create_subprocess_exec(
        "tmux",
        "-L",
        sock,
        "kill-session",
        "-t",
        name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return False
    if proc.returncode == 0:
        return True
    text = (stderr or b"").decode(errors="replace").strip()
    if "can't find session" in text or "no server running" in text:
        return False
    logger.warning("tmux kill-session on socket=%r name=%r failed: %s", sock, name, text)
    return False


async def detach_orphan_clients(*, socket: str | None = None, timeout_s: float = 3.0) -> int:
    """Detach every client currently attached to the scribe socket.

    Intended to be called ONCE on meeting-scribe startup. Any attached
    client at that moment is by definition from a previous process
    that died without cleaning up — meeting-scribe doesn't attach
    ambient clients. Detaching orphans prevents the "blank pane"
    failure mode where tmux defers output to a dead client.

    The session itself is preserved. Returns the number of clients
    that were asked to detach (best-effort, tmux may have already
    cleaned some up by the time the second command runs).
    """
    if shutil.which("tmux") is None:
        return 0
    sock = socket or socket_name()
    # First: enumerate attached clients. `list-clients` with no args
    # lists every client on the server.
    list_proc = await asyncio.create_subprocess_exec(
        "tmux",
        "-L",
        sock,
        "list-clients",
        "-F",
        "#{client_name}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, _stderr = await asyncio.wait_for(list_proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        list_proc.kill()
        await list_proc.wait()
        return 0
    if list_proc.returncode != 0:
        # "no server running" is fine — nothing to detach.
        return 0
    client_names = [
        line.strip()
        for line in (stdout or b"").decode(errors="replace").splitlines()
        if line.strip()
    ]
    if not client_names:
        return 0
    # Detach each client by its tmux-assigned name (typically the tty).
    for cname in client_names:
        proc = await asyncio.create_subprocess_exec(
            "tmux",
            "-L",
            sock,
            "detach-client",
            "-t",
            cname,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except TimeoutError:
            proc.kill()
            await proc.wait()
    logger.info(
        "detached %d orphan tmux clients on socket=%r at startup",
        len(client_names),
        sock,
    )
    return len(client_names)


async def kill_server(*, socket: str | None = None) -> bool:
    """Explicit teardown of the tmux server. Intended for admin CLI only.

    meeting-scribe NEVER calls this from its lifecycle — that would
    defeat session persistence. Returns True if the server was running
    and was killed, False if there was no server to kill.
    """
    if shutil.which("tmux") is None:
        return False
    sock = socket or socket_name()
    proc = await asyncio.create_subprocess_exec(
        "tmux",
        "-L",
        sock,
        "kill-server",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode == 0:
        return True
    text = (stderr or b"").decode(errors="replace").strip()
    if "no server running" in text:
        return False
    logger.warning("tmux kill-server on socket %r failed: %s", sock, text)
    return False
