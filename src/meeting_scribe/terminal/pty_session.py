"""PTY-backed terminal session with WS-friendly flow control.

One :class:`TerminalSession` per WebSocket. The session owns a forked
child on a pty pair — typically ``tmux new-session -A`` — and brokers
bytes between that child and the WS with bounded buffers in both
directions.

Design decisions worth re-reading before editing:

* **Controlling TTY is explicit.** We set up the child with
  ``setsid()`` + ``ioctl(TIOCSCTTY)`` so Ctrl+C / Ctrl+Z / foreground
  process groups all behave correctly.
* **Env is allowlisted.** Anything not in :data:`ENV_ALLOWLIST` (plus
  ``LC_*``) is dropped — we never pass tokens or credentials into a
  shell the user can script.
* **Flow control is based on total pending output** (``out_buffer`` +
  ``unacked_bytes``), using monotonic cumulative ACK counters so a
  malicious client can't drive the bookkeeping negative.
* **Stdin is byte-accounted**, not item-counted — a single huge paste
  frame can't bypass the budget.
* **Close drops queued stdin.** If the user has disconnected we do
  *not* flush half-typed commands into the shell.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import fcntl
import logging
import os
import struct
import subprocess
import termios
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from meeting_scribe.terminal.protocol import (
    HIGH_WATER,
    INBOUND_FRAME_MAX,
    LOW_WATER,
    MAX_OUT_FRAME,
    OUT_BUFFER_HARD_CAP,
    PREFIX_OUTPUT,
    STDIN_BUDGET,
)

if TYPE_CHECKING:
    from fastapi import WebSocket

logger = logging.getLogger(__name__)


ENV_ALLOWLIST: Final[frozenset[str]] = frozenset(
    {
        "HOME",
        "PATH",
        "LANG",
        "TZ",
        "USER",
        "SHELL",
        "XDG_RUNTIME_DIR",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
    }
)

SEND_LOCK_TIMEOUT_S: Final[float] = 10.0
READ_CHUNK: Final[int] = 64 * 1024
CLOSE_WAIT_S: Final[float] = 2.0


async def _poll_exit(proc: subprocess.Popen[bytes], timeout_s: float) -> bool:
    """Wait up to *timeout_s* for *proc* to exit, polling with asyncio.sleep.

    Returns True if the process exited within the timeout. We avoid
    ``asyncio.to_thread(proc.wait)`` here because the default thread
    executor can stall during Starlette TestClient teardown, and any
    blocking wait on a cancelled coroutine would prevent the caller's
    registry-release from running.
    """
    deadline = time.monotonic() + timeout_s
    while proc.poll() is None:
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0.02)
    return True


def _default_cwd() -> str:
    """Where the shell lands.

    ``SCRIBE_TERM_CWD`` env var wins. Otherwise prefer ``~/sddcinfo`` so
    ``autosre`` / ``sddc`` commands Just Work on first keystroke; fall
    back to ``$HOME`` if that directory doesn't exist.
    """
    override = os.environ.get("SCRIBE_TERM_CWD")
    if override:
        p = os.path.expanduser(override)
        if os.path.isdir(p):
            return p
    home = os.path.expanduser("~")
    preferred = os.path.join(home, "sddcinfo")
    return preferred if os.path.isdir(preferred) else home


def _child_preexec() -> None:
    """Finish making the PTY slave the child's controlling TTY.

    Runs after fork, before exec. We rely on ``start_new_session=True`` on
    the Popen call to do ``setsid()``; this preexec only does the
    ``TIOCSCTTY`` binding, and only when the caller has opted in via
    :attr:`TerminalSession.claim_ctty`. tmux does its own session/TTY
    management internally, so claiming the TTY up front interferes with
    its daemon fork; for the tmux path we leave ``claim_ctty`` False.
    Direct shells (the test path) need ``claim_ctty=True`` for Ctrl+C
    semantics to work.
    """
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)


def _build_env(cols: int, rows: int, term: str = "xterm-256color") -> dict[str, str]:
    """Allowlist-driven environment for the spawned child."""
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in ENV_ALLOWLIST or key.startswith("LC_"):
            env[key] = value
    env["TERM"] = term
    env["COLORTERM"] = "truecolor"
    env["COLUMNS"] = str(cols)
    env["LINES"] = str(rows)
    env["SCRIBE_TERM"] = "1"
    # Tmux nesting protection: never leak outer TMUX* vars into the inner shell.
    for leaked in ("TMUX", "TMUX_PANE", "TMUX_TMPDIR"):
        env.pop(leaked, None)
    return env


@dataclass
class TerminalSession:
    """Lifecycle + I/O for one PTY, one tmux client, one WebSocket."""

    tmux_session: str
    master_fd: int
    proc: subprocess.Popen[bytes]
    ws: WebSocket
    cols: int
    rows: int
    created_at: float = field(default_factory=time.monotonic)

    # Outbound pipeline (PTY → WS)
    out_buffer: bytearray = field(default_factory=bytearray)
    bytes_sent_total: int = 0  # monotonic cumulative, O-frame bytes sent
    bytes_acked_total: int = 0  # monotonic cumulative, last validated client ack
    reader_attached: bool = False
    paused: bool = False
    _flush_pending: bool = False
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Inbound pipeline (WS → PTY)
    _stdin_deque: collections.deque[bytes] = field(default_factory=collections.deque)
    _stdin_cond: asyncio.Condition = field(default_factory=asyncio.Condition)
    pending_input_bytes: int = 0
    bytes_in: int = 0
    writer_task: asyncio.Task[None] | None = None

    # Lifecycle
    _closing: bool = False
    _close_reason: str = ""

    # Per-meeting history sink (set after spawn by the router).
    history: object = None  # type: ignore[assignment]  # TerminalHistoryLog | None

    # ── Classmethod: spawn ────────────────────────────────────────

    @classmethod
    async def spawn(
        cls,
        *,
        tmux_session: str,
        argv: list[str],
        cols: int,
        rows: int,
        ws: WebSocket,
        term: str = "xterm-256color",
        cwd: str | None = None,
        claim_ctty: bool | None = None,
    ) -> TerminalSession:
        """Spawn a child on a PTY pair.

        ``claim_ctty`` — if True, preexec_fn runs ``TIOCSCTTY`` to bind
        the slave as the controlling TTY. Required for direct shells so
        that Ctrl+C/Ctrl+Z deliver SIGINT/SIGTSTP to the foreground
        process group. For tmux (``argv[0] == "tmux"``) we default to
        False because tmux does its own session/TTY management
        internally; claiming the TTY up front interferes with its
        daemon fork. Auto-detected when ``None``.
        """
        master, slave = os.openpty()
        # Non-blocking master — we drive it from the event loop.
        flags = fcntl.fcntl(master, fcntl.F_GETFL)
        fcntl.fcntl(master, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        if claim_ctty is None:
            claim_ctty = not (argv and argv[0].endswith("tmux"))

        env = _build_env(cols, rows, term=term)
        proc = subprocess.Popen(
            argv,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            start_new_session=True,
            preexec_fn=_child_preexec if claim_ctty else None,
            close_fds=True,
            env=env,
            cwd=cwd or _default_cwd(),
        )
        os.close(slave)

        # Initial window size.
        fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

        session = cls(
            tmux_session=tmux_session,
            master_fd=master,
            proc=proc,
            ws=ws,
            cols=cols,
            rows=rows,
        )
        loop = asyncio.get_running_loop()
        loop.add_reader(master, session._on_master_readable)
        session.reader_attached = True
        session.writer_task = asyncio.create_task(session._stdin_writer_loop())
        logger.info(
            "terminal session spawned: tmux=%s pid=%d cols=%d rows=%d",
            tmux_session,
            proc.pid,
            cols,
            rows,
        )
        return session

    # ── Properties ────────────────────────────────────────────────

    @property
    def unacked_bytes(self) -> int:
        return self.bytes_sent_total - self.bytes_acked_total

    @property
    def pending_out(self) -> int:
        return len(self.out_buffer) + self.unacked_bytes

    # ── Outbound: PTY → WS ────────────────────────────────────────

    def _on_master_readable(self) -> None:
        """add_reader callback. Runs on the event-loop thread, no await."""
        if self._closing:
            return
        # If we'd immediately overflow, stop reading and wait for ACKs.
        if self.pending_out >= HIGH_WATER:
            self._pause_reader()
            return
        try:
            chunk = os.read(self.master_fd, READ_CHUNK)
        except BlockingIOError, InterruptedError:
            return
        except OSError as e:
            logger.debug("pty read error (%s): %s — closing", self.tmux_session, e)
            self._schedule_close(reason="read_error")
            return
        if not chunk:
            # EOF = child exited.
            self._schedule_close(reason="pty_exited")
            return
        # Tee to the history log BEFORE any flow-control decisions so the
        # on-disk record matches what the PTY actually produced, not just
        # what made it to the wire.
        if self.history is not None:
            try:
                self.history.write(chunk)
            except Exception:
                pass
        # Hard overrun cap: the client isn't ACKing and we're about to OOM.
        if len(self.out_buffer) + len(chunk) > OUT_BUFFER_HARD_CAP:
            logger.warning(
                "terminal %s overrun: buffer=%d+%d cap=%d — closing",
                self.tmux_session,
                len(self.out_buffer),
                len(chunk),
                OUT_BUFFER_HARD_CAP,
            )
            self._schedule_close(reason="overrun")
            return
        self.out_buffer.extend(chunk)
        if not self._flush_pending:
            self._flush_pending = True
            asyncio.create_task(self._flush_out())

    async def _flush_out(self) -> None:
        try:
            async with self.send_lock:
                while self.out_buffer and not self._closing and self.ws is not None:
                    frame_size = min(len(self.out_buffer), MAX_OUT_FRAME)
                    frame = bytes(self.out_buffer[:frame_size])
                    del self.out_buffer[:frame_size]
                    payload = PREFIX_OUTPUT + frame
                    try:
                        await asyncio.wait_for(
                            self.ws.send_bytes(payload), timeout=SEND_LOCK_TIMEOUT_S
                        )
                    except TimeoutError:
                        logger.warning(
                            "terminal %s: WS send timeout — closing",
                            self.tmux_session,
                        )
                        self._schedule_close(reason="send_timeout")
                        return
                    except Exception as e:
                        logger.debug(
                            "terminal %s: WS send failed (%s) — closing",
                            self.tmux_session,
                            e,
                        )
                        self._schedule_close(reason="send_failed")
                        return
                    self.bytes_sent_total += frame_size
                    if self.pending_out >= HIGH_WATER:
                        self._pause_reader()
        finally:
            self._flush_pending = False

    def on_client_ack(self, cumulative: int) -> None:
        """Apply a monotonic cumulative ACK from the client.

        Non-monotonic values are silently ignored; values that exceed
        bytes_sent_total (malicious or out-of-sync client) are clamped.
        """
        if cumulative < self.bytes_acked_total:
            return  # stale / reordered — ignore
        if cumulative > self.bytes_sent_total:
            logger.warning(
                "terminal %s: ack_total=%d exceeds sent=%d; clamping",
                self.tmux_session,
                cumulative,
                self.bytes_sent_total,
            )
            cumulative = self.bytes_sent_total
        self.bytes_acked_total = cumulative
        if self.paused and self.pending_out < LOW_WATER:
            self._resume_reader()

    def _pause_reader(self) -> None:
        if not self.reader_attached or self._closing:
            return
        loop = asyncio.get_running_loop()
        loop.remove_reader(self.master_fd)
        self.reader_attached = False
        self.paused = True

    def _resume_reader(self) -> None:
        if self.reader_attached or self._closing:
            return
        loop = asyncio.get_running_loop()
        loop.add_reader(self.master_fd, self._on_master_readable)
        self.reader_attached = True
        self.paused = False

    # ── Inbound: WS → PTY ────────────────────────────────────────

    async def write_stdin(self, data: bytes) -> None:
        """Enqueue a chunk for the PTY. Applies byte-level backpressure."""
        if len(data) > INBOUND_FRAME_MAX:
            raise ValueError(f"inbound frame too large: {len(data)} > {INBOUND_FRAME_MAX}")
        async with self._stdin_cond:
            while self.pending_input_bytes + len(data) > STDIN_BUDGET and not self._closing:
                await self._stdin_cond.wait()
            if self._closing:
                return
            self._stdin_deque.append(data)
            self.pending_input_bytes += len(data)
            self._stdin_cond.notify_all()

    async def _stdin_writer_loop(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while not self._closing:
                async with self._stdin_cond:
                    while not self._stdin_deque and not self._closing:
                        await self._stdin_cond.wait()
                    if self._closing:
                        return
                    data = self._stdin_deque.popleft()
                    self.pending_input_bytes -= len(data)
                    self._stdin_cond.notify_all()
                # Drain outside the condition so other producers can enqueue.
                while data and not self._closing:
                    try:
                        n = os.write(self.master_fd, data)
                    except BlockingIOError:
                        # Kernel write buffer is full; wait for writable.
                        fut: asyncio.Future[None] = loop.create_future()

                        def _cb() -> None:
                            if not fut.done():
                                fut.set_result(None)

                        loop.add_writer(self.master_fd, _cb)
                        try:
                            await fut
                        finally:
                            loop.remove_writer(self.master_fd)
                        continue
                    except (OSError, BrokenPipeError) as e:
                        logger.debug(
                            "terminal %s: stdin write failed (%s) — draining",
                            self.tmux_session,
                            e,
                        )
                        return
                    data = data[n:]
                    self.bytes_in += n
        except asyncio.CancelledError:
            raise

    # ── Resize ────────────────────────────────────────────────────

    async def resize(self, cols: int, rows: int) -> None:
        self.cols, self.rows = cols, rows
        try:
            fcntl.ioctl(
                self.master_fd,
                termios.TIOCSWINSZ,
                struct.pack("HHHH", rows, cols, 0, 0),
            )
        except OSError as e:
            logger.debug("resize ioctl failed (%s): %s", self.tmux_session, e)

    # ── Close ─────────────────────────────────────────────────────

    def _schedule_close(self, *, reason: str) -> None:
        """Fire close() from a sync context (add_reader callback)."""
        if self._closing:
            return
        asyncio.create_task(self.close(reason=reason))

    async def close(self, *, reason: str) -> None:
        """Tear down the PTY and child. Idempotent, cancellation-tolerant."""
        if self._closing:
            return
        self._closing = True
        self._close_reason = reason
        self._pause_reader()

        # Drop anything queued — do NOT execute half-typed commands after the
        # user is gone.
        async with self._stdin_cond:
            self._stdin_deque.clear()
            self.pending_input_bytes = 0
            self._stdin_cond.notify_all()

        if self.writer_task is not None:
            self.writer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.writer_task

        # os.close(master) triggers SIGHUP on the child via the PTY driver.
        # For tmux clients this prompts a clean detach — we give the client a
        # generous grace period to detach so the tmux server doesn't see a
        # SIGKILL'd client and treat the session state ambiguously.
        with contextlib.suppress(OSError):
            os.close(self.master_fd)
        with contextlib.suppress(ProcessLookupError):
            self.proc.terminate()
        if not await _poll_exit(self.proc, CLOSE_WAIT_S):
            with contextlib.suppress(ProcessLookupError):
                self.proc.kill()
            await _poll_exit(self.proc, CLOSE_WAIT_S)
        if self.history is not None:
            try:
                self.history.close()
            except Exception:
                pass
        logger.info(
            "terminal session closed: tmux=%s reason=%s bytes_in=%d sent=%d acked=%d",
            self.tmux_session,
            reason,
            self.bytes_in,
            self.bytes_sent_total,
            self.bytes_acked_total,
        )

    # ── Introspection ─────────────────────────────────────────────

    def summary(self) -> dict[str, object]:
        return {
            "tmux_session": self.tmux_session,
            "pid": self.proc.pid,
            "cols": self.cols,
            "rows": self.rows,
            "bytes_in": self.bytes_in,
            "bytes_sent_total": self.bytes_sent_total,
            "bytes_acked_total": self.bytes_acked_total,
            "paused": self.paused,
            "pending_out": self.pending_out,
            "created_at": self.created_at,
        }
