"""Tests for the PTY-backed terminal session.

Runs against ``/bin/sh`` so no tmux is required. Exercises the
controlling-TTY setup, flow control, ACK bookkeeping, resize, and
close semantics.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from meeting_scribe.terminal import pty_session as ps
from meeting_scribe.terminal.pty_session import TerminalSession

# ── Fake WS that records what the session sent ────────────────────


@dataclass
class FakeWS:
    """Minimal stand-in for Starlette/FastAPI WebSocket — capture-only."""

    sent_binary: list[bytes] = field(default_factory=list)
    sent_text: list[str] = field(default_factory=list)
    send_delay: float = 0.0  # delay applied to each send_bytes call
    closed: bool = False

    async def send_bytes(self, data: bytes) -> None:
        if self.closed:
            raise RuntimeError("WS closed")
        if self.send_delay:
            await asyncio.sleep(self.send_delay)
        self.sent_binary.append(data)

    async def send_text(self, data: str) -> None:
        if self.closed:
            raise RuntimeError("WS closed")
        self.sent_text.append(data)


def _stdout_bytes(ws: FakeWS) -> bytes:
    """Concatenate all PTY output frames (stripping 1-byte O prefix)."""
    return b"".join(frame[1:] for frame in ws.sent_binary)


async def _wait_for(predicate, *, timeout: float = 3.0, poll: float = 0.02) -> None:
    """Poll ``predicate()`` until it returns truthy or timeout elapses."""
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        if predicate():
            return
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timed out waiting for {predicate!r}")
        await asyncio.sleep(poll)


async def _wait_output_contains(ws: FakeWS, needle: bytes, *, timeout: float = 3.0) -> None:
    await _wait_for(lambda: needle in _stdout_bytes(ws), timeout=timeout)


# ── Fixtures ──────────────────────────────────────────────────────


async def _spawn_sh(
    *, cols: int = 80, rows: int = 24, ws_delay: float = 0.0
) -> tuple[TerminalSession, FakeWS]:
    ws = FakeWS(send_delay=ws_delay)
    session = await TerminalSession.spawn(
        tmux_session="test",
        argv=["/bin/sh"],
        cols=cols,
        rows=rows,
        ws=ws,  # type: ignore[arg-type]
    )
    return session, ws


async def _spawn_custom(
    argv: list[str], *, cols: int = 80, rows: int = 24
) -> tuple[TerminalSession, FakeWS]:
    ws = FakeWS()
    session = await TerminalSession.spawn(
        tmux_session="test",
        argv=argv,
        cols=cols,
        rows=rows,
        ws=ws,  # type: ignore[arg-type]
    )
    return session, ws


# ── Happy path ────────────────────────────────────────────────────


async def test_spawn_and_echo():
    session, ws = await _spawn_sh()
    try:
        await session.write_stdin(b"echo hello-world\n")
        await _wait_output_contains(ws, b"hello-world")
    finally:
        await session.close(reason="test")


async def test_bytes_in_counted():
    session, _ws = await _spawn_sh()
    try:
        await session.write_stdin(b"true\n")
        # Wait for writer to drain
        await _wait_for(lambda: session.bytes_in >= len(b"true\n"))
        assert session.bytes_in == 5
    finally:
        await session.close(reason="test")


async def test_controlling_tty_ctrl_c_kills_foreground():
    # Running sleep directly — Ctrl+C from the PTY delivers SIGINT to its
    # foreground process group. If the controlling-TTY setup is wrong,
    # sleep never receives the signal.
    session, _ws = await _spawn_custom(["/bin/sleep", "30"])
    try:
        # Give sleep a moment to start.
        await asyncio.sleep(0.1)
        assert session.proc.poll() is None, "sleep exited too early"
        # Send Ctrl+C
        await session.write_stdin(b"\x03")
        # It should die well under a second
        returncode = await asyncio.wait_for(asyncio.to_thread(session.proc.wait), timeout=2.0)
        # Killed by SIGINT — negative returncode under Popen.
        assert returncode < 0, f"expected signal death, got returncode={returncode}"
    finally:
        await session.close(reason="test")


async def test_resize_propagates_to_child():
    session, ws = await _spawn_sh(cols=80, rows=24)
    try:
        await session.resize(cols=132, rows=43)
        await session.write_stdin(b"stty size\n")
        await _wait_output_contains(ws, b"43 132")
        assert session.cols == 132 and session.rows == 43
    finally:
        await session.close(reason="test")


# ── Flow control ──────────────────────────────────────────────────


async def test_flow_control_pauses_and_resumes():
    session, _ws = await _spawn_sh()
    try:
        # Produce a big burst of output — 300 KiB > HIGH_WATER (128 KiB).
        await session.write_stdin(b"yes x 2>/dev/null | head -c 300000\n")
        # Wait for pause to engage (reader detached)
        await _wait_for(lambda: session.paused, timeout=3.0)
        assert session.unacked_bytes >= ps.LOW_WATER
        # ACK everything we've sent so far
        session.on_client_ack(session.bytes_sent_total)
        # Should resume reading
        await _wait_for(lambda: not session.paused, timeout=3.0)
        # Drain the remaining output — final ack catches up
        await asyncio.sleep(0.2)
        session.on_client_ack(session.bytes_sent_total)
    finally:
        await session.close(reason="test")


async def test_ack_monotonic_ignores_stale():
    session, _ws = await _spawn_sh()
    try:
        # Stale ack (below current) — ignored
        session.bytes_sent_total = 1000
        session.bytes_acked_total = 500
        session.on_client_ack(100)
        assert session.bytes_acked_total == 500
        # Forward ack updates
        session.on_client_ack(700)
        assert session.bytes_acked_total == 700
    finally:
        await session.close(reason="test")


async def test_ack_clamps_when_exceeds_sent(caplog):
    session, _ws = await _spawn_sh()
    try:
        session.bytes_sent_total = 1000
        session.bytes_acked_total = 0
        # Client claims to have rendered 9999 bytes when we only sent 1000.
        session.on_client_ack(9999)
        assert session.bytes_acked_total == 1000
    finally:
        await session.close(reason="test")


async def test_overrun_closes_session(monkeypatch):
    # Disable the normal pause watermark so we can actually hit overrun;
    # tighten OUT_BUFFER_HARD_CAP so the test runs quickly.
    monkeypatch.setattr(ps, "HIGH_WATER", 100 * 1024 * 1024)
    monkeypatch.setattr(ps, "OUT_BUFFER_HARD_CAP", 2048)
    # Slow WS so the out_buffer accumulates faster than it drains.
    ws = FakeWS(send_delay=0.5)
    session = await TerminalSession.spawn(
        tmux_session="test",
        argv=["/bin/sh", "-c", "yes 2>&1 | head -c 50000"],
        cols=80,
        rows=24,
        ws=ws,  # type: ignore[arg-type]
    )
    try:
        await _wait_for(lambda: session._closing, timeout=4.0)
        assert session._close_reason == "overrun"
    finally:
        await session.close(reason="cleanup")


# ── Inbound: stdin bounds ─────────────────────────────────────────


async def test_oversized_inbound_frame_rejected():
    session, _ws = await _spawn_sh()
    try:
        oversized = b"x" * (ps.INBOUND_FRAME_MAX + 1)
        with pytest.raises(ValueError, match="too large"):
            await session.write_stdin(oversized)
    finally:
        await session.close(reason="test")


async def test_bounded_stdin_backpressures_producer():
    """A producer that outpaces the PTY must block, not balloon memory."""
    # Use `cat > /dev/null` so the PTY consumes at a steady pace.
    session, _ws = await _spawn_custom(["/bin/sh", "-c", "cat > /dev/null"])
    max_pending_seen = 0
    try:
        # Sample pending_input_bytes while pushing 4x budget through.
        async def sampler():
            nonlocal max_pending_seen
            for _ in range(400):
                max_pending_seen = max(max_pending_seen, session.pending_input_bytes)
                await asyncio.sleep(0.001)

        async def pusher():
            chunk = b"x" * (ps.INBOUND_FRAME_MAX // 2)  # 128 KiB each
            for _ in range(32):  # 4 MiB total
                await session.write_stdin(chunk)

        sampler_task = asyncio.create_task(sampler())
        await pusher()
        await asyncio.sleep(0.1)
        # The sampler may already have finished its 400-iter loop on a
        # slow runner before the pusher completed (locally pusher beats
        # sampler; on the GitHub Ubuntu runner it's the other way around).
        # Either outcome is fine — what matters is that we get the peak
        # reading without leaving a dangling task.
        if not sampler_task.done():
            sampler_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sampler_task
        # Budget must not have been exceeded at any sample.
        assert max_pending_seen <= ps.STDIN_BUDGET, (
            f"pending_input_bytes peaked at {max_pending_seen}, "
            f"exceeding STDIN_BUDGET={ps.STDIN_BUDGET}"
        )
    finally:
        await session.close(reason="test")


async def test_close_drops_queued_stdin():
    """Bytes queued after close() must not reach the PTY."""
    session, ws = await _spawn_sh()
    # Fill up the inbound queue a bit.
    await session.write_stdin(b"echo pre-close\n")
    # Give writer time to consume it so it actually ran.
    await _wait_output_contains(ws, b"pre-close")
    bytes_before = session.bytes_in

    # Queue more, then close before the writer task drains.
    await session.write_stdin(b"echo post-close\n")
    await session.close(reason="test")

    # bytes_in may have increased slightly (the item popped from the
    # queue before close) — what matters is the deque is empty and no
    # new "post-close" bytes leak into the PTY after close finishes.
    assert len(session._stdin_deque) == 0
    assert session.pending_input_bytes == 0
    assert session._closing
    # Child process is already dead/reaped; no race with more writes.
    assert session.proc.poll() is not None


# ── Env scrub ────────────────────────────────────────────────────


async def test_env_allowlist_drops_secrets(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "super-secret-value-xyz")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "another-secret")
    monkeypatch.setenv("LC_ALL", "C")  # should survive
    monkeypatch.setenv("RANDOM_UNKNOWN", "should-not-leak")

    session, _ = await _spawn_sh()
    try:
        environ = Path(f"/proc/{session.proc.pid}/environ").read_bytes()
        entries = environ.decode(errors="replace").split("\x00")
        env: dict[str, str] = {}
        for entry in entries:
            if "=" in entry:
                k, _, v = entry.partition("=")
                env[k] = v
        # Secrets must be absent
        assert "GH_TOKEN" not in env
        assert "CLOUDFLARE_API_TOKEN" not in env
        assert "RANDOM_UNKNOWN" not in env
        # Standard vars survive
        assert env.get("TERM") == "xterm-256color"
        assert env.get("COLORTERM") == "truecolor"
        assert env.get("LC_ALL") == "C"
        assert env.get("SCRIBE_TERM") == "1"
    finally:
        await session.close(reason="test")


async def test_env_drops_outer_tmux_vars(monkeypatch):
    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1234,0")
    monkeypatch.setenv("TMUX_PANE", "%0")

    session, _ = await _spawn_sh()
    try:
        environ = Path(f"/proc/{session.proc.pid}/environ").read_bytes()
        entries = environ.decode(errors="replace").split("\x00")
        names = {e.split("=", 1)[0] for e in entries if "=" in e}
        assert "TMUX" not in names
        assert "TMUX_PANE" not in names
    finally:
        await session.close(reason="test")


# ── Close is idempotent ───────────────────────────────────────────


async def test_close_is_idempotent():
    session, _ = await _spawn_sh()
    await session.close(reason="first")
    # Second close must not raise or hang.
    await asyncio.wait_for(session.close(reason="second"), timeout=2.0)
    assert session._close_reason == "first"  # first reason wins


async def test_summary_shape():
    session, ws = await _spawn_sh()
    try:
        await session.write_stdin(b"echo SUMMARY-OK\n")
        await _wait_output_contains(ws, b"SUMMARY-OK")
        s = session.summary()
        assert s["tmux_session"] == "test"
        assert isinstance(s["pid"], int)
        assert s["cols"] == 80
        assert s["rows"] == 24
        assert isinstance(s["bytes_sent_total"], int)
        assert s["bytes_sent_total"] > 0
    finally:
        await session.close(reason="test")


# ── Session leader / process group sanity ────────────────────────


async def test_child_is_session_leader():
    """After setsid, the child's pid should equal its sid."""
    session, _ = await _spawn_sh()
    try:
        # /proc/<pid>/stat field 6 is sid. Format:
        #   pid (comm) state ppid pgrp sid ...
        stat = Path(f"/proc/{session.proc.pid}/stat").read_text()
        # 'comm' can contain spaces or parens; use regex to find the last ')'
        m = re.match(r"\d+\s+\(.*\)\s+\S+\s+\d+\s+\d+\s+(\d+)", stat)
        assert m, f"unexpected /proc/<pid>/stat format: {stat!r}"
        sid = int(m.group(1))
        assert sid == session.proc.pid, (
            f"child pid {session.proc.pid} is not session leader (sid={sid})"
        )
    finally:
        await session.close(reason="test")
