"""Tests for TerminalHistoryLog — the per-meeting terminal output log.

Covers: basic append + read, parent-directory auto-creation, byte cap
enforcement, atomic rotation, idempotent open/close, and single-write
larger than cap (the pathological "pasted 10 MB" case).
"""

from __future__ import annotations

from pathlib import Path

from meeting_scribe.terminal.history import (
    DEFAULT_CAP_BYTES,
    REPLAY_BYTES,
    TerminalHistoryLog,
)


def test_append_and_read(tmp_path: Path) -> None:
    log = TerminalHistoryLog(tmp_path / "terminal.log", cap_bytes=4096)
    log.open()
    log.write(b"hello ")
    log.write(b"world\n")
    log.close()
    assert log.read_tail() == b"hello world\n"


def test_parent_directory_auto_created(tmp_path: Path) -> None:
    # Parent dir 'meetings/abc/' does NOT exist yet.
    path = tmp_path / "meetings" / "abc" / "terminal.log"
    log = TerminalHistoryLog(path)
    log.open()
    log.write(b"first meeting bytes")
    log.close()
    assert path.exists()
    assert path.parent.is_dir()
    assert log.read_tail() == b"first meeting bytes"


def test_read_tail_returns_only_last_n_bytes(tmp_path: Path) -> None:
    log = TerminalHistoryLog(tmp_path / "x.log", cap_bytes=4096)
    log.open()
    log.write(b"ABCDEFGHIJ")
    log.write(b"1234567890")
    log.close()
    assert log.read_tail(5) == b"67890"
    assert log.read_tail(100) == b"ABCDEFGHIJ1234567890"


def test_read_tail_before_any_open(tmp_path: Path) -> None:
    # Resolver returns a path but we haven't opened yet — reading must
    # return an empty bytes object, not raise.
    log = TerminalHistoryLog(tmp_path / "never-opened.log")
    assert log.read_tail() == b""


def test_cap_triggers_rotation(tmp_path: Path) -> None:
    cap = 256
    log = TerminalHistoryLog(tmp_path / "rotate.log", cap_bytes=cap)
    log.open()
    # Write 6x the cap in 64-byte chunks.
    for i in range(24):
        log.write(bytes([ord("A") + (i % 26)]) * 64)
    log.close()
    size = (tmp_path / "rotate.log").stat().st_size
    # After rotation, size is bounded by cap_bytes + the marker line.
    assert size <= cap + 80, f"file ballooned to {size} bytes (cap={cap})"
    tail = log.read_tail(64)
    assert len(tail) == 64


def test_single_write_larger_than_cap(tmp_path: Path) -> None:
    cap = 1024
    log = TerminalHistoryLog(tmp_path / "big.log", cap_bytes=cap)
    log.open()
    # One write of 10x the cap — should land us with only the tail.
    huge = b"Z" * (cap * 10)
    log.write(huge)
    log.close()
    size = (tmp_path / "big.log").stat().st_size
    assert size <= cap, f"single oversize write bypassed cap: {size}"


def test_close_is_idempotent(tmp_path: Path) -> None:
    log = TerminalHistoryLog(tmp_path / "idem.log")
    log.open()
    log.write(b"once")
    log.close()
    log.close()  # second close is a no-op


def test_write_after_close_is_noop(tmp_path: Path) -> None:
    log = TerminalHistoryLog(tmp_path / "after.log")
    log.open()
    log.write(b"alive")
    log.close()
    log.write(b"ghost")  # must not raise or resurrect the fd
    assert log.read_tail() == b"alive"


def test_rotation_preserves_most_recent_bytes(tmp_path: Path) -> None:
    """After a cap event, the tail of the log must contain the newest
    bytes we wrote — not something stale from before the rotation.
    """
    cap = 128
    log = TerminalHistoryLog(tmp_path / "tail.log", cap_bytes=cap)
    log.open()
    for i in range(50):
        log.write(f"line-{i:04d}\n".encode())
    log.close()
    tail = log.read_tail(200)
    # The very last line must be present.
    assert b"line-0049" in tail
    # An early line must have been pruned.
    assert b"line-0000" not in tail


def test_constants_are_sensible() -> None:
    # Cap is large enough for a meeting's terminal output but small
    # enough to avoid ballooning disk; replay is a subset.
    assert DEFAULT_CAP_BYTES >= 64 * 1024
    assert REPLAY_BYTES <= DEFAULT_CAP_BYTES
