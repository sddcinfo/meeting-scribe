"""Rolling terminal output log, tied to the current meeting folder.

Every byte the PTY emits is tee'd into a per-meeting file so the user
can scroll back through prior shell activity after a page reload or
WebSocket drop. Works exactly the same way slides do: the file lives
inside ``data/meetings/<id>/terminal.log`` so it's part of the meeting
artifact bundle.

File is capped at ``cap_bytes`` via a simple halving truncate: when the
file exceeds the cap we drop the first half of the content, write a
marker line, and continue appending. This costs one bulk copy per cap
event, which is rare for interactive shell use (megabytes of output per
hour at most).

The log is format-transparent — we store the raw PTY bytes including
ANSI escape sequences. On replay the client's xterm renders them back
to their original visual state, so colors + cursor moves + clear-screen
all round-trip correctly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)


DEFAULT_CAP_BYTES: Final[int] = 512 * 1024  # 512 KiB per meeting is plenty
REPLAY_BYTES: Final[int] = 256 * 1024  # how much we send on reattach


class TerminalHistoryLog:
    """Append-only byte log with a rolling byte cap.

    Intentionally small surface area — open, write, read_tail, close.
    Not thread-safe; the PTY session drives it from a single event-loop
    callback, which is already serialized.
    """

    __slots__ = ("_closed", "_fh", "_size", "cap_bytes", "path")

    def __init__(self, path: Path, *, cap_bytes: int = DEFAULT_CAP_BYTES) -> None:
        self.path = Path(path)
        self.cap_bytes = int(cap_bytes)
        self._fh = None  # type: ignore[assignment]
        self._size = 0
        self._closed = False

    def open(self) -> None:
        """Prepare the file for appending. Idempotent."""
        if self._fh is not None or self._closed:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Touch so read_tail works even before the first write.
        self.path.touch(exist_ok=True)
        try:
            self._size = self.path.stat().st_size
        except OSError:
            self._size = 0
        # Long-lived append handle owned by the writer; closed in close().
        self._fh = open(self.path, "ab", buffering=0)  # noqa: SIM115

    def write(self, chunk: bytes) -> None:
        """Append *chunk*, rotating if it would exceed the cap."""
        if self._closed or not chunk:
            return
        if self._fh is None:
            self.open()
        assert self._fh is not None

        # If a single write is already larger than the cap, keep only
        # the tail. This is the "pasted 10MB of output" pathological case.
        if len(chunk) >= self.cap_bytes:
            chunk = chunk[-self.cap_bytes // 2 :]
            self._truncate_head(to_bytes=0)

        if self._size + len(chunk) > self.cap_bytes:
            # Keep the newest half, drop the oldest half, then append.
            self._truncate_head(to_bytes=self.cap_bytes // 2)

        try:
            self._fh.write(chunk)
            self._size += len(chunk)
        except OSError as e:
            logger.warning("terminal history write failed: %s", e)

    def read_tail(self, max_bytes: int = REPLAY_BYTES) -> bytes:
        """Return the last ``max_bytes`` of stored history.

        Returned bytes may start mid-escape-sequence — xterm tolerates
        that and will recover on the next complete sequence.
        """
        try:
            with open(self.path, "rb") as r:
                try:
                    r.seek(0, os.SEEK_END)
                    size = r.tell()
                    offset = max(0, size - max_bytes)
                    r.seek(offset)
                    return r.read()
                except OSError:
                    return b""
        except FileNotFoundError:
            return b""

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._fh is not None:
            try:
                self._fh.flush()
            except OSError:
                pass
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None

    # ── internals ────────────────────────────────────────────────────

    def _truncate_head(self, *, to_bytes: int) -> None:
        """Drop everything except the last ``to_bytes`` from disk.

        Writes a small marker line so a human reading the log file knows
        where the rotation happened.
        """
        try:
            # Flush pending writes so stat+read see the current tail.
            if self._fh is not None:
                self._fh.flush()
            with open(self.path, "rb") as r:
                r.seek(0, os.SEEK_END)
                size = r.tell()
                start = max(0, size - to_bytes)
                r.seek(start)
                tail = r.read()
        except OSError as e:
            logger.warning("terminal history truncate read failed: %s", e)
            return

        marker = b"\n\x1b[38;5;244m-- log rolled --\x1b[0m\n"
        payload = marker + tail if to_bytes > 0 else b""

        # Atomic replace via *.tmp so a crash mid-truncate doesn't nuke
        # the log.
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with open(tmp_path, "wb") as w:
                w.write(payload)
            os.replace(tmp_path, self.path)
        except OSError as e:
            logger.warning("terminal history truncate replace failed: %s", e)
            return
        finally:
            with contextlib_suppress_oserror():
                tmp_path.unlink(missing_ok=True)

        # Reopen the append fd on the new file.
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
        self._fh = open(self.path, "ab", buffering=0)  # noqa: SIM115  # see open()
        self._size = len(payload)


# Small inline to avoid importing contextlib at module load time.
class contextlib_suppress_oserror:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return exc_type is not None and issubclass(exc_type, OSError)
