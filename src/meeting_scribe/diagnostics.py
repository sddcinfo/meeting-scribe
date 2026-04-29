"""In-process diagnostics: ring buffer of recent warnings + rotating log file.

Powers the web UI Diagnostics view so users can investigate issues without
needing terminal access. Two sinks are attached to the ``meeting_scribe``
logger:

- ``RecentLogRingBuffer`` — keeps the last N WARNING+ records in memory with
  structured metadata (timestamp, level, logger name, message, optional
  exc_info text). Surfaced as the "Issues" feed.
- ``RotatingFileHandler`` — writes the full INFO+ stream to
  ``<data_root>/diagnostics/server.log`` so the UI can tail it live.

Both are idempotent: calling ``setup_diagnostics_logging`` twice on the same
logger is a no-op.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import deque
from collections.abc import Iterator
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
RING_BUFFER_CAPACITY = 1000
LOG_FILE_MAX_BYTES = 5 * 1024 * 1024
LOG_FILE_BACKUP_COUNT = 3


class RecentLogRingBuffer(logging.Handler):
    """Bounded in-memory ring buffer of recent log records.

    Thread-safe. Stored entries are plain dicts so they can be serialised to
    JSON directly by the diag endpoint without re-touching the LogRecord.
    """

    def __init__(self, capacity: int = RING_BUFFER_CAPACITY, level: int = logging.WARNING) -> None:
        super().__init__(level=level)
        self._records: deque[dict] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._seq = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            exc_text: str | None = None
            if record.exc_info:
                exc_text = self.format(record).split("\n", 1)[-1] if record.exc_info else None
                # Fallback: build a clean traceback if format() didn't include one.
                if not exc_text or exc_text == record.getMessage():
                    import traceback as _tb

                    exc_text = "".join(_tb.format_exception(*record.exc_info))
            with self._lock:
                self._seq += 1
                self._records.append(
                    {
                        "id": self._seq,
                        "ts": record.created,
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                        "exc": exc_text,
                    }
                )
        except Exception:
            self.handleError(record)

    def snapshot(
        self,
        *,
        since_id: int | None = None,
        since_ts: float | None = None,
        level: str | None = None,
        component: str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        with self._lock:
            items = list(self._records)
        if since_id is not None:
            items = [r for r in items if r["id"] > since_id]
        if since_ts is not None:
            items = [r for r in items if r["ts"] > since_ts]
        if level:
            wanted = level.upper()
            order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if wanted in order:
                min_idx = order.index(wanted)
                items = [r for r in items if r["level"] in order[min_idx:]]
        if component:
            needle = component.lower()
            items = [r for r in items if needle in r["logger"].lower()]
        if limit and len(items) > limit:
            items = items[-limit:]
        return items


_ring_buffer: RecentLogRingBuffer | None = None
_log_file_path: Path | None = None
_setup_done = False


def setup_diagnostics_logging(data_root: Path, *, logger_name: str = "meeting_scribe") -> None:
    """Wire ring buffer + rotating file handler onto ``logger_name``.

    Idempotent: subsequent calls are no-ops. ``data_root`` is the directory
    that contains per-meeting subdirs; we create a sibling ``diagnostics/``
    folder for the rotating log.
    """
    global _ring_buffer, _log_file_path, _setup_done
    if _setup_done:
        return

    diag_dir = data_root / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    log_path = diag_dir / "server.log"

    target = logging.getLogger(logger_name)
    target.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=LOG_FILE_MAX_BYTES,
        backupCount=LOG_FILE_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    target.addHandler(file_handler)

    ring = RecentLogRingBuffer()
    ring.setFormatter(logging.Formatter(LOG_FORMAT))
    target.addHandler(ring)

    _ring_buffer = ring
    _log_file_path = log_path
    _setup_done = True


def get_ring_buffer() -> RecentLogRingBuffer | None:
    return _ring_buffer


def get_log_file_path() -> Path | None:
    return _log_file_path


def tail_log_lines(
    *,
    max_lines: int = 500,
    level: str | None = None,
    search: str | None = None,
) -> list[str]:
    """Return the trailing ``max_lines`` lines from the server log.

    ``level`` filters by minimum level; ``search`` filters case-insensitively
    by substring (applied AFTER the tail so the user always sees the most
    recent matches).
    """
    if not _log_file_path or not _log_file_path.exists():
        return []
    lines = _read_tail(_log_file_path, max_lines * 4 if (level or search) else max_lines)

    if level:
        wanted = level.upper()
        order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if wanted in order:
            keep = set(order[order.index(wanted) :])
            lines = [ln for ln in lines if any(f" {lvl} " in ln for lvl in keep)]
    if search:
        needle = search.lower()
        lines = [ln for ln in lines if needle in ln.lower()]
    return lines[-max_lines:]


def _read_tail(path: Path, max_lines: int) -> list[str]:
    """Efficient tail without loading the whole file into memory."""
    block = 4096
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            data = b""
            seen = 0
            while size > 0 and seen <= max_lines:
                read = min(block, size)
                size -= read
                f.seek(size)
                chunk = f.read(read)
                data = chunk + data
                seen = data.count(b"\n")
            text = data.decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = text.splitlines()
    return lines[-max_lines:]


async def stream_log_lines(*, poll_interval_s: float = 0.5) -> Iterator[str]:
    """Async generator that yields *new* lines appended to the server log.

    Tracks file position; reopens after rotation (detected via inode change
    or file shrinking). Yields one line per iteration so SSE callers can
    forward each as its own event.
    """
    import asyncio as _asyncio

    if not _log_file_path:
        return
    path = _log_file_path
    pos = 0
    inode = None
    if path.exists():
        try:
            st = path.stat()
            pos = st.st_size
            inode = st.st_ino
        except OSError:
            pos = 0

    while True:
        try:
            st = path.stat() if path.exists() else None
            if st is None:
                await _asyncio.sleep(poll_interval_s)
                continue
            # Rotation: inode changed or file truncated below our cursor.
            if inode is not None and (st.st_ino != inode or st.st_size < pos):
                pos = 0
                inode = st.st_ino
            if st.st_size > pos:
                with path.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
                for line in chunk.splitlines():
                    if line:
                        yield line
            await _asyncio.sleep(poll_interval_s)
        except OSError:
            await _asyncio.sleep(poll_interval_s)
