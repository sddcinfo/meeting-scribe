"""Atomic filesystem helpers shared across the codebase."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, data: Any) -> None:
    """Write ``data`` to ``path`` atomically (tmp file + rename).

    The tmp filename carries a uuid suffix so concurrent writers to the
    same target path can't race on the same tmp file. Without the uuid,
    thread A's rename completes first and thread B's rename then fails
    with FileNotFoundError — observed on slide deck meta.json updates
    when the extract-text and render-originals phases ran in parallel.

    ``ensure_ascii=False`` so non-ASCII content (e.g. translated slide
    text) round-trips losslessly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


def atomic_append_jsonl(path: Path, item: dict[str, Any]) -> None:
    """Append a single JSONL record to ``path`` durably.

    Used by the translation-queue quiesce path — once a meeting is
    quiesced, every ``submit()`` call MUST persist its work before
    returning so a process crash between the submit and finalize
    can't drop it. The file is the authoritative store for any
    not-yet-translated work.

    Concurrency contract:
      * Caller MUST hold a per-meeting serializing lock so two
        appenders in the same process can't interleave.
      * For inter-process safety we rely on POSIX's atomicity guarantee
        for ``write()`` calls of less than ``PIPE_BUF`` (4 KiB on
        Linux) to an ``O_APPEND`` file. Records exceeding that size
        fall through to the temp-file-rename path and merge on read.

    Durability:
      * ``fsync`` is called after every write so the record survives a
        process kill.

    JSON shape:
      * ``ensure_ascii=False`` so non-ASCII translation text round-trips.
      * Single line; no embedded newlines (callers must not include
        them in string fields).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(item, ensure_ascii=False).encode("utf-8") + b"\n"
    if len(line) < 4096:
        # Fast path: single atomic write to O_APPEND fd.
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line)
            os.fsync(fd)
        finally:
            os.close(fd)
        return

    # Slow path for oversized records: read existing, rewrite atomically.
    existing = path.read_bytes() if path.exists() else b""
    tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    tmp.write_bytes(existing + line)
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file as a list of dicts. Skips malformed lines."""
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
