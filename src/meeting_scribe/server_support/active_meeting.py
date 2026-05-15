"""Active-meeting marker persistence (crash-recovery).

A small JSON file on disk records the meeting ID + start time of the
in-progress meeting. On clean stop the marker is removed. On startup
the server reads the marker to detect an interrupted recording — the
UI then prompts the user to resume or discard.

The path is fixed at ``/tmp/meeting-scribe-active.json`` so a process
crash that loses the heap still leaves the marker for the next boot.
"""

from __future__ import annotations

import time
from pathlib import Path

_ACTIVE_MEETING_FILE = Path("/tmp/meeting-scribe-active.json")


def _persist_active_meeting(meeting_id: str) -> None:
    """Write current meeting ID to disk for crash recovery."""
    import json as _json

    _ACTIVE_MEETING_FILE.write_text(
        _json.dumps({"meeting_id": meeting_id, "start_time": time.time()}) + "\n"
    )


def _clear_active_meeting() -> None:
    """Remove the active meeting marker on clean stop."""
    _ACTIVE_MEETING_FILE.unlink(missing_ok=True)


def _get_interrupted_meeting() -> str | None:
    """Check if a meeting was active when the server last crashed."""
    import json as _json

    if not _ACTIVE_MEETING_FILE.exists():
        return None
    try:
        data = _json.loads(_ACTIVE_MEETING_FILE.read_text())
        return data.get("meeting_id")
    except Exception:
        return None
