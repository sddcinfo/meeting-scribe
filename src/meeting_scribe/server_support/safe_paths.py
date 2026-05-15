"""Path validation helpers — prevent directory traversal via meeting_id
or segment_id.

Every route that accepts a ``{meeting_id}`` or ``{segment_id}`` path
parameter passes it through one of these helpers BEFORE touching the
filesystem. Returning ``None`` lets the caller respond with a 400 /
404 instead of letting a crafted ``..`` walk out of the meetings
tree.
"""

from __future__ import annotations

from pathlib import Path

from meeting_scribe.runtime import state


def _safe_meeting_dir(meeting_id: str) -> Path | None:
    """Resolve a meeting directory path, rejecting traversal attacks."""
    if not meeting_id or ".." in meeting_id or "/" in meeting_id or "\\" in meeting_id:
        return None
    meeting_dir = (state.storage._meetings_dir / meeting_id).resolve()
    if not meeting_dir.is_relative_to(state.storage._meetings_dir.resolve()):
        return None
    return meeting_dir


def _safe_segment_path(meeting_id: str, subdir: str, filename: str) -> Path | None:
    """Resolve a file path within a meeting subdirectory, rejecting traversal."""
    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir:
        return None
    if ".." in filename or "/" in filename or "\\" in filename:
        return None
    path = (meeting_dir / subdir / filename).resolve()
    if not path.is_relative_to(meeting_dir.resolve()):
        return None
    return path
