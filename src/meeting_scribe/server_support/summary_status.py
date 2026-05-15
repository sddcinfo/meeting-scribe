"""Summary-generation status envelope, separate from ``summary.json``.

`summary.json` itself is the success artifact: when it exists, downstream
readers (`storage.py:870`, `routes/meeting_crud.py:67,91`,
`routes/speaker.py:311`, `routes/meeting_lifecycle.py:817`) treat the
file's existence as "summary ready". We keep that contract intact and
write failure / in-progress / staleness signals into a sibling file
``summary.status.json`` that has its own shape.

Why a separate file: writing an error payload into ``summary.json``
would (1) trigger the regen-skip gate at ``meeting_lifecycle.py:817``,
(2) confuse readers expecting `topics`/`action_items`, and (3) risk
persisting raw exception text — which can include transcript
fragments, prompts, hostnames, or other PII.

The status envelope is enum-only:

  * ``status`` ∈ {``generating``, ``complete``, ``error``, ``stale``}
  * ``error_code`` ∈ ``SummaryErrorCode``
  * ``user_safe_message`` is looked up from a fixed table — no runtime
    text from exceptions ever lands on disk via this helper.

Raw exception strings are emitted to the admin-only server log at WARN
(short, classified) and DEBUG (full) — never persisted, never returned
through the public ``/api/meetings/{id}/summary-status`` route.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from meeting_scribe.util.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)

STATUS_FILENAME: Final[str] = "summary.status.json"
SCHEMA_VERSION: Final[int] = 1


class SummaryStatus(StrEnum):
    GENERATING = "generating"
    COMPLETE = "complete"
    ERROR = "error"
    STALE = "stale"


class SummaryErrorCode(StrEnum):
    VLLM_TIMEOUT = "vllm_timeout"
    VLLM_5XX = "vllm_5xx"
    VLLM_UNREACHABLE = "vllm_unreachable"
    TRANSCRIPT_EMPTY = "transcript_empty"
    TRANSCRIPT_TOO_LARGE = "transcript_too_large"
    INTERNAL = "internal"


_USER_SAFE_MESSAGES: Final[dict[SummaryErrorCode, str]] = {
    SummaryErrorCode.VLLM_TIMEOUT: "Summary backend timed out",
    SummaryErrorCode.VLLM_5XX: "Summary backend returned an error",
    SummaryErrorCode.VLLM_UNREACHABLE: "Summary backend unreachable",
    SummaryErrorCode.TRANSCRIPT_EMPTY: "No transcript content available",
    SummaryErrorCode.TRANSCRIPT_TOO_LARGE: "Transcript exceeds size limit",
    SummaryErrorCode.INTERNAL: "Summary generation failed (see admin logs)",
}

_RETRYABLE: Final[frozenset[SummaryErrorCode]] = frozenset(
    {
        SummaryErrorCode.VLLM_TIMEOUT,
        SummaryErrorCode.VLLM_5XX,
        SummaryErrorCode.VLLM_UNREACHABLE,
        SummaryErrorCode.INTERNAL,
    }
)


def classify_summary_error(exc_or_dict: BaseException | dict | str) -> SummaryErrorCode:
    """Map an exception or ``{"error": ...}`` dict to an enum code.

    Pattern-matches on exception type or a tiny set of substrings in the
    error string. Never returns the input — only the enum.
    """
    if isinstance(exc_or_dict, dict):
        raw = str(exc_or_dict.get("error") or "")
    elif isinstance(exc_or_dict, BaseException):
        raw = f"{type(exc_or_dict).__name__}: {exc_or_dict}"
    else:
        raw = str(exc_or_dict)

    low = raw.lower()
    if "timeout" in low or "timedout" in low or "readtimeout" in low:
        return SummaryErrorCode.VLLM_TIMEOUT
    if "connecterror" in low or "unreachable" in low or "refused" in low:
        return SummaryErrorCode.VLLM_UNREACHABLE
    if re.search(r"\b5\d\d\b", raw) or "internal server error" in low:
        return SummaryErrorCode.VLLM_5XX
    if "no transcript" in low or "transcript_empty" in low:
        return SummaryErrorCode.TRANSCRIPT_EMPTY
    if "too large" in low or "max_transcript" in low:
        return SummaryErrorCode.TRANSCRIPT_TOO_LARGE
    return SummaryErrorCode.INTERNAL


def transcript_hash(journal_path: Path) -> str:
    """Stable content hash of ``journal.jsonl``.

    Tied to the status envelope so a later journal mutation (e.g.
    ``reprocess --summaries-only``) marks any prior status STALE on
    read.
    """
    if not journal_path.exists():
        return "sha256:empty"
    h = hashlib.sha256()
    with journal_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def read_status(meeting_dir: Path) -> dict[str, Any] | None:
    """Return the status envelope dict, or ``None`` if absent/corrupt."""
    p = meeting_dir / STATUS_FILENAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError, OSError:
        return None


def next_attempt_id(meeting_dir: Path) -> int:
    """Increment the attempt counter from any prior status (1 if absent)."""
    prior = read_status(meeting_dir)
    if prior is None:
        return 1
    try:
        return int(prior.get("attempt_id", 0)) + 1
    except TypeError, ValueError:
        return 1


def write_status(
    meeting_dir: Path,
    status: SummaryStatus,
    *,
    attempt_id: int,
    journal_path: Path,
    error_code: SummaryErrorCode | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Atomically write the status envelope.

    ``error_code`` is required when status==ERROR; ignored otherwise.
    ``extra`` may carry caller-controlled metadata (e.g. ``partial``,
    ``pending_translation_count``) — caller is responsible for ensuring
    nothing in ``extra`` contains exception text or user content.
    """
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status.value,
        "error_code": error_code.value if error_code else None,
        "user_safe_message": (_USER_SAFE_MESSAGES[error_code] if error_code else None),
        "retryable": (error_code in _RETRYABLE) if error_code else None,
        "attempt_id": attempt_id,
        "transcript_hash": transcript_hash(journal_path),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        payload.update(extra)
    atomic_write_json(meeting_dir / STATUS_FILENAME, payload)


def is_stale(meeting_dir: Path, journal_path: Path) -> bool:
    """True when the status envelope's transcript_hash diverges from disk."""
    s = read_status(meeting_dir)
    if s is None:
        return False
    return s.get("transcript_hash") != transcript_hash(journal_path)
