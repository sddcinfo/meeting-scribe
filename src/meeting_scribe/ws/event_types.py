"""Registry of every server-emitted WebSocket event ``type`` value.

Why this exists: the popout's view-WS handler (and the admin's audio-WS
handler) cascade over `msg.type === '...'` cases, with a catch-all
``else`` at the end. When a new server-emitted type is introduced
without updating both client cascades, it falls into the catch-all and
gets silently funnelled through `ingestFromLiveWs → store.ingest`,
which corrupts the SegmentStore (the popout-clear-on-pulse class of
bugs). Schema validation alone doesn't catch this — the message shape
is fine; the routing is wrong.

The fix is process-level:

  1. Every server-emitted ``type`` is registered here as a StrEnum
     value.
  2. ``_broadcast_json`` validates that the payload's ``type`` is
     in this enum (in dev / debug mode); ``WS_EVENT_TYPES_STRICT=1``
     turns the warning into a hard error.
  3. A pytest test (``test_ws_event_handler_coverage``) replays a
     canonical sample of each type through the JS client handlers
     and asserts exactly one named branch ran for each — no
     catch-all fallbacks.
  4. A JS-side dev-mode ``default:`` clause in ``scribe-app.js`` logs
     a counter when an unhandled type arrives so the test can detect
     drift.

Adding a new type? Add it here, AND a sample at
``tests/contracts/ws_event_samples/<type>.json``, AND named handlers
in both client cascades.
"""

from __future__ import annotations

from enum import StrEnum


class WsEventType(StrEnum):
    """Server-emitted WebSocket event ``type`` values.

    NOTE: TranscriptEvent does NOT carry a ``type`` field — it's the
    "default" arm and is identified by its ``segment_id``. This enum
    only enumerates control messages broadcast via ``_broadcast_json``.
    """

    # Lifecycle (carry ``meeting_id``)
    DEV_RESET = "dev_reset"
    MEETING_CANCELLED = "meeting_cancelled"
    MEETING_STOPPED = "meeting_stopped"
    MEETING_WARNING = "meeting_warning"
    MEETING_WARNING_CLEARED = "meeting_warning_cleared"

    # Pipeline / status (no segment_id; per-meeting state)
    AUDIO_DRIFT = "audio_drift"
    FINALIZE_PROGRESS = "finalize_progress"
    SUMMARY_REGENERATED = "summary_regenerated"
    TRANSCRIPT_REVISION = "transcript_revision"

    # Speakers / room
    ROOM_LAYOUT_UPDATE = "room_layout_update"
    SPEAKER_ASSIGNMENT = "speaker_assignment"
    SPEAKER_CORRECTION = "speaker_correction"
    SPEAKER_PULSE = "speaker_pulse"
    SPEAKER_REMAP = "speaker_remap"
    SPEAKER_RENAME = "speaker_rename"

    # Slides
    SLIDE_CHANGE = "slide_change"
    SLIDE_DECK_CHANGED = "slide_deck_changed"
    SLIDE_JOB_PROGRESS = "slide_job_progress"
    SLIDE_PARTIAL_READY = "slide_partial_ready"


# Frozen set for O(1) membership checks.
WS_EVENT_TYPES: frozenset[str] = frozenset(t.value for t in WsEventType)


def is_known_type(type_value: str | None) -> bool:
    """Return True when ``type_value`` is a recognized control event type."""
    return isinstance(type_value, str) and type_value in WS_EVENT_TYPES
