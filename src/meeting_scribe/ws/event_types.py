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
  4. A JS-side dev-mode ``default:`` clause in the admin SPA logs
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
    MEETING_STARTED = "meeting_started"
    MEETING_CANCELLED = "meeting_cancelled"
    MEETING_STOPPED = "meeting_stopped"
    MEETING_WARNING = "meeting_warning"
    MEETING_WARNING_CLEARED = "meeting_warning_cleared"

    # Pipeline / status (no segment_id; per-meeting state)
    AUDIO_DRIFT = "audio_drift"
    # Live mic input level (peak amplitude, 0-100). Broadcast from
    # ws/audio_input._handle_audio every Nth chunk during a recording —
    # covers both browser-mic and SP325/server-mic paths since both
    # funnel through that handler. Drives the header `.meter-bar`.
    MIC_LEVEL = "mic_level"
    FINALIZE_PROGRESS = "finalize_progress"
    # Phase B's progress channel — fires for the meeting whose Phase B
    # is running in the background after Stop. Always carries
    # ``meeting_id`` so clients can route to the corner toast (NOT the
    # main blocking modal) without ambiguity.
    BACKGROUND_FINALIZE_PROGRESS = "background_finalize_progress"
    SUMMARY_REGENERATED = "summary_regenerated"
    TRANSCRIPT_REVISION = "transcript_revision"
    INTERPRETATION_STATUS = "interpretation_status"
    BT_STATUS = "bt_status"

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

    # Pop-out layout: server-authoritative state that the laptop admin
    # mirrors to the kiosk HDMI mirror in <1 s. Fired from the
    # ``PUT /api/admin/popout-layout`` handler after persistence;
    # carries the full layout shape + a monotonic ``version`` and the
    # initiating tab's ``source_tab_id`` so the originator can ignore
    # its own echo.
    POPOUT_LAYOUT_CHANGED = "popout_layout_changed"


# Frozen set for O(1) membership checks.
WS_EVENT_TYPES: frozenset[str] = frozenset(t.value for t in WsEventType)


# Event types kiosk-role subscribers may receive. Operator-only
# signals (BT pairing, audio drift, finalize progress, slide-job
# internals, transcript revisions) are NOT in this set: kiosk users
# can't act on them and they leak operator diagnostics to anyone with
# a kiosk cookie.
#
# Update this allowlist when adding kiosk-relevant content events.
# ``server_support/broadcast.py`` enforces it on every broadcast site.
KIOSK_ALLOWED_EVENT_TYPES: frozenset[str] = frozenset(
    {
        WsEventType.MEETING_STARTED.value,
        WsEventType.MEETING_CANCELLED.value,
        WsEventType.MEETING_STOPPED.value,
        WsEventType.SUMMARY_REGENERATED.value,
        WsEventType.SPEAKER_PULSE.value,
        WsEventType.SPEAKER_RENAME.value,
        WsEventType.SLIDE_CHANGE.value,
        WsEventType.SLIDE_DECK_CHANGED.value,
        WsEventType.POPOUT_LAYOUT_CHANGED.value,
    }
)


def is_known_type(type_value: str | None) -> bool:
    """Return True when ``type_value`` is a recognized control event type."""
    return isinstance(type_value, str) and type_value in WS_EVENT_TYPES


def is_kiosk_allowed(type_value: str | None) -> bool:
    """True when ``type_value`` may be delivered to a Role.KIOSK socket."""
    return isinstance(type_value, str) and type_value in KIOSK_ALLOWED_EVENT_TYPES
