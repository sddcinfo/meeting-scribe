"""Core data models for meeting transcription events.

TranscriptEvent is the fundamental unit — every ASR result, revision,
and translation flows through this model. UI renders by segment_id,
showing only the highest revision.
"""

from __future__ import annotations

import uuid
from enum import StrEnum

from pydantic import BaseModel, Field


class TranslationStatus(StrEnum):
    """Translation lifecycle states."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class MeetingState(StrEnum):
    """Meeting lifecycle state machine.

    created → recording → finalizing → complete | interrupted
    """

    CREATED = "created"
    RECORDING = "recording"
    FINALIZING = "finalizing"
    COMPLETE = "complete"
    INTERRUPTED = "interrupted"


class SpeakerAttribution(BaseModel):
    """Speaker identity for a transcript segment.

    Supports overlap: a segment can have multiple SpeakerAttributions.
    """

    cluster_id: int
    identity: str | None = None  # "Tanaka" or None if unidentified
    identity_confidence: float = 0.0  # 0.0-1.0
    source: str = "unknown"  # "enrolled" | "cluster_only" | "unknown"


class TranslationState(BaseModel):
    """Translation status and result for a transcript segment."""

    status: TranslationStatus = TranslationStatus.PENDING
    text: str | None = None
    target_language: str = "en"


class TranscriptEvent(BaseModel):
    """A single transcript event — the atomic unit of the system.

    Events are keyed by segment_id. Each segment may have multiple
    revisions as ASR refines its output. UI shows highest revision only.
    Full revision history is preserved in the journal for review.
    """

    segment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    revision: int = 0  # Monotonic per segment_id
    is_final: bool = False  # No further revisions expected
    start_ms: int = 0  # From client sample counter
    end_ms: int = 0
    language: str = "unknown"  # "ja" | "en" | "unknown"
    text: str = ""
    speakers: list[SpeakerAttribution] = Field(default_factory=list)
    translation: TranslationState | None = None

    def with_translation(
        self, status: TranslationStatus, text: str | None = None
    ) -> TranscriptEvent:
        """Return a new event with updated translation state."""
        return self.model_copy(
            update={
                "translation": TranslationState(
                    status=status,
                    text=text,
                    target_language="en" if self.language == "ja" else "ja",
                ),
            },
        )


class MeetingMeta(BaseModel):
    """Meeting metadata — persisted as meta.json with atomic write-rename."""

    meeting_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    state: MeetingState = MeetingState.CREATED
    created_at: str = ""  # ISO 8601
    organizer_token_hash: str = ""
    invite_code_hash: str = ""
    max_attendees: int = 10
    audio_sample_rate: int = 16000
    enrolled_speakers: dict[str, str] = Field(default_factory=dict)


# ── Room Layout ──────────────────────────────────────────────


class TableObject(BaseModel):
    """A table in the room — position, size, shape are all configurable.

    Coordinates and dimensions are percentages (0-100) of the container.
    borderRadius 0 = sharp corners, 50 = fully rounded (ellipse/circle).
    """

    table_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    x: float = Field(ge=0, le=100, default=50.0)  # center x %
    y: float = Field(ge=0, le=100, default=50.0)  # center y %
    width: float = Field(gt=0, le=100, default=50.0)  # width %
    height: float = Field(gt=0, le=100, default=30.0)  # height %
    border_radius: float = Field(ge=0, le=50, default=50.0)  # 0=rect, 50=ellipse
    label: str = ""


class SeatPosition(BaseModel):
    """A seat in the room with position and optional speaker enrollment."""

    seat_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    x: float = Field(ge=0, le=100, default=50.0)
    y: float = Field(ge=0, le=100, default=50.0)
    enrollment_id: str | None = None
    speaker_name: str = ""


class RoomLayout(BaseModel):
    """Room configuration — tables + seats, all freely positionable.

    Persisted as room.json in each meeting directory.
    During setup, exists as a draft in server memory.
    Presets (boardroom, classroom, etc.) set initial positions but
    everything is editable after.
    """

    preset: str = "rectangle"  # which preset was last applied
    tables: list[TableObject] = Field(default_factory=list)
    seats: list[SeatPosition] = Field(default_factory=list)


# ── Speaker Identity ─────────────────────────────────────────


class DetectedSpeaker(BaseModel):
    """A speaker discovered during a meeting.

    Separate from enrolled reference speakers — per-meeting state only.
    May be matched to an enrolled speaker or remain as "Speaker N".
    Stored in detected_speakers.json per meeting.
    """

    speaker_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    display_name: str = ""
    matched_enrollment_id: str | None = None
    match_confidence: float = 0.0
    segment_count: int = 0
    first_seen_ms: int = 0
    last_seen_ms: int = 0
