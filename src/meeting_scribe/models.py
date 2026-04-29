"""Core data models for meeting transcription events.

TranscriptEvent is the fundamental unit — every ASR result, revision,
and translation flows through this model. UI renders by segment_id,
showing only the highest revision.
"""

from __future__ import annotations

import uuid
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from meeting_scribe.languages import is_valid_languages


class TranslationStatus(StrEnum):
    """Translation lifecycle states."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class MeetingState(StrEnum):
    """Meeting lifecycle state machine.

    created -> recording -> finalizing -> complete | interrupted
    complete -> reprocessing -> complete   (transient, cleared on success)
    """

    CREATED = "created"
    RECORDING = "recording"
    FINALIZING = "finalizing"
    COMPLETE = "complete"
    INTERRUPTED = "interrupted"
    # Set by reprocess_meeting() at step 0 and cleared at step 7. If the
    # server is killed mid-reprocess (e.g. systemd TimeoutStopUSec fires),
    # the flag persists. Listed in the enum so MeetingMeta can round-trip
    # a reprocessing meta.json through pydantic; recover_interrupted()
    # flips it back to COMPLETE on the next startup.
    REPROCESSING = "reprocessing"


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
    completed_at: float | None = None
    # Monotonic ts when translation finished. Used as the TTS deadline
    # origin (upstream-aware anchor) and for upstream_lag_ms metrics.


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
    # HTML-annotated form of `text` with `<ruby>kanji<rt>kana</rt></ruby>`
    # wraps around every kanji run. Generated async by the furigana
    # backend; None when the language isn't JA, the backend is
    # unavailable, or annotation hasn't landed yet. UI renders this
    # instead of `text` when present so the raw text stays as the
    # fallback path for any consumer that doesn't want ruby.
    furigana_html: str | None = None
    speakers: list[SpeakerAttribution] = Field(default_factory=list)
    translation: TranslationState | None = None
    utterance_end_at: float | None = Field(default=None, exclude=True)
    # Monotonic wall-clock ts at which the speaker's audio actually ended
    # (computed from audio_wall_at_start + end_ms/1000). This is the
    # authoritative origin for the TTS speech-end SLA. excluded from
    # serialisation — internal server-side use only.

    def with_translation(
        self,
        status: TranslationStatus,
        text: str | None = None,
        target_language: str | None = None,
    ) -> TranscriptEvent:
        """Return a new event with updated translation state."""
        return self.model_copy(
            update={
                "translation": TranslationState(
                    status=status,
                    text=text,
                    target_language=target_language or "",
                ),
            },
        )


class MeetingMeta(BaseModel):
    """Meeting metadata — persisted as meta.json with atomic write-rename."""

    meeting_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    state: MeetingState = MeetingState.CREATED
    created_at: str = ""  # ISO 8601 — when the meeting record was created
    # Wall-clock anchor for the recording. Set to the unix epoch (ms)
    # of the FIRST audio sample in recording.pcm. Together with
    # audio_sample_rate this makes the PCM file an absolute-time record:
    #     wall_clock_ms = recording_started_epoch_ms + byte_offset / (sample_rate * 2 / 1000)
    # The transcript `start_ms` fields are byte-offset-relative to the
    # audio file, so combining them with this anchor yields the real
    # UTC timestamp each segment was spoken.
    recording_started_epoch_ms: int = 0
    organizer_token_hash: str = ""
    invite_code_hash: str = ""
    max_attendees: int = 10
    audio_sample_rate: int = 16000
    # Languages spoken in the meeting. Length 1 = monolingual (no translation
    # work is scheduled, UI collapses the second pane); length 2 = bilingual
    # pair with distinct codes. The field is named ``language_pair`` for
    # historical reasons (it predates monolingual support); kept as-is because
    # it is baked into persisted meta.json, journal.ndjson, and exported
    # artifacts — all of which are internal-only and can still evolve freely.
    #
    # The Pydantic validator below is the authoritative shape check; every
    # code path that constructs a ``MeetingMeta`` (API, reload, fixtures)
    # runs through it, so invalid shapes can never enter the system.
    language_pair: list[str] = Field(default_factory=lambda: ["en", "ja"])

    @field_validator("language_pair")
    @classmethod
    def _validate_language_pair(cls, v: list[str]) -> list[str]:
        if not is_valid_languages(v):
            raise ValueError(
                "language_pair must be 1 or 2 distinct codes from the language registry"
            )
        return v

    @property
    def is_monolingual(self) -> bool:
        """True iff the meeting has a single language (no translation work)."""
        return len(self.language_pair) == 1

    enrolled_speakers: dict[str, str] = Field(default_factory=dict)
    # User-toggled "useful for demo / reference" mark. Surfaced in the
    # meetings list so favorites are easy to spot at a glance.
    is_favorite: bool = False

    def audio_offset_to_epoch_ms(self, byte_offset: int) -> int | None:
        """Translate a byte position in recording.pcm to a unix epoch ms.

        Returns None until the recording anchor has been set (i.e.
        until the first audio chunk has been written). Assumes s16le
        mono audio at ``audio_sample_rate``.
        """
        if self.recording_started_epoch_ms <= 0:
            return None
        bytes_per_ms = (self.audio_sample_rate * 2) / 1000.0
        return int(self.recording_started_epoch_ms + byte_offset / bytes_per_ms)

    def transcript_ms_to_epoch_ms(self, start_ms: int) -> int | None:
        """Translate a transcript ``start_ms`` (audio-file-relative) to
        unix epoch ms. Companion to ``audio_offset_to_epoch_ms`` for
        the common case where the caller has a transcript segment's
        timestamp in hand and wants its wall-clock equivalent."""
        if self.recording_started_epoch_ms <= 0:
            return None
        return self.recording_started_epoch_ms + int(start_ms)


# ── Room Layout ──────────────────────────────────────────────


class TableObject(BaseModel):
    """A table in the room — position, size, shape are all configurable.

    Coordinates and dimensions are percentages (0-100) of the container.
    borderRadius 0 = sharp corners, 50 = fully rounded (ellipse/circle).
    """

    table_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    x: float = Field(ge=0, le=100, default=50.0)  # center x %
    y: float = Field(ge=0, le=100, default=50.0)  # center y %
    width: float = Field(gt=0, le=100, default=50.0)  # width %
    height: float = Field(gt=0, le=100, default=30.0)  # height %
    border_radius: float = Field(ge=0, le=50, default=50.0)  # 0=rect, 50=ellipse
    label: str = ""


class SeatPosition(BaseModel):
    """A seat in the room with position and optional speaker enrollment."""

    seat_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
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

    speaker_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    display_name: str = ""
    matched_enrollment_id: str | None = None
    match_confidence: float = 0.0
    segment_count: int = 0
    first_seen_ms: int = 0
    last_seen_ms: int = 0
