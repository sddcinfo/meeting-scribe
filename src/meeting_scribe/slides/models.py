"""Data models for the slide translation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class StageStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageProgress:
    status: StageStatus = StageStatus.PENDING
    progress: str | None = None  # e.g. "12/20"
    error: str | None = None

    def to_dict(self) -> dict:
        d: dict = {"status": self.status.value}
        if self.progress:
            d["progress"] = self.progress
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class SlideMeta:
    """Persisted as meta.json in the deck directory."""

    deck_id: str
    total_slides: int = 0
    source_lang: str = ""
    target_lang: str = ""
    stage: str = "validating"
    stages: dict[str, StageProgress] = field(
        default_factory=lambda: {
            "validating": StageProgress(),
            "rendering_original": StageProgress(),
            "extracting_text": StageProgress(),
            "translating": StageProgress(),
            "reinserting": StageProgress(),
            "rendering_translated": StageProgress(),
        }
    )
    error: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "deck_id": self.deck_id,
            "total_slides": self.total_slides,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "stage": self.stage,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class TextRun:
    """A single text run extracted from a PPTX shape."""

    id: str  # e.g. "s3_shape12_p0_r0"
    slide_index: int
    shape_id: int
    para_index: int
    run_index: int
    text: str
    font_name: str | None = None
    font_size: int | None = None  # EMU
    bold: bool | None = None
    italic: bool | None = None


@dataclass
class SlideText:
    """All text runs from a single slide."""

    index: int
    runs: list[TextRun] = field(default_factory=list)


@dataclass
class ValidationResult:
    valid: bool
    error: str | None = None
    slide_count: int = 0
    slide_width: int = 0  # EMU
    slide_height: int = 0  # EMU
