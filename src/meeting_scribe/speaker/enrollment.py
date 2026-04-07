"""Speaker enrollment — extract and store voice embeddings.

Uses SpeechBrain's ECAPA-TDNN for speaker embedding extraction.
Enrollment audio (3-10 seconds) is processed into a fixed-dimension
embedding vector used for later verification.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class EnrolledSpeaker:
    """A speaker with a stored voice embedding."""

    name: str
    embedding: np.ndarray  # Fixed-dimension vector from ECAPA-TDNN
    enrollment_id: str = ""
    audio_duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "enrollment_id": self.enrollment_id,
            "embedding": self.embedding.tolist(),
            "audio_duration_seconds": self.audio_duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EnrolledSpeaker:
        return cls(
            name=data["name"],
            enrollment_id=data.get("enrollment_id", ""),
            embedding=np.array(data["embedding"], dtype=np.float32),
            audio_duration_seconds=data.get("audio_duration_seconds", 0.0),
        )


class SpeakerEnrollmentStore:
    """Manages enrolled speaker embeddings.

    Stores embeddings in-memory with JSON persistence.
    One store per meeting.
    """

    def __init__(self, storage_path: Path | None = None) -> None:
        self._speakers: dict[str, EnrolledSpeaker] = {}
        self._storage_path = storage_path

    @property
    def speakers(self) -> dict[str, EnrolledSpeaker]:
        return dict(self._speakers)

    def add(self, speaker: EnrolledSpeaker) -> None:
        """Add or update an enrolled speaker."""
        self._speakers[speaker.enrollment_id] = speaker
        self._persist()
        logger.info("Enrolled speaker: %s (id=%s)", speaker.name, speaker.enrollment_id)

    def remove(self, enrollment_id: str) -> bool:
        """Remove an enrolled speaker."""
        if enrollment_id in self._speakers:
            del self._speakers[enrollment_id]
            self._persist()
            return True
        return False

    def get_all_embeddings(self) -> list[tuple[str, str, np.ndarray]]:
        """Return (enrollment_id, name, embedding) for all enrolled speakers."""
        return [(s.enrollment_id, s.name, s.embedding) for s in self._speakers.values()]

    def _persist(self) -> None:
        """Save to JSON if storage path is set."""
        if self._storage_path is None:
            return
        data = {eid: s.to_dict() for eid, s in self._speakers.items()}
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage_path.write_text(json.dumps(data, indent=2))

    def load(self) -> None:
        """Load from JSON if storage path exists."""
        if self._storage_path is None or not self._storage_path.exists():
            return
        data = json.loads(self._storage_path.read_text())
        self._speakers = {eid: EnrolledSpeaker.from_dict(s) for eid, s in data.items()}
        logger.info("Loaded %d enrolled speakers", len(self._speakers))
