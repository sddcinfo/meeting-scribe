"""Abstract base classes for inference backends.

All inference is behind swappable ABCs. The same server/UI works
on POC (MacBook, MLX/Ollama) and production (GB10, vLLM+TurboQuant).

Backend hierarchy:
    ASRBackend      — speech-to-text (mlx-whisper | vLLM+Qwen3-ASR)
    DiarizeBackend  — speaker diarization (diart+pyannote | Sortformer v2)
    TranslateBackend — text translation (Ollama | vLLM+Qwen3.5)
    TTSBackend      — text-to-speech (future: Qwen3-TTS)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from meeting_scribe.models import SpeakerAttribution, TranscriptEvent


class ASRBackend(ABC):
    """Abstract speech-to-text backend.

    Receives resampled 16kHz mono float32 audio chunks.
    Yields TranscriptEvents with segment_id, revision tracking,
    and is_final flag.
    """

    @abstractmethod
    async def start(self) -> None:
        """Initialize the model (load weights, warm up)."""

    @abstractmethod
    async def stop(self) -> None:
        """Release resources."""

    @abstractmethod
    async def process_audio(
        self,
        audio: np.ndarray,
        sample_offset: int,
        sample_rate: int = 16000,
    ) -> AsyncIterator[TranscriptEvent]:
        """Process an audio chunk and yield transcript events.

        Args:
            audio: Float32 mono 16kHz audio samples.
            sample_offset: Monotonic sample counter from client.
            sample_rate: Sample rate (always 16000 after resample).

        Yields:
            TranscriptEvent with partial or final results.
        """
        yield  # type: ignore[misc]  # pragma: no cover

    @abstractmethod
    async def flush(self) -> AsyncIterator[TranscriptEvent]:
        """Flush any buffered audio and emit final events."""
        yield  # type: ignore[misc]  # pragma: no cover


class DiarizeBackend(ABC):
    """Abstract speaker diarization backend.

    Assigns cluster_id to audio segments. Separate from speaker
    identification (which maps cluster_id to enrolled identities).
    """

    @abstractmethod
    async def start(self, max_speakers: int = 4) -> None:
        """Initialize diarization model."""

    @abstractmethod
    async def stop(self) -> None:
        """Release resources."""

    @abstractmethod
    async def process_audio(
        self,
        audio: np.ndarray,
        sample_offset: int,
        sample_rate: int = 16000,
    ) -> list[SpeakerAttribution]:
        """Assign speakers to an audio chunk.

        Returns a list of SpeakerAttributions (may be >1 for overlapping speech).
        """

    @abstractmethod
    async def enroll_speaker(
        self,
        name: str,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """Enroll a speaker from a reference audio clip.

        Args:
            name: Display name (e.g., "Tanaka").
            audio: Reference audio (3-10 seconds recommended).
            sample_rate: Audio sample rate.

        Returns:
            Enrollment ID string.
        """


class TranslateBackend(ABC):
    """Abstract translation backend.

    Translates text between Japanese and English.
    """

    @abstractmethod
    async def start(self) -> None:
        """Initialize translation model."""

    @abstractmethod
    async def stop(self) -> None:
        """Release resources."""

    @abstractmethod
    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        """Translate text.

        Args:
            text: Source text.
            source_language: "ja" or "en".
            target_language: "ja" or "en".

        Returns:
            Translated text.

        Raises:
            TimeoutError: If translation exceeds configured timeout.
        """


class TTSBackend(ABC):
    """Abstract text-to-speech backend (Phase 6, future).

    Synthesizes speech from text, optionally cloning a speaker's voice
    from enrollment audio.
    """

    @abstractmethod
    async def start(self) -> None:
        """Initialize TTS model."""

    @abstractmethod
    async def stop(self) -> None:
        """Release resources."""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        language: str,
        voice_reference: np.ndarray | None = None,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """Synthesize speech from text.

        Args:
            text: Text to speak.
            language: Target language ("ja" or "en").
            voice_reference: Optional reference audio for voice cloning (3s).
            sample_rate: Output sample rate.

        Returns:
            Float32 audio samples.
        """
