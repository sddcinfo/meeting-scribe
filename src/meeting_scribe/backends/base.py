"""Abstract base classes for inference backends.

All inference is behind swappable ABCs for GB10 production.

Backend hierarchy:
    ASRBackend       — Qwen3-ASR-1.7B via vLLM
    DiarizeBackend   — pyannote.audio speaker diarization
    TranslateBackend — vLLM+Qwen3.6
    TTSBackend       — Qwen3-TTS via vLLM
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

    # ── Shared state attributes (declared on ABC so server.py can
    # access them without narrowing to a concrete subclass).  Concrete
    # backends are expected to populate these in __init__; the ABC
    # provides default sentinels so mypy sees the attributes exist on
    # the union of all backend types.
    _buffer: list[np.ndarray]
    _buffer_samples: int
    _segment_id: str | None
    _base_offset: int
    audio_wall_at_start: float | None

    @abstractmethod
    async def start(self) -> None:
        """Initialize the model (load weights, warm up)."""

    @abstractmethod
    async def stop(self) -> None:
        """Release resources."""

    async def process_audio_bytes(
        self,
        pcm_s16le: bytes,
        sample_offset: int | None = None,
    ) -> None:
        """Process raw s16le PCM bytes.

        Byte-oriented convenience wrapper around ``process_audio``.
        Default implementation decodes the bytes as little-endian int16,
        converts to float32, and drains ``process_audio`` into the
        backend's internal event-dispatch pipeline.  Concrete backends
        override for zero-copy paths.  Declared here so server.py can
        call it without narrowing to a specific subclass.
        """
        audio = np.frombuffer(pcm_s16le, dtype=np.int16).astype(np.float32) / 32768.0
        effective_offset = sample_offset if sample_offset is not None else 0
        async for _ in self.process_audio(audio, effective_offset):
            pass

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

    def set_languages(self, languages: list[str] | tuple[str, ...]) -> None:
        """Update the ASR prompt to target these languages for the next
        meeting. Default is a no-op so backends that don't support
        runtime language switching (e.g. a hypothetical monolingual
        model) inherit a safe default. Concrete language-aware
        backends (``VllmASRBackend``) override to invalidate their
        system-prompt cache. Declared on the ABC so server.py can
        call it without narrowing to a specific subclass.
        """
        return None


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
        prior_context: list[tuple[str, str]] | None = None,
        meeting_id: str | None = None,
    ) -> str:
        """Translate text.

        Args:
            text: Source text.
            source_language: "ja" or "en".
            target_language: "ja" or "en".
            prior_context: Optional rolling window of earlier
                ``(source_text, translation)`` tuples from the same
                meeting.  When passed, the backend is expected to
                include them in the system prompt so the model can
                anchor on the running topic/speakers rather than
                hallucinate full sentences from fragmented utterances.
                Must be oldest → newest.  ``None`` preserves the
                stateless per-utterance prompt (current live-path
                behavior).
            meeting_id: Optional UUID of the active meeting.  Backends
                that write JSONL rows tag every row with this id so the
                validation harness can attribute load to a specific
                meeting.  ``None`` for callers outside of an active
                meeting (warmup, ad-hoc tests).

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
