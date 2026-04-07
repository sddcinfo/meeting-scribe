"""Direct MLX Whisper ASR backend — simple, reliable batch inference.

Accumulates audio in a buffer, runs inference every ~2 seconds.
No WhisperLiveKit dependency — just raw mlx-whisper transcribe().
Uses the smallest model (tiny) for maximum reliability and speed.

This is the "it just works" backend. Quality comes from the GB10 with vLLM.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import AsyncIterator

import numpy as np

from meeting_scribe.backends.base import ASRBackend
from meeting_scribe.models import TranscriptEvent

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
VAD_ENERGY_THRESHOLD = 0.005  # Lower threshold for mic audio
NO_SPEECH_THRESHOLD = 0.6

# CJK character detection for language classification
_CJK_RE = re.compile(r"[\u3000-\u9fff\uff00-\uffef]")

# Known hallucination patterns
_HALLUCINATIONS = {
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "ご視聴ありがとうございました",
    "チャンネル登録",
    "字幕",
    "subtitles",
}


def _detect_language(text: str) -> str:
    if not text or len(text.strip()) < 2:
        return "unknown"
    cjk = len(_CJK_RE.findall(text))
    total = len(text.replace(" ", ""))
    if total == 0:
        return "unknown"
    return "ja" if cjk / total > 0.3 else "en"


def _is_hallucination(text: str) -> bool:
    lower = text.lower().strip()
    for p in _HALLUCINATIONS:
        if p in lower:
            return True
    # Repeated words (3+ identical consecutive words)
    words = lower.split()
    if len(words) >= 3 and words[0] == words[1] == words[2]:
        return True

    stripped = lower.replace(" ", "").replace(",", "").replace(".", "")
    if len(stripped) < 4:
        return False

    # Single char dominance (>50%)
    from collections import Counter

    _most, count = Counter(stripped).most_common(1)[0]
    if count / len(stripped) > 0.5:
        return True

    # Repeating short patterns (1-4 chars repeated 4+ times, at any offset)
    for plen in range(1, 5):
        for start in range(plen):
            if len(stripped) - start >= plen * 4:
                pat = stripped[start : start + plen]
                repeats = 0
                for i in range(start, len(stripped) - plen + 1, plen):
                    if stripped[i : i + plen] == pat:
                        repeats += 1
                    else:
                        break
                if repeats >= 4:
                    return True

    # Text is unreasonably long without spaces (>60 chars continuous)
    return any(len(word) > 60 for word in text.split())


class MlxWhisperBackend(ASRBackend):
    """Direct mlx-whisper batch inference. Simple and reliable.

    Accumulates ~4s of audio, runs whisper.transcribe(), emits events.
    No streaming policy, no buffer management, no stale detection needed.
    """

    def __init__(
        self, model: str = "mlx-community/whisper-tiny", buffer_seconds: float = 4.0
    ) -> None:
        self._model_name = model
        self._model = None
        self._buffer: list[np.ndarray] = []
        self._buffer_samples = 0
        self._buffer_threshold = int(buffer_seconds * SAMPLE_RATE)
        self._segment_id: str | None = None
        self._revision = 0
        self._base_offset = 0
        self._on_event = None

    def set_event_callback(self, fn) -> None:
        """Register async callback for event delivery (matches WLK backend interface)."""
        self._on_event = fn

    async def start(self) -> None:
        import mlx_whisper

        logger.info("Loading mlx-whisper: %s", self._model_name)
        # Warm up
        silent = np.zeros(SAMPLE_RATE, dtype=np.float32)
        mlx_whisper.transcribe(
            silent, path_or_hf_repo=self._model_name, language="en", no_speech_threshold=0.9
        )
        self._model = mlx_whisper
        logger.info("mlx-whisper loaded: %s", self._model_name)

    async def stop(self) -> None:
        self._model = None
        self._buffer.clear()
        self._buffer_samples = 0

    async def process_audio(
        self, audio: np.ndarray, sample_offset: int, sample_rate: int = 16000
    ) -> AsyncIterator[TranscriptEvent]:
        """Buffer audio and run inference when threshold is reached."""
        if self._model is None:
            return

        if self._segment_id is None:
            self._segment_id = str(uuid.uuid4())
            self._revision = 0
            self._base_offset = sample_offset

        self._buffer.append(audio)
        self._buffer_samples += len(audio)

        if self._buffer_samples >= self._buffer_threshold:
            combined = np.concatenate(self._buffer)
            rms = float(np.sqrt(np.mean(combined**2)))

            if rms < VAD_ENERGY_THRESHOLD:
                # Silence — if we had a segment going, finalize it empty
                self._buffer.clear()
                self._buffer_samples = 0
                if self._segment_id and self._revision > 0:
                    self._segment_id = None
                    self._revision = 0
                return

            # Run inference
            result = self._model.transcribe(
                combined,
                path_or_hf_repo=self._model_name,
                no_speech_threshold=NO_SPEECH_THRESHOLD,
                condition_on_previous_text=False,
            )

            text = (result.get("text") or "").strip()
            whisper_lang = result.get("language", "unknown")

            # Filter hallucinations and unreasonably long text
            if text and (len(text) > 200 or _is_hallucination(text)):
                logger.debug("Filtered hallucination: '%s'", text[:40])
                text = ""

            if text:
                self._revision += 1
                lang = _detect_language(text) if whisper_lang == "unknown" else whisper_lang
                # Normalize language to ja/en only
                if lang not in ("ja", "en"):
                    lang = _detect_language(text)

                start_ms = int(self._base_offset / SAMPLE_RATE * 1000)
                end_ms = start_ms + int(len(combined) / SAMPLE_RATE * 1000)

                event = TranscriptEvent(
                    segment_id=self._segment_id,
                    revision=self._revision,
                    is_final=True,  # Each batch is a complete segment
                    start_ms=start_ms,
                    end_ms=end_ms,
                    language=lang,
                    text=text,
                )

                # Deliver via callback if set (matches WLK pattern)
                if self._on_event:
                    await self._on_event(event)
                else:
                    yield event

            # Reset for next segment
            self._segment_id = str(uuid.uuid4())
            self._revision = 0
            self._buffer.clear()
            self._buffer_samples = 0
            self._base_offset += len(combined)

    async def flush(self) -> AsyncIterator[TranscriptEvent]:
        """Flush remaining buffer."""
        if self._buffer_samples > 0 and self._model:
            async for event in self.process_audio(np.array([], dtype=np.float32), 0):
                yield event
            # Force process remaining
            if self._buffer_samples >= SAMPLE_RATE // 2:  # at least 0.5s
                self._buffer_threshold = 0  # force trigger
                dummy = np.zeros(0, dtype=np.float32)
                async for event in self.process_audio(dummy, 0):
                    yield event
                self._buffer_threshold = int(2.0 * SAMPLE_RATE)

    async def process_audio_bytes(self, pcm_s16le: bytes) -> None:
        """Accept raw s16le 16kHz PCM bytes (matches WLK backend interface)."""
        audio = np.frombuffer(pcm_s16le, dtype=np.int16).astype(np.float32) / 32768.0
        async for _ in self.process_audio(audio, self._base_offset + self._buffer_samples):
            pass  # events delivered via callback
