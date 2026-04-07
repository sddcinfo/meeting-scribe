"""Qwen3-ASR backend — high-quality multilingual ASR via MLX.

Uses mlx-qwen3-asr for Apple Silicon optimized inference.
Dramatically better than Whisper for Japanese (perfect transcription).
52 languages supported with 97.9% language ID accuracy.

RTF ~0.06-0.25 on M4 Max (4-16x real-time).
Memory: ~1.2GB.
"""

from __future__ import annotations

import logging
import re
import tempfile
import uuid
from collections import Counter
from collections.abc import AsyncIterator

import numpy as np
import soundfile as sf

from meeting_scribe.backends.base import ASRBackend
from meeting_scribe.models import TranscriptEvent

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
VAD_ENERGY_THRESHOLD = 0.005

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


def _is_hallucination(text: str) -> bool:
    lower = text.lower().strip()
    for p in _HALLUCINATIONS:
        if p in lower:
            return True
    words = lower.split()
    if len(words) >= 3 and words[0] == words[1] == words[2]:
        return True
    stripped = lower.replace(" ", "").replace(",", "").replace(".", "")
    if len(stripped) >= 4:
        _most, count = Counter(stripped).most_common(1)[0]
        if count / len(stripped) > 0.5:
            return True
    return any(len(word) > 60 for word in text.split())


def _normalize_language(lang: str) -> str:
    """Normalize Qwen3-ASR language names to ja/en only.

    Qwen3-ASR returns full language names like 'Japanese', 'English',
    'Chinese', 'Cantonese', etc. We only support ja/en — everything
    else maps to 'unknown' and gets classified by text content.
    """
    lang = (lang or "").lower().strip()
    if lang in ("japanese", "ja", "jpn"):
        return "ja"
    if lang in ("english", "en", "eng"):
        return "en"
    # Chinese/Korean/etc. are NOT Japanese — don't let CJK heuristic misclassify
    if lang in (
        "chinese",
        "mandarin",
        "cantonese",
        "zh",
        "zho",
        "cmn",
        "yue",
        "korean",
        "ko",
        "kor",
        "thai",
        "th",
        "tha",
        "vietnamese",
        "vi",
    ):
        return "unknown"
    return "unknown"


class Qwen3ASRBackend(ASRBackend):
    """Qwen3-ASR-0.6B via MLX — best quality for JA+EN.

    Accumulates 4s of audio, writes to temp wav, transcribes.
    Perfect Japanese with correct language detection.
    """

    def __init__(self, buffer_seconds: float = 4.0) -> None:
        self._transcribe = None
        self._buffer: list[np.ndarray] = []
        self._buffer_samples = 0
        self._buffer_threshold = int(buffer_seconds * SAMPLE_RATE)
        self._segment_id: str | None = None
        self._revision = 0
        self._base_offset = 0
        self._on_event = None
        self.last_audio_chunk: np.ndarray | None = None  # available for speaker matching

    def set_event_callback(self, fn) -> None:
        self._on_event = fn

    async def start(self) -> None:
        from mlx_qwen3_asr import transcribe

        logger.info("Loading Qwen3-ASR-0.6B (MLX)...")
        # Warm up with silence
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
            transcribe(f.name)
        self._transcribe = transcribe
        logger.info("Qwen3-ASR loaded")

    async def stop(self) -> None:
        self._transcribe = None
        self._buffer.clear()
        self._buffer_samples = 0

    async def process_audio(
        self,
        audio: np.ndarray,
        sample_offset: int,
        sample_rate: int = 16000,
    ) -> AsyncIterator[TranscriptEvent]:
        if self._transcribe is None:
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
                self._buffer.clear()
                self._buffer_samples = 0
                if self._segment_id and self._revision > 0:
                    self._segment_id = None
                    self._revision = 0
                return

            # Write to temp wav for Qwen3-ASR
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, combined, SAMPLE_RATE)
                result = self._transcribe(f.name)

            text = (result.text or "").strip()
            raw_lang = getattr(result, "language", "") or ""
            lang = _normalize_language(raw_lang)

            # If model detected a non-JA/EN language, discard — it's likely
            # hallucinating from noise (Chinese, Korean, Thai, etc.)
            if lang == "unknown" and raw_lang.lower().strip() not in ("", "unknown"):
                logger.debug("Discarded non-JA/EN segment (detected %s): '%s'", raw_lang, text[:40])
                text = ""

            # If language still unknown but we have text, use CJK heuristic
            if lang == "unknown" and text:
                cjk = len(re.findall(r"[\u3000-\u9fff\uff00-\uffef]", text))
                total = len(text.replace(" ", ""))
                if total > 0:
                    ratio = cjk / total
                    # Only classify as JA if predominantly hiragana/katakana (not just CJK)
                    kana = len(re.findall(r"[\u3040-\u30ff]", text))
                    if kana > 0 and ratio > 0.3:
                        lang = "ja"
                    elif ratio < 0.1:
                        lang = "en"
                    else:
                        # Ambiguous CJK — could be Chinese. Discard to be safe.
                        logger.debug("Discarded ambiguous CJK: '%s'", text[:40])
                        text = ""

            if text and (len(text) > 200 or _is_hallucination(text)):
                logger.debug("Filtered: '%s'", text[:40])
                text = ""

            if text:
                self._revision += 1
                self.last_audio_chunk = combined  # save for speaker matching
                start_ms = int(self._base_offset / SAMPLE_RATE * 1000)
                end_ms = start_ms + int(len(combined) / SAMPLE_RATE * 1000)

                event = TranscriptEvent(
                    segment_id=self._segment_id,
                    revision=self._revision,
                    is_final=True,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    language=lang,
                    text=text,
                )

                if self._on_event:
                    await self._on_event(event)
                else:
                    yield event

            self._segment_id = str(uuid.uuid4())
            self._revision = 0
            self._buffer.clear()
            self._buffer_samples = 0
            self._base_offset += len(combined)

    async def flush(self) -> AsyncIterator[TranscriptEvent]:
        if self._buffer_samples >= SAMPLE_RATE // 2 and self._transcribe:
            self._buffer_threshold = 0
            async for event in self.process_audio(np.array([], dtype=np.float32), 0):
                yield event
            self._buffer_threshold = int(4.0 * SAMPLE_RATE)

    async def process_audio_bytes(self, pcm_s16le: bytes) -> None:
        audio = np.frombuffer(pcm_s16le, dtype=np.int16).astype(np.float32) / 32768.0
        async for _ in self.process_audio(audio, self._base_offset + self._buffer_samples):
            pass
