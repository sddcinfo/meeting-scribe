"""WhisperLiveKit streaming ASR backend.

Wraps WhisperLiveKit's TranscriptionEngine + AudioProcessor behind our
ASRBackend ABC. Provides real streaming ASR with Silero VAD, SimulStreaming
(AlignAtt) policy, and optional diarization via Sortformer.

Platform-agnostic: macOS uses mlx-whisper backend, GB10 uses faster-whisper (CUDA).
Controlled by config.asr_backend_type.

Architecture note:
    WLK processes audio asynchronously — results arrive from a background
    generator, not synchronously from process_audio(). We use a callback
    pattern: set_event_callback(fn) registers a function that gets called
    whenever WLK produces new transcript events. The server registers its
    broadcast function as the callback.

Audio flow:
    Server resample (float32 16kHz) → s16le bytes → WLK process_audio()
    WLK results_formatter() → FrontData → _map_front_data → callback(TranscriptEvent)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING

import numpy as np

from meeting_scribe.backends.base import ASRBackend
from meeting_scribe.models import SpeakerAttribution, TranscriptEvent

if TYPE_CHECKING:
    from whisperlivekit import TranscriptionEngine
    from whisperlivekit.timed_objects import FrontData

logger = logging.getLogger(__name__)

# Default seconds of silence (no text OR buffer change) before finalizing.
# Configurable via SCRIBE_FINALIZE_IDLE_SECONDS env var.
_FINALIZE_IDLE_SECONDS = float(os.environ.get("SCRIBE_FINALIZE_IDLE_SECONDS", "2.0"))

# Regex for detecting predominantly CJK text (Japanese/Chinese)
_CJK_RE = re.compile(r"[\u3000-\u9fff\uff00-\uffef]")

# Known Whisper hallucination patterns
_HALLUCINATION_PATTERNS = [
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "ご視聴ありがとうございました",
    "ご覧いただきありがとうございます",
    "チャンネル登録",
    "字幕",
    "subtitles",
]


def _is_hallucination(text: str) -> bool:
    """Detect common Whisper hallucination patterns and repetition loops."""
    lower = text.lower().strip()
    if not lower:
        return False

    # Known hallucination phrases
    for pattern in _HALLUCINATION_PATTERNS:
        if pattern in lower:
            return True

    # Repetition detection: any word repeated 3+ times in a row
    words = lower.split()
    if len(words) >= 3:
        for i in range(len(words) - 2):
            if words[i] == words[i + 1] == words[i + 2]:
                return True

    # Same word makes up >60% of all words
    if len(words) >= 4:
        from collections import Counter

        most_common_word, count = Counter(words).most_common(1)[0]
        if count / len(words) > 0.6 and len(most_common_word) > 1:
            return True

    # Text is very short but non-speech (single repeated character)
    stripped = lower.replace(" ", "")
    return bool(len(stripped) >= 4 and len(set(stripped)) <= 2)


def _detect_language(text: str) -> str:
    """Heuristic language detection from text content.

    Only returns "ja" or "en" — those are the two languages we support.
    CJK characters → ja, otherwise → en.
    """
    if not text or len(text.strip()) < 2:
        return "unknown"
    cjk_count = len(_CJK_RE.findall(text))
    total = len(text.replace(" ", ""))
    if total == 0:
        return "unknown"
    cjk_ratio = cjk_count / total
    if cjk_ratio > 0.3:
        return "ja"  # Always ja, never zh — we only support ja/en
    return "en"


def _parse_time_seconds(t: float) -> int:
    """Convert WLK segment time (seconds as float) to milliseconds."""
    return int(t * 1000)


class WhisperLiveKitBackend(ASRBackend):
    """Streaming ASR via WhisperLiveKit.

    Lifecycle:
        start()              → loads TranscriptionEngine (model weights, Silero VAD)
        set_event_callback() → registers async callback for event delivery
        process_audio()      → lazily creates AudioProcessor, feeds s16le PCM
        flush()              → signals end-of-stream, waits for remaining events
        stop()               → cleans up processor and engine

    Events are delivered via the callback (not yielded from process_audio),
    because WLK produces results asynchronously from a background generator.
    """

    def __init__(
        self,
        *,
        model: str = "large-v3-turbo",
        backend: str = "auto",
        language: str = "ja",
        streaming_policy: str = "simulstreaming",
        diarization: bool = False,
        diarization_backend: str = "sortformer",
    ) -> None:
        self._model = model
        self._backend = backend
        self._language = language
        self._streaming_policy = streaming_policy
        self._diarization = diarization
        self._diarization_backend = diarization_backend

        self._engine: TranscriptionEngine | None = None
        self._processor = None  # AudioProcessor (lazy per-session)
        self._results_task: asyncio.Task | None = None
        self._on_event: Callable[[TranscriptEvent], Awaitable[None]] | None = None

        # Track state for FrontData → TranscriptEvent mapping
        self._line_segment_ids: list[str] = []
        self._line_texts: list[str] = []
        self._line_revisions: list[int] = []
        self._line_languages: list[str] = []
        self._line_finalized: list[bool] = []
        self._line_last_changed: list[float] = []
        # Buffer (partial uncommitted text, not yet in any line)
        self._buffer_segment_id: str | None = None
        self._buffer_revision = 0
        # Stale detection — reset session if no change for this many seconds
        self._stale_reset_seconds = 10.0
        self._last_any_output_time = 0.0
        self._last_fd_signature = ""

    def set_event_callback(self, fn: Callable[[TranscriptEvent], Awaitable[None]]) -> None:
        """Register an async callback for event delivery.

        Called from the background results consumer whenever WLK produces
        new transcript events. Must be set before process_audio().
        """
        self._on_event = fn

    async def start(self) -> None:
        """Load the TranscriptionEngine (model weights + VAD)."""
        from whisperlivekit import TranscriptionEngine, WhisperLiveKitConfig

        config = WhisperLiveKitConfig(
            model_size=self._model,
            backend=self._backend,
            backend_policy=self._streaming_policy,
            lan=self._language,
            diarization=self._diarization,
            diarization_backend=self._diarization_backend,
            vad=True,
            vac=True,
            pcm_input=True,
            transcription=True,
            confidence_validation=True,  # Fast-commit high-confidence tokens
            buffer_trimming="segment",
            buffer_trimming_sec=10.0,  # Force trim at 10s to prevent stale buffer
            min_chunk_size=0.5,
        )

        logger.info(
            "Loading WhisperLiveKit: model=%s backend=%s policy=%s lang=%s diarize=%s",
            self._model,
            self._backend,
            self._streaming_policy,
            self._language,
            self._diarization,
        )
        self._engine = TranscriptionEngine(config=config)
        logger.info("WhisperLiveKit engine loaded")

    async def stop(self) -> None:
        """Release all resources."""
        await self._destroy_session()
        self._engine = None

    async def process_audio(
        self,
        audio: np.ndarray,
        sample_offset: int,
        sample_rate: int = 16000,
    ) -> AsyncIterator[TranscriptEvent]:
        """Feed float32 audio to WLK (ABC compatibility). Converts to s16le."""
        if self._engine is None:
            return

        if self._processor is None:
            await self._create_session()

        # Convert float32 → s16le. No normalization — browser AGC handles levels.
        pcm_s16 = (audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        await self._processor.process_audio(pcm_s16)

        return
        yield  # ABC generator compatibility

    async def process_audio_bytes(self, pcm_s16le: bytes) -> None:
        """Feed raw s16le 16kHz PCM bytes directly to WLK.

        Preferred path — AudioWorklet sends s16le directly, no conversion needed.
        """
        if self._engine is None:
            return

        if self._processor is None:
            await self._create_session()

        await self._processor.process_audio(pcm_s16le)

    async def flush(self) -> AsyncIterator[TranscriptEvent]:
        """Signal end-of-stream and wait for remaining events."""
        if self._processor is None:
            return
            yield

        # Send empty bytes to signal EOS
        await self._processor.process_audio(b"")

        # Wait for WLK to finish processing
        if self._results_task:
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(self._results_task, timeout=10.0)

        await self._destroy_session()

    # ── Session lifecycle ──────────────────────────────────────

    async def _create_session(self) -> None:
        """Create a new AudioProcessor session."""
        from whisperlivekit import AudioProcessor

        self._processor = AudioProcessor(transcription_engine=self._engine)
        results_gen = await self._processor.create_tasks()

        # Background task: consume FrontData and push events via callback
        self._results_task = asyncio.create_task(
            self._consume_results(results_gen),
            name="wlk-results-consumer",
        )

        self._line_segment_ids = []
        self._line_texts = []
        self._line_revisions = []
        self._line_languages = []
        self._line_finalized = []
        self._line_last_changed = []
        self._buffer_segment_id = None
        self._buffer_revision = 0
        self._last_buffer_text = ""
        self._last_buffer_change = 0.0
        self._hallucination_detected = False
        self._last_any_output_time = time.monotonic()
        self._last_fd_signature = ""
        self._fd_count = 0
        logger.info("WLK AudioProcessor session created")

    async def _destroy_session(self) -> None:
        """Destroy the current AudioProcessor session."""
        if self._results_task and not self._results_task.done():
            self._results_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._results_task

        if self._processor:
            try:
                await self._processor.cleanup()
            except Exception as e:
                logger.warning("WLK cleanup error: %s", e)

        self._processor = None
        self._results_task = None
        logger.info("WLK AudioProcessor session destroyed")

    # ── Results pipeline ───────────────────────────────────────

    async def _consume_results(self, gen) -> None:
        """Background task: read FrontData from WLK, map to events, push via callback."""
        try:
            async for front_data in gen:
                # Stale detection: check if FrontData has changed
                lines_sig = "|".join(
                    getattr(ln, "text", "")[:30] for ln in (front_data.lines or [])
                )
                sig = f"{lines_sig}:{(front_data.buffer_transcription or '')[:50]}"
                if sig != self._last_fd_signature:
                    self._last_fd_signature = sig
                    self._last_any_output_time = time.monotonic()
                elif self._last_any_output_time > 0:
                    stale_seconds = time.monotonic() - self._last_any_output_time
                    if stale_seconds > self._stale_reset_seconds:
                        logger.warning(
                            "WLK stale for %.0fs — resetting session",
                            stale_seconds,
                        )
                        # Finalize anything we have, then reset
                        if self._on_event:
                            for event in self.finalize_all_lines():
                                with contextlib.suppress(Exception):
                                    await self._on_event(event)
                        await self._destroy_session()
                        await self._create_session()
                        return  # New consumer will be spawned by _create_session

                events = self._map_front_data(front_data)

                # Check if hallucination was detected — reset session
                if getattr(self, "_hallucination_detected", False):
                    self._hallucination_detected = False
                    logger.warning("Resetting WLK session due to hallucination")
                    if self._on_event:
                        for event in self.finalize_all_lines():
                            with contextlib.suppress(Exception):
                                await self._on_event(event)
                    await self._destroy_session()
                    await self._create_session()
                    return  # New consumer spawned

                if self._on_event:
                    for event in events:
                        try:
                            await self._on_event(event)
                        except Exception:
                            logger.exception("Event callback error for seg=%s", event.segment_id)

            # Generator ended (WLK finished) — finalize all lines
            if self._on_event:
                for event in self.finalize_all_lines():
                    try:
                        await self._on_event(event)
                    except Exception:
                        logger.exception("Finalize callback error for seg=%s", event.segment_id)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("WLK results consumer error: %s", e)

    def _map_front_data(self, fd: FrontData) -> list[TranscriptEvent]:
        """Convert a WLK FrontData into TranscriptEvents.

        Handles two finalization triggers:
        1. New line N+1 appears → line N is final (WLK detected silence/speaker change)
        2. Line text unchanged for _FINALIZE_IDLE_SECONDS → line is final (idle timeout)

        Language is detected from text content (CJK ratio heuristic) since
        WLK doesn't always populate detected_language on segments.
        """
        events: list[TranscriptEvent] = []

        if fd.status == "error":
            logger.error("WLK error: %s", fd.error)
            return events

        now = time.monotonic()
        lines = fd.lines or []
        buf = (fd.buffer_transcription or "").strip()

        # Log every 20th FrontData for debugging
        self._fd_count = getattr(self, "_fd_count", 0) + 1
        if self._fd_count % 20 == 1 or lines or buf:
            line_texts = [getattr(ln, "text", "")[:30] for ln in lines] if lines else []
            logger.info(
                "FD#%d: status=%s lines=%d(%s) buf='%s'",
                self._fd_count,
                fd.status,
                len(lines),
                line_texts,
                buf[:40] if buf else "",
            )

        # Finalize lines that have a successor (WLK added a new line)
        for i in range(len(lines) - 1):
            if (
                i < len(self._line_finalized)
                and not self._line_finalized[i]
                and self._line_texts[i]
            ):
                events.extend(self._finalize_line(i, lines[i]))

        # Finalize lines idle for too long.
        # "idle" = no change to EITHER the committed line text OR the buffer text.
        # This prevents premature finalization while user is still speaking.
        buffer_text = (fd.buffer_transcription or "").strip()
        if buffer_text != getattr(self, "_last_buffer_text", ""):
            self._last_buffer_change = now
            self._last_buffer_text = buffer_text
        last_any_change = max(
            getattr(self, "_last_buffer_change", 0),
            max((t for t in self._line_last_changed), default=0),
        )
        idle_seconds = now - last_any_change if last_any_change > 0 else 0

        for i in range(len(self._line_finalized)):
            if (
                not self._line_finalized[i]
                and self._line_texts[i]
                and idle_seconds > _FINALIZE_IDLE_SECONDS
            ):
                # Merge buffer into the line before finalizing
                if buffer_text:
                    full_text = f"{self._line_texts[i]} {buffer_text}".strip()
                    self._line_texts[i] = full_text
                    self._line_languages[i] = _detect_language(full_text)
                    buffer_text = ""  # Consumed
                    logger.info("Finalize with buffer merge: '%s'", self._line_texts[i][:60])

                line = lines[i] if i < len(lines) else None
                events.extend(self._finalize_line(i, line))
                logger.info(
                    "Segment finalized after %.1fs idle: '%s'",
                    idle_seconds,
                    self._line_texts[i][:60],
                )

        # Process all lines for text changes
        for i, line in enumerate(lines):
            text = (line.text or "").strip()
            if not text or _is_hallucination(text):
                continue

            # New line — initialize tracking
            if i >= len(self._line_segment_ids):
                self._line_segment_ids.append(str(uuid.uuid4()))
                self._line_texts.append("")
                self._line_revisions.append(0)
                self._line_languages.append("unknown")
                self._line_finalized.append(False)
                self._line_last_changed.append(now)

            if self._line_finalized[i]:
                continue

            if text == self._line_texts[i]:
                continue

            self._line_texts[i] = text
            self._line_revisions[i] += 1
            self._line_last_changed[i] = now

            # Language: use WLK detected_language but normalize to ja/en only
            raw_lang = (
                line.detected_language
                if hasattr(line, "detected_language") and line.detected_language
                else None
            )
            if raw_lang and raw_lang not in ("ja", "en"):
                raw_lang = None  # WLK sometimes detects zh/ko — normalize
            self._line_languages[i] = raw_lang or _detect_language(text)

            logger.debug(
                "Line %d rev %d: '%s' [%s]",
                i,
                self._line_revisions[i],
                text[:40],
                self._line_languages[i],
            )

            events.append(
                TranscriptEvent(
                    segment_id=self._line_segment_ids[i],
                    revision=self._line_revisions[i],
                    is_final=False,
                    start_ms=_parse_time_seconds(line.start),
                    end_ms=_parse_time_seconds(line.end),
                    language=self._line_languages[i],
                    text=text,
                    speakers=self._speakers_for(line),
                )
            )

        # Buffer text — primary content stream for both policies.
        # Filter hallucinations from buffer — if detected, flag for session reset
        if buffer_text and _is_hallucination(buffer_text):
            logger.warning("Hallucination detected, flagging reset: '%s'", buffer_text[:40])
            buffer_text = ""
            self._hallucination_detected = True

        # Emit as partial while changing, finalize when stable for _FINALIZE_IDLE_SECONDS.
        if buffer_text:
            if not self._buffer_segment_id:
                self._buffer_segment_id = str(uuid.uuid4())
                self._buffer_revision = 0

            if buffer_text != self._last_buffer_text:
                # Text changed — emit partial
                self._buffer_revision += 1
                events.append(
                    TranscriptEvent(
                        segment_id=self._buffer_segment_id,
                        revision=self._buffer_revision,
                        is_final=False,
                        language=_detect_language(buffer_text),
                        text=buffer_text,
                    )
                )
            elif idle_seconds > _FINALIZE_IDLE_SECONDS and len(buffer_text) >= 2:
                # Buffer text unchanged for idle period — finalize it
                lang = _detect_language(buffer_text)
                events.append(
                    TranscriptEvent(
                        segment_id=self._buffer_segment_id,
                        revision=self._buffer_revision,
                        is_final=True,
                        language=lang,
                        text=buffer_text,
                    )
                )
                logger.info(
                    "Buffer finalized (stable %.1fs): [%s] '%s'",
                    idle_seconds,
                    lang,
                    buffer_text[:60],
                )
                # Reset for next segment
                self._buffer_segment_id = str(uuid.uuid4())
                self._buffer_revision = 0
                self._last_buffer_text = ""
                buffer_text = ""  # consumed

        elif self._buffer_segment_id and self._last_buffer_text:
            # Buffer cleared — finalize immediately (silence gap / text committed to lines)
            lang = _detect_language(self._last_buffer_text)
            events.append(
                TranscriptEvent(
                    segment_id=self._buffer_segment_id,
                    revision=self._buffer_revision,
                    is_final=True,
                    language=lang,
                    text=self._last_buffer_text,
                )
            )
            logger.info("Buffer cleared → final: [%s] '%s'", lang, self._last_buffer_text[:60])
            self._buffer_segment_id = None
            self._buffer_revision = 0
            self._last_buffer_text = ""

        return events

    def _finalize_line(self, i: int, line=None) -> list[TranscriptEvent]:
        """Emit a final event for line i. Reuses current revision (no double-count)."""
        self._line_finalized[i] = True
        start_ms = _parse_time_seconds(line.start) if line and hasattr(line, "start") else 0
        end_ms = _parse_time_seconds(line.end) if line and hasattr(line, "end") else 0
        return [
            TranscriptEvent(
                segment_id=self._line_segment_ids[i],
                revision=self._line_revisions[i],  # Same revision, just is_final=True
                is_final=True,
                start_ms=start_ms,
                end_ms=end_ms,
                language=self._line_languages[i],
                text=self._line_texts[i],
                speakers=self._speakers_for(line) if line else [],
            )
        ]

    @staticmethod
    def _speakers_for(line) -> list[SpeakerAttribution]:
        """Extract speaker attribution from a WLK Segment."""
        if line and hasattr(line, "speaker") and line.speaker is not None:
            try:
                sid = int(line.speaker)
                if sid >= 0:
                    return [SpeakerAttribution(cluster_id=sid, source="cluster_only")]
            except (ValueError, TypeError):
                pass
        return []

    def finalize_all_lines(self) -> list[TranscriptEvent]:
        """Mark all unfinalized lines and buffer as final. Called at end of stream."""
        events: list[TranscriptEvent] = []
        buffer = getattr(self, "_last_buffer_text", "").strip()

        for i in range(len(self._line_segment_ids)):
            if self._line_texts[i] and not self._line_finalized[i]:
                if buffer:
                    self._line_texts[i] = f"{self._line_texts[i]} {buffer}".strip()
                    self._line_languages[i] = _detect_language(self._line_texts[i])
                    buffer = ""
                events.extend(self._finalize_line(i))

        # Finalize any remaining buffer text as its own segment
        if buffer and self._buffer_segment_id:
            events.append(
                TranscriptEvent(
                    segment_id=self._buffer_segment_id,
                    revision=self._buffer_revision,
                    is_final=True,
                    language=_detect_language(buffer),
                    text=buffer,
                )
            )
            self._buffer_segment_id = None
            self._last_buffer_text = ""

        return events
