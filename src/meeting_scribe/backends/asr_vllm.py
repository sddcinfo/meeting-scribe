"""vLLM ASR backend — Qwen3-ASR via OpenAI-compatible endpoint.

Primary ASR backend. Sends audio buffers to a vLLM-served Qwen3-ASR
model via HTTP. Superior Japanese quality.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable

import httpx
import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from meeting_scribe.backends.asr_filters import (
    _is_hallucination,
    _parse_qwen3_asr_response,
)
from meeting_scribe.backends.base import ASRBackend
from meeting_scribe.models import TranscriptEvent

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
VAD_ENERGY_THRESHOLD = 0.005


class VllmASRBackend(ASRBackend):
    """Qwen3-ASR via vLLM OpenAI-compatible endpoint.

    Buffers 4s of audio, encodes as WAV, sends to vLLM for transcription.
    Uses the same hallucination filtering and language detection as
    the local Qwen3-ASR backend.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8003",
        # Live-meeting ASR finalization cadence. Changed 2026-04-15 from
        # 1.5s → 3.5s after observing three linked pathologies on the
        # 2026-04-15 ja/en meeting:
        #   1. Translation quality suffered because 1.5s of audio is
        #      rarely a complete thought — the model got `'はい'`,
        #      `'うん'`, `'あの'` fragments with no context.
        #   2. TTS request rate exceeded the 2-container × 1-slot pool
        #      capacity; requests queued ~4s inside the container,
        #      bursting past the 5s synth deadline → mass dropouts.
        #   3. Short segments straddled speaker-change boundaries in
        #      the 16s diarization window, producing wrong/unstable
        #      speaker attribution even though diarization itself was
        #      fine. 3.5s segments align much better with the 4s
        #      diarization flush cadence.
        # Latency cost: +2s to on-screen transcript and first-audio.
        # Override via SCRIBE_ASR_BUFFER_SECONDS if you need the old
        # behavior (e.g. for a demo that needs sub-2s response).
        buffer_seconds: float = float(os.environ.get("SCRIBE_ASR_BUFFER_SECONDS", "3.5")),
        languages: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._model: str | None = None
        # Set of 1 or 2 language codes for the meeting. Length 1 = monolingual
        # (prompt biases ASR toward that one language; no translation hint).
        self._language_pair: set[str] = set(languages) if languages else set()
        # Cache the prompt so we don't rebuild it on every chunk. Invalidated
        # when _language_pair is reassigned (start_meeting does this).
        self._cached_prompt_pair: frozenset[str] | None = None
        self._cached_prompt: str = ""

        self._buffer: list[np.ndarray] = []
        self._buffer_samples = 0
        self._buffer_threshold = int(buffer_seconds * SAMPLE_RATE)
        self._segment_id: str | None = None
        self._revision = 0
        # Absolute sample offset (in the meeting's audio file) of the
        # FIRST sample currently sitting in the buffer. Set by the caller
        # on every process_audio_bytes call — the audio file is the
        # source of truth for timestamps, NOT an internal counter, so
        # server restart / meeting resume can't desync the transcript
        # from the audio it belongs to.
        self._buffer_start_sample: int = 0
        # Back-compat: old callers may still read _base_offset. Keep as
        # a mirror of the last-seen buffer-start sample. The backend no
        # longer USES it to stamp events.
        self._base_offset = 0
        self._on_event: Callable[[TranscriptEvent], Awaitable[None]] | None = None
        self.last_audio_chunk: np.ndarray | None = None
        # _last_response_time is None until the FIRST successful response.
        # This prevents the watchdog from firing on the first chunk of a new
        # meeting just because the server has been idle between meetings.
        self._last_response_time: float | None = None
        self._watchdog_timeout: float = 10.0  # seconds
        # Minimum buffer before the watchdog may flush — guards against
        # flushing tiny ~4000-sample chunks that ASR can't reliably transcribe
        self._min_watchdog_samples: int = int(0.8 * SAMPLE_RATE)  # 0.8s
        # Monotonic wall-clock ts at meeting start. Set by the caller when a
        # new meeting starts (server.py start_meeting). Used to compute
        # event.utterance_end_at = audio_wall_at_start + end_ms/1000, the
        # authoritative origin for the TTS speech-end SLA. When None, the
        # backend raises on event emission rather than silently using a wrong
        # origin — [P1-2-i5] strict policy.
        self.audio_wall_at_start: float | None = None

    def set_event_callback(self, fn: Callable[[TranscriptEvent], Awaitable[None]]) -> None:
        self._on_event = fn

    def _build_system_prompt(self) -> str:
        """Build the ASR system prompt with the meeting's expected languages.

        Without naming the languages explicitly Qwen3-ASR over-confidently
        labels Germanic-family speech (Dutch, German, sometimes Norwegian)
        as English. Spelling out the exact pair gives it a strong prior.
        """
        pair = frozenset(self._language_pair)
        if pair == self._cached_prompt_pair and self._cached_prompt:
            return self._cached_prompt

        from meeting_scribe.languages import LANGUAGE_REGISTRY

        if not pair:
            self._cached_prompt = (
                "Transcribe the audio in the original spoken language. "
                "Do not translate."
            )
        else:
            # Map each ISO code to its English name (e.g. "nl" → "Dutch").
            # Falls back to upper-cased code if the language registry
            # doesn't have it.
            names: list[str] = []
            for code in sorted(pair):
                lang = LANGUAGE_REGISTRY.get(code)
                if lang is not None:
                    names.append(f"{lang.name} ({code})")
                else:
                    names.append(code.upper())
            joined = " or ".join(names) if len(names) <= 2 else ", ".join(names)
            self._cached_prompt = (
                f"Transcribe the audio in the original spoken language. "
                f"The speaker is using {joined}. "
                f"Do NOT translate. If the speech is in one of these "
                f"languages, output it verbatim in that language."
            )
        self._cached_prompt_pair = pair
        return self._cached_prompt

    def set_languages(self, languages: list[str] | tuple[str, ...]) -> None:
        """Update the meeting's languages (1 = monolingual, 2 = bilingual pair)
        and invalidate the prompt cache.
        """
        self._language_pair = set(languages)
        self._cached_prompt_pair = None
        self._cached_prompt = ""

    async def start(self) -> None:
        """Connect to vLLM endpoint and detect model."""
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=30.0)
        # Reset watchdog state on every start — this prevents the watchdog from
        # firing on the first chunk of a new meeting when the backend has been
        # idle between meetings.
        self._last_response_time = None
        self._buffer.clear()
        self._buffer_samples = 0
        self._segment_id = None
        self._revision = 0

        # Health check
        resp = await self._client.get("/health")
        resp.raise_for_status()

        # Auto-detect model
        resp = await self._client.get("/v1/models")
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models:
            self._model = models[0]["id"]
        else:
            msg = "No models available at vLLM ASR endpoint"
            raise RuntimeError(msg)

        logger.info("vLLM ASR connected: model=%s", self._model)

    async def stop(self) -> None:
        """Release HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._buffer.clear()
        self._buffer_samples = 0
        self._last_response_time = None  # so next meeting starts clean

    async def process_audio(
        self,
        audio: np.ndarray,
        sample_offset: int,
        sample_rate: int = 16000,
    ) -> AsyncIterator[TranscriptEvent]:
        """Buffer audio and transcribe when threshold is reached.

        ``sample_offset`` is the absolute sample index of the FIRST
        sample in ``audio`` within the meeting's audio file. Events
        emitted from this call are stamped with (start, end) derived
        from the first sample of the buffer, not an internal counter.
        """
        if self._client is None:
            return

        if self._segment_id is None:
            self._segment_id = str(uuid.uuid4())
            self._revision = 0

        # First append to an empty buffer anchors the buffer's start
        # sample to where THIS chunk lives in the audio file.
        if self._buffer_samples == 0:
            self._buffer_start_sample = sample_offset
            self._base_offset = sample_offset  # back-compat mirror

        self._buffer.append(audio)
        self._buffer_samples += len(audio)

        # Watchdog: force flush if no ASR response within timeout AND we have
        # enough buffered audio for ASR to produce meaningful output.
        # Guards against:
        #   - Firing on the first chunk of a new meeting (last_response_time is None)
        #   - Flushing tiny ~4000-sample chunks that ASR can't reliably transcribe
        watchdog_triggered = (
            self._last_response_time is not None
            and self._buffer_samples >= self._min_watchdog_samples
            and self._buffer_samples < self._buffer_threshold
            and time.monotonic() - self._last_response_time > self._watchdog_timeout
        )
        if watchdog_triggered:
            logger.warning(
                "ASR watchdog: no response in %.0fs, forcing flush (%d samples)",
                self._watchdog_timeout,
                self._buffer_samples,
            )

        if self._buffer_samples >= self._buffer_threshold or watchdog_triggered:
            combined = np.concatenate(self._buffer)
            rms = float(np.sqrt(np.mean(combined**2)))

            if rms < VAD_ENERGY_THRESHOLD:
                self._buffer.clear()
                self._buffer_samples = 0
                if self._segment_id and self._revision > 0:
                    self._segment_id = None
                    self._revision = 0
                return

            # Encode audio as WAV in memory
            wav_buf = io.BytesIO()
            sf.write(wav_buf, combined, SAMPLE_RATE, format="WAV")
            wav_bytes = wav_buf.getvalue()
            audio_b64 = base64.b64encode(wav_bytes).decode()

            # Send to vLLM via OpenAI-compatible chat completions.
            # Qwen3-ASR auto-detects language from the audio and returns
            # "language X<asr_text>Y" format. The system prompt names the
            # MEETING's expected languages so the model has a strong prior —
            # without this, Dutch was being transcribed as English (similar
            # phonemes for many words; Qwen3-ASR's English bias wins by
            # default). The list of names is built per-meeting from the
            # active language_pair so any pair the user picks gets the hint.
            try:
                system_prompt = self._build_system_prompt()
                resp = await self._client.post(
                    "/v1/chat/completions",
                    json={
                        "model": self._model,
                        "messages": [
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_audio",
                                        "input_audio": {
                                            "data": audio_b64,
                                            "format": "wav",
                                        },
                                    },
                                ],
                            },
                        ],
                        "max_tokens": 512,
                        "temperature": 0.0,
                        # Live ASR is the tightest-SLA path on the Omni-consolidation
                        # stack (user hears silence otherwise). vLLM priority: lower
                        # = earlier. See plan `vLLM Consolidation + Omni Spike`.
                        "priority": -20,
                    },
                )
                resp.raise_for_status()
                result = resp.json()
                raw = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

                # Parse Qwen3-ASR response format: "language English<asr_text>actual text"
                text, lang = _parse_qwen3_asr_response(raw)

                # Second opinion from lingua, constrained to the meeting's
                # language pair. Catches the Germanic→English / Romance→
                # English bias that survived even after the system-prompt
                # naming-the-languages fix. Only overrides when lingua is
                # confident; otherwise trusts the ASR tag.
                if text and self._language_pair:
                    try:
                        from meeting_scribe.language_correction import (
                            correct_segment_language,
                        )

                        corrected = correct_segment_language(
                            text, lang, self._language_pair
                        )
                        if corrected != lang:
                            lang = corrected
                    except Exception:
                        logger.debug("lingua post-correction failed", exc_info=True)

                # Constrain to meeting language pair — drop segments in wrong languages
                if self._language_pair and lang not in self._language_pair:
                    # ASR confidently said this is a different language — drop it
                    logger.debug(
                        "Dropping %s segment (meeting pair: %s): '%s'",
                        lang,
                        self._language_pair,
                        text[:40] if text else "",
                    )
                    text = ""
            except (httpx.HTTPError, KeyError, IndexError) as e:
                logger.warning("vLLM ASR request failed: %s", e)
                text = ""
                lang = "unknown"

            # Filter hallucinations
            if text and (len(text) > 200 or _is_hallucination(text)):
                logger.debug("Filtered hallucination: '%s'", text[:40])
                text = ""

            self._last_response_time = time.monotonic()

            if text:
                self._revision += 1
                self.last_audio_chunk = combined
                start_ms = int(self._buffer_start_sample / SAMPLE_RATE * 1000)
                end_ms = start_ms + int(len(combined) / SAMPLE_RATE * 1000)

                # utterance_end_at is the authoritative TTS SLA origin
                # (audio-end wall clock, not emission time). Computed from
                # the per-meeting audio_wall_at_start set by start_meeting,
                # plus the meeting-relative end_ms. See [P1-1-i1].
                utterance_end_at: float | None = None
                if self.audio_wall_at_start is not None:
                    utterance_end_at = self.audio_wall_at_start + end_ms / 1000.0
                else:
                    logger.warning(
                        "ASR emit without audio_wall_at_start — utterance_end_at "
                        "will be None and TTS will refuse this segment. This is a "
                        "server init bug (start_meeting must set audio_wall_at_start)."
                    )

                event = TranscriptEvent(
                    segment_id=self._segment_id,
                    revision=self._revision,
                    is_final=True,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    language=lang,
                    text=text,
                    utterance_end_at=utterance_end_at,
                )

                if self._on_event:
                    await self._on_event(event)
                else:
                    yield event

            self._segment_id = str(uuid.uuid4())
            self._revision = 0
            self._buffer.clear()
            self._buffer_samples = 0
            # Next buffer starts where this one ended. The caller can
            # override by passing a fresh sample_offset on the next
            # process_audio_bytes call (e.g. after a mic gap).
            self._buffer_start_sample += len(combined)
            self._base_offset = self._buffer_start_sample

    async def flush(self) -> AsyncIterator[TranscriptEvent]:
        """Flush buffered audio."""
        if self._buffer_samples >= SAMPLE_RATE // 2 and self._client:
            self._buffer_threshold = 0
            async for event in self.process_audio(np.array([], dtype=np.float32), 0):
                yield event
            self._buffer_threshold = int(4.0 * SAMPLE_RATE)

    async def process_audio_bytes(self, pcm_s16le: bytes, sample_offset: int | None = None) -> None:
        """Process raw s16le PCM bytes.

        ``sample_offset`` is the absolute sample index of the FIRST
        sample in ``pcm_s16le`` within the meeting's audio file. When
        None, the backend falls back to its running internal counter —
        but callers that own a meeting-wide audio file (the live
        server) MUST pass the real offset so transcript alignment
        survives server restarts and meeting resume.
        """
        audio = np.frombuffer(pcm_s16le, dtype=np.int16).astype(np.float32) / 32768.0
        if sample_offset is None:
            sample_offset = self._buffer_start_sample + self._buffer_samples
        async for _ in self.process_audio(audio, sample_offset):
            pass
