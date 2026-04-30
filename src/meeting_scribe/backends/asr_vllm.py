"""vLLM ASR backend — Qwen3-ASR via OpenAI-compatible endpoint.

Primary ASR backend. Sends audio buffers to a vLLM-served Qwen3-ASR
model via HTTP. Superior Japanese quality.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

import httpx
import numpy as np

from meeting_scribe.runtime import state


# ── W6a — Recovery state machine ────────────────────────────────
#
# The 2026-04-30 ASR cascade exposed three coupled gaps:
#
#   1. Watchdog fired 40+ times with no escalation. Fixed in W6b.
#   2. Recovery, when it eventually happened (Docker auto-restart),
#      lost ~30s of audio because the backend treated the in-flight
#      request's failure as "no transcript for this segment, advance".
#      Recording.pcm kept growing the whole time, but ASR never
#      revisited that range. Fixed here.
#   3. Stale httpx requests can complete late, after the backend has
#      already been recovered + replayed. Without a generation guard
#      they emit duplicate / out-of-order transcripts. Fixed here
#      via `_recovery_generation`.

_RecoveryState = Literal["NORMAL", "RECOVERY_PENDING", "REPLAYING"]

# Defensive ceiling on `_submissions` size — see W6a plan note.
# At a worst-case sustained submission rate of ~3 Hz during failure,
# 500 entries covers ~3 minutes of complete inability to recover
# before any submissions get dropped.
_MAX_UNRESOLVED_SUBMISSIONS = 500

# Hard cap on total replay duration. If a single recovery's replay
# would take >120s of audio, log ERROR and skip the remainder —
# better degraded than dead.
_REPLAY_DURATION_CAP_S = 120.0


@dataclass
class InflightSubmission:
    """Tracks a single ASR submission's audio offsets + status, so
    after a watchdog escalation we can replay from the EARLIEST
    unresolved offset (not from the much-later escalation tail).

    Eviction policy:
    - status="complete" → REMOVED from `_submissions` immediately
      (successful submissions never accumulate).
    - status="failed" or "inflight" → KEPT until either a successful
      replay supersedes them (offset < replay_end_offset) or the
      defensive ceiling fires.
    """

    request_id: int
    audio_start_offset: int   # recording.pcm byte offset of first chunk
    audio_end_offset: int     # one past the last chunk
    submitted_at: float       # time.monotonic()
    status: Literal["inflight", "complete", "failed"]
    generation: int           # _recovery_generation at submission time
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

        # ── W6a — Recovery state machine ────────────────────────
        # Per-submission offset tracking + recovery-generation epoch.
        # See module docstring for the design rationale; W6b drives
        # state transitions via `_begin_recovery_pending()` and the
        # supervisor task in `runtime/recovery_supervisor.py`.
        self._submissions: list[InflightSubmission] = []
        self._next_request_id: int = 0
        self._recovery_state: _RecoveryState = "NORMAL"
        self._recovery_start_offset: int | None = None
        self._recovery_generation: int = 0
        # Asyncio event the supervisor awaits; set by _begin_recovery_pending.
        self._recovery_requested: asyncio.Event = asyncio.Event()
        self._inflight_tasks: set[asyncio.Task] = set()

        # ── W6b — Watchdog escalation counter ───────────────────
        # Increments on every watchdog fire (the dashboard tile shows
        # fires from the FIRST fire — see W5 watchdog_fires_total
        # contract). On reaching 3 consecutive fires (~30s of inference
        # hang), calls _begin_recovery_pending() to enter
        # RECOVERY_PENDING and signal the background supervisor.
        # Reset to 0 on every successful response.
        self._watchdog_consecutive_fires: int = 0
        # Threshold for transition. Configurable so tests can drive
        # the path with a smaller value, but prod stays at 3.
        self._watchdog_escalation_threshold: int = 3

    def set_event_callback(self, fn: Callable[[TranscriptEvent], Awaitable[None]]) -> None:
        self._on_event = fn

    # ── W6a — Recovery state machine helpers ────────────────────
    # All offset capture goes through `_begin_recovery_pending()`. Do
    # NOT inline `current_recording_pcm_offset()` on the watchdog
    # escalation path elsewhere — that's the iteration-3/4 bug the
    # plan explicitly forbids.

    def _begin_recovery_pending(self) -> None:
        """Single source of truth for offset capture on watchdog escalation.

        Replay must cover audio submitted to a hung backend during the
        multi-watchdog-fire window before escalation, so we capture
        the earliest unresolved submission's start offset, NOT the
        current tail of recording.pcm.

        Idempotent: if state is already RECOVERY_PENDING or REPLAYING
        the call is a no-op. This blocks stale-task-driven re-entry
        — a superseded httpx request whose timeout fires mid-recovery
        cannot bump the watchdog counter past 3 again and re-trigger
        this method (the response-path generation guard already
        ignores its result; this guard is the second line of defence).
        """
        if self._recovery_state != "NORMAL":
            return  # already in recovery; no-op

        # Bump epoch FIRST — older submissions become stale before
        # they have a chance to mutate any shared state.
        self._recovery_generation += 1

        # Best-effort cancel of in-flight tasks. The generation guard
        # remains the safety net regardless of cancellation.
        for task in list(self._inflight_tasks):
            if not task.done():
                task.cancel()

        unresolved = [
            s for s in self._submissions
            if s.status in ("inflight", "failed")
        ]
        if unresolved:
            self._recovery_start_offset = min(
                s.audio_start_offset for s in unresolved
            )
        else:
            # Defensive fallback. Shouldn't happen because watchdog
            # fires imply timed-out submissions exist; if it does we
            # may have lost some audio — log loudly.
            logger.warning(
                "ASR recovery escalation with empty in-flight submissions "
                "deque; falling back to current recording offset, "
                "transcript may have a hole"
            )
            self._recovery_start_offset = state.current_recording_pcm_offset()

        self._recovery_state = "RECOVERY_PENDING"
        try:
            state.metrics.watchdog_escalations_total += 1
        except AttributeError:
            pass  # metrics not yet wired (warmup); harmless
        self._recovery_requested.set()

    def _track_submission_start(
        self, audio_start_offset: int, audio_end_offset: int
    ) -> InflightSubmission:
        """Record an outgoing transcribe request. Caller is
        responsible for calling `_track_submission_complete` /
        `_track_submission_failed` exactly once per submission (with
        the generation guard built in). Enforces the
        _MAX_UNRESOLVED_SUBMISSIONS ceiling defensively."""
        self._next_request_id += 1
        sub = InflightSubmission(
            request_id=self._next_request_id,
            audio_start_offset=audio_start_offset,
            audio_end_offset=audio_end_offset,
            submitted_at=time.monotonic(),
            status="inflight",
            generation=self._recovery_generation,
        )
        self._submissions.append(sub)

        # Defensive ceiling — drop oldest "failed" entry if exceeded.
        # Never drop "inflight" (those are still pending real outcome).
        if len(self._submissions) > _MAX_UNRESOLVED_SUBMISSIONS:
            for i, s in enumerate(self._submissions):
                if s.status == "failed":
                    del self._submissions[i]
                    logger.error(
                        "ASR submission tracking exceeded %d unresolved "
                        "entries; dropping oldest failed entry "
                        "(request_id=%d offset=%d) — transcript may have "
                        "a hole",
                        _MAX_UNRESOLVED_SUBMISSIONS,
                        s.request_id,
                        s.audio_start_offset,
                    )
                    # Future: emit a structured `submission_tracking_overflow`
                    # WS event (W6b will own that wiring).
                    break

        return sub

    def _track_submission_complete(self, sub: InflightSubmission) -> bool:
        """Mark a submission complete. Returns True if the result
        should be processed (generation matches), False if it's a
        stale-generation response that must be silently discarded."""
        if sub.generation != self._recovery_generation:
            logger.debug(
                "ignoring stale response from generation %d (current=%d)",
                sub.generation,
                self._recovery_generation,
            )
            return False
        # Success → REMOVE the entry (do not let completed submissions
        # accumulate; the deque retains only inflight + failed).
        try:
            self._submissions.remove(sub)
        except ValueError:
            pass  # already evicted by replay; harmless
        return True

    def _track_submission_failed(
        self, sub: InflightSubmission, exc: BaseException
    ) -> bool:
        """Mark a submission failed. Returns True if the failure
        should be processed (generation matches), False if it's a
        stale-generation result that must be silently discarded.

        Failed entries are RETAINED in `_submissions` so the next
        recovery escalation knows the earliest unresolved offset.
        Replay (or the defensive ceiling) is what eventually evicts
        them."""
        if sub.generation != self._recovery_generation:
            logger.debug(
                "ignoring stale failure from generation %d (current=%d): %s",
                sub.generation,
                self._recovery_generation,
                exc,
            )
            return False
        sub.status = "failed"
        return True

    async def replay_until_caught_up(self, start_offset: int) -> int:
        """Replay recording.pcm from `start_offset` to the current
        write head, submitting each chunk through the production
        transcribe path. Returns the final replay-end offset.

        Bounded by `_REPLAY_DURATION_CAP_S` (120s of audio); past
        that, log ERROR + advance offset to live + return. Better
        degraded than dead.

        Caller (W6b supervisor) is responsible for setting state to
        REPLAYING before calling this and resetting to NORMAL after
        completion. The replay submissions themselves go through the
        normal `process_audio_bytes` path with `_is_replay=True` so
        the live-suppression check skips them."""
        offset = start_offset
        bytes_per_sec = 16000 * 2  # s16le @ 16 kHz
        replay_started_at = time.monotonic()
        # Read a 3.5s buffer's worth of audio per replay step — same
        # natural cadence as the normal flush threshold.
        chunk_bytes = int(3.5 * bytes_per_sec)
        chunk_bytes -= chunk_bytes % 2  # keep alignment

        meeting_dir = (
            state.current_meeting and state.storage.meeting_dir(state.current_meeting.meeting_id)
        )
        pcm_path = meeting_dir / "audio" / "recording.pcm" if meeting_dir else None
        if pcm_path is None or not pcm_path.exists():
            logger.warning(
                "replay_until_caught_up: no recording.pcm path; nothing to replay"
            )
            return offset

        with pcm_path.open("rb") as fh:
            while True:
                live_offset = state.current_recording_pcm_offset()
                if offset >= live_offset:
                    break  # caught up

                # Cap total replay duration.
                replayed_s = (offset - start_offset) / bytes_per_sec
                if replayed_s > _REPLAY_DURATION_CAP_S:
                    skipped_s = (live_offset - offset) / bytes_per_sec
                    logger.error(
                        "ASR replay duration exceeded %.0fs, skipping "
                        "%.1fs of audio (offset=%d → %d)",
                        _REPLAY_DURATION_CAP_S,
                        skipped_s,
                        offset,
                        live_offset,
                    )
                    offset = live_offset
                    break

                fh.seek(offset)
                read_n = min(chunk_bytes, live_offset - offset)
                pcm = fh.read(read_n)
                if not pcm:
                    break
                # Sample offset for transcript alignment is byte-offset
                # divided by 2 (s16le).
                sample_offset = offset // 2
                await self.process_audio_bytes(
                    pcm, sample_offset=sample_offset, _is_replay=True
                )
                offset += len(pcm)

        replay_end_offset = offset

        # Drop submissions superseded by replay (those whose
        # audio_start_offset falls in [start_offset, replay_end_offset)).
        self._submissions = [
            s for s in self._submissions
            if s.audio_start_offset >= replay_end_offset
        ]
        logger.info(
            "ASR replay caught up at offset=%d (replayed %d bytes, %.1fs, %.1fs wall)",
            replay_end_offset,
            replay_end_offset - start_offset,
            (replay_end_offset - start_offset) / bytes_per_sec,
            time.monotonic() - replay_started_at,
        )
        return replay_end_offset

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
                "Transcribe the audio in the original spoken language. Do not translate."
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
        *,
        _is_replay: bool = False,
    ) -> AsyncIterator[TranscriptEvent]:
        """Buffer audio and transcribe when threshold is reached.

        ``sample_offset`` is the absolute sample index of the FIRST
        sample in ``audio`` within the meeting's audio file. Events
        emitted from this call are stamped with (start, end) derived
        from the first sample of the buffer, not an internal counter.

        ``_is_replay`` is set by W6a's `replay_until_caught_up` so
        replay submissions bypass the live-suppression check during
        RECOVERY_PENDING / REPLAYING. Live audio still goes to
        recording.pcm via the unrelated audio_writer path; replay
        re-reads from there and feeds it through the recovered backend.
        """
        if self._client is None:
            return

        # ── W6a — Live suppression during recovery ─────────────
        # During RECOVERY_PENDING / REPLAYING, the inference endpoint
        # is presumed wedged or being recreated. Live audio chunks
        # still arrive via WS and still get appended to recording.pcm
        # (independent path), so the audio is preserved on disk.
        # Submitting them to the wedged endpoint would just generate
        # more failed submissions and pollute the deque. Reset the
        # in-memory chunk buffer and return early.
        if not _is_replay and self._recovery_state in ("RECOVERY_PENDING", "REPLAYING"):
            self._buffer.clear()
            self._buffer_samples = 0
            if self._segment_id is not None and self._revision > 0:
                self._segment_id = None
                self._revision = 0
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
            # ── W6b — Count + escalate ───────────────────────────
            # EVERY fire bumps the dashboard counter (W5 contract).
            # Consecutive fires accumulate independently; on reaching
            # the threshold, transition to RECOVERY_PENDING via the
            # single source of truth helper. The audio path returns
            # immediately after — recovery work runs in the
            # background supervisor task.
            self._watchdog_consecutive_fires += 1
            try:
                state.metrics.watchdog_fires_total += 1
                state.metrics._watchdog_fire_timestamps.append(time.monotonic())
            except AttributeError:
                pass  # metrics not yet wired (warmup); harmless

            if self._watchdog_consecutive_fires >= self._watchdog_escalation_threshold:
                self._begin_recovery_pending()

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
            _rtt_t0 = time.monotonic()
            # W6a: track this submission's audio offsets so a future
            # watchdog escalation can replay from the earliest
            # unresolved one. Bytes-per-sample = 2 for s16le.
            _audio_start_offset = self._buffer_start_sample * 2
            _audio_end_offset = _audio_start_offset + len(combined) * 2
            _submission = self._track_submission_start(
                _audio_start_offset, _audio_end_offset
            )
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
                        # Live ASR is the tightest-SLA path in the stack (user
                        # hears silence otherwise). vLLM priority: lower = earlier.
                        "priority": -20,
                    },
                )
                resp.raise_for_status()
                # W6a: stale-response guard. If recovery escalated
                # while this request was in flight, the response is
                # from an old generation — silently discard. The
                # supervisor's replay path will re-submit this audio
                # range under the current generation.
                if not self._track_submission_complete(_submission):
                    return
                # W6b: any successful response resets the consecutive
                # fire counter so a single hiccup doesn't accumulate
                # toward an unwarranted recovery escalation.
                self._watchdog_consecutive_fires = 0
                # W5: backend RTT histogram. Sampled only on successful
                # requests so failures (which dominate the early seconds
                # of a CUDA-wedge incident) don't pollute the percentile.
                try:
                    state.metrics.asr_request_rtt_ms.append(
                        (time.monotonic() - _rtt_t0) * 1000
                    )
                except AttributeError:
                    pass  # state.metrics not yet initialised at warmup time
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

                        corrected = correct_segment_language(text, lang, self._language_pair)
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
                # W6a: stale-failure guard. A late-arriving timeout
                # or HTTP error from a superseded generation must NOT
                # mutate state — the new generation's submission for
                # this audio range owns the outcome.
                if not self._track_submission_failed(_submission, e):
                    return
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

    async def process_audio_bytes(
        self,
        pcm_s16le: bytes,
        sample_offset: int | None = None,
        *,
        _is_replay: bool = False,
    ) -> None:
        """Process raw s16le PCM bytes.

        ``sample_offset`` is the absolute sample index of the FIRST
        sample in ``pcm_s16le`` within the meeting's audio file. When
        None, the backend falls back to its running internal counter —
        but callers that own a meeting-wide audio file (the live
        server) MUST pass the real offset so transcript alignment
        survives server restarts and meeting resume.

        ``_is_replay`` is forwarded to ``process_audio`` so W6a's
        replay path bypasses the live-suppression check during
        RECOVERY_PENDING / REPLAYING.
        """
        audio = np.frombuffer(pcm_s16le, dtype=np.int16).astype(np.float32) / 32768.0
        if sample_offset is None:
            sample_offset = self._buffer_start_sample + self._buffer_samples
        async for _ in self.process_audio(audio, sample_offset, _is_replay=_is_replay):
            pass
