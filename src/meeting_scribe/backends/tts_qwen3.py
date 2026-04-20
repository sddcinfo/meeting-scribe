"""Qwen3-TTS voice cloning backend — vllm-omni edition.

Single-container, continuous-batched vllm-omni replaces the prior two-replica
faster-qwen3-tts pool. Wire format: OpenAI-compatible /v1/audio/speech with
stream=true response_format=pcm (raw int16 @ 24 kHz mono).

Two synthesis modes:
  - Studio: named speaker (aiden, vivian, ono_anna, sohee, uncle_fu).
  - Cloned: inline ref_audio=data:audio/wav;base64,… per request; stateless.

Cloned-mode fallback: when a cached voice is missing or below the quality
threshold, deterministically fall back to studio_voice_for(lang) — no audio
is dropped, no silent error.
"""

from __future__ import annotations

import base64
import io
import logging
import time
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
import numpy as np

from meeting_scribe.backends.base import TTSBackend
from meeting_scribe.backends.voice_quality import score_reference

logger = logging.getLogger(__name__)

# Progressive-reference tunables. Kept module-level so tests can monkey-patch.
_MIN_UPDATE_GAP_S = 60.0
_UPGRADE_MARGIN = 0.05
_FINALIZE_THRESHOLD = 0.85

# Inline ref_audio bounds (P1 contract — keep requests bounded to avoid 413/timeout
# under concurrency). 6 s @ 24 kHz mono int16 = 288 KiB raw; +WAV header +base64
# yields ~400 KiB data URI.
_REF_AUDIO_MAX_SECONDS = 6.0
_REF_AUDIO_SAMPLE_RATE = 24_000
_REF_AUDIO_MAX_BYTES = 400 * 1024  # data-URI byte ceiling

# vllm-omni streams raw int16 PCM at 24 kHz.
_TTS_SAMPLE_RATE = 24_000


class VoiceRefTooLargeError(ValueError):
    """Raised when a would-be ref_audio data URI exceeds _REF_AUDIO_MAX_BYTES."""


@dataclass
class _VoiceEntry:
    """Per-speaker reference state for progressive quality upgrades."""

    audio: np.ndarray
    score: float
    last_update_ts: float
    finalized: bool
    source: str  # "enrollment", "runtime", "seed"


class Qwen3TTSBackend(TTSBackend):
    """Qwen3-TTS via faster-qwen3-tts with optional multi-replica pool.

    The faster-qwen3-tts container is **GIL-bound** — Python tokenization,
    sample formatting, and CUDA dispatch all happen on a single core, so a
    single replica caps at one synthesis at a time and pegs that core at
    100 % during decode. The fix is process-level parallelism: run two
    replicas (qwen3-tts on :8002 and qwen3-tts-2 on :8012) and round-robin
    requests across them. ``vllm_url`` accepts a comma-separated list to
    enable this — the docker-compose stack already runs both containers,
    only the routing was missing.

    **Per-replica failure tracking**: a single replica entering a
    CUDA-corrupt state returns HTTP 200 on ``/health`` but HTTP 500 on
    ``/v1/audio/speech``. The global consecutive-failure counter used to
    bounce back to 0 on every successful request from the *other*
    replica — so the pool never flagged the dead replica and kept
    sending half the traffic into a black hole. Consecutive failures are
    now tracked **per URL** in ``_url_failures``; after
    ``MAX_CONSECUTIVE_FAILURES`` hits on a specific URL, that URL is
    quarantined (skipped by ``_next_url``) for
    ``_QUARANTINE_REPROBE_S`` seconds. After the cooldown, the next
    ``_next_url`` call sends a probe request to the quarantined URL and
    restores it on success. Observed 2026-04-15 when ``scribe-tts-2``
    went CUDA-corrupt mid-meeting and took half the audio with it.
    """

    MAX_CONSECUTIVE_FAILURES = 3
    # Seconds to hold a quarantined replica before attempting to re-probe.
    # Short enough that a transient failure (e.g. a compose-driven restart)
    # heals fast; long enough that a genuinely broken replica doesn't eat
    # ~3 requests per minute of probe traffic.
    _QUARANTINE_REPROBE_S = 30.0

    def __init__(self, vllm_url: str | None = None) -> None:
        self._mode: str = "disabled"
        # Parse `vllm_url` as a comma-separated pool. A single URL still
        # works — it just becomes a pool of size 1. Whitespace tolerated.
        self._urls: list[str] = [u.strip() for u in (vllm_url or "").split(",") if u.strip()]
        # Backwards-compat single-url accessor used by health probes / logs.
        self._vllm_url: str | None = self._urls[0] if self._urls else None
        self._round_robin_idx: int = 0
        self._http_client: httpx.AsyncClient | None = None
        self._voice_cache: dict[str, _VoiceEntry] = {}
        # Legacy global failure counter — kept so existing callers and
        # the degraded flag still work, but the quarantine logic is driven
        # by the per-URL counter below.
        self._consecutive_failures: int = 0
        self._last_error: str | None = None
        self._degraded: bool = False
        # Per-URL consecutive failure counter. Reset to 0 on any
        # successful response from that URL.
        self._url_failures: dict[str, int] = {}
        # URL → monotonic ts at which this URL was quarantined. Empty
        # when the URL is healthy. A URL in this dict is skipped by
        # `_next_url` until `ts + _QUARANTINE_REPROBE_S` has passed,
        # at which point it gets one probe attempt.
        self._quarantined: dict[str, float] = {}

    def _next_url(self) -> str | None:
        """Pick the next URL via round-robin, skipping quarantined replicas.

        If a quarantined replica has been parked for longer than
        ``_QUARANTINE_REPROBE_S`` it is treated as eligible again for
        this one call (a probe). If the probe fails, the URL is re-
        quarantined with a fresh timestamp. If it succeeds, the success
        path in ``synthesize_stream`` calls ``_mark_url_success`` which
        removes the quarantine entry.

        If ALL URLs are quarantined AND none is overdue for a re-probe,
        returns the oldest quarantined URL anyway — better to make a
        doomed attempt than to silently return None and drop the entire
        segment. The caller will see the failure and the per-URL
        counter is already maxed out, so nothing gets worse.
        """
        if not self._urls:
            return None
        now = time.monotonic()
        eligible: list[str] = []
        for url in self._urls:
            q_ts = self._quarantined.get(url)
            if q_ts is None:
                eligible.append(url)
            elif now - q_ts >= self._QUARANTINE_REPROBE_S:
                eligible.append(url)  # probe candidate
        if not eligible:
            # Everyone's quarantined. Return the oldest-quarantined URL
            # as a "best-effort try" — this happens when both replicas
            # went down at once and we still need to service requests.
            oldest = min(self._quarantined.items(), key=lambda kv: kv[1])
            return oldest[0]
        # Round-robin within the eligible set. Using a global counter
        # that persists across recoveries keeps load-balance fair when
        # replicas bounce.
        url = eligible[self._round_robin_idx % len(eligible)]
        self._round_robin_idx += 1
        return url

    def _mark_url_failure(self, url: str, err: str) -> None:
        """Record a failed request against one URL; quarantine at threshold."""
        n = self._url_failures.get(url, 0) + 1
        self._url_failures[url] = n
        if n >= self.MAX_CONSECUTIVE_FAILURES:
            if url not in self._quarantined:
                logger.error(
                    "TTS replica quarantined: %s after %d consecutive failures (%s)",
                    url,
                    n,
                    err[:80],
                )
            self._quarantined[url] = time.monotonic()

    def _mark_url_success(self, url: str) -> None:
        """Clear failure state for one URL after a successful response."""
        prev = self._url_failures.get(url, 0)
        if prev > 0:
            self._url_failures[url] = 0
            logger.info("TTS replica recovered: %s", url)
        if url in self._quarantined:
            del self._quarantined[url]

    async def start(self) -> None:
        """Probe /health on every URL in the pool. Drop any that fail."""
        if not self._urls:
            self._mode = "disabled"
            logger.info("TTS: disabled (no endpoint configured)")
            return

        alive: list[str] = []
        async with httpx.AsyncClient(timeout=5) as c:
            for url in self._urls:
                try:
                    r = await c.get(f"{url}/health")
                    if r.status_code == 200:
                        alive.append(url)
                    else:
                        logger.warning(
                            "TTS %s returned HTTP %d at startup probe", url, r.status_code
                        )
                except Exception as e:
                    logger.warning("TTS endpoint not reachable at %s: %s", url, e)

        if not alive:
            self._mode = "disabled"
            self._last_error = "no replica responded to /health"
            return

        self._urls = alive
        self._vllm_url = alive[0]  # for legacy single-url log paths
        self._mode = "vllm"
        if len(alive) == 1:
            logger.info("TTS: vllm-omni ready at %s (single replica)", alive[0])
        else:
            logger.info(
                "TTS: vllm-omni ready, %d replicas in pool: %s", len(alive), ", ".join(alive)
            )

    async def stop(self) -> None:
        self._voice_cache.clear()
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
        self._http_client = None

    def reset_voice_cache(self) -> None:
        if self._voice_cache:
            logger.info(
                "TTS reset_voice_cache: dropping %d cached voice reference(s)",
                len(self._voice_cache),
            )
        self._voice_cache.clear()

    @staticmethod
    def _best_segment(audio: np.ndarray, segment_len: int = 48000) -> np.ndarray:
        """Return the segment of *segment_len* samples with the highest RMS energy."""
        if len(audio) <= segment_len:
            return audio
        best_start = 0
        best_rms = 0.0
        step = max(segment_len // 2, 1)
        for start in range(0, len(audio) - segment_len + 1, step):
            seg = audio[start : start + segment_len]
            rms = float(np.sqrt(np.mean(seg**2)))
            if rms > best_rms:
                best_rms = rms
                best_start = start
        return audio[best_start : best_start + segment_len]

    def seed_voice(self, speaker_id: str, audio: np.ndarray, *, source: str = "enrollment") -> None:
        """Seed a reference from an enrollment clip (caller has vetted it)."""
        if len(audio) < 16000:
            logger.info(
                "TTS seed_voice: skipping '%s' — audio too short (%d samples)",
                speaker_id,
                len(audio),
            )
            return
        segment = self._best_segment(audio, segment_len=3 * 16000)
        score = score_reference(segment, 16000)
        self._voice_cache[speaker_id] = _VoiceEntry(
            audio=segment,
            score=score.total,
            last_update_ts=time.monotonic(),
            finalized=score.total >= _FINALIZE_THRESHOLD,
            source=source,
        )
        logger.info(
            "TTS seed_voice: '%s' src=%s score=%.2f snr=%.1fdB voiced=%.2f dur=%.1fs finalized=%s",
            speaker_id,
            source,
            score.total,
            score.snr_db,
            score.voiced_ratio,
            score.duration_s,
            self._voice_cache[speaker_id].finalized,
        )

    def cache_voice(self, speaker_id: str, audio_chunk: np.ndarray) -> None:
        """Cache or progressively upgrade a speaker's reference clip."""
        if len(audio_chunk) < 16000:
            logger.info(
                "TTS cache_voice: skipping '%s' — audio too short (%d < 16000)",
                speaker_id,
                len(audio_chunk),
            )
            return

        segment = self._best_segment(audio_chunk, segment_len=3 * 16000)
        score = score_reference(segment, 16000)
        existing = self._voice_cache.get(speaker_id)
        now = time.monotonic()

        if existing is None:
            rms = float(np.sqrt(np.mean(audio_chunk**2)))
            if rms < 0.01:
                logger.info(
                    "TTS cache_voice: skipping first clip '%s' rms=%.4f < 0.01",
                    speaker_id,
                    rms,
                )
                return
            self._voice_cache[speaker_id] = _VoiceEntry(
                audio=segment,
                score=score.total,
                last_update_ts=now,
                finalized=score.total >= _FINALIZE_THRESHOLD,
                source="runtime",
            )
            logger.info(
                "TTS cache_voice: first ref '%s' score=%.2f snr=%.1fdB finalized=%s",
                speaker_id,
                score.total,
                score.snr_db,
                self._voice_cache[speaker_id].finalized,
            )
            return

        if existing.finalized:
            return
        if score.total < existing.score + _UPGRADE_MARGIN:
            return
        if (now - existing.last_update_ts) < _MIN_UPDATE_GAP_S:
            return

        existing.audio = segment
        existing.score = score.total
        existing.last_update_ts = now
        existing.finalized = score.total >= _FINALIZE_THRESHOLD
        existing.source = "runtime"
        logger.info(
            "TTS cache_voice: upgraded '%s' → score=%.2f snr=%.1fdB finalized=%s",
            speaker_id,
            score.total,
            score.snr_db,
            existing.finalized,
        )

    def has_voice(self, speaker_id: str) -> bool:
        return speaker_id in self._voice_cache

    def get_voice(self, speaker_id: str) -> np.ndarray | None:
        entry = self._voice_cache.get(speaker_id)
        return entry.audio if entry else None

    def get_voice_score(self, speaker_id: str) -> float | None:
        entry = self._voice_cache.get(speaker_id)
        return entry.score if entry else None

    @property
    def available(self) -> bool:
        return self._mode != "disabled" and not self._degraded

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def last_error(self) -> str | None:
        return self._last_error

    async def check_health(self) -> bool:
        """Deep health check.

        Probe chain:
        1. ``GET /health`` — must return 200. This is the hard gate.
        2. ``GET /v1/models`` — treated as an INFORMATIONAL signal only.
           The legacy faster-qwen3-tts container doesn't implement this
           endpoint (returns 404); vllm-omni does. Not blocking.
        3. Warmed synth probe — SKIPPED when ``_last_error`` matches known
           protocol mismatches against the legacy container, and skipped
           when we've seen a hard backend failure recently (avoids
           hammering a known-broken endpoint on every retry tick, which
           was contributing to event-loop lag).
        """
        if not self._vllm_url:
            return False
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{self._vllm_url}/health")
                if r.status_code != 200:
                    return False
                # /v1/models is optional — legacy server returns 404.
                try:
                    m = await c.get(f"{self._vllm_url}/v1/models")
                    if m.status_code != 200 and m.status_code != 404:
                        self._last_error = f"/v1/models HTTP {m.status_code}"
                except Exception:
                    pass

            # Skip the synth probe when the backend is a known-legacy
            # custom_voice server (recognized by its last error message)
            # or when we're in healthy steady state — the /health 200
            # already implies the container is alive.
            legacy_marker = "create_voice_clone_prompt"
            if self._last_error and legacy_marker in self._last_error:
                return True  # trust /health only for legacy

            if self._degraded:
                logger.info("TTS recovered from degraded state")
                self._degraded = False
                self._consecutive_failures = 0
                self._last_error = None
            return True
        except Exception as e:
            self._last_error = str(e)
            return False

    async def synthesize(
        self,
        text: str,
        language: str,
        voice_reference: np.ndarray | None = None,
        sample_rate: int = _TTS_SAMPLE_RATE,
        studio_voice: str | None = None,
    ) -> np.ndarray:
        """Non-streaming convenience: fully consume synthesize_stream and concat.

        Kept as a minimal wrapper to satisfy the abstract base class; real
        callers use synthesize_stream directly for per-chunk fanout.
        """
        chunks: list[np.ndarray] = []
        async for chunk in self.synthesize_stream(
            text=text,
            language=language,
            voice_reference=voice_reference,
            studio_voice=studio_voice,
        ):
            chunks.append(chunk)
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    async def synthesize_stream(
        self,
        text: str,
        language: str,
        voice_reference: np.ndarray | None = None,
        studio_voice: str | None = None,
    ) -> AsyncIterator[np.ndarray]:
        """Stream float32 audio chunks from vllm-omni at 24 kHz mono.

        Wire format: POST /v1/audio/speech with stream=true response_format=pcm.
        Response is raw int16 LE PCM @ _TTS_SAMPLE_RATE. HTTP chunk boundaries
        are arbitrary, so we carry an odd residual byte across reads.

        Cloned-mode fallback: if voice_reference is provided but encoding
        exceeds bounds, fall back to studio_voice (caller supplies it).
        """
        from meeting_scribe.languages import is_tts_native

        if not is_tts_native(language):
            logger.info("TTS skip: language %r not natively supported", language)
            return

        if self._mode != "vllm" or not self._urls:
            return

        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30)

        # Pick a replica per request via round-robin. With two replicas
        # this halves wall-clock pressure on each one and lets us double
        # throughput without changing anything else in the pipeline.
        target_url = self._next_url() or self._vllm_url
        if target_url is None:
            logger.error("TTS synthesis: no URLs configured")
            self._last_error = "no TTS URL configured"
            return

        body: dict[str, object] = {
            "model": "qwen3-tts",
            "input": text,
            "language": language,
            "stream": True,
            "response_format": "pcm",
            # vLLM priority: lower = earlier under --scheduling-policy=priority.
            # Live TTS preempts translate/coding when Omni consolidation lands.
            "priority": -10,
            # Deterministic seed so a listener hears the SAME studio voice
            # across every segment for a given language. Without this the
            # underlying model's sampler produces enough prosodic variance
            # that listeners perceive it as "the voice keeps changing", which
            # came up 2026-04-15. Fixing the seed does NOT change the speaker
            # identity (that's set by `voice` below) — it only removes the
            # stochastic per-call jitter in pitch, pace and intonation.
            "seed": 42,
            "temperature": 0.0,
        }

        if voice_reference is not None and len(voice_reference) > 0:
            try:
                body["ref_audio"] = self._encode_voice_ref(voice_reference)
                body["voice"] = "custom"
            except VoiceRefTooLargeError as e:
                if studio_voice:
                    logger.info(
                        "TTS: ref_audio too large (%s); falling back to %s", e, studio_voice
                    )
                    body["voice"] = studio_voice
                else:
                    logger.warning("TTS: ref_audio too large and no studio fallback; skipping")
                    return
        elif studio_voice:
            body["voice"] = studio_voice
        else:
            logger.warning("TTS: no voice specified and no ref_audio; skipping")
            return

        # Explicit log of what we're actually asking the container for.
        # Added 2026-04-15 because the existing "TTS synthesize:" log in
        # server.py logs `mode=studio speaker_id=cluster_N` which describes
        # the DIARIZATION cluster, not the TTS voice — and with listeners
        # reporting perceived voice changes that ambiguity hid whether the
        # voice field was actually constant. This pins it.
        logger.debug(
            "TTS dispatch: url=%s voice=%s lang=%s chars=%d seed=%s",
            target_url,
            body.get("voice"),
            language,
            len(text),
            body.get("seed"),
        )

        try:
            async with self._http_client.stream(
                "POST", f"{target_url}/v1/audio/speech", json=body
            ) as resp:
                resp.raise_for_status()
                residual = bytearray()
                total_samples = 0
                async for part in resp.aiter_bytes():
                    if not part:
                        continue
                    residual.extend(part)
                    take = len(residual) - (len(residual) % 2)
                    if take == 0:
                        continue
                    pcm_bytes = bytes(residual[:take])
                    del residual[:take]
                    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
                    total_samples += len(samples)
                    yield samples.astype(np.float32) / 32768.0
                if residual:
                    logger.debug("TTS: dropping %d trailing byte(s) at stream end", len(residual))
            # A response with zero audio samples is a silent failure —
            # HTTP 200 but no output. Treat it as a per-URL failure so
            # the replica quarantines if this keeps happening, but do
            # not re-raise.
            if total_samples == 0:
                self._mark_url_failure(target_url, "stream returned 0 samples")
                self._last_error = "empty stream"
                return
            # Genuine success.
            self._mark_url_success(target_url)
            if self._consecutive_failures > 0:
                logger.info("TTS synthesis recovered after %d failures", self._consecutive_failures)
            self._consecutive_failures = 0
            self._last_error = None
        except Exception as e:
            self._mark_url_failure(target_url, str(e))
            self._consecutive_failures += 1
            self._last_error = str(e)
            if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                self._degraded = True
                logger.error(
                    "TTS marked degraded after %d consecutive failures: %s",
                    self._consecutive_failures,
                    e,
                )
            else:
                logger.warning(
                    "TTS synthesis failed on %s (%d/%d): %s",
                    target_url,
                    self._consecutive_failures,
                    self.MAX_CONSECUTIVE_FAILURES,
                    e,
                )
            return

    @staticmethod
    def _encode_voice_ref(audio: np.ndarray, sample_rate: int = _REF_AUDIO_SAMPLE_RATE) -> str:
        """Build a bounded data:audio/wav;base64,… URI for inline ref_audio.

        Slices to the best RMS segment within _REF_AUDIO_MAX_SECONDS, encodes
        as 16-bit WAV at sample_rate, base64s, wraps as a data URI. Raises
        VoiceRefTooLargeError if the result exceeds _REF_AUDIO_MAX_BYTES.
        """
        max_samples = int(_REF_AUDIO_MAX_SECONDS * sample_rate)
        segment = (
            audio
            if len(audio) <= max_samples
            else Qwen3TTSBackend._best_segment(audio, segment_len=max_samples)
        )

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            pcm = (np.clip(segment, -1.0, 1.0) * 32767).astype(np.int16)
            wf.writeframes(pcm.tobytes())
        wav_bytes = buf.getvalue()
        b64 = base64.b64encode(wav_bytes).decode("ascii")
        uri = f"data:audio/wav;base64,{b64}"
        if len(uri) > _REF_AUDIO_MAX_BYTES:
            raise VoiceRefTooLargeError(
                f"ref_audio {len(uri)} B > {_REF_AUDIO_MAX_BYTES} B ceiling"
            )
        return uri
