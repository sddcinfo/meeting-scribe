"""Synthetic-inference probes for ASR / translate / diarize.

Single source of truth for the probe contract — used by both
``routes/meeting_lifecycle._meeting_start_preflight`` (W4) and
``runtime/recovery_supervisor`` (W6b). Sharing the implementation
guarantees a backend healthy enough to pass meeting-start preflight
will also be considered healthy enough to exit recovery — no
false-failure asymmetry between the two call sites.

Why this lives outside the cached ``/api/status`` health path
(``server_support/backend_health.py``): real synthetic-inference
probes generate GPU load. ``/api/status`` is polled by the dashboard
on a stale-while-revalidate cache, so putting heavy probes there
would generate periodic background inference contention on the
exact shared GB10 path implicated in the 2026-04-30 incident.

Probe success criterion is **HTTP 200 + valid response schema** —
NOT non-empty transcribed text. The fixture audio is a 200 Hz sine
tone (above the production VAD threshold so it reaches vLLM) but
not speech, so a healthy backend may legitimately return empty
text. The probe's purpose is to confirm the inference pipeline
isn't wedged; an empty transcription with a valid schema proves
the kernel ran without hanging.

Adaptive timeout: if the corresponding backend's RTT histogram has
≥10 samples, threshold = ``p95 × 2`` (capped at the backend's
ceiling). On cold boot, falls back to a conservative fixed
deadline.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE_AUDIO_PATH = REPO_ROOT / "tests" / "fixtures" / "probe_audio_1500ms.wav"

# Adaptive-timeout knobs — surface as module constants so tests can
# patch and so the W6b supervisor reuses identical values.
ADAPTIVE_MIN_SAMPLES = 10
ADAPTIVE_P95_MULTIPLIER = 2.0

ASR_COLD_DEFAULT_S = 5.0
ASR_CEILING_S = 5.0
TRANSLATE_COLD_DEFAULT_S = 3.0
TRANSLATE_CEILING_S = 5.0
DIARIZE_COLD_DEFAULT_S = 3.0
DIARIZE_CEILING_S = 5.0


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of a single synthetic-inference probe call."""

    status: Literal["ok", "timeout", "schema_error", "http_error"]
    latency_ms: float
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


_HARD_MIN_TIMEOUT_S = 0.5  # never fire a probe with less than 0.5s budget


def _adaptive_timeout(
    histogram: deque[float], cold_default_s: float, ceiling_s: float
) -> float:
    """Threshold for a single probe call.

    Adapts to recent backend behaviour:
    - <10 samples (cold boot, fresh container): fall back to
      ``cold_default_s``. The histogram has no signal yet, so we
      give the backend the conservative budget per the plan.
    - ≥10 samples: threshold = ``p95 × 2``, capped at ``ceiling_s``,
      with a hard floor of 0.5s so a very-fast cluster doesn't fire
      timeout-fails on transient micro-stutters.

    The ceiling is a hard cap — even a backend with sustained slow
    p95 won't extend probe timeouts past this. If a backend's true
    p95 exceeds the ceiling it'll fail probes; that's the desired
    behaviour (the user-visible meeting SLA itself fails at that
    level)."""
    samples = list(histogram)
    if len(samples) < ADAPTIVE_MIN_SAMPLES:
        return cold_default_s
    samples.sort()
    idx = max(0, min(len(samples) - 1, int(round(0.95 * (len(samples) - 1)))))
    p95_ms = samples[idx]
    threshold_s = (p95_ms * ADAPTIVE_P95_MULTIPLIER) / 1000.0
    return max(_HARD_MIN_TIMEOUT_S, min(ceiling_s, threshold_s))


# ─────────────────────────────────────────────────────────────────────
# Probe payload builders.  Each one assembles the SAME request shape
# the live backend uses, so a probe exercises the same code path real
# meeting traffic does.  Heavy state (httpx client, audio fixture) is
# loaded lazily so importing this module from a pure-CPU test context
# (e.g. test_synthetic_probe.py for the timeout helper) doesn't pay
# the file-I/O or module-import cost.
# ─────────────────────────────────────────────────────────────────────


_probe_audio_cache: bytes | None = None


def _load_probe_audio() -> bytes:
    """Read the fixture WAV bytes, cached on first read."""
    global _probe_audio_cache
    if _probe_audio_cache is None:
        _probe_audio_cache = PROBE_AUDIO_PATH.read_bytes()
    return _probe_audio_cache


def _build_asr_payload(audio_bytes: bytes, model_id: str) -> dict:
    """Mirror the production VllmASRBackend request shape (audio_b64
    in OpenAI chat-completions audio format)."""
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "Probe."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    }
                ],
            },
        ],
        # Probe traffic is the lowest-priority load on the shared
        # vLLM scheduler — must NEVER preempt live meeting work.
        "priority": 50,
    }


def _build_translate_payload(model_id: str) -> dict:
    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a translator. Translate the user's text to English."},
            {"role": "user", "content": "ping"},
        ],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "priority": 50,
    }


def _wav_pcm_payload() -> tuple[bytes, int]:
    """Strip the WAV header from the fixture so the diarize endpoint
    gets raw int16 PCM with a sample-rate header (its production
    contract)."""
    audio_bytes = _load_probe_audio()
    with wave.open(io.BytesIO(audio_bytes), "rb") as r:
        sample_rate = r.getframerate()
        n_frames = r.getnframes()
        pcm = r.readframes(n_frames)
    return pcm, sample_rate


# ─────────────────────────────────────────────────────────────────────
# Backend probes.  Each one captures latency, classifies the outcome
# into one of four ``status`` codes, and never raises — callers can
# react on ``result.status`` and ``result.detail`` directly.
# ─────────────────────────────────────────────────────────────────────


async def asr_synthetic_probe(
    base_url: str,
    model_id: str,
    histogram: deque[float],
    *,
    timeout_override_s: float | None = None,
) -> ProbeResult:
    """Probe the ASR vLLM endpoint with the production audio request
    shape. Returns ``ok`` on HTTP 200 + valid OpenAI schema."""
    timeout_s = timeout_override_s or _adaptive_timeout(
        histogram, ASR_COLD_DEFAULT_S, ASR_CEILING_S
    )
    payload = _build_asr_payload(_load_probe_audio(), model_id)
    return await _post_and_validate(
        url=f"{base_url}/v1/chat/completions",
        json_payload=payload,
        timeout_s=timeout_s,
        validator=_openai_chat_response_valid,
    )


async def translate_synthetic_probe(
    base_url: str,
    model_id: str,
    histogram: deque[float],
    *,
    timeout_override_s: float | None = None,
) -> ProbeResult:
    """Probe the translate vLLM endpoint with a tiny chat request."""
    timeout_s = timeout_override_s or _adaptive_timeout(
        histogram, TRANSLATE_COLD_DEFAULT_S, TRANSLATE_CEILING_S
    )
    payload = _build_translate_payload(model_id)
    return await _post_and_validate(
        url=f"{base_url}/v1/chat/completions",
        json_payload=payload,
        timeout_s=timeout_s,
        validator=_openai_chat_response_valid,
    )


async def diarize_synthetic_probe(
    base_url: str,
    histogram: deque[float],
    *,
    max_speakers: int = 4,
    timeout_override_s: float | None = None,
) -> ProbeResult:
    """Probe the pyannote diarize endpoint with raw PCM. Returns ``ok``
    on HTTP 200 + JSON-decodable response."""
    timeout_s = timeout_override_s or _adaptive_timeout(
        histogram, DIARIZE_COLD_DEFAULT_S, DIARIZE_CEILING_S
    )
    pcm, sample_rate = _wav_pcm_payload()
    return await _post_and_validate(
        url=f"{base_url}/v1/diarize",
        json_payload=None,
        timeout_s=timeout_s,
        validator=_diarize_response_valid,
        content=pcm,
        headers={
            "Content-Type": "application/octet-stream",
            "X-Sample-Rate": str(sample_rate),
            "X-Max-Speakers": str(max_speakers),
            "X-Min-Speakers": "0",
        },
    )


# ─────────────────────────────────────────────────────────────────────


async def _post_and_validate(
    url: str,
    *,
    timeout_s: float,
    validator,
    json_payload: dict | None = None,
    content: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> ProbeResult:
    """Single shared post-and-classify implementation. Captures
    wall-clock latency; never raises."""
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            kwargs: dict = {}
            if json_payload is not None:
                kwargs["json"] = json_payload
            if content is not None:
                kwargs["content"] = content
            if headers is not None:
                kwargs["headers"] = headers
            resp = await client.post(url, **kwargs)
    except (httpx.TimeoutException, asyncio.TimeoutError) as exc:
        return ProbeResult(
            status="timeout",
            latency_ms=(time.monotonic() - t0) * 1000,
            detail=f"timeout after {timeout_s:.1f}s: {exc!s}",
        )
    except httpx.HTTPError as exc:
        return ProbeResult(
            status="http_error",
            latency_ms=(time.monotonic() - t0) * 1000,
            detail=f"http error: {exc!s}",
        )

    latency_ms = (time.monotonic() - t0) * 1000
    if resp.status_code != 200:
        return ProbeResult(
            status="http_error",
            latency_ms=latency_ms,
            detail=f"status {resp.status_code}: {resp.text[:200]}",
        )
    try:
        data = resp.json()
    except ValueError as exc:
        return ProbeResult(
            status="schema_error",
            latency_ms=latency_ms,
            detail=f"non-JSON response: {exc!s}",
        )
    ok, why = validator(data)
    if not ok:
        return ProbeResult(status="schema_error", latency_ms=latency_ms, detail=why)
    return ProbeResult(status="ok", latency_ms=latency_ms, detail=None)


def _openai_chat_response_valid(data: object) -> tuple[bool, str | None]:
    """Validate the OpenAI chat-completions response shape.
    Does NOT assert non-empty content — a pure tone may legitimately
    transcribe to empty text on a healthy backend."""
    if not isinstance(data, dict):
        return False, f"top-level not a dict: {type(data).__name__}"
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return False, "choices missing or not a non-empty list"
    first = choices[0]
    if not isinstance(first, dict):
        return False, "choices[0] not a dict"
    msg = first.get("message")
    if not isinstance(msg, dict):
        return False, "choices[0].message missing"
    if "content" not in msg:
        return False, "choices[0].message.content missing"
    return True, None


def _diarize_response_valid(data: object) -> tuple[bool, str | None]:
    """Validate the pyannote diarize JSON response shape."""
    if not isinstance(data, dict):
        return False, f"top-level not a dict: {type(data).__name__}"
    # pyannote returns a list of segments under various keys depending
    # on container version; accept any of the known shapes.
    if "segments" in data and isinstance(data["segments"], list):
        return True, None
    if "diarization" in data and isinstance(data["diarization"], list):
        return True, None
    return False, f"no 'segments'/'diarization' key in response (keys={list(data)})"
