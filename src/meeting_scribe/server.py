"""FastAPI server — GB10 real-time bilingual meeting transcription.

Single meeting at a time. No auth (localhost only). No TLS needed.
Wires together: storage, audio resample, ASR (vLLM Qwen3-ASR), translation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import socket
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fastapi
import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from meeting_scribe.audio.resample import (  # kept for legacy/test clients
    Resampler,
)
from meeting_scribe.config import ServerConfig
from meeting_scribe.models import (
    DetectedSpeaker,
    MeetingMeta,
    MeetingState,
    RoomLayout,
    SpeakerAttribution,
    TranscriptEvent,
)
from meeting_scribe.speaker.enrollment import EnrolledSpeaker, SpeakerEnrollmentStore
from meeting_scribe.storage import AudioWriter, AudioWriterProcess, MeetingStorage
from meeting_scribe.translation.queue import TranslationQueue

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    import numpy as np

    from meeting_scribe.backends.base import ASRBackend, TranslateBackend
    from meeting_scribe.backends.mse_encoder import Fmp4AacEncoder

logger = logging.getLogger(__name__)

# Force INFO level for our modules AND attach a handler so messages
# actually reach stdout. Uvicorn configures its own uvicorn.* loggers but
# leaves the root logger on the default lastResort handler at WARNING —
# so setLevel(INFO) alone isn't enough: INFO messages pass the logger
# filter, propagate up, and then get dropped by lastResort. The missing
# handler is why "TTS fire:", "TTS synthesize:", "cache_voice" INFO logs
# that should have been visible during live meetings were silently
# swallowed, making TTS failures look like nothing was happening at all.
_ms_logger = logging.getLogger("meeting_scribe")
_ms_logger.setLevel(logging.INFO)
if not _ms_logger.handlers:
    _ms_handler = logging.StreamHandler()
    _ms_handler.setLevel(logging.INFO)
    _ms_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _ms_logger.addHandler(_ms_handler)
    _ms_logger.propagate = False

# Global state
config: ServerConfig
storage: MeetingStorage
resampler: Resampler
translation_queue: TranslationQueue | None = None

# Backends
asr_backend: ASRBackend | None = None
translate_backend: TranslateBackend | None = None
furigana_backend = None  # FuriganaBackend — initialized lazily in lifespan
tts_backend = None  # Qwen3TTSBackend (optional)
diarize_backend = None  # SortformerBackend (optional)
name_extractor = None  # LLMNameExtractor (optional)

# Speaker enrollment + room layout draft (session-scoped)
enrollment_store: SpeakerEnrollmentStore = SpeakerEnrollmentStore()
_draft_layouts: dict[str, RoomLayout] = {}
_draft_layout_access: dict[str, float] = {}
_DEFAULT_SESSION = "default"


def _get_session_id(request: fastapi.Request) -> str:
    """Get or create a session ID from cookie."""
    return request.cookies.get("scribe_session", _DEFAULT_SESSION)


def _get_draft_layout(session_id: str) -> RoomLayout:
    """Get the draft layout for a session, creating if needed."""
    if session_id not in _draft_layouts:
        _draft_layouts[session_id] = RoomLayout()
    _draft_layout_access[session_id] = time.monotonic()
    return _draft_layouts[session_id]


def _set_draft_layout(session_id: str, layout: RoomLayout) -> None:
    """Set the draft layout for a session."""
    _draft_layouts[session_id] = layout
    _draft_layout_access[session_id] = time.monotonic()


# Per-client session preferences (language for interpretation, audio-out)
@dataclass
class ClientSession:
    preferred_language: str = ""
    send_audio: bool = False
    interpretation_mode: str = (
        "translation"  # "translation" (TTS only) or "full" (passthrough + TTS)
    )
    # "studio" — Qwen3-TTS named speaker per target language (default, fast)
    # "cloned" — clone each participant's voice from live meeting audio
    voice_mode: str = "studio"
    # Wire format negotiated via `set_format` on the audio-out WS. `None`
    # means the negotiation grace period is still running — audio for
    # this listener is buffered into `pending_audio` instead of being
    # sent immediately. After `_AUDIO_FORMAT_GRACE_S` the default is
    # "wav-pcm" for backward compatibility with cached legacy clients.
    audio_format: str | None = None
    # Lazy: stays None until the first audio delivery after a listener
    # negotiates "mse-fmp4-aac". Holds the per-connection PyAV encoder.
    # Import-cycle concern is handled by the TYPE_CHECKING-only import
    # at the top of this module.
    mse_encoder: Fmp4AacEncoder | None = None
    # PCM delivery attempts that arrived while `audio_format is None`.
    # Each item is `(pcm: np.ndarray, source_sample_rate: int)`. Capped
    # to about 1 second of audio to bound memory for stuck handshakes.
    pending_audio: list = field(default_factory=list)
    # Monotonic deadline after which `audio_format=None` is promoted to
    # "wav-pcm" on the next delivery attempt. Set when the WS accepts.
    grace_deadline: float = 0.0
    # MSE stuck-health bookkeeping. Never drives encoder recreation —
    # only logs a diagnostic WARNING via the stuck-detection path.
    last_fragment_at: float = 0.0
    bytes_in_since_last_emit: int = 0


# Single active meeting + connections
current_meeting: MeetingMeta | None = None
ws_connections: set[WebSocket] = set()
_client_prefs: dict[WebSocket, ClientSession] = {}
_audio_out_clients: set[WebSocket] = set()
_audio_out_prefs: dict[WebSocket, ClientSession] = {}
audio_writer: AudioWriter | AudioWriterProcess | None = None
meeting_start_time: float = 0.0  # monotonic time for audio alignment
detected_speakers: list[DetectedSpeaker] = []  # per-meeting speaker state
speaker_verifier = None  # SpeakerVerifier instance
_background_tasks: set[asyncio.Task] = set()  # prevent GC of fire-and-forget tasks
_speaker_pulse_task: asyncio.Task | None = None  # periodic heartbeat
_speaker_catchup_task: asyncio.Task | None = None  # retroactive speaker attribution
_eager_summary_task: asyncio.Task | None = None  # periodic draft summary during recording
_eager_summary_cache: dict | None = None  # last draft summary from eager loop

# Lazy-initialized singleton ThreadPoolExecutor for the speaker-embedding
# worker pool. Created on first use inside _process_event and reused for
# the lifetime of the server.  Moved to module scope so mypy can see the
# type (previously stashed as an attribute on _process_event itself).
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor

_speaker_executor_singleton: _ThreadPoolExecutor | None = None
_eager_summary_event_count: int = 0  # event count when draft was generated

# Slide translation
slide_job_runner = None  # SlideJobRunner — initialized in lifespan if worker image available
slides_enabled: bool = False

# Time-proximity pseudo-speaker tracking — fallback when diarization hasn't
# produced a result yet. Segments within 3 seconds of the previous final
# event's end get the same pseudo-cluster; bigger gaps allocate a new one.
# This gives users a meaningful "Speaker 1 / 2 / 3" grouping immediately,
# which the catch-up loop can later refine to real diarization clusters.
_last_pseudo_cluster_id: int = 0
_last_pseudo_end_ms: int = 0
_next_pseudo_cluster_id: int = 100  # start at 100 so pseudo IDs don't
# collide with real diarization IDs (which are small integers starting at 0 or 1)

# Backend health tracking: record consecutive failures per backend so
# /api/status can report "error" state even when the backend object is
# alive in memory but every request is failing (e.g., GPU context corrupted).
_backend_failure_counts: dict[str, int] = {"asr": 0, "diarize": 0, "translate": 0}
_backend_last_errors: dict[str, str] = {}
BACKEND_FAILURE_THRESHOLD = 3  # 3 consecutive failures → mark degraded


def _record_backend_failure(name: str, err: str) -> None:
    _backend_failure_counts[name] = _backend_failure_counts.get(name, 0) + 1
    _backend_last_errors[name] = err[:200]
    if _backend_failure_counts[name] >= BACKEND_FAILURE_THRESHOLD:
        logger.error(
            "Backend '%s' marked degraded after %d consecutive failures: %s",
            name,
            _backend_failure_counts[name],
            err[:200],
        )


def _record_backend_success(name: str) -> None:
    if _backend_failure_counts.get(name, 0) > 0:
        logger.info("Backend '%s' recovered after %d failures", name, _backend_failure_counts[name])
    _backend_failure_counts[name] = 0
    _backend_last_errors.pop(name, None)


def _backend_is_degraded(name: str) -> bool:
    return _backend_failure_counts.get(name, 0) >= BACKEND_FAILURE_THRESHOLD


# ── Container auto-restart ──────────────────────────────────────
#
# When a container's CUDA context is corrupted, only a restart clears it.
# This is a second line of defense behind Docker's autoheal sidecar.

_container_restart_cooldown: dict[str, float] = {}
_RESTART_COOLDOWN_S = 120.0
# Never auto-restart vLLM containers — model reload takes 3+ minutes.
_RESTART_BLACKLIST = frozenset({"scribe-asr", "autosre-vllm-local"})


async def _restart_container(container_name: str) -> bool:
    """Restart a Docker container if not on cooldown. Returns True if issued."""
    if container_name in _RESTART_BLACKLIST:
        logger.warning(
            "Skipping auto-restart of %s (blacklisted — slow model load)",
            container_name,
        )
        return False

    now = time.monotonic()
    last = _container_restart_cooldown.get(container_name, 0)
    if now - last < _RESTART_COOLDOWN_S:
        return False

    _container_restart_cooldown[container_name] = now
    logger.warning("Auto-restarting degraded container: %s", container_name)

    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "restart",
            container_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode == 0:
            logger.info("Container %s restart issued successfully", container_name)
            return True
        logger.error(
            "Container %s restart failed (rc=%d): %s",
            container_name,
            proc.returncode,
            stderr.decode()[:200],
        )
    except Exception as e:
        logger.error("Failed to restart container %s: %s", container_name, e)
    return False


# ── Deep backend health (verifies real inference, not just object existence) ──
#
# Each backend is probed with a real lightweight request to confirm it can
# actually serve work. Results are cached to avoid burning resources on
# every /api/status call.

_deep_health_cache: dict[str, tuple[float, dict]] = {}
# Bumped 2026-04-15 from 10 → 30 s. The 10 s TTL meant that every ~10 s a
# guest /api/status poll burst-fired four concurrent httpx probes to the
# backend containers, each creating its own AsyncClient (TLS + SSL context
# per call). During a live meeting the synth-busy TTS container's /health
# response was slow enough to hold the gather() for ~2.5 s, which showed up
# as an event-loop stall that killed the audio WebSocket ping. Backends
# genuinely changing state inside 30 s is rare — if a container dies, the
# next probe will catch it, and the TTS backend has its own degraded-flag
# short-circuit above.
_DEEP_HEALTH_TTL = 30.0


async def _deep_check_asr() -> dict:
    """Actually call the ASR vLLM endpoint — catches model-loading/crashed state."""
    if asr_backend is None:
        return {"ready": False, "detail": "not initialized"}
    if getattr(asr_backend, "degraded", False):
        return {"ready": False, "detail": getattr(asr_backend, "last_error", "degraded")}
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{config.asr_vllm_url}/v1/models")
            if r.status_code != 200:
                return {"ready": False, "detail": f"/v1/models HTTP {r.status_code}"}
            data = r.json().get("data", [])
            if not data:
                return {"ready": False, "detail": "no models loaded"}
        return {"ready": True, "detail": None}
    except Exception as e:
        return {"ready": False, "detail": f"{type(e).__name__}: {str(e)[:80]}"}


async def _deep_check_translate() -> dict:
    """Actually call the translate vLLM endpoint — catches model loading."""
    if translate_backend is None:
        return {"ready": False, "detail": "not initialized"}
    # Reject if the retry counter has been tripping
    if _backend_is_degraded("translate"):
        return {"ready": False, "detail": _backend_last_errors.get("translate", "degraded")}
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{config.translate_vllm_url}/v1/models")
            if r.status_code != 200:
                return {"ready": False, "detail": f"/v1/models HTTP {r.status_code}"}
            data = r.json().get("data", [])
            if not data:
                return {"ready": False, "detail": "no models loaded"}
        return {"ready": True, "detail": None}
    except Exception as e:
        return {"ready": False, "detail": f"{type(e).__name__}: {str(e)[:80]}"}


async def _deep_check_diarize() -> dict:
    """Actually call the diarize container /health — catches CUDA corruption."""
    if diarize_backend is None:
        return {"ready": False, "detail": "not initialized"}
    if getattr(diarize_backend, "degraded", False):
        return {
            "ready": False,
            "detail": getattr(diarize_backend, "last_error", None) or "degraded",
        }
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{config.diarize_url}/health")
            if r.status_code != 200:
                return {"ready": False, "detail": f"HTTP {r.status_code}"}
            body = r.json()
            if body.get("status") != "ok":
                return {"ready": False, "detail": str(body)[:80]}
        return {"ready": True, "detail": None}
    except Exception as e:
        return {"ready": False, "detail": f"{type(e).__name__}: {str(e)[:80]}"}


async def _deep_check_tts() -> dict:
    """Actually call the TTS container /health — catches CUDA corruption."""
    if tts_backend is None:
        return {"ready": False, "detail": "not initialized"}
    if getattr(tts_backend, "degraded", False):
        return {
            "ready": False,
            "detail": getattr(tts_backend, "last_error", None) or "degraded",
        }
    url = (getattr(tts_backend, "_vllm_url", None) or config.tts_vllm_url or "").strip()
    if not url:
        return {"ready": False, "detail": "no TTS endpoint configured"}

    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            try:
                r = await c.get(f"{url}/health")
            except Exception as e:
                return {"ready": False, "detail": f"{type(e).__name__}: {str(e)[:60]}"}
            if r.status_code != 200:
                return {"ready": False, "detail": f"HTTP {r.status_code}"}
            # /v1/models is informational only. The legacy faster-qwen3-tts
            # server returns 404 because it's not an OpenAI-compatible
            # endpoint — that is expected for the pre-vllm-omni container
            # and must NOT surface as "TTS not ready" in /api/status.
            body = None
            with suppress(Exception):
                body = (
                    r.json()
                    if r.headers.get("content-type", "").startswith("application/json")
                    else None
                )
            if isinstance(body, dict):
                status = body.get("status", "unknown")
                if status and status != "healthy":
                    return {"ready": False, "detail": f"health status={status}"}
        return {"ready": True, "detail": None}
    except Exception as e:
        return {"ready": False, "detail": f"{type(e).__name__}: {str(e)[:80]}"}


async def _deep_check_furigana() -> dict:
    """Verify pykakasi is loaded and produces ruby output for a kanji probe."""
    if furigana_backend is None:
        return {"ready": False, "detail": "not initialized (pykakasi import failed?)"}
    if getattr(furigana_backend, "_kks", None) is None:
        return {"ready": False, "detail": "pykakasi dictionary not loaded"}
    try:
        html = await furigana_backend.annotate("会議")
        if not html or "<ruby>" not in html:
            return {"ready": False, "detail": "probe returned no ruby markup"}
        return {"ready": True, "detail": None}
    except Exception as e:
        return {"ready": False, "detail": f"{type(e).__name__}: {str(e)[:80]}"}


async def _deep_backend_health(force: bool = False) -> dict[str, dict]:
    """Run all deep health checks in parallel, with a short cache.

    Returns {backend_name: {"ready": bool, "detail": str|None}}.
    force=True bypasses the cache (e.g., for /api/meeting/start).
    """
    now = time.monotonic()
    results: dict[str, dict] = {}
    to_check: list[tuple[str, Callable[[], Awaitable[dict]]]] = []
    checks = {
        "asr": _deep_check_asr,
        "translate": _deep_check_translate,
        "diarize": _deep_check_diarize,
        "tts": _deep_check_tts,
        "furigana": _deep_check_furigana,
    }
    for name, fn in checks.items():
        if not force:
            cached = _deep_health_cache.get(name)
            if cached and now - cached[0] < _DEEP_HEALTH_TTL:
                results[name] = cached[1]
                continue
        to_check.append((name, fn))

    if to_check:
        fresh = await asyncio.gather(*[fn() for _, fn in to_check])
        for (name, _), result in zip(to_check, fresh):
            _deep_health_cache[name] = (now, result)
            results[name] = result

    return results


# Events waiting for diarization results to catch up. Keyed by segment_id.
# An event lands here when _process_event runs BEFORE diarization has
# produced results for that time range (typically because diarization
# buffers 4s of audio vs ASR's faster finalization). The catch-up loop
# polls for results and re-broadcasts with cluster_ids attached.
from collections import OrderedDict as _OrderedDict

_pending_speaker_events: _OrderedDict[str, TranscriptEvent] = _OrderedDict()
# Retroactive updates are also applied to the audio writer position, so
# we track the monotonic time the event was first queued.
_pending_speaker_timestamps: dict[str, float] = {}

# TTS pipeline: a single long-lived worker task drains a FIFO queue of
# translation events. A real queue (instead of the old drop-latest-wins
# policy) means that under transient overload — e.g. translation gets
# slow for a few seconds — listeners still hear every segment, in order,
# just delayed. The queue is bounded so sustained overload can't grow
# memory without limit; when the cap is hit we drop the OLDEST item
# (front of queue) to make room, so audio stays roughly in sync with
# live speech rather than playing ancient backlog.
_tts_semaphore: asyncio.Semaphore | None = None  # init guard, still set at startup
_tts_queue: asyncio.Queue | None = None

# Short-window text-hash dedup for ASR finals. Guards against scribe-asr
# emitting the same final under a fresh segment_id after a container restart
# or other flakiness (LocalAgreement revision dedup keys on seg_id and can't
# catch that case). See _process_event.
_DEDUP_WINDOW_S = 8.0
_recent_finals: dict[tuple[str, str], tuple[float, str]] = {}

# Serialize meeting start/stop. The UI can fire /api/meeting/start twice
# (user double-click, browser retry, auto-recovery) and without this lock
# two concurrent handlers race through the "create new meeting + open
# audio writer" path, leaving one meeting orphaned with a dangling audio
# writer and a confused WS client. The lock makes start/stop atomic —
# the second caller waits for the first to finish and gets the current
# meeting state back.
_meeting_lifecycle_lock: asyncio.Lock | None = None


def _get_meeting_lifecycle_lock() -> asyncio.Lock:
    global _meeting_lifecycle_lock
    if _meeting_lifecycle_lock is None:
        _meeting_lifecycle_lock = asyncio.Lock()
    return _meeting_lifecycle_lock


# Silence watchdog — tracks the last wall-clock arrival of an audio chunk.
# `_silence_watchdog_loop` broadcasts a meeting_warning when this goes
# stale during a recording-state meeting so the UI can show "reconnect
# required" instead of looking silently hung.
_last_audio_chunk_ts: float = 0.0
_SILENCE_WARN_THRESHOLD_S = 10.0
_silence_warn_sent: bool = False
_tts_worker_tasks: list[asyncio.Task] = []

# Bound on the in-flight TTS backlog. Keep this SMALL so TTS stays
# near-live: faster-qwen3-tts produces ~1-2s of synthesis per segment,
# live speech produces segments every ~1s, and with a single serial
# worker a 40-slot queue would accumulate ~60s of backlog within a
# minute. 3 slots + drop-oldest policy means the listener always hears
# the most recent translations and silently misses at most a handful
# when the worker pool falls behind — preferable to playing ancient
# audio that's drifted 30s off the live transcript.
# With two replicas + streaming, two segments synthesise in parallel and
# listeners hear first audio within ~300 ms of synth start. Allowing four
# in the backlog plus two in flight (~6 segments buffered) is still
# comfortably under the lag budget while letting brief bursts coast
# without triggering oldest-drop.
_TTS_QUEUE_MAXSIZE = 4

# Number of concurrent TTS worker tasks. Matches the translation queue's
# concurrency=4 so TTS can keep up with translation on bursty meetings.
# Parallel HTTP calls let the TTS backend (vLLM / faster-qwen3-tts) batch
# across requests and hide network + encode/decode overhead between
# syntheses, restoring near-live audio output when live speech produces
# segments faster than a single serial worker can synthesise.
_TTS_WORKER_COUNT = 4

# Container outbound-concurrency cap [P1-1-i2 + i4 + i5]. faster-qwen3-tts
# serialises synthesis on the GPU, so parallel HTTP POSTs pile up at the
# container instead of running in parallel. Cap outbound concurrency to 1
# for now; Phase 3a bumps this to 4 when we swap to vLLM-Omni. Workers
# acquire _tts_backend_semaphore BEFORE dequeuing from _tts_queue, so only
# one worker is ever committed to a segment — the other N-1 are parked on
# the semaphore holding nothing. See _tts_worker_loop.
# With the two-replica TTS pool (qwen3-tts + qwen3-tts-2) wired up via
# round-robin in tts_qwen3.Qwen3TTSBackend._next_url, we can have up to N
# workers submitting simultaneously where N == replica count. Default 2;
# override with SCRIBE_TTS_CONTAINER_CONCURRENCY for a single-replica
# deployment.
_TTS_CONTAINER_MAX_CONCURRENCY = int(os.environ.get("SCRIBE_TTS_CONTAINER_CONCURRENCY", "2"))
_tts_backend_semaphore: asyncio.Semaphore | None = None  # init in _start_tts_worker

# Hard TTS lag cap. Deadline = tts_origin + this value, where tts_origin is
# translation.completed_at (upstream-aware anchor). With streaming + two
# replicas, first-audio latency is sub-second and most full-synth calls
# land inside ~2 s. Tightening from 8 s → 5 s drops stale segments that
# would otherwise play 5+ s behind the speaker — freshness beats
# completeness for live interpretation.
_TTS_MAX_SPEECH_LAG_S = 5.0

# Slack threshold at the producer gate — drop if less than this much of the
# deadline remains when the event arrives AND there is real outstanding work.
_TTS_PRODUCER_MIN_SLACK_S = 0.5

# Fallback expected-synth budget when we don't yet have enough synth_ms samples
# to compute a real P95. GB10 baseline: voice cloning ~0.8s + synthesis ~1.2s
# = ~2.0s total. Once real P95 data flows, the adaptive estimate takes over.
_TTS_EXPECTED_SYNTH_DEFAULT_S = 2.0

# Size-aware wait_for ceiling: base + per-char budget. A 25-char segment gets
# ~8.75 s, a 300-char (MAX_TTS_CHARS) segment gets 23 s. The effective timeout
# used in wait_for is min(size_timeout, max(0.2, deadline - now)) so the
# deadline always wins — see [P1-1-i2].
#
# Bumped 2026-04-15 from base=5.0/per_char=0.03 to base=8.0/per_char=0.05 after
# observing that faster-qwen3-tts round-robin across two replicas can incur
# ~4 s of container-internal queue wait when requests burst in, leaving the
# old 5 s budget with only ~1 s for actual synthesis. Paired with the ASR
# buffer_seconds bump (1.5 → 3.5) which cuts TTS request rate by ~2.5x, the
# two changes together keep the deadline honest while the model-level
# freshness cap (_TTS_MAX_SPEECH_LAG_S = 5 s) remains the real freshness SLA.
_TTS_SYNTH_TIMEOUT_BASE_S = 8.0
_TTS_SYNTH_TIMEOUT_PER_CHAR_S = 0.05

# Minimum samples required before percentile stats are reported — prevents
# tiny-window noise at meeting start. [P1-6-i1]
_MIN_SAMPLES_FOR_PCT = 10

# Whitelist of acknowledgment tokens that should be dropped as "filler" when
# the pipeline has real outstanding work. [P2-2-i1] — char-length was too
# aggressive and would swallow "No", "Stop", names. Env-configurable.
_TTS_ACK_WHITELIST: frozenset[str] = frozenset(
    {
        "はい",
        "うん",
        "ええ",
        "そうですね",
        "なるほど",
        "i see",
        "yeah",
        "uh",
        "um",
        "mhm",
        "ok",
        "okay",
        "right",
    }
)
_env_ws = os.environ.get("SCRIBE_TTS_ACK_WHITELIST")
if _env_ws:
    _TTS_ACK_WHITELIST = frozenset(t.strip().lower() for t in _env_ws.split(",") if t.strip())

# In-flight synthesis tracking. _tts_in_flight is the authoritative
# "workers_busy" count — it is incremented inside _do_tts_synthesis right
# before the backend call and decremented in a finally. This is NOT the same
# as "how many worker tasks exist" — with _TTS_CONTAINER_MAX_CONCURRENCY=1
# the max in-flight is always 1 regardless of how many workers are spawned.
_tts_in_flight: int = 0
# seg_id → monotonic ts when synth started; used by the health evaluator
# to detect "stuck" requests (oldest in-flight age exceeds expected budget).
_tts_inflight_started: dict[tuple[str, str], float] = {}

refinement_worker = None  # RefinementWorker instance (started per meeting)


@dataclass
class _DrainEntry:
    """One meeting's refinement drain state + counter snapshot.

    The registry is a *list*, not a dict, because ``meeting_id`` can
    repeat — a single meeting that is stopped, re-started mid-session
    (future path), and stopped again produces two independent drain
    entries.  Keying by meeting_id alone would overwrite the older
    entry and lose its authoritative counter snapshot; validation
    relies on the ``drain_id`` pin to query a specific arm.

    Counters are snapshotted twice:
      * at drain kickoff, while the worker is still reachable, so the
        endpoint can serve something even if ``_drain_refinement``
        hasn't run yet;
      * after ``worker.stop()`` returns, so the final values include
        any calls that landed during ``_process_remaining``.
    """

    drain_id: int
    meeting_id: str
    task: asyncio.Task
    state: str  # "draining" | "complete" | "partial" | "failed"
    started_at: float
    finished_at: float | None = None
    error: str | None = None
    translate_calls: int = 0
    asr_calls: int = 0
    errors_at_stop: int = 0


_drain_seq: int = 0
# Bounded LRU (cap 32). Oldest completed entries are evicted when a
# new drain kicks off so long-running processes don't leak memory, but
# the recently-finished entries stay available for validation queries.
_refinement_drains: list[_DrainEntry] = []
_REFINEMENT_DRAINS_CAP = 32


async def _drain_refinement(worker: Any, meeting_id: str, drain_id: int) -> None:
    """Background driver for one refinement drain.

    Looks up its entry by ``drain_id`` (unambiguous — meeting_id can
    repeat), awaits ``worker.stop()`` under a 60 s budget that matches
    the user-facing "post-meeting delay < 60 s" claim in
    ``refinement.py`` module doc, and re-snapshots the worker's
    counters into the entry so post-drain validation reads the final
    numbers (including anything ``_process_remaining`` added).
    """
    entry = next((e for e in _refinement_drains if e.drain_id == drain_id), None)
    if entry is None:
        logger.warning("Refinement drain %d has no registry entry", drain_id)
        return
    try:
        await asyncio.wait_for(worker.stop(), timeout=60.0)
        entry.state = "complete"
    except TimeoutError:
        entry.state = "partial"
        entry.error = "drain exceeded 60s budget"
        logger.warning(
            "Refinement drain for meeting %s exceeded 60s budget (partial)",
            meeting_id,
        )
    except Exception as exc:
        entry.state = "failed"
        entry.error = f"{type(exc).__name__}: {exc}"
        logger.exception("Refinement drain for meeting %s failed", meeting_id)
    finally:
        entry.finished_at = time.time()
        entry.translate_calls = getattr(worker, "translate_call_count", 0)
        entry.asr_calls = getattr(worker, "asr_call_count", 0)
        entry.errors_at_stop = getattr(worker, "last_error_count", 0)


def _evict_completed_drains(limit: int = _REFINEMENT_DRAINS_CAP) -> None:
    """Drop oldest completed entries so the list stays bounded."""
    while len(_refinement_drains) > limit:
        for i, e in enumerate(_refinement_drains):
            if e.state != "draining":
                _refinement_drains.pop(i)
                break
        else:
            # All entries still draining — leave them; we can't evict
            # in-flight drains without losing their task handle.
            break


def _find_drains_by_meeting(meeting_id: str) -> list[_DrainEntry]:
    return [e for e in _refinement_drains if e.meeting_id == meeting_id]


def _drain_entry_to_dict(entry: _DrainEntry) -> dict:
    return {
        "drain_id": entry.drain_id,
        "meeting_id": entry.meeting_id,
        "state": entry.state,
        "started_at": entry.started_at,
        "finished_at": entry.finished_at,
        "error": entry.error,
        "translate_calls": entry.translate_calls,
        "asr_calls": entry.asr_calls,
        "errors_at_stop": entry.errors_at_stop,
    }


# ── Phase 2: crash detection, health evaluator, loop lag monitor ──

# Sanitised crash metadata. Populated by the excepthook / asyncio
# exception handler on any unhandled exception. Exposed via /api/status
# with opaque error code only — no type names, no messages, no
# tracebacks. Full details live in the server log. [P2-1-i2]
_crash_state: dict | None = None


def _sanitised_crash_state() -> dict | None:
    """Return the sanitised crash metadata dict for /api/status."""
    return _crash_state


def _record_crash(component: str, exc: BaseException) -> None:
    """Record a crash in a way that doesn't leak internals to /api/status.

    The full traceback is logged at CRITICAL level to stdout; only a
    timestamp, component name, and opaque sha256 code land in
    ``_crash_state``. The UI red-dots the server pill based on the
    ``state`` field — nothing more. [P2-1-i2]
    """
    global _crash_state
    import hashlib
    import traceback as _tb

    try:
        tb_list = _tb.extract_tb(exc.__traceback__)
        if tb_list:
            frame = tb_list[-1]
            fingerprint = f"{type(exc).__name__}|{frame.filename}|{frame.lineno}"
        else:
            fingerprint = f"{type(exc).__name__}|<no traceback>"
        opaque = hashlib.sha256(fingerprint.encode()).hexdigest()
    except Exception:
        opaque = "unknown"

    logger.critical(
        "crash in %s: %s",
        component,
        exc,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    _crash_state = {
        "ts": round(time.monotonic(), 2),
        "component": component,
        "state": "crashed",
        "code": opaque,
    }


def _install_crash_hooks() -> None:
    """Install sys.excepthook + asyncio exception handler."""

    def _sys_hook(exc_type, exc, tb):
        _record_crash("sync", exc)

    sys.excepthook = _sys_hook

    def _async_hook(loop, context):
        exc = context.get("exception")
        task = context.get("task")
        task_name = task.get_name() if task else "unknown"

        # WebSocket disconnects (ping timeout, client gone) are normal network
        # events — don't record them as server crashes.
        if exc is not None:
            exc_name = type(exc).__name__
            if exc_name in ("ConnectionClosedError", "ConnectionClosedOK", "WebSocketDisconnect"):
                logger.debug("WS disconnect (not a crash): %s", exc)
                return
            # TLS handshake rejections from clients that don't trust our
            # self-signed admin cert (Safari/iOS, curl without -k, etc.)
            # surface here as asyncio `_accept_connection2` errors. They
            # are a *client* decision, not a server fault — demote to
            # debug so we don't log CRITICAL "crash in other" every time
            # a browser tab re-opens on a device that hasn't trusted
            # the cert yet.
            import ssl as _ssl

            if isinstance(exc, _ssl.SSLError):
                logger.debug("TLS handshake rejected by client (not a crash): %s", exc)
                return

        component = "other"
        if "tts-worker" in task_name:
            component = "tts_worker"
        elif "loop-lag" in task_name:
            component = "loop_lag_monitor"
        elif "health-evaluator" in task_name:
            component = "health_evaluator"
        elif "catchup" in task_name:
            component = "speaker_catchup"
        if exc is not None:
            _record_crash(component, exc)
        else:
            logger.warning("asyncio exception (no exc): %s", context.get("message"))

    try:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(_async_hook)
        # When a single sync callback blocks the loop for > 500 ms,
        # asyncio's debug mode logs the slow callback's repr + location.
        # This is how we pinpoint which periodic loop is causing the
        # 2.5 s periodic stall seen on the baseline. Cheap to enable.
        loop.slow_callback_duration = 0.5
        # `slow_callback_duration` is only honoured when the loop is in
        # debug mode. Without this line the threshold is silently
        # ignored and we never see the culprit. Also force the
        # asyncio logger to WARNING so the slow-callback message is
        # not buried below the default level.
        loop.set_debug(True)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
    except RuntimeError:
        # Not inside a running loop — the hook will be installed by
        # whichever task calls _install_crash_hooks at startup.
        pass


# ── Event-loop lag monitor [Phase 2] ──────────────────────────────

_LOOP_LAG_TICK_S = 0.5
_LOOP_LAG_WARN_MS = 250.0


async def _loop_lag_monitor() -> None:
    """Measure async event loop starvation every 500 ms.

    Records lag = (actual wake delta) - (scheduled sleep) in
    ``metrics.loop_lag_ms``. Warns when a single tick > 250 ms — means
    something is blocking the loop and TTS / WS fan-out will suffer
    regardless of queue depth.

    On a big spike (>1500 ms) also dumps a stack sample of every
    running task so we can see WHICH coroutine was blocking the loop.
    """
    while True:
        try:
            before = time.monotonic()
            await asyncio.sleep(_LOOP_LAG_TICK_S)
            lag_ms = (time.monotonic() - before - _LOOP_LAG_TICK_S) * 1000.0
            metrics.loop_lag_ms.append(max(0.0, lag_ms))
            if lag_ms > _LOOP_LAG_WARN_MS:
                logger.warning("Event loop lag: %.0fms", lag_ms)
            # Big spike → capture what was actually running so we can
            # pinpoint the culprit periodic loop. Cheap — only fires on
            # pathological lags, not routine 300 ms hiccups.
            if lag_ms > 1500:
                try:
                    tasks = asyncio.all_tasks()
                    interesting = []
                    for t in tasks:
                        if t.done():
                            continue
                        name = t.get_name()
                        # Grab a single top frame from each task
                        stack = t.get_stack(limit=1)
                        loc = "?"
                        if stack:
                            f = stack[0]
                            loc = f"{f.f_code.co_filename.split('/')[-1]}:{f.f_lineno} in {f.f_code.co_name}"
                        interesting.append(f"{name}@{loc}")
                    logger.warning(
                        "Loop-lag %dms stack sample: %s",
                        int(lag_ms),
                        " | ".join(interesting[:12]),
                    )
                except Exception:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("loop lag monitor crashed: %s", e)


# ── TTS health evaluator [Phase 2 + P1-3-i2 + P1-4-i2] ────────────

_TTS_NO_PROGRESS_STALL_S = 8.0
_TTS_STALL_DWELL_S = 3.0
_TTS_DEGRADED_DWELL_S = 1.5
_tts_stall_candidate_since: float | None = None
_tts_degraded_candidate_since: float | None = None


def _commit_tts_health(new_state: str, now: float) -> None:
    if new_state != metrics.tts_health_state:
        logger.warning("TTS health: %s → %s", metrics.tts_health_state, new_state)
        metrics.tts_health_state = new_state
        metrics.tts_health_since = now


async def _tts_health_evaluator() -> None:
    """Background state machine for TTS health [P1-3-i2 + P1-4-i2].

    Runs every 500 ms and mutates ``metrics.tts_health_state``. Uses
    separate candidate timestamps for stall vs degraded so hysteresis is
    honoured (the "committed since" timestamp is NOT used for dwell).
    Reads progress-based signals (last_delivery_at, _tts_inflight_started)
    as well as percentile and saturation signals so a no-progress hang
    is caught even when percentiles stop moving.
    """
    global _tts_stall_candidate_since, _tts_degraded_candidate_since
    while True:
        try:
            await asyncio.sleep(0.5)
            now = time.monotonic()

            qsize = _tts_queue.qsize() if _tts_queue else 0
            in_flight = _tts_in_flight
            queue_saturated = (
                in_flight >= _TTS_CONTAINER_MAX_CONCURRENCY and qsize >= _TTS_QUEUE_MAXSIZE
            )

            e2e_p95 = _percentile(sorted(metrics.end_to_end_lag_ms), 0.95)

            no_progress_stall = (
                in_flight > 0
                and metrics.last_delivery_at > 0
                and (now - metrics.last_delivery_at) > _TTS_NO_PROGRESS_STALL_S
            )
            oldest_inflight_age = 0.0
            if _tts_inflight_started:
                oldest_inflight_age = now - min(_tts_inflight_started.values())
            expected = max(
                _TTS_EXPECTED_SYNTH_DEFAULT_S,
                (metrics.tts_synth_ms_p95 or 0) / 1000.0,
            )
            stuck_request = oldest_inflight_age > (expected + 2.0)

            backend_degraded = bool(tts_backend and getattr(tts_backend, "degraded", False))

            stall_condition = (
                backend_degraded or no_progress_stall or (e2e_p95 is not None and e2e_p95 > 6000)
            )
            degraded_condition = (
                queue_saturated or stuck_request or (e2e_p95 is not None and e2e_p95 > 3500)
            )

            if stall_condition:
                if _tts_stall_candidate_since is None:
                    _tts_stall_candidate_since = now
                if now - _tts_stall_candidate_since >= _TTS_STALL_DWELL_S:
                    _commit_tts_health("stalled", now)
                _tts_degraded_candidate_since = None
            else:
                _tts_stall_candidate_since = None
                if degraded_condition:
                    if _tts_degraded_candidate_since is None:
                        _tts_degraded_candidate_since = now
                    if now - _tts_degraded_candidate_since >= _TTS_DEGRADED_DWELL_S:
                        _commit_tts_health("degraded", now)
                else:
                    _tts_degraded_candidate_since = None
                    _commit_tts_health("healthy", now)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("TTS health evaluator crashed: %s", e)


# Active meeting state file — persisted so crash recovery can detect
# an interrupted meeting and the audio writer process can keep running.
_ACTIVE_MEETING_FILE = Path("/tmp/meeting-scribe-active.json")


def _persist_active_meeting(meeting_id: str) -> None:
    """Write current meeting ID to disk for crash recovery."""
    import json as _json

    _ACTIVE_MEETING_FILE.write_text(
        _json.dumps({"meeting_id": meeting_id, "start_time": time.time()}) + "\n"
    )


def _clear_active_meeting() -> None:
    """Remove the active meeting marker on clean stop."""
    _ACTIVE_MEETING_FILE.unlink(missing_ok=True)


def _get_interrupted_meeting() -> str | None:
    """Check if a meeting was active when the server last crashed."""
    import json as _json

    if not _ACTIVE_MEETING_FILE.exists():
        return None
    try:
        data = _json.loads(_ACTIVE_MEETING_FILE.read_text())
        return data.get("meeting_id")
    except Exception:
        return None


# ── Real-time metrics ──────────────────────────────────────────


def _percentile(samples: list[float], q: float, *, presorted: bool = False) -> float | None:
    """Return the qth percentile (q in [0, 1]) or None for tiny windows.

    Guards against `statistics.quantiles` edge cases on empty / 1-sample
    windows [P1-6-i1]. Under _MIN_SAMPLES_FOR_PCT samples → None.
    """
    n = len(samples)
    if n < _MIN_SAMPLES_FOR_PCT:
        return None
    srt = samples if presorted else sorted(samples)
    # Nearest-rank method — simple, stable, no interpolation noise on
    # small windows.
    idx = min(n - 1, max(0, round(q * (n - 1))))
    return round(srt[idx], 2)


def _percentile_dict(samples) -> dict:
    """Serialise a rolling deque into a p50/p95/p99/sample_count dict.

    Returns null percentiles + actual sample_count when the window has
    fewer than _MIN_SAMPLES_FOR_PCT items. [P1-6-i1]
    """
    arr = list(samples)
    sample_count = len(arr)
    if sample_count < _MIN_SAMPLES_FOR_PCT:
        return {"p50": None, "p95": None, "p99": None, "sample_count": sample_count}
    srt = sorted(arr)
    return {
        "p50": _percentile(srt, 0.50, presorted=True),
        "p95": _percentile(srt, 0.95, presorted=True),
        "p99": _percentile(srt, 0.99, presorted=True),
        "sample_count": sample_count,
    }


class Metrics:
    """Server-wide performance metrics, reset per meeting."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        from collections import deque

        self.meeting_start: float = 0.0
        self.audio_chunks: int = 0
        self.audio_seconds: float = 0.0
        self.asr_events: int = 0
        self.asr_partials: int = 0
        self.asr_finals: int = 0
        self.asr_finals_filler_dropped: int = 0
        self.asr_finals_deduped: int = 0
        self.translations_submitted: int = 0
        self.translations_completed: int = 0
        self.translations_failed: int = 0
        self.translation_total_ms: float = 0.0
        self.last_asr_event_time: float = 0.0
        self.translate_warmup_ms: float = 0.0
        self.asr_load_ms: float = 0.0

        # ── TTS counters [Phase 1] ──────────────────────────────
        self.tts_submitted: int = 0
        self.tts_delivered: int = 0
        self.tts_dropped_filler: int = 0
        self.tts_dropped_stale_producer: int = 0
        self.tts_dropped_stale_worker: int = 0
        self.tts_dropped_pre_synth: int = 0
        self.tts_dropped_post_synth: int = 0
        self.tts_dropped_queue_full: int = 0
        self.tts_dropped_missing_origin: int = 0
        self.tts_synth_timeouts: int = 0
        self.last_delivery_at: float = 0.0
        self.tts_synth_ms_p95: float | None = None

        # ── TTS latency histograms (256-sample rolling windows) ─
        self.tts_synth_ms: deque[float] = deque(maxlen=256)
        self.upstream_lag_ms: deque[float] = deque(maxlen=256)
        self.tts_post_translation_lag_ms: deque[float] = deque(maxlen=256)
        self.end_to_end_lag_ms: deque[float] = deque(maxlen=256)

        # ── Listener transport metrics (per send_bytes) [P1-5-i2] ─
        self.listener_send_ms: deque[float] = deque(maxlen=256)
        self.listener_deliveries: int = 0
        self.listener_send_failed: int = 0
        self.listener_removed_on_send_error: int = 0

        # ── Event-loop lag monitor [Phase 2] ────────────────────
        self.loop_lag_ms: deque[float] = deque(maxlen=256)

        # ── TTS health state (mutated by _tts_health_evaluator) ─
        self.tts_health_state: str = "healthy"
        self.tts_health_since: float = 0.0

    @property
    def elapsed_seconds(self) -> float:
        if self.meeting_start == 0:
            return 0.0
        return time.monotonic() - self.meeting_start

    @property
    def asr_events_per_second(self) -> float:
        elapsed = self.elapsed_seconds
        return self.asr_events / elapsed if elapsed > 0 else 0.0

    @property
    def avg_translation_ms(self) -> float:
        if self.translations_completed == 0:
            return 0.0
        return self.translation_total_ms / self.translations_completed

    def to_dict(self) -> dict:
        tts_block = {
            "queue_depth": _tts_queue.qsize() if _tts_queue else 0,
            "queue_maxsize": _TTS_QUEUE_MAXSIZE,
            "workers_busy": _tts_in_flight,
            "workers_total": _TTS_WORKER_COUNT,
            "container_concurrency": _TTS_CONTAINER_MAX_CONCURRENCY,
            "submitted": self.tts_submitted,
            "delivered": self.tts_delivered,
            "drops": {
                "filler": self.tts_dropped_filler,
                "stale_producer": self.tts_dropped_stale_producer,
                "stale_worker": self.tts_dropped_stale_worker,
                "pre_synth": self.tts_dropped_pre_synth,
                "post_synth": self.tts_dropped_post_synth,
                "queue_full": self.tts_dropped_queue_full,
                "missing_origin": self.tts_dropped_missing_origin,
            },
            "timeouts": self.tts_synth_timeouts,
            "synth_ms": _percentile_dict(self.tts_synth_ms),
            "upstream_lag_ms": _percentile_dict(self.upstream_lag_ms),
            "tts_post_translation_lag_ms": _percentile_dict(self.tts_post_translation_lag_ms),
            "end_to_end_lag_ms": _percentile_dict(self.end_to_end_lag_ms),
            "oldest_inflight_age_ms": (
                int((time.monotonic() - min(_tts_inflight_started.values())) * 1000)
                if _tts_inflight_started
                else 0
            ),
            "last_delivery_age_ms": (
                int((time.monotonic() - self.last_delivery_at) * 1000)
                if self.last_delivery_at
                else None
            ),
            "health": self.tts_health_state,
            "health_since": round(self.tts_health_since, 2),
        }
        listener_block = {
            "connected": len(_audio_out_clients),
            "deliveries": self.listener_deliveries,
            "send_failed": self.listener_send_failed,
            "removed_on_send_error": self.listener_removed_on_send_error,
            "send_ms": _percentile_dict(self.listener_send_ms),
        }
        return {
            "elapsed_s": round(self.elapsed_seconds, 1),
            "audio_chunks": self.audio_chunks,
            "audio_s": round(self.audio_seconds, 1),
            "asr_events": self.asr_events,
            "asr_partials": self.asr_partials,
            "asr_finals": self.asr_finals,
            "asr_eps": round(self.asr_events_per_second, 1),
            "translations_submitted": self.translations_submitted,
            "translations_completed": self.translations_completed,
            "translations_failed": self.translations_failed,
            "avg_translation_ms": round(self.avg_translation_ms),
            "translate_warmup_ms": round(self.translate_warmup_ms),
            "asr_load_ms": round(self.asr_load_ms),
            "tts": tts_block,
            "listener": listener_block,
            "loop_lag_ms": _percentile_dict(self.loop_lag_ms),
            "crash": _sanitised_crash_state(),
        }


metrics = Metrics()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    global config, storage, resampler, translation_queue
    global asr_backend, translate_backend, diarize_backend, name_extractor

    config = ServerConfig.from_env()
    storage = MeetingStorage(config)
    resampler = Resampler()

    storage.recover_interrupted()
    storage.cleanup_retention()

    # Regulatory domain must be JP before any WiFi AP work happens. We
    # enforce + verify at startup so a cold boot doesn't produce an AP
    # that phones can't connect to (country 00 caps 5 GHz TX power). We
    # ALSO install the persistent modprobe file so cfg80211 starts in JP
    # on the next reboot without any runtime sudo.
    try:
        loop = asyncio.get_event_loop()
        persistent_ok = await loop.run_in_executor(None, _ensure_regdomain_persistent)
        if not persistent_ok:
            logger.warning(
                "could not install persistent /etc/modprobe.d/cfg80211-jp.conf — "
                "regdomain will still be set at runtime but may revert on reboot",
            )
        runtime_ok = await loop.run_in_executor(None, _ensure_regdomain)
        target_regdomain = _effective_regdomain()
        if runtime_ok:
            logger.info(
                "regulatory domain validated at startup: country=%s",
                target_regdomain,
            )
        else:
            logger.error(
                "STARTUP CHECK FAILED: regulatory domain is %r (expected %s) — "
                "WiFi hotspot will not work until this is fixed. Try: "
                "sudo iw reg set %s && sudo modprobe -r cfg80211 && sudo modprobe cfg80211",
                _current_regdomain(),
                target_regdomain,
                target_regdomain,
            )
    except Exception as exc:
        logger.error("regdomain startup check raised: %s", exc)

    # Auto-bring-up WiFi AP if wifi_mode != "off" in settings. This
    # replaces the old `sddc gb10 hotspot up` step that used to run
    # before meeting-scribe. wifi_up() handles regdomain, captive
    # portal, firewall, AP activation, and state file write.
    try:
        from meeting_scribe.wifi import _load_settings as _wifi_settings
        from meeting_scribe.wifi import build_config as _build_cfg
        from meeting_scribe.wifi import wifi_up as _wifi_up

        wifi_mode = _wifi_settings().get("wifi_mode", "off")
        if wifi_mode != "off":
            logger.info("WiFi auto-bring-up: mode=%s", wifi_mode)
            cfg = _build_cfg(wifi_mode, None, None, "a", 36)
            await _wifi_up(cfg)
            logger.info("WiFi AP started in %s mode", wifi_mode)
        else:
            logger.info("WiFi mode is 'off' — skipping AP bring-up")
    except Exception as exc:
        logger.error("WiFi auto-bring-up failed: %s", exc)

    # Module-scope _init_tts / _init_diarization are defined below so
    # _retry_failed_backends can reference them after lifespan init returns.
    # (They were previously nested inside lifespan, causing NameError in the
    # retry loop — silently eaten by logger.debug and leaving TTS/diarize
    # permanently dead if they failed to come up in the first 30s.)

    async def _init_name_extractor():
        global name_extractor
        if (
            config.name_extraction_backend not in ("llm", "auto")
            or config.translate_backend != "vllm"
        ):
            return
        try:
            from meeting_scribe.speaker.name_llm import LLMNameExtractor

            extractor = LLMNameExtractor(base_url=config.translate_vllm_url)
            await extractor.start()
            if extractor.available:
                name_extractor = extractor
        except Exception as e:
            logger.info("LLM name extraction disabled: %s", e)

    async def _init_translation_and_queue():
        global translate_backend, translation_queue
        await _init_translation(default_pair)
        if translate_backend and config.translate_enabled:
            translation_queue = TranslationQueue(
                maxsize=config.translate_queue_maxsize,
                concurrency=config.translate_queue_concurrency,
                timeout=config.translate_timeout_seconds,
                on_result=_broadcast_translation,
                languages=default_pair,
            )
            await translation_queue.start(translate_backend)

    async def _init_furigana():
        """Start the furigana backend. pykakasi is a hard dep — if it
        fails to load we raise so lifespan aborts and the operator sees
        the problem immediately instead of discovering mid-meeting that
        Japanese segments have no ruby text."""
        global furigana_backend
        from meeting_scribe.backends.furigana import FuriganaBackend

        furigana_backend = FuriganaBackend()
        await furigana_backend.start()
        logger.info("Furigana backend ready")

    from meeting_scribe.languages import parse_languages

    default_pair = parse_languages(config.default_language_pair)

    await asyncio.gather(
        _init_asr(default_pair),
        _init_translation_and_queue(),
        _init_tts(),  # module-scope below
        _init_diarization(),  # module-scope below
        _init_name_extractor(),
        _init_furigana(),
    )

    # Install crash hooks now that the loop is running.
    _install_crash_hooks()

    # Start background retry for any backends that failed to init
    _retry_task = asyncio.create_task(
        _retry_failed_backends(default_pair), name="retry-failed-backends"
    )

    # Phase 2: event-loop lag monitor + TTS health evaluator run for the
    # whole process lifetime (not just until backends are ready).
    _loop_lag_task = asyncio.create_task(_loop_lag_monitor(), name="loop-lag-monitor")
    _health_eval_task = asyncio.create_task(_tts_health_evaluator(), name="tts-health-evaluator")
    _silence_watchdog_task = asyncio.create_task(_silence_watchdog_loop(), name="silence-watchdog")

    logger.info("Meeting Scribe ready on port %d", config.port)

    # Dev mode: auto-resume interrupted meeting on server restart, but
    # ONLY if it was interrupted recently. An old interrupted meeting
    # would otherwise pull the entire audio through the reprocess pipeline
    # (5-chunk chunked diarization for a 33-min meeting took 90s of
    # event-loop starvation on 2026-04-13) in parallel with any new live
    # meeting, which manifests as audio drift, WS disconnects, and what
    # feels like "meetings randomly crashing".
    if _is_dev_mode():
        interrupted = _get_interrupted_meeting()
        if interrupted:
            stale_s: float | None = None
            try:
                pcm_path = storage._meeting_dir(interrupted) / "audio" / "recording.pcm"
                if pcm_path.exists():
                    stale_s = time.time() - pcm_path.stat().st_mtime
            except Exception:
                stale_s = None

            _AUTO_RESUME_MAX_AGE_S = 120.0
            if stale_s is None or stale_s > _AUTO_RESUME_MAX_AGE_S:
                logger.info(
                    "Dev mode: NOT auto-resuming interrupted meeting %s "
                    "(audio age=%s — over %.0fs threshold; it will appear in "
                    "the meetings list for manual re-open or reprocess).",
                    interrupted,
                    f"{stale_s:.0f}s" if stale_s is not None else "unknown",
                    _AUTO_RESUME_MAX_AGE_S,
                )
            else:
                logger.info(
                    "Dev mode: auto-resuming interrupted meeting %s (audio age=%.0fs)",
                    interrupted,
                    stale_s,
                )
                try:
                    result = await _do_resume_meeting(interrupted)
                    if result.status_code == 200:
                        logger.info("Dev mode: meeting %s resumed successfully", interrupted)
                    else:
                        raw_body = getattr(result, "body", b"") or b""
                        body = (
                            raw_body.decode()
                            if isinstance(raw_body, bytes)
                            else bytes(raw_body).decode()
                        )
                        logger.warning(
                            "Dev mode: meeting resume returned %d: %s",
                            result.status_code,
                            body[:200],
                        )
                except Exception as e:
                    logger.warning("Dev mode: auto-resume failed: %s", e)

    # ── Slide translation worker ──────────────────────────────
    global slide_job_runner, slides_enabled
    try:
        from meeting_scribe.slides.job import SlideJobRunner
        from meeting_scribe.slides.worker import check_worker_available

        worker_ok = await check_worker_available()
        if worker_ok:

            async def _slide_translate_fn(
                text: str,
                source_lang: str,
                target_lang: str,
                system_prompt: str = "",
                max_tokens: int = 128,
            ) -> str | None:
                """Translate slide text via shared vLLM with lower priority.

                Reads ``slide_translate_url`` from runtime-config per call
                so a hot-reload (Phase 6/7 of the 3.6 plan) flips the slide
                pipeline's endpoint alongside the live-translation one.
                Falls back to ``translate_vllm_url`` so production behavior
                is unchanged when the runtime knob is unset.
                """
                if translate_backend is None:
                    return None

                from meeting_scribe import runtime_config as _rc

                client = getattr(translate_backend, "_client", None)
                model = getattr(translate_backend, "_model", None)
                if client is None or model is None:
                    return None
                base_url = _rc.get("slide_translate_url", config.translate_vllm_url)
                try:
                    resp = await client.post(
                        f"{base_url.rstrip('/')}/v1/chat/completions",
                        json={
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": text},
                            ],
                            "temperature": 0.3,
                            "max_tokens": max_tokens,
                            "stream": False,
                            "chat_template_kwargs": {"enable_thinking": False},
                            # Same priority as coding agent (10) — lower than
                            # live transcript (-10) under vLLM's priority scheduler
                            "priority": 10,
                        },
                    )
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"].strip()
                except Exception as exc:
                    logger.warning("Slide translation failed: %s", exc)
                    return None

            slide_job_runner = SlideJobRunner(
                meetings_dir=storage._meetings_dir,
                translate_fn=_slide_translate_fn,
                broadcast_fn=_broadcast_json,
            )
            slides_enabled = True
            logger.info("Slide translation enabled (worker image found)")
        else:
            logger.info("Slide translation disabled (LibreOffice not found)")
    except Exception as exc:
        logger.warning("Slide translation init failed: %s", exc)

    # Inline post-ready bookkeeping: write `last-good-boot`, clear any
    # stale `BOOT_BLOCKED` marker, and tell systemd we're ready. Doing
    # this in-process (rather than via a separate ExecStartPost hook)
    # guarantees we only record "boot is healthy" *after* every backend
    # has actually initialised, because if the lifespan coroutine hadn't
    # run to this point we wouldn't reach these calls.
    try:
        from meeting_scribe import preflight as _pf

        _pf.write_post_ready()
    except Exception as e:
        logger.warning("post-ready bookkeeping failed (non-fatal): %r", e)

    _notify_systemd("READY=1")

    yield

    _notify_systemd("STOPPING=1")

    for _t in (_retry_task, _loop_lag_task, _health_eval_task, _silence_watchdog_task):
        _t.cancel()
    for _t in (_retry_task, _loop_lag_task, _health_eval_task, _silence_watchdog_task):
        try:
            await _t
        except asyncio.CancelledError:
            pass

    # Shutdown
    if name_extractor:
        await name_extractor.stop()
    if diarize_backend:
        await diarize_backend.stop()
    if translation_queue:
        await translation_queue.stop()
    if tts_backend:
        await tts_backend.stop()
    if asr_backend:
        await asr_backend.stop()
    if translate_backend:
        await translate_backend.stop()


async def _retry_failed_backends(default_pair: list[str] | tuple[str, ...]) -> None:
    """Background task: periodically retry init for backends that failed at startup.

    Checks every 10s. Stops retrying each backend once it succeeds.
    This handles the case where containers are still loading when the server starts.
    """
    while True:
        await asyncio.sleep(10)
        try:
            if translate_backend is None and config.translate_enabled:
                logger.info("Retrying translation backend init...")
                await _init_translation(default_pair)
                if translate_backend:
                    global translation_queue
                    if translation_queue is None:
                        translation_queue = TranslationQueue(
                            maxsize=config.translate_queue_maxsize,
                            concurrency=config.translate_queue_concurrency,
                            timeout=config.translate_timeout_seconds,
                            on_result=_broadcast_translation,
                            languages=default_pair,
                        )
                        await translation_queue.start(translate_backend)
                    logger.info("Translation backend recovered")

            if diarize_backend is None:
                logger.info("Retrying diarization backend init...")
                await _init_diarization()
                if diarize_backend:
                    logger.info("Diarization backend recovered")
            elif getattr(diarize_backend, "degraded", False):
                # Backend exists but CUDA crashed — check if container restarted
                try:
                    healthy = await diarize_backend.check_health()
                    if healthy:
                        logger.info("Diarization recovered from degraded state")
                    else:
                        await _restart_container("scribe-diarization")
                except Exception:
                    await _restart_container("scribe-diarization")

            if tts_backend is None:
                logger.info("Retrying TTS backend init...")
                await _init_tts()
                if tts_backend:
                    logger.info("TTS backend recovered")
            elif tts_backend.degraded:
                # TTS exists but CUDA crashed — check if container restarted
                healthy = await tts_backend.check_health()
                if healthy:
                    logger.info("TTS recovered from degraded state (container restarted)")
                else:
                    await _restart_container("scribe-tts")

            if asr_backend is None:
                logger.info("Retrying ASR backend init...")
                await _init_asr(default_pair)
                if asr_backend:
                    logger.info("ASR backend recovered")

            # Continuous background health polling [Phase 2]: keep looping
            # for the whole process lifetime so we detect late failures
            # (e.g. TTS container CUDA dispatch failure mid-meeting) and
            # auto-recover via the existing degraded-check paths above.
            # The old "return on all-ready" branch was removed — it
            # silenced every post-startup regression.
        except Exception as e:
            # Upgraded from debug → warning: this handler previously hid
            # real failures (including NameError from the dead
            # `_init_tts`/`_init_diarization` references that silently broke
            # TTS/diarization recovery forever). If you land here, the retry
            # loop is not actually retrying — fix the caller, don't just
            # accept the log line.
            logger.warning("Backend retry error: %r", e, exc_info=True)


async def _init_tts() -> None:
    """Initialize the TTS backend. Module-scope so `_retry_failed_backends`
    can call it after lifespan startup has returned. The retry loop used
    to reference a nested version of this function that did not exist at
    module scope, which silently raised NameError and left TTS permanently
    dead if it missed the initial 30s startup window.

    Idempotent: a successful call sets the module-level `tts_backend` and
    `_tts_semaphore` globals and starts the FIFO TTS worker. If `tts_backend`
    is already set we are a no-op.
    """
    global tts_backend, _tts_semaphore
    if tts_backend is not None:
        return
    try:
        from meeting_scribe.backends.tts_qwen3 import Qwen3TTSBackend

        tts_url = (
            config.omni_tts_url
            or config.tts_vllm_url
            or (config.translate_vllm_url if config.translate_backend == "vllm" else None)
        )
        tts = Qwen3TTSBackend(vllm_url=tts_url or None)
        # Retry: TTS container may still be starting during parallel init.
        for attempt in range(15):
            await tts.start()
            if tts.available:
                tts_backend = tts
                _tts_semaphore = asyncio.Semaphore(1)
                _start_tts_worker()
                logger.info("TTS backend ready (url=%s)", tts_url)
                return
            last_err = getattr(tts, "_last_error", None)
            if attempt < 14:
                logger.info(
                    "TTS not ready (attempt %d/15, last_error=%r), retrying in 2s...",
                    attempt + 1,
                    last_err,
                )
                await asyncio.sleep(2)
        logger.warning(
            "TTS disabled: not available after 30s of retries (last_error=%r)",
            getattr(tts, "_last_error", None),
        )
    except Exception as e:
        logger.warning("TTS disabled: %r", e, exc_info=True)


async def _init_diarization() -> None:
    """Initialize the diarization backend. Module-scope for the retry loop.

    See `_init_tts` for why this was extracted from its previous nested
    definition inside `lifespan`.
    """
    global diarize_backend
    if diarize_backend is not None:
        return
    if not config.diarize_enabled:
        return
    try:
        from meeting_scribe.backends.diarize_sortformer import SortformerBackend
        from meeting_scribe.speaker.verification import SpeakerVerifier

        verifier = SpeakerVerifier(enrollment_store)
        # Rolling-window diarization config:
        # window_seconds=16 — pyannote needs ~15s of multi-speaker audio to
        # separate clusters; shorter chunks collapse everything into one.
        # flush_interval_seconds=4 — re-diarize every 4s of new audio; each
        # flush re-runs the FULL window so pyannote always has context.
        # max_speakers=6 — typical 4-6 person meetings.
        diarize_be = SortformerBackend(
            url=config.diarize_url,
            verifier=verifier,
            flush_interval_seconds=4.0,
            window_seconds=16.0,
        )
        await diarize_be.start(max_speakers=6)
        diarize_backend = diarize_be
        logger.info("Diarization backend ready")
    except Exception as e:
        logger.warning("Diarization disabled: %r", e, exc_info=True)


async def _init_translation(default_pair: list[str] | tuple[str, ...] = ("en", "ja")) -> None:
    """Initialize translation backend based on config."""
    global translate_backend

    if not config.translate_enabled:
        return

    t0 = time.monotonic()
    try:
        from meeting_scribe.backends.translate_vllm import VllmTranslateBackend

        # Omni-consolidation override takes precedence; then realtime URL; then main URL.
        realtime_url = (
            config.omni_translate_url
            or config.translate_realtime_vllm_url
            or config.translate_vllm_url
        )
        be = VllmTranslateBackend(
            base_url=realtime_url,
            model=config.translate_vllm_model or None,
        )
        await be.start()
        # First request may be slow (model compilation/warmup) — use extended timeout
        old_client = be._client
        be._client = httpx.AsyncClient(timeout=120.0)
        if old_client:
            await old_client.aclose()
        warmup = await be.translate("Hello", default_pair[1], default_pair[0])
        await be._client.aclose()
        be._client = httpx.AsyncClient(timeout=be._timeout)
        metrics.translate_warmup_ms = (time.monotonic() - t0) * 1000
        translate_backend = be
        logger.info(
            "Translation: vLLM (%.0fms, test: 'Hello' → '%s')",
            metrics.translate_warmup_ms,
            warmup[:30],
        )
    except Exception as e:
        logger.warning("Translation vLLM backend failed: %s", e)


async def _init_asr(default_pair: list[str] | tuple[str, ...] = ("en", "ja")) -> None:
    """Initialize the Qwen3-ASR vLLM backend. This is the only ASR path
    on GB10 — the WhisperLiveKit fallback was removed 2026-04-13."""
    global asr_backend

    from meeting_scribe.backends.asr_vllm import VllmASRBackend

    be = VllmASRBackend(
        base_url=(config.omni_asr_url or config.asr_vllm_url),
        languages=default_pair,
    )
    be.set_event_callback(_process_event)
    t0 = time.monotonic()
    try:
        await be.start()
    except Exception as e:
        logger.warning("vLLM ASR failed to start: %s", e)
        return
    metrics.asr_load_ms = (time.monotonic() - t0) * 1000
    asr_backend = be
    logger.info("ASR: vLLM Qwen3-ASR loaded in %.0fms", metrics.asr_load_ms)


# --- App ---

app = FastAPI(title="Meeting Scribe", lifespan=lifespan)
STATIC_DIR = Path(__file__).parent.parent.parent / "static"

# GZip compression — cuts 67K CSS and 134K JS down to ~15K and ~35K over WiFi
from starlette.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=500)

# ── Path validation ──────────────────────────────────────────
# Prevent directory traversal via meeting_id or segment_id


def _safe_meeting_dir(meeting_id: str) -> Path | None:
    """Resolve a meeting directory path, rejecting traversal attacks."""
    if not meeting_id or ".." in meeting_id or "/" in meeting_id or "\\" in meeting_id:
        return None
    meeting_dir = (storage._meetings_dir / meeting_id).resolve()
    if not meeting_dir.is_relative_to(storage._meetings_dir.resolve()):
        return None
    return meeting_dir


def _safe_segment_path(meeting_id: str, subdir: str, filename: str) -> Path | None:
    """Resolve a file path within a meeting subdirectory, rejecting traversal."""
    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir:
        return None
    if ".." in filename or "/" in filename or "\\" in filename:
        return None
    path = (meeting_dir / subdir / filename).resolve()
    if not path.is_relative_to(meeting_dir.resolve()):
        return None
    return path


# ── Hotspot access control ────────────────────────────────────
# Clients on the hotspot subnet (10.42.0.x) are restricted to the
# guest live-view page only. No static files, no admin UI, no past
# meetings, no controls. Even the allowed APIs require an active meeting.

HOTSPOT_SUBNET = os.environ.get("SCRIBE_HOTSPOT_SUBNET", "10.42.0.")

# Paths hotspot clients can access (only during an active meeting,
# except / and captive probes which are always allowed)
_HOTSPOT_ALWAYS_ALLOWED = frozenset(
    (
        "/",
        "/reader",
        "/api/status",
        "/api/languages",
        "/api/captive",  # RFC 8910 captive-portal probe
        "/hotspot-detect.html",
        "/generate_204",
        "/gen_204",
        "/canonical.html",
        "/connecttest.txt",
        "/ncsi.txt",
        "/success.txt",
        "/redirect",
    )
)

_HOTSPOT_MEETING_ALLOWED = (
    "/api/status",
    "/api/languages",
    "/api/ws/view",
    "/api/ws/audio-out",
    # Guest-side audio-chain diagnostics. Without this, hotspot clients
    # POSTing their AudioContext / WS / decode state back to us got 403
    # and we had zero visibility into why playback was silent — exactly
    # the hole that hid the client-side bug 2026-04-15.
    "/api/diag/listener",
)


def _is_hotspot_client(request_or_ws) -> bool:
    """Check if the request comes from the hotspot WiFi subnet."""
    client = getattr(request_or_ws, "client", None)
    client_ip = client.host if client else ""
    return client_ip.startswith(HOTSPOT_SUBNET)


def _is_guest_scope(request_or_ws) -> bool:
    """Return True if this request should be treated as a guest request.

    A request is in guest scope if EITHER:
      - the client IP is on the hotspot subnet (existing check), OR
      - the request arrived over plain HTTP (scheme == 'http' or 'ws').

    The second condition hardens the HTTP-only guest listener on port 80:
    even a LAN user who hits ``http://<gb10-lan-ip>:80/`` by mistake gets
    the guest-restricted view rather than admin. Admin is reachable only
    via the HTTPS listener (https://<gb10-lan-ip>:8080/).
    """
    if _is_hotspot_client(request_or_ws):
        return True
    url = getattr(request_or_ws, "url", None)
    scheme = getattr(url, "scheme", "") if url is not None else ""
    return scheme in ("http", "ws")


def _has_active_meeting() -> bool:
    return current_meeting is not None and current_meeting.state.value == "recording"


def _norm_lang(code: str | None) -> str:
    """Normalize a language code to ISO 639-1 for comparison.

    ``"en-US"`` / ``"en_US"`` / ``"EN"`` → ``"en"``.

    Used everywhere we match a listener's preferred_language against a
    translation's target_language. Without normalization, a browser
    sending its locale (``"en-US"``) would never match the translation
    backend's output (``"en"``) and the listener would silently receive
    no audio.
    """
    if not code:
        return ""
    head = code.strip().split("-", 1)[0].split("_", 1)[0]
    return head.lower()


# Demand-driven multi-target translation + TTS fan-out. Default off
# until TranscriptEvent consumers (journal, replay, WS broadcast, UI
# state, exports, captions) are audited for one-segment/many-translations.
# Gates both the translation queue fan-out and the per-target TTS keying.
from meeting_scribe.translation.queue import MULTI_TARGET_ENABLED as _MULTI_TARGET_ENABLED


def _compute_translation_demand(
    event: TranscriptEvent,
) -> tuple[frozenset[str], frozenset[str]]:
    """Compute baseline and optional translation targets for a segment.

    Baseline = the meeting's language_pair cross-translation (always
    translated so journal/captions/exports have it). For monolingual
    meetings (length-1 ``language_pair``) the baseline is always empty —
    no translation work runs. Optional = live audio-out listener
    ``preferred_language`` values that are NOT in baseline and NOT
    equal to the source language. Optional targets are droppable under
    load.

    Under legacy mode the returned tuple still works: the queue uses
    baseline when demand is supplied, optional is dropped at synth time
    by the TTS listener filter.
    """
    source = _norm_lang(event.language)
    baseline: set[str] = set()
    if current_meeting and current_meeting.language_pair:
        pair = tuple(current_meeting.language_pair)
        if len(pair) == 2:
            a, b = _norm_lang(pair[0]), _norm_lang(pair[1])
            if source == a and b:
                baseline.add(b)
            elif source == b and a:
                baseline.add(a)
    optional: set[str] = set()
    for pref in _audio_out_prefs.values():
        lang = _norm_lang(getattr(pref, "preferred_language", "") or "")
        if not lang or lang == source:
            continue
        if lang in baseline:
            continue
        optional.add(lang)
    return frozenset(baseline), frozenset(optional)


def _listener_tts_demand(target_lang: str) -> set[str]:
    """Return the set of voice_modes listeners want for ``target_lang``.

    Empty set means no listener wants TTS for this language → skip synth.
    Uses normalized language comparison so ``en-US`` listener matches
    ``en`` translation.
    """
    target_norm = _norm_lang(target_lang)
    modes: set[str] = set()
    for pref in _audio_out_prefs.values():
        pref_lang = _norm_lang(getattr(pref, "preferred_language", "") or "")
        if pref_lang and pref_lang != target_norm:
            continue
        modes.add(getattr(pref, "voice_mode", "studio"))
    return modes


# TTS deferral feature flag. When True, _broadcast_translation skips TTS
# on events with empty speakers[] and lets the catch-up loop fire TTS
# once it has attached a real speaker attribution. Defaults off in Part A;
# Part B of the 2026-04 speaker-separation refactor flips it on.
_TTS_DEFER_UNTIL_CATCH_UP = os.environ.get("SCRIBE_TTS_DEFER_UNTIL_CATCH_UP", "0") == "1"


@app.middleware("http")
async def hotspot_guard(request: fastapi.Request, call_next):
    """Restrict guest-scope requests to the guest live view.

    A request is "guest-scope" when it comes from a hotspot-subnet IP OR
    it arrived over plain HTTP. Both are routed through the same guest
    allowlist so the HTTP-only guest listener on port 80 and the
    hotspot-subnet IP check stay in lockstep.
    """
    if not _is_guest_scope(request):
        return await call_next(request)

    path = request.url.path

    # Static files are always allowed for guest-scope requests — guest.html
    # is self-contained but we want /static/*.html and captive-portal probes
    # served for other OS probes.
    if path.startswith("/static/"):
        return await call_next(request)

    # Always allow: guest index page + captive portal probes
    if path in _HOTSPOT_ALWAYS_ALLOWED:
        return await call_next(request)

    # Meeting-gated: only during active recording
    if _has_active_meeting() and any(path.startswith(p) for p in _HOTSPOT_MEETING_ALLOWED):
        return await call_next(request)

    # Slide viewer: guests can GET slide metadata and images during a meeting.
    # Write endpoints (POST upload, PUT advance) are blocked here AND guarded
    # by _require_admin() defense-in-depth.
    if (
        _has_active_meeting()
        and request.method == "GET"
        and "/slides" in path
        and path.startswith("/api/meetings/")
    ):
        return await call_next(request)

    # Block everything else — admin API, past meetings, controls.
    return JSONResponse(
        {"error": "Not available on guest WiFi (use https://<gb10>:8080/ for admin)"},
        status_code=403,
    )


# ── HTML page cache ──────────────────────────────────────────
# Read all HTML pages into memory at import time. Avoids disk I/O on
# every request (~10-50ms per read_text on slow storage).

_HTML: dict[str, str] = {}


def _cache_html() -> None:
    """Populate _HTML cache from static files."""
    pages = {
        "index": STATIC_DIR / "index.html",
        "guest": STATIC_DIR / "guest.html",
        "portal": STATIC_DIR / "portal.html",
        "reader": STATIC_DIR / "reader.html",
        "demo": STATIC_DIR / "demo" / "index.html",
        "voice-clone": STATIC_DIR / "demo" / "voice-clone.html",
    }
    for key, path in pages.items():
        if path.exists():
            _HTML[key] = path.read_text()


_cache_html()


# Static files with caching headers — CSS/JS rarely change in dev,
# and on guest hotspot the browser re-downloads 300KB+ without them.
@app.middleware("http")
async def request_timing(request: fastapi.Request, call_next):
    """Log any HTTP request that takes > 500 ms end-to-end.

    Added 2026-04-15 to pinpoint which handler was causing periodic
    ~2.5 s event-loop stalls that kept killing the audio-out WS. The
    asyncio slow-callback warning told us "a handler" was slow but
    not which one. This middleware closes that gap — the next slow
    request lands in the log with its exact URL + duration, no more
    guessing.
    """
    t0 = time.monotonic()
    response = await call_next(request)
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    if elapsed_ms > 500:
        logger.warning(
            "SLOW HTTP %s %s — %.0f ms (client=%s)",
            request.method,
            request.url.path,
            elapsed_ms,
            request.client.host if request.client else "?",
        )
    return response


@app.middleware("http")
async def static_cache_headers(request: fastapi.Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "public, max-age=3600"
    return response


# Static files mounted AFTER middleware — middleware intercepts first
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index(request: fastapi.Request) -> HTMLResponse:
    if _is_guest_scope(request):
        # Any hit on the portal root from a hotspot client ACKs them
        # so subsequent captive-portal probes return "not captive" and
        # the CNA sheet (iOS) / sign-in notification (Android) dismisses.
        # Without this the CNA stays open forever because iOS keeps
        # polling hotspot-detect.html and never sees the Success body.
        _captive_ack(request)
        if request.cookies.get("scribe_portal") == "done":
            return HTMLResponse(_HTML.get("guest", ""))
        return HTMLResponse(_HTML.get("portal", ""))
    return HTMLResponse(_HTML.get("index", ""))


@app.get("/reader")
async def reader_view(request: fastapi.Request) -> HTMLResponse:
    """Large-font, text-only reader view for guest displays (iPad, TV)."""
    return HTMLResponse(_HTML.get("reader", ""))


@app.get("/demo")
@app.get("/demo/")
async def demo_landing() -> HTMLResponse:
    """Demo landing page — links to all interactive demos."""
    return HTMLResponse(_HTML.get("demo", ""))


@app.get("/demo/guest")
async def demo_guest_page() -> HTMLResponse:
    """Preview of the guest view from the admin side."""
    return HTMLResponse(_HTML.get("guest", ""))


@app.get("/demo/voice-clone")
async def demo_voice_clone_page() -> HTMLResponse:
    """Standalone voice clone demo — own mic, own pipeline."""
    return HTMLResponse(_HTML.get("voice-clone", ""))


@app.post("/api/demo/voice-clone")
async def demo_voice_clone(request: fastapi.Request) -> JSONResponse:
    """Standalone voice clone: voice reference + text → translate → TTS.

    JSON body:
      - audio_b64: base64 s16le 16kHz mono PCM (voice reference)
      - text: text to translate and speak (if empty, ASR transcribes the audio)
      - target_language: BCP-47 target language code

    Returns JSON with original_text, translated_text, audio_b64 (s16le PCM),
    source_language, target_language, sample_rate.
    Completely independent of any running meeting.
    """
    import base64
    import io

    import numpy as np

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    audio_b64 = body.get("audio_b64", "")
    text = body.get("text", "").strip()
    tgt_lang = body.get("target_language", "en")

    if not audio_b64:
        return JSONResponse({"error": "audio_b64 is required"}, status_code=400)

    # Decode voice reference from base64 s16le PCM
    try:
        raw = base64.b64decode(audio_b64)
        pcm = np.frombuffer(raw, dtype=np.int16)
        voice_f32 = pcm.astype(np.float32) / 32768.0
    except Exception as e:
        return JSONResponse({"error": f"Could not decode audio: {e}"}, status_code=400)

    if len(voice_f32) < 8000:
        return JSONResponse(
            {"error": "Voice reference too short (need at least 0.5s)"}, status_code=400
        )

    # If no text provided, transcribe the voice reference via ASR
    src_lang = None
    if not text:
        import soundfile as sf  # type: ignore[import-untyped]

        wav_buf = io.BytesIO()
        sf.write(wav_buf, voice_f32, 16000, format="WAV")
        asr_b64 = base64.b64encode(wav_buf.getvalue()).decode()

        try:
            async with httpx.AsyncClient(timeout=30) as c:
                asr_resp = await c.post(
                    f"{config.asr_vllm_url}/v1/chat/completions",
                    json={
                        "model": "Qwen/Qwen3-ASR-1.7B",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_audio",
                                        "input_audio": {"data": asr_b64, "format": "wav"},
                                    },
                                    {"type": "text", "text": "<|startoftranscript|>"},
                                ],
                            }
                        ],
                        "max_tokens": 512,
                        "temperature": 0.0,
                    },
                )
                asr_resp.raise_for_status()
                text = asr_resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return JSONResponse({"error": f"ASR failed: {e}"}, status_code=502)

    if not text:
        return JSONResponse(
            {"error": "No text to translate (speech not detected)"}, status_code=400
        )

    # Detect source language
    cjk = sum(1 for ch in text if "\u3000" <= ch <= "\u9fff" or "\uff00" <= ch <= "\uffef")
    src_lang = "ja" if cjk > len(text) * 0.3 else "en"
    if tgt_lang == src_lang:
        tgt_lang = "en" if src_lang == "ja" else "ja"

    # Translate
    try:
        if translate_backend:
            translated = await translate_backend.translate(text, src_lang, tgt_lang)
        else:
            translated = text
    except Exception as e:
        return JSONResponse({"error": f"Translation failed: {e}"}, status_code=502)

    # TTS — synthesize in the speaker's cloned voice
    audio_out_b64 = None
    sample_rate = 16000
    try:
        if tts_backend and tts_backend.available:
            tts_audio = await tts_backend.synthesize(
                text=translated,
                language=tgt_lang,
                voice_reference=voice_f32,
            )
            out_pcm = (np.clip(tts_audio, -1.0, 1.0) * 32767).astype(np.int16)
            audio_out_b64 = base64.b64encode(out_pcm.tobytes()).decode()
    except Exception as e:
        logger.warning("Demo voice-clone TTS failed: %s", e)

    from meeting_scribe.languages import is_tts_native

    return JSONResponse(
        {
            "original_text": text,
            "translated_text": translated,
            "source_language": src_lang,
            "target_language": tgt_lang,
            "audio_b64": audio_out_b64,
            "sample_rate": sample_rate,
            "tts_supported": is_tts_native(tgt_lang),
        }
    )


@app.get("/how-it-works")
async def how_it_works() -> RedirectResponse:
    return RedirectResponse("/static/how-it-works.html")


# ── WiFi QR Code ─────────────────────────────────────────────


def _qr_svg(data: str) -> str:
    """Generate an SVG QR code for the given data string."""
    import io

    import qrcode  # type: ignore[import-untyped]
    import qrcode.image.svg  # type: ignore[import-untyped]

    img = qrcode.make(data, image_factory=qrcode.image.svg.SvgPathImage, box_size=10)
    buf = io.BytesIO()
    img.save(buf)
    return buf.getvalue().decode()


def _wifi_qr_svg(ssid: str, password: str) -> str:
    """Generate an SVG QR code for WiFi auto-join."""
    return _qr_svg(f"WIFI:T:WPA;S:{ssid};P:{password};;")


def _load_hotspot_state() -> dict | None:
    """Read hotspot state written by `sddc gb10 hotspot up`."""
    import json as _json

    state_file = Path("/tmp/meeting-hotspot.json")
    if not state_file.exists():
        return None
    try:
        return _json.loads(state_file.read_text())
    except Exception:
        return None


AP_CON_NAME = "DellDemo-AP"
AP_IP = "10.42.0.1"
HOTSPOT_STATE_FILE = Path("/tmp/meeting-hotspot.json")

# Regulatory domain enforcement.
#
# The regdomain can silently drift back to the default "world" domain
# (country 00) if cfg80211 picks up a foreign beacon during a scan, or
# if cfg80211 is reloaded. With country 00 on 5 GHz the kernel caps
# transmit power to a useless level and phones can't associate reliably.
#
# The target country is CONFIGURABLE via ``config.wifi_regdomain`` (set
# via the ``SCRIBE_WIFI_REGDOMAIN`` env var or persisted in
# ``~/.config/meeting-scribe/settings.json`` — the admin UI writes to
# that path through the ``PUT /api/admin/settings`` endpoint).
#
# Defense in depth:
#   1. /etc/modprobe.d/cfg80211-<code>.conf sets ieee80211_regdom at
#      module load time, so boots with meeting-scribe installed start
#      in the configured country.
#   2. _ensure_regdomain() runs ``iw reg set <code>`` before every AP
#      rotation and at server startup, and VERIFIES via ``iw reg get``
#      that the setting took effect.
#   3. The verify step logs an error if the code couldn't be set —
#      no silent drift.

# Persisted overrides — survives process restart. The admin UI's
# settings endpoint writes here. Env vars still override.
SETTINGS_OVERRIDE_FILE = Path.home() / ".config" / "meeting-scribe" / "settings.json"

# Default regdomain used when config hasn't been initialized yet (e.g.
# during unit tests that don't run the lifespan startup).
_DEFAULT_REGDOMAIN = "JP"


_settings_cache: dict | None = None
_settings_cache_mtime: float = 0.0


def _load_settings_override() -> dict:
    """Read persisted admin-UI settings overrides. Best-effort.

    Cached by file mtime — safe for the /api/status hot path (~3s poll).
    Invalidated on write via _save_settings_override().
    """
    global _settings_cache, _settings_cache_mtime
    import json as _json

    if not SETTINGS_OVERRIDE_FILE.exists():
        _settings_cache = {}
        return {}
    try:
        mtime = SETTINGS_OVERRIDE_FILE.stat().st_mtime
        if _settings_cache is not None and mtime == _settings_cache_mtime:
            return _settings_cache
        data = _json.loads(SETTINGS_OVERRIDE_FILE.read_text())
        result = data if isinstance(data, dict) else {}
        _settings_cache = result
        _settings_cache_mtime = mtime
        return result
    except (OSError, _json.JSONDecodeError):
        return _settings_cache or {}


def _save_settings_override(updates: dict) -> None:
    """Merge ``updates`` into the persisted settings override file."""
    global _settings_cache, _settings_cache_mtime
    import json as _json

    SETTINGS_OVERRIDE_FILE.parent.mkdir(parents=True, exist_ok=True)
    current = _load_settings_override()
    current.update(updates)
    tmp = SETTINGS_OVERRIDE_FILE.with_suffix(SETTINGS_OVERRIDE_FILE.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(_json.dumps(current, indent=2) + "\n")
    tmp.replace(SETTINGS_OVERRIDE_FILE)
    _settings_cache = current
    _settings_cache_mtime = SETTINGS_OVERRIDE_FILE.stat().st_mtime


def _effective_regdomain() -> str:
    """Return the regulatory domain to enforce.

    Priority:
      1. Live ``config.wifi_regdomain`` (from env var, profile, or
         persisted settings override).
      2. Persisted override file (read directly if config isn't initialized).
      3. ``_DEFAULT_REGDOMAIN`` fallback.

    The returned code is upper-cased to match what ``iw`` expects.
    """
    try:
        from_config = getattr(config, "wifi_regdomain", None)
    except NameError:
        from_config = None
    if isinstance(from_config, str) and from_config.strip():
        return from_config.strip().upper()

    override = _load_settings_override().get("wifi_regdomain")
    if isinstance(override, str) and override.strip():
        return override.strip().upper()
    return _DEFAULT_REGDOMAIN


def _regdomain_modprobe_path(country: str) -> Path:
    """Canonical modprobe conf path for a given 2-letter country code."""
    safe = country.strip().upper() or _DEFAULT_REGDOMAIN
    return Path("/etc/modprobe.d") / f"cfg80211-{safe.lower()}.conf"


# Curated list of 2-letter ISO 3166-1 country codes the WiFi card supports.
# The MT7925e mt7925e driver + cfg80211 regdb cover every ISO country, but
# surfacing all ~250 in a dropdown is unusable. This list is the "useful
# demo deployment" subset; add a country via PR or the
# ``SCRIBE_WIFI_REGDOMAIN`` env var (the env var accepts any valid code).
_WIFI_REGDOMAIN_OPTIONS: tuple[tuple[str, str], ...] = (
    ("JP", "Japan"),
    ("US", "United States"),
    ("CA", "Canada"),
    ("GB", "United Kingdom"),
    ("IE", "Ireland"),
    ("DE", "Germany"),
    ("FR", "France"),
    ("IT", "Italy"),
    ("ES", "Spain"),
    ("PT", "Portugal"),
    ("NL", "Netherlands"),
    ("BE", "Belgium"),
    ("LU", "Luxembourg"),
    ("CH", "Switzerland"),
    ("AT", "Austria"),
    ("SE", "Sweden"),
    ("NO", "Norway"),
    ("FI", "Finland"),
    ("DK", "Denmark"),
    ("IS", "Iceland"),
    ("PL", "Poland"),
    ("CZ", "Czechia"),
    ("SK", "Slovakia"),
    ("HU", "Hungary"),
    ("GR", "Greece"),
    ("RO", "Romania"),
    ("BG", "Bulgaria"),
    ("EE", "Estonia"),
    ("LV", "Latvia"),
    ("LT", "Lithuania"),
    ("AU", "Australia"),
    ("NZ", "New Zealand"),
    ("SG", "Singapore"),
    ("HK", "Hong Kong"),
    ("TW", "Taiwan"),
    ("KR", "South Korea"),
    ("CN", "China"),
    ("IN", "India"),
    ("TH", "Thailand"),
    ("MY", "Malaysia"),
    ("ID", "Indonesia"),
    ("PH", "Philippines"),
    ("VN", "Vietnam"),
    ("AE", "United Arab Emirates"),
    ("SA", "Saudi Arabia"),
    ("IL", "Israel"),
    ("TR", "Turkey"),
    ("ZA", "South Africa"),
    ("BR", "Brazil"),
    ("MX", "Mexico"),
    ("AR", "Argentina"),
    ("CL", "Chile"),
    ("CO", "Colombia"),
)


def _is_valid_regdomain(code: str) -> bool:
    """Return True if ``code`` is in the curated supported-country list."""
    return code.upper() in {c for c, _ in _WIFI_REGDOMAIN_OPTIONS}


# ── Timezone ──────────────────────────────────────────────────────────

_DEFAULT_TIMEZONE = ""  # empty = use the server's local time


def _effective_timezone() -> str:
    """Return the display timezone to use (IANA name) or '' for local.

    Priority mirrors ``_effective_regdomain``:
      1. ``config.timezone``
      2. persisted override file
      3. default (empty string — server local time).
    """
    try:
        from_config = getattr(config, "timezone", None)
    except NameError:
        from_config = None
    if isinstance(from_config, str) and from_config.strip():
        return from_config.strip()

    override = _load_settings_override().get("timezone")
    if isinstance(override, str) and override.strip():
        return override.strip()
    return _DEFAULT_TIMEZONE


def _is_dev_mode() -> bool:
    """Return True if dev mode is enabled (SSID rotation skipped)."""
    override = _load_settings_override().get("dev_mode")
    if isinstance(override, bool):
        return override
    return os.environ.get("SCRIBE_DEV_MODE", "0") == "1"


def _seed_tts_from_enrollments() -> None:
    """Seed the TTS voice cache with each enrolled speaker's reference WAV.

    Runs at meeting start, after ``reset_voice_cache`` has cleared any
    cross-meeting leakage. Missing or unreadable WAV files are skipped
    with a log line so a single bad file doesn't break the whole seed.
    """
    import wave

    if tts_backend is None or not hasattr(tts_backend, "seed_voice"):
        return
    for eid, speaker in enrollment_store.speakers.items():
        ref_path = getattr(speaker, "reference_wav_path", "")
        if not ref_path:
            continue
        path = Path(ref_path)
        if not path.exists():
            logger.info("TTS seed: enrollment '%s' ref wav missing at %s", eid, ref_path)
            continue
        try:
            with wave.open(str(path), "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                pcm = np.frombuffer(frames, dtype=np.int16)
                audio = pcm.astype(np.float32) / 32767.0
            tts_backend.seed_voice(eid, audio, source="enrollment")
        except Exception as e:
            logger.warning("TTS seed failed for '%s': %s", eid, e)


def _effective_tts_voice_mode() -> str:
    """Return the server-wide default TTS voice mode.

    "studio" — Qwen3-TTS named speaker per language (fast, commercial-safe).
    "cloned" — clone each meeting participant's voice (slower, personal).
    Individual listeners can still override per-session via WS ``set_voice``.
    """
    override = _load_settings_override().get("tts_voice_mode")
    if isinstance(override, str) and override in ("studio", "cloned"):
        return override
    return "studio"


def _timezone_options() -> list[str]:
    """Return all IANA timezone names the runtime knows about.

    Filters out bare legacy aliases (``UTC``, ``GMT``, ``EST`` …) that
    don't follow the ``Region/City`` format — those are confusing in a
    dropdown next to modern names. Sorted alphabetically.
    """
    from zoneinfo import available_timezones

    names = available_timezones()
    # Keep only names with a '/' — drops legacy shortcuts like UTC, GMT,
    # EST, MST, Zulu, etc. UTC is re-added at the top for visibility.
    tz_list = sorted(n for n in names if "/" in n)
    return ["UTC", *tz_list]


def _is_valid_timezone(name: str) -> bool:
    """Return True if ``name`` is a valid IANA timezone name."""
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        ZoneInfo(name)
        return True
    except (ZoneInfoNotFoundError, ValueError):
        return False


def _current_regdomain() -> str | None:
    """Return the current 2-letter country code from ``iw reg get``.

    Returns the country code (``JP``, ``US``, ``00``, …) or ``None`` if
    ``iw`` isn't available or the output can't be parsed.
    """
    try:
        result = subprocess.run(
            ["iw", "reg", "get"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("country "):
            # Format: "country JP: DFS-JP" — extract the 2-char code.
            token = stripped.split(":", 1)[0].removeprefix("country ").strip()
            return token or None
    return None


def _ensure_regdomain() -> bool:
    """Ensure the WiFi regulatory domain matches ``_effective_regdomain()``.

    Runs ``sudo iw reg set <code>`` and verifies the result via ``iw reg get``.
    Returns True on success, False otherwise. Never raises — failures are
    logged and the caller decides whether to abort.
    """
    import time

    target = _effective_regdomain()
    current = _current_regdomain()
    if current == target:
        return True

    try:
        subprocess.run(
            ["sudo", "iw", "reg", "set", target],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.error(
            "failed to invoke 'iw reg set %s': %s (current=%r)",
            target,
            exc,
            current,
        )
        return False

    # Verify the setting actually took effect. The kernel may reject
    # the change (e.g. on some MT7925 firmware versions) or silently
    # revert due to a concurrent scan result.
    time.sleep(0.2)
    after = _current_regdomain()
    if after == target:
        if current != after:
            logger.info("regdomain set to %s (was %r)", target, current)
        return True

    logger.error(
        "regdomain set to %s failed: iw reg get still reports %r — "
        "5 GHz AP transmit power will be capped; clients cannot connect",
        target,
        after,
    )
    return False


def _ensure_regdomain_persistent() -> bool:
    """Install ``/etc/modprobe.d/cfg80211-<code>.conf`` so cfg80211 boots
    in the configured country. Idempotent — rewrites only if the target
    code has changed. Returns True on success, False on failure (e.g.
    no sudo, write error). Never raises.
    """
    target = _effective_regdomain()
    path = _regdomain_modprobe_path(target)
    body = f"options cfg80211 ieee80211_regdom={target}\n"

    # If an existing file for a DIFFERENT country is present, remove it so
    # we don't end up with two conflicting modprobe entries.
    try:
        for existing in Path("/etc/modprobe.d").glob("cfg80211-*.conf"):
            if existing == path:
                continue
            try:
                subprocess.run(
                    ["sudo", "rm", "-f", str(existing)],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                pass
    except OSError:
        pass

    # Skip the write if the content is already correct.
    try:
        if path.exists() and path.read_text() == body:
            return True
    except OSError:
        pass

    try:
        result = subprocess.run(
            ["sudo", "tee", str(path)],
            input=body,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("could not write %s: %s", path, exc)
        return False

    if result.returncode != 0:
        logger.warning(
            "sudo tee %s failed (rc=%d): %s",
            path,
            result.returncode,
            result.stderr.strip(),
        )
        return False
    return True


# Activation polling: `nmcli con up` can exit before wpa_supplicant finishes
# negotiating, and NM may auto-retry a failed activation on its own. We poll
# for up to 45s so the supplicant-timeout path (~27s on the MT7925) still
# succeeds if NM recovers it.
_AP_ACTIVATION_WAIT_SECONDS = 45
_AP_ACTIVATION_POLL_INTERVAL = 1.0

# nmcli con up timeout needs to cover the supplicant negotiation window.
# Observed: 27s supplicant-timeout + NM retry on the MT7925 driver.
_NMCLI_CON_UP_TIMEOUT = 60
_NMCLI_CON_DOWN_TIMEOUT = 15
_NMCLI_CON_MODIFY_TIMEOUT = 5
_NMCLI_CON_SHOW_TIMEOUT = 5


def _run_nmcli_sync(args: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    """Run ``sudo nmcli <args>`` synchronously. Errors are returned, not raised."""
    import subprocess as _subprocess

    return _subprocess.run(
        ["sudo", "nmcli", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _parse_nmcli_fields(output: str) -> dict[str, str]:
    """Parse ``nmcli -t -f a,b,c con show NAME`` terse output into a dict.

    The -t (terse) + -f (fields) combo emits one ``key:value`` per line. A
    field may contain additional ``:`` characters (common for SSIDs with
    punctuation), so we only split on the FIRST colon.
    """
    result: dict[str, str] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        result[key.strip()] = value
    return result


def _nmcli_read_live_ap_credentials() -> tuple[str, str] | None:
    """Return ``(ssid, psk)`` for the active AP profile, or ``None``.

    Reads via ``nmcli --show-secrets`` so the psk is returned in plaintext.
    This is the authoritative source for what the radio is actually
    broadcasting — the state file is a derived cache of this.
    """
    proc = _run_nmcli_sync(
        [
            "--show-secrets",
            "-t",
            "-f",
            "802-11-wireless.ssid,802-11-wireless-security.psk",
            "con",
            "show",
            AP_CON_NAME,
        ],
        timeout=_NMCLI_CON_SHOW_TIMEOUT,
    )
    if proc.returncode != 0:
        return None
    fields = _parse_nmcli_fields(proc.stdout)
    ssid = fields.get("802-11-wireless.ssid", "")
    psk = fields.get("802-11-wireless-security.psk", "")
    if not ssid:
        return None
    return ssid, psk


def _nmcli_ap_is_active() -> bool:
    """Return True if the AP profile is currently active on the radio."""
    proc = _run_nmcli_sync(
        ["-t", "-f", "NAME,DEVICE", "con", "show", "--active"],
        timeout=_NMCLI_CON_SHOW_TIMEOUT,
    )
    if proc.returncode != 0:
        return False
    prefix = f"{AP_CON_NAME}:"
    return any(line.startswith(prefix) for line in proc.stdout.splitlines())


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write ``data`` to ``path`` atomically (tmp file + rename)."""
    import json as _json

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(_json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def _write_hotspot_state_sync() -> bool:
    """Sync ``HOTSPOT_STATE_FILE`` with the live AP credentials (sync path).

    Returns True if the state file now matches a currently-broadcasting AP,
    False otherwise. The state file is ONLY written when the AP is both
    readable (``_nmcli_read_live_ap_credentials``) AND actively broadcasting
    (``_nmcli_ap_is_active``). If the profile has credentials but the radio
    is disconnected, the state file is left alone — otherwise the
    ``/api/meeting/wifi`` endpoint would ship a QR for a non-broadcasting
    network.

    ``port`` is hard-coded to 80 to match the guest HTTP listener. The QR
    URL omits the port entirely (``http://10.42.0.1/``); this field is
    retained for backwards compatibility with existing readers.

    Never raises — failures are logged at debug level.
    """
    try:
        creds = _nmcli_read_live_ap_credentials()
        if creds is None:
            logger.debug("hotspot state sync: AP profile not readable")
            return False
        # Critical: don't write state for a profile whose radio is down.
        # Otherwise get_wifi_info would return credentials for a network
        # that isn't broadcasting and guests would fail to connect.
        if not _nmcli_ap_is_active():
            logger.debug("hotspot state sync: AP profile readable but not active")
            return False
        ssid, psk = creds
        state = {
            "ssid": ssid,
            "password": psk,
            "ap_ip": AP_IP,
            "port": 80,  # guest HTTP listener
        }
        _atomic_write_json(HOTSPOT_STATE_FILE, state)
        return True
    except Exception as exc:
        logger.debug("hotspot state sync failed: %s", exc)
        return False


# ── Rotation deduplication ──────────────────────────────────────────
#
# `_start_wifi_ap` is scheduled as a fire-and-forget task from BOTH
# `start_meeting` and `resume_meeting`. Without a dedup, each call
# rotates credentials afresh, which caused the production bug where the
# admin panel and the pop-out view displayed different SSIDs (each fetched
# `/api/meeting/wifi` between two rotations and cached a different answer).
#
# Guarantees:
#   - At most one rotation is in flight at any time (`_AP_ROTATION_LOCK`).
#   - At most one rotation per meeting_id (`_LAST_ROTATED_MEETING_ID`).
#   - A no-op call still reconciles the state file and starts the captive
#     portal + firewall so transient drift (e.g. state file clobbered by
#     an external process) still gets healed.
_AP_ROTATION_LOCK = asyncio.Lock()
_LAST_ROTATED_MEETING_ID: str | None = None


def _reset_rotation_state_for_tests() -> None:
    """Clear the rotation dedup state. Unit-test only."""
    global _LAST_ROTATED_MEETING_ID
    _LAST_ROTATED_MEETING_ID = None


async def _start_wifi_ap(meeting_id: str | None = None) -> None:
    """Rotate AP credentials, wait for NM to activate them, and sync state.

    Idempotent per ``meeting_id``: if called twice for the same meeting,
    the second call does NOT rotate credentials — it only re-syncs the
    state file and ensures the captive portal + firewall are applied. This
    prevents the consistency bug where admin view and pop-out view fetch
    ``/api/meeting/wifi`` between two rotations and display different SSIDs.

    The implementation is structured so that the state file in
    ``/tmp/meeting-hotspot.json`` can *never* drift from the live AP:

    1. Acquire rotation lock (serializes concurrent calls).
    2. If already rotated for this ``meeting_id`` AND the AP is still active,
       skip rotation but still reconcile state file + portal (no-op fast path).
    3. Otherwise: generate fresh credentials and push them into the NM profile.
    4. Bounce the connection (down → up). NOTE: ``nmcli con up`` can return
       non-zero (supplicant-timeout) even when NM subsequently auto-retries
       and succeeds — we do NOT treat the exit code as authoritative.
    5. **Poll** ``nmcli con show --active`` for up to 45s, catching the
       auto-retry window.
    6. Once the AP is active, read the live SSID/psk back from nmcli
       ``--show-secrets`` and write them to the state file. The state file
       is ALWAYS derived from the radio, never from in-memory values.
    7. Start the captive portal and apply the hotspot firewall. Idempotent.
    8. Record the ``meeting_id`` so the next call for the same meeting is
       a no-op.

    Fire-and-forget: this is scheduled via ``asyncio.create_task`` from
    ``start_meeting`` / ``resume_meeting``. All errors are logged; no
    exception escapes to the meeting-start flow.
    """
    import secrets

    global _LAST_ROTATED_MEETING_ID

    loop = asyncio.get_event_loop()

    async with _AP_ROTATION_LOCK:
        # Dev mode: skip SSID rotation entirely — keep the current AP
        # credentials so the hotspot network stays consistent across
        # meeting starts. Still reconcile state + portal + firewall.
        # If the AP isn't active yet, bring it up with existing credentials.
        if _is_dev_mode():
            logger.info(
                "Dev mode: skipping SSID rotation for meeting %s",
                meeting_id,
            )
            if not await loop.run_in_executor(None, _nmcli_ap_is_active):
                logger.info("Dev mode: AP not active, bringing up with existing credentials")
                await loop.run_in_executor(None, _ensure_regdomain)
                await loop.run_in_executor(
                    None,
                    lambda: _run_nmcli_sync(["con", "up", AP_CON_NAME], timeout=45),
                )
                deadline = asyncio.get_event_loop().time() + _AP_ACTIVATION_WAIT_SECONDS
                while asyncio.get_event_loop().time() < deadline:
                    if await loop.run_in_executor(None, _nmcli_ap_is_active):
                        break
                    await asyncio.sleep(_AP_ACTIVATION_POLL_INTERVAL)
            await loop.run_in_executor(None, _write_hotspot_state_sync)
            await _start_captive_portal()
            await _apply_hotspot_firewall()
            if meeting_id is not None:
                _LAST_ROTATED_MEETING_ID = meeting_id
            return

        # Step 0: make sure the WiFi regulatory domain is JP before we touch
        # the AP. If it drifted back to the default "world" domain (country 00),
        # the kernel would cap our 5 GHz TX power to a level where phones
        # can't associate. Done on EVERY rotation, not just once at boot,
        # because scans can silently reset the regdomain.
        if not await loop.run_in_executor(None, _ensure_regdomain):
            logger.error(
                "refusing to rotate AP: regulatory domain is not %s — "
                "a phone attempting to connect would fail at association",
                _effective_regdomain(),
            )
            return

        # Dedup fast path: same meeting already rotated and AP still active.
        if (
            meeting_id is not None
            and meeting_id == _LAST_ROTATED_MEETING_ID
            and await loop.run_in_executor(None, _nmcli_ap_is_active)
        ):
            logger.info(
                "WiFi AP rotation skipped (already rotated for meeting %s)",
                meeting_id,
            )
            # Still reconcile state + portal in case either drifted.
            await loop.run_in_executor(None, _write_hotspot_state_sync)
            await _start_captive_portal()
            await _apply_hotspot_firewall()
            return

        session_id = secrets.token_hex(2).upper()
        new_ssid = f"Dell Demo {session_id}"
        new_password = secrets.token_hex(4).upper()

    def _rotate_profile_and_bounce() -> subprocess.CompletedProcess[str] | None:
        """Update the NM profile and bounce it. All sync to avoid race."""
        import time as _time

        modify = _run_nmcli_sync(
            [
                "con",
                "modify",
                AP_CON_NAME,
                "802-11-wireless.ssid",
                new_ssid,
                "802-11-wireless-security.psk",
                new_password,
            ],
            timeout=_NMCLI_CON_MODIFY_TIMEOUT,
        )
        if modify.returncode != 0:
            return modify

        _run_nmcli_sync(["con", "down", AP_CON_NAME], timeout=_NMCLI_CON_DOWN_TIMEOUT)
        _time.sleep(2)  # wifi driver needs a beat between down and up
        return _run_nmcli_sync(["con", "up", AP_CON_NAME], timeout=_NMCLI_CON_UP_TIMEOUT)

    # Step 1: rotate. Don't fail the whole flow if this raises — we still
    # want to reconcile in case the AP is running with stale credentials.
    try:
        rotate_result = await loop.run_in_executor(None, _rotate_profile_and_bounce)
        if rotate_result is None or rotate_result.returncode != 0:
            stderr = (rotate_result.stderr or "").strip()[:200] if rotate_result else ""
            logger.warning(
                "WiFi AP rotation nmcli failed (will still reconcile): ssid=%s err=%s",
                new_ssid,
                stderr or "<no stderr>",
            )
        else:
            logger.info("WiFi AP rotation submitted: ssid=%s", new_ssid)
    except Exception as exc:
        logger.warning("WiFi AP rotation raised (will still reconcile): %s", exc)

    # Step 2: poll for the AP to become active. NM may have auto-retried
    # even if our explicit `con up` timed out.
    deadline = asyncio.get_event_loop().time() + _AP_ACTIVATION_WAIT_SECONDS
    active = False
    while asyncio.get_event_loop().time() < deadline:
        if await loop.run_in_executor(None, _nmcli_ap_is_active):
            active = True
            break
        await asyncio.sleep(_AP_ACTIVATION_POLL_INTERVAL)

    if not active:
        logger.error(
            "WiFi AP did not become active within %ds — hotspot state file left unchanged",
            _AP_ACTIVATION_WAIT_SECONDS,
        )
        return

    # Step 3: reconcile state file from live nmcli. This is the authoritative
    # write — what's in the file now matches what the radio is broadcasting.
    if await loop.run_in_executor(None, _write_hotspot_state_sync):
        creds = await loop.run_in_executor(None, _nmcli_read_live_ap_credentials)
        live_ssid = creds[0] if creds else "<unknown>"
        logger.info("Hotspot state written from live nmcli: ssid=%s", live_ssid)
        if creds and creds[0] != new_ssid:
            logger.warning(
                "Live AP ssid %r does not match rotation target %r — "
                "rotation likely failed and NM served the previous profile",
                creds[0],
                new_ssid,
            )
    else:
        logger.error("Failed to sync hotspot state file after AP activation")

    # Step 4: captive portal + firewall. These are idempotent; calling them
    # after a successful reconciliation ensures clients can join whatever is
    # actually broadcasting.
    await _start_captive_portal()
    await _apply_hotspot_firewall()

    # Step 5: record the meeting_id so subsequent calls for the same meeting
    # are no-ops. Only set AFTER successful reconciliation — if rotation or
    # reconciliation failed above, we want a retry path to still attempt
    # rotation on the next call.
    if meeting_id is not None:
        _LAST_ROTATED_MEETING_ID = meeting_id


async def _write_hotspot_state() -> None:
    """Sync the hotspot state file from live nmcli.

    This is kept as an async wrapper around ``_write_hotspot_state_sync``
    for existing callers. New code should prefer the sync helper directly
    or call this from an async context.
    """
    loop = asyncio.get_event_loop()
    if await loop.run_in_executor(None, _write_hotspot_state_sync):
        logger.info("Hotspot state synced from live nmcli")
    else:
        logger.debug("Hotspot state sync: no change or failure")


SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
CAPTIVE_PID_80 = Path("/tmp/meeting-captive-80.pid")
# Stale PID file from the deleted port-443 TLS redirector. Kept as a
# constant only so _stop_captive_portal() can reap it on upgrade; nothing
# writes to it anymore. Safe to delete once all live hosts have cycled.
CAPTIVE_PID_443 = Path("/tmp/meeting-captive-443.pid")
# sddc-cli's captive portal PID file (port 80 only). We clean it up on
# stop so sddc-cli's redirector can't shadow ours when both writers are
# invoked in the same session.
SDDC_CLI_PORTAL_PID = Path("/tmp/meeting-captive-portal.pid")


async def _start_captive_portal() -> None:
    """Captive-portal lifecycle hook.

    The guest portal is **HTTP-only on port 80**. The in-process guest
    uvicorn binds ``{127.0.0.1,10.42.0.1}:80`` and serves every captive-
    portal probe route (``/hotspot-detect.html``, ``/generate_204``,
    ``/api/captive``, etc.) directly via the FastAPI routes registered
    above.

    There is **no TLS handler on port 443** and nothing is spawned
    here. HTTPS captive-portal MITM is dead since HSTS preload:
    apple.com, google.com, github.com, etc. cannot be intercepted with
    a self-signed cert because browsers refuse the click-through for
    HSTS-preloaded domains. Modern OS captive-portal detection uses
    HTTP probes (captive.apple.com/hotspot-detect.html,
    connectivitycheck.gstatic.com/generate_204, etc.) which hit port 80
    via the NAT REDIRECT rule in ``_apply_hotspot_firewall`` and work
    regardless. The firewall sends TCP RST on 443 so HTTPS attempts
    fail instantly and the OS falls back to its HTTP captive-portal
    detection.

    This function is kept as an explicit lifecycle seam (called by the
    hotspot bring-up flow) so future work can slot in — but today it
    just guarantees no stale 443 subprocess / PID files survive.
    """
    await _stop_captive_portal()


async def _stop_captive_portal() -> None:
    """Stop captive portal handlers.

    Cleans up BOTH meeting-scribe's own PID files (80 + 443) AND
    sddc-cli's PID file, because both writers bind port 80 and leaving
    the other writer's listener around would prevent us from rebinding.
    """
    import signal as _signal

    for pid_file in (CAPTIVE_PID_80, CAPTIVE_PID_443, SDDC_CLI_PORTAL_PID):
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                try:
                    os.kill(pid, _signal.SIGTERM)
                except ProcessLookupError:
                    pass
            except Exception:
                pass
            pid_file.unlink(missing_ok=True)

    # Kill any orphaned captive portal processes (match script names)
    await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: subprocess.run(
            ["pkill", "-f", "captive-portal-[48]"],
            capture_output=True,
            timeout=5,
            check=False,
        ),
    )
    # Also catch sddc-cli's script name (/tmp/meeting-captive-portal.py)
    await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: subprocess.run(
            ["pkill", "-f", "meeting-captive-portal.py"],
            capture_output=True,
            timeout=5,
            check=False,
        ),
    )


HOTSPOT_SUBNET_CIDR = os.environ.get("SCRIBE_HOTSPOT_SUBNET_CIDR", "10.42.0.0/24")
HOTSPOT_IPTABLES_COMMENT = "meeting-scribe-hotspot"


async def _apply_hotspot_firewall() -> None:
    """Hotspot firewall: always clear tagged rules and re-apply.

    Every call first removes any rules carrying our comment tag, then
    reinstalls the canonical rule set. This guarantees code changes to
    the rule set (ports, REJECT vs REDIRECT, etc.) actually take effect
    on the next meeting-scribe restart — the previous "skip if already
    present" guard silently locked in stale rules from old releases.
    The delete + add cycle is ~10 iptables invocations, a few hundred ms.
    """
    import subprocess

    await _remove_hotspot_firewall()

    rules = [
        # Allow: HTTP (guest portal on 80) only. Port 443 is REJECTed with
        # TCP RST so HTTPS attempts fail instantly — HSTS-preloaded domains
        # (apple.com, google.com, ...) cannot be intercepted with a self-
        # signed cert, and a TCP RST causes modern OS captive-portal
        # detection to fall back to its HTTP probes, which all land on
        # port 80 and hit the guest portal. Admin HTTPS (config.port /
        # 8080) is also REJECTed from the hotspot subnet — admin must be
        # reached from the LAN. Defense-in-depth behind the hotspot_guard
        # middleware.
        f"-I INPUT 1 -s {HOTSPOT_SUBNET_CIDR} -p tcp --dport 80 -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j ACCEPT",
        f"-I INPUT 2 -s {HOTSPOT_SUBNET_CIDR} -p tcp --dport 443 -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j REJECT --reject-with tcp-reset",
        f"-I INPUT 3 -s {HOTSPOT_SUBNET_CIDR} -p tcp --dport {config.port} -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j REJECT --reject-with tcp-reset",
        f"-I INPUT 4 -s {HOTSPOT_SUBNET_CIDR} -p udp --dport 53 -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j ACCEPT",
        f"-I INPUT 5 -s {HOTSPOT_SUBNET_CIDR} -p udp --dport 67 -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j ACCEPT",
        f"-I INPUT 6 -s {HOTSPOT_SUBNET_CIDR} -p icmp -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j ACCEPT",
        # REJECT (not DROP!) — gives instant ICMP unreachable instead of 30s timeout
        f"-I INPUT 7 -s {HOTSPOT_SUBNET_CIDR} -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j REJECT --reject-with icmp-port-unreachable",
        # CRITICAL: no internet — block ALL forwarding for hotspot subnet (both directions)
        f"-I FORWARD 1 -s {HOTSPOT_SUBNET_CIDR} -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j REJECT --reject-with icmp-port-unreachable",
        f"-I FORWARD 2 -d {HOTSPOT_SUBNET_CIDR} -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j REJECT --reject-with icmp-port-unreachable",
        # NAT redirect: only port 80. Anything else clients try (1.2.3.4:80,
        # random.com:80 after DNS wildcard, OS captive-portal probes) all
        # land on the in-process guest listener on 10.42.0.1:80.
        f"-t nat -I PREROUTING 1 -s {HOTSPOT_SUBNET_CIDR} -p tcp --dport 80 -m comment --comment {HOTSPOT_IPTABLES_COMMENT} -j REDIRECT --to-port 80",
    ]
    for rule in rules:
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda r=rule: subprocess.run(  # type: ignore[misc]
                ["sudo", "iptables"] + r.split(),
                capture_output=True,
                timeout=5,
            ),
        )
    logger.info(
        "Hotspot firewall applied (allow: 80/DNS/DHCP; reject: 443/admin %d)",
        config.port,
    )


async def _remove_hotspot_firewall() -> None:
    """Remove hotspot firewall rules."""
    import subprocess

    # Remove all rules with our comment
    for chain in ("INPUT", "FORWARD"):
        for _ in range(10):  # max 10 rules to remove
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda c=chain: subprocess.run(  # type: ignore[misc]
                    ["sudo", "iptables", "-S", c],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ),
            )
            found = False
            for line in (result.stdout or "").splitlines():
                if HOTSPOT_IPTABLES_COMMENT in line:
                    # Convert -A to -D for deletion
                    del_rule = line.replace(f"-A {chain}", f"-D {chain}", 1)
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda r=del_rule: subprocess.run(  # type: ignore[misc]
                            ["sudo", "iptables"] + r.split(),
                            capture_output=True,
                            timeout=5,
                        ),
                    )
                    found = True
                    break
            if not found:
                break
    # Also clean NAT PREROUTING rules
    for _ in range(10):
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                ["sudo", "iptables", "-t", "nat", "-S", "PREROUTING"],
                capture_output=True,
                text=True,
                timeout=5,
            ),
        )
        found = False
        for line in (result.stdout or "").splitlines():
            if HOTSPOT_IPTABLES_COMMENT in line:
                del_rule = line.replace("-A PREROUTING", "-D PREROUTING", 1)
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda r=del_rule: subprocess.run(  # type: ignore[misc]
                        ["sudo", "iptables", "-t", "nat"] + r.split(),
                        capture_output=True,
                        timeout=5,
                    ),
                )
                found = True
                break
        if not found:
            break
    logger.info("Hotspot firewall removed")


async def _stop_wifi_ap() -> None:
    """Bring down the WiFi AP when a meeting ends.

    Stops the AP and captive portal. Firewall rules are intentionally
    left in place — they are harmless when no hotspot traffic exists
    and will be reused by the next meeting.
    """
    import subprocess

    try:
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                ["sudo", "nmcli", "con", "down", AP_CON_NAME],
                capture_output=True,
                text=True,
                timeout=15,
            ),
        )
        Path("/tmp/meeting-hotspot.json").unlink(missing_ok=True)
        await _stop_captive_portal()
        # Firewall rules are persistent — never removed
        logger.info("WiFi AP + captive portal stopped (firewall rules retained)")
    except Exception as e:
        logger.debug("WiFi AP stop skipped: %s", e)


@app.get("/api/meeting/wifi")
async def get_wifi_info(request: fastapi.Request) -> JSONResponse:
    """Return WiFi hotspot info + QR code SVG for client auto-join.

    Pure read from the hotspot state file. No nmcli, no sudo, no dbus.

    The state file is the single source of truth — it's written atomically
    only from ``_start_wifi_ap()`` on meeting-start / rotation and from
    ``_start_captive_portal()`` on hotspot-up, both inside this process.
    Nothing else mutates it. The old version of this handler called
    ``_write_hotspot_state_sync()`` (one `sudo nmcli --show-secrets` + one
    `sudo nmcli con show --active`) on every GET "just in case the state
    file drifted" — observed 2026-04-15, this burned ~1.5 s per request on
    this box because sudo+PAM+dbus is slow, and the guest page polls this
    endpoint every ~10 s. 26 polls × 1.5 s blew past the asyncio executor
    thread pool and showed up as recurring 2.5 s event-loop stalls that
    killed the audio-out WS ping. There was no real drift scenario — just
    defensive code from a bug fear that doesn't actually exist in the
    single-writer world.
    """

    state = _load_hotspot_state()
    if not state:
        return JSONResponse({"error": "Hotspot not active"}, status_code=503)

    ssid = state["ssid"]
    password = state["password"]
    ap_ip = state["ap_ip"]

    wifi_qr_svg = _wifi_qr_svg(ssid, password)
    # Guest portal is served over plain HTTP on port 80 bound to the hotspot
    # gateway IP. Building the QR URL as http:// (no port) means:
    #   - No self-signed cert warnings when guests scan and join
    #   - The captive portal mini-browser on iOS actually follows the URL
    #   - All guest traffic flows through the HTTP listener, which the
    #     scheme-based guest_scope check already locks down.
    # Admin stays on https://<lan>:8080/; nothing guest-facing references it.
    meeting_url = f"http://{ap_ip}/"
    url_qr_svg = _qr_svg(meeting_url)

    # Silence the unused-variable warning now that we no longer need the
    # caller's URL scheme (the meeting_url is hard-coded to http).
    del request

    return JSONResponse(
        {
            "ssid": ssid,
            "password": password,
            "ap_ip": ap_ip,
            "meeting_url": meeting_url,
            "session_id": state.get("session_id", ""),
            "wifi_qr_svg": wifi_qr_svg,
            "url_qr_svg": url_qr_svg,
        }
    )


# ── Client-side audio diagnostics ─────────────────────────────────────
# Live telemetry from each Listen client, pushed by the browser every ~2s.
# We can then query GET /api/diag/listeners from the trace tool to see what
# is actually happening on the user's device — context state, queue depth,
# decode counts, errors — without asking them to read a banner off a phone.
_listener_diag: dict[str, dict] = {}  # client_id → state dict (in-memory only)


@app.post("/api/diag/listener")
async def diag_listener_post(request: fastapi.Request) -> JSONResponse:
    """Accept a tiny JSON ping from a browser-side audio listener.

    Body shape (all optional, free-form):
      {
        "client_id":   "<stable per-tab uuid>",
        "page":        "admin" | "guest" | "reader",
        "ctx_state":   "running" | "suspended" | "closed" | "null",
        "ctx_rate":    48000,
        "ws_state":    "OPEN" | "CONNECTING" | "CLOSING" | "CLOSED" | "NULL",
        "primed":      true,
        "queue":       0,
        "bytes_in":    12345,
        "blobs_in":    7,
        "decoded":     7,
        "decode_err":  0,
        "played":      6,
        "last_err":    "",
        "ua_short":    "Safari/iPhone",
      }

    Stored in a tiny in-memory ring (last 16 distinct client_ids). The trace
    tool reads /api/diag/listeners to surface the latest snapshot per client.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "expected object"}, status_code=400)
    client_id = str(body.get("client_id") or "anon")[:64]
    body["_received_at"] = time.time()
    try:
        body["_peer"] = f"{request.client.host}" if request.client else "?"
    except Exception:
        body["_peer"] = "?"
    _listener_diag[client_id] = body
    # Bound the dict so a noisy client can't OOM the server.
    if len(_listener_diag) > 16:
        oldest = sorted(_listener_diag.items(), key=lambda kv: kv[1].get("_received_at", 0))
        for k, _ in oldest[:-16]:
            _listener_diag.pop(k, None)
    return JSONResponse({"ok": True})


@app.get("/api/diag/listeners")
async def diag_listeners_get() -> JSONResponse:
    """Return the latest audio-listener telemetry from every recent client."""
    now = time.time()
    out = []
    for cid, state in _listener_diag.items():
        out.append(
            {**state, "client_id": cid, "age_s": round(now - state.get("_received_at", now), 1)}
        )
    out.sort(key=lambda r: r.get("age_s", 0))
    return JSONResponse({"listeners": out})


# ── Captive Portal ───────────────────────────────────────────
# Two-phase captive-portal behavior keyed on client IP:
#
#   Phase 1 — unacknowledged:
#     Every OS probe returns a 302 redirect (or "captive: true" for
#     RFC 8910) so the device's captive-portal assistant opens
#     automatically on WiFi association and shows the guest portal.
#
#   Phase 2 — acknowledged:
#     Once the client has actually loaded the portal page at ``/``,
#     their IP is added to ``_captive_acked``. Subsequent probes
#     return the platform-specific "not captive, you're online"
#     response so the CNA sheet dismisses and the blue tick appears.
#
# Without phase 2 iOS stays stuck in CNA forever — the OS keeps
# polling hotspot-detect.html and never sees the ``Success`` body
# it wants, so it never marks the network as "internet ready".
#
# The set is IP-keyed, not cookie-keyed, because iOS CNA is a
# separate WebKit context from Safari and does not share cookies.
# State is process-local (cleared on meeting-scribe restart) — fine
# for a demo AP where the SSID rotates per meeting anyway.

_PORTAL_URL = "http://10.42.0.1/"
_captive_acked: set[str] = set()


def _captive_ack(request: fastapi.Request) -> None:
    """Mark the requesting hotspot client IP as having seen the portal."""
    client = getattr(request, "client", None)
    ip = client.host if client else ""
    if ip and ip.startswith(HOTSPOT_SUBNET):
        _captive_acked.add(ip)


def _is_captive_acked(request: fastapi.Request) -> bool:
    client = getattr(request, "client", None)
    ip = client.host if client else ""
    return bool(ip) and ip in _captive_acked


@app.get("/hotspot-detect.html")
async def captive_apple(request: fastapi.Request) -> fastapi.responses.Response:
    """Apple iOS/macOS probe.

    Unacknowledged → 302 to portal → CNA opens.
    Acknowledged   → exact ``Success`` HTML → CNA dismisses, blue tick.
    """
    if _is_captive_acked(request):
        return fastapi.responses.HTMLResponse(
            "<HTML><HEAD><TITLE>Success</TITLE></HEAD><BODY>Success</BODY></HTML>"
        )
    return fastapi.responses.RedirectResponse(_PORTAL_URL, status_code=302)


@app.get("/generate_204")
@app.get("/gen_204")
@app.get("/canonical.html")
@app.get("/redirect")
async def captive_204(request: fastapi.Request) -> fastapi.responses.Response:
    """Android/ChromeOS/Firefox probes.

    Unacknowledged → 302 to portal → captive-portal sign-in UI.
    Acknowledged   → HTTP 204 No Content → network marked online.
    """
    if _is_captive_acked(request):
        return fastapi.responses.Response(status_code=204)
    return fastapi.responses.RedirectResponse(_PORTAL_URL, status_code=302)


@app.get("/connecttest.txt")
async def captive_windows(request: fastapi.Request) -> fastapi.responses.Response:
    """Windows NCSI probe."""
    if _is_captive_acked(request):
        return fastapi.responses.PlainTextResponse("Microsoft Connect Test")
    return fastapi.responses.RedirectResponse(_PORTAL_URL, status_code=302)


@app.get("/ncsi.txt")
async def captive_ncsi(request: fastapi.Request) -> fastapi.responses.Response:
    """Windows NCSI secondary probe."""
    if _is_captive_acked(request):
        return fastapi.responses.PlainTextResponse("Microsoft NCSI")
    return fastapi.responses.RedirectResponse(_PORTAL_URL, status_code=302)


@app.get("/success.txt")
async def captive_firefox(request: fastapi.Request) -> fastapi.responses.Response:
    """Firefox captive-portal probe."""
    if _is_captive_acked(request):
        return fastapi.responses.PlainTextResponse("success\n")
    return fastapi.responses.RedirectResponse(_PORTAL_URL, status_code=302)


@app.get("/api/captive")
async def captive_rfc8910(request: fastapi.Request) -> JSONResponse:
    """RFC 8910 captive-portal API.

    Modern OSes (iOS 14+, Android 11+) poll this after learning the
    URL from DHCP option 114. Flips from ``captive: true`` to
    ``captive: false`` once the client has loaded the portal page.
    """
    if _is_captive_acked(request):
        return JSONResponse(
            {"captive": False},
            headers={"Content-Type": "application/captive+json"},
        )
    return JSONResponse(
        {
            "captive": True,
            "user-portal-url": _PORTAL_URL,
        },
        headers={"Content-Type": "application/captive+json"},
    )


@app.get("/api/languages")
async def get_languages() -> JSONResponse:
    """Return the selectable language list + default pair for the setup UI."""
    from meeting_scribe.languages import to_api_response

    return JSONResponse(to_api_response())


async def _probe_vllm_status(url: str) -> dict:
    """Probe a vLLM endpoint for detailed loading status.

    Returns: {"ready": bool, "status": str, "detail": str|None}
    """
    import subprocess

    base = url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{base}/health")
            if r.status_code == 200:
                # Healthy — try to get model name
                try:
                    mr = await client.get(f"{base}/v1/models")
                    if mr.status_code == 200:
                        models = mr.json().get("data", [])
                        model_id = models[0]["id"] if models else "unknown"
                        return {"ready": True, "status": "active", "detail": model_id}
                except Exception:
                    pass
                return {"ready": True, "status": "active", "detail": None}
    except Exception:
        pass

    # Not healthy — check if container is running and parse loading progress
    port = base.split(":")[-1].split("/")[0]
    container_map = {
        "8010": "autosre-vllm-local",
        "8003": "scribe-asr",
        "8002": "scribe-tts",
        "8001": "scribe-diarization",
    }
    container = container_map.get(port)
    if not container:
        return {"ready": False, "status": "down", "detail": None}

    try:
        ps = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", container],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if ps.returncode != 0:
            return {"ready": False, "status": "not started", "detail": "Container not found"}

        docker_state = ps.stdout.strip()
        if docker_state == "restarting":
            return {"ready": False, "status": "restarting", "detail": "Container restarting"}
        if docker_state != "running":
            return {"ready": False, "status": docker_state, "detail": None}

        # Container running but health check failed — parse logs for loading progress
        logs = subprocess.run(
            ["docker", "logs", container, "--tail", "50"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        log_text = (logs.stdout or "") + (logs.stderr or "")

        # vLLM loading stages (same as autosre TUI)
        stages = [
            ("Application startup complete", "Starting API..."),
            ("Available KV cache memory", "Allocating KV cache..."),
            ("Model loading took", "Profiling..."),
            ("Initializing", "Initializing engine..."),
            ("non-default args", "Initializing engine..."),
            ("INT8 LM Head", "Applying patches..."),
            ("Starting vLLM", "Starting vLLM..."),
        ]

        # Weight loading progress
        matches = re.findall(
            r"Loading.*safetensors.*?(\d+)%\s+Completed\s+\|\s+(\d+)/(\d+)", log_text
        )
        if matches:
            pct, done, total = matches[-1]
            return {
                "ready": False,
                "status": "loading",
                "detail": f"Loading weights: {pct}% ({done}/{total})",
            }

        for marker, msg in stages:
            if marker in log_text:
                return {"ready": False, "status": "loading", "detail": msg}

        return {"ready": False, "status": "loading", "detail": "Starting container..."}
    except Exception:
        return {"ready": False, "status": "unknown", "detail": None}


def _admin_settings_payload() -> dict:
    """Build the GET /api/admin/settings response body.

    Includes current values AND the option lists, so the UI can render
    dropdowns without a second round-trip.
    """
    from meeting_scribe.wifi import (
        _load_settings as _wifi_load_settings,
    )
    from meeting_scribe.wifi import (
        _nmcli_ap_is_active,
        _nmcli_read_live_ap_credentials,
        _wpa_supplicant_ap_security,
    )

    wifi_settings = _wifi_load_settings()
    wifi_mode = wifi_settings.get("wifi_mode", "off")

    # Live AP state from nmcli/wpa_cli — NOT just the state file
    wifi_active = _nmcli_ap_is_active()
    live_creds = _nmcli_read_live_ap_credentials() if wifi_active else None
    live_security = _wpa_supplicant_ap_security() if wifi_active else None

    payload = {
        "wifi_regdomain": _effective_regdomain(),
        "wifi_regdomain_current": _current_regdomain(),
        "wifi_regdomain_options": [
            {"code": code, "name": name} for code, name in _WIFI_REGDOMAIN_OPTIONS
        ],
        "wifi_mode": wifi_mode,
        "wifi_mode_options": [
            {"code": "off", "name": "Off"},
            {"code": "meeting", "name": "Meeting (rotating SSID, captive portal)"},
            {"code": "admin", "name": "Admin (fixed SSID, admin UI over WiFi)"},
        ],
        "wifi_active": wifi_active,
        "wifi_ssid": live_creds[0] if live_creds else None,
        "wifi_security": live_security,
        "admin_ssid": wifi_settings.get("admin_ssid", ""),
        "admin_password_set": bool(wifi_settings.get("admin_password")),
        "timezone": _effective_timezone(),
        "timezone_options": _timezone_options(),
        "dev_mode": _is_dev_mode(),
        "tts_voice_mode": _effective_tts_voice_mode(),
        "tts_voice_mode_options": [
            {"code": "studio", "name": "Studio voice (Qwen3-TTS, studio quality)"},
            {"code": "cloned", "name": "Participant voice (clone each speaker)"},
        ],
    }
    return payload


@app.post("/api/admin/drain")
async def post_admin_drain(request: fastapi.Request) -> JSONResponse:
    """Wait up to ``timeout`` seconds for translation + slide work to idle.

    Query params:
      - ``timeout`` (float, default 1.0) — max seconds to wait on this call.
      - ``force`` (bool, default false) — cancel every not-yet-started
        translation item AND abort the in-flight slide pipeline before
        returning.  Used by ``meeting-scribe drain --force`` for incidents
        where an operator chooses to drop in-flight work rather than wait.

    Response body:
      ``{"idle": bool, "translation_active": int, "translation_pending": int,
         "merge_gate_held": bool, "slide_in_flight": {...},
         "force_cancelled": {"translation": int, "slide": bool}?}``

    Callers (``meeting-scribe drain``) poll in a loop until ``idle=true`` or
    their own overall timeout expires.  The endpoint flushes the merge gate
    once at entry so a held final event doesn't hide behind "empty queue".
    """
    try:
        timeout_s = float(request.query_params.get("timeout", "1.0"))
    except ValueError:
        timeout_s = 1.0
    timeout_s = max(0.1, min(timeout_s, 10.0))  # clamp

    force_raw = request.query_params.get("force", "").lower()
    force = force_raw in ("1", "true", "yes")

    # Flush the merge gate so a held final event becomes visible as pending.
    if translation_queue is not None:
        await translation_queue.flush_merge_gate()

    force_cancelled: dict[str, int | bool] = {}
    if force:
        t_cancelled = translation_queue.cancel_all() if translation_queue else 0
        s_cancelled = (
            await slide_job_runner.cancel_current_job() if slide_job_runner is not None else False
        )
        force_cancelled = {"translation": t_cancelled, "slide": s_cancelled}
        if t_cancelled or s_cancelled:
            logger.warning(
                "drain --force: cancelled %d translation item(s), slide_aborted=%s",
                t_cancelled,
                s_cancelled,
            )

    deadline = asyncio.get_event_loop().time() + timeout_s
    while True:
        t_active = translation_queue.active_count() if translation_queue else 0
        t_pending = translation_queue.pending_count() if translation_queue else 0
        t_held = translation_queue.merge_gate_held() if translation_queue else False
        slide_info = (
            slide_job_runner.in_flight() if slide_job_runner is not None else {"running": False}
        )
        idle = (
            t_active == 0 and t_pending == 0 and not t_held and not slide_info.get("running", False)
        )
        if idle or asyncio.get_event_loop().time() >= deadline:
            body: dict[str, object] = {
                "idle": idle,
                "translation_active": t_active,
                "translation_pending": t_pending,
                "merge_gate_held": t_held,
                "slide_in_flight": slide_info,
            }
            if force:
                body["force_cancelled"] = force_cancelled
            return JSONResponse(body)
        await asyncio.sleep(0.1)


@app.post("/api/admin/pause-translation")
async def post_admin_pause_translation() -> JSONResponse:
    """Gate new translation intake.  Already-queued + in-flight items continue.

    Paired with ``/api/admin/resume-translation`` for operator-driven
    model-swap windows.  Called by ``meeting-scribe pause-translation``
    before a drain so new ASR output doesn't pile up against an
    about-to-unload backend.  Idempotent.
    """
    if translation_queue is None:
        return JSONResponse({"error": "Translation queue not initialised"}, status_code=503)
    translation_queue.pause()
    return JSONResponse({"paused": translation_queue.is_paused()})


@app.post("/api/admin/resume-translation")
async def post_admin_resume_translation() -> JSONResponse:
    """Re-open translation intake.  Idempotent."""
    if translation_queue is None:
        return JSONResponse({"error": "Translation queue not initialised"}, status_code=503)
    translation_queue.resume()
    return JSONResponse({"paused": translation_queue.is_paused()})


@app.get("/api/admin/refinement-stats")
async def get_admin_refinement_stats(
    drain_id: int | None = None,
    meeting_id: str | None = None,
) -> JSONResponse:
    """Return refinement counter snapshots for validation.

    Resolution order:
      1. ``drain_id`` (unambiguous) — returns the exact entry if present.
      2. ``meeting_id`` — returns the most recent drain entry for that
         meeting, plus ``other_drain_ids`` if earlier drains exist, plus
         a ``live`` block if the worker for that meeting is still
         running.
      3. Otherwise 404 — caller is asking about a meeting that never
         started the worker.
    """
    if drain_id is not None:
        entry = next((e for e in _refinement_drains if e.drain_id == drain_id), None)
        if entry is None:
            return JSONResponse(
                {"error": f"no drain entry for drain_id={drain_id}"},
                status_code=404,
            )
        return JSONResponse(_drain_entry_to_dict(entry))

    if meeting_id:
        entries = _find_drains_by_meeting(meeting_id)
        live_block: dict | None = None
        if (
            refinement_worker is not None
            and getattr(refinement_worker, "_meeting_id", None) == meeting_id
        ):
            live_block = {
                "meeting_id": meeting_id,
                "translate_calls": refinement_worker.translate_call_count,
                "asr_calls": refinement_worker.asr_call_count,
                "errors_at_stop": refinement_worker.last_error_count,
            }
        if not entries and live_block is None:
            return JSONResponse(
                {"error": (f"no active or recent refinement for meeting_id={meeting_id}")},
                status_code=404,
            )
        # Most recent = last entry with matching meeting_id (list is append-only).
        newest = entries[-1] if entries else None
        others = [e.drain_id for e in entries[:-1]] if entries else []
        return JSONResponse(
            {
                "drain": _drain_entry_to_dict(newest) if newest else None,
                "live": live_block,
                "other_drain_ids": others,
            }
        )

    return JSONResponse(
        {"error": "must pass drain_id or meeting_id"},
        status_code=400,
    )


@app.get("/api/meeting/{meeting_id}/polished-status")
async def get_polished_status(meeting_id: str) -> JSONResponse:
    """Polish-drain progress for a meeting.

    Returns the latest drain entry for the meeting (most-recent-wins)
    with the counter snapshot inlined so a polling client fetches state
    + counters in one round-trip.  If no drain entry exists but
    ``polished.json`` is on disk (post-restart state), returns
    ``state=complete`` with the file's mtime so the harness doesn't
    false-alarm after a server restart.
    """
    entries = _find_drains_by_meeting(meeting_id)
    if entries:
        newest = entries[-1]
        body = _drain_entry_to_dict(newest)
        polished_path = storage._meeting_dir(meeting_id) / "polished.json"
        body["polished_json_mtime"] = (
            polished_path.stat().st_mtime if polished_path.is_file() else None
        )
        return JSONResponse(body)

    # No registry entry — but the file may be on disk from a previous
    # process. Treat that as "complete" from disk.
    polished_path = storage._meeting_dir(meeting_id) / "polished.json"
    if polished_path.is_file():
        return JSONResponse(
            {
                "meeting_id": meeting_id,
                "state": "complete",
                "polished_json_mtime": polished_path.stat().st_mtime,
                "drain_id": None,
                "note": "read from disk — no in-memory drain entry",
            }
        )

    return JSONResponse(
        {
            "meeting_id": meeting_id,
            "state": "absent",
            "polished_json_mtime": None,
            "drain_id": None,
        },
        status_code=404,
    )


@app.get("/api/admin/settings")
async def get_admin_settings(request: fastapi.Request) -> JSONResponse:
    """Return admin-configurable runtime settings + selectable options.

    Fields:
      - ``wifi_regdomain``     — effective 2-letter country code in force
      - ``wifi_regdomain_current`` — live ``iw reg get`` value
      - ``wifi_regdomain_options`` — [{code, name}] supported-country list
      - ``timezone``           — effective IANA tz (empty = local time)
      - ``timezone_options``   — sorted IANA tz list

    The effective values come from config > persisted override > default,
    so the admin UI always shows the value actually in force.
    """
    return JSONResponse(_admin_settings_payload())


@app.put("/api/admin/settings")
async def put_admin_settings(request: fastapi.Request) -> JSONResponse:
    """Update admin-configurable runtime settings.

    Body accepts any subset of:
      - ``wifi_regdomain``: 2-letter ISO country code from the supported
        list. Persisted, applied via ``iw reg set`` immediately, and
        written to ``/etc/modprobe.d/cfg80211-<code>.conf`` for boot.
      - ``timezone``: IANA timezone name (e.g. ``Asia/Tokyo``) or empty
        string to clear and use the server's local time. Validated via
        ``zoneinfo.ZoneInfo`` before persisting.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be a JSON object"}, status_code=400)

    updates: dict = {}

    if "wifi_regdomain" in body:
        raw_code = body.get("wifi_regdomain")
        if not isinstance(raw_code, str):
            return JSONResponse({"error": "wifi_regdomain must be a string"}, status_code=400)
        code = raw_code.strip().upper()
        if len(code) != 2 or not code.isalpha():
            return JSONResponse(
                {"error": "wifi_regdomain must be a 2-letter ISO country code"},
                status_code=400,
            )
        if not _is_valid_regdomain(code):
            return JSONResponse(
                {"error": (f"wifi_regdomain {code!r} is not in the supported country list")},
                status_code=400,
            )
        updates["wifi_regdomain"] = code

    if "timezone" in body:
        raw_tz = body.get("timezone")
        if not isinstance(raw_tz, str):
            return JSONResponse({"error": "timezone must be a string"}, status_code=400)
        tz_name = raw_tz.strip()
        if tz_name and not _is_valid_timezone(tz_name):
            return JSONResponse(
                {"error": f"timezone {tz_name!r} is not a valid IANA name"},
                status_code=400,
            )
        updates["timezone"] = tz_name

    if "dev_mode" in body:
        raw_dev = body.get("dev_mode")
        if not isinstance(raw_dev, bool):
            return JSONResponse({"error": "dev_mode must be a boolean"}, status_code=400)
        updates["dev_mode"] = raw_dev

    if "tts_voice_mode" in body:
        raw_mode = body.get("tts_voice_mode")
        if not isinstance(raw_mode, str) or raw_mode not in ("studio", "cloned"):
            return JSONResponse(
                {"error": "tts_voice_mode must be 'studio' or 'cloned'"},
                status_code=400,
            )
        updates["tts_voice_mode"] = raw_mode

    # ── WiFi fields ──────────────────────────────────────────
    wifi_mode_changed = False
    new_wifi_mode: str | None = None

    if "wifi_mode" in body:
        raw_wm = body.get("wifi_mode")
        if not isinstance(raw_wm, str) or raw_wm not in ("off", "meeting", "admin"):
            return JSONResponse(
                {"error": "wifi_mode must be 'off', 'meeting', or 'admin'"},
                status_code=400,
            )
        from meeting_scribe.wifi import _load_settings as _wifi_settings

        current_mode = _wifi_settings().get("wifi_mode", "off")
        if raw_wm != current_mode:
            wifi_mode_changed = True
            new_wifi_mode = raw_wm
        updates["wifi_mode"] = raw_wm

    if "admin_ssid" in body:
        raw_ssid = body.get("admin_ssid")
        if not isinstance(raw_ssid, str):
            return JSONResponse({"error": "admin_ssid must be a string"}, status_code=400)
        ssid = raw_ssid.strip()
        ssid_bytes = ssid.encode("utf-8")
        if not (1 <= len(ssid_bytes) <= 32):
            return JSONResponse({"error": "admin_ssid must be 1-32 bytes"}, status_code=400)
        if not all(0x20 <= ord(c) <= 0x7E for c in ssid):
            return JSONResponse({"error": "admin_ssid must be printable ASCII"}, status_code=400)
        updates["admin_ssid"] = ssid

    if "admin_password" in body:
        raw_pw = body.get("admin_password")
        if not isinstance(raw_pw, str):
            return JSONResponse({"error": "admin_password must be a string"}, status_code=400)
        if not (8 <= len(raw_pw) <= 63):
            return JSONResponse(
                {"error": "admin_password must be 8-63 characters (WPA2/WPA3 constraint)"},
                status_code=400,
            )
        if not all(0x20 <= ord(c) <= 0x7E for c in raw_pw):
            return JSONResponse(
                {"error": "admin_password must be printable ASCII"}, status_code=400
            )
        updates["admin_password"] = raw_pw

    if not updates:
        return JSONResponse({"error": "no recognized settings in body"}, status_code=400)

    # ── WiFi mode change → async cutover with 202 ───────────
    if wifi_mode_changed and new_wifi_mode is not None:
        from meeting_scribe.wifi import build_config, wifi_switch

        # Validate the new config BEFORE committing anything
        try:
            if new_wifi_mode == "off":
                new_cfg = None
            else:
                # Use updates for admin_ssid/admin_password if they were
                # in this same PUT, otherwise build_config reads settings.
                new_cfg = build_config(
                    new_wifi_mode,
                    updates.get("admin_ssid") if new_wifi_mode == "admin" else None,
                    updates.get("admin_password") if new_wifi_mode == "admin" else None,
                    "a",
                    36,
                )
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        # Save non-wifi settings immediately (timezone, regdomain, etc.)
        non_wifi = {
            k: v
            for k, v in updates.items()
            if k not in ("wifi_mode", "admin_ssid", "admin_password")
        }
        if non_wifi:
            _save_settings_override(non_wifi)

        async def _do_switch():
            from meeting_scribe.wifi import wifi_down as _wd

            if new_cfg is None:
                await _wd()
            else:
                await wifi_switch(new_cfg)

        asyncio.create_task(_do_switch())
        return JSONResponse(
            {"status": "switching", "wifi_mode": new_wifi_mode},
            status_code=202,
        )

    # ── Non-wifi-mode-change path (immediate apply) ──────────
    _save_settings_override(updates)

    for key in ("wifi_regdomain", "timezone"):
        if key in updates:
            try:
                setattr(config, key, updates[key])
            except Exception:
                pass

    persistent_ok = True
    runtime_ok = True
    if "wifi_regdomain" in updates:
        loop = asyncio.get_event_loop()
        persistent_ok = await loop.run_in_executor(None, _ensure_regdomain_persistent)
        runtime_ok = await loop.run_in_executor(None, _ensure_regdomain)

    response = _admin_settings_payload()
    response["persistent_ok"] = persistent_ok
    response["runtime_ok"] = runtime_ok
    return JSONResponse(response)


@app.get("/metrics")
async def get_metrics() -> fastapi.Response:
    """Prometheus-compatible text metrics [Phase 2 optional].

    Exposes the same TTS counters and latency histograms as ``/api/status``
    but in a scrape-friendly text format. No prometheus_client dependency —
    the format is trivial and stable. Gauges only (no histograms); the
    rolling percentiles are exposed as separate gauges because the
    underlying windows are bounded deques, not Prometheus histograms.
    """
    lines: list[str] = []

    def _g(name: str, value: float | int, help_text: str = "") -> None:
        if help_text:
            lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {value}")

    def _pct(prefix: str, d: dict) -> None:
        sc = d.get("sample_count", 0)
        _g(f"{prefix}_sample_count", sc)
        for q in ("p50", "p95", "p99"):
            v = d.get(q)
            if v is not None:
                _g(f"{prefix}_{q}", v)

    snap = metrics.to_dict()
    tts = snap.get("tts", {})
    listener = snap.get("listener", {})
    loop_lag = snap.get("loop_lag_ms", {})

    _g("scribe_meeting_elapsed_s", snap.get("elapsed_s", 0))
    _g("scribe_asr_events_total", snap.get("asr_events", 0))
    _g("scribe_translations_completed_total", snap.get("translations_completed", 0))
    _g("scribe_translations_failed_total", snap.get("translations_failed", 0))

    _g("scribe_tts_queue_depth", tts.get("queue_depth", 0))
    _g("scribe_tts_queue_maxsize", tts.get("queue_maxsize", 0))
    _g("scribe_tts_workers_busy", tts.get("workers_busy", 0))
    _g("scribe_tts_container_concurrency", tts.get("container_concurrency", 0))
    _g("scribe_tts_submitted_total", tts.get("submitted", 0))
    _g("scribe_tts_delivered_total", tts.get("delivered", 0))
    _g("scribe_tts_timeouts_total", tts.get("timeouts", 0))
    _g("scribe_tts_oldest_inflight_age_ms", tts.get("oldest_inflight_age_ms", 0))
    for k, v in (tts.get("drops") or {}).items():
        _g(f"scribe_tts_drops_{k}_total", v)
    _pct("scribe_tts_synth_ms", tts.get("synth_ms") or {})
    _pct("scribe_tts_upstream_lag_ms", tts.get("upstream_lag_ms") or {})
    _pct("scribe_tts_post_translation_lag_ms", tts.get("tts_post_translation_lag_ms") or {})
    _pct("scribe_tts_end_to_end_lag_ms", tts.get("end_to_end_lag_ms") or {})

    health = {"healthy": 0, "degraded": 1, "stalled": 2}.get(tts.get("health", "healthy"), 0)
    _g("scribe_tts_health_state", health, help_text="0=healthy 1=degraded 2=stalled")

    _g("scribe_listener_connected", listener.get("connected", 0))
    _g("scribe_listener_deliveries_total", listener.get("deliveries", 0))
    _g("scribe_listener_send_failed_total", listener.get("send_failed", 0))
    _pct("scribe_listener_send_ms", listener.get("send_ms") or {})

    _pct("scribe_loop_lag_ms", loop_lag)

    crash = snap.get("crash")
    _g("scribe_crash_state", 1 if crash else 0)

    return fastapi.Response(
        content="\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4",
    )


@app.get("/api/status")
async def get_status(request: fastapi.Request) -> JSONResponse:
    """Current server and meeting status."""
    from meeting_scribe.gpu_monitor import get_system_resources, get_vram_usage

    gpu_data = None
    vram = get_vram_usage()
    if vram:
        gpu_data = {
            "vram_used_mb": vram.used_mb,
            "vram_total_mb": vram.total_mb,
            "vram_pct": round(vram.pct, 1),
        }

    # ROOT CAUSE, 2026-04-15: `get_system_resources()` internally calls
    # `subprocess.run(["docker", "stats", "--no-stream", ...], timeout=8)`
    # to populate container metrics. `docker stats` blocks 1.5–2.5 s even
    # with --no-stream because the daemon still samples every container
    # once. The previous code called this SYNCHRONOUSLY from this async
    # handler, freezing the entire event loop for the duration of the
    # subprocess. Under a 10-sec admin poll, that showed up as clockwork
    # 2.5 s event-loop stalls every 10–12 s — which was what was killing
    # the hotspot listener's WebSocket ping. Running the whole thing in
    # a thread via run_in_executor means the loop stays responsive; the
    # existing 5 s sys + 10 s container caches still limit how often we
    # actually shell out to `docker`.
    sys_res = await asyncio.get_event_loop().run_in_executor(None, get_system_resources)
    system_data = None
    if sys_res:
        system_data = {
            "cpu_pct": sys_res.cpu_pct,
            "mem_used_mb": sys_res.mem_used_mb,
            "mem_total_mb": sys_res.mem_total_mb,
            "mem_pct": sys_res.mem_pct,
            "load": [sys_res.load_1m, sys_res.load_5m, sys_res.load_15m],
            "uptime_s": sys_res.uptime_s,
            "containers": sys_res.containers,
        }

    # DEEP backend health — every "ready" status reflects an actual live
    # inference check (or a recent cached one), not just "the Python object
    # exists in memory". Without this, /api/status used to lie: if the
    # backend object was constructed but its container hit CUDA errors,
    # we'd still report "active".
    deep = await _deep_backend_health()

    def _backend_extra(obj) -> dict:
        """Extract model/url/failures from a backend object for status."""
        extra: dict = {}
        for attr in ("_model", "_vllm_model", "_model_name"):
            m = getattr(obj, attr, None)
            if m:
                extra["model"] = m
                break
        for attr in ("_base_url", "_vllm_url", "_url"):
            u = getattr(obj, attr, None)
            if u:
                extra["url"] = str(u)
                break
        cf = getattr(obj, "_consecutive_failures", None)
        if cf is not None:
            extra["consecutive_failures"] = cf
        return extra

    backend_details: dict[str, dict] = {}
    loading_probes: list[tuple[str, str]] = []
    # TTS URL may be a comma-separated pool — the loading probe only needs
    # one live endpoint, so take the first. The live pool state is already
    # reflected in deep-health ("ready") via the backend's own tracking.
    tts_probe_url = (config.tts_vllm_url or "").split(",")[0].strip()
    for name, backend_obj, url in [
        ("asr", asr_backend, config.asr_vllm_url),
        ("translate", translate_backend, config.translate_vllm_url),
        ("diarize", diarize_backend, config.diarize_url),
        ("tts", tts_backend, tts_probe_url),
        ("furigana", furigana_backend, ""),
    ]:
        dh = deep.get(name) or {}
        extra = _backend_extra(backend_obj) if backend_obj else {}
        if dh.get("ready"):
            backend_details[name] = {
                "ready": True,
                "status": "active",
                "detail": None,
                **extra,
            }
            continue

        # Not deep-ready — distinguish between "degraded" (had a real error)
        # and "loading/starting" (container not finished initializing).
        if backend_obj is not None and getattr(backend_obj, "degraded", False):
            backend_details[name] = {
                "ready": False,
                "status": "error",
                "detail": (
                    getattr(backend_obj, "last_error", None)
                    or dh.get("detail")
                    or "Backend degraded"
                ),
                **extra,
            }
            continue

        # Look up container state to produce a meaningful loading status
        loading_probes.append((name, url))

    # Probe unready backends in parallel for loading progress
    if loading_probes:
        results = await asyncio.gather(*[_probe_vllm_status(url) for _, url in loading_probes])
        for (name, _), result in zip(loading_probes, results):
            # If the container probe says "ready" but deep health failed,
            # the backend is up but scribe-side wiring isn't complete.
            if result.get("ready"):
                detail = deep.get(name, {}).get("detail") or "Connecting..."
                backend_details[name] = {
                    "ready": False,
                    "status": "loading",
                    "detail": detail,
                }
            else:
                backend_details[name] = result

    return JSONResponse(
        {
            "meeting": {
                "id": current_meeting.meeting_id if current_meeting else None,
                "state": current_meeting.state.value if current_meeting else None,
            },
            "backends": {
                "asr": backend_details.get("asr", {}).get("ready", False),
                "diarize": backend_details.get("diarize", {}).get("ready", False),
                "translate": backend_details.get("translate", {}).get("ready", False),
                "tts": backend_details.get("tts", {}).get("ready", False),
                "furigana": backend_details.get("furigana", {}).get("ready", False),
            },
            "backend_details": backend_details,
            "language_correction": (
                __import__(
                    "meeting_scribe.language_correction", fromlist=["correction_stats"]
                ).correction_stats.snapshot()
            ),
            "connections": len(ws_connections),
            "audio_out_connections": len(_audio_out_clients),
            "metrics": metrics.to_dict(),
            "gpu": gpu_data,
            "system": system_data,
            "guest": _is_hotspot_client(request),
            "dev_mode": _is_dev_mode(),
        }
    )


# ── Meetings History ──────────────────────────────────────────


@app.get("/api/meetings")
async def list_meetings() -> JSONResponse:
    """List all meetings with metadata + short summary.

    Extracts `executive_summary` and first topic from summary.json so the
    meetings list can show context without a second fetch per meeting.
    """
    import json as _json

    meetings_dir = storage._meetings_dir
    results = []
    if meetings_dir.exists():
        for d in meetings_dir.iterdir():
            meta_path = d / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = _json.loads(meta_path.read_text())
                journal_path = d / "journal.jsonl"
                event_count = 0
                if journal_path.exists():
                    event_count = sum(1 for _ in journal_path.open())

                # Read summary.json for short context (executive_summary + topics)
                summary_path = d / "summary.json"
                executive_summary = None
                topics_preview: list[str] = []
                if summary_path.exists():
                    try:
                        sdata = _json.loads(summary_path.read_text())
                        es = sdata.get("executive_summary")
                        if isinstance(es, str):
                            executive_summary = es
                        topics_raw = sdata.get("topics", [])
                        if isinstance(topics_raw, list):
                            for t in topics_raw[:5]:
                                if isinstance(t, str):
                                    topics_preview.append(t)
                                elif isinstance(t, dict) and "title" in t:
                                    topics_preview.append(str(t["title"]))
                    except Exception:
                        pass

                results.append(
                    {
                        "meeting_id": meta["meeting_id"],
                        "state": meta["state"],
                        "created_at": meta.get("created_at", ""),
                        "event_count": event_count,
                        "has_room": (d / "room.json").exists(),
                        "has_speakers": (d / "speakers.json").exists(),
                        "has_summary": summary_path.exists(),
                        "has_timeline": (d / "timeline.json").exists(),
                        "has_slides": (d / "slides" / "active_deck_id").exists(),
                        "executive_summary": executive_summary,
                        "topics": topics_preview,
                        "is_favorite": bool(meta.get("is_favorite", False)),
                    }
                )
            except Exception:
                continue

    # Strict chronological order (newest first). Favorites used to rise
    # to the top here, but that reshuffled the timeline whenever a star
    # was toggled. The UI now has a separate "Favorites" tab that filters
    # the list, so favoriting can leave the chronological order intact.
    results.sort(key=lambda m: m["created_at"], reverse=True)
    return JSONResponse({"meetings": results})


@app.put("/api/meetings/{meeting_id}/events/{segment_id}/speaker")
async def update_segment_speaker(
    meeting_id: str, segment_id: str, request: fastapi.Request
) -> JSONResponse:
    """Rename a speaker. Updates journal, detected_speakers.json, and room.json."""
    import json as _json

    body = await request.json()
    speaker_name = (body.get("speaker_name") or body.get("display_name") or "").strip()
    old_name = (body.get("old_name") or "").strip()
    if not speaker_name:
        return JSONResponse({"error": "speaker_name required"}, status_code=400)

    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir:
        return JSONResponse({"error": "Invalid meeting ID"}, status_code=400)
    journal_path = meeting_dir / "journal.jsonl"
    if not journal_path.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    # 1. Append correction to journal
    correction = {
        "type": "speaker_correction",
        "segment_id": segment_id,
        "speaker_name": speaker_name,
    }
    with journal_path.open("a") as f:
        f.write(_json.dumps(correction) + "\n")

    # 2. Update detected_speakers.json (if old_name provided)
    if old_name:
        speakers_path = meeting_dir / "detected_speakers.json"
        if speakers_path.exists():
            with suppress(Exception):
                speakers = _json.loads(speakers_path.read_text())
                changed = False
                for s in speakers:
                    if s.get("display_name") == old_name:
                        s["display_name"] = speaker_name
                        changed = True
                if changed:
                    speakers_path.write_text(_json.dumps(speakers, indent=2))

        # 3. Update room.json seat names
        room_path = meeting_dir / "room.json"
        if room_path.exists():
            with suppress(Exception):
                room = _json.loads(room_path.read_text())
                changed = False
                for seat in room.get("seats", []):
                    if seat.get("speaker_name") == old_name:
                        seat["speaker_name"] = speaker_name
                        changed = True
                if changed:
                    room_path.write_text(_json.dumps(room, indent=2))

        # speaker_lanes.json uses cluster_id keys — names come from detected_speakers (already updated above)

    return JSONResponse(
        {"status": "updated", "segment_id": segment_id, "speaker_name": speaker_name}
    )


@app.get("/api/meetings/{meeting_id}/speakers")
async def get_meeting_speakers(meeting_id: str) -> JSONResponse:
    """Get detected speakers for a meeting."""
    speakers = storage.load_detected_speakers(meeting_id)
    return JSONResponse({"speakers": speakers})


@app.get("/api/debug/diarize")
async def debug_diarize() -> JSONResponse:
    """Debug endpoint: inspect the diarization backend state.

    Returns the result cache size, time range, pending catch-up queue,
    and other diagnostics for investigating why speakers aren't being
    attributed to events.
    """
    state: dict = {
        "diarize_backend_exists": diarize_backend is not None,
        "pending_catchup_events": len(_pending_speaker_events),
        "catchup_task_running": (
            _speaker_catchup_task is not None and not _speaker_catchup_task.done()
        ),
    }
    if diarize_backend is not None:
        state["base_offset_samples"] = getattr(diarize_backend, "_base_offset", None)
        state["buffer_samples"] = getattr(diarize_backend, "_buffer_samples", None)
        state["buffer_threshold_samples"] = getattr(diarize_backend, "_buffer_threshold", None)
        cache = getattr(diarize_backend, "_result_cache", None)
        if cache is not None:
            state["result_cache_size"] = len(cache)
            if cache:
                items = list(cache.values())
                first = items[0]
                last = items[-1]
                state["result_cache_range_ms"] = {
                    "first_start": first.start_ms,
                    "first_end": first.end_ms,
                    "last_start": last.start_ms,
                    "last_end": last.end_ms,
                }
                # Unique global cluster IDs present
                state["unique_cluster_ids"] = sorted({dr.cluster_id for dr in items})
        # Global centroid state
        gc = getattr(diarize_backend, "_global_centroids", None)
        if gc is not None:
            state["global_centroids_count"] = len(gc)
            state["next_global_id"] = getattr(diarize_backend, "_next_global_id", None)
        # Failure state
        state["degraded"] = getattr(diarize_backend, "degraded", False)
        state["consecutive_failures"] = getattr(diarize_backend, "_consecutive_failures", 0)
        state["last_error"] = getattr(diarize_backend, "last_error", None)

    # Sample a few pending events so we can see their time ranges
    if _pending_speaker_events:
        samples = []
        for sid in list(_pending_speaker_events.keys())[:5]:
            e = _pending_speaker_events[sid]
            samples.append(
                {
                    "segment_id": sid,
                    "start_ms": getattr(e, "start_ms", None),
                    "end_ms": getattr(e, "end_ms", None),
                    "text_preview": getattr(e, "text", "") or "",
                }
            )
        state["pending_samples"] = samples

    return JSONResponse(state)


@app.put("/api/meetings/{meeting_id}/clusters/{cluster_id}/name")
async def rename_cluster(
    meeting_id: str, cluster_id: str, request: fastapi.Request
) -> JSONResponse:
    """Rename all segments in a cluster at once.

    Writes speaker_correction entries for every segment belonging to the
    given cluster_id, updates detected_speakers.json, and broadcasts the
    rename to connected WebSocket clients so the live UI updates immediately.

    This is the "click speaker in virtual table → rename" flow.
    """
    import json as _json

    body = await request.json()
    new_name = (body.get("speaker_name") or body.get("display_name") or "").strip()
    if not new_name:
        return JSONResponse({"error": "speaker_name required"}, status_code=400)

    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir:
        return JSONResponse({"error": "Invalid meeting ID"}, status_code=400)
    journal_path = meeting_dir / "journal.jsonl"
    if not journal_path.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    # Parse cluster_id (may be int or string-int)
    try:
        target_cid = int(cluster_id)
    except (TypeError, ValueError):
        return JSONResponse({"error": "Invalid cluster_id"}, status_code=400)

    # Find all segments belonging to this cluster (apply corrections + dedup)
    affected_segment_ids: list[str] = []
    events_by_sid: dict[str, dict] = {}
    for line in journal_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = _json.loads(line)
        except Exception:
            continue
        if e.get("type") == "speaker_correction":
            continue
        if not e.get("is_final") or not e.get("text"):
            continue
        sid = e.get("segment_id")
        if not sid:
            continue
        rev = e.get("revision", 0)
        if sid not in events_by_sid or rev > events_by_sid[sid].get("revision", 0):
            events_by_sid[sid] = e

    for sid, event in events_by_sid.items():
        sp = (event.get("speakers") or [{}])[0]
        if sp.get("cluster_id") == target_cid:
            affected_segment_ids.append(sid)

    if not affected_segment_ids:
        return JSONResponse(
            {"error": f"No segments found for cluster_id={target_cid}"},
            status_code=404,
        )

    # Append bulk corrections to journal
    with journal_path.open("a") as f:
        for sid in affected_segment_ids:
            f.write(
                _json.dumps(
                    {
                        "type": "speaker_correction",
                        "segment_id": sid,
                        "speaker_name": new_name,
                    }
                )
                + "\n"
            )

    # Update detected_speakers.json — find the entry for this cluster_id
    speakers_path = meeting_dir / "detected_speakers.json"
    if speakers_path.exists():
        with suppress(Exception):
            speakers = _json.loads(speakers_path.read_text())
            for s in speakers:
                if s.get("cluster_id") == target_cid:
                    s["display_name"] = new_name
            speakers_path.write_text(_json.dumps(speakers, indent=2))

    # Broadcast to live WS clients so the UI updates immediately
    try:
        await _broadcast_json(
            {
                "type": "speaker_rename",
                "cluster_id": target_cid,
                "display_name": new_name,
                "affected_segments": len(affected_segment_ids),
            }
        )
    except Exception:
        pass

    logger.info(
        "Cluster rename: cluster %d → '%s' (%d segments)",
        target_cid,
        new_name,
        len(affected_segment_ids),
    )

    return JSONResponse(
        {
            "status": "renamed",
            "cluster_id": target_cid,
            "display_name": new_name,
            "affected_segments": len(affected_segment_ids),
        }
    )


@app.get("/api/meetings/{meeting_id}/room")
async def get_meeting_room(meeting_id: str) -> JSONResponse:
    """Get the persisted room layout for a meeting."""
    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir:
        return JSONResponse({"error": "Invalid meeting ID"}, status_code=400)
    layout = storage.load_room_layout(meeting_id)
    if layout is None:
        return JSONResponse({"error": "No room layout for this meeting"}, status_code=404)
    return JSONResponse(layout.model_dump())


@app.put("/api/meetings/{meeting_id}/room/layout")
async def put_meeting_room_layout(meeting_id: str, request: fastapi.Request) -> JSONResponse:
    """Persist an updated room layout for a meeting.

    Works for both active meetings (live edit) and past meetings (review edit).
    If the meeting is currently recording, broadcasts a room_layout_update
    event over WebSocket so all connected clients see the change.
    """
    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir:
        return JSONResponse({"error": "Invalid meeting ID"}, status_code=400)
    try:
        body = await request.json()
        layout = RoomLayout.model_validate(body)
    except Exception as e:
        return JSONResponse({"error": f"Invalid layout: {e}"}, status_code=422)

    storage.save_room_layout(meeting_id, layout)

    # Broadcast to all WS clients if this is the active meeting
    if current_meeting and current_meeting.meeting_id == meeting_id:
        await _broadcast_json(
            {
                "type": "room_layout_update",
                "layout": layout.model_dump(),
            }
        )

    return JSONResponse({"status": "ok", "layout": layout.model_dump()})


@app.post("/api/meetings/{meeting_id}/speakers/assign")
async def assign_speaker_to_seat(
    meeting_id: str,
    request: fastapi.Request,
) -> JSONResponse:
    """Bind a detected voice cluster to a seat and/or name.

    Body: {cluster_id: int, seat_id: str | null, display_name: str}

    Atomic multi-file update:
    - Updates room.json (sets speaker_name on matching seat)
    - Updates detected_speakers.json (sets display_name for cluster_id)
    - Rewrites journal.jsonl (sets speakers[0].identity for matching events)
    - Regenerates speaker_lanes.json via _generate_speaker_data

    Returns the updated detected_speakers list.
    """
    import json as _json

    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir:
        return JSONResponse({"error": "Invalid meeting ID"}, status_code=400)

    try:
        body = await request.json()
        cluster_id = int(body.get("cluster_id"))
        seat_id = body.get("seat_id")  # may be None to unbind
        display_name = str(body.get("display_name", "")).strip()
    except Exception as e:
        return JSONResponse({"error": f"Invalid request: {e}"}, status_code=422)

    if not display_name:
        return JSONResponse({"error": "display_name required"}, status_code=422)

    # 1. Update room.json — find seat_id, set speaker_name
    layout = storage.load_room_layout(meeting_id)
    if layout and seat_id:
        for seat in layout.seats:
            if seat.seat_id == seat_id:
                seat.speaker_name = display_name
            elif seat.speaker_name == display_name:
                # Unbind any other seat that had this name
                seat.speaker_name = ""
        storage.save_room_layout(meeting_id, layout)

    # 2. Update detected_speakers.json
    speakers_path = meeting_dir / "detected_speakers.json"
    if speakers_path.exists():
        try:
            speakers_list = _json.loads(speakers_path.read_text())
            for sp in speakers_list:
                # Match by cluster_id stored in speaker_id or as int
                sp_cluster = sp.get("cluster_id", sp.get("speaker_id"))
                try:
                    sp_cluster_int = int(sp_cluster) if sp_cluster is not None else -1
                except (ValueError, TypeError):
                    sp_cluster_int = -1
                if sp_cluster_int == cluster_id:
                    sp["display_name"] = display_name
            speakers_path.write_text(_json.dumps(speakers_list, indent=2))
        except (_json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to update detected_speakers.json: %s", e)

    # 3. Rewrite journal.jsonl — set speakers[0].identity for matching events
    updated_events = storage.update_journal_speaker_identity(
        meeting_id,
        cluster_id,
        display_name,
    )

    # 4. Regenerate speaker_lanes.json + timeline.json from the updated
    # journal so EVERY view that queries the meeting API (post-finalize
    # replay, participant list, timeline scrubber, summary) picks up the
    # rename without a separate refresh.
    try:
        journal_path = meeting_dir / "journal.jsonl"
        if journal_path.exists():
            _generate_speaker_data(meeting_dir, journal_path, _json)
        _generate_timeline(meeting_id)
    except Exception as e:
        logger.warning("Failed to regenerate speaker data: %s", e)

    # 4b. Summary.json bakes the old name into topics / action-items /
    # speaker_stats. Re-run summary in the background so the review view
    # surfaces the new name everywhere without blocking this response.
    # Best-effort: we've already persisted the rename, so summary regen
    # can fail silently.
    async def _regen_summary():
        try:
            summary_path = meeting_dir / "summary.json"
            if summary_path.exists():
                from meeting_scribe.summary import generate_summary

                summary = await generate_summary(meeting_dir, vllm_url=config.translate_vllm_url)
                if summary and not summary.get("error"):
                    summary_path.write_text(_json.dumps(summary, indent=2, ensure_ascii=False))
                    # Tell any open review UI to reload the summary panel.
                    await _broadcast_json(
                        {
                            "type": "summary_regenerated",
                            "meeting_id": meeting_id,
                        }
                    )
        except Exception as exc:
            logger.warning("Rename: summary regen failed: %s", exc)

    asyncio.create_task(_regen_summary())

    # 5. Broadcast if this is the active meeting
    if current_meeting and current_meeting.meeting_id == meeting_id:
        await _broadcast_json(
            {
                "type": "speaker_assignment",
                "cluster_id": cluster_id,
                "seat_id": seat_id,
                "display_name": display_name,
            }
        )

    updated_speakers = storage.load_detected_speakers(meeting_id)
    return JSONResponse(
        {
            "status": "ok",
            "cluster_id": cluster_id,
            "display_name": display_name,
            "seat_id": seat_id,
            "updated_events": updated_events,
            "speakers": updated_speakers,
        }
    )


@app.get("/api/meetings/{meeting_id}/tts/{segment_id}")
async def get_tts_audio(meeting_id: str, segment_id: str) -> fastapi.responses.Response:
    """Get synthesized TTS audio for a specific segment."""
    from fastapi.responses import FileResponse

    tts_path = _safe_segment_path(meeting_id, "tts", f"{segment_id}.wav")
    if not tts_path or not tts_path.exists():
        return JSONResponse({"error": "No TTS audio for this segment"}, status_code=404)
    return FileResponse(tts_path, media_type="audio/wav")


@app.get("/api/meetings/{meeting_id}/audio")
async def get_meeting_audio(
    meeting_id: str, request: fastapi.Request
) -> fastapi.responses.Response:
    """Stream meeting audio as WAV for a time range.

    Query params: start_ms (default 0), end_ms (default full duration).
    Returns audio/wav with proper headers for browser <audio> element.
    """
    import struct as _struct

    from fastapi.responses import StreamingResponse

    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir:
        return JSONResponse({"error": "Invalid meeting ID"}, status_code=400)
    pcm_path = meeting_dir / "audio" / "recording.pcm"
    if not pcm_path.exists():
        return JSONResponse({"error": "No audio recording"}, status_code=404)

    start_ms = int(request.query_params.get("start_ms", 0))
    end_ms_param = request.query_params.get("end_ms")
    total_duration = storage.audio_duration_ms(meeting_id)
    end_ms = int(end_ms_param) if end_ms_param else total_duration

    # Clamp
    start_ms = max(0, min(start_ms, total_duration))
    end_ms = max(start_ms, min(end_ms, total_duration))

    pcm_data = storage.read_audio_segment(meeting_id, start_ms, end_ms)
    if not pcm_data:
        return JSONResponse({"error": "No audio in range"}, status_code=404)

    # Build WAV header
    sample_rate = 16000
    channels = 1
    bits_per_sample = 16
    data_size = len(pcm_data)
    header = _struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        sample_rate * channels * bits_per_sample // 8,
        channels * bits_per_sample // 8,
        bits_per_sample,
        b"data",
        data_size,
    )

    def stream():
        yield header
        # Stream PCM in 8KB chunks
        for i in range(0, len(pcm_data), 8192):
            yield pcm_data[i : i + 8192]

    return StreamingResponse(
        stream(),
        media_type="audio/wav",
        headers={
            "Content-Length": str(44 + data_size),
            "Accept-Ranges": "bytes",
        },
    )


@app.get("/api/meetings/{meeting_id}/timeline")
async def get_timeline(meeting_id: str) -> JSONResponse:
    """Get the timeline manifest for podcast player and speaker lane view."""
    import json as _json

    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir:
        return JSONResponse({"error": "Invalid meeting ID"}, status_code=400)
    path = meeting_dir / "timeline.json"
    if not path.exists():
        return JSONResponse({"error": "No timeline"}, status_code=404)
    segments = _json.loads(path.read_text())
    if isinstance(segments, list):
        duration_ms = max((s.get("end_ms", 0) for s in segments), default=0)
    else:
        duration_ms = segments.get("duration_ms", 0)
        segments = segments.get("segments", [])

    # Include speaker lanes if available
    lanes_path = meeting_dir / "speaker_lanes.json"
    speaker_lanes = {}
    if lanes_path.exists():
        speaker_lanes = _json.loads(lanes_path.read_text())

    # Include detected speaker info
    speakers_path = meeting_dir / "detected_speakers.json"
    detected_speakers = []
    if speakers_path.exists():
        detected_speakers = _json.loads(speakers_path.read_text())

    return JSONResponse(
        {
            "duration_ms": duration_ms,
            "segments": segments,
            "speaker_lanes": speaker_lanes,
            "speakers": detected_speakers,
        }
    )


@app.get("/api/meetings/{meeting_id}/polished")
async def get_polished(meeting_id: str) -> JSONResponse:
    """Get polished transcript from the refinement worker."""
    import json as _json

    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir:
        return JSONResponse({"error": "Invalid meeting ID"}, status_code=400)
    path = meeting_dir / "polished.json"
    if not path.exists():
        return JSONResponse({"error": "No polished transcript"}, status_code=404)
    return JSONResponse(_json.loads(path.read_text()))


@app.get("/api/meetings/{meeting_id}/export")
async def export_meeting(
    meeting_id: str, format: str = "md", lang: str | None = None
) -> fastapi.responses.Response:
    """Export meeting in various formats.

    Query params:
        format: "md" (Markdown), "txt" (plain text), "zip" (full archive)
        lang: "en" or "ja" (for txt format only — filters by language)
    """
    from meeting_scribe.export import meeting_to_markdown, meeting_to_zip, transcript_to_text

    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir or not meeting_dir.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    if format == "md":
        content: str | bytes = meeting_to_markdown(meeting_dir)
        return fastapi.responses.Response(
            content=content,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="meeting-{meeting_id}.md"'},
        )
    elif format == "txt":
        journal = meeting_dir / "journal.jsonl"
        if not journal.exists():
            return JSONResponse({"error": "No transcript"}, status_code=404)
        content = transcript_to_text(journal, lang=lang)
        lang_suffix = f"-{lang}" if lang else ""
        return fastapi.responses.Response(
            content=content,
            media_type="text/plain; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="transcript-{meeting_id}{lang_suffix}.txt"'
            },
        )
    elif format == "zip":
        content = meeting_to_zip(meeting_dir)
        return fastapi.responses.Response(
            content=content,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="meeting-{meeting_id}.zip"'},
        )
    else:
        return JSONResponse({"error": f"Unknown format: {format}"}, status_code=400)


async def _translate_text_via_vllm(text: str, target_lang: str) -> str:
    """Translate a single text string via the live vLLM translation backend.

    Returns the original text on any failure — summary translation is
    best-effort and shouldn't blow up the endpoint if the backend is busy.
    """
    if not text or not text.strip():
        return text
    if translate_backend is None:
        return text
    client = getattr(translate_backend, "_client", None)
    model = getattr(translate_backend, "_model", None)
    if client is None or model is None:
        return text
    from meeting_scribe.languages import LANGUAGE_REGISTRY

    target_name = (
        LANGUAGE_REGISTRY[target_lang].name if target_lang in LANGUAGE_REGISTRY else target_lang
    )
    try:
        resp = await client.post(
            f"{config.translate_vllm_url.rstrip('/')}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            f"Translate the user's text into {target_name}. "
                            f"Preserve markdown, line breaks, and any speaker "
                            f"labels (e.g. 'Speaker 1:'). Return only the "
                            f"translation — no preamble, no quotation marks."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                "temperature": 0.2,
                "max_tokens": min(2048, max(64, int(len(text) * 1.5))),
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "priority": 10,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning("Summary text translation failed: %s", exc)
        return text


async def _translate_summary(summary: dict, target_lang: str) -> dict:
    """Translate every user-visible text field in a summary dict.

    Walks the v2 summary schema (executive_summary, key_insights[],
    action_items[], topics[], decisions[], key_quotes[]) and translates
    the textual leaves. Speaker names, cluster_ids, dates, and other
    non-prose fields are left intact.
    """
    import json as _json

    out = _json.loads(_json.dumps(summary))  # deep copy via JSON

    if isinstance(out.get("executive_summary"), str):
        out["executive_summary"] = await _translate_text_via_vllm(
            out["executive_summary"], target_lang
        )

    for insight in out.get("key_insights", []) or []:
        if isinstance(insight.get("title"), str):
            insight["title"] = await _translate_text_via_vllm(insight["title"], target_lang)
        if isinstance(insight.get("description"), str):
            insight["description"] = await _translate_text_via_vllm(
                insight["description"], target_lang
            )

    for item in out.get("action_items", []) or []:
        if isinstance(item.get("category"), str):
            item["category"] = await _translate_text_via_vllm(item["category"], target_lang)
        if isinstance(item.get("task"), str):
            item["task"] = await _translate_text_via_vllm(item["task"], target_lang)

    for topic in out.get("topics", []) or []:
        if isinstance(topic.get("title"), str):
            topic["title"] = await _translate_text_via_vllm(topic["title"], target_lang)
        if isinstance(topic.get("description"), str):
            topic["description"] = await _translate_text_via_vllm(topic["description"], target_lang)

    if isinstance(out.get("decisions"), list):
        out["decisions"] = [
            await _translate_text_via_vllm(d, target_lang) if isinstance(d, str) else d
            for d in out["decisions"]
        ]

    for q in out.get("key_quotes", []) or []:
        if isinstance(q, dict) and isinstance(q.get("text"), str):
            q["text"] = await _translate_text_via_vllm(q["text"], target_lang)

    out.setdefault("translation_meta", {})["language"] = target_lang
    return out


@app.get("/api/meetings/{meeting_id}/summary")
async def get_meeting_summary(
    meeting_id: str,
    lang: str | None = None,
) -> JSONResponse:
    """Get AI-generated meeting summary with speaker name corrections applied.

    ``lang``: optional ISO code from the meeting's language pair. When
    given AND the cached translation doesn't exist yet, every text field
    in the summary is translated via the live translation backend and
    cached as ``summary_{lang}.json``. Subsequent requests are instant.
    """
    import json as _json

    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir:
        return JSONResponse({"error": "Invalid meeting ID"}, status_code=400)
    path = meeting_dir / "summary.json"
    if not path.exists():
        return JSONResponse({"error": "No summary available"}, status_code=404)

    # Validate ``lang`` against the meeting's language pair so we don't
    # spawn translations into arbitrary unrelated languages.
    requested_lang = (lang or "").strip().lower() or None
    if requested_lang:
        meta_path = meeting_dir / "meta.json"
        try:
            meeting_meta = _json.loads(meta_path.read_text()) if meta_path.exists() else {}
            valid = set(meeting_meta.get("language_pair", []) or [])
        except Exception:
            valid = set()
        if requested_lang not in valid:
            return JSONResponse(
                {
                    "error": "lang not in this meeting's language pair",
                    "supported": sorted(valid),
                },
                status_code=400,
            )
        # Serve the translated cache if it exists
        cached_path = meeting_dir / f"summary_{requested_lang}.json"
        if cached_path.exists():
            path = cached_path
        else:
            try:
                summary = _json.loads(path.read_text())
                translated = await _translate_summary(summary, requested_lang)
                cached_path.write_text(_json.dumps(translated, ensure_ascii=False, indent=2))
                path = cached_path
            except Exception as exc:
                logger.exception("Summary translation to %s failed", requested_lang)
                return JSONResponse(
                    {"error": f"Summary translation failed: {exc}"},
                    status_code=500,
                )

    summary = _json.loads(path.read_text())

    # Build name mapping from detected_speakers.json (updated by rename API)
    # The summary was generated with original "Speaker N" names.
    # detected_speakers.json has been updated with renamed display_names.
    name_map: dict[str, str] = {}
    speakers_path = meeting_dir / "detected_speakers.json"
    if speakers_path.exists():
        with suppress(Exception):
            current_speakers = _json.loads(speakers_path.read_text())
            for s in current_speakers:
                cid = s.get("cluster_id")
                current_name = s.get("display_name", "")
                original_name = f"Speaker {cid}"
                if current_name and current_name != original_name:
                    name_map[original_name] = current_name

    # Apply name corrections to summary
    if name_map:
        # Update speaker_stats names
        for stat in summary.get("speaker_stats", []):
            if stat.get("name") in name_map:
                stat["name"] = name_map[stat["name"]]

        # Update action item assignees
        for item in summary.get("action_items", []):
            if item.get("assignee") in name_map:
                item["assignee"] = name_map[item["assignee"]]

        # Update text references in executive_summary, topics, decisions, v2 fields
        for old, new in name_map.items():
            if summary.get("executive_summary"):
                summary["executive_summary"] = summary["executive_summary"].replace(old, new)
            for topic in summary.get("topics", []):
                if topic.get("description"):
                    topic["description"] = topic["description"].replace(old, new)
            summary["decisions"] = [d.replace(old, new) for d in summary.get("decisions", [])]
            # V2 fields: key_insights speaker lists + descriptions
            for insight in summary.get("key_insights", []):
                if insight.get("description"):
                    insight["description"] = insight["description"].replace(old, new)
                insight["speakers"] = [new if s == old else s for s in insight.get("speakers", [])]
            # V2 fields: key_quotes speaker attribution
            for quote in summary.get("key_quotes", []):
                if isinstance(quote, dict):
                    if quote.get("speaker") == old:
                        quote["speaker"] = new
                    if quote.get("text"):
                        quote["text"] = quote["text"].replace(old, new)

    return JSONResponse(summary)


@app.post("/api/meetings/{meeting_id}/ask", response_model=None)
async def ask_meeting(
    meeting_id: str, request: fastapi.Request
) -> fastapi.responses.StreamingResponse | JSONResponse:
    """Ask a question about a meeting transcript. Streams SSE answer chunks."""
    from meeting_scribe.qa import ask_meeting_question

    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir:
        return JSONResponse({"error": "Invalid meeting ID"}, status_code=400)
    if not meeting_dir.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    question = body.get("question", "").strip()
    if not question:
        return JSONResponse({"error": "Missing 'question' field"}, status_code=400)
    if len(question) > 2000:
        return JSONResponse({"error": "Question too long (max 2000 chars)"}, status_code=400)

    return fastapi.responses.StreamingResponse(
        ask_meeting_question(
            meeting_dir=meeting_dir,
            question=question,
            vllm_url=config.translate_vllm_url,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/api/meetings/{meeting_id}")
async def delete_meeting(meeting_id: str) -> JSONResponse:
    """Delete a meeting and all its artifacts."""
    import shutil

    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir or not meeting_dir.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    # Don't delete the currently active meeting
    if current_meeting and current_meeting.meeting_id == meeting_id:
        return JSONResponse({"error": "Cannot delete active meeting"}, status_code=400)

    shutil.rmtree(meeting_dir)
    logger.info("Deleted meeting: %s", meeting_id)
    return JSONResponse({"status": "deleted", "meeting_id": meeting_id})


@app.patch("/api/meetings/{meeting_id}")
async def update_meeting(meeting_id: str, request: fastapi.Request) -> JSONResponse:
    """Update editable fields on a meeting's metadata.

    Currently supports: ``is_favorite`` (bool). Only the listed keys are
    written; everything else in the request body is ignored. Persists via
    the storage layer's atomic write-rename so concurrent reads never see
    a half-written meta.json.
    """
    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir or not meeting_dir.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "Body must be a JSON object"}, status_code=400)

    try:
        meta = storage._read_meta(meeting_id)
    except FileNotFoundError:
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    changed: list[str] = []
    if "is_favorite" in body:
        new_val = bool(body["is_favorite"])
        if meta.is_favorite != new_val:
            meta.is_favorite = new_val
            changed.append("is_favorite")

    if not changed:
        return JSONResponse({"meeting_id": meeting_id, "changed": []})

    storage._write_meta(meta)
    logger.info("Updated meeting %s: %s", meeting_id, ", ".join(changed))
    return JSONResponse(
        {
            "meeting_id": meeting_id,
            "changed": changed,
            "is_favorite": meta.is_favorite,
        }
    )


@app.get("/api/meetings/{meeting_id}")
async def get_meeting(meeting_id: str) -> JSONResponse:
    """Get a specific meeting's full data (meta + transcript + room layout)."""
    import json as _json

    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir or not meeting_dir.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    result = {}

    # Meta
    meta_path = meeting_dir / "meta.json"
    if meta_path.exists():
        result["meta"] = _json.loads(meta_path.read_text())

    # Room layout
    room_path = meeting_dir / "room.json"
    if room_path.exists():
        result["room"] = _json.loads(room_path.read_text())

    # Transcript events — load and apply speaker corrections
    journal_path = meeting_dir / "journal.jsonl"
    events = []
    corrections: dict[str, str] = {}  # segment_id → speaker_name
    if journal_path.exists():
        for line in journal_path.open():
            line = line.strip()
            if not line:
                continue
            with suppress(Exception):
                entry = _json.loads(line)
                if entry.get("type") == "speaker_correction":
                    corrections[entry["segment_id"]] = entry["speaker_name"]
                else:
                    events.append(entry)

    # Deduplicate by segment_id — keep best version per segment
    # Priority: version with translation > highest revision > first seen
    best: dict[str, dict] = {}
    for event in events:
        sid = event.get("segment_id")
        if not sid or not event.get("is_final") or not event.get("text"):
            continue

        has_tr = bool((event.get("translation") or {}).get("text"))
        existing = best.get(sid)

        if not existing:
            best[sid] = event
        elif has_tr and not (existing.get("translation") or {}).get("text"):
            # This version has translation, existing doesn't — always prefer
            best[sid] = event
        elif not has_tr and (existing.get("translation") or {}).get("text"):
            # Existing has translation, this doesn't — keep existing
            pass
        elif event.get("revision", 0) > existing.get("revision", 0):
            # Same translation state, higher revision wins
            # Preserve translation from existing if new one lacks it
            if (existing.get("translation") or {}).get("text") and not has_tr:
                event["translation"] = existing["translation"]
            best[sid] = event

    # Apply corrections to matching segments
    for sid, event in best.items():
        if sid in corrections:
            new_name = corrections[sid]
            speakers = event.get("speakers", [])
            if speakers:
                speakers[0]["identity"] = new_name
                speakers[0]["display_name"] = new_name
            else:
                event["speakers"] = [
                    {"identity": new_name, "display_name": new_name, "cluster_id": 0}
                ]

    # Sort by start_ms for consistent display order
    deduped = sorted(best.values(), key=lambda e: e.get("start_ms", 0))

    result["events"] = deduped
    result["total_events"] = len(deduped)

    return JSONResponse(result)


# ── Room Setup & Speaker Enrollment ────────────────────────────


@app.get("/api/room/layout")
async def get_room_layout(request: fastapi.Request) -> JSONResponse:
    """Get the current draft room layout for this session."""
    sid = _get_session_id(request)
    layout = _get_draft_layout(sid)
    resp = JSONResponse(layout.model_dump())
    if "scribe_session" not in request.cookies:
        resp.set_cookie("scribe_session", sid, path="/", samesite="lax", httponly=False)
    return resp


@app.put("/api/room/layout")
async def put_room_layout(request: fastapi.Request) -> JSONResponse:
    """Update the draft room layout for this session."""
    sid = _get_session_id(request)
    try:
        body = await request.json()
        _set_draft_layout(sid, RoomLayout.model_validate(body))
        resp = JSONResponse({"status": "ok"})
        if "scribe_session" not in request.cookies:
            resp.set_cookie("scribe_session", sid, path="/", samesite="lax", httponly=False)
        return resp
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=422)


@app.get("/api/room/presets")
async def get_presets() -> JSONResponse:
    """List available room layout presets."""
    return JSONResponse(
        {
            "presets": [
                {
                    "id": "boardroom",
                    "label": "Boardroom",
                    "description": "Oval table, seats around",
                },
                {
                    "id": "round",
                    "label": "Round Table",
                    "description": "Circular table, seats around",
                },
                {
                    "id": "square",
                    "label": "Square Table",
                    "description": "Square table, seats around",
                },
                {
                    "id": "rectangle",
                    "label": "Rectangle",
                    "description": "Rectangular table, seats around",
                },
                {
                    "id": "classroom",
                    "label": "Classroom",
                    "description": "Front desk, seats in rows",
                },
                {"id": "u_shape", "label": "U-Shape", "description": "U-shaped table arrangement"},
                {"id": "pods", "label": "Pods", "description": "Small group tables"},
                {
                    "id": "freeform",
                    "label": "Free-form",
                    "description": "No table, place seats freely",
                },
            ],
        }
    )


@app.get("/api/room/speakers")
async def get_speakers() -> JSONResponse:
    """Get all enrolled speakers."""
    speakers = enrollment_store.speakers
    return JSONResponse(
        {
            "speakers": [
                {
                    "enrollment_id": s.enrollment_id,
                    "name": s.name,
                    "audio_duration_seconds": s.audio_duration_seconds,
                }
                for s in speakers.values()
            ],
        }
    )


@app.post("/api/room/enroll")
async def enroll_speaker(request: fastapi.Request) -> JSONResponse:
    """Enroll a speaker from audio.

    Accepts raw s16le 16kHz PCM audio in body.
    Transcribes the audio to extract the speaker's name,
    then extracts a voice embedding for later identification.
    Optional ?name= query param overrides ASR-detected name.
    """
    import uuid

    import numpy as np

    body = await request.body()
    if len(body) < 16000:
        return JSONResponse({"error": "Need at least 0.5s of audio"}, status_code=400)

    audio = np.frombuffer(body, dtype=np.int16).astype(np.float32) / 32768.0
    duration = len(audio) / 16000

    # Transcribe audio to extract name
    name = request.query_params.get("name", "").strip()
    if not name:
        try:
            name = await _transcribe_name(audio)
        except Exception as e:
            logger.warning("Name transcription failed: %s", e)
            name = f"Speaker {len(enrollment_store.speakers) + 1}"

    if not name or len(name) < 1:
        name = f"Speaker {len(enrollment_store.speakers) + 1}"

    # Extract voice embedding
    try:
        embedding = await _extract_embedding(audio)
    except Exception as e:
        logger.warning("Embedding extraction failed: %s", e)
        embedding = _simple_embedding(audio)

    enrollment_id = str(uuid.uuid4())

    # Stash the raw enrollment WAV — reused as the TTS voice-clone seed
    # at meeting start so participant-voice mode is ready from segment 0.
    ref_path = ""
    try:
        import wave

        enroll_dir = Path.home() / ".config" / "meeting-scribe" / "enrollments"
        enroll_dir.mkdir(parents=True, exist_ok=True)
        wav_path = enroll_dir / f"{enrollment_id}.wav"
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm.tobytes())
        ref_path = str(wav_path)
    except Exception as e:
        logger.warning("Could not persist enrollment WAV: %s", e)

    speaker = EnrolledSpeaker(
        name=name,
        enrollment_id=enrollment_id,
        embedding=embedding,
        audio_duration_seconds=duration,
        reference_wav_path=ref_path,
    )
    enrollment_store.add(speaker)

    logger.info("Enrolled '%s' (id=%s, %.1fs audio)", name, enrollment_id, duration)
    return JSONResponse(
        {
            "enrollment_id": enrollment_id,
            "name": name,
            "audio_duration_seconds": round(duration, 1),
        }
    )


# Single-flight guard for the name-detect probe. The streaming
# enrollment flow polls every ~1.5s, but multiple clients/tabs (and the
# occasional front-end retry storm) will pile concurrent ASR calls onto
# the same vLLM backend. We saw the ASR worker die after 14+ probes in
# 7 seconds — the reprocess pipeline + the name-detect storm together
# exceeded the backend's working-set budget.
#
# A semaphore serializes the probes: at most one ASR call in flight at a
# time. If a probe arrives while another is running, we return immediately
# with a "busy" reason instead of queuing. The client polls again in
# ~1.5s anyway, so silent skipping is cheaper than queue-depth blowup.
_NAME_DETECT_SEMAPHORE = asyncio.Semaphore(1)


@app.post("/api/room/enroll/detect-name")
async def enroll_detect_name(request: fastapi.Request) -> JSONResponse:
    """Probe enrollment audio for a self-stated name.

    Used by the streaming enrollment flow: the client records continuously
    and POSTs the accumulated buffer every ~1.5s. The server runs ASR on
    the buffer and only reports a name when a self-intro pattern actually
    fires. The client stops recording as soon as it receives a name, then
    calls /api/room/enroll?name=<name> with the final audio to extract the
    voice embedding.

    Single-flight: at most one ASR probe runs at a time. Concurrent probes
    return ``{"reason": "busy"}`` immediately so a polling storm can't
    overwhelm the vLLM ASR worker (which previously died mid-inference
    when the storm coincided with reprocess-driven load).
    """
    import numpy as np

    if _NAME_DETECT_SEMAPHORE.locked():
        return JSONResponse({"name": "", "text": "", "reason": "busy"})

    body = await request.body()
    # Need roughly ~0.5s of audio before bothering with ASR — shorter clips
    # can work for short names but rarely produce usable transcriptions.
    # The name extraction patterns are conservative enough to reject garbage.
    if len(body) < 16000:
        return JSONResponse({"name": "", "text": "", "reason": "too_short"})

    audio = np.frombuffer(body, dtype=np.int16).astype(np.float32) / 32768.0
    async with _NAME_DETECT_SEMAPHORE:
        try:
            text = await _asr_transcribe(audio)
        except Exception as e:
            logger.warning("Name detection probe failed: %s", e)
            return JSONResponse({"name": "", "text": "", "reason": "asr_error"})

    # _extract_name_from_text (defined further down) returns None unless a
    # high-confidence self-intro pattern matches, which is exactly what we
    # want for probe-based streaming detection.
    name = _extract_name_from_text(text) if text else None
    if not name:
        return JSONResponse({"name": "", "text": text or "", "reason": "no_pattern_match"})

    return JSONResponse(
        {
            "name": name,
            "text": text,
            "duration_seconds": round(len(audio) / 16000, 2),
        }
    )


@app.post("/api/room/enroll/rename")
async def rename_speaker(request: fastapi.Request) -> JSONResponse:
    """Rename an enrolled speaker."""
    eid = request.query_params.get("id", "")
    name = request.query_params.get("name", "").strip()
    if not eid or not name:
        return JSONResponse({"error": "id and name required"}, status_code=400)

    speakers = enrollment_store.speakers
    if eid in speakers:
        speaker = speakers[eid]
        speaker.name = name
        enrollment_store.add(speaker)  # re-add to persist
        return JSONResponse({"status": "renamed", "name": name})
    return JSONResponse({"error": "not found"}, status_code=404)


@app.delete("/api/room/speakers/{enrollment_id}")
async def remove_speaker(enrollment_id: str) -> JSONResponse:
    """Remove an enrolled speaker."""
    if enrollment_store.remove(enrollment_id):
        return JSONResponse({"status": "removed"})
    return JSONResponse({"error": "not found"}, status_code=404)


async def _asr_transcribe(audio: np.ndarray) -> str:
    """Run ASR on a PCM audio buffer and return plain text via the
    already-running vLLM ASR endpoint."""
    import base64
    import io

    import httpx
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, 16000, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode()

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{config.asr_vllm_url}/v1/chat/completions",
            json={
                "model": config.asr_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Transcribe the audio in the original spoken language. Do not translate. Preserve the original script (e.g. Japanese in kanji/hiragana, Korean in hangul).",
                    },
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
                "temperature": 0.0,
                "max_tokens": 64,
            },
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]

    if "<asr_text>" in raw:
        _, _, text = raw.partition("<asr_text>")
        return text.strip()
    return raw.strip()


async def _transcribe_name(audio: np.ndarray) -> str:
    """Transcribe enrollment audio to extract the speaker's name.

    Used by the legacy fixed-duration flow. Runs ASR, then delegates name
    extraction to _extract_name_from_text (defined further below), falling
    back to a trimmed version of the raw transcription when no pattern
    matches so the old flow still produces *some* name.
    """
    text = await _asr_transcribe(audio)
    logger.info("Enrollment transcription: '%s'", text)
    if not text:
        return ""
    name = _extract_name_from_text(text)
    if name:
        return name
    # Fallback: use the raw transcription (trimmed) so the legacy flow
    # keeps producing something rather than dropping silently.
    lower = text.lower()
    for filler in ("hello", "hi", "hey", "um", "uh", "so", "okay", "well"):
        if lower.startswith(filler):
            text = text[len(filler) :].strip().lstrip(",").lstrip(".").strip()
            break
    return text[:20].strip().rstrip(".")


async def _extract_embedding(audio: np.ndarray) -> np.ndarray:
    """Extract speaker embedding using SpeechBrain ECAPA-TDNN."""

    try:
        import torch
        from speechbrain.inference.speaker import (
            EncoderClassifier,  # type: ignore[import-not-found]
        )

        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/speechbrain-ecapa",
        )
        waveform = torch.from_numpy(audio).unsqueeze(0)
        embedding = classifier.encode_batch(waveform).squeeze().numpy()
        return embedding
    except ImportError:
        logger.warning("SpeechBrain not installed — using simple embedding")
        return _simple_embedding(audio)


def _simple_embedding(audio: np.ndarray) -> np.ndarray:
    """Fallback: extract simple audio features as a pseudo-embedding.

    Uses MFCCs as a basic voice fingerprint. Not as good as ECAPA-TDNN
    but works without SpeechBrain.
    """
    import numpy as np

    # Compute basic spectral features
    n_fft = 512
    # Simple energy + zero-crossing based features
    frame_size = 16000  # 1s frames
    n_frames = max(1, len(audio) // frame_size)
    features = []
    for i in range(n_frames):
        frame = audio[i * frame_size : (i + 1) * frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        rms = float(np.sqrt(np.mean(frame**2)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(frame)))))
        # Simple FFT features (16 bins)
        fft = np.abs(np.fft.rfft(frame, n=n_fft))[: n_fft // 2]
        spectral = np.array([np.mean(fft[j::16]) for j in range(16)])
        features.append(np.concatenate([[rms, zcr], spectral]))

    embedding = np.mean(features, axis=0).astype(np.float32)
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


# ── Meeting Lifecycle ──────────────────────────────────────────


@app.post("/api/meeting/start")
async def start_meeting(request: fastapi.Request) -> JSONResponse:
    """Start a new meeting, or return the current one if already recording.

    GATED ON DEEP BACKEND HEALTH: refuses to start if ASR, Translation, or
    Diarization can't actually serve a real request. This prevents the
    "I started the meeting but nothing was translated" failure mode
    where status lied about backend readiness.

    SERIALIZED: all concurrent start/stop calls go through a single
    asyncio.Lock. Without this, a UI double-click or browser retry
    races through the "create new meeting + open audio writer" path
    and leaves one meeting orphaned with a dangling writer.
    """
    async with _get_meeting_lifecycle_lock():
        return await _start_meeting_locked(request)


async def _start_meeting_locked(request: fastapi.Request) -> JSONResponse:
    global current_meeting

    # Idempotent fast path: if we're already recording, the caller's
    # intent was "make sure a meeting is running". Return the existing
    # one so duplicate clicks are harmless.
    if current_meeting and current_meeting.state == MeetingState.RECORDING:
        logger.info(
            "start_meeting called while already recording — returning existing %s",
            current_meeting.meeting_id,
        )
        return JSONResponse(
            {
                "meeting_id": current_meeting.meeting_id,
                "state": "recording",
                "resumed": True,
                "language_pair": current_meeting.language_pair,
            }
        )

    # ── DEEP HEALTH GATE ──────────────────────────────────────────
    # Force a fresh check (bypass cache) — don't use a stale reading to
    # decide whether to start a meeting. We now require ALL primary
    # meeting backends (ASR + Translation + Diarization) so users can't
    # start a meeting that will silently lose speaker attribution or
    # freeze on an unready backend. TTS stays optional — meetings
    # still run without interpretation audio.
    health = await _deep_backend_health(force=True)
    REQUIRED = ["asr", "translate", "diarize", "furigana"]
    not_ready = [
        (name, health.get(name, {}).get("detail") or "not ready")
        for name in REQUIRED
        if not health.get(name, {}).get("ready")
    ]
    # Furigana is optional but cheap — fail start only if it CRASHED.
    # An uninitialized furigana_backend is fine (JA gets no ruby, meeting
    # still works end-to-end).
    if not_ready:
        logger.warning(
            "Refusing to start meeting — required backends not ready: %s",
            not_ready,
        )
        return JSONResponse(
            {
                "error": "Backends not ready",
                "not_ready": [{"backend": name, "detail": detail} for name, detail in not_ready],
                "message": (
                    "Required backends ("
                    + ", ".join(n.upper() for n in REQUIRED)
                    + ") are not ready yet. Wait for them to finish loading, then retry."
                ),
            },
            status_code=503,
        )

    # Stop any previous meeting
    if current_meeting and current_meeting.state in (MeetingState.CREATED, MeetingState.RECORDING):
        with suppress(Exception):
            storage.transition_state(current_meeting.meeting_id, MeetingState.INTERRUPTED)

    # Clear slide state from previous meeting so new meeting starts fresh
    if slide_job_runner is not None:
        slide_job_runner.active_deck_id = None
        slide_job_runner.current_slide_index = 0
        slide_job_runner._active_meta = None

    # Close any previous audio writer
    global audio_writer, meeting_start_time
    if audio_writer:
        audio_writer.close()
        audio_writer = None

    # Parse optional language_pair from request body. Accepts 1 or 2 distinct
    # codes — length 1 is a monolingual meeting (no translation work). Invalid
    # values are rejected with 400 rather than silently falling back to the
    # default, so a typo can never masquerade as a valid meeting (this is what
    # the original Deutsch/Dutch UI bug looked like).
    from meeting_scribe.languages import (
        DEFAULT_LANGUAGE_PAIR,
        is_valid_languages,
        parse_languages_strict,
    )

    language_pair = list(DEFAULT_LANGUAGE_PAIR)
    try:
        body = await request.json()
    except Exception:
        body = None
    if isinstance(body, dict) and "language_pair" in body:
        lp = body["language_pair"]
        parts: list[str] | None = None
        if isinstance(lp, str):
            parts = parse_languages_strict(lp)
        elif isinstance(lp, list):
            stripped = [str(p).strip() for p in lp]
            parts = stripped if is_valid_languages(stripped) else None
        if parts is None:
            return JSONResponse(
                {
                    "error": "Invalid language_pair",
                    "message": (
                        "language_pair must be 1 or 2 distinct codes from the "
                        "language registry (see GET /api/languages)."
                    ),
                    "received": lp,
                },
                status_code=400,
            )
        language_pair = parts

    # Create new meeting and persist room setup
    meta = storage.create_meeting(MeetingMeta(language_pair=language_pair))

    # Fresh start: clear in-memory voice enrollments AND any seat names /
    # enrollment_ids carried over from a previous meeting. The room layout
    # itself (seat positions, table shape) is preserved so the user keeps
    # the physical room they set up, but every chair starts unenrolled.
    # The user explicitly asked for this so a new meeting never inherits
    # voices or names.
    enrollment_store.clear()
    active_layout = _get_draft_layout(_get_session_id(request))
    for seat in active_layout.seats:
        seat.enrollment_id = None
        seat.speaker_name = ""
    _set_draft_layout(_get_session_id(request), active_layout)

    storage.save_room_layout(meta.meeting_id, active_layout)
    meeting_dir = storage._meeting_dir(meta.meeting_id)
    speakers_path = meeting_dir / "speakers.json"
    enrollment_store._storage_path = speakers_path
    enrollment_store._persist()  # writes empty speakers.json for this meeting
    logger.info(
        "Persisted room layout (%d seats, all reset) + empty speakers to %s",
        len(active_layout.seats),
        meeting_dir,
    )

    storage.transition_state(meta.meeting_id, MeetingState.RECORDING)
    current_meeting = meta
    current_meeting.state = MeetingState.RECORDING
    metrics.reset()
    metrics.meeting_start = time.monotonic()

    # Reset ASR backend watchdog state for the new meeting
    if asr_backend is not None and hasattr(asr_backend, "_last_response_time"):
        asr_backend._last_response_time = None
        asr_backend._buffer = []
        asr_backend._buffer_samples = 0
        asr_backend._segment_id = None
        asr_backend._base_offset = 0
    # Set the wall-clock origin for utterance_end_at. The ASR backend will
    # compute event.utterance_end_at = audio_wall_at_start + end_ms/1000
    # and that is the authoritative TTS speech-end SLA origin. See [P1-1-i1].
    if asr_backend is not None:
        asr_backend.audio_wall_at_start = metrics.meeting_start

    # Reset diarization global cluster state so each meeting starts fresh
    if diarize_backend is not None and hasattr(diarize_backend, "_global_centroids"):
        diarize_backend._global_centroids.clear()
        diarize_backend._global_centroid_counts.clear()
        diarize_backend._next_global_id = 1
        diarize_backend._last_mapping.clear()
        diarize_backend._result_cache.clear()
        # Rolling-window state (replaces the old fixed buffer)
        diarize_backend._rolling_audio.clear()
        diarize_backend._rolling_samples = 0
        diarize_backend._rolling_start_sample = 0
        diarize_backend._samples_since_flush = 0
        diarize_backend._last_emitted_end_ms = 0

    # Reset time-proximity pseudo-speaker state so each meeting starts fresh
    global _last_pseudo_cluster_id, _last_pseudo_end_ms, _next_pseudo_cluster_id
    _last_pseudo_cluster_id = 0
    _last_pseudo_end_ms = 0
    _next_pseudo_cluster_id = 100

    # Reset TTS voice cache so last meeting's speakers don't leak in.
    # pyannote reuses small cluster ids across meetings (0, 1, 2 …); without
    # this, a new meeting's first speaker would get the previous meeting's
    # cloned voice the moment their cluster_id happens to collide.
    if tts_backend is not None and hasattr(tts_backend, "reset_voice_cache"):
        tts_backend.reset_voice_cache()

    # Seed TTS voice cache from enrollment WAVs — participant-voice mode
    # otherwise has to wait for the first few seconds of each speaker's
    # audio before it can clone them. The seeded reference can still be
    # upgraded later if live audio scores higher.
    if tts_backend is not None and hasattr(tts_backend, "seed_voice"):
        _seed_tts_from_enrollments()

    # Open audio writer for recording
    audio_writer = storage.open_audio_writer(meta.meeting_id)
    meeting_start_time = time.monotonic()

    # Initialize speaker tracking
    global detected_speakers, speaker_verifier
    detected_speakers = []
    from meeting_scribe.speaker.verification import SpeakerVerifier

    speaker_verifier = SpeakerVerifier(enrollment_store)

    logger.info("Audio recording started for %s", meta.meeting_id)
    _persist_active_meeting(meta.meeting_id)

    # Start WiFi AP (background — don't block meeting start).
    # Passing meeting_id enables dedup: if the task runs multiple times for
    # the same meeting, only the first rotates credentials. All UI views
    # therefore see the same SSID throughout a meeting.
    asyncio.create_task(_start_wifi_ap(meeting_id=meta.meeting_id))

    # Start speaker pulse heartbeat
    global _speaker_pulse_task, _speaker_catchup_task, _eager_summary_task
    global _eager_summary_cache, _eager_summary_event_count
    if _speaker_pulse_task is None or _speaker_pulse_task.done():
        _speaker_pulse_task = asyncio.create_task(_speaker_pulse_loop())
    # Start retroactive speaker attribution catch-up
    if _speaker_catchup_task is None or _speaker_catchup_task.done():
        _speaker_catchup_task = asyncio.create_task(_speaker_catchup_loop())
    # Start eager summary pre-computation (low-priority draft during recording)
    _eager_summary_cache = None
    _eager_summary_event_count = 0
    if _eager_summary_task and not _eager_summary_task.done():
        _eager_summary_task.cancel()
    _eager_summary_task = asyncio.create_task(_eager_summary_loop(meta.meeting_id))

    # Push the meeting's languages into the live translation queue + ASR
    # language filter. Both were initialized at server startup with the
    # process-wide default, so without this step a meeting started with
    # e.g. it/en would still route all translations to ja. For a
    # monolingual meeting this also prevents any translation work from
    # being enqueued.
    meeting_languages = list(meta.language_pair)
    if translation_queue is not None:
        translation_queue.set_languages(meeting_languages)
        # Bind carries the live-context-window knobs (B1 window size +
        # B2 fragment gate) so every meeting honours the currently-
        # configured behaviour without the queue needing a reference
        # to ServerConfig.
        translation_queue.bind_meeting(
            meta.meeting_id,
            history_maxlen=config.live_translate_context_window_ja_en,
            fragment_gated=config.live_translate_context_fragment_gated,
        )
    if asr_backend is not None:
        # set_languages invalidates the ASR system prompt cache AND
        # rebuilds it with the meeting's language names. Without this,
        # Qwen3-ASR was misclassifying Dutch as English.
        asr_backend.set_languages(meeting_languages)

    # Near-realtime refinement worker (gated OFF by default via
    # config.refinement_enabled).  The worker trails 45 s behind the
    # live recording, re-transcribes with higher quality, and writes
    # polished.json at meeting end.  Only supports bilingual meetings;
    # 3+ language pairs are skipped (the refinement worker's prior-
    # context collector is same-direction only and can't anchor a
    # multi-direction history).  Backend URL respects
    # translate_offline_vllm_url by default so production deployments
    # that have already separated the offline backend keep their
    # semantics; the validation harness sets
    # SCRIBE_REFINEMENT_FORCE_SHARED_BACKEND=1 to force the live
    # translate backend and measure shared-GPU contention.
    global refinement_worker
    if config.refinement_enabled:
        if len(meta.language_pair) != 2:
            logger.info(
                "Refinement worker skipped for meeting %s: %d languages, bilingual only",
                meta.meeting_id,
                len(meta.language_pair),
            )
        else:
            try:
                from meeting_scribe.refinement import RefinementWorker

                force_shared = os.environ.get("SCRIBE_REFINEMENT_FORCE_SHARED_BACKEND", "0") == "1"
                refinement_translate_url = (
                    config.translate_vllm_url
                    if force_shared
                    else (config.translate_offline_vllm_url or config.translate_vllm_url)
                )
                logger.info(
                    "Refinement worker: asr=%s translate=%s (force_shared=%s)",
                    config.asr_vllm_url,
                    refinement_translate_url,
                    force_shared,
                )
                refinement_worker = RefinementWorker(
                    meeting_id=meta.meeting_id,
                    meeting_dir=storage._meeting_dir(meta.meeting_id),
                    asr_url=config.asr_vllm_url,
                    translate_url=refinement_translate_url,
                    language_pair=(meta.language_pair[0], meta.language_pair[1]),
                    context_window_segments=config.refinement_context_window_segments,
                )
                await refinement_worker.start()
            except Exception:
                logger.exception("Refinement worker failed to start")
                refinement_worker = None

    logger.info("Meeting started: %s", meta.meeting_id)

    # Warm the translate model so the first real utterance doesn't pay
    # the compile/cold-path tax. Measured at 855ms vs ~400ms steady-state
    # on the current 35B setup. One tiny throwaway call + a best-effort
    # exception swallow — no user visibility if it fails.
    #
    # Phase B4: when the live context-window is on, pass a short
    # representative prior_context so vLLM's prefix cache pre-compiles
    # the "Earlier in this meeting" scaffold shape.  Zero user cost —
    # the warmup result is discarded either way — but saves the first
    # real JA→EN utterance from paying the prompt-expand cost on top
    # of the cold-path cost.
    if translate_backend is not None and not meta.is_monolingual:
        _warmup_prior_context: list[tuple[str, str]] | None = None
        if (
            config.live_translate_context_window_ja_en > 0
            and tuple(meta.language_pair[:2]) == ("ja", "en")
        ):
            _warmup_prior_context = [("warmup ja", "warmup en")]

        async def _warm_translate():
            try:
                await translate_backend.translate(
                    "warmup",
                    meta.language_pair[0],
                    meta.language_pair[1],
                    prior_context=_warmup_prior_context,
                )
            except Exception:
                pass

        asyncio.create_task(_warm_translate())

    return JSONResponse(
        {
            "meeting_id": meta.meeting_id,
            "state": "recording",
            "resumed": False,
            "language_pair": meta.language_pair,
        }
    )


@app.post("/api/meeting/dev-reset")
async def dev_reset_meeting() -> JSONResponse:
    """Reset the current meeting without stopping it (DEV mode only).

    Flushes and discards accumulated ASR/translation/speaker state so the
    developer can iterate on code changes without the overhead of a full
    stop → start cycle.  The meeting stays in RECORDING state,
    websockets stay open, and all backends remain connected.

    Only available when dev_mode is enabled (via settings or env var).
    """
    if not _is_dev_mode():
        return JSONResponse(
            {"error": "dev-reset is only available in DEV mode"},
            status_code=403,
        )

    global audio_writer, meeting_start_time, refinement_worker

    if not current_meeting:
        return JSONResponse({"error": "No active meeting"}, status_code=400)

    mid = current_meeting.meeting_id
    logger.info("DEV RESET: clearing state for meeting %s", mid)

    # 1. Flush ASR — finalize any in-flight segments then discard
    if asr_backend:
        async for _event in asr_backend.flush():
            pass  # discard events

    # 2. Drain translation queue — B1 bind-to-None-first ordering.
    # Unbind first (increments epoch, quarantines in-flight work);
    # flush waits for workers to settle; clear drops the old history
    # dict; rebind creates a fresh history under the same meeting_id
    # with an even newer epoch.  The resulting two epoch bumps plus
    # the history-dict pop close the same-meeting-id leak that
    # meeting-id equality alone cannot.
    if translation_queue:
        translation_queue.bind_meeting(None)
        await translation_queue.flush_merge_gate()
        for _i in range(50):  # up to 5s
            if translation_queue.is_idle():
                break
            await asyncio.sleep(0.1)
        translation_queue.clear_meeting(mid)
        translation_queue.bind_meeting(
            mid,
            history_maxlen=config.live_translate_context_window_ja_en,
            fragment_gated=config.live_translate_context_fragment_gated,
        )

    # 2b. Tear down refinement worker if it is running.  Gated on the
    # worker handle, not on config.refinement_enabled — if the flag
    # was toggled off mid-meeting the already-running worker must
    # still be torn down.  Drain runs async off-lock so the dev-reset
    # response returns promptly.
    global _drain_seq
    if refinement_worker is not None:
        _resetting_worker = refinement_worker
        refinement_worker = None
        _drain_seq += 1
        entry = _DrainEntry(
            drain_id=_drain_seq,
            meeting_id=mid,
            task=asyncio.create_task(
                _drain_refinement(_resetting_worker, mid, _drain_seq),
                name=f"refinement-drain-{mid}-{_drain_seq}",
            ),
            state="draining",
            started_at=time.time(),
            translate_calls=_resetting_worker.translate_call_count,
            asr_calls=_resetting_worker.asr_call_count,
            errors_at_stop=_resetting_worker.last_error_count,
        )
        _refinement_drains.append(entry)
        _evict_completed_drains()

    # 3. Flush journal to disk (preserves history), then truncate it
    storage.flush_journal(mid)
    meeting_dir = storage._meeting_dir(mid)
    journal_path = meeting_dir / "journal.jsonl"
    if journal_path.exists():
        journal_path.write_text("")
    logger.info("DEV RESET: journal cleared for %s", mid)

    # 4. Reset ASR backend state for fresh segments
    if asr_backend is not None:
        if hasattr(asr_backend, "_last_response_time"):
            asr_backend._last_response_time = None
            asr_backend._buffer = []
            asr_backend._buffer_samples = 0
            asr_backend._segment_id = None
            asr_backend._base_offset = 0
        asr_backend.audio_wall_at_start = time.monotonic()

    # 5. Reset diarization global cluster state
    if diarize_backend is not None and hasattr(diarize_backend, "_global_centroids"):
        diarize_backend._global_centroids.clear()
        diarize_backend._global_centroid_counts.clear()
        diarize_backend._next_global_id = 1
        diarize_backend._last_mapping.clear()
        diarize_backend._result_cache.clear()
        diarize_backend._rolling_audio.clear()
        diarize_backend._rolling_samples = 0
        diarize_backend._rolling_start_sample = 0
        diarize_backend._samples_since_flush = 0
        diarize_backend._last_emitted_end_ms = 0

    # 6. Reset pseudo-speaker state
    global _last_pseudo_cluster_id, _last_pseudo_end_ms, _next_pseudo_cluster_id
    _last_pseudo_cluster_id = 0
    _last_pseudo_end_ms = 0
    _next_pseudo_cluster_id = 100

    # 7. Reset TTS voice cache
    if tts_backend is not None and hasattr(tts_backend, "reset_voice_cache"):
        tts_backend.reset_voice_cache()

    # 8. Reset audio writer — close old, open fresh
    if audio_writer:
        audio_writer.close()
    audio_writer = storage.open_audio_writer(mid)
    meeting_start_time = time.monotonic()
    metrics.reset()
    metrics.meeting_start = meeting_start_time

    # 8b. Restart refinement worker if it was in use.  The prior worker's
    # drain (kicked off in step 2b) continues in the background under
    # its own drain_id; a new worker on the same meeting_id is started
    # fresh so the second half of the meeting continues to be polished.
    if config.refinement_enabled and len(current_meeting.language_pair) == 2:
        try:
            from meeting_scribe.refinement import RefinementWorker

            force_shared = os.environ.get("SCRIBE_REFINEMENT_FORCE_SHARED_BACKEND", "0") == "1"
            refinement_translate_url = (
                config.translate_vllm_url
                if force_shared
                else (config.translate_offline_vllm_url or config.translate_vllm_url)
            )
            refinement_worker = RefinementWorker(
                meeting_id=mid,
                meeting_dir=storage._meeting_dir(mid),
                asr_url=config.asr_vllm_url,
                translate_url=refinement_translate_url,
                language_pair=(
                    current_meeting.language_pair[0],
                    current_meeting.language_pair[1],
                ),
                context_window_segments=config.refinement_context_window_segments,
            )
            await refinement_worker.start()
        except Exception:
            logger.exception("Refinement worker restart after dev-reset failed")
            refinement_worker = None

    # 9. Reset speaker tracking
    global detected_speakers, speaker_verifier
    detected_speakers = []

    # 10. Broadcast reset to all connected clients so UI clears
    await _broadcast_json(
        {
            "type": "dev_reset",
            "meeting_id": mid,
            "message": "Meeting state reset — ready for next iteration",
        }
    )

    logger.info("DEV RESET complete for %s — meeting still RECORDING", mid)
    return JSONResponse(
        {
            "status": "reset",
            "meeting_id": mid,
            "message": "State cleared, meeting still recording. Start speaking to test.",
        }
    )


@app.post("/api/meetings/{meeting_id}/resume")
async def resume_meeting(meeting_id: str) -> JSONResponse:
    """Resume an interrupted meeting — continue recording into the same meeting."""
    global current_meeting, audio_writer, meeting_start_time

    try:
        return await _do_resume_meeting(meeting_id)
    except Exception as e:
        logger.exception("Resume failed for %s", meeting_id)
        return JSONResponse(
            {"error": f"Resume failed: {type(e).__name__}: {e}"},
            status_code=500,
        )


async def _do_resume_meeting(meeting_id: str) -> JSONResponse:
    """Inner resume implementation — wrapped for guaranteed JSON error responses."""
    global current_meeting, audio_writer, meeting_start_time

    if current_meeting and current_meeting.state == MeetingState.RECORDING:
        return JSONResponse({"error": "Another meeting is already recording"}, status_code=409)

    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir or not meeting_dir.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    import json as _json

    meta_path = meeting_dir / "meta.json"
    if not meta_path.exists():
        return JSONResponse({"error": "Meeting metadata not found"}, status_code=404)

    meta_data = _json.loads(meta_path.read_text())
    if meta_data.get("state") not in ("interrupted", "complete"):
        return JSONResponse(
            {"error": f"Cannot resume meeting in state: {meta_data.get('state')}"}, status_code=400
        )

    # Rebuild MeetingMeta. The Pydantic field validator on language_pair
    # is the authoritative shape check — a corrupt persisted list (empty,
    # >2 codes, duplicate, unknown code) raises and the resume is refused
    # rather than silently starting the meeting in a weird state.
    lp = meta_data.get("language_pair", ["en", "ja"])
    try:
        meta = MeetingMeta(
            meeting_id=meeting_id,
            language_pair=lp if isinstance(lp, list) else ["en", "ja"],
        )
    except Exception as e:
        return JSONResponse(
            {"error": f"Corrupt language_pair in persisted meta: {e}"},
            status_code=400,
        )
    meta.state = MeetingState.RECORDING

    # Transition state
    # Allow interrupted → recording
    meta_data["state"] = "recording"
    meta_path.write_text(_json.dumps(meta_data, indent=2))

    current_meeting = meta
    metrics.reset()
    metrics.meeting_start = time.monotonic()

    # Set the wall-clock origin for utterance_end_at so TTS deadlines work.
    # Without this, ASR events have utterance_end_at=None and TTS drops
    # every segment at the missing-origin gate.
    if asr_backend is not None:
        asr_backend.audio_wall_at_start = metrics.meeting_start

    # Open audio writer in APPEND mode (isolated process for crash resilience)
    from meeting_scribe.storage import AudioWriterProcess

    audio_dir = meeting_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    pcm_path = audio_dir / "recording.pcm"
    writer = AudioWriterProcess(pcm_path, append=True)
    writer.start()
    audio_writer = writer
    meeting_start_time = time.monotonic()

    # Restore speaker enrollment from meeting
    speakers_path = meeting_dir / "speakers.json"
    if speakers_path.exists():
        try:
            enrollment_store._storage_path = speakers_path
            enrollment_store.load()
        except Exception as e:
            # Don't fail the resume — log and continue with empty enrollment
            logger.warning("Failed to load speaker enrollment for resume: %s", e)

    global detected_speakers, speaker_verifier
    detected_speakers = []
    from meeting_scribe.speaker.verification import SpeakerVerifier

    speaker_verifier = SpeakerVerifier(enrollment_store)

    # Same fix as start_meeting — propagate the meeting's languages
    # into the translation queue + ASR filter on resume.
    meeting_languages = list(meta.language_pair)
    if translation_queue is not None:
        translation_queue.set_languages(meeting_languages)
    if asr_backend is not None:
        asr_backend.set_languages(meeting_languages)

    logger.info("Meeting resumed: %s (appending audio)", meeting_id)
    _persist_active_meeting(meeting_id)

    # Start WiFi AP. Resume reuses the original meeting_id, so if
    # _start_wifi_ap was already called during the initial start, this
    # call is a no-op (dedup via _LAST_ROTATED_MEETING_ID).
    asyncio.create_task(_start_wifi_ap(meeting_id=meeting_id))

    # Start speaker pulse + catch-up
    global _speaker_pulse_task, _speaker_catchup_task
    if _speaker_pulse_task is None or _speaker_pulse_task.done():
        _speaker_pulse_task = asyncio.create_task(_speaker_pulse_loop())
    if _speaker_catchup_task is None or _speaker_catchup_task.done():
        _speaker_catchup_task = asyncio.create_task(_speaker_catchup_loop())

    return JSONResponse(
        {
            "meeting_id": meeting_id,
            "state": "recording",
            "resumed": True,
            "language_pair": meta.language_pair,
        }
    )


@app.post("/api/meetings/{meeting_id}/finalize")
async def finalize_meeting(
    meeting_id: str,
    force: bool = False,
    expected_speakers: int | None = None,
) -> JSONResponse:
    """Finalize/reprocess a meeting — generate timeline, speakers, summary.

    Runs full-audio diarization on the recording so speaker separation
    works even when the live streaming path didn't produce results. The
    new diarization cluster_ids are written back to the journal as new
    revisions so subsequent reads dedup to the diarized version.

    Always regenerates speaker data and timeline. Regenerates summary if
    missing or force=True.

    ``expected_speakers``: optional hard pin on the speaker count when
    the user knows it. Pyannote tends to over-cluster on long noisy
    audio; pinning min=max=N constrains it tightly. Range 1-12.
    """
    import json as _json

    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir or not meeting_dir.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    meta_path = meeting_dir / "meta.json"
    if not meta_path.exists():
        return JSONResponse({"error": "Meeting metadata missing"}, status_code=404)
    meta = _json.loads(meta_path.read_text())

    journal_path = meeting_dir / "journal.jsonl"
    pcm_path = meeting_dir / "audio" / "recording.pcm"

    # ── Full-audio diarization pass ──────────────────────────
    # Walks the existing journal and attaches diarization cluster_ids via
    # time-range overlap. Works for meetings of any length (chunked with
    # embedding-based cluster merging — see reprocess._diarize_full_audio).
    diarize_info = {"ran": False, "segments": 0, "unique_speakers": 0}
    audio_quality = None
    if journal_path.exists() and pcm_path.exists():
        try:
            from meeting_scribe.reprocess import (
                _attach_speakers_to_events,
                _audio_quality_report,
                _diarize_full_audio,
            )

            # Load final events with translation preservation.
            # Priority: version with translation > highest revision > first.
            events_by_sid: dict[str, dict] = {}
            corrections: dict[str, str] = {}
            for line in journal_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    e = _json.loads(line)
                except Exception:
                    continue
                if e.get("type") == "speaker_correction":
                    corrections[e.get("segment_id", "")] = e.get("speaker_name", "")
                    continue
                if not e.get("is_final") or not e.get("text"):
                    continue
                sid = e.get("segment_id")
                if not sid:
                    continue
                existing = events_by_sid.get(sid)
                if not existing:
                    events_by_sid[sid] = e
                else:
                    e_has_tr = bool((e.get("translation") or {}).get("text"))
                    ex_has_tr = bool((existing.get("translation") or {}).get("text"))
                    e_rev = e.get("revision", 0)
                    ex_rev = existing.get("revision", 0)
                    if e_has_tr and not ex_has_tr:
                        events_by_sid[sid] = e
                    elif not e_has_tr and ex_has_tr:
                        if e_rev > ex_rev:
                            e["translation"] = existing["translation"]
                            events_by_sid[sid] = e
                    elif e_rev > ex_rev:
                        if ex_has_tr and not e_has_tr:
                            e["translation"] = existing["translation"]
                        events_by_sid[sid] = e

            if events_by_sid:
                logger.info(
                    "Finalize: running full-audio diarization on %d events (%.0fMB audio)",
                    len(events_by_sid),
                    pcm_path.stat().st_size / 1024 / 1024,
                )
                pcm_data = pcm_path.read_bytes()
                audio_quality = _audio_quality_report(pcm_data)
                if not audio_quality["usable"]:
                    logger.warning(
                        "Finalize: audio is %0.1f%% zero-filled "
                        "(longest gap %dms) — diarization quality will be poor. "
                        "This is a recording bug in the old audio writer. "
                        "New meetings recorded after the fix will be clean.",
                        audio_quality["zero_fill_pct"],
                        audio_quality["longest_zero_run_ms"],
                    )
                # When the caller pins expected_speakers, hint pyannote to
                # that count (min=max=N is a soft hint per pyannote#1405)
                # AND tell the cross-chunk merger to force-absorb extras
                # down to exactly N. Otherwise stay wide.
                if expected_speakers is not None and 1 <= expected_speakers <= 12:
                    diar_min = expected_speakers
                    diar_max = expected_speakers
                    expected_for_merge = expected_speakers
                    logger.info(
                        "Finalize: pinning diarization to %d speakers (caller hint)",
                        expected_speakers,
                    )
                else:
                    diar_min = 2
                    diar_max = 8
                    expected_for_merge = None

                diarize_segments = await _diarize_full_audio(
                    pcm_data,
                    config.diarize_url,
                    max_speakers=diar_max,
                    min_speakers=diar_min,
                    expected_speakers=expected_for_merge,
                )

                if diarize_segments:
                    # Attach cluster_ids in place
                    events_list = list(events_by_sid.values())
                    _attach_speakers_to_events(events_list, diarize_segments)

                    # Append new revisions to the journal so readers pick up
                    # the diarized versions (highest revision wins on read).
                    with journal_path.open("a") as f:
                        for event in events_list:
                            if not event.get("speakers"):
                                continue  # skip events the diarizer couldn't attribute
                            new_revision = (event.get("revision", 0) or 0) + 1
                            event["revision"] = new_revision
                            f.write(_json.dumps(event, ensure_ascii=False) + "\n")

                    unique = len({s["cluster_id"] for s in diarize_segments})
                    diarize_info = {
                        "ran": True,
                        "segments": len(diarize_segments),
                        "unique_speakers": unique,
                    }
                    logger.info(
                        "Finalize diarization: %d segments, %d unique global speakers",
                        len(diarize_segments),
                        unique,
                    )
                else:
                    logger.warning("Finalize: diarization returned no segments")
        except Exception as e:
            logger.exception("Finalize diarization failed: %s", e)

    # Regenerate speaker data and timeline from the (now updated) journal
    if journal_path.exists():
        _generate_speaker_data(
            meeting_dir,
            journal_path,
            _json,
            expected_speakers=expected_speakers,
        )

        # Sanity check: did every cluster pyannote found make it into
        # detected_speakers? If diarize reported N unique speakers but
        # detected_speakers has fewer, one or more speakers got lost
        # in the ASR-gap scenario described in
        # `_attach_speakers_to_events` — the new diarize cluster's
        # time ranges don't coincide with any existing ASR event.
        # Only /reprocess can recover these. Warn loudly so the user
        # doesn't silently lose speakers.
        if diarize_info.get("ran") and diarize_info.get("unique_speakers", 0) > 0:
            try:
                _ds_raw = (meeting_dir / "detected_speakers.json").read_text()
                _ds = _json.loads(_ds_raw)
                diarize_count = diarize_info["unique_speakers"]
                detected_count = len(_ds)
                if detected_count < diarize_count:
                    lost = diarize_count - detected_count
                    logger.warning(
                        "finalize: diarize found %d unique speakers but "
                        "only %d made it into detected_speakers.json "
                        "(%d lost). Lost speakers live in ASR gaps — "
                        "run /api/meetings/%s/reprocess to recover them.",
                        diarize_count,
                        detected_count,
                        lost,
                        meeting_id,
                    )
                    diarize_info["lost_in_asr_gap"] = lost
            except Exception:
                pass

    timeline_path = meeting_dir / "timeline.json"
    _generate_timeline(meeting_id)

    # Generate summary (if missing or force reprocess, and journal has content)
    summary_path = meeting_dir / "summary.json"
    summary = {}
    if (not summary_path.exists() or force) and journal_path.exists():
        try:
            from meeting_scribe.summary import generate_summary

            summary = await generate_summary(meeting_dir, vllm_url=config.translate_vllm_url)
        except Exception as e:
            logger.warning("Finalize summary failed: %s", e)
            summary = {"error": str(e)}

    # Update state to complete
    meta["state"] = "complete"
    meta_path.write_text(_json.dumps(meta, indent=2))

    return JSONResponse(
        {
            "status": "finalized",
            "meeting_id": meeting_id,
            "has_timeline": timeline_path.exists(),
            "has_summary": "error" not in summary,
            "has_speakers": (meeting_dir / "detected_speakers.json").exists(),
            "diarization": diarize_info,
            "audio_quality": audio_quality,
        }
    )


@app.post("/api/meetings/{meeting_id}/reprocess")
async def reprocess_meeting_endpoint(
    meeting_id: str,
    expected_speakers: int | None = None,
) -> JSONResponse:
    """Full reprocess — re-run ASR + translation + diarization on raw audio.

    Backs up the existing journal, re-transcribes from recording.pcm,
    re-translates every segment, re-runs full-audio diarization, and
    regenerates all derived artifacts (journal, detected_speakers,
    speaker_lanes, timeline, summary). This is the "redo everything from
    the original audio for higher quality" path — distinct from
    /finalize which only re-runs diarization + summary on the existing
    transcript.

    ``expected_speakers``: optional pin on the speaker count when the
    user knows it. Constrains pyannote per-chunk and triggers the
    forced-collapse pass at both the diarize-merge and journal-rewrite
    layers — same pipeline as /finalize.
    """
    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir or not meeting_dir.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)

    pcm_path = meeting_dir / "audio" / "recording.pcm"
    if not pcm_path.exists():
        return JSONResponse({"error": "No audio recording found"}, status_code=404)

    # Get language pair from meeting metadata
    import json as _json

    meta_path = meeting_dir / "meta.json"
    language_pair = ("en", "ja")
    if meta_path.exists():
        meta = _json.loads(meta_path.read_text())
        lp = meta.get("language_pair", ["en", "ja"])
        language_pair = (lp[0], lp[1]) if isinstance(lp, list) and len(lp) == 2 else ("en", "ja")

    from meeting_scribe.reprocess import reprocess_meeting

    result = await reprocess_meeting(
        meeting_dir,
        asr_url=(config.omni_asr_url or config.asr_vllm_url),
        translate_url=(config.omni_translate_url or config.translate_vllm_url),
        diarize_url=config.diarize_url,
        language_pair=language_pair,
        expected_speakers=expected_speakers,
    )

    if "error" in result:
        return JSONResponse(result, status_code=400)

    return JSONResponse(
        {
            "status": "reprocessed",
            "meeting_id": meeting_id,
            **result,
        }
    )


@app.get("/api/meetings/{meeting_id}/versions")
async def list_meeting_versions(meeting_id: str) -> JSONResponse:
    """List snapshot versions for a meeting (newest first).

    Each reprocess auto-snapshots the prior journal/summary/timeline/
    speakers under ``meetings/{id}/versions/`` so two runs can be
    compared. Returns the manifest for each snapshot.
    """
    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir or not meeting_dir.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)
    from meeting_scribe.versions import list_versions

    return JSONResponse({"meeting_id": meeting_id, "versions": list_versions(meeting_dir)})


@app.get("/api/meetings/{meeting_id}/versions/diff")
async def diff_meeting_versions(
    meeting_id: str,
    baseline: str | None = None,
    compare: str | None = None,
) -> JSONResponse:
    """Diff two versions (or the latest snapshot vs current state).

    ``baseline`` and ``compare`` are version directory names (from
    ``GET /versions``). Either omitted: ``baseline`` defaults to the
    most recent snapshot, ``compare`` defaults to the current state.
    Returns per-dimension verdicts (better/worse/same) so the caller
    can grade whether a code/model change improved transcription quality.
    """
    meeting_dir = _safe_meeting_dir(meeting_id)
    if not meeting_dir or not meeting_dir.exists():
        return JSONResponse({"error": "Meeting not found"}, status_code=404)
    from meeting_scribe.versions import (
        diff_versions,
        list_versions,
        metrics_for_current,
        metrics_for_version,
    )

    snaps = list_versions(meeting_dir)
    if not snaps and baseline is None:
        return JSONResponse(
            {"error": "No snapshots yet — run reprocess first"},
            status_code=404,
        )

    if baseline is None:
        baseline = snaps[0]["name"]
    base_metrics = metrics_for_version(meeting_dir, baseline)

    if compare is None:
        cmp_metrics = metrics_for_current(meeting_dir)
        cmp_label = "(current)"
    else:
        cmp_metrics = metrics_for_version(meeting_dir, compare)
        cmp_label = compare

    diff = diff_versions(base_metrics, cmp_metrics)
    return JSONResponse(
        {
            "meeting_id": meeting_id,
            "baseline": baseline,
            "compare": cmp_label,
            "diff": diff,
        }
    )


@app.post("/api/meeting/cancel")
async def cancel_meeting() -> JSONResponse:
    """Cancel the current meeting — discard all artifacts without finalization."""
    async with _get_meeting_lifecycle_lock():
        return await _cancel_meeting_locked()


async def _cancel_meeting_locked() -> JSONResponse:
    """Cancel: stop recording, delete everything, no summary/diarization.

    Use cases: accidental starts, test meetings, private content that
    shouldn't be retained. The meeting directory is fully removed so
    the meeting never appears in the sidebar.
    """
    global current_meeting, audio_writer, meeting_start_time, refinement_worker
    global _speaker_pulse_task, _speaker_catchup_task
    global _eager_summary_task, _eager_summary_cache, _eager_summary_event_count

    if not current_meeting:
        return JSONResponse({"error": "No active meeting"}, status_code=400)

    mid = current_meeting.meeting_id
    logger.info("Cancelling meeting: %s", mid)

    # Close audio writer (no fsync needed — we're deleting everything)
    if audio_writer:
        audio_writer.close()
        audio_writer = None
        meeting_start_time = 0.0

    # Tear down the refinement worker if it was started for this meeting.
    # On cancel we skip the 60s drain — all artifacts are being deleted,
    # so any polished.json that was about to land would just be erased.
    # Best-effort stop with a short timeout; worker is then abandoned.
    if refinement_worker is not None:
        _cancelling_worker = refinement_worker
        refinement_worker = None
        with suppress(Exception):
            await asyncio.wait_for(_cancelling_worker.stop(), timeout=2.0)

    # Drop the translation-queue binding + history so the to-be-deleted
    # meeting_id does not leak into log rows or live-context history
    # for the next meeting.
    if translation_queue is not None:
        translation_queue.bind_meeting(None)
        translation_queue.clear_meeting(mid)

    # Cancel all background tasks immediately
    if _eager_summary_task and not _eager_summary_task.done():
        _eager_summary_task.cancel()
        _eager_summary_task = None
    _eager_summary_cache = None
    _eager_summary_event_count = 0

    if _speaker_pulse_task and not _speaker_pulse_task.done():
        _speaker_pulse_task.cancel()
        _speaker_pulse_task = None
    if _speaker_catchup_task and not _speaker_catchup_task.done():
        _speaker_catchup_task.cancel()
        _speaker_catchup_task = None
    _pending_speaker_events.clear()
    _pending_speaker_timestamps.clear()

    for task in list(_background_tasks):
        if not task.done():
            task.cancel()
    _background_tasks.clear()

    # Cancel all pending translations (don't wait for in-flight work)
    if translation_queue:
        for item in list(getattr(translation_queue, "_items", [])):
            if not getattr(item, "started", False):
                item.cancelled = True

    # Clean up slide artifacts
    if slide_job_runner is not None:
        slide_job_runner.cleanup_meeting(mid)

    # Delete the entire meeting directory
    import shutil as _cancel_shutil

    meeting_dir = storage._meeting_dir(mid)
    if meeting_dir.exists():
        _cancel_shutil.rmtree(meeting_dir)
        logger.info("Cancelled meeting %s — all artifacts deleted", mid)

    # Broadcast cancellation to all clients
    await _broadcast_json(
        {
            "type": "meeting_cancelled",
            "meeting_id": mid,
        }
    )

    # Close all websockets
    await asyncio.sleep(0.3)
    for ws in list(ws_connections):
        with suppress(Exception):
            await ws.close(1000, "Meeting cancelled")
    ws_connections.clear()
    _client_prefs.clear()
    for ws in list(_audio_out_clients):
        with suppress(Exception):
            await ws.close(1000, "Meeting cancelled")
    _audio_out_clients.clear()
    _audio_out_prefs.clear()

    current_meeting = None
    _clear_active_meeting()
    asyncio.create_task(_stop_wifi_ap())

    return JSONResponse({"status": "cancelled", "meeting_id": mid})


@app.post("/api/meeting/stop")
async def stop_meeting(preserve_empty: bool = False) -> JSONResponse:
    """Stop the current meeting (serialized via the lifecycle lock)."""
    async with _get_meeting_lifecycle_lock():
        return await _stop_meeting_locked(preserve_empty)


async def _stop_meeting_locked(preserve_empty: bool = False) -> JSONResponse:
    global current_meeting, audio_writer, meeting_start_time, refinement_worker
    global _speaker_pulse_task, _speaker_catchup_task
    global _eager_summary_task, _eager_summary_cache, _eager_summary_event_count

    if not current_meeting:
        return JSONResponse({"error": "No active meeting"}, status_code=400)

    mid = current_meeting.meeting_id
    # Env override so tests don't have to thread the query param through
    # every call. In production this env var is unset → False.
    if os.environ.get("SCRIBE_PRESERVE_EMPTY_ON_STOP", "0") == "1":
        preserve_empty = True
    audio_duration_s = audio_writer.duration_ms / 1000 if audio_writer else 0
    # Estimate finalization time based on audio duration
    est_seconds = max(10, int(audio_duration_s * 0.3))  # ~30% of audio duration

    # Broadcast finalization progress to all clients
    await _broadcast_json(
        {
            "type": "finalize_progress",
            "step": 1,
            "total_steps": 5,
            "label": "Flushing ASR...",
            "eta_seconds": est_seconds,
        }
    )

    # Flush ASR — finalizes all pending lines
    if asr_backend:
        async for event in asr_backend.flush():
            await _process_event(event)

    # Clear the active meeting EARLY so a new meeting can start
    # while finalization continues in the background
    current_meeting = None

    # Slides become part of the meeting record after finalize — keep the
    # on-disk artifacts, just drop the in-memory active-deck pointer so a
    # new meeting starts with a clean slide viewer.
    if slide_job_runner is not None:
        slide_job_runner.clear_active_state(mid)

    await _broadcast_json(
        {
            "type": "finalize_progress",
            "step": 2,
            "total_steps": 5,
            "label": "Completing translations...",
            "eta_seconds": max(5, est_seconds - 5),
        }
    )

    # Flush translation merge gate and wait for pending translations.
    # Flush-first-then-unbind ordering: items complete under the stopping
    # meeting's binding so tail translations deliver to the user before
    # the queue is unbound.  Stragglers past the 10s budget that land
    # after the unbind will be dropped by B1's post-await epoch check.
    # Phase B1 completes the ordering: flush first (items deliver under
    # the stopping meeting's epoch), then bind(None) increments the
    # epoch, then clear_meeting drops the history dict.
    if translation_queue:
        await translation_queue.flush_merge_gate()
        logger.info("Waiting for pending translations...")
        for _i in range(100):  # up to 10s
            if translation_queue.is_idle():
                break
            await asyncio.sleep(0.1)
        else:
            logger.warning("Translation queue still busy after 10s — proceeding with stop")
        translation_queue.bind_meeting(None)
        translation_queue.clear_meeting(mid)

    storage.flush_journal(mid)

    # Cancel eager summary so it doesn't compete with finalization on vLLM
    if _eager_summary_task and not _eager_summary_task.done():
        _eager_summary_task.cancel()
        _eager_summary_task = None

    # Close audio writer
    if audio_writer:
        duration_ms = audio_writer.duration_ms
        audio_writer.close()
        audio_writer = None
        meeting_start_time = 0.0
        logger.info("Audio recording: %dms (%.1fs)", duration_ms, duration_ms / 1000)

    # Kick off refinement drain OFF the lifecycle lock.  worker.stop()
    # flushes ``_process_remaining`` which can take tens of seconds on
    # a long meeting; running it on the locked stop path would block
    # the next /api/meeting/start by up to a minute.  The worker is
    # stopped only if it is actually running — gate is on the worker
    # handle, not config.refinement_enabled, so a flag toggled off
    # mid-meeting still tears down the already-running worker.
    global _drain_seq
    if refinement_worker is not None:
        _stopping_worker = refinement_worker
        refinement_worker = None
        _drain_seq += 1
        entry = _DrainEntry(
            drain_id=_drain_seq,
            meeting_id=mid,
            task=asyncio.create_task(
                _drain_refinement(_stopping_worker, mid, _drain_seq),
                name=f"refinement-drain-{mid}-{_drain_seq}",
            ),
            state="draining",
            started_at=time.time(),
            # Kickoff snapshot — worker is still alive and reachable here.
            # _drain_refinement overwrites these fields when stop() returns
            # so /api/admin/refinement-stats and /polished-status reflect
            # the counters captured after _process_remaining.
            translate_calls=_stopping_worker.translate_call_count,
            asr_calls=_stopping_worker.asr_call_count,
            errors_at_stop=_stopping_worker.last_error_count,
        )
        _refinement_drains.append(entry)
        _evict_completed_drains()

    # ── Empty-meeting cleanup ─────────────────────────────────────
    # If the user clicked Start then Stop without ever producing a final
    # transcript event, the meeting dir is just empty overhead. Delete it
    # here instead of going through the full finalize / summary pipeline
    # and then leaving a "0 events · complete" entry cluttering the sidebar.
    import json as _empty_json
    import shutil as _empty_shutil

    _meeting_dir = storage._meeting_dir(mid)
    _journal_path = _meeting_dir / "journal.jsonl"
    _has_final = False
    if _journal_path.exists():
        try:
            for _line in _journal_path.read_text().splitlines():
                if not _line.strip():
                    continue
                try:
                    _e = _empty_json.loads(_line)
                except Exception:
                    continue
                if _e.get("is_final") and (_e.get("text") or "").strip():
                    _has_final = True
                    break
        except Exception:
            _has_final = True  # on read error, be conservative and keep it
    if not _has_final and preserve_empty:
        # Integration-test path: keep the meeting directory so
        # /api/meetings/{id}/* endpoints return 200. Fall through to the
        # normal finalize path below, which will produce a tiny summary
        # and mark the meeting complete. The `preserve_empty` flag is set
        # by the stop endpoint's query param or SCRIBE_PRESERVE_EMPTY_ON_STOP.
        logger.info("Meeting %s stopped with zero events — preserving (preserve_empty=True)", mid)
    elif not _has_final:
        # SAFETY INVARIANT (added 2026-04-14): never delete a meeting
        # dir that still has real audio. See storage.py for the same
        # guard on startup cleanup. Audio is unrecoverable; journal/
        # timeline/speaker files can be regenerated via reprocess.
        _pcm = _meeting_dir / "audio" / "recording.pcm"
        _has_audio = _pcm.exists() and _pcm.stat().st_size > 0
        if _has_audio:
            logger.info(
                "Meeting %s stopped with zero journal events but has "
                "%d bytes of audio — preserving for reprocess",
                mid,
                _pcm.stat().st_size,
            )
        else:
            logger.info("Meeting %s stopped with zero events — deleting", mid)
            try:
                if _meeting_dir.exists():
                    _empty_shutil.rmtree(_meeting_dir)
            except Exception as e:
                logger.warning("Failed to delete empty meeting %s: %s", mid, e)
        # Broadcast a completion message so any finalization modals/UI
        # clean up, then short-circuit all downstream finalize work.
        await _broadcast_json(
            {
                "type": "finalize_progress",
                "step": 6,
                "label": "Empty meeting discarded",
                "meeting_id": mid,
                "summary": {"discarded": True},
            }
        )
        await asyncio.sleep(0.3)
        for ws in list(ws_connections):
            with suppress(Exception):
                await ws.close(1000, "Meeting ended")
        ws_connections.clear()
        _client_prefs.clear()
        for ws in list(_audio_out_clients):
            with suppress(Exception):
                await ws.close(1000, "Meeting ended")
        _audio_out_clients.clear()
        _audio_out_prefs.clear()
        if _speaker_pulse_task and not _speaker_pulse_task.done():
            _speaker_pulse_task.cancel()
            _speaker_pulse_task = None
        if _speaker_catchup_task and not _speaker_catchup_task.done():
            _speaker_catchup_task.cancel()
            _speaker_catchup_task = None
        _pending_speaker_events.clear()
        _pending_speaker_timestamps.clear()
        current_meeting = None
        for task in list(_background_tasks):
            if not task.done():
                task.cancel()
        _background_tasks.clear()
        _clear_active_meeting()
        asyncio.create_task(_stop_wifi_ap())
        return JSONResponse({"status": "discarded", "meeting_id": mid, "reason": "no_events"})

    # Save detected speakers (live-attribution state)
    await _broadcast_json(
        {
            "type": "finalize_progress",
            "step": 3,
            "total_steps": 6,
            "label": "Saving speaker data...",
            "eta_seconds": max(3, est_seconds - 10),
        }
    )
    if detected_speakers:
        storage.save_detected_speakers(mid, detected_speakers)
        logger.info("Saved %d detected speakers", len(detected_speakers))

    # ── Full-audio diarization pass ──────────────────────────
    # This is the SAME pipeline /api/meetings/{id}/finalize uses — run it
    # here so the stop flow produces the exact same final result as the
    # Re-finalize button. Without this, live-streaming attribution (which
    # sees only 6s chunks at a time) leaves events stuck on pseudo clusters.
    import json as _json

    meeting_dir = storage._meeting_dir(mid)
    journal_path = meeting_dir / "journal.jsonl"
    pcm_path = meeting_dir / "audio" / "recording.pcm"

    await _broadcast_json(
        {
            "type": "finalize_progress",
            "step": 4,
            "total_steps": 6,
            "label": "Running full-audio diarization...",
            "eta_seconds": max(5, int(audio_duration_s * 0.15)),
        }
    )

    # ── Decide whether the eager draft summary is usable ────────
    # The draft is usable if the transcript hasn't grown by more than
    # 20% since the draft was generated. In that case we skip the
    # expensive LLM call at finalization time entirely.
    _n_final_events = 0
    if journal_path.exists():
        for _line in journal_path.read_text().splitlines():
            if not _line.strip():
                continue
            try:
                _ev = _json.loads(_line)
            except Exception:
                continue
            if _ev.get("is_final") and (_ev.get("text") or "").strip():
                _n_final_events += 1

    _use_draft = (
        _eager_summary_cache is not None
        and _eager_summary_event_count > 0
        and _n_final_events > 0
        and (_n_final_events - _eager_summary_event_count) / _eager_summary_event_count <= 0.20
    )
    if _use_draft:
        logger.info(
            "Eager summary: draft usable (%d draft events, %d final events, %.0f%% growth)",
            _eager_summary_event_count,
            _n_final_events,
            (_n_final_events - _eager_summary_event_count) / _eager_summary_event_count * 100,
        )
    elif _eager_summary_cache is not None:
        logger.info(
            "Eager summary: draft stale (%d draft events → %d final events), will regenerate",
            _eager_summary_event_count,
            _n_final_events,
        )

    # ── Start summary LLM call in parallel with diarization ───
    # When no usable draft exists, fire the summary generation concurrently
    # with diarization so they overlap. The summary uses the pre-diarization
    # journal (live speaker attributions) which is good enough — speaker
    # stats are recalculated from the post-diarization journal afterwards.
    _parallel_summary_task: asyncio.Task | None = None
    if not _use_draft:
        from meeting_scribe.summary import generate_summary as _gen_summary

        _summary_dir = storage._meeting_dir(mid)

        async def _parallel_summary():
            return await _gen_summary(
                _summary_dir,
                vllm_url=config.translate_vllm_url,
            )

        _parallel_summary_task = asyncio.create_task(_parallel_summary())

    if journal_path.exists() and pcm_path.exists():
        try:
            from meeting_scribe.reprocess import (
                _attach_speakers_to_events,
                _audio_quality_report,
                _diarize_full_audio,
            )

            # Load final events — dedup with translation preservation.
            # Priority: version with translation > highest revision > first.
            # Without this, the translated journal line (same revision as
            # the original) gets discarded, and the diarization revision
            # inherits the non-translated version, permanently losing it.
            events_by_sid: dict[str, dict] = {}
            for line in journal_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    event = _json.loads(line)
                except Exception:
                    continue
                if event.get("type") == "speaker_correction":
                    continue
                if not event.get("is_final") or not event.get("text"):
                    continue
                sid = event.get("segment_id")
                if not sid:
                    continue
                existing = events_by_sid.get(sid)
                if not existing:
                    events_by_sid[sid] = event
                else:
                    e_has_tr = bool((event.get("translation") or {}).get("text"))
                    ex_has_tr = bool((existing.get("translation") or {}).get("text"))
                    e_rev = event.get("revision", 0)
                    ex_rev = existing.get("revision", 0)
                    if e_has_tr and not ex_has_tr:
                        events_by_sid[sid] = event
                    elif not e_has_tr and ex_has_tr:
                        # Higher revision without translation: merge it in
                        if e_rev > ex_rev:
                            event["translation"] = existing["translation"]
                            events_by_sid[sid] = event
                    elif e_rev > ex_rev:
                        if ex_has_tr and not e_has_tr:
                            event["translation"] = existing["translation"]
                        events_by_sid[sid] = event

            if events_by_sid:
                logger.info(
                    "Stop: running full-audio diarization on %d events (%.0fMB audio)",
                    len(events_by_sid),
                    pcm_path.stat().st_size / 1024 / 1024,
                )
                pcm_data = pcm_path.read_bytes()
                quality = _audio_quality_report(pcm_data)
                if not quality["usable"]:
                    logger.warning(
                        "Stop finalize: audio is %.1f%% zero-filled — diarization quality limited",
                        quality["zero_fill_pct"],
                    )
                diarize_segments = await _diarize_full_audio(
                    pcm_data,
                    config.diarize_url,
                    max_speakers=8,
                    min_speakers=2,
                )
                if diarize_segments:
                    events_list = list(events_by_sid.values())
                    _attach_speakers_to_events(events_list, diarize_segments)
                    # Append new revisions so journal dedup picks them up
                    with journal_path.open("a") as f:
                        for event_dict in events_list:
                            if not event_dict.get("speakers"):
                                continue
                            event_dict["revision"] = (event_dict.get("revision", 0) or 0) + 1
                            f.write(_json.dumps(event_dict, ensure_ascii=False) + "\n")
                    unique = len({s["cluster_id"] for s in diarize_segments})
                    logger.info(
                        "Stop diarization: %d segments, %d unique global speakers",
                        len(diarize_segments),
                        unique,
                    )
                else:
                    logger.warning("Stop: full-audio diarization returned no segments")
        except Exception as e:
            logger.exception("Stop-time diarization failed: %s", e)

    # Regenerate speaker data + timeline from the (now diarized) journal
    if journal_path.exists():
        _generate_speaker_data(meeting_dir, journal_path, _json)

    await _broadcast_json(
        {
            "type": "finalize_progress",
            "step": 5,
            "total_steps": 6,
            "label": "Generating timeline...",
            "eta_seconds": 5,
        }
    )
    _generate_timeline(mid)

    # ── Collect summary ──────────────────────────────────────────
    await _broadcast_json(
        {
            "type": "finalize_progress",
            "step": 6,
            "total_steps": 6,
            "label": "Generating meeting summary...",
            "eta_seconds": 3,
        }
    )
    summary: dict[str, Any] = {}
    if _use_draft:
        # Promote the cached draft — update speaker stats from the
        # post-diarization journal so names reflect final attribution.
        summary = _eager_summary_cache or {}
        try:
            from meeting_scribe.summary import (
                _calculate_speaker_stats,
                build_transcript_text,
            )

            events, _ = build_transcript_text(meeting_dir)
            speakers_path = meeting_dir / "detected_speakers.json"
            speakers: list[dict] = []
            if speakers_path.exists():
                import json as _sp_json

                with suppress(Exception):
                    speakers = _sp_json.loads(speakers_path.read_text())
            speaker_stats = _calculate_speaker_stats(events, speakers)
            duration_ms = max(e.get("end_ms", 0) for e in events) if events else 0
            languages = set(e.get("language", "unknown") for e in events)
            summary["speaker_stats"] = speaker_stats
            summary["metadata"].update(
                {
                    "meeting_id": mid,
                    "duration_min": round(duration_ms / 60000, 1),
                    "num_segments": len(events),
                    "num_speakers": len(speaker_stats),
                    "languages": sorted(languages - {"unknown"}),
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "is_draft": False,
                    "promoted_from_draft": True,
                }
            )
            # Save promoted draft to disk
            summary_path = meeting_dir / "summary.json"
            summary_path.write_text(_json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
            logger.info(
                "Eager summary: promoted draft to final (%d events)",
                len(events),
            )
        except Exception as e:
            logger.warning("Failed to promote draft summary, regenerating: %s", e)
            _use_draft = False

    if not _use_draft:
        try:
            if _parallel_summary_task:
                # Await the summary that was running in parallel with diarization
                summary = await _parallel_summary_task
            else:
                from meeting_scribe.summary import generate_summary

                meeting_dir = storage._meeting_dir(mid)
                summary = await generate_summary(
                    meeting_dir,
                    vllm_url=config.translate_vllm_url,
                )
            if "error" not in summary:
                logger.info(
                    "Meeting summary generated: %d topics, %d action items",
                    len(summary.get("topics", [])),
                    len(summary.get("action_items", [])),
                )
        except Exception as e:
            logger.warning("Summary generation failed: %s", e)
            summary = {"error": str(e)}

    # Clear eager summary state
    _eager_summary_cache = None
    _eager_summary_event_count = 0

    try:
        storage.transition_state(mid, MeetingState.FINALIZING)
        storage.transition_state(mid, MeetingState.COMPLETE)
    except Exception as e:
        logger.warning("Stop transition error: %s", e)

    # Broadcast completion with summary
    await _broadcast_json(
        {
            "type": "finalize_progress",
            "step": 6,
            "label": "Done!",
            "meeting_id": mid,
            "summary": summary,
        }
    )
    # Also send an explicit meeting_stopped event so UIs that don't listen
    # for finalize_progress (the guest popout, notifications) can act on it.
    # `reason` distinguishes user-initiated stop from auto-stop paths; we
    # use "user_stop" here because this function is only reached via the
    # explicit POST /api/meeting/stop handler.
    await _broadcast_json(
        {
            "type": "meeting_stopped",
            "meeting_id": mid,
            "reason": "user_stop",
        }
    )

    # Close all websockets (after summary broadcast so clients receive it)
    await asyncio.sleep(0.5)  # Give clients time to receive the final message
    for ws in list(ws_connections):
        with suppress(Exception):
            await ws.close(1000, "Meeting ended")
    ws_connections.clear()
    _client_prefs.clear()
    # Close audio-out clients
    for ws in list(_audio_out_clients):
        with suppress(Exception):
            await ws.close(1000, "Meeting ended")
    _audio_out_clients.clear()
    _audio_out_prefs.clear()

    # Stop speaker pulse + catch-up
    if _speaker_pulse_task and not _speaker_pulse_task.done():
        _speaker_pulse_task.cancel()
        _speaker_pulse_task = None
    if _speaker_catchup_task and not _speaker_catchup_task.done():
        _speaker_catchup_task.cancel()
        _speaker_catchup_task = None
    # Clear pending catch-up state between meetings
    _pending_speaker_events.clear()
    _pending_speaker_timestamps.clear()

    current_meeting = None
    # Cancel all background tasks (TTS synthesis, etc.) — prevent ghost processing
    for task in list(_background_tasks):
        if not task.done():
            task.cancel()
    _background_tasks.clear()

    _clear_active_meeting()
    logger.info("Meeting stopped: %s", mid)

    # Stop WiFi AP only (not the full demo stack)
    asyncio.create_task(_stop_wifi_ap())

    return JSONResponse({"status": "complete", "meeting_id": mid})


def _generate_speaker_data(
    meeting_dir,
    journal_path,
    _json,
    expected_speakers: int | None = None,
) -> None:
    """Generate detected_speakers.json and speaker_lanes.json from journal events.

    Applies speaker_correction journal entries (the same way get_meeting does)
    so reprocessed speaker stats include user-assigned names, not just raw
    cluster IDs. Without this, reprocessing wipes speaker identities.

    ``expected_speakers``: when set, after the size-based ghost dissolve
    runs, additionally force-collapse the smallest surviving clusters into
    their nearest neighbor (by time) until exactly N remain. This is the
    "user told us the count, deliver exactly that" lever — it catches
    leftover stale-cluster events that the size threshold missed.
    """
    # First pass: collect corrections (segment_id → speaker_name)
    corrections: dict[str, str] = {}
    best: dict[str, dict] = {}
    for line in journal_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = _json.loads(line)
        except Exception:
            continue
        if e.get("type") == "speaker_correction":
            corrections[e.get("segment_id", "")] = e.get("speaker_name", "")
            continue
        if not e.get("is_final") or not e.get("text"):
            continue
        sid = e.get("segment_id")
        rev = e.get("revision", 0)
        if sid not in best or rev > best[sid].get("revision", 0):
            best[sid] = e

    # Apply corrections to event speakers
    for sid, e in best.items():
        if sid in corrections:
            name = corrections[sid]
            speakers = e.get("speakers", [])
            if speakers:
                speakers[0]["identity"] = name
                speakers[0]["display_name"] = name
            else:
                e["speakers"] = [{"identity": name, "display_name": name, "cluster_id": 0}]

    events = sorted(best.values(), key=lambda e: e.get("start_ms", 0))

    # Orphan speaker reassignment — no more "Speaker Unknown".
    # Events that landed before diarize caught up can have empty speakers[]
    # or cluster_id == 0. Strategy:
    #   1. First pass: reassign orphans to nearest real speaker WITHIN 5s
    #      (high confidence — same conversational beat).
    #   2. Second pass: any orphan that still has no neighbor falls back
    #      to the nearest-in-time real speaker with NO distance limit.
    #      Meetings often open with one host talking for a minute before
    #      anyone else joins; the 5s window can never reach them, but
    #      attributing those segments to the only nearby real voice is
    #      far better than dropping a whole minute of audio under cluster
    #      id 0 — which (a) leaks into the seq_index allocation and
    #      poisons "Speaker 1", and (b) leaves a gaping unattributed
    #      block in the timeline.
    #   3. If the meeting has NO real speakers at all, leave orphans
    #      alone — that's a genuinely speakerless recording.
    _ORPHAN_WINDOW_MS = 5000
    real_events = [
        e for e in events if e.get("speakers") and (e["speakers"][0].get("cluster_id") or 0) > 0
    ]

    def _attribute_to(orphan_e, donor_e) -> None:
        donor_sp = donor_e["speakers"][0]
        orphan_e["speakers"] = [
            {
                "cluster_id": donor_sp.get("cluster_id"),
                "display_name": donor_sp.get("identity") or donor_sp.get("display_name"),
                "identity": donor_sp.get("identity") or donor_sp.get("display_name"),
                "source": "orphan_reassigned",
            }
        ]

    if real_events:
        for e in events:
            sp = e.get("speakers") or []
            cid = sp[0].get("cluster_id") if sp else None
            if sp and cid and cid > 0:
                continue
            mid_ms = (e.get("start_ms", 0) + e.get("end_ms", e.get("start_ms", 0))) / 2
            nearest = min(
                real_events,
                key=lambda r: abs(((r.get("start_ms", 0) + r.get("end_ms", 0)) / 2) - mid_ms),
            )
            # Always attribute — distance is informational only. Better to
            # guess "the only voice in the room right now" than to surface
            # a black hole as cluster 0.
            _attribute_to(e, nearest)

    # ── Tiny-cluster dissolution + expected-count collapse ─────
    # The full-audio diarize pass + its consolidation work on diarize
    # output, but events the new pass DIDN'T temporally overlap keep
    # their stale live-diarize cluster_id. Those leftovers create ghost
    # clusters in the final speakers list (the "Speaker 2 with 2 segs /
    # 7s" symptom). We do TWO passes:
    #
    #  1. Size-based dissolve: any cluster below the ghost floor (<15s
    #     OR <5 segments) gets its events reassigned to the nearest
    #     real cluster by time.
    #  2. Expected-count collapse: when the caller passed
    #     ``expected_speakers=N``, additionally fold the smallest
    #     surviving clusters into their nearest neighbor until exactly
    #     N remain. This catches stale leftovers that survive the size
    #     check but the user knows aren't real.
    #
    # All reassignments happen on the in-memory ``events`` list, which
    # is the single source of truth for detected_speakers.json,
    # speaker_lanes.json, and the journal rewrite below — keeping all
    # four artifacts aligned.
    _DISSOLVE_MAX_DURATION_MS = 15_000
    _DISSOLVE_MAX_SEGMENTS = 5

    def _cluster_sizes() -> tuple[dict[int, int], dict[int, int]]:
        durations: dict[int, int] = {}
        counts: dict[int, int] = {}
        for ev in events:
            sp = ev.get("speakers") or []
            cid = sp[0].get("cluster_id") if sp else 0
            if not cid or cid <= 0:
                continue
            durations[cid] = durations.get(cid, 0) + max(
                0, ev.get("end_ms", 0) - ev.get("start_ms", 0)
            )
            counts[cid] = counts.get(cid, 0) + 1
        return durations, counts

    def _reassign_events_from(targets: set[int], reason: str) -> int:
        if not targets:
            return 0
        keepers = [
            e
            for e in events
            if (e.get("speakers") or [{}])[0].get("cluster_id", 0) > 0
            and (e.get("speakers") or [{}])[0].get("cluster_id") not in targets
        ]
        if not keepers:
            return 0
        for ev in events:
            sp = ev.get("speakers") or []
            cid = sp[0].get("cluster_id") if sp else 0
            if not cid or cid not in targets:
                continue
            mid_ms = (ev.get("start_ms", 0) + ev.get("end_ms", ev.get("start_ms", 0))) / 2
            nearest = min(
                keepers,
                key=lambda r: abs(((r.get("start_ms", 0) + r.get("end_ms", 0)) / 2) - mid_ms),
            )
            _attribute_to(ev, nearest)
        logger.info(
            "%s: reassigned events from %d cluster(s) %s to nearest real speakers",
            reason,
            len(targets),
            sorted(targets),
        )
        return len(targets)

    # Pass 1: size-based dissolve
    durations, counts = _cluster_sizes()
    ghost_ids = {
        cid
        for cid in durations
        if durations[cid] < _DISSOLVE_MAX_DURATION_MS
        and counts.get(cid, 0) <= _DISSOLVE_MAX_SEGMENTS
    }
    _reassign_events_from(ghost_ids, "Ghost dissolve")

    # Pass 2: expected-count collapse
    if expected_speakers is not None and expected_speakers > 0:
        durations, _ = _cluster_sizes()
        live = sorted(durations.keys(), key=lambda c: durations[c], reverse=True)
        if len(live) > expected_speakers:
            collapse_ids = set(live[expected_speakers:])
            _reassign_events_from(
                collapse_ids,
                f"Expected-count collapse (target={expected_speakers})",
            )

    # Build speaker data from corrected events.
    #
    # The cluster's participant-list display_name is chosen by MAJORITY
    # VOTE over the identities present on its events — not just the
    # first one encountered. This matters when a cluster carries a mix
    # of identities (e.g. pyannote clustered Joel + Nikul + Danny
    # together on f38d5807): the sidebar shows the dominant voice name,
    # while individual transcript rows keep their own `identity` field
    # set by upstream corrections.
    from collections import Counter as _Counter

    speaker_stats: dict[int, dict] = {}
    speaker_lanes: dict[str, list] = {}
    cluster_identity_votes: dict[int, _Counter] = {}

    for e in events:
        sp = e.get("speakers", [])
        cluster_id = 0 if not sp else sp[0].get("cluster_id", 0)

        # Skip the orphan sentinel entirely. Reassignment above already
        # tried to attribute every event to a real speaker — anything
        # still tagged cluster_id=0 is unattributable (e.g. a meeting
        # with literally zero recognized voices). Letting cluster_id=0
        # fall through here would (a) eat seq_index=1 from the real
        # speakers, and (b) emit a phantom "Speaker 1" lane in the UI.
        if cluster_id == 0:
            continue

        cid_str = str(cluster_id)
        start_ms = e.get("start_ms", 0)
        end_ms = e.get("end_ms", start_ms + 1500)

        # Build lanes
        if cid_str not in speaker_lanes:
            speaker_lanes[cid_str] = []
        speaker_lanes[cid_str].append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "segment_id": e.get("segment_id", ""),
            }
        )

        # Build stats — prefer corrected identity over raw cluster name.
        # Store sequential "Speaker N" based on first-seen order (not raw
        # cluster_id which may be 100, 101 etc. from the pseudo-cluster
        # fallback). This keeps the UI consistent with the client-side
        # registry that uses sequential numbering.
        # Vote for this cluster's display name using this event's
        # identity (if the user explicitly labeled it). Generic
        # "Speaker N" labels don't vote — they'd dilute real names.
        if sp:
            candidate = sp[0].get("identity") or sp[0].get("display_name")
            if candidate:
                import re as _re

                if not _re.match(r"^Speaker\s+\d+$", candidate.strip()):
                    cluster_identity_votes.setdefault(cluster_id, _Counter())[candidate] += 1

        if cluster_id not in speaker_stats:
            seq_index = len(speaker_stats) + 1
            speaker_stats[cluster_id] = {
                "display_name": f"Speaker {seq_index}",  # placeholder, resolved below
                "cluster_id": cluster_id,
                "seq_index": seq_index,
                "segment_count": 0,
                "total_speaking_ms": 0,
                "first_seen_ms": start_ms,
                "last_seen_ms": end_ms,
            }
        speaker_stats[cluster_id]["segment_count"] += 1
        speaker_stats[cluster_id]["total_speaking_ms"] += max(0, end_ms - start_ms)
        speaker_stats[cluster_id]["last_seen_ms"] = max(
            speaker_stats[cluster_id]["last_seen_ms"], end_ms
        )

    # After counting, assign each cluster its dominant identity.
    for cid, votes in cluster_identity_votes.items():
        if cid in speaker_stats and votes:
            dominant_name = votes.most_common(1)[0][0]
            speaker_stats[cid]["display_name"] = dominant_name

    # Save.
    # Important: remap the raw cluster_ids we see in-flight (which can
    # number in the dozens after a long meeting with lots of diarize
    # fragmentation) to the stable "Speaker N" seq_index we just assigned.
    # Post-finalize consumers (timeline playback, detected-speakers list,
    # speaker_lanes) ALL key on this seq_index so the participant list and
    # the raw transcript view agree on "how many speakers" and "which
    # speaker". Raw cluster_id stays on each entry for traceability.
    # Drop cluster_id=0 (the orphan sentinel) — after the reassignment
    # pass above it should only hold events we truly couldn't attribute,
    # and surfacing "Unknown" is worse than silently omitting a few
    # sub-second fragments from the participant list.
    speakers_list = sorted(
        [s for s in speaker_stats.values() if s["cluster_id"] > 0],
        key=lambda s: -s["segment_count"],
    )
    # Build the cluster_id → seq_index mapping now that every speaker_stats
    # entry has its seq_index populated.
    cluster_to_seq: dict[int, int] = {s["cluster_id"]: s["seq_index"] for s in speakers_list}

    # Rewrite the lanes dict with seq_index as the key (stringified for JSON).
    # Skip any cluster_id that didn't survive into the speakers list
    # (orphans, stale ids) so the UI never gets a phantom lane.
    remapped_lanes: dict[str, list] = {}
    for cid_str, entries in speaker_lanes.items():
        try:
            cid = int(cid_str)
        except ValueError:
            continue
        if cid <= 0:
            continue
        seq = cluster_to_seq.get(cid)
        if seq is None:
            continue
        remapped_lanes.setdefault(str(seq), []).extend(entries)

    speakers_path = meeting_dir / "detected_speakers.json"
    speakers_path.write_text(_json.dumps(speakers_list, indent=2))

    lanes_path = meeting_dir / "speaker_lanes.json"
    lanes_path.write_text(_json.dumps(remapped_lanes, indent=2))

    # Rewrite journal.jsonl so post-finalize views of the raw transcript
    # show "Speaker N" consistent with the participants list AND so
    # orphan-reassigned events carry their newly-attributed speaker
    # (otherwise the first minute of a meeting where diarization hadn't
    # caught up renders with no speaker highlighting).
    #
    # We rewrite from the in-memory ``events`` list — which already has
    # corrections applied AND orphan reassignment applied — rather than
    # re-reading the journal from disk, so all three downstream files
    # (detected_speakers, speaker_lanes, journal) agree second-by-second.
    journal_path = meeting_dir / "journal.jsonl"
    if journal_path.exists():
        # Rewrite the journal from the IN-MEMORY best/events list,
        # NOT by re-iterating the on-disk journal. The on-disk journal
        # is append-only — every re-finalize call stacks another
        # revision on top, and iterating it produces duplicates. We
        # emit exactly one line per segment_id (the highest-revision
        # event with the fully-resolved speakers) + all
        # speaker_correction sentinel rows verbatim.
        #
        # This is the fix for the 2026-04-14 journal-doubling bug:
        # running /api/meetings/<id>/finalize twice used to produce
        # 2× → 4× → 8× lines because each run appended rev=N+1 and
        # then the rewrite loop processed EVERY on-disk line again.
        resolved_events: dict[str, dict] = {}  # sid → full event dict
        for e in events:
            sid = e.get("segment_id")
            if not sid:
                continue
            sp_list = e.get("speakers") or []
            new_sp_list: list[dict] = []
            for sp_entry in sp_list:
                raw = sp_entry.get("cluster_id")
                if raw is None or raw <= 0:
                    continue
                seq = cluster_to_seq.get(raw)
                if seq is None:
                    continue
                # Stamp the seq_index over cluster_id so the UI's join
                # against detected_speakers.json works. Keep the raw
                # cluster_id under _raw_cluster_id for diarize debugging.
                merged = dict(sp_entry)
                merged["_raw_cluster_id"] = raw
                merged["cluster_id"] = seq
                new_sp_list.append(merged)
            # Write back the resolved speakers onto the event dict,
            # then record the full event for journal output.
            e_out = dict(e)
            e_out["speakers"] = new_sp_list
            resolved_events[sid] = e_out

        # Collect speaker_correction rows from the current on-disk
        # journal so they survive the rewrite. These are sentinels
        # the UI writes when a user renames a speaker.
        correction_lines: list[str] = []
        for line in journal_path.open("r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                evt = _json.loads(line)
            except Exception:
                correction_lines.append(line)  # preserve unparseable
                continue
            if evt.get("type") == "speaker_correction":
                correction_lines.append(line)

        # Build the new journal: one final event per segment_id
        # (sorted by start_ms for reproducibility), then the
        # speaker_corrections.
        out_lines: list[str] = []
        for e in sorted(resolved_events.values(), key=lambda x: x.get("start_ms", 0)):
            out_lines.append(_json.dumps(e, ensure_ascii=False))
        out_lines.extend(correction_lines)
        journal_path.write_text("\n".join(out_lines) + "\n")

    logger.info(
        "Generated speaker data: %d speakers, %d lane entries (journal remapped)",
        len(speakers_list),
        sum(len(v) for v in remapped_lanes.values()),
    )


def _generate_timeline(meeting_id: str, meeting_dir=None) -> None:
    """Generate timeline.json from journal for podcast player.

    ``meeting_dir`` is optional — callers that already have the path
    (e.g. reprocess.py which runs before the server's storage global
    is initialized) can pass it directly to avoid touching the module
    global.
    """
    import json as _json

    if meeting_dir is None:
        meeting_dir = storage._meeting_dir(meeting_id)
    journal_path = meeting_dir / "journal.jsonl"
    timeline_path = meeting_dir / "timeline.json"

    # Deduplicate by segment_id, keeping highest revision
    best: dict[str, dict] = {}
    if journal_path.exists():
        for line in journal_path.open():
            line = line.strip()
            if not line:
                continue
            try:
                e = _json.loads(line)
                if e.get("text") and e.get("is_final"):
                    sid = e.get("segment_id")
                    rev = e.get("revision", 0)
                    if sid not in best or rev > best[sid].get("revision", 0):
                        best[sid] = e
            except Exception:
                continue

    # Build segments sorted by start_ms
    segments = sorted(
        [
            {
                "segment_id": e.get("segment_id"),
                "start_ms": e.get("start_ms", 0),
                "end_ms": e.get("end_ms", 0),
                "language": e.get("language", "unknown"),
                "speaker_id": (e.get("speakers") or [{}])[0].get("cluster_id")
                if e.get("speakers")
                else None,
                "text": e.get("text", "")[:100],
            }
            for e in best.values()
        ],
        key=lambda s: s["start_ms"],
    )

    # Audio duration via storage if available; fall back to reading the
    # recording.pcm directly if the module-global storage hasn't been
    # initialized (reprocess.py invocation path — no lifespan yet).
    audio_ms = 0
    try:
        audio_ms = storage.audio_duration_ms(meeting_id)
    except Exception:
        pcm = meeting_dir / "audio" / "recording.pcm"
        if pcm.exists():
            audio_ms = int(pcm.stat().st_size // 2 / 16000 * 1000)
    # Timeline duration = audio file duration. This is the canonical
    # length users see in the scrubber / speaker lanes. If events only
    # cover part of the audio (e.g. a crashed session or a meeting that
    # was never fully ASR'd), the proper fix is to re-run full-reprocess
    # so the journal gets ASR events for the whole recording — NOT to
    # shrink the timeline and hide untranscribed audio.
    duration_ms = audio_ms if audio_ms > 0 else max((s["end_ms"] for s in segments), default=0)
    for s in segments:
        if s["end_ms"] > duration_ms:
            s["end_ms"] = duration_ms
        if s["start_ms"] > duration_ms:
            s["start_ms"] = duration_ms
    timeline = {"duration_ms": duration_ms, "segments": segments}
    timeline_path.write_text(_json.dumps(timeline, indent=2))
    logger.info("Generated timeline.json: %d segments, %dms", len(segments), duration_ms)


# ── Slide translation endpoints ──────────────────────────────


def _require_admin(request: fastapi.Request) -> None:
    """Raise 403 if the request is guest-scope. Defense-in-depth."""
    if _is_guest_scope(request):
        raise fastapi.HTTPException(403, "Admin access required")


@app.post("/api/meetings/{meeting_id}/slides/upload")
async def upload_slides(meeting_id: str, request: fastapi.Request):
    """Upload a PPTX file for slide translation.

    Triggers the full pipeline: validate -> render -> extract -> translate
    -> reinsert -> render translated. Original slides are viewable as soon
    as the rendering stage completes.
    """
    _require_admin(request)

    if not slides_enabled or slide_job_runner is None:
        return JSONResponse(
            {"error": "Slide processing unavailable (worker container not found)"},
            status_code=503,
        )

    form = await request.form()
    upload = form.get("file")
    if upload is None or isinstance(upload, str):
        return JSONResponse({"error": "No file uploaded"}, status_code=400)

    contents = await upload.read()
    upload_filename = getattr(upload, "filename", None) or ""
    logger.info(
        "Slide upload: %d bytes filename=%r",
        len(contents),
        upload_filename,
    )

    # Default to THIS MEETING's language pair (so a Dutch meeting doesn't
    # silently get the global ja↔en pair). Fall back to the global default
    # only if the meeting's meta is unreadable. Explicit form overrides win
    # and bypass auto-detection — useful when the detector misclassifies
    # (e.g. a Japanese deck dominated by English brand names).
    meeting_source = ""
    meeting_target = ""
    meeting_monolingual = False
    try:
        _meta = storage._read_meta(meeting_id)
        if _meta and _meta.language_pair:
            if _meta.is_monolingual:
                meeting_source = _meta.language_pair[0]
                meeting_target = ""  # no target for monolingual decks
                meeting_monolingual = True
            elif len(_meta.language_pair) == 2:
                meeting_source, meeting_target = _meta.language_pair[0], _meta.language_pair[1]
    except Exception:
        pass
    if not meeting_monolingual and not (meeting_source and meeting_target):
        # Fall back to the process-wide default. Only reached for decks
        # uploaded against a meeting whose meta is unreadable.
        parts = [p.strip() for p in config.default_language_pair.split(",")]
        if len(parts) == 1:
            meeting_source = parts[0]
            meeting_target = ""
            meeting_monolingual = True
        elif len(parts) >= 2:
            meeting_source, meeting_target = parts[0], parts[1]
    _src = form.get("source_lang") or ""
    _tgt = form.get("target_lang") or ""
    source_lang = (_src if isinstance(_src, str) else "").strip() or meeting_source
    target_lang = (_tgt if isinstance(_tgt, str) else "").strip() or meeting_target
    explicit_override = bool(form.get("source_lang") and form.get("target_lang"))

    if explicit_override:
        # An explicit target override wins over the meeting's monolingual
        # default — the operator is asking for a specific translation.
        meeting_monolingual = False
        logger.info(
            "Slide upload: explicit language override %s→%s",
            source_lang,
            target_lang,
        )

    deck_id = await slide_job_runner.start_job(
        meeting_id,
        contents,
        source_lang,
        target_lang,
        skip_language_detection=explicit_override,
        upload_filename=upload_filename,
        monolingual=meeting_monolingual,
    )

    return JSONResponse({"deck_id": deck_id, "status": "processing"})


@app.get("/api/meetings/{meeting_id}/decks")
async def list_meeting_decks(meeting_id: str):
    """List every slide deck on disk for a meeting (newest first).

    Each deck entry includes its meta + a `is_active` flag. Used by the
    popout's deck switcher so the user can flip between multiple
    uploaded decks during a meeting OR review past decks afterwards.
    """
    if slide_job_runner is None:
        return JSONResponse({"error": "Slides not enabled"}, status_code=503)
    return JSONResponse(
        {
            "meeting_id": meeting_id,
            "decks": slide_job_runner.list_decks(meeting_id),
        }
    )


@app.put("/api/meetings/{meeting_id}/decks/active")
async def set_active_meeting_deck(meeting_id: str, request: fastapi.Request):
    """Switch the active deck for a meeting. Body: ``{"deck_id": "..."}``.

    Broadcasts ``slide_deck_changed`` so connected viewers swap to the
    chosen deck without a manual reload. Works on both live and past
    meetings — past meeting "switching" just changes which deck the
    UI/endpoints serve next.
    """
    if slide_job_runner is None:
        return JSONResponse({"error": "Slides not enabled"}, status_code=503)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    deck_id = (body or {}).get("deck_id", "").strip() if isinstance(body, dict) else ""
    if not deck_id:
        return JSONResponse({"error": "Missing deck_id"}, status_code=400)
    meta = await slide_job_runner.set_active_deck(meeting_id, deck_id)
    if meta is None:
        return JSONResponse({"error": "Deck not found"}, status_code=404)
    return JSONResponse({"meeting_id": meeting_id, "deck_id": deck_id, "meta": meta})


@app.get("/api/meetings/{meeting_id}/slides")
async def get_slides_metadata(meeting_id: str):
    """Get deck metadata for any meeting that has slides on disk.

    Works for both the live meeting and past completed meetings (slides
    are part of the meeting record, not just live state). The
    ``current_slide_index`` field is only populated for the singleton
    active deck — past meetings get the persisted final value or 0.
    """
    if slide_job_runner is None:
        return JSONResponse({"error": "Slides not enabled"}, status_code=503)

    # Past-meeting path: read deck meta directly from disk for this
    # specific meeting, without disturbing the singleton active state.
    meta = slide_job_runner.get_meeting_deck_meta(meeting_id)
    if meta is None:
        return JSONResponse({"error": "No deck for this meeting"}, status_code=404)

    # If THIS meeting is the live one, layer in the in-memory current index
    # so the popout can resume from where the presenter left off.
    if slide_job_runner.active_deck_id and slide_job_runner.active_deck_id == meta.get("deck_id"):
        meta["current_slide_index"] = slide_job_runner.current_slide_index
    else:
        meta.setdefault("current_slide_index", 0)
    return JSONResponse(meta)


@app.get("/api/meetings/{meeting_id}/slides/original.pdf")
async def get_slides_original_pdf(meeting_id: str):
    """Serve the original slides as a PDF (works for past meetings too)."""
    if slide_job_runner is None:
        return JSONResponse({"error": "Slides not enabled"}, status_code=503)

    pdf_path = slide_job_runner.get_original_pdf_path(meeting_id)
    if pdf_path is None:
        return JSONResponse({"error": "PDF not yet rendered or no deck"}, status_code=404)

    from starlette.responses import FileResponse

    return FileResponse(pdf_path, media_type="application/pdf")


@app.get("/api/meetings/{meeting_id}/slides/source.pptx")
async def get_slides_source_pptx(meeting_id: str):
    """Serve the original PPTX (works for past meetings too)."""
    if slide_job_runner is None:
        return JSONResponse({"error": "Slides not enabled"}, status_code=503)

    pptx_path = slide_job_runner.get_source_pptx_path(meeting_id)
    if pptx_path is None:
        return JSONResponse({"error": "Source PPTX not found"}, status_code=404)

    from starlette.responses import FileResponse

    return FileResponse(
        pptx_path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename="presentation.pptx",
    )


# OnlyOffice was previously surfaced via /api/onlyoffice/callback +
# /onlyoffice/* reverse proxy. Removed: the popout viewer now uses the
# PNG render path exclusively (no second toolbar, no "Download failed"
# overlay, no loading state). Backend container can be left running but
# nothing reaches it from the app anymore — safe to remove from
# docker-compose if you want to free its memory.


@app.get("/api/meetings/{meeting_id}/slides/{index}/original")
async def get_slide_original(meeting_id: str, index: int):
    """Serve an original slide PNG. Works for both the live meeting AND
    past meetings (slides persist as part of the meeting record now)."""
    if slide_job_runner is None:
        return JSONResponse({"error": "Slides not enabled"}, status_code=503)

    if index < 0:
        return JSONResponse({"error": f"Invalid slide index {index}"}, status_code=422)

    path = slide_job_runner.get_slide_image_path(meeting_id, index, translated=False)
    if path is None:
        return JSONResponse({"error": "Slide not yet rendered or no deck"}, status_code=404)

    from starlette.responses import FileResponse

    return FileResponse(path, media_type="image/png")


@app.get("/api/meetings/{meeting_id}/slides/{index}/translated")
async def get_slide_translated(meeting_id: str, index: int):
    """Serve a translated slide PNG. Works for both the live meeting AND
    past meetings (slides persist as part of the meeting record now)."""
    if slide_job_runner is None:
        return JSONResponse({"error": "Slides not enabled"}, status_code=503)

    if index < 0:
        return JSONResponse({"error": f"Invalid slide index {index}"}, status_code=422)

    path = slide_job_runner.get_slide_image_path(meeting_id, index, translated=True)
    if path is None:
        return JSONResponse({"error": "Translation not yet ready or no deck"}, status_code=404)

    from starlette.responses import FileResponse

    return FileResponse(path, media_type="image/png")


@app.put("/api/meetings/{meeting_id}/slides/current")
async def set_current_slide(meeting_id: str, request: fastapi.Request):
    """Set the current slide index and broadcast to all viewers."""
    _require_admin(request)

    if slide_job_runner is None:
        return JSONResponse({"error": "Slides not enabled"}, status_code=503)

    if slide_job_runner.active_deck_id is None:
        return JSONResponse({"error": "No active deck"}, status_code=404)

    body = await request.json()
    index = body.get("index")
    if index is None or not isinstance(index, int):
        return JSONResponse({"error": "Missing or invalid 'index'"}, status_code=400)

    if not slide_job_runner.advance_slide(index):
        return JSONResponse(
            {"error": f"Slide index {index} out of range (0-{slide_job_runner.total_slides - 1})"},
            status_code=422,
        )

    await _broadcast_json(
        {
            "type": "slide_change",
            "deck_id": slide_job_runner.active_deck_id,
            "slide_index": index,
        }
    )

    return JSONResponse({"ok": True, "slide_index": index})


# --- WebSocket ---


@app.websocket("/api/ws")
async def websocket_audio(websocket: WebSocket) -> None:
    """WebSocket for audio streaming and transcript events.

    Sends binary audio, receives JSON transcript events.
    Also accepts JSON text messages for language preference:
        {"type": "set_language", "language": "en"}

    Admin-only: rejects hotspot clients AND any client connecting over the
    plain-HTTP guest listener (``ws://`` scheme). Admin must use ``wss://``
    via the HTTPS listener on the LAN.
    """
    if _is_guest_scope(websocket):
        await websocket.close(code=4003, reason="Admin endpoint — use wss://<gb10>:8080")
        return
    await websocket.accept()
    ws_connections.add(websocket)
    _client_prefs[websocket] = ClientSession()
    logger.info("WS connected (total=%d)", len(ws_connections))

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("bytes"):
                await _handle_audio(msg["bytes"])
            elif msg.get("text"):
                try:
                    import json as _json

                    parsed = _json.loads(msg["text"])
                    if isinstance(parsed, dict) and parsed.get("type") == "set_language":
                        lang = parsed.get("language", "")
                        if lang:
                            _client_prefs[websocket].preferred_language = lang
                            logger.info("Audio WS set language preference: %s", lang)
                except Exception:
                    pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("WS error: %s", e)
    finally:
        ws_connections.discard(websocket)
        _client_prefs.pop(websocket, None)
        logger.info("WS disconnected (remaining=%d)", len(ws_connections))


@app.websocket("/api/ws/view")
async def websocket_view(websocket: WebSocket) -> None:
    """View-only WebSocket — receives transcript events without sending audio.

    Used by second browsers or client devices that want to watch the
    live transcript without mic access.
    Accepts JSON text messages for language preference:
        {"type": "set_language", "language": "en"}
    """
    await websocket.accept()
    ws_connections.add(websocket)
    _client_prefs[websocket] = ClientSession()
    logger.info("View-only WS connected (total=%d)", len(ws_connections))

    # Replay current meeting's journal so late-joining clients catch up
    if current_meeting:
        try:
            lines = storage.read_journal_raw(current_meeting.meeting_id, max_lines=500)
            if lines:
                total = len(lines)
                for line in lines:
                    if line.strip():
                        await websocket.send_text(line)
                logger.info("Replayed %d journal events to view WS", total)
        except Exception as e:
            logger.warning("Journal replay failed: %s", e)

    try:
        while True:
            text = await websocket.receive_text()
            # Parse language preference messages
            try:
                import json as _json

                msg = _json.loads(text)
                if isinstance(msg, dict) and msg.get("type") == "set_language":
                    lang = msg.get("language", "")
                    if lang:
                        _client_prefs[websocket].preferred_language = lang
                        logger.info("View WS set language preference: %s", lang)
            except Exception:
                pass  # Ignore non-JSON (pings, etc.)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        ws_connections.discard(websocket)
        _client_prefs.pop(websocket, None)
        logger.info("View-only WS disconnected (remaining=%d)", len(ws_connections))


# ── Audio-out wire protocol constants ───────────────────────────────
# Hotspot listener hard cap. Hotspot guard already restricts this
# endpoint to the meeting subnet, but this is belt-and-braces against
# accidental tab-spam or a misbehaving client reopening the WS in a
# loop.
_MAX_AUDIO_OUT_CLIENTS = 32
# Grace window after WS accept during which a client may negotiate
# `audio_format` via `{"type": "set_format", ...}`. If the grace
# expires with no negotiation, we default to "wav-pcm" (backward
# compat with legacy clients). Audio deliveries that arrive during
# the grace window are held in `pref.pending_audio` (capped) and
# flushed once the format is known.
_AUDIO_FORMAT_GRACE_S = 1.0
# Max buffered audio seconds while waiting for `set_format`. Per-
# listener bound so a stuck handshake can't balloon memory.
_AUDIO_FORMAT_PENDING_CAP_S = 1.0
_VALID_AUDIO_FORMATS = frozenset({"wav-pcm", "mse-fmp4-aac"})


def _create_audio_out_session(ws: WebSocket) -> ClientSession:
    """Create a ClientSession for an audio-out listener and register it.

    Pure helper — always succeeds.  Does NOT check capacity or scope
    (those guards live in the handler).  Does NOT touch
    ``_audio_out_clients`` — slot reservation is the handler's job.
    """
    pref = ClientSession(
        send_audio=True,
        voice_mode=_effective_tts_voice_mode(),
        grace_deadline=time.monotonic() + _AUDIO_FORMAT_GRACE_S,
    )
    _audio_out_prefs[ws] = pref
    return pref


def _unregister_audio_out_client(ws: WebSocket) -> None:
    """Remove an audio-out listener from globals and close its encoder.

    Idempotent — safe to call multiple times for the same ``ws``.
    """
    pref = _audio_out_prefs.pop(ws, None)
    _audio_out_clients.discard(ws)
    if pref is not None and pref.mse_encoder is not None:
        try:
            pref.mse_encoder.close()
        except Exception as e:
            logger.debug("mse_encoder close on disconnect: %r", e)
        pref.mse_encoder = None


async def _handle_audio_out_message(ws: WebSocket, pref: ClientSession, text: str) -> str | None:
    """Parse and apply one JSON control message from an audio-out client.

    Returns the ``format_ack`` JSON string to send back when the message
    is a valid ``set_format``, or ``None`` otherwise.
    """
    import json as _json

    try:
        msg = _json.loads(text)
    except Exception:
        return None
    if not isinstance(msg, dict):
        return None

    msg_type = msg.get("type")
    if msg_type == "set_format":
        fmt = msg.get("format", "")
        if fmt not in _VALID_AUDIO_FORMATS:
            logger.warning(
                "Audio-out set_format: rejected %r from %s",
                fmt,
                _peer_str(ws),
            )
            return None
        old_format = pref.audio_format
        pref.audio_format = fmt
        logger.info("Audio-out format: %s for %s", fmt, _peer_str(ws))
        # If the client changed its mind about an existing encoder,
        # tear the old one down.
        if old_format is not None and old_format != fmt and pref.mse_encoder is not None:
            try:
                pref.mse_encoder.close()
            except Exception as e:
                logger.debug("old mse_encoder close: %r", e)
            pref.mse_encoder = None
        # Flush any audio held during the negotiation grace window.
        await _flush_pending_audio(ws, pref)
        return _json.dumps({"type": "format_ack", "format": fmt})

    if msg_type == "set_language":
        lang = _norm_lang(msg.get("language", ""))
        if lang:
            pref.preferred_language = lang
            logger.info("Audio-out WS set language preference: %s", lang)
    elif msg_type == "set_mode":
        mode = msg.get("mode", "translation")
        if mode in ("translation", "full"):
            pref.interpretation_mode = mode
            logger.info("Audio-out WS set interpretation mode: %s", mode)
    elif msg_type == "set_voice":
        voice = msg.get("voice", "studio")
        if voice in ("studio", "cloned"):
            pref.voice_mode = voice
            logger.info("Audio-out WS set voice mode: %s", voice)
    return None


@app.websocket("/api/ws/audio-out")
async def websocket_audio_out(websocket: WebSocket) -> None:
    """Audio-out WebSocket — sends synthesized TTS audio to guest-scope clients.

    Clients send JSON text messages for preference updates:
        {"type": "set_format",   "format":   "wav-pcm" | "mse-fmp4-aac"}
        {"type": "set_language", "language": "en"}
        {"type": "set_mode",     "mode":     "translation" | "full"}
        {"type": "set_voice",    "voice":    "studio" | "cloned"}

    On `set_format`, server responds with a text message:
        {"type": "format_ack", "format": "<accepted-format>"}

    Server sends binary audio frames:
        - wav-pcm listeners:   one complete RIFF WAV per audio segment,
                               no prefix byte (backward-compatible with
                               legacy clients)
        - mse-fmp4-aac listeners: one fMP4 init frame prefixed with
                                  b'\\x49' ('I'), then media fragments
                                  prefixed with b'\\x46' ('F').

    Legacy clients that never send `set_format` receive wav-pcm via the
    grace-window default — no URL versioning or protocol upgrade prompt
    required across a deploy.
    """
    # Audio-out is guest-scope ONLY. The admin interface must never play
    # interpretation audio — the operator is typically in the room with
    # the speaker, and bleeding TTS out of the admin console creates
    # instant feedback loops and confuses the meeting.
    if not _is_guest_scope(websocket):
        await websocket.close(code=1008, reason="audio-out is guest-scope only")
        return
    # Atomic cap check + slot reservation (no await between them).
    # This prevents another coroutine from consuming the last slot
    # while we're awaiting accept().
    if len(_audio_out_clients) >= _MAX_AUDIO_OUT_CLIENTS:
        logger.warning(
            "Audio-out WS refused: at capacity (%d/%d)",
            len(_audio_out_clients),
            _MAX_AUDIO_OUT_CLIENTS,
        )
        await websocket.close(code=1013, reason="audio-out at capacity")
        return
    _audio_out_clients.add(websocket)  # reserve slot synchronously
    try:
        await websocket.accept()
    except Exception:
        _audio_out_clients.discard(websocket)  # release slot on accept failure
        return
    pref = _create_audio_out_session(websocket)
    client = getattr(websocket, "client", None)
    logger.info(
        "Audio-out WS connected from %s (hotspot=%s, total=%d)",
        client.host if client else "?",
        _is_hotspot_client(websocket),
        len(_audio_out_clients),
    )

    try:
        while True:
            text = await websocket.receive_text()
            ack = await _handle_audio_out_message(websocket, pref, text)
            if ack:
                await websocket.send_text(ack)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _unregister_audio_out_client(websocket)
        logger.info("Audio-out WS disconnected (remaining=%d)", len(_audio_out_clients))


async def _handle_audio(data: bytes) -> None:
    """Handle audio from browser AudioWorklet.

    Wire format (per chunk, legacy-compatible):
        - NEW: [4B LE uint32 sample_rate][N*2 bytes s16le PCM at native rate]
          Server resamples to 16kHz using torchaudio Kaiser-windowed sinc.
        - LEGACY: [N*2 bytes s16le PCM, assumed 16kHz]
          Detected by the first 4 bytes NOT being a plausible sample rate.

    Why server-side resampling: browser JS linear interpolation causes aliasing.
    torchaudio's Kaiser sinc is effectively transparent and runs on GPU.
    Bandwidth cost (~3× at 48kHz) is trivial on local WiFi.
    """
    global _last_audio_chunk_ts
    _last_audio_chunk_ts = time.monotonic()

    if len(data) < 6:
        return

    import numpy as np

    # Parse optional sample-rate header. A valid header has a uint32 in the
    # expected range (8000..192000). Anything else is legacy raw 16kHz PCM.
    header_rate = int.from_bytes(data[:4], "little")
    if 8000 <= header_rate <= 192000:
        source_rate = header_rate
        pcm_bytes = data[4:]
    else:
        source_rate = 16000
        pcm_bytes = data

    if len(pcm_bytes) < 2:
        return

    metrics.audio_chunks += 1
    num_source_samples = len(pcm_bytes) // 2
    metrics.audio_seconds += num_source_samples / source_rate

    # Decode s16le → float32 [-1, 1]
    source = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Resample to 16kHz using torchaudio Kaiser sinc (anti-aliased).
    # The Resampler caches a torchaudio transform per source_rate so the
    # kernel is computed once and reused across chunks.
    if source_rate != 16000:
        try:
            resampled = resampler.resample(source, source_rate=source_rate, target_rate=16000)
        except Exception as e:
            logger.warning("Resample %d→16000 failed: %s — falling back to raw", source_rate, e)
            resampled = source
    else:
        resampled = source

    # Back to s16le for downstream consumers (audio writer + ASR)
    # Clip to [-1, 1] before quantizing to avoid wraparound on occasional overshoot
    resampled_clipped = np.clip(resampled, -1.0, 1.0)
    pcm16_bytes = (resampled_clipped * 32767).astype(np.int16).tobytes()

    # Log audio level at chunk #1 and every 40 after — WARNING level so it
    # shows regardless of root logger config. This is a critical diagnostic:
    # if this log stops appearing, the audio pipeline is broken.
    if metrics.audio_chunks == 1 or metrics.audio_chunks % 40 == 0:
        peak = float(np.max(np.abs(resampled_clipped))) if len(resampled_clipped) > 0 else 0.0
        rms = float(np.sqrt(np.mean(resampled_clipped**2))) if len(resampled_clipped) > 0 else 0.0
        logger.warning(
            "Audio chunk #%d: %dHz→16kHz, %d→%d samples, peak=%.4f rms=%.4f",
            metrics.audio_chunks,
            source_rate,
            num_source_samples,
            len(resampled_clipped),
            peak,
            rms,
        )

    # Write 16kHz audio to recording file (time-aligned) and capture
    # the absolute sample offset this chunk lands at. That offset is
    # the single source of truth for ASR + timeline alignment — the
    # audio writer is what playback reads, so transcript events must
    # be stamped relative to the same file, not an in-memory counter.
    chunk_sample_offset: int | None = None
    if audio_writer and meeting_start_time > 0:
        elapsed_ms = int((time.monotonic() - meeting_start_time) * 1000)
        # Snapshot the audio writer's byte position BEFORE this write —
        # that's where the chunk will land in the PCM file.
        pre_bytes = audio_writer.total_bytes
        audio_writer.write_at(pcm16_bytes, elapsed_ms)
        chunk_sample_offset = pre_bytes // 2  # s16le → 1 sample = 2 bytes

        # One-shot wall-clock anchor for the recording file. Set on the
        # very first sample so `recording.pcm` becomes an absolute-time
        # record: byte offset → unix epoch is deterministic forever.
        # Only set it if this write actually extended the file from
        # zero; on resume (append mode) pre_bytes>0 and the anchor
        # was already written at the original start.
        if (
            current_meeting is not None
            and pre_bytes == 0
            and getattr(current_meeting, "recording_started_epoch_ms", 0) == 0
        ):
            current_meeting.recording_started_epoch_ms = int(time.time() * 1000)
            with suppress(Exception):
                storage._write_meta(current_meeting)
            logger.info(
                "Recording anchor: meeting=%s started_epoch_ms=%d",
                current_meeting.meeting_id,
                current_meeting.recording_started_epoch_ms,
            )

        # Audio drift detection: wall clock vs cumulative audio bytes
        audio_elapsed_ms = int(metrics.audio_seconds * 1000)
        drift_ms = elapsed_ms - audio_elapsed_ms
        if abs(drift_ms) > 500 and metrics.audio_chunks % 20 == 0:
            logger.warning(
                "Audio drift detected: %dms (wall=%dms, audio=%dms)",
                drift_ms,
                elapsed_ms,
                audio_elapsed_ms,
            )
            await _broadcast_json({"type": "audio_drift", "drift_ms": drift_ms})

    if not asr_backend:
        return

    # Feed 16kHz s16le bytes to ASR with the absolute audio-file sample
    # offset so every emitted event is stamped against the ACTUAL audio
    # playback position, not an internal counter that resets on restart.
    await asr_backend.process_audio_bytes(pcm16_bytes, sample_offset=chunk_sample_offset)

    # Also feed to diarization backend (runs in parallel, buffers internally)
    if diarize_backend:
        try:
            await diarize_backend.process_audio(
                resampled_clipped.astype(np.float32),
                metrics.audio_chunks * len(resampled_clipped),
            )
            _record_backend_success("diarize")
        except Exception as e:
            _record_backend_failure("diarize", str(e))
            # Diarization failures shouldn't block ASR


from meeting_scribe.speaker.name_extraction import (
    extract_name as _extract_name_from_text,
)


async def _process_event(event: TranscriptEvent) -> None:
    """Store event, broadcast to WS clients, queue translation."""
    metrics.asr_events += 1
    metrics.last_asr_event_time = time.monotonic()
    if event.is_final:
        metrics.asr_finals += 1
    else:
        metrics.asr_partials += 1

    # Filler-only finals: ASR frequently emits a final for a single
    # backchannel syllable ("ああ", "うん", "はい", "そう") between real
    # utterances. These carry no information, they inflate translate
    # cost, and they clutter the transcript. Drop them here before any
    # downstream fanout. The allowlist below matches the most common JA
    # + EN filler patterns; we keep text if it has ≥ 2 CJK chars or ≥ 3
    # non-space ASCII chars outside the allowlist.
    if event.is_final and event.text:
        _filler = {
            "ああ",
            "ああ。",
            "うん",
            "うん。",
            "はい",
            "はい。",
            "えっと",
            "えっと。",
            "あ",
            "あ。",
            "そう",
            "そう。",
            "ね",
            "ね。",
            "なん",
            "なん。",
            "ええ",
            "ええ。",
            "uh",
            "uh.",
            "um",
            "um.",
            "ah",
            "ah.",
            "mm",
            "mm.",
            "yeah",
            "yeah.",
            "ok",
            "ok.",
        }
        normalized = event.text.strip().lower()
        if normalized in _filler or len(normalized) <= 1:
            metrics.asr_finals_filler_dropped = getattr(metrics, "asr_finals_filler_dropped", 0) + 1
            logger.info(
                "Filler final dropped: seg=%s text='%s'",
                event.segment_id,
                event.text[:40],
            )
            return

    # Short-window text-hash dedup on FINALS ONLY.
    # When scribe-asr recovers from a container restart (e.g. GPU contention
    # killed its worker), it can replay buffered audio under fresh segment_ids
    # so LocalAgreement's revision-based dedup (keyed on seg_id) can't catch it.
    # Drop a final whose exact (language, normalized-text) we already saw
    # within _DEDUP_WINDOW_S, before it fans out to storage/WS/translate.
    if event.is_final and event.text:
        now = time.monotonic()
        key = (event.language or "", event.text.strip())
        prior = _recent_finals.get(key)
        if prior is not None and (now - prior[0]) < _DEDUP_WINDOW_S:
            metrics.asr_finals_deduped = getattr(metrics, "asr_finals_deduped", 0) + 1
            logger.warning(
                "Dup ASR final dropped: seg=%s prev_seg=%s dt=%.2fs text='%s'",
                event.segment_id,
                prior[1],
                now - prior[0],
                event.text,
            )
            return
        _recent_finals[key] = (now, event.segment_id)
        if len(_recent_finals) > 256:
            stale = [k for k, (ts, _) in _recent_finals.items() if now - ts > _DEDUP_WINDOW_S]
            for k in stale:
                _recent_finals.pop(k, None)

    # Enforce language pair: if ASR detected a language outside the active pair
    # (or couldn't detect one at all), remap it to the best match so the frontend
    # doesn't silently drop the event. The renderer's filter
    # `lang !== langA && lang !== langB` rejects anything not in the pair —
    # INCLUDING "unknown" — which used to cause events to disappear.
    #
    # We *also* override the ASR's claimed language when the text's script
    # disagrees with it (e.g. ASR returns lang=en for kana-heavy text, or
    # lang=en for hangul "내장."). This is the gate that decides whether
    # furigana fires AND whether out-of-pair-script text leaks into the wrong
    # column, so a script/label mismatch causes both bugs at once. The
    # override fires whenever the script and label disagree, *regardless* of
    # whether the script's language is in the active pair — out-of-pair
    # script content is then mapped to the best in-pair fallback (CJK→first
    # CJK lang in pair, else first non-CJK lang in pair).
    if current_meeting and hasattr(current_meeting, "language_pair") and event.text:
        pair = current_meeting.language_pair
        from meeting_scribe.backends.asr_filters import _detect_language_from_text

        script_lang = _detect_language_from_text(event.text)

        def _fallback_for_script(script: str) -> str:
            """Pick the closest in-pair language for an out-of-pair script."""
            cjk_set = {"ja", "zh", "ko"}
            cjk_langs = [lang for lang in pair if lang in cjk_set]
            if script in cjk_set and cjk_langs:
                return cjk_langs[0]
            non_cjk = [lang for lang in pair if lang not in cjk_set]
            return non_cjk[0] if non_cjk else pair[0]

        needs_remap = event.language not in pair
        # Case 1: the script disagrees with the ASR label — trust the script.
        # If script_lang is in pair, use it directly. Otherwise pick the
        # closest in-pair fallback. This is how "내장." tagged en gets
        # re-routed to ja in a ja↔en meeting instead of leaking into the
        # English column. Skip when script_lang is "unknown" (e.g. all
        # punctuation / numbers) — we have no signal to act on.
        if not needs_remap and script_lang != "unknown" and script_lang != event.language:
            target = script_lang if script_lang in pair else _fallback_for_script(script_lang)
            if target != event.language:
                logger.info(
                    "Language remap: ASR=%s → script=%s → %s (text=%r)",
                    event.language,
                    script_lang,
                    target,
                    event.text[:40],
                )
                event.language = target
        elif needs_remap:
            if script_lang in pair:
                event.language = script_lang
            else:
                event.language = _fallback_for_script(script_lang)

    # Speaker matching: compare audio chunk against enrolled speaker embeddings
    if event.is_final and event.text and enrollment_store.speakers:
        try:
            audio_chunk = getattr(asr_backend, "last_audio_chunk", None)
            if audio_chunk is not None and len(audio_chunk) > 0:
                # Run embedding extraction off the event loop.
                # NOTE: do NOT `import asyncio` locally here — Python scoping
                # would make asyncio a function-local and every earlier
                # `asyncio.create_task` reference in _process_event would
                # raise UnboundLocalError until this conditional ran. The
                # module already imports asyncio at the top.
                from concurrent.futures import ThreadPoolExecutor

                global _speaker_executor_singleton
                if _speaker_executor_singleton is None:
                    _speaker_executor_singleton = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="speaker"
                    )
                _speaker_executor = _speaker_executor_singleton

                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(
                    _speaker_executor, _simple_embedding, audio_chunk
                )

                best_name = None
                best_score = 0.0
                best_eid = None
                from meeting_scribe.speaker.verification import cosine_similarity

                for eid, name, enrolled_emb in enrollment_store.get_all_embeddings():
                    score = cosine_similarity(embedding, enrolled_emb)
                    if score > best_score:
                        best_score = score
                        best_name = name
                        best_eid = eid

                # ENRICH the diarization result with enrolled identity rather
                # than REPLACE it. Preserves per-speaker cluster_ids so downstream
                # timeline lanes and stats show all speakers correctly.
                # Threshold raised from 0.3 → 0.55: 0.3 was too permissive, causing
                # every segment to be claimed by the first enrolled speaker.
                if best_score > 0.55 and best_name:
                    existing_speakers = list(event.speakers or [])
                    if existing_speakers:
                        # Keep original cluster_id, add identity on top
                        original = existing_speakers[0]
                        existing_speakers[0] = SpeakerAttribution(
                            cluster_id=original.cluster_id,
                            identity=best_name,
                            identity_confidence=best_score,
                            source="enrolled",
                        )
                    else:
                        existing_speakers = [
                            SpeakerAttribution(
                                cluster_id=0,
                                identity=best_name,
                                identity_confidence=best_score,
                                source="enrolled",
                            )
                        ]
                    event = event.model_copy(update={"speakers": existing_speakers})
                    existing = [s for s in detected_speakers if s.speaker_id == best_eid]
                    if existing:
                        existing[0].segment_count += 1
                        existing[0].last_seen_ms = event.end_ms
                    else:
                        detected_speakers.append(
                            DetectedSpeaker(
                                speaker_id=best_eid or "",
                                display_name=best_name,
                                matched_enrollment_id=best_eid,
                                match_confidence=best_score,
                                segment_count=1,
                                first_seen_ms=event.start_ms,
                                last_seen_ms=event.end_ms,
                            )
                        )
        except Exception as e:
            logger.debug("Speaker matching error: %s", e)

    # Automatic name discovery has been intentionally removed.
    # Names come from only two sources:
    #   1. Pre-meeting voice enrollment (explicit seat assignment) — handled above
    #   2. Explicit user corrections via the UI (update_segment_speaker endpoint)
    # Any speaker without a known identity will display as "Speaker {cluster_id}".
    #
    # Previously self-introduction detection + LLM name extraction could silently
    # auto-assign names to unnamed seats, which was unreliable and confusing.

    # Diarization: attach speaker cluster_id from diarization backend
    # First attempt is synchronous — if results are already cached we use them.
    # If not (diarization buffers audio and lags ASR), we queue the event
    # for the background catch-up loop which re-attributes and re-broadcasts
    # once diarization produces results.
    if (
        diarize_backend
        and event.is_final
        and hasattr(diarize_backend, "get_results_for_range")
        and not event.speakers  # don't overwrite enrolled identification
    ):
        try:
            results = diarize_backend.get_results_for_range(event.start_ms, event.end_ms)
            if results:
                event = event.model_copy(
                    update={
                        "speakers": [
                            SpeakerAttribution(
                                cluster_id=results[0].cluster_id,
                                source="diarization",
                            )
                        ]
                    }
                )
            elif event.text and len(event.text.strip()) > 0:
                # Queue for retroactive attribution by the catch-up loop
                _pending_speaker_events[event.segment_id] = event
                _pending_speaker_timestamps[event.segment_id] = time.monotonic()
                # Cap the pending queue (oldest 200 segments — ~10 minutes)
                while len(_pending_speaker_events) > 200:
                    old_sid, _ = _pending_speaker_events.popitem(last=False)
                    _pending_speaker_timestamps.pop(old_sid, None)
        except Exception:
            pass

    # SPEAKER ATTRIBUTION DELIBERATELY NOT GUESSED HERE.
    #
    # Pre-2026-04 this code allocated a time-proximity "pseudo cluster"
    # (cluster_id ≥ 100) when diarization hadn't produced a result yet,
    # so the UI wouldn't show "Unknown". That was a guess — same speaker
    # across a silence gap might get a different pseudo-cluster, or two
    # different speakers might share one if they spoke back-to-back.
    #
    # The user's directive for Part B of the 2026-04 refactor: ASR should
    # identify SPEECH only. The catch-up loop is the sole source of
    # speaker attribution, and it only runs when diarization has a real
    # answer. Segments broadcast here with empty speakers[] are queued
    # in _pending_speaker_events above, and _speaker_catchup_loop will
    # broadcast a revised event (same segment_id, incremented revision)
    # once pyannote has assigned a real cluster. The frontend dedups by
    # segment_id, so the UI updates in place: transcript line appears
    # first with no speaker badge, then the badge fills in seconds later.
    #
    # Enrollment matching runs earlier in this function (above). That
    # path is kept because enrolled-voice match is deterministic and
    # high-confidence — not a guess.

    if current_meeting:
        storage.append_event(current_meeting.meeting_id, event)

    await _broadcast(event)

    # Full interpretation mode: pass through original audio to clients
    # whose preferred language matches the speaker's language
    if event.is_final and event.text and _audio_out_clients:
        audio_chunk = getattr(asr_backend, "last_audio_chunk", None)
        if audio_chunk is not None and len(audio_chunk) > 0:
            await _send_passthrough_audio(audio_chunk, event.language)

    if event.is_final and translation_queue:
        metrics.translations_submitted += 1
        logger.info(
            "Submitting for translation: seg=%s lang=%s text='%s'",
            event.segment_id,
            event.language,
            event.text[:60],
        )
        baseline, optional = _compute_translation_demand(event)
        await translation_queue.submit(event, baseline_targets=baseline, optional_targets=optional)
        # Flush merge gate immediately — don't hold segments waiting for merge
        await translation_queue.flush_merge_gate()

    # Furigana annotation — concurrent, non-blocking, best-effort.
    # Runs as a separate asyncio.create_task so it never delays the
    # translate submit. Uses its own httpx pool so the connection pool
    # is separate from translate_queue. Result is broadcast as an event
    # revision with a `furigana_html` field when it lands; the UI picks
    # it up and re-renders the transcript block in place. Priority is
    # intentionally +5 (lower than translate's -10) so vLLM's priority
    # scheduler preempts for the critical path under contention.
    if event.is_final and event.language == "ja" and event.text and furigana_backend is not None:

        async def _annotate_and_broadcast(ev):
            try:
                html = await furigana_backend.annotate(ev.text)
                if not html:
                    logger.info(
                        "furigana: no annotation for seg=%s text=%s (no kanji or cache miss)",
                        ev.segment_id,
                        ev.text[:30],
                    )
                    return
                logger.info(
                    "furigana: seg=%s text=%s → %d chars",
                    ev.segment_id,
                    ev.text[:30],
                    len(html),
                )
                updated = ev.model_copy(
                    update={
                        "furigana_html": html,
                        "revision": (ev.revision or 0) + 1,
                    }
                )
                if current_meeting:
                    try:
                        storage.append_event(current_meeting.meeting_id, updated)
                    except Exception as exc:
                        logger.warning("furigana: journal append failed: %s", exc)
                await _broadcast(updated)
            except Exception as e:
                logger.warning("furigana task failed seg=%s: %s", ev.segment_id, e)

        task = asyncio.create_task(
            _annotate_and_broadcast(event),
            name=f"furigana-{event.segment_id}",
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)


async def _eager_summary_loop(meeting_id: str) -> None:
    """Periodically generate draft summaries during recording.

    Runs every 90 seconds after an initial 2-minute warmup. Uses vLLM
    priority 5 (well below translation's -10) so real-time translation
    is never starved. The cached draft is used at stop time if the
    transcript hasn't grown significantly, avoiding the full LLM call
    during finalization.

    The loop self-terminates when current_meeting changes or is cleared.
    """
    global _eager_summary_cache, _eager_summary_event_count

    # Long warmup — don't compete with translation/ASR during first 5 minutes.
    # Most meetings under 5 min don't benefit from eager summary anyway.
    await asyncio.sleep(300)

    while current_meeting and current_meeting.meeting_id == meeting_id:
        try:
            meeting_dir = storage._meeting_dir(meeting_id)
            journal_path = meeting_dir / "journal.jsonl"
            if not journal_path.exists():
                await asyncio.sleep(300)
                continue

            # Count current finals to decide if a draft is worthwhile
            import json as _esj

            n_finals = 0
            for line in journal_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    e = _esj.loads(line)
                except Exception:
                    continue
                if e.get("is_final") and e.get("text", "").strip():
                    n_finals += 1

            # Skip if too few events (< 50) or transcript hasn't grown
            # significantly since last draft (< 30% growth)
            if n_finals < 50 or n_finals == _eager_summary_event_count:
                await asyncio.sleep(300)
                continue
            if _eager_summary_event_count > 0 and n_finals < _eager_summary_event_count * 1.3:
                await asyncio.sleep(300)
                continue

            logger.info(
                "Eager summary: generating draft (%d events, prev=%d)",
                n_finals,
                _eager_summary_event_count,
            )

            from meeting_scribe.summary import generate_draft_summary

            draft, event_count = await generate_draft_summary(
                meeting_dir,
                vllm_url=config.translate_vllm_url,
            )

            if draft and current_meeting and current_meeting.meeting_id == meeting_id:
                _eager_summary_cache = draft
                _eager_summary_event_count = event_count
                logger.info(
                    "Eager summary: draft cached (%d events, %.0fms)",
                    event_count,
                    draft.get("metadata", {}).get("generation_ms", 0),
                )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("Eager summary loop error (non-fatal): %s", e)

        await asyncio.sleep(300)  # 5 min between drafts to avoid GPU pressure

    logger.info("Eager summary loop exiting (meeting changed or stopped)")


async def _speaker_pulse_loop() -> None:
    """Periodic speaker pulse broadcast (every 200ms).

    Sends active speaker info to all WebSocket clients for smoother
    speaker indicator animations. "Currently speaking" = the single
    most recently heard speaker, within a short recency window.

    The window must be short enough that natural back-and-forth doesn't
    leave the previous speaker lit up while the next one starts — 2s is
    wide enough that A/B conversations highlight both seats at once.
    700ms still survives the small silence gaps inside an utterance but
    collapses to one seat during a handoff.
    """
    import json as _json

    SPEAKING_WINDOW_MS = 700

    while True:
        await asyncio.sleep(0.2)

        if not ws_connections or not current_meeting:
            continue

        now = time.monotonic()
        meeting_elapsed_ms = int((now - metrics.meeting_start) * 1000)
        active_speakers: list[dict] = []

        # Pick the SINGLE most-recently-active detected speaker within the
        # window. Conversations rarely have true overlap, and highlighting
        # only the latest speaker prevents the previous speaker's seat from
        # lingering in a pulse during a handoff.
        most_recent = None
        most_recent_last_seen = -1
        for i, speaker in enumerate(detected_speakers):
            if not speaker.last_seen_ms:
                continue
            age_ms = meeting_elapsed_ms - speaker.last_seen_ms
            if 0 <= age_ms < SPEAKING_WINDOW_MS and speaker.last_seen_ms > most_recent_last_seen:
                most_recent = (i, speaker)
                most_recent_last_seen = speaker.last_seen_ms

        if most_recent is not None:
            i, speaker = most_recent
            active_speakers.append(
                {
                    "seat_index": i,
                    "name": speaker.display_name,
                    "confidence": speaker.match_confidence,
                }
            )

        # From diarization backend — only when no enrolled speaker is
        # already claiming the pulse. Also tightened to the same window so
        # diarization tails don't drag previous clusters along.
        if (
            not active_speakers
            and diarize_backend
            and hasattr(diarize_backend, "get_results_for_range")
        ):
            recent_results = diarize_backend.get_results_for_range(
                max(0, meeting_elapsed_ms - SPEAKING_WINDOW_MS), meeting_elapsed_ms
            )
            if recent_results:
                latest = max(recent_results, key=lambda dr: dr.end_ms)
                active_speakers.append(
                    {
                        "cluster_id": latest.cluster_id,
                        "confidence": latest.confidence,
                    }
                )

        pulse_data = _json.dumps(
            {
                "type": "speaker_pulse",
                "active_speakers": active_speakers,
                "timestamp_ms": int((now - metrics.meeting_start) * 1000),
            }
        )

        dead: list[WebSocket] = []
        for ws in list(ws_connections):
            try:
                await ws.send_text(pulse_data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            ws_connections.discard(ws)


async def _silence_watchdog_loop() -> None:
    """Emit a meeting_warning WS broadcast when audio ingestion stalls.

    The browser's audio WebSocket occasionally drops (NAT, tab throttle,
    CPU starvation) and never reconnects. Server-side state stays
    ``recording`` so the UI looks hung. This loop flips a one-shot
    warning once no audio chunk has arrived for `_SILENCE_WARN_THRESHOLD_S`
    and resets it on the next chunk. Clients pick the event up off the
    regular `/api/ws` channel and can render a banner / prompt a refresh.
    """
    global _silence_warn_sent
    POLL = 1.0
    while True:
        await asyncio.sleep(POLL)
        try:
            if not current_meeting or not _last_audio_chunk_ts:
                _silence_warn_sent = False
                continue
            age = time.monotonic() - _last_audio_chunk_ts
            if age >= _SILENCE_WARN_THRESHOLD_S and not _silence_warn_sent:
                logger.warning(
                    "Silence watchdog: no audio in %.0fs for meeting %s — notifying clients",
                    age,
                    current_meeting.meeting_id,
                )
                await _broadcast_json(
                    {
                        "type": "meeting_warning",
                        "reason": "no_audio",
                        "age_s": round(age, 1),
                        "meeting_id": current_meeting.meeting_id,
                    }
                )
                _silence_warn_sent = True
            elif age < _SILENCE_WARN_THRESHOLD_S and _silence_warn_sent:
                logger.info(
                    "Silence watchdog: audio resumed after %.1fs for meeting %s",
                    age,
                    current_meeting.meeting_id,
                )
                await _broadcast_json(
                    {
                        "type": "meeting_warning_cleared",
                        "meeting_id": current_meeting.meeting_id,
                    }
                )
                _silence_warn_sent = False
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("silence watchdog error: %s", e)


async def _speaker_catchup_loop() -> None:
    """Background catch-up task for retroactive speaker attribution.

    Fixes the timing mismatch between ASR (finalizes in ~1-2s) and diarization
    (buffers 4s before processing). Events that landed in _process_event before
    diarization had results sit in _pending_speaker_events. Every 1s this loop:

    1. Walks the pending queue oldest-first
    2. Asks diarize_backend.get_results_for_range for each pending event's range
    3. If results exist, attaches cluster_id, writes a new revision to the
       journal, and re-broadcasts the event — the frontend updates the block
       in-place by segment_id across all views (admin, popout, guest)
    4. Also applies enrolled-speaker names retroactively when diarization
       embeddings match a later-enrolled voice
    5. Gives up on events older than 45s (diarization has processed that
       region long ago; if still no result, it was likely silence)
    """
    # WARNING level so it shows up in logs regardless of handler config
    logger.warning("Speaker catch-up loop started (aggressive mode: 400ms poll)")
    MAX_AGE_SECONDS = 45.0
    POLL_INTERVAL = 0.4  # Aggressive polling — GB10 has plenty of headroom
    _tick_count = 0
    _resolved_count = 0

    try:
        while True:
            await asyncio.sleep(POLL_INTERVAL)
            _tick_count += 1

            # Heartbeat every 10s so we can see the loop is alive + see what
            # state the diarize cache is in while we're debugging.
            if _tick_count % 25 == 0:
                cache_size = (
                    len(getattr(diarize_backend, "_result_cache", {})) if diarize_backend else 0
                )
                logger.warning(
                    "Catch-up heartbeat: pending=%d, diarize_cache=%d, resolved_total=%d",
                    len(_pending_speaker_events),
                    cache_size,
                    _resolved_count,
                )

            # Drain any pending centroid-consolidation renames (fix 4) and
            # broadcast them to the UI. The client's speaker registry then
            # collapses "Speaker 41" onto the surviving cluster's label so
            # live view matches the post-finalize view without waiting for
            # the meeting to end.
            if diarize_backend is not None:
                pending_renames = getattr(diarize_backend, "_cluster_rename", None)
                if pending_renames:
                    renames = dict(pending_renames)
                    pending_renames.clear()
                    if renames:
                        logger.info("Broadcasting %d speaker_remap(s) to UI", len(renames))
                        await _broadcast_json(
                            {
                                "type": "speaker_remap",
                                "renames": {str(k): v for k, v in renames.items()},
                            }
                        )

            if not current_meeting or not diarize_backend:
                continue
            if not _pending_speaker_events:
                continue

            now = time.monotonic()
            to_resolve: list[str] = []
            to_evict: list[str] = []

            for segment_id, event in list(_pending_speaker_events.items()):
                age = now - _pending_speaker_timestamps.get(segment_id, now)
                if age > MAX_AGE_SECONDS:
                    to_evict.append(segment_id)
                    continue

                # Look up diarization results for this event's time range
                try:
                    results = diarize_backend.get_results_for_range(
                        event.start_ms,
                        event.end_ms,
                    )
                except Exception:
                    continue

                if not results:
                    continue

                # Overlap-aware speaker attribution: primary = longest-
                # overlapping cluster, secondaries kept when co-speech is
                # meaningful. Same thresholds as _attach_speakers_to_events
                # in reprocess.py so live + finalize paths agree.
                def _overlap(dr):
                    return max(0, min(dr.end_ms, event.end_ms) - max(dr.start_ms, event.start_ms))

                ev_dur = max(1, event.end_ms - event.start_ms)
                overlap_by_cluster: dict[int, float] = {}
                from meeting_scribe.backends.diarize_sortformer import DiarizationResult as _DR

                results_by_cluster: dict[int, _DR] = {}
                for dr in results:
                    ov = _overlap(dr)
                    if ov <= 0:
                        continue
                    overlap_by_cluster[dr.cluster_id] = (
                        overlap_by_cluster.get(dr.cluster_id, 0) + ov
                    )
                    # Keep the representative result (max-confidence) per cluster.
                    prev = results_by_cluster.get(dr.cluster_id)
                    if prev is None or (dr.confidence or 0) > (getattr(prev, "confidence", 0) or 0):
                        results_by_cluster[dr.cluster_id] = dr
                if not overlap_by_cluster:
                    continue
                ranked_cids = sorted(overlap_by_cluster.items(), key=lambda kv: -kv[1])
                primary_cid, primary_ov = ranked_cids[0]
                best = results_by_cluster[primary_cid]

                # Resolve identity via enrollment if embedding is available
                identity = None
                identity_conf = 0.0
                source = "diarization"
                if best.embedding is not None and enrollment_store.speakers:
                    try:
                        from meeting_scribe.speaker.verification import cosine_similarity

                        best_score = 0.0
                        best_name = None
                        for eid, name, enrolled_emb in enrollment_store.get_all_embeddings():
                            score = cosine_similarity(best.embedding, enrolled_emb)
                            if score > best_score:
                                best_score = score
                                best_name = name
                        if best_score > 0.55 and best_name:
                            identity = best_name
                            identity_conf = best_score
                            source = "enrolled"
                    except Exception:
                        pass

                # Build the updated speakers list: primary first, then any
                # secondary clusters whose overlap is ≥ 30 % of the event
                # AND ≥ 50 % of the primary's overlap.
                updated_speakers = [
                    SpeakerAttribution(
                        cluster_id=best.cluster_id,
                        identity=identity,
                        identity_confidence=identity_conf,
                        source=source,
                    )
                ]
                for cid, ov in ranked_cids[1:]:
                    if ov / ev_dur < 0.30 or ov / primary_ov < 0.50:
                        continue
                    dr2 = results_by_cluster[cid]
                    updated_speakers.append(
                        SpeakerAttribution(
                            cluster_id=dr2.cluster_id,
                            identity=None,
                            identity_confidence=dr2.confidence or 0.0,
                            source="diarization_overlap",
                        )
                    )
                updated = event.model_copy(
                    update={
                        "speakers": updated_speakers,
                        "revision": (event.revision or 0) + 1,
                    }
                )

                # Persist the new revision to the journal (readers dedup by
                # segment_id keeping highest revision)
                if current_meeting:
                    try:
                        storage.append_event(current_meeting.meeting_id, updated)
                    except Exception as e:
                        logger.debug("Catch-up journal append failed: %s", e)

                # Broadcast to all live views (admin, popout, guest) so their
                # rendered blocks update in place by segment_id.
                try:
                    await _broadcast(updated)
                except Exception:
                    pass

                to_resolve.append(segment_id)
                _resolved_count += 1

                logger.warning(
                    "Speaker catch-up: seg=%s age=%.1fs → cluster %d%s",
                    segment_id,
                    age,
                    best.cluster_id,
                    f" ({identity})" if identity else "",
                )

            # Clean up
            for sid in to_resolve + to_evict:
                _pending_speaker_events.pop(sid, None)
                _pending_speaker_timestamps.pop(sid, None)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception("Speaker catch-up loop crashed: %s", e)


async def _broadcast(event: TranscriptEvent) -> None:
    """Send transcript event to all connected WebSocket clients.

    Reaches admin audio WS, view-only WS (popout + guest), and hotspot
    guests since they all share the ws_connections set. Frontends look up
    the block by segment_id and update in place.
    """
    data = event.model_dump_json()
    dead: list[WebSocket] = []
    for ws in list(ws_connections):
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_connections.discard(ws)


async def _broadcast_json(data: dict) -> None:
    """Send arbitrary JSON to all connected WebSocket clients."""
    import json as _json

    text = _json.dumps(data)
    dead: list[WebSocket] = []
    for ws in list(ws_connections):
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_connections.discard(ws)


async def _broadcast_translation(event: TranscriptEvent) -> None:
    """Callback from translation queue — track metrics, persist, and broadcast."""
    if event.translation:
        status = event.translation.status.value
        if status == "done":
            metrics.translations_completed += 1
            # Persist the translation to the journal
            if current_meeting:
                storage.append_event(current_meeting.meeting_id, event)
        elif status == "failed":
            metrics.translations_failed += 1

    # Broadcast to all clients (includes in_progress, done, failed, skipped)
    await _broadcast(event)

    # TTS: synthesize translated text in speaker's voice (background — don't block translation queue)
    # ONLY runs when at least one listener is connected — no point burning GPU for nobody.
    # Voice references are still cached so listeners get speaker's voice when they join later.
    #
    # When _TTS_DEFER_UNTIL_CATCH_UP is True (Part B of the 2026-04 speaker
    # separation refactor), segments with empty event.speakers are skipped
    # at broadcast time and TTS is fired from the catch-up loop instead,
    # once the speaker attribution has been resolved by diarization.
    if (
        tts_backend
        and tts_backend.available
        and event.translation
        and event.translation.status.value == "done"
        and event.translation.text
        and not (_TTS_DEFER_UNTIL_CATCH_UP and not event.speakers)
    ):
        # Cache voice reference synchronously regardless of listeners
        # (so future listeners get the speaker's voice immediately).
        #
        # Derive a stable speaker_id for the voice cache:
        #   1. The speaker's enrolled identity ("Tanaka") when set
        #   2. The diarization cluster_id ("cluster_0", "cluster_1") when
        #      the speaker is detected but not yet enrolled
        #   3. "default" only when there is no speaker attribution at all
        # The old code was `speaker.identity if speaker else "default"`,
        # which returned None for freshly-detected speakers (identity is
        # None on every new cluster until enrollment), and the voice
        # cache skipped them silently because `if audio_chunk and
        # speaker_id:` evaluates False on None. With no voice cached,
        # `_do_tts_synthesis` would fall through `if ref is None: continue`
        # for every segment and Listen would produce nothing.
        speaker = event.speakers[0] if event.speakers else None
        if speaker is None:
            speaker_id = "default"
        elif speaker.identity:
            speaker_id = speaker.identity
        else:
            speaker_id = f"cluster_{speaker.cluster_id}"

        audio_chunk = getattr(asr_backend, "last_audio_chunk", None)
        if audio_chunk is not None:
            tts_backend.cache_voice(speaker_id, audio_chunk)

        # Skip synthesis entirely if no listeners — saves GPU
        if not _audio_out_clients:
            return

        logger.info(
            "TTS fire: seg=%s speaker=%s text=%r",
            event.segment_id,
            speaker_id,
            (event.translation.text or "")[:60],
        )

        _enqueue_tts(event, speaker_id)


def _tts_outstanding() -> int:
    """Total outstanding TTS work: in-flight + queued.

    Used by the producer backlog gates AND the health evaluator. With the
    semaphore-first worker loop, workers parked on _tts_backend_semaphore
    hold NO queue items, so `_tts_in_flight + qsize()` is the complete
    outstanding count. [P1-1-i3 + i4]
    """
    return _tts_in_flight + (_tts_queue.qsize() if _tts_queue else 0)


def _enqueue_tts(event, speaker_id: str) -> None:
    """Push a translation segment onto the FIFO TTS backlog.

    Applies three producer-side gates before enqueue:
      1. Missing-origin drop — event.utterance_end_at is None is a code
         bug (asr_vllm MUST populate it). Refuse the segment. [P1-2-i5]
      2. Whitelist filler — drop short acknowledgment tokens
         ("はい", "I see", ...) when there's real outstanding work, so
         one-word interjections don't block real content. [P2-2-i1]
      3. Stale-on-enqueue — if the deadline has less than MIN_SLACK left
         AND we have outstanding work, drop rather than compete for a
         worker slot that will miss its budget anyway. [P1-2-i1]
    Otherwise the existing drop-oldest policy handles queue saturation.
    """
    global _tts_queue
    if _tts_queue is None:
        return

    # Gate 1 [P1-2-i5]: missing speech-end origin is a code bug. Refuse.
    if event.utterance_end_at is None:
        logger.warning(
            "TTS refuse seg=%s: utterance_end_at missing — asr_vllm did not "
            "populate it (bug: start_meeting must set audio_wall_at_start)",
            event.segment_id,
        )
        metrics.tts_dropped_missing_origin += 1
        return

    # Deadline anchored to when translation completed (upstream-aware), not
    # when the speaker stopped talking. utterance_end_at is audio-time which
    # doesn't account for ASR + translation latency — those can consume 5-6s
    # of a speech-end-based budget before TTS even sees the segment.
    # translation.completed_at is the wall-clock moment upstream finished,
    # giving TTS a clean budget for synthesis + delivery only.
    tts_origin = (
        event.translation.completed_at
        if event.translation and event.translation.completed_at is not None
        else time.monotonic()
    )
    deadline = tts_origin + _TTS_MAX_SPEECH_LAG_S
    outstanding = _tts_outstanding()

    # Gate 2 [P2-2-i1]: whitelist filler drop.
    text = (event.translation.text or "").strip().lower()
    if text in _TTS_ACK_WHITELIST and outstanding > 0:
        logger.info(
            "TTS skip ack seg=%s text=%r (outstanding=%d)",
            event.segment_id,
            text,
            outstanding,
        )
        metrics.tts_dropped_filler += 1
        return

    # Gate 3 [P1-2-i1]: stale-on-enqueue.
    now = time.monotonic()
    remaining = deadline - now
    if remaining < _TTS_PRODUCER_MIN_SLACK_S and outstanding > 0:
        logger.info(
            "TTS skip stale-on-enqueue seg=%s remaining=%.2fs (outstanding=%d)",
            event.segment_id,
            remaining,
            outstanding,
        )
        metrics.tts_dropped_stale_producer += 1
        return

    # Drop-oldest-on-full preserves the "play recent content" invariant.
    while _tts_queue.full():
        try:
            dropped = _tts_queue.get_nowait()
            _tts_queue.task_done()
            dropped_event = dropped[0]
            metrics.tts_dropped_queue_full += 1
            logger.warning(
                "TTS backlog full (%d) — dropping oldest seg=%s to make room for %s",
                _TTS_QUEUE_MAXSIZE,
                getattr(dropped_event, "segment_id", "?"),
                event.segment_id,
            )
        except asyncio.QueueEmpty:
            break
    try:
        # Queue tuple shape: (event, speaker_id, deadline, queued_at)
        _tts_queue.put_nowait((event, speaker_id, deadline, now))
        metrics.tts_submitted += 1
    except asyncio.QueueFull:
        logger.warning("TTS enqueue failed despite make-room loop")


def _start_tts_worker() -> None:
    """Create the TTS queue, container semaphore, and worker pool. Idempotent.

    Spawns ``_TTS_WORKER_COUNT`` parallel worker tasks that drain the same
    shared queue under a semaphore-first flow — each worker acquires
    ``_tts_backend_semaphore`` BEFORE dequeuing, so at most
    ``_TTS_CONTAINER_MAX_CONCURRENCY`` workers are ever committed to a
    segment at once. See ``_tts_worker_loop``.
    """
    global _tts_queue, _tts_worker_tasks, _tts_backend_semaphore
    if _tts_queue is None:
        _tts_queue = asyncio.Queue(maxsize=_TTS_QUEUE_MAXSIZE)
    if _tts_backend_semaphore is None:
        _tts_backend_semaphore = asyncio.Semaphore(_TTS_CONTAINER_MAX_CONCURRENCY)
    # Prune any dead workers so a restart re-spawns them
    _tts_worker_tasks = [t for t in _tts_worker_tasks if not t.done()]
    while len(_tts_worker_tasks) < _TTS_WORKER_COUNT:
        idx = len(_tts_worker_tasks)
        _tts_worker_tasks.append(
            asyncio.create_task(_tts_worker_loop(idx), name=f"tts-worker-{idx}")
        )
    logger.info(
        "TTS worker pool started (workers=%d, container_concurrency=%d, maxsize=%d)",
        len(_tts_worker_tasks),
        _TTS_CONTAINER_MAX_CONCURRENCY,
        _TTS_QUEUE_MAXSIZE,
    )


async def _tts_worker_loop(worker_idx: int = 0) -> None:
    """Semaphore-first TTS worker [P1-1-i4].

    Acquires ``_tts_backend_semaphore`` BEFORE dequeuing, so workers
    without the semaphore hold nothing — no queue item, no in-flight
    count, no metrics. With ``_TTS_CONTAINER_MAX_CONCURRENCY=1`` only
    ONE worker is ever committed to a segment; the other N-1 are parked
    on the semaphore.
    """
    assert _tts_queue is not None
    assert _tts_backend_semaphore is not None
    while True:
        try:
            # 1. ACQUIRE CONTAINER SLOT FIRST — parks here until a slot is free.
            async with _tts_backend_semaphore:
                # 2. Now dequeue exactly one item. Blocks here if empty.
                event, speaker_id, deadline, _queued_at = await _tts_queue.get()
                try:
                    # Guard: skip if no active meeting (ghost synthesis after stop).
                    if not current_meeting:
                        continue
                    # Guard: no listeners → skip synthesis entirely to save GPU.
                    if not _audio_out_clients:
                        continue
                    # 3. Dequeue deadline check [P1-1-i1].
                    if time.monotonic() > deadline:
                        logger.warning(
                            "TTS drop stale-at-dequeue seg=%s over=%.2fs",
                            event.segment_id,
                            time.monotonic() - deadline,
                        )
                        metrics.tts_dropped_stale_worker += 1
                        continue
                    # 4. Run the synth (pre-synth budget check + synthesis
                    #    + post-synth re-check + segment-level histograms
                    #    all live inside _do_tts_synthesis).
                    await _do_tts_synthesis(event, speaker_id, deadline)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning(
                        "TTS worker error on seg=%s: %s",
                        getattr(event, "segment_id", "?"),
                        e,
                    )
                finally:
                    try:
                        _tts_queue.task_done()
                    except ValueError:
                        pass
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Defensive: any unexpected error in the semaphore path should
            # not kill the worker task. Log and loop.
            logger.exception("TTS worker %d outer loop error: %s", worker_idx, e)


def _record_segment_lag(event, now: float) -> None:
    """Record segment-level lag histograms ONCE per segment [P1-5-i2].

    Called exactly once per successful synthesis from _do_tts_synthesis,
    BEFORE fan-out to listeners. N listeners cannot inflate these
    histograms to N samples.
    """
    if event.utterance_end_at is not None:
        metrics.end_to_end_lag_ms.append((now - event.utterance_end_at) * 1000.0)
        if event.translation and event.translation.completed_at is not None:
            metrics.upstream_lag_ms.append(
                (event.translation.completed_at - event.utterance_end_at) * 1000.0
            )
    if event.translation and event.translation.completed_at is not None:
        metrics.tts_post_translation_lag_ms.append((now - event.translation.completed_at) * 1000.0)


async def _do_tts_synthesis(event, speaker_id, deadline: float) -> None:
    """Inner TTS synthesis — runs under the container semaphore.

    Flow:
      1. Cap text to MAX_TTS_CHARS (sentence-boundary aware).
      2. Filter listeners by language/voice_mode → voice_modes_needed.
      3. Pre-synth budget check: drop if remaining < expected. [P1-2-i1]
      4. Size-aware + deadline-aware wait_for. [P1-5-i1 + P1-1-i2]
      5. Post-synth deadline re-check: drop late audio. [P1-1-i2]
      6. Record segment-level histograms ONCE. [P1-5-i2]
      7. Fan out (transport metrics only). [P1-5-i2]

    The caller (_tts_worker_loop) already holds _tts_backend_semaphore,
    so this function does not re-acquire it.
    """
    global _tts_in_flight

    # Caller guarantees tts_backend is non-None by gating TTS work on
    # the global's presence at submit/worker-spawn time — assert it
    # here so mypy can narrow the optional and every downstream
    # `tts_backend.<method>` call typechecks.
    assert tts_backend is not None

    text = event.translation.text
    # Cap text to ~30 seconds of audio (~100 words ≈ 200 chars).
    MAX_TTS_CHARS = 300
    if len(text) > MAX_TTS_CHARS:
        cut = text[:MAX_TTS_CHARS].rfind("。")
        if cut < 100:
            cut = text[:MAX_TTS_CHARS].rfind(". ")
        if cut < 100:
            cut = MAX_TTS_CHARS
        text = text[: cut + 1]
        logger.info("TTS text truncated: %d → %d chars", len(event.translation.text), len(text))

    # Determine which voice modes are needed by listeners.
    target_lang = _norm_lang(event.translation.target_language)
    voice_modes_needed: set[str] = set()
    total_listeners = 0
    skipped_lang = 0
    for ws in _audio_out_clients:
        pref = _audio_out_prefs.get(ws)
        if not pref:
            continue
        total_listeners += 1
        pref_lang = _norm_lang(pref.preferred_language)
        if pref_lang and pref_lang != target_lang:
            skipped_lang += 1
            continue
        voice_modes_needed.add(getattr(pref, "voice_mode", "studio"))

    if not voice_modes_needed:
        logger.info(
            "TTS skip seg=%s: no matching voice modes (target_lang=%s, "
            "listeners=%d, skipped_for_lang=%d)",
            event.segment_id,
            target_lang,
            total_listeners,
            skipped_lang,
        )
        return

    # Synthesize each requested voice mode (default voice, no cloning)
    for voice_mode in voice_modes_needed:
        # Pre-synth budget check [P1-2-i1]. The queue + semaphore wait may
        # have eaten budget since the producer admitted this event. Compute
        # expected synth time from rolling P95 (fallback to 2 s).
        remaining = deadline - time.monotonic()
        # Cap the P95 estimate — one slow request shouldn't poison all
        # future pre-synth checks and cause a cascade of drops.
        p95_s = min((metrics.tts_synth_ms_p95 or 0) / 1000.0, _TTS_SYNTH_TIMEOUT_BASE_S)
        expected = max(_TTS_EXPECTED_SYNTH_DEFAULT_S, p95_s)
        if remaining < expected:
            logger.warning(
                "TTS drop pre-synth seg=%s mode=%s remaining=%.2fs expected=%.2fs",
                event.segment_id,
                voice_mode,
                remaining,
                expected,
            )
            metrics.tts_dropped_pre_synth += 1
            return

        used_fallback = False
        logger.info(
            "TTS synthesize: seg=%s mode=%s speaker_id=%s fallback=%s target_lang=%s chars=%d",
            event.segment_id,
            voice_mode,
            speaker_id,
            used_fallback,
            target_lang,
            len(text),
        )

        # Deadline-aware wait_for ceiling [P1-1-i2 + P1-5-i1].
        size_timeout = _TTS_SYNTH_TIMEOUT_BASE_S + _TTS_SYNTH_TIMEOUT_PER_CHAR_S * len(text)
        effective_timeout = min(size_timeout, max(0.2, remaining))

        _tts_in_flight += 1
        _tts_inflight_started[(event.segment_id, target_lang)] = time.monotonic()
        synth_start = time.monotonic()
        streamed_chunks: list[np.ndarray] = []
        first_chunk_ms: float | None = None
        try:
            from meeting_scribe.backends.tts_voices import studio_voice_for

            studio_voice = studio_voice_for(target_lang)
            voice_reference = None
            if voice_mode == "cloned":
                voice_reference = tts_backend.get_voice(speaker_id)
                # If no cached voice, fall back deterministically to the studio
                # voice for the target language (logged as cloned_fallback.studio).
                if voice_reference is None:
                    logger.debug(
                        "tts.cloned_fallback.studio seg=%s speaker=%s reason=no_cached_voice",
                        event.segment_id,
                        speaker_id,
                    )

            try:

                async def _run_stream():
                    nonlocal first_chunk_ms
                    # Stream FROM the backend (keeps the container's
                    # generator busy and gives us a real TTFA metric), but
                    # do NOT fan out to listeners mid-stream. Every chunk
                    # boundary was producing an audible click because each
                    # chunk was wrapped in its own WAV/RIFF header and
                    # played as a separate BufferSource on the client;
                    # consecutive buffers rarely join at a zero-crossing.
                    # Accumulate here and send ONE coherent WAV per segment
                    # below. Latency cost: the listener hears each segment
                    # ~synth_time later than the old behaviour, but given
                    # the 3.5 s ASR buffer that's ~1 s additional delay,
                    # which listeners tolerate far better than constant
                    # clicking.
                    async for chunk in tts_backend.synthesize_stream(
                        text=text,
                        language=target_lang,
                        voice_reference=voice_reference,
                        studio_voice=studio_voice,
                    ):
                        if len(chunk) == 0:
                            continue
                        if first_chunk_ms is None:
                            first_chunk_ms = (time.monotonic() - synth_start) * 1000.0
                        streamed_chunks.append(chunk)

                await asyncio.wait_for(_run_stream(), timeout=effective_timeout)
            except TimeoutError:
                logger.warning(
                    "TTS stream TIMEOUT seg=%s chars=%d timeout=%.2fs waited=%.2fs",
                    event.segment_id,
                    len(text),
                    effective_timeout,
                    time.monotonic() - synth_start,
                )
                metrics.tts_synth_timeouts += 1
                return
        finally:
            _tts_in_flight -= 1
            _tts_inflight_started.pop((event.segment_id, target_lang), None)

        audio = (
            np.concatenate(streamed_chunks) if streamed_chunks else np.zeros(0, dtype=np.float32)
        )

        # Post-synth deadline re-check [P1-1-i2]. Budget blown → drop.
        if time.monotonic() > deadline:
            logger.warning(
                "TTS post-synth deadline blown seg=%s over=%.2fs",
                event.segment_id,
                time.monotonic() - deadline,
            )
            metrics.tts_dropped_post_synth += 1
            return

        # Double-check meeting still active after synthesis.
        if not current_meeting:
            logger.info(
                "TTS post-synthesize: meeting stopped, dropping seg=%s",
                event.segment_id,
            )
            return

        if len(audio) == 0:
            logger.warning(
                "TTS synthesize returned empty audio: seg=%s mode=%s",
                event.segment_id,
                voice_mode,
            )
            continue

        # === Segment-level histograms (ONCE per segment, BEFORE fan-out) ===
        # [P1-5-i2]. This is the ONLY place lag metrics are recorded.
        synth_ms = (time.monotonic() - synth_start) * 1000.0
        metrics.tts_synth_ms.append(synth_ms)
        metrics.tts_synth_ms_p95 = _percentile(sorted(metrics.tts_synth_ms), 0.95)
        _record_segment_lag(event, time.monotonic())
        metrics.tts_delivered += 1
        metrics.last_delivery_at = time.monotonic()

        # Save TTS output to disk (for replay/download).
        if voice_mode in ("studio", "cloned") and current_meeting:
            tts_dir = storage._meeting_dir(current_meeting.meeting_id) / "tts"
            tts_dir.mkdir(exist_ok=True)
            import soundfile as _sf

            # Multi-target fan-out writes `{segment_id}.{target_lang}.wav`
            # so two targets for the same segment don't collide. Legacy
            # path stays `{segment_id}.wav` so replay/exports keep working.
            wav_name = (
                f"{event.segment_id}.{target_lang}.wav"
                if _MULTI_TARGET_ENABLED
                else f"{event.segment_id}.wav"
            )
            _sf.write(str(tts_dir / wav_name), audio, 24000)

        # Single-shot fan-out: one coherent WAV per segment, no chunk
        # joins on the client side. The per-chunk call that used to live
        # inside _run_stream was removed 2026-04-15 to kill click
        # artifacts — if this call ever goes missing the listener gets
        # nothing, so guard it with an explicit comment.
        if _audio_out_clients:
            await _send_audio_to_listeners(audio, target_lang, voice_mode)

        logger.info(
            "TTS streamed: seg=%s mode=%s chunks=%d audio_samples=%d ttfa_ms=%.0f total_ms=%.0f",
            event.segment_id,
            voice_mode,
            len(streamed_chunks),
            len(audio),
            first_chunk_ms or 0.0,
            synth_ms,
        )


def _peer_str(ws: WebSocket) -> str:
    """Best-effort `host:port` string for a WebSocket client, for logs."""
    try:
        return f"{ws.client.host}:{ws.client.port}" if ws.client else "?"
    except Exception:
        return "?"


def _build_riff_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Build a complete RIFF WAV (16-bit PCM, mono) for float32 audio in [-1, 1].

    Centralizes the struct.pack header construction that used to be
    duplicated between `_send_passthrough_audio` and
    `_send_audio_to_listeners`. Zero behavior change for legacy clients.
    """
    import struct as _struct

    import numpy as np

    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    data_bytes = pcm.tobytes()
    header = _struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(data_bytes),
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        1,  # mono
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        len(data_bytes),
    )
    return header + data_bytes


def _buffer_pending_audio(pref: ClientSession, audio: np.ndarray, source_rate: int) -> None:
    """Queue audio for a listener still mid-format-negotiation, bounded.

    Capped at ``_AUDIO_FORMAT_PENDING_CAP_S`` seconds per listener so a
    stuck handshake cannot balloon memory. When the cap is exceeded, the
    new item is silently dropped — by the time this fires the grace
    period has been running for well under 1 s and the worst-case loss
    is tiny.
    """
    import numpy as np  # noqa: F401  (numpy already used above; keep import local)

    total = sum(len(a) / max(sr, 1) for a, sr in pref.pending_audio) + len(audio) / max(
        source_rate, 1
    )
    if total > _AUDIO_FORMAT_PENDING_CAP_S:
        return
    pref.pending_audio.append((audio, source_rate))


async def _deliver_audio_to_listener(
    ws: WebSocket,
    pref: ClientSession,
    audio: np.ndarray,
    source_rate: int,
    wav_cache: dict | None,
) -> bool:
    """Encode + send one segment of audio to one listener in their format.

    Dispatches on ``pref.audio_format``:
      - ``None`` during grace → buffer into `pref.pending_audio`
      - ``None`` after grace  → default to ``"wav-pcm"`` and continue
      - ``"wav-pcm"``          → send one RIFF WAV binary frame, no prefix
      - ``"mse-fmp4-aac"``     → lazy-create the encoder (sending the init
                                 frame prefixed with ``b'\\x49'``), then
                                 feed `audio` and send any resulting
                                 fragment prefixed with ``b'\\x46'``.

    Returns True on success OR when the listener is in a valid held
    state (grace buffer, empty below-threshold MSE return). Returns
    False only when the WS send raised — caller discards the listener.
    """
    import numpy as np  # noqa: F401

    fmt = pref.audio_format
    if fmt is None:
        now = time.monotonic()
        if now > pref.grace_deadline:
            pref.audio_format = "wav-pcm"
            fmt = "wav-pcm"
            logger.info(
                "Audio-out format: defaulted to wav-pcm for %s (no set_format received)",
                _peer_str(ws),
            )
        else:
            # Still in grace; buffer and flush when set_format arrives.
            _buffer_pending_audio(pref, audio, source_rate)
            return True

    if fmt == "wav-pcm":
        # Reuse one cached WAV across every wav-pcm listener in this
        # fan-out call. Cache key covers the source rate because
        # passthrough (16 kHz) and TTS (24 kHz) coexist.
        cache_key = ("wav", source_rate, id(audio))
        wav_bytes: bytes | None = None
        if wav_cache is not None:
            wav_bytes = wav_cache.get(cache_key)
        if wav_bytes is None:
            wav_bytes = _build_riff_wav(audio, source_rate)
            if wav_cache is not None:
                wav_cache[cache_key] = wav_bytes
        try:
            await ws.send_bytes(wav_bytes)
            metrics.listener_deliveries += 1
            return True
        except Exception as e:
            logger.warning("audio_out wav send failed peer=%s: %s", _peer_str(ws), e)
            metrics.listener_send_failed += 1
            return False

    if fmt == "mse-fmp4-aac":
        from meeting_scribe.backends.mse_encoder import Fmp4AacEncoder

        # Lazy encoder creation on first audio delivery. The init frame
        # is sent once, immediately before the first media fragment
        # for this listener. After this point pref.mse_encoder stays
        # live until WS disconnect.
        if pref.mse_encoder is None:
            try:
                pref.mse_encoder = Fmp4AacEncoder()
            except Exception as e:
                logger.warning(
                    "mse encoder construction failed peer=%s: %s",
                    _peer_str(ws),
                    e,
                )
                return False
            logger.info("mse encoder lazy-created for %s", _peer_str(ws))
            try:
                await ws.send_bytes(b"\x49" + pref.mse_encoder.init_segment())
            except Exception as e:
                logger.warning("mse init send failed peer=%s: %s", _peer_str(ws), e)
                return False

        # Encode. Empty return is normal below-threshold accumulation —
        # NOT a fault. Do not log, do not recreate the encoder.
        try:
            fragment = pref.mse_encoder.encode(audio, source_rate)
        except Exception as e:
            logger.warning("mse encode failed peer=%s: %s", _peer_str(ws), e)
            return False
        pref.bytes_in_since_last_emit += len(audio)
        if not fragment:
            # Accumulating — still a successful "delivery" from the
            # listener's perspective. Stuck-detection is a separate
            # concern handled by the health check below.
            return True
        try:
            await ws.send_bytes(b"\x46" + fragment)
            metrics.listener_deliveries += 1
            pref.last_fragment_at = time.monotonic()
            pref.bytes_in_since_last_emit = 0
            return True
        except Exception as e:
            logger.warning("mse fragment send failed peer=%s: %s", _peer_str(ws), e)
            metrics.listener_send_failed += 1
            return False

    # Unknown format — impossible in practice (validated in set_format
    # handler) but fail closed.
    logger.warning("audio_out deliver: unknown format %r for %s", fmt, _peer_str(ws))
    return False


async def _flush_pending_audio(websocket: WebSocket, pref: ClientSession) -> None:
    """Drain the format-negotiation grace-window buffer for one listener."""
    pending = pref.pending_audio
    pref.pending_audio = []
    if not pending:
        return
    for audio, source_rate in pending:
        ok = await _deliver_audio_to_listener(websocket, pref, audio, source_rate, None)
        if not ok:
            break


async def _send_passthrough_audio(audio: np.ndarray, source_language: str) -> None:
    """Send original audio to 'full' interpretation clients whose preferred language matches.

    Only sent to clients in 'full' mode where preferred_language == source_language.
    Translation-only clients never receive pass-through audio.
    """
    source_norm = _norm_lang(source_language)
    recipients = [
        ws
        for ws in _audio_out_clients
        if _audio_out_prefs.get(ws)
        and _audio_out_prefs[ws].interpretation_mode == "full"
        and _norm_lang(_audio_out_prefs[ws].preferred_language) == source_norm
    ]
    if not recipients:
        return

    wav_cache: dict = {}
    sent = 0
    dead: list[WebSocket] = []
    for ws in recipients:
        pref = _audio_out_prefs.get(ws)
        if pref is None:
            continue
        ok = await _deliver_audio_to_listener(ws, pref, audio, 16000, wav_cache)
        if ok:
            sent += 1
        else:
            dead.append(ws)
    for ws in dead:
        _audio_out_clients.discard(ws)
        _audio_out_prefs.pop(ws, None)
        metrics.listener_removed_on_send_error += 1

    # One INFO line per delivery so we can validate "full" mode end-to-end
    # from the journal. Rate is bounded by the final-ASR-segment cadence
    # (~1/sec at most), so this won't flood.
    if sent:
        logger.info(
            "passthrough sent: source_lang=%s listeners=%d",
            source_norm,
            sent,
        )


async def _send_audio_to_listeners(
    audio: np.ndarray,
    target_language: str,
    voice_mode: str = "cloned",
) -> int:
    """Send synthesized audio to audio-out clients in their negotiated format.

    Filters listeners by:
      - preferred_language matching target_language (normalized — both
        sides pass through ``_norm_lang`` so ``en-US`` matches ``en``)
      - voice_mode matching the synthesized voice mode

    Returns the number of listeners the audio was successfully sent to.
    Each matching listener receives the audio in the format it negotiated
    via `set_format` — wav-pcm (one RIFF WAV frame) or mse-fmp4-aac
    (init frame on first delivery, fragments thereafter). See
    `_deliver_audio_to_listener` for the dispatch logic.
    """
    target_norm = _norm_lang(target_language)

    wav_cache: dict = {}
    sent = 0
    dead: list[WebSocket] = []
    for ws in list(_audio_out_clients):
        pref = _audio_out_prefs.get(ws)
        if pref is None:
            continue
        pref_lang = _norm_lang(pref.preferred_language)
        if pref_lang and pref_lang != target_norm:
            continue
        client_voice = getattr(pref, "voice_mode", "studio")
        if client_voice != voice_mode:
            continue
        send_start = time.monotonic()
        ok = await _deliver_audio_to_listener(ws, pref, audio, 24000, wav_cache)
        if ok:
            sent += 1
            metrics.listener_send_ms.append((time.monotonic() - send_start) * 1000.0)
        else:
            dead.append(ws)
    for ws in dead:
        _audio_out_clients.discard(ws)
        _audio_out_prefs.pop(ws, None)
        metrics.listener_removed_on_send_error += 1
    return sent


def _notify_systemd(message: str) -> None:
    """Send a ``sd_notify(3)`` message to the service manager if we are
    running under ``Type=notify``. No-op (and no dependency on
    ``systemd-python``/``sdnotify``) when ``$NOTIFY_SOCKET`` is unset,
    e.g. in foreground dev runs.

    Implements the datagram protocol directly via the Unix socket so
    there's nothing to install: the whole contract is "connect to the
    socket at ``$NOTIFY_SOCKET`` and send one UTF-8 message".
    """
    import socket as _socket

    sock_path = os.environ.get("NOTIFY_SOCKET")
    if not sock_path:
        return
    try:
        addr: bytes | str
        if sock_path.startswith("@"):
            # Abstract Linux socket.
            addr = "\0" + sock_path[1:]
        else:
            addr = sock_path
        s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_DGRAM | _socket.SOCK_CLOEXEC)
        try:
            s.sendto(message.encode("utf-8"), addr)
        finally:
            s.close()
    except OSError as e:
        logger.warning("sd_notify(%r) failed: %r", message, e)


def _detect_management_ip_via_nm() -> str | None:
    """Query NetworkManager for the IPv4 address of the active wired connection.

    Positive selection: only considers ``802-3-ethernet`` connections, so
    VPN tunnels (``wg0``, ``tun0``), wireless hotspots, docker bridges,
    USB tethering, and mobile broadband are never selected.

    Returns the IP string if exactly one active ethernet connection with an
    IPv4 address is found, otherwise ``None`` (ambiguous or unavailable).
    """
    try:
        cons = subprocess.run(
            ["nmcli", "-t", "-f", "TYPE,DEVICE", "connection", "show", "--active"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None

    # Filter to wired ethernet connections.
    ethernet_devices: list[str] = []
    for line in cons.splitlines():
        parts = line.split(":")
        if len(parts) >= 2 and parts[0] == "802-3-ethernet" and parts[1]:
            ethernet_devices.append(parts[1])

    if len(ethernet_devices) != 1:
        # Zero or multiple wired connections — refuse to guess.
        if ethernet_devices:
            logger.info(
                "NM: %d active ethernet connections — refusing to guess management IP",
                len(ethernet_devices),
            )
        return None

    device = ethernet_devices[0]
    try:
        dev_info = subprocess.run(
            ["nmcli", "-t", "-f", "IP4.ADDRESS", "device", "show", device],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None

    # Parse "IP4.ADDRESS[1]:192.168.8.153/24" → "192.168.8.153"
    for line in dev_info.splitlines():
        if line.startswith("IP4.ADDRESS"):
            _, _, addr = line.partition(":")
            addr = addr.strip()
            if "/" in addr:
                addr = addr.split("/")[0]
            if addr:
                logger.info("NM: management IP from ethernet device %s: %s", device, addr)
                return addr

    return None


def _detect_management_ip() -> str:
    """Detect the management IPv4 address via a three-tier cascade.

    1. ``SCRIBE_MANAGEMENT_IP`` env override (tests, unusual layouts).
    2. ``ip -4 route get 1.1.1.1`` with retry budget — the kernel's
       preferred source address for outbound traffic.
    3. NetworkManager positive selection — the IPv4 of the sole active
       ``802-3-ethernet`` connection.
    4. Fallback to ``127.0.0.1`` — admin listener is localhost-only
       (degraded mode). The guest hotspot portal is unaffected.

    Never raises. Returns ``"127.0.0.1"`` as the final fallback so the
    server always starts, even without a network.
    """
    override = os.environ.get("SCRIBE_MANAGEMENT_IP", "").strip()
    if override:
        return override

    try:
        budget = max(1, int(os.environ.get("SCRIBE_MGMT_IP_WAIT", "30")))
    except ValueError:
        budget = 30

    # Tier 2: ip route get with retry budget.
    for attempt in range(1, budget + 1):
        try:
            out = subprocess.run(
                ["ip", "-4", "route", "get", "1.1.1.1"],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            ).stdout
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            if attempt < budget:
                logger.info(
                    "waiting for default route (attempt %d/%d): %s",
                    attempt,
                    budget,
                    e,
                )
                time.sleep(1)
                continue
            logger.info("ip route get exhausted after %d attempts: %s", budget, e)
            break

        tokens = out.split()
        if "src" in tokens:
            if attempt > 1:
                logger.info("management IP detected after %d attempts", attempt)
            return tokens[tokens.index("src") + 1]
        if attempt < budget:
            logger.info(
                "no 'src' field in ip route get output yet (attempt %d/%d)",
                attempt,
                budget,
            )
            time.sleep(1)
            continue
        logger.info("no 'src' field in ip route get output: %r", out)
        break

    # Tier 3: NetworkManager ethernet lookup.
    nm_ip = _detect_management_ip_via_nm()
    if nm_ip:
        return nm_ip

    # Tier 4: localhost fallback — degraded mode.
    logger.warning(
        "no management IP detected via route or NetworkManager — "
        "admin listener will be localhost-only (degraded mode)"
    )
    return "127.0.0.1"


def _make_tcp_socket(host: str, port: int, freebind: bool = False) -> socket.socket:
    """Create and bind a TCP listening socket.

    When ``freebind=True`` sets ``IP_FREEBIND`` so the bind succeeds even
    if the target IP isn't yet assigned to any local interface. We use
    that for the guest listener on the hotspot AP IP (``10.42.0.1``)
    which only exists after ``nmcli`` brings the AP up.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if freebind:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_FREEBIND, 1)
    sock.bind((host, port))
    sock.listen(128)
    sock.setblocking(False)
    return sock


class _NoSignalServer(uvicorn.Server):
    """uvicorn.Server variant that skips its own SIGINT/SIGTERM handlers.

    When two Servers run in one process, only the last installed signal
    handler wins. We disable per-Server handlers and install a single
    outer handler in ``main()`` that sets ``should_exit = True`` on both
    instances together, so Ctrl-C cleanly drains both listeners.
    """

    def install_signal_handlers(self) -> None:
        return None


async def _wait_for_management_ip(
    admin_server: _NoSignalServer,
    port: int,
    poll_interval: int = 15,
) -> None:
    """Background task: hot-add an admin LAN socket when the network appears.

    Launched only when ``_detect_management_ip()`` returned ``127.0.0.1``
    at startup (degraded mode). Polls the same detection cascade every
    ``poll_interval`` seconds with a minimal inner budget
    (``SCRIBE_MGMT_IP_WAIT=1``) to avoid blocking the event loop.

    When a non-loopback IP is found, creates a new TCP listener socket,
    wires it into the running admin uvicorn server using the same protocol
    factory pattern as uvicorn's own ``startup()``, and exits (one-shot).
    """

    def _detect_quick() -> str:
        """Run the detection cascade with a 1-attempt budget."""
        saved = os.environ.get("SCRIBE_MGMT_IP_WAIT")
        try:
            os.environ["SCRIBE_MGMT_IP_WAIT"] = "1"
            return _detect_management_ip()
        finally:
            if saved is None:
                os.environ.pop("SCRIBE_MGMT_IP_WAIT", None)
            else:
                os.environ["SCRIBE_MGMT_IP_WAIT"] = saved

    try:
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(poll_interval)

            # Run detection in a thread so subprocess calls don't block
            # the event loop (worst case: ip 3s + 2× nmcli 5s = ~13s).
            ip = await loop.run_in_executor(None, _detect_quick)
            if ip == "127.0.0.1":
                continue

            # Found a real management IP — hot-add the admin socket.
            try:
                sock = _make_tcp_socket(ip, port)
            except OSError as e:
                logger.warning(
                    "LAN admin socket bind failed for %s:%d: %s — will retry",
                    ip,
                    port,
                    e,
                )
                continue

            config = admin_server.config
            # uvicorn's http_protocol_class is typed as `type[Protocol]` on
            # the uvicorn.Config surface but the concrete call site takes
            # config/server_state/app_state kwargs (see uvicorn.protocols.*).
            # mypy can't see through the Protocol aliasing.
            server = await loop.create_server(
                lambda: config.http_protocol_class(  # type: ignore[call-arg]
                    config=config,
                    server_state=admin_server.server_state,
                    app_state=admin_server.lifespan.state,
                ),
                sock=sock,
                ssl=config.ssl,
                backlog=config.backlog,
            )
            admin_server.servers.append(server)
            logger.info("LAN admin listener recovered: https://%s:%d", ip, port)
            return

    except asyncio.CancelledError:
        return


async def _serve_dual(
    admin_server: _NoSignalServer,
    admin_sockets: list,
    guest_server: _NoSignalServer,
    guest_sockets: list,
    *,
    deferred_admin_bind: tuple[int, int] | None = None,
) -> None:
    """Run admin + guest uvicorn servers sharing one FastAPI app.

    Startup order matters: admin runs the FastAPI lifespan (model loads,
    backend wiring, WiFi regdomain checks). Guest has ``lifespan="off"``
    and must only start accepting connections **after** admin's startup
    is complete, otherwise a fast phone hitting the guest portal could
    land in an app with uninitialised globals.

    When ``deferred_admin_bind`` is set (``(port, poll_interval)``), a
    background task polls for the management IP and hot-adds an admin LAN
    socket when the network appears. This handles the degraded-start case
    where no LAN IP was available at boot.
    """
    import signal

    loop = asyncio.get_running_loop()

    def _request_shutdown() -> None:
        admin_server.should_exit = True
        guest_server.should_exit = True

    def _request_config_reload() -> None:
        """Re-read runtime-config from disk.

        Bound to SIGHUP so ``meeting-scribe config reload`` (and the
        Phase 7 rollback procedure) can flip ``translate_url`` /
        ``slide_translate_url`` / ``slide_use_json_schema`` live.  The
        translate backend re-reads on every request, so no restart is
        needed after the reload fires.
        """
        from meeting_scribe import runtime_config

        try:
            runtime_config.reload_from_disk()
            logger.info(
                "runtime-config reloaded on SIGHUP: %s", runtime_config.instance().as_dict()
            )
        except Exception:
            logger.exception("runtime-config reload failed")

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _request_shutdown)
    loop.add_signal_handler(signal.SIGHUP, _request_config_reload)

    admin_task = asyncio.create_task(
        admin_server.serve(sockets=admin_sockets),
        name="admin-uvicorn",
    )

    while not admin_server.started:
        if admin_task.done():
            # Admin died during startup — surface the exception.
            await admin_task
            return
        await asyncio.sleep(0.05)

    # Launch LAN recovery task if we started in degraded mode.
    recovery_task: asyncio.Task | None = None
    if deferred_admin_bind is not None:
        port, interval = deferred_admin_bind
        recovery_task = asyncio.create_task(
            _wait_for_management_ip(admin_server, port, interval),
            name="lan-recovery",
        )

    guest_task = asyncio.create_task(
        guest_server.serve(sockets=guest_sockets),
        name="guest-uvicorn",
    )

    try:
        await asyncio.gather(admin_task, guest_task)
    finally:
        # If one server crashed, make sure the other drains too.
        admin_server.should_exit = True
        guest_server.should_exit = True
        if recovery_task is not None and not recovery_task.done():
            recovery_task.cancel()
            try:
                await recovery_task
            except asyncio.CancelledError:
                pass


def main() -> None:
    """Entry point — run admin (HTTPS) + guest (HTTP) listeners in one process.

    Binding:
      - Admin HTTPS on TWO sockets (or ONE in degraded mode):
          * ``127.0.0.1:<port>``  — local tools (tests, autosre TUI,
            healthcheck script, ``meeting-scribe status``).
          * ``<management_ip>:<port>`` — LAN admin access. Omitted when
            no management IP is detected (degraded mode); a background
            task hot-adds it when the network appears.
        Never ``0.0.0.0``: the admin socket must not exist on the
        hotspot (wlan0) interface under any circumstances.
      - Guest HTTP on ``<AP_IP>:<guest_port>`` (``IP_FREEBIND``). Pre-
        binds the hotspot IP before the AP is up so a single bind
        survives meeting rotations. Unaffected by degraded mode.

    Both servers share the same ``app`` object, so in-process globals
    (``current_meeting``, ``_audio_out_clients``, ``_tts_queue``,
    ``ws_connections``) are shared. The FastAPI ``hotspot_guard``
    middleware + per-WebSocket ``_is_guest_scope`` checks keep guest
    traffic restricted to the live-translation / TTS paths regardless of
    which listener accepted the connection.
    """
    server_config = ServerConfig.from_env()

    management_ip = _detect_management_ip()
    is_degraded = (
        management_ip == "127.0.0.1" and not os.environ.get("SCRIBE_MANAGEMENT_IP", "").strip()
    )
    if is_degraded:
        logger.warning(
            "no LAN IP detected — admin listener is localhost-only; "
            "will recover when network appears"
        )
    else:
        logger.info("management IP detected: %s", management_ip)

    project_root = Path(__file__).resolve().parent.parent.parent
    ssl_key = project_root / "certs" / "key.pem"
    ssl_cert = project_root / "certs" / "cert.pem"
    if not (ssl_key.exists() and ssl_cert.exists()):
        raise RuntimeError(
            f"admin TLS certs missing: expected {ssl_key} and {ssl_cert}. "
            "Run `meeting-scribe setup` to generate a self-signed pair."
        )

    # Admin: pre-bind two sockets (loopback + management IP). Both serve
    # HTTPS via one uvicorn.Server so there's only one lifespan and one
    # set of in-process globals.
    admin_hosts: list[str] = ["127.0.0.1"]
    if management_ip != "127.0.0.1":
        admin_hosts.append(management_ip)
    admin_sockets = [_make_tcp_socket(h, server_config.port) for h in admin_hosts]

    admin_config = uvicorn.Config(
        app,
        log_level="info",
        ssl_keyfile=str(ssl_key),
        ssl_certfile=str(ssl_cert),
        lifespan="on",
        # Explicit WS keepalive. Bumped 2026-04-15 from 10/10 → 20/45 after
        # observing the audio-out WS getting closed mid-meeting by the
        # periodic deep-health probes (every 10–12 s) blocking the event
        # loop for ~2.5 s, which accumulated across cycles and pushed the
        # pong round-trip past the 10 s timeout. 20 s send interval × 45 s
        # receive timeout gives ~2× headroom for any single stall and still
        # detects a truly dead peer within ~65 s. Real fix for the stalls
        # is upstream in _DEEP_HEALTH_TTL and the shared httpx client, but
        # this is the direct "don't kill the WS" lever.
        ws_ping_interval=20,
        ws_ping_timeout=45,
    )
    admin_server = _NoSignalServer(admin_config)

    # Guest: plain HTTP on two sockets.
    #   * ``<AP_IP>:<guest_port>`` via IP_FREEBIND — the real hotspot
    #     entry point. FREEBIND lets us pre-bind before nmcli assigns
    #     10.42.0.1 to wlan0, so a single bind survives meeting rotations.
    #   * ``127.0.0.1:<guest_port>`` — keeps local tooling working
    #     (autosre TUI's portal probe, `curl http://127.0.0.1/`, tests).
    #     Middleware + per-WS ``_is_guest_scope`` still treat every HTTP
    #     request as guest-scope, so loopback cannot reach admin paths
    #     via this socket.
    # Port <1024 requires CAP_NET_BIND_SERVICE on the interpreter; the
    # meeting-scribe CLI grants that via `setcap` before spawn.
    guest_sockets = [
        _make_tcp_socket("127.0.0.1", server_config.guest_port),
        _make_tcp_socket(AP_IP, server_config.guest_port, freebind=True),
    ]
    guest_config = uvicorn.Config(
        app,
        log_level="info",
        lifespan="off",
        # See admin_config above for the 20/45 rationale — same stall
        # budget applies, and the audio-out WS that hotspot guests open
        # is the one actually exposed to this risk (admin WS is just
        # local operator / dev).
        ws_ping_interval=20,
        ws_ping_timeout=45,
    )
    guest_server = _NoSignalServer(guest_config)

    logger.info(
        "starting listeners: admin=https://{127.0.0.1,%s}:%d, guest=http://{127.0.0.1,%s}:%d",
        management_ip,
        server_config.port,
        AP_IP,
        server_config.guest_port,
    )

    deferred = (server_config.port, 15) if is_degraded else None

    try:
        asyncio.run(
            _serve_dual(
                admin_server,
                admin_sockets,
                guest_server,
                guest_sockets,
                deferred_admin_bind=deferred,
            )
        )
    finally:
        for s in admin_sockets:
            with suppress(OSError):
                s.close()
        for s in guest_sockets:
            with suppress(OSError):
                s.close()


if __name__ == "__main__":
    main()
