"""Canonical home for cross-module mutable runtime state.

This module is **declarations only**. It never constructs backends,
queues, stores, or singletons at import time. Real objects are built
in ``server.lifespan`` (or in ``_init_*`` helpers it calls) and
assigned to module attributes here:

    from meeting_scribe.runtime import state
    state.asr_backend = await _init_asr()

Reads and writes from any module — ``server.py``, route modules,
WebSocket handlers, the hotspot subsystem — go through this module
to avoid sideways cycles and to keep there one source of truth for
shared mutable state.

Access pattern:

    # CORRECT — module-attribute access; sees latest mutations.
    from meeting_scribe.runtime import state
    if state.asr_backend is None:
        ...

    # WRONG — captures the value at import time, never sees mutations.
    from meeting_scribe.runtime.state import asr_backend  # do not do this

The ``Any`` type is used liberally below to avoid import cycles back
into ``server.py`` for classes defined there (``Metrics``,
``ClientSession``, etc.). Future refactors can tighten these annotations
once the relevant classes move out of ``server.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import WebSocket

    from meeting_scribe.backends.base import ASRBackend, TranslateBackend
    from meeting_scribe.config import ServerConfig
    from meeting_scribe.models import DetectedSpeaker, MeetingMeta
    from meeting_scribe.storage import AudioWriter, AudioWriterProcess


# ── Backends ───────────────────────────────────────────────────
# All backend handles are populated in ``server.lifespan``. They are
# ``None`` before startup and after shutdown. Handlers should treat
# ``None`` as "backend unavailable" and respond accordingly.
asr_backend: ASRBackend | None = None
translate_backend: TranslateBackend | None = None
furigana_backend: Any = None  # FuriganaBackend — initialized lazily
tts_backend: Any = None  # Qwen3TTSBackend (optional)
diarize_backend: Any = None  # SortformerBackend (optional)
name_extractor: Any = None  # LLMNameExtractor (optional)


# ── Active meeting + WebSocket connections ────────────────────
# A meeting transitions through ``state.current_meeting`` for its
# lifetime; ``None`` means no recording is active. ``ws_connections``
# holds the live transcript broadcast set; ``_client_prefs`` carries
# per-connection preferences negotiated over the WebSocket. The
# audio-out side is symmetric: ``_audio_out_clients`` are listeners
# subscribed to interpreted audio, with ``_audio_out_prefs`` for
# their negotiated wire format.
current_meeting: MeetingMeta | None = None
ws_connections: set[WebSocket] = set()
_client_prefs: dict[WebSocket, Any] = {}  # dict[WebSocket, ClientSession]
_audio_out_clients: set[WebSocket] = set()
_audio_out_prefs: dict[WebSocket, Any] = {}  # dict[WebSocket, ClientSession]
audio_writer: AudioWriter | AudioWriterProcess | None = None


def current_recording_pcm_offset() -> int:
    """Current byte offset of the live recording.pcm write head, or 0
    if no audio writer is open. W6a's recovery state machine uses
    this to bound replay windows after ASR recovers — see
    `meeting_scribe.backends.asr_vllm` for the consumer."""
    if audio_writer is None:
        return 0
    # AudioWriter and AudioWriterProcess both expose .current_offset.
    return getattr(audio_writer, "current_offset", 0)


meeting_start_time: float = 0.0  # monotonic time for audio alignment
detected_speakers: list[DetectedSpeaker] = []  # per-meeting speaker state
speaker_verifier: Any = None  # SpeakerVerifier instance


# ── Background tasks + caches ─────────────────────────────────
# Hold references to fire-and-forget asyncio tasks so they aren't
# garbage-collected before completing. The pulse/catchup/eager-summary
# tasks are owned by an active meeting and torn down on stop.
import asyncio as _asyncio

_background_tasks: set[_asyncio.Task] = set()
_speaker_pulse_task: _asyncio.Task | None = None
_speaker_catchup_task: _asyncio.Task | None = None
_eager_summary_task: _asyncio.Task | None = None
_eager_summary_cache: dict | None = None  # last draft summary from eager loop
_eager_summary_event_count: int = 0  # event count when draft was generated

# Telemetry for the eager-summary loop. Updated at every iteration of
# ``runtime.meeting_loops._eager_summary_loop`` so we can prove the
# loop is alive (last_start_at recent) AND making progress
# (last_success_at within draft cadence). Exposed via
# ``GET /api/admin/eager-summary-status``. NEVER stores raw exception
# text — only an enum classifier from ``server_support.summary_status``.
from dataclasses import dataclass as _dc


@_dc
class EagerSummaryMetrics:
    last_start_at: float | None = None
    last_success_at: float | None = None
    last_error_code: str | None = None  # SummaryErrorCode value, never raw
    last_skipped_reason: str | None = None  # "warmup" | "in_flight" | "no_growth"
    in_flight: bool = False
    draft_event_count_at_last_run: int = 0
    runs_total: int = 0
    errors_total: int = 0


_eager_summary_metrics: EagerSummaryMetrics = EagerSummaryMetrics()

# Lazy-initialized singleton ThreadPoolExecutor for the speaker-embedding
# worker pool, created on first use inside _process_event and reused for
# the lifetime of the server.
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor

_speaker_executor_singleton: _ThreadPoolExecutor | None = None


# ── Speaker enrollment + room layout drafts ────────────────────
# ``enrollment_store`` and ``_terminal_*`` singletons are created at
# server-module import time (still in server.py — see note below) and
# assigned here so other modules can reach them. ``state.py`` itself
# does no construction.
enrollment_store: Any = None  # SpeakerEnrollmentStore — assigned by server.py at import
_draft_layouts: dict[str, Any] = {}  # dict[str, RoomLayout]
_draft_layout_access: dict[str, float] = {}


# ── Embedded admin terminal ────────────────────────────────────
# All four singletons are constructed at server-module import time
# because ``register_bootstrap_routes(...)`` and
# ``register_terminal_routes(...)`` (called at module level) capture
# them as constructor arguments. Construction stays in server.py;
# the assignments here are for cross-module access only.
_terminal_admin_secret: Any = None  # AdminSecretStore
_terminal_cookie_signer: Any = None  # CookieSigner
_terminal_ticket_store: Any = None  # TicketStore
_terminal_registry: Any = None  # ActiveTerminals


# ── Slide translation worker ───────────────────────────────────
# Initialized in lifespan if the worker container image is available.
slide_job_runner: Any = None  # SlideJobRunner
slides_enabled: bool = False


# ── Server-wide config singleton ───────────────────────────────
# Populated in ``server.lifespan`` from environment + persisted
# overrides. The bulk migration uses the ``\bconfig\.`` regex so it
# only touches attribute accesses (``config.foo``); string literals
# like ``~/.config/`` and module imports like
# ``meeting_scribe.config`` stay intact.
config: ServerConfig | None = None


# ── Pseudo-speaker tracking (proximity-based fallback) ────────
# Time-proximity pseudo-speaker tracking — fallback when
# diarization hasn't produced a result yet. Segments within 3
# seconds of the previous final event's end get the same pseudo-
# cluster; bigger gaps allocate a new one. Pseudo IDs start at
# 100 so they don't collide with real diarization IDs (small
# integers starting at 0 or 1). All three reset to defaults at
# the start of each meeting.
_last_pseudo_cluster_id: int = 0
_last_pseudo_end_ms: int = 0
_next_pseudo_cluster_id: int = 100


# ── Pending speaker events (catchup queue) ─────────────────────
# Events waiting for diarization results to catch up. Keyed by
# segment_id. An event lands here when ``_process_event`` runs
# BEFORE diarization has produced results for that time range
# (typically because diarization buffers 4 s of audio vs ASR's
# faster finalization). The catch-up loop polls for results and
# re-broadcasts with cluster_ids attached.
from collections import OrderedDict as _OrderedDict

_pending_speaker_events: _OrderedDict = _OrderedDict()  # OrderedDict[str, TranscriptEvent]

# Retroactive updates are also applied to the audio writer position, so
# we track the monotonic time the event was first queued.
_pending_speaker_timestamps: dict[str, float] = {}


# ── Storage / pipelines / metrics / refinement worker ──────────
# ``MeetingStorage`` and ``Resampler`` are constructed in
# ``server.lifespan``. ``TranslationQueue`` is constructed lazily on
# the first translate call (so a server start with the translate
# backend down doesn't fail). ``Metrics`` is reset per meeting in
# ``/api/meeting/start`` — it stays alive between meetings, but its
# counters are zeroed. ``refinement_worker`` is started in
# ``meeting/start`` / ``meeting/{id}/resume`` and torn down in
# ``meeting/stop`` / ``meeting/cancel``.
storage: Any = None  # MeetingStorage
resampler: Any = None  # Resampler
translation_queue: Any = None  # TranslationQueue | None
# Constructed at import time so tests + every consumer can call
# ``state.metrics.reset()`` without coupling to server.py being imported
# first. The ``Metrics`` constructor is side-effect-free (just resets a
# bunch of histograms), so import-time construction is safe.
from meeting_scribe.runtime.metrics import Metrics as _Metrics

metrics: Any = _Metrics()
refinement_worker: Any = None  # RefinementWorker | None


# ── Silence watchdog signals ───────────────────────────────────
# ``last_audio_chunk_ts`` is bumped by ``ws.audio_input`` on every
# inbound audio frame; the silence watchdog reads it to detect a
# stalled browser-side audio WebSocket and emit a ``meeting_warning``
# broadcast so the UI can prompt a reconnect. ``silence_warn_sent``
# is the one-shot guard that prevents the warning from firing on
# every poll while the stall persists.
last_audio_chunk_ts: float = 0.0
silence_warn_sent: bool = False


# ── TTS pipeline runtime state + config ────────────────────────
# Mutable handles owned by the TTS worker pool but inspected by the
# metrics + health evaluator + status route. The pipeline-level
# config constants (queue size, worker count, container concurrency,
# expected synth budget) live here too so the metrics module can
# render them in /api/status without a server-module round-trip.
import os as _os

tts_queue: _asyncio.Queue | None = None
tts_semaphore: _asyncio.Semaphore | None = None
tts_backend_semaphore: _asyncio.Semaphore | None = None
tts_in_flight: int = 0
tts_inflight_started: dict[tuple[str, str], float] = {}
tts_worker_tasks: list[_asyncio.Task] = []

TTS_QUEUE_MAXSIZE: int = 4
TTS_WORKER_COUNT: int = 4
TTS_CONTAINER_MAX_CONCURRENCY: int = int(_os.environ.get("SCRIBE_TTS_CONTAINER_CONCURRENCY", "2"))
TTS_EXPECTED_SYNTH_DEFAULT_S: float = 2.0
