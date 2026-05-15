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
# Subset of ``ws_connections`` whose handshake carried a valid
# ``scribe_kiosk`` cookie. ``broadcast._broadcast_json`` consults this
# set to filter operator-only event types out of the kiosk fan-out.
# Membership is maintained by ``ws/view_broadcast.py``.
kiosk_ws_connections: set[WebSocket] = set()
# Subset of ``ws_connections`` that holds the AUDIO recorder side of
# the pipeline (``/api/ws`` in ``ws/audio_input.py``). The meeting
# reconciler's "stale recorder still here?" + "upgrade to recorder?"
# decisions read ``status.connections`` and the count must reflect
# recorder sockets ONLY - not view-only popouts or the HDMI kiosk
# mirror, which would otherwise wedge the admin tab in view-only
# mode forever (kiosk's WS never goes away).
recorder_ws_connections: set[WebSocket] = set()
_client_prefs: dict[WebSocket, Any] = {}  # dict[WebSocket, ClientSession]
_audio_out_clients: set[Any] = set()  # set[AudioListener] — WebSocket and BTSpeakerListener
_audio_out_prefs: dict[Any, Any] = {}  # dict[AudioListener, ClientSession]
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

# Server-side mic capture (audio_routing). Owned by the lifespan and
# the admin /api/admin/audio/route endpoint. ``server_mic`` is the
# live :class:`ServerMicCapture` (from
# ``meeting_scribe.audio.server_mic``) when a target node is
# configured + active, otherwise None. ``server_mic_active`` is the
# fast-path flag the WS audio-input handler reads on every binary
# frame to drop browser-mic data while server-side capture owns the
# ASR pipeline (operator chose "server mic replaces browser mic"
# semantics).
server_mic: Any = None  # ServerMicCapture | None
server_mic_active: bool = False
interpretation_buffer: Any = None  # InterpretationBuffer | None

# Soft input-mute. Runtime-only — NOT persisted across process boots and
# NOT carried across meeting boundaries. ``_start_meeting_locked`` resets
# this to False on every fresh meeting so a stale operator toggle from
# the previous meeting cannot silently suppress recording. When True,
# ``ws.audio_input`` and ``audio.server_mic`` drop captured frames before
# either the audio writer or the ASR forward — true privacy pause, not
# just a UI label.
mic_input_muted: bool = False

# Per-process identity stamped into Phase B progress sidecars. Allocated
# once at lifespan boot. ``storage.sweep_orphan_phase_b_sidecars`` uses
# this to distinguish "live in-flight from THIS process" from "in-flight
# stamped by a previous process that crashed" — orphans get rewritten to
# the interrupted-failure shape before any client can read them.
session_uuid: str = ""


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


# ── Per-boot signing subkey + session revocation ───────────────
# ``boot_session_id`` is regenerated every time the server starts. The
# CookieSigner's HMAC key is derived from ``(admin_secret,
# boot_session_id)`` via HKDF, so a server restart invalidates every
# previously-issued admin cookie — the strong logout-all guarantee for
# v1.0. See ``terminal/auth.py:derive_cookie_subkey``.
boot_session_id: bytes = b""

# Logout / re-auth revocation set. Maps ``session_id`` (16-byte hex from
# ``CookieSigner.issue``) → expiry epoch (when the cookie's max_age
# would have run out). Live for the lifetime of the boot only;
# ``boot_session_id`` rotation supersedes it across restarts.
_revoked_sessions: dict[str, float] = {}

# Active admin WebSockets per session. ``register_admin_ws`` (the async
# context manager in terminal/auth.py) populates this; logout looks up the
# logging-out session_id and closes every WS in its set; re-auth on the
# same browser does the same for the prior session before minting a fresh
# cookie.
_admin_ws_by_session: dict[str, set[Any]] = {}


# ── Slide translation worker ───────────────────────────────────
# Initialized in lifespan if the worker container image is available.
slide_job_runner: Any = None  # SlideJobRunner
slides_enabled: bool = False


# ── Server-wide config singleton ───────────────────────────────
# Populated in ``server.lifespan`` from environment + persisted
# overrides.
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


# ── Storage / pipelines / metrics ──────────────────────────────
# ``MeetingStorage`` and ``Resampler`` are constructed in
# ``server.lifespan``. ``TranslationQueue`` is constructed lazily on
# the first translate call (so a server start with the translate
# backend down doesn't fail). ``Metrics`` is reset per meeting in
# ``/api/meeting/start`` — it stays alive between meetings, but its
# counters are zeroed.
storage: Any = None  # MeetingStorage
resampler: Any = None  # Resampler
translation_queue: Any = None  # TranslationQueue | None
# Constructed at import time so tests + every consumer can call
# ``state.metrics.reset()`` without coupling to server.py being imported
# first. The ``Metrics`` constructor is side-effect-free (just resets a
# bunch of histograms), so import-time construction is safe.
from meeting_scribe.runtime.metrics import Metrics as _Metrics

metrics: Any = _Metrics()


# ── Silence watchdog signals ───────────────────────────────────
# ``last_audio_chunk_ts`` is bumped by ``ws.audio_input`` on every
# inbound audio frame regardless of content. Kept for backwards
# compatibility + the chunks-arrived metric; the silence watchdog now
# reads ``last_nonzero_audio_ts`` (peak-gated) below so the alarm fires
# on frames-of-zero as well as missing-frames. ``silence_warn_sent``
# is the one-shot guard that prevents the warning from firing on
# every poll while the stall persists.
last_audio_chunk_ts: float = 0.0
# Bumped by ``ws.audio_input`` only when the chunk has speech-like energy.
# Room-speaker interpretation uses this as the live quiet gate; ASR final
# timestamps alone can be misleading when watchdog flushes emit partial
# segments while someone is still talking.
last_speech_audio_ts: float = 0.0
# Bumped by ``ws.audio_input`` and ``audio.server_mic`` whenever a frame's
# peak crosses ``_AUDIO_LIVENESS_FLOOR`` (= 1e-4, well below the speech
# threshold). Distinct from ``last_speech_audio_ts``: this proves "the ADC
# is sampling" — captures ambient room noise, USB preamp DC offset, anything
# non-zero. The silence watchdog and the meeting-start mic probe consume
# this to detect "frames arriving but all-zero" failures that the older
# ``last_audio_chunk_ts`` heartbeat could not catch (it bumped on every
# frame regardless of content). Reset to 0.0 by the meeting-start probe to
# enforce a probe-local epoch so prior-session activity cannot mask a dead
# mic at the new meeting's start.
last_nonzero_audio_ts: float = 0.0
# Route health summary set by ``audio_routing.reconcile_audio_routing()``.
# One of ``"ok"`` (no issue), ``"ambiguous"`` (multiple sources match the
# persisted stable_id and discriminator can't narrow), ``"unresolved"``
# (device not currently present — likely transient disconnect), or
# ``"capture_failed"`` (routing resolved but capture process refused to
# start). Phase-5 meeting-start preflight rejects all non-"ok" values.
audio_route_status: str = "ok"
# Active admin notifications surfaced via the ``.meeting-banner`` SPA
# component. Keyed by ``kind`` (e.g. ``"mic_rebound"``, ``"mic_ambiguous"``,
# ``"mic_unresolved"``, ``"mic_capture_failed"``). Persisted to settings via
# ``state_support.admin_notifications`` so reloads survive. ``/api/status``
# emits only un-dismissed entries.
pending_admin_notifications: dict[str, dict[str, Any]] = {}
# Local room-sink TTS playback guard. ``audio.local_sink`` extends this
# while it writes GB10/Poly speaker output; ``audio.server_mic`` drops
# captured frames inside the window so the room mic does not feed TTS
# audio back into ASR/translation.
local_tts_playback_until: float = 0.0
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

# Token queue gating entry to ``Qwen3TTSBackend.synthesize_stream``. Both
# translation TTS workers and speakerphone button-feedback go through
# this single primitive so the speakerphone path can do an atomic
# non-blocking reservation (``get_nowait()``) and drop cleanly when the
# backend is busy — see ``speakerphone/api.py::apply_speak``. The
# ``asyncio.Queue`` primitive is used (rather than Semaphore) because
# ``get_nowait()`` is a true synchronous atomic try-acquire; the
# ``wait_for(sem.acquire(), timeout=0)`` idiom is broken (the freshly
# created acquire awaitable is never complete at t=0).
tts_dispatch_gate: _asyncio.Queue | None = None

# The app owns backpressure in front of the single GB10 TTS container. Keep
# this queue large enough to absorb a real meeting burst; explicit lag gates in
# ``tts.worker`` decide when old audio is no longer useful. A tiny queue drops
# valid interpretation work before the serializer has a chance to drain it.
TTS_QUEUE_MAXSIZE: int = int(_os.environ.get("SCRIBE_TTS_QUEUE_MAXSIZE", "128"))
TTS_WORKER_COUNT: int = int(_os.environ.get("SCRIBE_TTS_WORKER_COUNT", "4"))
TTS_CONTAINER_MAX_CONCURRENCY: int = int(_os.environ.get("SCRIBE_TTS_CONTAINER_CONCURRENCY", "2"))
TTS_EXPECTED_SYNTH_DEFAULT_S: float = float(
    _os.environ.get("SCRIBE_TTS_EXPECTED_SYNTH_DEFAULT_S", "2.0")
)
